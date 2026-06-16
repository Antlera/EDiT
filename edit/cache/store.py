# Pluggable STORAGE backend for the feature cache.
#
# The mechanism (framework.py) never touches torch storage directly — it reads and
# writes cached tensors (the zeroth-order residual, optional history records)
# through a CacheStore. That makes "where/how the cache lives" a swappable research
# axis, orthogonal to the cache POLICY (when to skip) and the GRANULARITY (unit
# size):
#
#   - GPUStore        : keep tensors as-is on their device, full precision. This is
#                       the original EDiT behavior (no copy, no overhead).
#   - CPUOffloadStore : park tensors in (pinned) CPU memory, copy back to the GPU on
#                       read. Trades PCIe bandwidth for GPU memory — the canonical
#                       offload study for long / many-unit / continual generation.
#   - <your subclass> : quantized / low-rank / delta / disk / paged ... Implement
#                       _encode/_decode/_nbytes on KeyedTensorStore, or the bare
#                       CacheStore interface for something exotic.
#
# Keys are opaque hashables built by the mechanism: (gen_id, unit_index, branch,
# slot). `gen_id` lets several generations coexist in one store (continual
# generation); a capacity budget + LRU eviction then bounds memory across them.

import os
from abc import ABC, abstractmethod
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import torch

# --- async prefetch machinery -------------------------------------------------- #
# A single side stream carries offload H2D copies so they overlap default-stream
# compute; a small thread pool carries blocking disk reads (torch.load releases the
# GIL during file I/O) so they overlap too. Both are created lazily on first use so
# importing this module never touches CUDA.
_PREFETCH_STREAM = None
_IO_POOL = None


def _prefetch_stream():
    global _PREFETCH_STREAM
    if _PREFETCH_STREAM is None:
        _PREFETCH_STREAM = torch.cuda.Stream()
    return _PREFETCH_STREAM


def _io_pool():
    global _IO_POOL
    if _IO_POOL is None:
        _IO_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cache-io")
    return _IO_POOL


class CacheStore(ABC):
    """Minimal storage interface the mechanism depends on.

    A store maps an opaque hashable key to one tensor. It MAY transform tensors on
    put (move device, quantize, compress) and MUST return a ready-to-use tensor of
    the original dtype/device on get. `get` returns None for an unknown or evicted
    key, and the mechanism then recomputes — so eviction is always safe.
    """

    @abstractmethod
    def put(self, key, tensor: torch.Tensor) -> None: ...

    @abstractmethod
    def get(self, key):
        """Return the tensor for `key`, ready to use, or None if absent/evicted."""

    @abstractmethod
    def drop(self, key) -> None: ...

    @abstractmethod
    def clear(self) -> None:
        """Drop everything (called on a fresh generation unless reuse is requested)."""

    def drop_where(self, predicate) -> None:
        """Drop every key for which predicate(key) is True (e.g. one generation)."""
        raise NotImplementedError

    def prefetch(self, key) -> None:
        """Hint that `key` will be read soon: begin staging it toward the GPU OFF the
        critical path. A later get(key) returns the staged tensor, blocking only if
        the transfer has not finished. Default: a no-op (e.g. GPUStore — already
        resident). Offloading stores override this to overlap the copy / disk read
        with compute. Issuing prefetch for an absent/evicted key is harmless."""
        """Approximate bytes currently held — instrumentation for storage research."""
        return 0

    @property
    def num_evictions(self) -> int:
        return 0


class KeyedTensorStore(CacheStore):
    """In-memory CacheStore with an optional byte budget + LRU eviction.

    Subclasses define only how a tensor is encoded for storage and decoded back:
      _encode(tensor) -> payload   (e.g. move to CPU, quantize)
      _decode(payload) -> tensor   (inverse; must return a GPU-ready tensor)
      _nbytes(payload) -> int      (resident size of a payload)

    All bookkeeping (LRU order, byte accounting, eviction, drop_where) lives here.
    """

    def __init__(self, *, capacity_bytes: int | None = None):
        self._data: "OrderedDict" = OrderedDict()   # key -> payload, ordered LRU (oldest first)
        self._bytes = 0
        self._evictions = 0
        self.capacity_bytes = capacity_bytes
        self._inflight: dict = {}                   # key -> staged prefetch (set by subclasses)

    # ---- subclass hooks ---------------------------------------------------- #
    def _encode(self, tensor):
        raise NotImplementedError

    def _decode(self, payload):
        raise NotImplementedError

    def _nbytes(self, payload) -> int:
        return payload.element_size() * payload.nelement()

    def _free(self, payload) -> None:
        """Release a payload's backing resource on removal (e.g. delete a file).
        No-op for in-memory payloads; overridden by disk-backed stores."""

    # ---- CacheStore interface --------------------------------------------- #
    def put(self, key, tensor) -> None:
        old = self._data.pop(key, None)
        if old is not None:
            self._bytes -= self._nbytes(old)
            self._free(old)
        payload = self._encode(tensor)
        self._data[key] = payload                 # newest -> end
        self._bytes += self._nbytes(payload)
        self._maybe_evict(protect=key)

    def get(self, key):
        payload = self._data.get(key)
        if payload is None:
            return None
        self._data.move_to_end(key)               # mark as recently used
        return self._decode(payload)

    def drop(self, key) -> None:
        self._inflight.pop(key, None)             # cancel any pending stage for this key
        payload = self._data.pop(key, None)
        if payload is not None:
            self._bytes -= self._nbytes(payload)
            self._free(payload)

    def clear(self) -> None:
        self._inflight.clear()
        for payload in self._data.values():
            self._free(payload)
        self._data.clear()
        self._bytes = 0

    def drop_where(self, predicate) -> None:
        for k in [k for k in self._data if predicate(k)]:
            self.drop(k)

    def _maybe_evict(self, *, protect) -> None:
        if self.capacity_bytes is None:
            return
        while self._bytes > self.capacity_bytes and len(self._data) > 1:
            k = next(iter(self._data))            # least recently used
            if k == protect:
                break
            self.drop(k)
            self._evictions += 1

    @property
    def bytes_resident(self) -> int:
        return self._bytes

    @property
    def num_evictions(self) -> int:
        return self._evictions


class GPUStore(KeyedTensorStore):
    """Keep tensors exactly where they are (no copy). Original EDiT behavior;
    zero overhead. With a capacity_bytes budget it becomes an LRU GPU cache."""

    def _encode(self, tensor):
        return tensor

    def _decode(self, payload):
        return payload


class CPUOffloadStore(KeyedTensorStore):
    """Offload cached tensors to CPU memory; bring them back on read.

    Copies are issued non-blocking and (by default) staged through pinned memory,
    so the GPU↔CPU transfer overlaps with stream work. Reading a cached residual
    therefore costs one host→device copy — the cost this store exists to study.
    """

    def __init__(self, *, pin_memory: bool = True, capacity_bytes: int | None = None):
        super().__init__(capacity_bytes=capacity_bytes)
        self.pin_memory = pin_memory

    def _encode(self, tensor):
        pin = self.pin_memory and tensor.is_cuda
        cpu = torch.empty_like(tensor, device="cpu", pin_memory=pin)
        cpu.copy_(tensor, non_blocking=pin)
        return (cpu, tensor.device)               # remember origin for _decode

    def _decode(self, payload):
        cpu, device = payload
        return cpu.to(device, non_blocking=True)

    def _nbytes(self, payload) -> int:
        cpu, _ = payload
        return cpu.element_size() * cpu.nelement()

    def prefetch(self, key) -> None:
        if key in self._inflight or key not in self._data:
            return
        cpu, device = self._data[key]
        self._data.move_to_end(key)
        stream = _prefetch_stream()
        with torch.cuda.stream(stream):           # H2D on the side stream...
            gpu = cpu.to(device, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record(stream)                         # ...flagged complete by this event
        # keep `cpu` referenced until the async copy is consumed (no use-after-free)
        self._inflight[key] = (gpu, ev, cpu)

    def get(self, key):
        staged = self._inflight.pop(key, None)
        if staged is not None:
            gpu, ev, _cpu = staged
            torch.cuda.current_stream().wait_event(ev)   # order compute after the copy
            self._data.move_to_end(key)
            return gpu
        payload = self._data.get(key)
        if payload is None:
            return None
        self._data.move_to_end(key)
        return self._decode(payload)


class DiskOffloadStore(KeyedTensorStore):
    """Offload cached tensors to disk (NVMe) — the deepest tier.

    One file per entry; the tensor is brought back to its original device on read.
    For very long or many-generation continual runs where even CPU RAM is tight.
    The encode (D2H copy + write) is synchronous, so reads see complete data; the
    write/read cost is exactly the NVMe-tier trade-off this store exists to study.
    """

    def __init__(self, *, directory: str = None, capacity_bytes: int | None = None):
        super().__init__(capacity_bytes=capacity_bytes)
        import tempfile

        self.dir = directory or tempfile.mkdtemp(prefix="edit_cache_")
        os.makedirs(self.dir, exist_ok=True)
        self._counter = 0

    def _encode(self, tensor):
        self._counter += 1
        path = os.path.join(self.dir, f"{self._counter}.pt")
        cpu = tensor.to("cpu")                    # synchronous D2H; data ready to write
        torch.save(cpu, path)
        return (path, cpu.element_size() * cpu.nelement(), tensor.device)

    def _decode(self, payload):
        path, _, device = payload
        return torch.load(path, map_location="cpu").to(device)

    @staticmethod
    def _read_pinned(path):
        cpu = torch.load(path, map_location="cpu")    # blocking file read (worker thread)
        return cpu.pin_memory() if not cpu.is_pinned() else cpu

    def prefetch(self, key) -> None:
        if key in self._inflight or key not in self._data:
            return
        path, _, device = self._data[key]
        self._data.move_to_end(key)
        # The slow part — the disk read — runs in a worker thread, overlapping compute.
        self._inflight[key] = (_io_pool().submit(self._read_pinned, path), device)

    def get(self, key):
        item = self._inflight.pop(key, None)
        if item is not None:
            fut, device = item
            cpu = fut.result()                        # wait for the (overlapped) disk read
            self._data.move_to_end(key)
            return cpu.to(device, non_blocking=True)  # small H2D from pinned, stream-ordered
        payload = self._data.get(key)
        if payload is None:
            return None
        self._data.move_to_end(key)
        return self._decode(payload)

    def _nbytes(self, payload) -> int:
        return payload[1]

    def _free(self, payload) -> None:
        try:
            os.remove(payload[0])
        except OSError:
            pass


class TieredStore(CacheStore):
    """Cascade several stores by a per-tier byte budget.

    An entry lives in the hottest tier that fits; when a tier exceeds its budget its
    least-recently-used entries SPILL DOWN to the next tier (moved, not dropped). On
    read, a cold entry is brought back and (by default) promoted to the top tier,
    which may push others down in turn. The last tier should be unbounded (the
    backstop). The headline use is a GPU bound that overflows to NVMe:

        TieredStore([(GPUStore(),         4 * 1024**3),   # hot: up to 4 GB on the GPU
                     (DiskOffloadStore(), None)])         # cold: unbounded, on NVMe

    A CPU tier can sit in between for a 3-level GPU -> RAM -> NVMe hierarchy.

    prefetch(key) routes to the holding tier so a soon-to-be-read residual is staged
    toward the GPU off the critical path (the copy / disk read overlaps compute); the
    later get(key) then blocks only on whatever transfer has not yet finished.
    """

    def __init__(self, tiers, *, promote: bool = True):
        # tiers: list of (CacheStore, capacity_bytes|None). Backends should be
        # unbounded themselves (capacity_bytes=None) — TieredStore owns placement.
        if not tiers:
            raise ValueError("TieredStore needs at least one tier.")
        self.tiers = [t for t, _ in tiers]
        self.caps = [c for _, c in tiers]
        self.promote = promote
        self._loc: "OrderedDict" = OrderedDict()   # key -> [tier_index, nbytes], LRU (oldest first)
        self._tbytes = [0] * len(self.tiers)
        self._spills = 0
        self._evictions = 0

    @staticmethod
    def _bytes_of(tensor) -> int:
        return tensor.element_size() * tensor.nelement()

    def _place(self, key, tier_idx, nbytes) -> None:
        self._loc[key] = [tier_idx, nbytes]
        self._loc.move_to_end(key)                 # most recently touched
        self._tbytes[tier_idx] += nbytes

    def _spill_overflow(self) -> None:
        # For every over-budget tier: demote its LRU entries to the next tier;
        # for the LAST tier (nowhere lower) drop them entirely (total-size bound).
        last = len(self.tiers) - 1
        for ti in range(len(self.tiers)):
            cap = self.caps[ti]
            if cap is None:
                continue
            while self._tbytes[ti] > cap:
                victim = next((k for k, (t, _) in self._loc.items() if t == ti), None)
                if victim is None:
                    break
                _, nb = self._loc[victim]
                if ti == last:
                    self.tiers[ti].drop(victim)            # evict: bounded backstop
                    self._tbytes[ti] -= nb
                    del self._loc[victim]
                    self._evictions += 1
                else:
                    tensor = self.tiers[ti].get(victim)    # bring up (GPUStore: a ref)
                    self.tiers[ti].drop(victim)
                    self._tbytes[ti] -= nb
                    self.tiers[ti + 1].put(victim, tensor)  # spill down (copies into next tier)
                    self._tbytes[ti + 1] += nb
                    self._loc[victim] = [ti + 1, nb]
                    self._loc.move_to_end(victim, last=False)  # demoted => least-recent
                    self._spills += 1

    def put(self, key, tensor) -> None:
        if key in self._loc:
            ti, nb = self._loc.pop(key)
            self.tiers[ti].drop(key)
            self._tbytes[ti] -= nb
        nb = self._bytes_of(tensor)
        self.tiers[0].put(key, tensor)
        self._place(key, 0, nb)
        self._spill_overflow()

    def get(self, key):
        loc = self._loc.get(key)
        if loc is None:
            return None
        ti, nb = loc
        tensor = self.tiers[ti].get(key)
        self._loc.move_to_end(key)
        if ti != 0 and self.promote:
            self.tiers[ti].drop(key)
            self._tbytes[ti] -= nb
            self.tiers[0].put(key, tensor)
            self._loc[key] = [0, nb]
            self._loc.move_to_end(key)
            self._tbytes[0] += nb
            self._spill_overflow()
        return tensor

    def prefetch(self, key) -> None:
        """Stage `key` toward the GPU from whatever tier holds it (no-op if it is
        already on the GPU tier, absent, or evicted). Does not change placement —
        promotion still happens on the eventual get()."""
        loc = self._loc.get(key)
        if loc is None:
            return
        self.tiers[loc[0]].prefetch(key)

    def drop(self, key) -> None:
        loc = self._loc.pop(key, None)
        if loc is not None:
            ti, nb = loc
            self.tiers[ti].drop(key)
            self._tbytes[ti] -= nb

    def clear(self) -> None:
        for t in self.tiers:
            t.clear()
        self._loc.clear()
        self._tbytes = [0] * len(self.tiers)

    def drop_where(self, predicate) -> None:
        for k in [k for k in self._loc if predicate(k)]:
            self.drop(k)

    @property
    def tier_bytes(self) -> list:
        """Resident bytes per tier — e.g. [GPU_bytes, NVMe_bytes]."""
        return list(self._tbytes)

    @property
    def bytes_resident(self) -> int:
        return sum(self._tbytes)

    @property
    def num_spills(self) -> int:
        """How many times an entry was moved down a tier (GPU bound exceeded)."""
        return self._spills

    @property
    def num_evictions(self) -> int:
        return self._evictions + sum(t.num_evictions for t in self.tiers)
