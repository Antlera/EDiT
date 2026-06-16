"""Verify P0: async prefetch overlaps the cache swap-in with compute.

The skip path reads a cached residual with store.get(). When that residual lives on
CPU or NVMe, a synchronous get() stalls the forward on the copy / disk read. The new
store.prefetch(key) stages it toward the GPU off the critical path, so a get() issued
*after* a compute block finds it (nearly) ready.

This bench replays the realistic reuse pattern — "I know which residual step i will
reuse, so I prefetch it `depth` iterations ahead and read it back after computing" —
for the CPU and NVMe tiers, and compares wall-clock against the synchronous baseline.
It also checks every fetched tensor is bit-identical, so prefetch is correctness-safe.

Run:  python tests/bench_prefetch.py
NVMe: ./.nano_cache_prefetch (set NANO_CACHE_DIR for a real NVMe mount; avoid /tmp).
"""

import os
import time

import torch

from edit.cache.store import CPUOffloadStore, DiskOffloadStore

# realistic Wan2.1-1.3B residual: [1, 14040, 1536] bf16 = 41 MB
B, SEQ, DIM = 1, 14040, 1536
DTYPE = torch.bfloat16
N = 24                      # residuals reused over the run
DEPTH = 2                   # prefetch this many iterations ahead
MB = 1024 * 1024


COMPUTE_ITERS = 48          # sized so one block ≈ a small transformer forward (tens of ms)


def compute_block(a):
    """A stand-in transformer-ish GPU workload to overlap the swap-in against. A real
    Wan forward is 200-600 ms, far larger than a residual read — i.e. swap-in is
    comfortably hideable; we use a smaller block here just to keep the bench short."""
    x = a
    for _ in range(COMPUTE_ITERS):
        x = torch.relu(x @ a)
    return x


def calibrate(a):
    for _ in range(3):
        compute_block(a)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        compute_block(a)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / 5 * 1000


def fill(store, refs):
    for i, r in enumerate(refs):
        store.put((i,), r)


def run_sync(store, refs, work):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    bad = 0
    for i in range(N):
        compute_block(work)
        r = store.get((i,))
        if not torch.equal(r, refs[i]):
            bad += 1
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000, bad


def run_prefetch(store, refs, work, depth):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    bad = 0
    for i in range(min(depth, N)):
        store.prefetch((i,))
    for i in range(N):
        if i + depth < N:
            store.prefetch((i + depth,))   # stage a future read off the critical path
        compute_block(work)                # ... which overlaps this compute
        r = store.get((i,))                # ready by now
        if not torch.equal(r, refs[i]):
            bad += 1
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000, bad


def main():
    assert torch.cuda.is_available(), "need CUDA"
    nvme = os.environ.get("NANO_CACHE_DIR", os.path.join(os.getcwd(), ".nano_cache_prefetch"))
    res_mb = B * SEQ * DIM * 2 / MB
    refs = [torch.randn(B, SEQ, DIM, device="cuda", dtype=DTYPE) for _ in range(N)]
    work = torch.randn(4096, 4096, device="cuda", dtype=DTYPE)

    comp_ms = calibrate(work)
    print(f"Residual {res_mb:.0f}MB x{N} reused   |   compute block ≈ {comp_ms:.1f} ms each   |   prefetch depth {DEPTH}\n")
    print(f"{'tier':18}{'sync ms':>10}{'prefetch ms':>14}{'hidden':>9}{'speedup':>9}{'mismatch':>10}")

    for name, store in (("CPUOffloadStore", CPUOffloadStore(pin_memory=True)),
                        ("DiskOffloadStore", DiskOffloadStore(directory=nvme))):
        fill(store, refs)
        # warm the path (allocator, pinned pool, file cache) so we time steady state
        run_prefetch(store, refs, work, DEPTH)
        s_ms, s_bad = run_sync(store, refs, work)
        p_ms, p_bad = run_prefetch(store, refs, work, DEPTH)
        hidden = (s_ms - p_ms) / N
        print(f"{name:18}{s_ms:10.1f}{p_ms:14.1f}{hidden:8.2f}m{s_ms/p_ms:8.2f}x{(s_bad+p_bad):>10}")
        store.clear()

    base = comp_ms * N
    print(f"\nPure compute (no swap) ≈ {base:.0f} ms for {N} blocks. The closer prefetch gets to that,"
          f"\nthe more of the swap-in it hid behind compute. 'mismatch' must be 0 (prefetch is exact).")
    print("Note: warm runs read .pt files from the OS page cache (RAM), not the SSD — for true"
          "\nNVMe latency point NANO_CACHE_DIR at a cold mount and drop caches. A real Wan forward"
          "\n(200-600 ms) dwarfs a residual read, so in practice the swap-in hides almost entirely.")
    if os.path.isdir(nvme) and not os.environ.get("NANO_CACHE_DIR"):
        import shutil
        shutil.rmtree(nvme, ignore_errors=True)


if __name__ == "__main__":
    main()
