# Pluggable feature-cache framework for single-GPU diffusers Wan transformers.
#
# Three orthogonal axes, so a researcher varies one without touching the others:
#   - POLICY  (CachePolicy)      : decides, each step, whether a unit is recomputed.
#   - GRANULARITY (CachedUnit)   : how much of the block stack one cache unit covers
#                                  (whole stack / per-block / groups of k).
#   - STORAGE (CacheStore)       : where/how cached tensors live (GPU / CPU-offload /
#                                  quantized / ...). See store.py. The mechanism never
#                                  touches torch storage directly — it goes through the
#                                  store, keyed by (gen_id, unit, branch, slot).
#
# A single MECHANISM owns all bookkeeping (per-CFG-branch CONTROL state, the
# store-backed residual/history, step counter, instrumentation); the policy only
# decides. The block stack is intercepted exactly like xDiT's flux adapter:
# transformer.forward is wrapped so that, for the duration of each forward,
# `self.blocks` is swapped for the list of CachedUnits (via unittest.mock.patch.object).
# Each unit's call signature matches a real Wan block, so it receives `timestep_proj`
# (== official TeaCache e0) for free. The original forward body (patch-embed, condition
# embedder, norm_out, unpatchify) is reused unchanged.
#
# Continual generation: the mechanism keys storage by gen_id and the pipeline resets
# only when asked (reset_cache), so cache state can persist or coexist across
# generations rather than being wiped every __call__.

import functools
from collections import deque
from dataclasses import dataclass
from unittest import mock

import torch
from torch import nn

from edit.cache.store import CacheStore, GPUStore


def relative_l1(current: torch.Tensor, previous: torch.Tensor) -> float:
    """Mean-absolute relative L1 distance, in fp32 (matches official TeaCache
    numerics regardless of the transformer dtype)."""
    current = current.float()
    previous = previous.float()
    return ((current - previous).abs().mean() / previous.abs().mean()).item()


@dataclass
class HistoryView:
    """One past COMPUTE step, materialized from the store for extrapolation policies."""
    step: int
    residual: torch.Tensor
    e0: torch.Tensor


class UnitState:
    """Per-(unit, CFG-branch) CONTROL state: step counter, policy scratch, and
    instrumentation. The cached TENSORS (residual, history) live in the store, not
    here — that is what makes storage a separate, swappable axis.

    `user` is free scratch for the policy (accumulators, previous signal, ...).
    `hist_steps` records which past steps currently have a history record in the
    store (for eviction ordering when history_len > 0).
    """

    __slots__ = ("cnt", "hist_steps", "user", "calc", "skip")

    def __init__(self):
        self.cnt = 0
        self.hist_steps = deque()          # steps with a stored history record (FIFO)
        self.user = {}
        self.calc = 0
        self.skip = 0


class StepContext:
    """Read-only-ish bundle handed to the policy for one (unit, step) decision.

    `e0` is the timestep projection (official TeaCache e0); `e` is the base time
    embedding (official e), captured by the adapter. `first_block_*` are populated
    by the mechanism only when the policy sets needs_first_block_probe. Cached
    tensors are read through the store via read_residual() / history(), so the
    policy never depends on the storage layout.
    """

    __slots__ = (
        "step", "num_steps", "branch", "unit_index", "num_units",
        "hidden", "encoder_hidden", "e0", "e", "rotary_emb", "state",
        "first_block_output", "first_block_residual", "last_residual", "unit",
    )

    def __init__(self, *, step, num_steps, branch, unit_index, num_units,
                 hidden, encoder_hidden, e0, e, rotary_emb, state, unit):
        self.step = step
        self.num_steps = num_steps
        self.branch = branch
        self.unit_index = unit_index
        self.num_units = num_units
        self.hidden = hidden
        self.encoder_hidden = encoder_hidden
        self.e0 = e0
        self.e = e
        self.rotary_emb = rotary_emb
        self.state = state
        self.unit = unit
        self.first_block_output = None
        self.first_block_residual = None
        self.last_residual = None          # filled by the mechanism on the skip path

    def read_residual(self):
        """The last computed residual for this unit (fetched from the store), or
        None if none is cached. Used by reconstruct(); fetched lazily / cached."""
        if self.last_residual is None:
            self.last_residual = self.unit._store_get("res")
        return self.last_residual

    def history(self):
        """List of past COMPUTE steps (oldest first) as HistoryView, materialized
        from the store. Empty unless the cache was installed with history_len > 0."""
        u = self.unit
        return [
            HistoryView(s, u._store_get(("hist", s, "res")), u._store_get(("hist", s, "e0")))
            for s in self.state.hist_steps
        ]


class CachePolicy:
    """Extension point. Subclass and implement should_compute().

    - should_compute(ctx) -> bool : True => recompute the unit; False => reuse.
      May read/write ctx.state.user for its own scratch (accumulators, prev signal).
    - reconstruct(ctx) -> Tensor  : output when skipping. Default is zeroth-order
      (input + last computed residual, read through the store). Override for
      extrapolation (e.g. Taylor) using ctx.history() (enable with history_len > 0).
    - needs_first_block_probe     : if True, the mechanism runs the unit's first
      block before should_compute and fills ctx.first_block_{output,residual}; on
      recompute the stack continues from that output (FBCache-style signals).
    """

    needs_first_block_probe = False

    def reset(self, state: UnitState) -> None:
        """Initialize per-unit scratch when a branch's state is first created or reset."""

    def should_compute(self, ctx: StepContext) -> bool:
        raise NotImplementedError

    def reconstruct(self, ctx: StepContext) -> torch.Tensor:
        return ctx.hidden + ctx.read_residual()


# ----------------------------- registry ----------------------------- #
POLICY_REGISTRY: dict[str, type] = {}


def register_policy(name: str):
    def deco(cls):
        POLICY_REGISTRY[name] = cls
        cls.policy_name = name
        return cls
    return deco


# ----------------------------- mechanism ----------------------------- #
class CachedUnit(nn.Module):
    """Wraps a contiguous slice of the real Wan blocks as one cache unit. Its
    forward matches a single Wan block's signature so it can stand in for blocks
    inside the diffusers forward loop. Cached tensors are read/written through the
    controller's store, keyed by (gen_id, unit_index, branch, slot)."""

    def __init__(self, controller, blocks, index: int):
        super().__init__()
        self.controller = controller          # plain object, not registered as submodule
        self._blocks = list(blocks)           # aliases the real blocks (not re-registered)
        self.index = index
        self._states: dict[str, UnitState] = {}

    def reset(self) -> None:
        self._states.clear()

    def _state(self) -> UnitState:
        st = self._states.get(self.controller.branch)
        if st is None:
            st = UnitState()
            self.controller.policy.reset(st)
            self._states[self.controller.branch] = st
        return st

    def _key(self, slot):
        c = self.controller
        return (c.gen_id, self.index, c.branch, slot)

    def _store_get(self, slot):
        return self.controller.store.get(self._key(slot))

    def prefetch(self, slot="res") -> None:
        """Stage this unit's cached tensor (current gen/branch) toward the GPU ahead
        of a reuse read, so the swap-in overlaps compute. No-op unless the store
        offloads. See CacheController.prefetch_unit to stage another gen/branch."""
        self.controller.store.prefetch(self._key(slot))

    def _reuse_residual(self):
        """The residual to reconstruct from on a skip: this generation's own residual
        if present, else a blend (mean) of the reuse-window generations' residuals for
        this (unit, branch). Returns None if nothing is available (=> recompute)."""
        own = self._store_get("res")
        if own is not None:
            return own
        ctrl = self.controller
        parts = []
        for g in ctrl.reuse_window:
            r = ctrl.store.get((g, self.index, ctrl.branch, "res"))
            if r is not None:
                parts.append(r)
        if not parts:
            return None
        return parts[0] if len(parts) == 1 else torch.stack(parts).mean(0)

    def _record_history(self, st, residual, e0) -> None:
        store = self.controller.store
        store.put(self._key(("hist", st.cnt, "res")), residual)
        store.put(self._key(("hist", st.cnt, "e0")), e0)
        st.hist_steps.append(st.cnt)
        while len(st.hist_steps) > self.controller.history_len:
            old = st.hist_steps.popleft()
            store.drop(self._key(("hist", old, "res")))
            store.drop(self._key(("hist", old, "e0")))

    def forward(self, hidden_states, encoder_hidden_states, temb, rotary_emb):
        ctrl = self.controller
        policy = ctrl.policy
        store = ctrl.store
        st = self._state()

        ctx = StepContext(
            step=st.cnt, num_steps=ctrl.num_steps, branch=ctrl.branch,
            unit_index=self.index, num_units=ctrl.num_units,
            hidden=hidden_states, encoder_hidden=encoder_hidden_states,
            e0=temb, e=ctrl.e_base, rotary_emb=rotary_emb, state=st, unit=self,
        )

        start = 0
        if policy.needs_first_block_probe:
            fb = self._blocks[0](hidden_states, encoder_hidden_states, temb, rotary_emb)
            ctx.first_block_output = fb
            ctx.first_block_residual = fb - hidden_states
            start = 1

        compute = policy.should_compute(ctx)

        # Try the skip path only when the policy allows it AND a residual is actually
        # available (it may have been evicted under a capacity budget — then we fall
        # through and recompute, which is always safe). When this generation has no
        # residual yet (e.g. a new segment's first touch of the unit), fall back to
        # the reuse window — the previous segment(s)' residual — for a warm start.
        residual = None if compute else self._reuse_residual()

        if residual is not None:
            ctx.last_residual = residual
            out = policy.reconstruct(ctx)
            st.skip += 1
        else:
            h = ctx.first_block_output if start == 1 else hidden_states
            for block in self._blocks[start:]:
                h = block(h, encoder_hidden_states, temb, rotary_emb)
            out = h
            res = out - hidden_states                 # full residual vs unit input
            store.put(self._key("res"), res)
            if ctrl.history_len:
                self._record_history(st, res, temb)
            st.calc += 1

        st.cnt = (st.cnt + 1) % ctrl.num_steps
        return out


class CacheController:
    """Holds the cache units + shared policy + the store + the current branch/step/
    generation. Driven by EditWanPipeline via set_branch()/reset()/configure_steps()."""

    def __init__(self, policy: CachePolicy, num_steps: int, store: CacheStore = None, history_len: int = 0):
        self.policy = policy
        self.num_steps = int(num_steps)
        self.history_len = int(history_len)
        self.store = store if store is not None else GPUStore()
        self.branch = "cond"
        self.gen_id = 0                       # bump (set_generation) to keep generations apart
        self.e_base = None                    # base time embedding e, captured per forward
        self.units: list[CachedUnit] = []
        self.num_units = 0
        self.reuse_window: list = []          # prior gen_ids to warm-start this segment from

    def set_branch(self, name: str) -> None:
        self.branch = name

    def set_generation(self, gen_id) -> None:
        """Tag subsequent forwards with a generation id (part of every store key),
        so several generations can coexist in one store for continual generation."""
        self.gen_id = gen_id

    def set_reuse_window(self, gen_ids) -> None:
        """Prior generations a new segment may warm-start from when it has no residual
        of its own yet (cross-segment reuse). The skip path blends (means) whichever of
        these are still cached; prefetch_reuse() stages them ahead of the forward."""
        self.reuse_window = list(gen_ids)

    def prefetch_reuse(self, branches=("cond", "uncond")) -> None:
        """Stage every reuse-window generation's residual, for every unit and branch,
        toward the GPU — so the warm-start reads during the next forward overlap
        compute instead of stalling it on CPU/NVMe swap-in. Call right before pipe()."""
        for g in self.reuse_window:
            for b in branches:
                for u in self.units:
                    self.store.prefetch((g, u.index, b, "res"))

    def prefetch_unit(self, unit_index, *, branch=None, gen_id=None, slot="res") -> None:
        """Stage one unit's cached tensor toward the GPU ahead of a reuse read, so the
        copy / disk read overlaps compute. branch and gen_id default to the current
        ones; pass them explicitly to warm-start the next segment from a previous
        generation's residuals (the cross-segment reuse case). Safe if absent/evicted."""
        b = self.branch if branch is None else branch
        g = self.gen_id if gen_id is None else gen_id
        self.store.prefetch((g, unit_index, b, slot))

    def reset(self, *, clear_store: bool = True) -> None:
        """Clear control state for a fresh generation. With clear_store=False the
        stored residuals/history survive, warm-starting a continual generation."""
        self.branch = "cond"
        self.e_base = None
        for u in self.units:
            u.reset()
        if clear_store:
            self.store.clear()

    def clear_generation(self, gen_id) -> None:
        """Drop just one generation's tensors from the store (continual research)."""
        self.store.drop_where(lambda k: k[0] == gen_id)

    def configure_steps(self, num_steps: int) -> None:
        self.num_steps = int(num_steps)

    @property
    def stats(self) -> dict:
        """Per-branch (calc, skip), summed across units."""
        agg: dict = {}
        for u in self.units:
            for b, st in u._states.items():
                d = agg.setdefault(b, {"calc": 0, "skip": 0})
                d["calc"] += st.calc
                d["skip"] += st.skip
        return agg

    @property
    def store_stats(self) -> dict:
        """Storage instrumentation: resident bytes and evictions."""
        return {"bytes_resident": self.store.bytes_resident, "num_evictions": self.store.num_evictions}

    def unit_stats(self) -> list:
        """Per-unit, per-branch (calc, skip) — for per-block research."""
        return [
            {b: {"calc": st.calc, "skip": st.skip} for b, st in u._states.items()}
            for u in self.units
        ]


def _build_units(controller, blocks, granularity):
    blocks = list(blocks)
    if granularity == "stack":
        return [CachedUnit(controller, blocks, 0)]
    if granularity == "per_block":
        return [CachedUnit(controller, [b], i) for i, b in enumerate(blocks)]
    if isinstance(granularity, int) and granularity > 0:
        return [
            CachedUnit(controller, blocks[i:i + granularity], idx)
            for idx, i in enumerate(range(0, len(blocks), granularity))
        ]
    raise ValueError(f"granularity must be 'stack', 'per_block', or a positive int; got {granularity!r}")


def apply_cache_on_transformer(
    transformer,
    *,
    policy,
    num_inference_steps: int,
    granularity="stack",
    store: CacheStore = None,
    history_len: int = 0,
    **policy_kwargs,
):
    """Install a pluggable feature cache on a diffusers WanTransformer3DModel.

    Args:
        policy: a registered policy name (e.g. "teacache", "fbcache") or a
            CachePolicy instance.
        num_inference_steps: denoise steps per CFG branch (the counter wraps here).
        granularity: "stack" (one unit over all blocks), "per_block", or an int k
            (groups of k blocks).
        store: a CacheStore controlling where/how cached tensors live (default
            GPUStore = on-device, full precision, no overhead). Pass e.g.
            CPUOffloadStore() or your own subclass to study a storage architecture.
        history_len: how many past COMPUTE steps to keep per unit (for extrapolation
            policies); 0 keeps only the last residual.
        **policy_kwargs: forwarded to the policy constructor when `policy` is a name.
    """
    if not hasattr(transformer, "blocks") or not hasattr(transformer, "condition_embedder"):
        raise TypeError("apply_cache_on_transformer expects a diffusers WanTransformer3DModel.")

    if isinstance(policy, str):
        if policy not in POLICY_REGISTRY:
            raise ValueError(f"Unknown policy {policy!r}; registered: {sorted(POLICY_REGISTRY)}")
        policy = POLICY_REGISTRY[policy](**policy_kwargs)
    elif policy_kwargs:
        raise ValueError("policy_kwargs are only used when `policy` is a registry name.")

    controller = CacheController(policy, num_steps=num_inference_steps, store=store, history_len=history_len)
    controller.units = _build_units(controller, transformer.blocks, granularity)
    controller.num_units = len(controller.units)

    cached_blocks = nn.ModuleList(controller.units)
    original_forward = transformer.forward
    original_ce_forward = transformer.condition_embedder.forward

    def capturing_condition_embedder(*a, **k):
        out = original_ce_forward(*a, **k)
        controller.e_base = out[0]            # diffusers temb == official TeaCache e
        return out

    @functools.wraps(original_forward)
    def new_forward(self, *args, **kwargs):
        with mock.patch.object(self, "blocks", cached_blocks), mock.patch.object(
            self.condition_embedder, "forward", capturing_condition_embedder
        ):
            return original_forward(*args, **kwargs)

    transformer.forward = new_forward.__get__(transformer)
    object.__setattr__(transformer, "_nano_cache", controller)
    object.__setattr__(transformer, "_nano_original_forward", original_forward)
    return transformer


def remove_cache_from_transformer(transformer):
    """Undo apply_cache_on_transformer, restoring the original forward."""
    if hasattr(transformer, "_nano_original_forward"):
        transformer.forward = transformer._nano_original_forward
        object.__delattr__(transformer, "_nano_original_forward")
        object.__delattr__(transformer, "_nano_cache")
    return transformer
