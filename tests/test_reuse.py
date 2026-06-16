"""P1 mechanism test: cross-segment warm-start reuse + reuse-prefetch are correct.

Drives a CacheController over several "segments" (generations) sharing one tiered
store. A warm-start policy skips each segment's first step and reconstructs it from
the reuse window — the previous segment's residual — exercising the cross-segment
read path the long-video engine relies on. Checks:

  1. Segment 0 (no prior) recomputes safely; later segments actually reuse.
  2. The reconstructed output equals hidden + the prior segment's residual (blend),
     i.e. the warm start reads the right cross-generation tensor.
  3. Issuing prefetch_reuse() first does NOT change any result (async staging is exact).

Run:  python tests/test_reuse.py
"""

import tempfile

import torch

from edit.cache.framework import CacheController, CachedUnit, CachePolicy
from edit.cache.store import GPUStore, CPUOffloadStore, DiskOffloadStore, TieredStore

MB = 1024 * 1024
SEQ, DIM = 1024, 256                  # residual [1,1024,256] fp32 = 1 MB / unit
NUM_UNITS, STEPS = 6, 4


class WarmStart(CachePolicy):
    """Skip (reuse) the first step of each segment; compute the rest."""
    def should_compute(self, ctx):
        return ctx.step != 0


class Block:
    def __init__(self, d):
        self.d = float(d)

    def __call__(self, h, enc, temb, rot):
        return h + self.d


def build(store):
    ctrl = CacheController(WarmStart(), num_steps=STEPS, store=store)
    ctrl.units = [CachedUnit(ctrl, [Block(i + 1)], i) for i in range(NUM_UNITS)]
    ctrl.num_units = NUM_UNITS
    return ctrl


def segment(ctrl, h0, *, prefetch):
    """One segment: STEPS steps over all units, branch cond. Returns step-0 outputs."""
    if prefetch:
        ctrl.prefetch_reuse(("cond",))
    step0_out = []
    for step in range(STEPS):
        ctrl.set_branch("cond")
        h = h0
        outs = []
        for u in ctrl.units:
            h = u.forward(h, None, None, None)
            outs.append(h)
        if step == 0:
            step0_out = [o.clone() for o in outs]
    return step0_out


def main():
    assert torch.cuda.is_available(), "need CUDA"
    nvme = tempfile.mkdtemp(prefix="edit_reuse_test_")
    store = TieredStore(
        [(GPUStore(), 3 * MB),
         (CPUOffloadStore(), 4 * MB),
         (DiskOffloadStore(directory=nvme), None)],
        promote=False,
    )
    ctrl = build(store)
    h0 = torch.zeros(1, SEQ, DIM, device="cuda")

    # Segment 0: no reuse window -> warm-start step must safely recompute.
    ctrl.set_generation(0)
    ctrl.set_reuse_window([])
    ctrl.reset(clear_store=False)
    base_skips = sum(st.skip for u in ctrl.units for st in u._states.values())
    seg0_step0 = segment(ctrl, h0, prefetch=False)
    seg0_skips = sum(st.skip for u in ctrl.units for st in u._states.values()) - base_skips
    assert seg0_skips == 0, f"segment 0 should not reuse (no prior), got {seg0_skips} skips"

    # Each unit i applied STEPS times adds (i+1) per step; segment 0 computed every step.
    # The residual stored for unit i is its single-step delta = (i+1).
    # Segment 1 step 0 should SKIP and reconstruct = hidden + prior residual.
    ctrl.set_generation(1)
    ctrl.set_reuse_window([0])
    ctrl.reset(clear_store=False)

    # Run WITHOUT prefetch, capture step-0 outputs.
    ctrl_state_skip = sum(st.skip for u in ctrl.units for st in u._states.values())
    seg1_noprefetch = segment(ctrl, h0, prefetch=False)
    seg1_skips = sum(st.skip for u in ctrl.units for st in u._states.values()) - ctrl_state_skip
    assert seg1_skips == NUM_UNITS, f"segment 1 step 0 should reuse all {NUM_UNITS} units, got {seg1_skips}"

    # Reconstructed step-0 output for unit i = cumulative input + prior residual(i+1).
    # Input to unit i at step 0 = h0 + sum_{j<i}(reconstructed deltas). Each unit's
    # warm-started delta is exactly its stored residual (i+1), so the running sum is
    # the same as a fully-computed pass: out_i = sum_{j<=i}(j+1).
    for i, out in enumerate(seg1_noprefetch):
        expect = float(sum(range(1, i + 2)))      # 1+2+...+(i+1)
        got = out.mean().item()
        assert abs(got - expect) < 1e-4, f"unit {i}: warm-start reuse gave {got}, expected {expect}"

    # Now segment 2 reusing [1]: run WITH prefetch and WITHOUT, results must match.
    def run_seg2(prefetch):
        ctrl.set_generation(2)
        ctrl.set_reuse_window([1])
        ctrl.reset(clear_store=False)
        return segment(ctrl, h0, prefetch=prefetch)

    a = run_seg2(prefetch=False)
    b = run_seg2(prefetch=True)
    mism = sum(0 if torch.equal(x, y) else 1 for x, y in zip(a, b))
    assert mism == 0, f"prefetch changed {mism} unit outputs (must be exact)"

    print(f"tier split during reuse: GPU/CPU/NVMe = {[round(x/MB,1) for x in store.tier_bytes]} MB")
    store.clear()
    import shutil
    shutil.rmtree(nvme, ignore_errors=True)
    print("All P1 reuse checks passed: cross-segment warm-start reads the right residual,"
          "\nsegment 0 recomputes safely, and reuse-prefetch is bit-exact.")


if __name__ == "__main__":
    main()
