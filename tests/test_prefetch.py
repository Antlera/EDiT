"""P0 integration test: prefetch driven through the cache framework is correct.

Runs a real CacheController/CachedUnit stack over a 3-tier store sized so the stored
residuals spill onto CPU and NVMe, then reads each unit's residual back via the
framework's prefetch hooks (CachedUnit.prefetch / CacheController.prefetch_unit) and
checks it is bit-identical to a plain synchronous get. This exercises the whole path
— controller -> store.prefetch -> tier routing -> draining get — not just the raw
store, and confirms async staging never changes the value.

Run:  python tests/test_prefetch.py
"""

import os
import tempfile

import torch

from edit.cache.framework import CacheController, CachedUnit, CachePolicy
from edit.cache.store import GPUStore, CPUOffloadStore, DiskOffloadStore, TieredStore

MB = 1024 * 1024
SEQ, DIM = 2048, 512                  # residual [1,2048,512] fp32 = 4 MB per unit
NUM_UNITS, STEPS = 12, 4


class AlwaysCompute(CachePolicy):
    def should_compute(self, ctx):    # always compute => every unit stores a residual
        return True


class Block:
    def __init__(self, d):
        self.d = float(d)

    def __call__(self, h, enc, temb, rot):
        return h + self.d             # distinct residual per unit


def main():
    assert torch.cuda.is_available(), "need CUDA"
    nvme = tempfile.mkdtemp(prefix="edit_prefetch_test_")
    # Tiny GPU + CPU bounds so 12 units x 4 MB = 48 MB spills GPU -> CPU -> NVMe.
    store = TieredStore(
        [(GPUStore(), 8 * MB),
         (CPUOffloadStore(pin_memory=True), 12 * MB),
         (DiskOffloadStore(directory=nvme), None)],
        promote=False,                # reads stay put, so we test each tier in place
    )
    ctrl = CacheController(AlwaysCompute(), num_steps=STEPS, store=store)
    ctrl.units = [CachedUnit(ctrl, [Block(i + 1)], i) for i in range(NUM_UNITS)]
    ctrl.num_units = NUM_UNITS
    ctrl.set_generation(7)
    ctrl.set_branch("cond")

    # Populate one residual per unit (compute path stores out - hidden).
    h0 = torch.zeros(1, SEQ, DIM, device="cuda")
    for u in ctrl.units:
        u.forward(h0, None, None, None)

    placed = store.tier_bytes
    tier_names = ("GPU", "CPU", "NVMe")
    print("after populate, resident per tier:",
          {n: f"{b/MB:.0f}MB" for n, b in zip(tier_names, placed)})
    assert placed[1] > 0 and placed[2] > 0, "test needs residuals on CPU and NVMe"

    # Read every unit back via the framework prefetch hooks; compare to a sync get.
    mism = 0
    tiers_hit = set()
    for u in ctrl.units:
        key = u._key("res")
        tiers_hit.add(tier_names[store._loc[key][0]])
        ref = store.get(key).clone()                  # synchronous reference

        u.prefetch("res")                             # CachedUnit hook (current gen/branch)
        got_unit = store.get(key)
        ctrl.prefetch_unit(u.index, branch="cond", gen_id=7)  # controller hook (explicit)
        got_ctrl = store.get(key)

        if not (torch.equal(ref, got_unit) and torch.equal(ref, got_ctrl)):
            mism += 1

    # A prefetch for an evicted/unknown key must be a harmless no-op.
    ctrl.prefetch_unit(999, branch="cond", gen_id=7)
    assert store.get((7, 999, "cond", "res")) is None

    store.clear()
    import shutil
    shutil.rmtree(nvme, ignore_errors=True)

    print(f"tiers exercised by reuse reads: {sorted(tiers_hit)}")
    assert mism == 0, f"{mism} prefetched residuals differed from sync get"
    print("All prefetch framework checks passed (0 mismatches, CPU+NVMe staged correctly).")


if __name__ == "__main__":
    main()
