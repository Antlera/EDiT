"""Long-run continual-generation memory bench.

Simulates many back-to-back generations through the real cache mechanism, tagging
each with a fresh gen_id (the multi-generation continual case — the one that grows
if left unbounded). Tracks GPU VRAM, host RAM (RSS), and NVMe usage together, for:

  1. plain GPUStore (unbounded)          -> VRAM climbs every generation (eventually OOMs)
  2. TieredStore(GPU bound -> NVMe)       -> VRAM held flat at the bound; the overflow
                                             spills to NVMe; host RAM stays flat.

No diffusers needed — fake blocks produce residuals of a realistic per-unit size.

Run: python tests/bench_longrun.py   (NVMe dir: ./.nano_cache_longrun or $NANO_CACHE_DIR)
"""

import os
import time

import torch

from edit.cache.framework import CacheController, CachedUnit, CachePolicy
from edit.cache.store import GPUStore, DiskOffloadStore, TieredStore

MB = 1024 * 1024
SEQ, DIM = 2048, 512                 # residual [1, 2048, 512] fp32 = 4 MB per unit
NUM_UNITS, STEPS = 8, 8
GPU_BOUND = 256 * MB                 # keep at most 256 MB of cache on the GPU
NVME_BOUND = 1024 * MB              # and at most 1 GB on NVMe (run [3])
UNBOUNDED_GENS = 40                  # short: this one grows, don't OOM the box
BOUNDED_GENS = 150                   # long: GPU flat, NVMe grows
CAPPED_GENS = 150                    # long: GPU + NVMe both flat -> truly indefinite


class Block:
    def __init__(self, d):
        self.d = float(d)

    def __call__(self, h, enc, temb, rot):
        return h + self.d


class Drive(CachePolicy):
    def should_compute(self, ctx):
        return ctx.step % 2 == 0      # compute half the steps, skip (reuse) the rest


def build(store):
    ctrl = CacheController(Drive(), num_steps=STEPS, store=store)
    ctrl.units = [CachedUnit(ctrl, [Block(i + 1)], i) for i in range(NUM_UNITS)]
    ctrl.num_units = NUM_UNITS
    return ctrl


def one_generation(ctrl, h0):
    for step in range(STEPS):
        for br in ("cond", "uncond"):
            ctrl.set_branch(br)
            h = h0
            for u in ctrl.units:
                h = u.forward(h, None, None, None)


def rss_mb():
    with open("/proc/self/statm") as f:
        resident_pages = int(f.read().split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE") / MB


def gpu_mb():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / MB


def run(label, store, gens, checkpoints):
    torch.cuda.empty_cache()
    h0 = torch.zeros(1, SEQ, DIM, device="cuda")
    g0_gpu, g0_rss = gpu_mb(), rss_mb()
    print(f"\n{label}")
    print(f"  start: GPU={g0_gpu:.0f}MB  RSS={g0_rss:.0f}MB")
    t0 = time.perf_counter()
    for g in range(gens):
        ctrl.set_generation(g)        # fresh gen_id => unique keys => the growing case

        one_generation(ctrl, h0)
        if (g + 1) in checkpoints or g == gens - 1:
            cold = store.tier_bytes[-1] / MB if hasattr(store, "tier_bytes") else 0.0
            spills = getattr(store, "num_spills", 0)
            dt = (time.perf_counter() - t0) / (g + 1) * 1000
            print(f"  gen {g+1:4}/{gens}: GPU={gpu_mb():7.0f}MB  RSS={rss_mb():7.0f}MB  "
                  f"NVMe={cold:7.0f}MB  spills={spills:<6} {dt:5.1f}ms/gen")


def main():
    assert torch.cuda.is_available(), "need CUDA"
    nvme = os.environ.get("NANO_CACHE_DIR", os.path.join(os.getcwd(), ".nano_cache_longrun"))
    cache_per_gen = NUM_UNITS * 2 * (SEQ * DIM * 4) / MB
    print(f"Per generation: {NUM_UNITS} units x 2 branches x {SEQ*DIM*4/MB:.0f}MB "
          f"= {cache_per_gen:.0f}MB of fresh cache (unique gen_id each gen)")
    print(f"GPU bound for the tiered store: {GPU_BOUND/MB:.0f}MB   NVMe dir: {nvme}")

    global ctrl
    # 1) unbounded GPU store — watch VRAM climb
    ctrl = build(GPUStore())
    run(f"[1] plain GPUStore (unbounded) — {UNBOUNDED_GENS} gens",
        ctrl.store, UNBOUNDED_GENS, checkpoints={1, 5, 10, 20, 30})
    ctrl.reset()
    del ctrl
    torch.cuda.empty_cache()

    # 2) bounded: GPU up to GPU_BOUND, overflow spills to NVMe — long run, flat VRAM
    tiered = TieredStore([(GPUStore(), GPU_BOUND), (DiskOffloadStore(directory=nvme), None)])
    ctrl = build(tiered)
    run(f"[2] TieredStore(GPU<={GPU_BOUND//MB}MB -> NVMe) — {BOUNDED_GENS} gens",
        tiered, BOUNDED_GENS, checkpoints={1, 5, 10, 25, 50, 100})

    print(f"\nFinal tiered split: GPU={tiered.tier_bytes[0]/MB:.0f}MB  "
          f"NVMe={tiered.tier_bytes[1]/MB:.0f}MB  spills={tiered.num_spills}")
    tiered.clear()
    del ctrl, tiered
    torch.cuda.empty_cache()

    # 3) fully bounded: GPU AND NVMe capped -> GPU flat, NVMe flat, total bounded
    capped = TieredStore([(GPUStore(), GPU_BOUND),
                          (DiskOffloadStore(directory=nvme), NVME_BOUND)])
    ctrl = build(capped)
    run(f"[3] TieredStore(GPU<={GPU_BOUND//MB}MB -> NVMe<={NVME_BOUND//MB}MB) — {CAPPED_GENS} gens",
        capped, CAPPED_GENS, checkpoints={1, 5, 10, 25, 50, 100})
    print(f"\nFinal capped split: GPU={capped.tier_bytes[0]/MB:.0f}MB  "
          f"NVMe={capped.tier_bytes[1]/MB:.0f}MB  spills={capped.num_spills}  "
          f"evictions={capped.num_evictions}")
    capped.clear()
    if os.path.isdir(nvme) and not os.environ.get("NANO_CACHE_DIR"):
        import shutil
        shutil.rmtree(nvme, ignore_errors=True)

    print("\nRead: [1] GPU grows ~{:.0f}MB/gen (unbounded continual => OOM eventually)."
          "\n      [2] GPU pinned at the bound; overflow spills to NVMe, which then grows."
          "\n      [3] GPU + NVMe both capped -> all flat, oldest generations evicted => runs forever."
          "\n  (Host RSS may still creep from glibc arena retention churning CPU staging buffers;"
          "\n   run with MALLOC_TRIM_THRESHOLD_=0 or jemalloc for a flat RSS.)"
          .format(cache_per_gen))


if __name__ == "__main__":
    main()
