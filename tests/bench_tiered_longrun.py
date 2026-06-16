"""Tiered-cache long-run monitor: per-tier memory growth + the speed cost of depth.

A long, continual generation run through a 3-tier feature cache

    GPU (bounded) -> CPU (bounded) -> NVMe (bounded)

at a realistic Wan residual size. Two things are measured and dumped to JSON for
plotting (tests/plot_tiered_longrun.py):

  1. MEMORY OVER TIME. Each "generation" writes its units' residuals (store.put).
     New entries land on the GPU tier; once a tier is over budget its least-recently
     -used entries SPILL DOWN to the next tier. So the cache fills GPU, then CPU,
     then NVMe, and each tier's resident bytes are pinned at its bound forever —
     the "runs indefinitely without OOM" story. We record per-tier MB every gen.

  2. THE COST OF TIER DEPTH. Two angles:
     - per generation, the wall-clock spent MOVING the overflow down the tiers
       (the price you pay to keep VRAM bounded) — near-zero while it fits on the
       GPU, rising once data must be evacuated to CPU / NVMe.
     - a clean micro-benchmark of one cached residual's round trip (put + get) on
       each tier in isolation (warmup + reps): GPU (a reference, ~free) << CPU (one
       H2D/D2H copy) < NVMe (a file write + read). This is the monotonic unit cost.

Synthetic on purpose — no checkpoint, so the storage cost is measured in isolation
rather than hidden behind the transformer forward.

Run:  python tests/bench_tiered_longrun.py
NVMe: writes under ./.nano_cache_tiered (set NANO_CACHE_DIR=/path/on/real/nvme;
      avoid /tmp, which is often tmpfs / RAM and would mislabel RAM as NVMe).
Then: python tests/plot_tiered_longrun.py   to render PNGs + the GIF into assets/.
"""

import json
import os
import time

import torch

from edit.cache.store import GPUStore, CPUOffloadStore, DiskOffloadStore, TieredStore

MB = 1024 * 1024

# --- realistic-ish Wan residual; scaled so all three tiers engage within the run ---
B, SEQ, DIM = 1, 4096, 1536          # bf16 => 12.0 MB per residual
DTYPE = torch.bfloat16
UNITS = 8                            # per_block-style: this many cache units ...
BRANCHES = ("cond", "uncond")        # ... times 2 CFG branches per generation
GENS = 30
READS_PER_GEN = 16                   # fixed read budget, spread across all history

GPU_BOUND = 384 * MB                 # ~2 generations of cache stay on the GPU
CPU_BOUND = 768 * MB                 # ~4 more generations in host RAM
NVME_BOUND = 2304 * MB               # ~12 more on NVMe, then the oldest are evicted

TIER_NAMES = ("GPU", "CPU", "NVMe")
DATA_PATH = os.path.join(os.path.dirname(__file__), "tiered_longrun_data.json")


def residual():
    return torch.randn(B, SEQ, DIM, device="cuda", dtype=DTYPE)


def even_spread(hi, n):
    """n indices spread evenly over [0, hi] inclusive (oldest..newest), deduped."""
    if hi <= 0:
        return [0]
    return sorted({round(i * hi / (n - 1)) for i in range(n)}) if n > 1 else [hi]


def tier_of(store, key):
    """Which tier a key lives in (0/1/2), or -1 if evicted. promote=False, so a read
    does not move it — we can classify before timing the get."""
    loc = store._loc.get(key)
    return loc[0] if loc is not None else -1


def bench_op(fn, reps=30, warmup=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1000.0    # ms


def tier_unit_cost(nvme_dir):
    """Clean per-tier round-trip (put + get) latency for one residual, in isolation."""
    r = residual()
    out = []
    for name, store in (("GPU", GPUStore()),
                        ("CPU", CPUOffloadStore(pin_memory=True)),
                        ("NVMe", DiskOffloadStore(directory=nvme_dir))):
        put_ms = bench_op(lambda: store.put(("k",), r))
        get_ms = bench_op(lambda: store.get(("k",)))
        store.clear()
        out.append(dict(tier=name, put_ms=put_ms, get_ms=get_ms, roundtrip_ms=put_ms + get_ms))
    del r
    return out


def run():
    assert torch.cuda.is_available(), "need CUDA"
    nvme = os.environ.get("NANO_CACHE_DIR", os.path.join(os.getcwd(), ".nano_cache_tiered"))
    res_mb = B * SEQ * DIM * 2 / MB
    per_gen_mb = UNITS * len(BRANCHES) * res_mb

    store = TieredStore(
        [(GPUStore(), GPU_BOUND),
         (CPUOffloadStore(pin_memory=True), CPU_BOUND),
         (DiskOffloadStore(directory=nvme), NVME_BOUND)],
        promote=False,               # reads do not reshuffle tiers (clean measurement)
    )

    print(f"Residual [{B},{SEQ},{DIM}] {str(DTYPE).split('.')[-1]} = {res_mb:.1f}MB | "
          f"{UNITS} units x{len(BRANCHES)} CFG = {per_gen_mb:.0f}MB cache written per generation")
    print(f"Tiers: GPU<={GPU_BOUND//MB}MB -> CPU<={CPU_BOUND//MB}MB -> NVMe<={NVME_BOUND//MB}MB   ({nvme})")
    print(f"Each generation also reads {READS_PER_GEN} residuals spread across all history.\n")

    hdr = (f"{'gen':>4} {'GPU MB':>7} {'CPU MB':>7} {'NVMe MB':>8} {'total':>7} "
           f"{'reads G/C/N/miss':>17} {'move ms':>8} {'read ms':>8} {'ms/gen':>7}")
    print(hdr); print("-" * len(hdr))

    trace = []
    for g in range(GENS):
        # ---- WRITE this generation's residuals (grows the cache, triggers spills) ----
        torch.cuda.synchronize(); tw = time.perf_counter()
        for u in range(UNITS):
            for br in BRANCHES:
                store.put((g, u, br, "res"), residual())
        torch.cuda.synchronize(); move_ms = (time.perf_counter() - tw) * 1000

        # ---- READ a fixed spread across history; classify tier, then time the get ----
        gens = even_spread(g, READS_PER_GEN)
        keys = [(gi, gi % UNITS, "cond", "res") for gi in gens]
        hits = [0, 0, 0]; miss = 0
        for k in keys:
            t = tier_of(store, k)
            if t < 0:
                miss += 1
            else:
                hits[t] += 1
        torch.cuda.synchronize(); tr = time.perf_counter()
        for k in keys:
            store.get(k)
        torch.cuda.synchronize(); read_ms = (time.perf_counter() - tr) * 1000

        tb = [b / MB for b in store.tier_bytes]
        trace.append(dict(gen=g + 1, gpu=tb[0], cpu=tb[1], nvme=tb[2], total=sum(tb),
                          hits=hits, miss=miss, move_ms=move_ms, read_ms=read_ms,
                          ms=move_ms + read_ms))
        r = trace[-1]
        print(f"{r['gen']:>4} {r['gpu']:>7.0f} {r['cpu']:>7.0f} {r['nvme']:>8.0f} {r['total']:>7.0f} "
              f"{f'{hits[0]}/{hits[1]}/{hits[2]}/{miss}':>17} "
              f"{r['move_ms']:>8.1f} {r['read_ms']:>8.1f} {r['ms']:>7.1f}", flush=True)

    print("\nMeasuring clean per-tier unit cost (put+get of one residual)...", flush=True)
    store.clear()
    tier_cost = tier_unit_cost(nvme)

    def first(pred):
        return next((r["gen"] for r in trace if pred(r)), None)
    engage = dict(cpu=first(lambda r: r["cpu"] > 0),
                  nvme=first(lambda r: r["nvme"] > 0),
                  evict=first(lambda r: r["miss"] > 0))

    data = dict(
        config=dict(res_mb=res_mb, units=UNITS, branches=len(BRANCHES), gens=GENS,
                    per_gen_mb=per_gen_mb, reads_per_gen=READS_PER_GEN,
                    gpu_bound_mb=GPU_BOUND / MB, cpu_bound_mb=CPU_BOUND / MB,
                    nvme_bound_mb=NVME_BOUND / MB, tier_names=list(TIER_NAMES)),
        trace=trace, tier_cost=tier_cost, engage=engage,
    )
    with open(DATA_PATH, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nTiers engage: CPU at gen {engage['cpu']}, NVMe at gen {engage['nvme']}, "
          f"eviction (NVMe full) at gen {engage['evict']}  =>  footprint bounded, runs forever.")
    print("Per-residual round trip (put+get): " +
          "  ".join(f"{t['tier']}={t['roundtrip_ms']:.3f}ms" for t in tier_cost))
    gpu_rt = next(t["roundtrip_ms"] for t in tier_cost if t["tier"] == "GPU")
    for t in tier_cost:
        if t["tier"] != "GPU":
            print(f"    {t['tier']} costs {t['roundtrip_ms']/max(gpu_rt,1e-6):.0f}x a GPU-resident hit "
                  f"(put {t['put_ms']:.3f}ms + get {t['get_ms']:.3f}ms)")
    print(f"\nWrote {DATA_PATH}\nNext: python tests/plot_tiered_longrun.py")

    store.clear()
    if os.path.isdir(nvme) and not os.environ.get("NANO_CACHE_DIR"):
        import shutil
        shutil.rmtree(nvme, ignore_errors=True)


if __name__ == "__main__":
    run()
