"""Storage-tier latency / VRAM trade-off benchmark for the pluggable CacheStore.

The cache skip path costs `store.get()` (bring the residual to the GPU) + one add;
each compute step costs `store.put()` (offload the residual). This benchmark measures
those two operations for GPUStore / CPUOffloadStore / DiskOffloadStore at a REALISTIC
Wan residual size, then derives how much end-to-end speedup each tier keeps versus how
much GPU VRAM it frees. No diffusers / checkpoint needed — only the residual shape.

Run:  python tests/bench_storage.py
NVMe: by default writes under ./.nano_cache_bench (set NANO_CACHE_DIR=/path/on/nvme).
      Avoid /tmp — it is often tmpfs (RAM), which would mislabel RAM as "NVMe".
"""

import os
import time

import torch

from edit.cache.store import GPUStore, CPUOffloadStore, DiskOffloadStore

# --- realistic Wan2.1-T2V-1.3B residual at 480x832x33f -----------------------
# hidden_states = [B, seq_len, inner_dim]; inner_dim = heads*head_dim = 12*128 = 1536.
# seq_len = num_latent_frames * (H/16) * (W/16) = 9 * 30 * 52 = 14040.
B, SEQ, DIM = 1, 14040, 1536
DTYPE = torch.bfloat16
NUM_LAYERS = 30                      # per_block would cache this many residuals (x2 CFG)
STEPS = 30
SKIP_FRAC = 0.5                      # typical TeaCache skip ratio
REF_FORWARD_MS = (200.0, 400.0, 600.0)   # plausible Wan-1.3B per-forward latencies; plug in yours
MB = 1024 * 1024


def residual():
    return torch.randn(B, SEQ, DIM, device="cuda", dtype=DTYPE)


def bench(fn, reps=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1000.0    # ms


def measure(name, store):
    r = residual()
    nbytes = r.element_size() * r.nelement()
    key = ("k",)

    # put: offload a freshly-computed residual (overwrites same key each rep)
    put_ms = bench(lambda: store.put(key, r))
    # get: bring the cached residual back to the GPU (the skip-path cost)
    get_ms = bench(lambda: store.get(key))

    held = store.bytes_resident if name == "GPUStore" else 0  # VRAM the cache holds
    # GPUStore.get returns a reference (held on GPU); others materialize a fresh GPU copy
    put_bw = nbytes / (put_ms / 1000.0) / 1e9
    get_bw = nbytes / (get_ms / 1000.0) / 1e9
    store.clear()
    return dict(name=name, mb=nbytes / MB, put_ms=put_ms, get_ms=get_ms,
                put_bw=put_bw, get_bw=get_bw)


def main():
    assert torch.cuda.is_available(), "need CUDA"
    r = residual()
    res_mb = r.element_size() * r.nelement() / MB
    del r
    print(f"Residual: [{B}, {SEQ}, {DIM}] {str(DTYPE).split('.')[-1]}  = {res_mb:.1f} MB each")
    print(f"per_block would hold {NUM_LAYERS} residuals x2 CFG = {2*NUM_LAYERS*res_mb/1024:.2f} GB of cache\n")

    nvme_dir = os.environ.get("NANO_CACHE_DIR", os.path.join(os.getcwd(), ".nano_cache_bench"))
    print(f"NVMe dir: {nvme_dir}  (set NANO_CACHE_DIR to a real NVMe mount; /tmp is often tmpfs)\n")

    # one elementwise add (hidden + residual) — the rest of the skip path, tier-independent
    a, b = residual(), residual()
    add_ms = bench(lambda: a.add(b))
    del a, b

    tiers = [
        ("GPUStore", GPUStore()),
        ("CPUOffloadStore", CPUOffloadStore(pin_memory=True)),
        ("CPU(unpinned)", CPUOffloadStore(pin_memory=False)),
        ("DiskOffloadStore", DiskOffloadStore(directory=nvme_dir)),
    ]
    rows = [measure(n, s) for n, s in tiers]

    print(f"reconstruct add (hidden+residual): {add_ms:.3f} ms\n")
    print(f"{'tier':18}{'put ms':>9}{'put GB/s':>10}{'get ms':>9}{'get GB/s':>10}"
          f"{'skip-hit ms':>13}{'VRAM/res':>10}")
    for r in rows:
        skip_hit = r["get_ms"] + add_ms       # cost paid on each skipped step
        vram = f"{r['mb']:.0f}MB" if r["name"] == "GPUStore" else "0MB"
        print(f"{r['name']:18}{r['put_ms']:9.3f}{r['put_bw']:10.1f}{r['get_ms']:9.3f}"
              f"{r['get_bw']:10.1f}{skip_hit:13.3f}{vram:>10}")

    # --- end-to-end trade-off: speedup kept vs VRAM freed ---------------------
    # Per CFG branch over STEPS: C computed (pay put), K skipped (pay get+add).
    C = round(STEPS * (1 - SKIP_FRAC))
    K = STEPS - C
    print(f"\nEnd-to-end (STEPS={STEPS}, skip={SKIP_FRAC:.0%} -> {C} computed / {K} skipped, x2 CFG):")
    print(f"{'forward':>9}{'baseline':>10}{'GPUStore':>20}{'CPUOffload':>20}{'NVMe':>20}")
    g = next(r for r in rows if r["name"] == "GPUStore")
    c = next(r for r in rows if r["name"] == "CPUOffloadStore")
    d = next(r for r in rows if r["name"] == "DiskOffloadStore")
    for T in REF_FORWARD_MS:
        base = 2 * STEPS * T
        def total(store_row):
            return 2 * (C * (T + store_row["put_ms"]) + K * (store_row["get_ms"] + add_ms))
        tg, tc, td = total(g), total(c), total(d)
        print(f"{T:8.0f}ms{base/1000:9.2f}s"
              f"{tg/1000:7.2f}s ({base/tg:4.2f}x)"
              f"{tc/1000:7.2f}s ({base/tc:4.2f}x)"
              f"{td/1000:7.2f}s ({base/td:4.2f}x)")

    print("\nRead: GPUStore keeps the most speedup but holds all residuals in VRAM;"
          "\nCPU/NVMe free that VRAM, the cost is the skip-hit `get` latency above."
          "\nOffload is worth it when get+put << forward time (it usually is) and VRAM is the bottleneck.")
    if os.path.isdir(nvme_dir) and not os.environ.get("NANO_CACHE_DIR"):
        import shutil
        shutil.rmtree(nvme_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
