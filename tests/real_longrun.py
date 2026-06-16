"""Real Wan continual-generation long-run with a tiered (GPU -> NVMe) cache.

Unlike tests/bench_longrun.py (synthetic blocks, isolates storage), this drives the
ACTUAL Wan2.1 text-to-video pipeline back-to-back, tagging each generation with a
fresh gen_id so the feature cache accumulates across generations — the continual
case that grows unbounded. The cache lives in a TieredStore:

    GPU (bounded) -> NVMe (bounded)

so VRAM held by the cache is pinned at the GPU bound, the overflow spills to NVMe,
and once NVMe hits its bound the oldest generations are evicted => it runs forever.

Per generation we print: model+activation VRAM, host RSS, the cache's resident
split (GPU MB / NVMe MB), spills/evictions, TeaCache calc/skip, and seconds.

Run (pick a free GPU):
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python tests/real_longrun.py --gens 8
"""

import argparse
import os
import time

import torch

from edit import EditWanPipeline
from edit.cache import TieredStore, GPUStore, DiskOffloadStore

MB = 1024 * 1024

PROMPTS = [
    "A cat and a dog baking a cake together in a kitchen, warm sunlight.",
    "A red fox trotting through a snowy forest at dawn.",
    "A paper boat sailing down a rain-soaked city gutter.",
    "A hummingbird sipping from a bright orange flower, slow motion.",
    "A lighthouse beam sweeping over crashing night waves.",
    "A potter's hands shaping a clay bowl on a spinning wheel.",
    "A hot air balloon rising over green rolling hills at sunrise.",
    "Neon reflections on a wet street as a tram passes by.",
]

NEG = "blurry, low quality, distorted, static, watermark"


def rss_mb():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / MB


def gpu_mb():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / MB


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    p.add_argument("--variant", default="t2v-1.3B")
    p.add_argument("--gens", type=int, default=8, help="number of back-to-back generations; <=0 runs forever")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num-frames", type=int, default=33)
    p.add_argument("--thresh", type=float, default=0.2)
    p.add_argument("--gpu-cache-mb", type=int, default=512, help="cache bytes kept on GPU")
    p.add_argument("--nvme-cache-mb", type=int, default=4096, help="cache bytes kept on NVMe")
    p.add_argument("--nvme-dir", default=os.path.join(os.getcwd(), ".nano_cache_real"))
    p.add_argument("--save-each", action="store_true", help="export every gen's mp4 (else only the first)")
    return p.parse_args()


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "need CUDA"
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from diffusers.utils import export_to_video

    print(f"Loading {args.model} (first run downloads weights)...", flush=True)
    t = time.time()
    pipe = EditWanPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16, device="cuda")
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
    print(f"  loaded in {time.time()-t:.1f}s   model VRAM={gpu_mb():.0f}MB", flush=True)

    store = TieredStore([
        (GPUStore(), args.gpu_cache_mb * MB),
        (DiskOffloadStore(directory=args.nvme_dir), args.nvme_cache_mb * MB),
    ])
    pipe.enable_cache(
        policy="teacache",
        num_inference_steps=args.steps,
        rel_l1_thresh=args.thresh,
        wan_variant=args.variant,
        store=store,
    )
    cache = pipe.cache

    forever = args.gens <= 0
    print(f"\nTieredStore: GPU<={args.gpu_cache_mb}MB -> NVMe<={args.nvme_cache_mb}MB  ({args.nvme_dir})")
    print(f"{'∞' if forever else args.gens} generations x {args.steps} steps "
          f"@ {args.width}x{args.height}x{args.num_frames}f\n")
    hdr = f"{'gen':>4} {'VRAM':>8} {'RSS':>8} {'cacheGPU':>9} {'cacheNVMe':>10} {'spill':>6} {'evict':>6} {'skip':>12} {'sec':>6}"
    print(hdr)
    print("-" * len(hdr))

    t_all = time.time()
    g = -1
    while forever or g + 1 < args.gens:
        g += 1
        cache.set_generation(g)          # fresh gen_id => cache accumulates (continual)
        cache.reset(clear_store=False)   # reset control state, keep stored tensors
        t0 = time.time()
        video = pipe(
            prompt=PROMPTS[g % len(PROMPTS)],
            negative_prompt=NEG,
            height=args.height, width=args.width, num_frames=args.num_frames,
            num_inference_steps=args.steps, guidance_scale=5.0,
            generator=torch.Generator("cuda").manual_seed(g),
            reset_cache=False,           # keep the tiered store across generations
        )[0]
        dt = time.time() - t0

        gpu_c = store.tier_bytes[0] / MB
        nvme_c = store.tier_bytes[1] / MB
        st = cache.stats
        skip = "/".join(f"{b}:{d['skip']}/{d['calc']+d['skip']}" for b, d in sorted(st.items()))
        print(f"{g+1:>4} {gpu_mb():>7.0f}M {rss_mb():>7.0f}M {gpu_c:>8.0f}M {nvme_c:>9.0f}M "
              f"{store.num_spills:>6} {store.num_evictions:>6} {skip:>12} {dt:>6.1f}", flush=True)

        if g == 0 or args.save_each:
            out = f"real_gen{g}.mp4" if args.save_each else "real_gen0.mp4"
            try:
                export_to_video(video, out, fps=16)
            except Exception as e:
                print(f"     (video export skipped: {type(e).__name__}: {e})", flush=True)

    print(f"\nTotal {args.gens} gens in {time.time()-t_all:.1f}s")
    print(f"Final cache split: GPU={store.tier_bytes[0]/MB:.0f}MB  NVMe={store.tier_bytes[1]/MB:.0f}MB  "
          f"spills={store.num_spills}  evictions={store.num_evictions}")
    print("VRAM held by the cache stays pinned at the GPU bound; overflow lives on NVMe;\n"
          "once NVMe is full the oldest generations are evicted => generation never stops.")


if __name__ == "__main__":
    main()
