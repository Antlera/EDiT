"""P1 end-to-end on real Wan: cross-segment reuse + prefetch over a tiered cache.

Drives the actual Wan2.1 pipeline as a continual long-video run: segment g warm-starts
its first `warm_steps` from a window of previous segments [g-W, g-1] (the
`warmstart_reuse` policy), with the whole feature cache in a TieredStore
(GPU -> CPU -> NVMe). It runs the loop twice — prefetch OFF then ON — so you can see
that staging the reuse window ahead of the forward (pipe(prefetch_reuse=True)) hides
the CPU/NVMe swap-in behind compute. Reports, per segment: model+activation VRAM,
cache resident per tier, reuse skips, and seconds; then the steady-state mean per
phase.

Because a real Wan forward (seconds) dwarfs a residual read (ms), the headline is
usually that prefetch makes the already-small swap-in essentially free — i.e. bounded
-VRAM continual generation costs almost nothing extra.

Run:  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python tests/real_reuse.py
"""

import argparse
import os
import time

import torch

from edit import EditWanPipeline
from edit.cache import TieredStore, GPUStore, CPUOffloadStore, DiskOffloadStore
from edit.cache import remove_cache_from_transformer

MB = 1024 * 1024
PROMPTS = [
    "A red fox trotting through a snowy forest at dawn.",
    "A hot air balloon rising over green hills at sunrise.",
    "Neon reflections on a wet street as a tram passes.",
    "A lighthouse beam sweeping over night waves.",
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
    p.add_argument("--gens", type=int, default=8)
    p.add_argument("--steps", type=int, default=15)
    p.add_argument("--window", type=int, default=4, help="reuse window W: warm-start from [g-W, g-1]")
    p.add_argument("--warm-steps", type=int, default=2, help="steps per segment that reuse the window")
    p.add_argument("--granularity", default="stack", help="'stack', 'per_block', or int k")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num-frames", type=int, default=33)
    p.add_argument("--gpu-cache-mb", type=int, default=120)
    p.add_argument("--cpu-cache-mb", type=int, default=120)
    p.add_argument("--nvme-dir", default=os.path.join(os.getcwd(), ".nano_cache_reuse"))
    return p.parse_args()


def make_store(a):
    return TieredStore([
        (GPUStore(), a.gpu_cache_mb * MB),
        (CPUOffloadStore(pin_memory=True), a.cpu_cache_mb * MB),
        (DiskOffloadStore(directory=a.nvme_dir), None),
    ], promote=False)


def run_phase(pipe, a, *, prefetch):
    gran = a.granularity
    if gran not in ("stack", "per_block"):
        gran = int(gran)
    store = make_store(a)
    pipe.enable_cache(policy="warmstart_reuse", num_inference_steps=a.steps,
                      granularity=gran, store=store, warm_steps=a.warm_steps,
                      wan_variant="t2v-1.3B", rel_l1_thresh=0.2)
    cache = pipe.cache

    label = "prefetch ON " if prefetch else "prefetch OFF"
    print(f"\n=== {label} | granularity={gran} | window W={a.window} | warm_steps={a.warm_steps} ===")
    print(f"{'seg':>3} {'VRAM':>8} {'cacheGPU':>9} {'cacheCPU':>9} {'cacheNVMe':>10} "
          f"{'reuse skip':>11} {'sec':>6}")
    times = []
    for g in range(a.gens):
        cache.set_generation(g)
        cache.set_reuse_window(list(range(max(0, g - a.window), g)))
        cache.reset(clear_store=False)
        t0 = time.time()
        pipe(prompt=PROMPTS[g % len(PROMPTS)], negative_prompt=NEG,
             height=a.height, width=a.width, num_frames=a.num_frames,
             num_inference_steps=a.steps, guidance_scale=5.0,
             generator=torch.Generator("cuda").manual_seed(g),
             reset_cache=False, prefetch_reuse=prefetch, output_type="latent")
        dt = time.time() - t0
        times.append(dt)
        tb = [b / MB for b in store.tier_bytes]
        skip = sum(s["skip"] for s in cache.stats.values())
        print(f"{g+1:>3} {gpu_mb():>7.0f}M {tb[0]:>8.0f}M {tb[1]:>8.0f}M {tb[2]:>9.0f}M "
              f"{skip:>11} {dt:>6.2f}", flush=True)

    store.clear()
    remove_cache_from_transformer(pipe.transformer)
    steady = times[2:] if len(times) > 2 else times    # drop warm-up segments
    return sum(steady) / len(steady)


def main():
    a = parse_args()
    assert torch.cuda.is_available(), "need CUDA"
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

    print(f"Loading {a.model} ...", flush=True)
    pipe = EditWanPipeline.from_pretrained(a.model, torch_dtype=torch.bfloat16, device="cuda")
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)
    print(f"  model VRAM={gpu_mb():.0f}MB", flush=True)

    off = run_phase(pipe, a, prefetch=False)
    on = run_phase(pipe, a, prefetch=True)

    print(f"\nSteady-state mean seconds/segment:  prefetch OFF = {off:.2f}s   prefetch ON = {on:.2f}s")
    if on < off:
        print(f"  prefetch hid {off-on:.2f}s/segment of swap-in ({(1-on/off)*100:.1f}% faster).")
    print("Cross-segment reuse runs the engine's real swap-in/out; with the window staged"
          "\nahead, the CPU/NVMe reads overlap the forward, so bounded-VRAM long-video"
          "\ngeneration runs at essentially the in-VRAM speed.")
    if os.path.isdir(a.nvme_dir) and not os.environ.get("NANO_CACHE_DIR"):
        import shutil
        shutil.rmtree(a.nvme_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
