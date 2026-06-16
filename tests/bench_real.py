"""Real-checkpoint end-to-end throughput benchmark: baseline vs TeaCache.

Downloads Wan2.1-T2V-1.3B-Diffusers on first run, then measures the ACTUAL
skip ratio and wall-clock speedup TeaCache achieves at a few thresholds.

Run: python tests/bench_real.py
"""

import time

import torch

from edit import EditWanPipeline
from edit.cache import remove_cache_from_transformer

MODEL = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
VARIANT = "t2v-1.3B"
HEIGHT, WIDTH, NUM_FRAMES = 480, 832, 33
STEPS = 30
GUIDANCE = 5.0
FLOW_SHIFT = 3.0  # 480p
SEED = 0

PROMPT = "A cat and a dog baking a cake together in a kitchen, warm sunlight through the window."
NEGATIVE = (
    "Bright tones, overexposed, static, blurred details, subtitles, worst quality, "
    "low quality, JPEG compression residue, ugly, deformed, disfigured, messy background."
)
THRESHOLDS = [0.1, 0.2]


def gen(pipe):
    torch.cuda.reset_peak_memory_stats()
    g = torch.Generator(device="cuda").manual_seed(SEED)
    torch.cuda.synchronize()
    t0 = time.time()
    out = pipe(
        prompt=PROMPT, negative_prompt=NEGATIVE,
        height=HEIGHT, width=WIDTH, num_frames=NUM_FRAMES,
        num_inference_steps=STEPS, guidance_scale=GUIDANCE,
        generator=g, output_type="latent",
    )
    torch.cuda.synchronize()
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9
    return out, dt, peak


def main():
    print(f"Loading {MODEL} ...")
    pipe = EditWanPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device="cuda")
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=FLOW_SHIFT)
    print(f"Loaded. Setup: {HEIGHT}x{WIDTH} x {NUM_FRAMES}f, {STEPS} steps, CFG {GUIDANCE}\n")

    # warmup (cudnn autotune / allocs) so the timed baseline is steady-state
    _ = pipe(prompt=PROMPT, negative_prompt=NEGATIVE, height=HEIGHT, width=WIDTH,
             num_frames=NUM_FRAMES, num_inference_steps=2, guidance_scale=GUIDANCE,
             generator=torch.Generator(device="cuda").manual_seed(SEED), output_type="latent")

    base_out, base_t, base_peak = gen(pipe)
    print(f"[baseline no-cache] {base_t:6.2f} s   peak {base_peak:5.1f} GB")

    for thr in THRESHOLDS:
        pipe.enable_cache(algorithm="teacache", num_inference_steps=STEPS,
                          rel_l1_thresh=thr, wan_variant=VARIANT, use_ret_steps=True)
        out, dt, peak = gen(pipe)
        stats = pipe.cache.stats
        total = sum(s["calc"] + s["skip"] for s in stats.values())
        skipped = sum(s["skip"] for s in stats.values())
        # latent L2 difference vs baseline as a coarse quality proxy
        rel = (out - base_out).norm() / base_out.norm()
        print(f"[teacache thr={thr:<4}] {dt:6.2f} s   speedup {base_t/dt:4.2f}x   "
              f"skip {skipped}/{total} ({100*skipped/total:.0f}%)   "
              f"peak {peak:5.1f} GB   latent_rel_l2 {rel:.3f}   stats={stats}")
        remove_cache_from_transformer(pipe.transformer)

    print("\nDone. (skip% and speedup here are the REAL values for this prompt/schedule.)")


if __name__ == "__main__":
    main()
