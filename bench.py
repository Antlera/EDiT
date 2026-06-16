"""Simple end-to-end benchmark: baseline (no cache) vs TeaCache on a real
Wan2.1-T2V-1.3B checkpoint. Prints a Markdown table of time / speedup / skip
ratio. Downloads the model on first run.

Run: python bench.py
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
FLOW_SHIFT = 3.0
SEED = 0
THRESHOLDS = [0.1, 0.2, 0.3]

PROMPT = "A cat and a dog baking a cake together in a kitchen, warm sunlight through the window."
NEGATIVE = "blurry, low quality, distorted, static, overexposed, worst quality"


@torch.no_grad()
def gen(pipe):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    out = pipe(
        prompt=PROMPT, negative_prompt=NEGATIVE,
        height=HEIGHT, width=WIDTH, num_frames=NUM_FRAMES,
        num_inference_steps=STEPS, guidance_scale=GUIDANCE,
        generator=torch.Generator("cuda").manual_seed(SEED), output_type="latent",
    )
    torch.cuda.synchronize()
    return out, time.time() - t0, torch.cuda.max_memory_allocated() / 1e9


def main():
    pipe = EditWanPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device="cuda")
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=FLOW_SHIFT)

    # warmup (cudnn autotune) so the timed baseline is steady-state
    pipe(prompt=PROMPT, negative_prompt=NEGATIVE, height=HEIGHT, width=WIDTH, num_frames=NUM_FRAMES,
         num_inference_steps=2, guidance_scale=GUIDANCE,
         generator=torch.Generator("cuda").manual_seed(SEED), output_type="latent")

    base_out, base_t, base_mem = gen(pipe)

    rows = [("baseline (no cache)", base_t, 1.0, 0, base_mem)]
    for thr in THRESHOLDS:
        pipe.enable_cache(policy="teacache", num_inference_steps=STEPS,
                          rel_l1_thresh=thr, wan_variant=VARIANT, use_ret_steps=True)
        _, dt, mem = gen(pipe)
        stats = pipe.cache.stats
        total = sum(s["calc"] + s["skip"] for s in stats.values())
        skip = sum(s["skip"] for s in stats.values())
        rows.append((f"TeaCache thr={thr}", dt, base_t / dt, round(100 * skip / total), mem))
        remove_cache_from_transformer(pipe.transformer)

    print(f"\nWan2.1-T2V-1.3B | {HEIGHT}x{WIDTH} x {NUM_FRAMES}f | {STEPS} steps | CFG {GUIDANCE}\n")
    print("| Config | Time (s) | Speedup | Steps skipped | Peak VRAM (GB) |")
    print("|---|---|---|---|---|")
    for name, t, sp, skp, mem in rows:
        print(f"| {name} | {t:.1f} | {sp:.2f}x | {skp}% | {mem:.1f} |")


if __name__ == "__main__":
    main()
