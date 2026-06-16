"""Example: plug a brand-new cache policy into EDiT.

This defines the simplest possible cache — "interval caching": recompute the
transformer every `interval` steps and reuse the cached residual in between (no
signal, no calibration). It shows the whole extension surface: subclass
CachePolicy, implement should_compute(), register a name, then enable_cache().

Run: python examples/custom_cache_policy.py
"""

import time

import torch

from edit import EditWanPipeline
from edit.cache import CachePolicy, register_policy, remove_cache_from_transformer


# ---- the entire new strategy: ~6 lines ----
@register_policy("interval")
class IntervalCachePolicy(CachePolicy):
    """Recompute every `interval` steps; also force the first and last step."""

    def __init__(self, *, interval=3):
        self.interval = int(interval)

    def should_compute(self, ctx):
        return ctx.step % self.interval == 0 or ctx.step == ctx.num_steps - 1


MODEL = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
STEPS, GUIDANCE = 30, 5.0
PROMPT = "A cat and a dog baking a cake together in a kitchen, warm sunlight through the window."
NEGATIVE = "blurry, low quality, distorted, static"


@torch.no_grad()
def gen(pipe):
    torch.cuda.synchronize()
    t0 = time.time()
    out = pipe(
        prompt=PROMPT, negative_prompt=NEGATIVE, height=480, width=832, num_frames=33,
        num_inference_steps=STEPS, guidance_scale=GUIDANCE,
        generator=torch.Generator("cuda").manual_seed(0), output_type="latent",
    )
    torch.cuda.synchronize()
    return out, time.time() - t0


def main():
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

    pipe = EditWanPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device="cuda")
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)

    # warmup
    pipe(prompt=PROMPT, negative_prompt=NEGATIVE, height=480, width=832, num_frames=33,
         num_inference_steps=2, guidance_scale=GUIDANCE,
         generator=torch.Generator("cuda").manual_seed(0), output_type="latent")

    _, base_t = gen(pipe)
    print(f"baseline (no cache):           {base_t:5.1f} s")

    pipe.enable_cache(policy="interval", num_inference_steps=STEPS, interval=3)
    _, dt = gen(pipe)
    stats = pipe.cache.stats
    skip = sum(s["skip"] for s in stats.values())
    total = sum(s["calc"] + s["skip"] for s in stats.values())
    print(f"interval cache (every 3 steps): {dt:5.1f} s   {base_t/dt:.2f}x   "
          f"skip {skip}/{total} ({100*skip/total:.0f}%)   stats={stats}")
    remove_cache_from_transformer(pipe.transformer)


if __name__ == "__main__":
    main()
