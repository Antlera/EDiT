"""Wan 2.1 text-to-video with EDiT TeaCache on a single GPU.

Example:
    python examples/wan_t2v_teacache.py \
        --model Wan-AI/Wan2.1-T2V-14B-Diffusers \
        --variant t2v-14B \
        --steps 50 --thresh 0.2 \
        --prompt "A cat and a dog baking a cake together in a kitchen."

Set --thresh to 0 to disable skipping (baseline), or larger (e.g. 0.2-0.3) for
more aggressive acceleration. Use --algorithm fbcache for First-Block-Cache.
"""

import argparse
import time

import torch
from diffusers.utils import export_to_video

from edit import EditWanPipeline


DEFAULT_NEGATIVE = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG "
    "compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Wan-AI/Wan2.1-T2V-14B-Diffusers")
    p.add_argument("--variant", default="t2v-14B",
                   help="coefficient set: t2v-14B / t2v-1.3B / i2v-480p / i2v-720p")
    p.add_argument("--algorithm", default="teacache", choices=["teacache", "fbcache"])
    p.add_argument("--prompt", default="A cat and a dog baking a cake together in a kitchen.")
    p.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num-frames", type=int, default=81)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance-scale", type=float, default=5.0)
    p.add_argument("--flow-shift", type=float, default=3.0, help="5.0 for 720p, 3.0 for 480p")
    p.add_argument("--thresh", type=float, default=0.2, help="cache rel-L1 threshold; 0 disables skipping")
    p.add_argument("--no-ret-steps", action="store_true", help="use the non-ret coefficient set")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="output.mp4")
    return p.parse_args()


def main():
    args = parse_args()
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

    pipe = EditWanPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16, device="cuda")
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=args.flow_shift)

    if args.thresh > 0:
        cache_kwargs = dict(
            policy=args.algorithm,
            num_inference_steps=args.steps,
            rel_l1_thresh=args.thresh,
        )
        if args.algorithm == "teacache":
            cache_kwargs.update(wan_variant=args.variant, use_ret_steps=not args.no_ret_steps)
        pipe.enable_cache(**cache_kwargs)

    def report(i, t, latents, stats):
        if stats is not None:
            print(f"  step {i:3d}  t={float(t):8.2f}  cache={stats}")

    g = torch.Generator(device="cuda").manual_seed(args.seed)
    start = time.time()
    video = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=g,
        callback_on_step=report,
    )[0]
    elapsed = time.time() - start
    print(f"Generated in {elapsed:.1f}s")
    if pipe.cache is not None:
        print(f"Final cache stats: {pipe.cache.stats}")

    export_to_video(video, args.out, fps=16)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
