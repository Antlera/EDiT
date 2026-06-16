"""Minimal EDiT text-to-video example with TeaCache.

Downloads Wan2.1-T2V-1.3B on first run. See examples/wan_t2v_teacache.py for a
full CLI (FBCache, thresholds, per-step cache stats, i2v coefficients).
"""

import torch
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video

from edit import EditWanPipeline

pipe = EditWanPipeline.from_pretrained(
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", torch_dtype=torch.bfloat16, device="cuda"
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)

# Enable TeaCache step-skipping (rel_l1_thresh: larger => faster, lower fidelity).
pipe.enable_cache(
    policy="teacache",             # or "fbcache", "perblock_demo", or your own
    num_inference_steps=30,
    rel_l1_thresh=0.2,
    wan_variant="t2v-1.3B",
)

video = pipe(
    prompt="A cat and a dog baking a cake together in a kitchen, warm sunlight through the window.",
    negative_prompt="blurry, low quality, distorted, static",
    height=480,
    width=832,
    num_frames=33,
    num_inference_steps=30,
    guidance_scale=5.0,
    generator=torch.Generator("cuda").manual_seed(0),
)[0]

export_to_video(video, "output.mp4", fps=16)
print("saved output.mp4 |", pipe.cache.stats)
