"""Throughput benchmark for EDiT on a single GPU.

Because TeaCache/FBCache skip ENTIRE transformer forwards (the cost of the
denoise loop is ~99% transformer), the wall-clock speedup is governed by how
many of the N denoise steps are actually computed. This benchmark measures the
real per-forward transformer latency at a realistic Wan latent shape and reports
the resulting throughput / speedup as a function of the skip ratio. It uses
RANDOM weights (no checkpoint needed), so the *latency* numbers are real but the
*skip ratio actually achieved by TeaCache* depends on the real model + prompt +
threshold and must be measured on a real checkpoint (see the note printed).

Run: python tests/bench_throughput.py
"""

import time

import torch

from diffusers import WanTransformer3DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from edit import EditWanPipeline, apply_cache_on_transformer
from edit.cache import remove_cache_from_transformer

DEVICE = "cuda"
DTYPE = torch.bfloat16

# Wan2.1-T2V-1.3B config.
CFG_1_3B = dict(
    patch_size=(1, 2, 2), num_attention_heads=12, attention_head_dim=128,
    in_channels=16, out_channels=16, text_dim=4096, freq_dim=256,
    ffn_dim=8960, num_layers=30, cross_attn_norm=True, eps=1e-6, rope_max_seq_len=1024,
)

# Realistic 480p video latent: 480x832, 33 frames.
HEIGHT, WIDTH, NUM_FRAMES = 480, 832, 33
STEPS = 30
GUIDANCE = 5.0


def build():
    torch.manual_seed(0)
    m = WanTransformer3DModel(**CFG_1_3B).to(DEVICE, DTYPE).eval()
    n_params = sum(p.numel() for p in m.parameters())
    return m, n_params


def time_forward(transformer, latents, ts, prompt, reps=6, warmup=3):
    """Mean per-forward latency (ms) of a single transformer call."""
    for _ in range(warmup):
        transformer(hidden_states=latents, timestep=ts, encoder_hidden_states=prompt, return_dict=False)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        transformer(hidden_states=latents, timestep=ts, encoder_hidden_states=prompt, return_dict=False)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / reps


@torch.no_grad()
def main():
    transformer, n_params = build()
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
    pipe = EditWanPipeline(transformer=transformer, scheduler=scheduler)

    g = torch.Generator(device=DEVICE).manual_seed(1)
    num_latent_frames = (NUM_FRAMES - 1) // 4 + 1
    latents = torch.randn(1, 16, num_latent_frames, HEIGHT // 8, WIDTH // 8, device=DEVICE, dtype=torch.float32, generator=g)
    seq = num_latent_frames * (HEIGHT // 16) * (WIDTH // 16)
    prompt = torch.randn(1, 512, 4096, device=DEVICE, dtype=DTYPE, generator=g)
    negative = torch.randn(1, 512, 4096, device=DEVICE, dtype=DTYPE, generator=g)

    print(f"Model: Wan-1.3B config, {n_params/1e9:.2f}B params, bf16")
    print(f"Latent: {tuple(latents.shape)}  ->  seq_len={seq} tokens   steps={STEPS}  CFG={GUIDANCE}")
    print(f"VRAM allocated: {torch.cuda.memory_allocated()/1e9:.1f} GB\n")

    # --- per-forward latency (the unit the cache skips) ---
    ts = torch.tensor([500.0], device=DEVICE)
    lat_full = time_forward(transformer, latents.to(DTYPE), ts, prompt)
    print(f"Per transformer forward (full {CFG_1_3B['num_layers']} blocks): {lat_full:.1f} ms")

    # --- skip-path cost (TeaCache: 0 blocks; rel-L1 + residual add only) ---
    apply_cache_on_transformer(transformer, policy="teacache", num_inference_steps=STEPS,
                               rel_l1_thresh=1e9, coefficients=[1.0, 0.0], use_ret_steps=True)
    cache = transformer._nano_cache
    cache.reset(); cache.set_branch("cond")
    # prime the residual so the next call takes the skip path
    transformer(hidden_states=latents.to(DTYPE), timestep=ts, encoder_hidden_states=prompt, return_dict=False)
    cache.set_branch("cond")
    lat_skip = time_forward(transformer, latents.to(DTYPE), ts, prompt)
    remove_cache_from_transformer(transformer)
    print(f"Per transformer forward (TeaCache skip path):           {lat_skip:.2f} ms  "
          f"({100*lat_skip/lat_full:.1f}% of full)\n")

    # --- end-to-end baseline (no cache) ---
    torch.cuda.synchronize(); t0 = time.time()
    pipe(prompt_embeds=prompt, negative_prompt_embeds=negative, latents=latents.clone(),
         num_inference_steps=STEPS, guidance_scale=GUIDANCE, output_type="latent")
    torch.cuda.synchronize(); base_t = time.time() - t0
    print(f"End-to-end baseline (no cache): {base_t:.2f} s  ({base_t/STEPS*1000:.0f} ms/step, 2 forwards/step)\n")

    # --- speedup vs skip ratio (analytic from measured latencies, exact since skip≈free) ---
    print("Wall-clock speedup vs fraction of denoise steps skipped (TeaCache):")
    print(f"{'skip%':>6} {'computed/'+str(STEPS):>12} {'est. time':>10} {'speedup':>9}")
    for skip_frac in (0.0, 0.3, 0.4, 0.5, 0.6, 0.7):
        computed = round(STEPS * (1 - skip_frac))
        skipped = STEPS - computed
        est = 2 * (computed * lat_full + skipped * lat_skip) / 1000.0
        print(f"{skip_frac*100:5.0f}% {computed:>12} {est:9.2f}s {base_t/est:8.2f}x")

    print("\nNote: the *skip ratio actually achieved* depends on the real checkpoint,"
          "\nprompt and rel_l1_thresh. Official TeaCache reports ~0.4-0.6 skip (≈1.6-2.2x)"
          "\nfor Wan2.1 at the recommended thresholds. Run on a real checkpoint to confirm.")


if __name__ == "__main__":
    main()
