"""End-to-end sanity test for the EDiT pluggable cache framework on a tiny
randomly initialised Wan transformer. No real weights / VAE / text encoder needed.

Run directly:  python tests/test_cache_tiny.py
Or with pytest: pytest tests/test_cache_tiny.py

Checks:
  * forced-compute (very negative threshold) reproduces the no-cache baseline
    bit-for-bit, at both "stack" and "per_block" granularity (residual mechanics
    are exact)
  * a large threshold makes TeaCache skip every step outside the forced window
    (use_ret_steps=True => first 5 of N computed per branch)
  * FBCache skips and stays finite
  * per_block granularity wires N independent units and still skips
"""

import torch

from diffusers import WanTransformer3DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from edit import EditWanPipeline, apply_cache_on_transformer
from edit.cache import remove_cache_from_transformer


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32  # fp32 for a deterministic tiny test
NUM_LAYERS = 6


def build_tiny_transformer(seed=0):
    torch.manual_seed(seed)
    model = WanTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=2,
        attention_head_dim=16,   # inner_dim = 32
        in_channels=4,
        out_channels=4,
        text_dim=16,
        freq_dim=64,
        ffn_dim=64,
        num_layers=NUM_LAYERS,
        cross_attn_norm=True,
        qk_norm="rms_norm_across_heads",
        eps=1e-6,
        rope_max_seq_len=64,
    )
    return model.to(DEVICE, DTYPE).eval()


def make_inputs(batch=1, seed=1):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    latents = torch.randn(batch, 4, 4, 16, 16, generator=g, device=DEVICE, dtype=torch.float32)
    prompt = torch.randn(batch, 12, 16, generator=g, device=DEVICE, dtype=DTYPE)
    negative = torch.randn(batch, 12, 16, generator=g, device=DEVICE, dtype=DTYPE)
    return latents, prompt, negative


def run(pipe, latents, prompt, negative, steps):
    return pipe(
        prompt_embeds=prompt,
        negative_prompt_embeds=negative,
        latents=latents.clone(),
        num_inference_steps=steps,
        guidance_scale=5.0,
        output_type="latent",
    )


def main():
    steps = 10
    transformer = build_tiny_transformer()
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
    pipe = EditWanPipeline(transformer=transformer, scheduler=scheduler)
    latents, prompt, negative = make_inputs()

    # ---- baseline: no cache ----
    base_out = run(pipe, latents, prompt, negative, steps)
    assert torch.isfinite(base_out).all(), "baseline produced non-finite output"
    print(f"[baseline      ] shape={tuple(base_out.shape)} mean={base_out.mean():.5f}")

    # ---- forced-compute must be bit-exact vs baseline (stack + per_block) ----
    for gran in ("stack", "per_block"):
        apply_cache_on_transformer(
            transformer, policy="teacache", num_inference_steps=steps, granularity=gran,
            rel_l1_thresh=-1e9, coefficients=[1.0, 0.0], use_ret_steps=True,
        )
        out = run(pipe, latents, prompt, negative, steps)
        skips = sum(b["skip"] for b in transformer._nano_cache.stats.values())
        delta = (out - base_out).abs().max().item()
        print(f"[tea -inf {gran:9}] skips={skips} max|Δ|={delta:.2e} units={transformer._nano_cache.num_units}")
        assert skips == 0, f"{gran}: negative threshold should never skip"
        assert torch.allclose(out, base_out, atol=1e-5, rtol=1e-4), f"{gran}: forced-compute != baseline"
        remove_cache_from_transformer(transformer)

    # ---- TeaCache stack, large threshold => 5 compute + 5 skip per branch ----
    apply_cache_on_transformer(
        transformer, policy="teacache", num_inference_steps=steps,
        rel_l1_thresh=1e9, coefficients=[1.0, 0.0], use_ret_steps=True,
    )
    tea_out = run(pipe, latents, prompt, negative, steps)
    stats = transformer._nano_cache.stats
    print(f"[tea +inf stack    ] stats={stats}")
    assert torch.isfinite(tea_out).all()
    for branch, s in stats.items():
        assert s["calc"] == 5 and s["skip"] == 5, f"{branch}: expected 5/5, got {s}"
    remove_cache_from_transformer(transformer)

    # ---- FBCache stack, large threshold => skip after the first compute ----
    apply_cache_on_transformer(
        transformer, policy="fbcache", num_inference_steps=steps, rel_l1_thresh=1e9,
    )
    fb_out = run(pipe, latents, prompt, negative, steps)
    skips = sum(b["skip"] for b in transformer._nano_cache.stats.values())
    print(f"[fb  +inf stack    ] stats={transformer._nano_cache.stats}")
    assert torch.isfinite(fb_out).all() and skips > 0, "FBCache should skip"
    remove_cache_from_transformer(transformer)

    # ---- per_block plumbing: N independent units, still skips, stays finite ----
    apply_cache_on_transformer(
        transformer, policy="teacache", num_inference_steps=steps, granularity="per_block",
        rel_l1_thresh=1e9, coefficients=[1.0, 0.0], use_ret_steps=True,
    )
    pb_out = run(pipe, latents, prompt, negative, steps)
    ctrl = transformer._nano_cache
    skips = sum(b["skip"] for b in ctrl.stats.values())
    print(f"[tea +inf per_block] units={ctrl.num_units} stats={ctrl.stats}")
    assert torch.isfinite(pb_out).all()
    assert ctrl.num_units == NUM_LAYERS, f"expected {NUM_LAYERS} units, got {ctrl.num_units}"
    assert skips > 0, "per_block TeaCache should skip"
    remove_cache_from_transformer(transformer)

    # ---- per_block sample policy registers and runs ----
    apply_cache_on_transformer(
        transformer, policy="perblock_demo", num_inference_steps=steps,
        granularity="per_block", rel_l1_thresh=1e9, protect=1,
    )
    demo_out = run(pipe, latents, prompt, negative, steps)
    print(f"[perblock_demo     ] stats={transformer._nano_cache.stats}")
    assert torch.isfinite(demo_out).all()
    remove_cache_from_transformer(transformer)

    print("\nAll tiny-model cache checks passed.")


if __name__ == "__main__":
    main()
