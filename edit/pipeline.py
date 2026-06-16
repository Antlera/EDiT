# EditWanPipeline: an explicit, instrumentable Wan text-to-video denoising loop.
#
# This deliberately reimplements the denoising loop of diffusers' WanPipeline
# (instead of calling its black-box __call__) so every step is visible for
# research: you can inspect latents per step, read cache hit/skip statistics, and
# drop in a per-step callback. It reuses the diffusers components unchanged
# (UMT5 text encoder, AutoencoderKLWan, FlowMatchEulerDiscreteScheduler,
# WanTransformer3DModel, VideoProcessor).
#
# Classifier-free guidance is run as two separate transformer forwards
# (conditional, then unconditional); before each, the feature cache (if enabled)
# is told which branch it is writing, so the two passes keep independent caches —
# exactly as the official TeaCache even/odd buffers do.

import torch

from edit.cache import apply_cache_on_transformer


class EditWanPipeline:
    def __init__(
        self,
        transformer,
        scheduler,
        vae=None,
        text_encoder=None,
        tokenizer=None,
        video_processor=None,
        vae_scale_factor_temporal: int = 4,
        vae_scale_factor_spatial: int = 8,
        base=None,
    ):
        self.transformer = transformer
        self.scheduler = scheduler
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.video_processor = video_processor
        self.vae_scale_factor_temporal = vae_scale_factor_temporal
        self.vae_scale_factor_spatial = vae_scale_factor_spatial
        # Optional underlying diffusers WanPipeline, used only to delegate
        # prompt encoding (UMT5 tokenization is verbose to reproduce).
        self._base = base

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_pretrained(cls, model_id: str, *, torch_dtype=torch.bfloat16, device="cuda", **kwargs):
        """Load a diffusers WanPipeline and wrap its components."""
        from diffusers import AutoencoderKLWan, WanPipeline

        vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
        base = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch_dtype, **kwargs)
        base.to(device)
        return cls(
            transformer=base.transformer,
            scheduler=base.scheduler,
            vae=base.vae,
            text_encoder=base.text_encoder,
            tokenizer=base.tokenizer,
            video_processor=base.video_processor,
            vae_scale_factor_temporal=base.vae_scale_factor_temporal,
            vae_scale_factor_spatial=base.vae_scale_factor_spatial,
            base=base,
        )

    def enable_cache(self, **kwargs):
        """Install a TeaCache / FBCache on the transformer. See
        apply_cache_on_transformer for the arguments."""
        apply_cache_on_transformer(self.transformer, **kwargs)
        return self

    @property
    def cache(self):
        return getattr(self.transformer, "_nano_cache", None)

    @property
    def device(self):
        return self.transformer.device

    @property
    def dtype(self):
        return self.transformer.dtype

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def encode_prompt(self, prompt, negative_prompt, do_cfg, max_sequence_length=512):
        if self._base is None:
            raise RuntimeError(
                "No text encoder available; either build EditWanPipeline via "
                "from_pretrained, or pass prompt_embeds/negative_prompt_embeds to __call__."
            )
        prompt_embeds, negative_prompt_embeds = self._base.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_cfg,
            max_sequence_length=max_sequence_length,
            device=self.device,
        )
        return prompt_embeds, negative_prompt_embeds

    def prepare_latents(self, batch_size, num_channels_latents, height, width, num_frames, dtype, device, generator):
        from diffusers.utils.torch_utils import randn_tensor

        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )
        return randn_tensor(shape, generator=generator, device=device, dtype=dtype)

    def decode_latents(self, latents, output_type="np"):
        latents = latents.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        video = self.vae.decode(latents, return_dict=False)[0]
        return self.video_processor.postprocess_video(video, output_type=output_type)

    # ------------------------------------------------------------------ #
    # Denoising loop
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def __call__(
        self,
        prompt=None,
        negative_prompt=None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        generator=None,
        latents=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        output_type: str = "np",
        max_sequence_length: int = 512,
        callback_on_step=None,
        reset_cache: bool = True,
    ):
        device = self.device
        transformer_dtype = self.dtype
        do_cfg = guidance_scale > 1.0

        # 1. Prompt embeddings
        if prompt_embeds is None:
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt, negative_prompt, do_cfg, max_sequence_length=max_sequence_length
            )
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)
        batch_size = prompt_embeds.shape[0]

        # 2. Timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        self.scheduler.set_begin_index(0)

        # 3. Latents
        num_channels_latents = self.transformer.config.in_channels
        if latents is None:
            latents = self.prepare_latents(
                batch_size, num_channels_latents, height, width, num_frames, torch.float32, device, generator
            )
        else:
            latents = latents.to(device=device, dtype=torch.float32)

        # 4. Sync the cache to this run (step count drives counter wrap + windows).
        #    reset_cache=False keeps the stored residuals/history across calls, so a
        #    continual generation can warm-start from the previous one.
        cache = self.cache
        if cache is not None:
            cache.configure_steps(num_inference_steps)
            if reset_cache:
                cache.reset()

        # 5. Denoising loop (explicit)
        for i, t in enumerate(timesteps):
            latent_model_input = latents.to(transformer_dtype)
            timestep = t.expand(latents.shape[0])

            if cache is not None:
                cache.set_branch("cond")
            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]

            if do_cfg:
                if cache is not None:
                    cache.set_branch("uncond")
                noise_uncond = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=negative_prompt_embeds,
                    return_dict=False,
                )[0]
                noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if callback_on_step is not None:
                callback_on_step(
                    i, t, latents, cache.stats if cache is not None else None
                )

        # 6. Decode
        if output_type == "latent":
            return latents
        return self.decode_latents(latents, output_type=output_type)
