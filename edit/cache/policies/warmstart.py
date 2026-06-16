# Warm-start reuse policy — TeaCache plus cross-segment warm start.
#
# For continual / long-video generation: the first `warm_steps` of each segment reuse
# the PREVIOUS segment(s)' residual (the controller's reuse window) instead of
# recomputing, on the assumption that consecutive segments are similar. After the
# warm-up the policy is exactly TeaCache. The mechanism falls back to a normal
# recompute whenever the reuse window has nothing cached (segment 0, or evicted), so
# this is always safe. Pair with:
#
#     ctrl.set_reuse_window([g - 1, ...])     # which prior segments to warm-start from
#     pipe(..., reset_cache=False, prefetch_reuse=True)   # overlap the swap-in
#
# prefetch_reuse stages those residuals toward the GPU before the forward, so the
# warm-start reads from CPU/NVMe overlap compute instead of stalling it.

from edit.cache.framework import register_policy
from edit.cache.policies.teacache import TeaCachePolicy


@register_policy("warmstart_reuse")
class WarmStartReusePolicy(TeaCachePolicy):
    def __init__(self, *, warm_steps=1, **teacache_kwargs):
        super().__init__(**teacache_kwargs)
        self.warm_steps = int(warm_steps)

    def should_compute(self, ctx):
        # Request reuse for the opening steps; CachedUnit blends the reuse window and,
        # if nothing is cached there, recomputes (safe). Afterwards: plain TeaCache.
        if ctx.step < self.warm_steps:
            return False
        return super().should_compute(ctx)
