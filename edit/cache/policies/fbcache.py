# First-Block-Cache policy (ParaAttention / xDiT style), for Wan.
#
# Signal: the unit's first-block residual (block0_out - input). The mechanism runs
# the first block for us (needs_first_block_probe) and fills ctx.first_block_residual;
# we skip the rest of the unit when that residual is close to the last computed
# step's. The reference is refreshed only on COMPUTE steps (canonical FBCache).
#
# Note: this signal is inherently whole-unit — it predicts "the rest of the unit
# from its first block". It is meaningful at "stack" granularity; at "per_block"
# granularity a unit is a single block, so the probe is the whole computation and
# nothing can be skipped. Use TeaCache (or a custom signal) for per-block research.

from edit.cache.framework import CachePolicy, register_policy, relative_l1


@register_policy("fbcache")
class FBCachePolicy(CachePolicy):
    needs_first_block_probe = True

    def __init__(self, *, rel_l1_thresh=0.1):
        self.rel_l1_thresh = float(rel_l1_thresh)

    def reset(self, state):
        state.user["prev"] = None

    def should_compute(self, ctx):
        u = ctx.state.user
        prev = u.get("prev")
        if prev is None:
            compute = True
        else:
            compute = relative_l1(ctx.first_block_residual, prev) >= self.rel_l1_thresh
        if compute:
            u["prev"] = ctx.first_block_residual.clone()
        return compute
