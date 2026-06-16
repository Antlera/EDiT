# Template policy for PER-BLOCK cache research.
#
# This is a minimal example showing how to write a policy that behaves differently
# per block, using the per-unit context the framework provides. Install with
# granularity="per_block" so each block is its own cache unit with independent
# state:
#
#   apply_cache_on_transformer(transformer, policy="perblock_demo",
#                              num_inference_steps=N, granularity="per_block")
#
# It demonstrates the two things per-block research needs:
#   - ctx.unit_index / ctx.num_units : act differently per block (here: always
#     compute the first/last `protect` blocks, which tend to be quality-critical).
#   - ctx.state.user                 : independent per-(block, branch) scratch.
#
# The decision signal here is just e0 stability (cheap, no extra block runs), so a
# block reuses its own cached residual while the timestep embedding is stable.
# Swap in a smarter per-block signal to do real research.

from edit.cache.framework import CachePolicy, register_policy, relative_l1


@register_policy("perblock_demo")
class PerBlockDemoPolicy(CachePolicy):
    def __init__(self, *, rel_l1_thresh=0.05, protect=2):
        self.rel_l1_thresh = float(rel_l1_thresh)
        self.protect = int(protect)

    def reset(self, state):
        state.user["prev"] = None

    def should_compute(self, ctx):
        # Always recompute the most sensitive blocks at the ends of the stack.
        if ctx.unit_index < self.protect or ctx.unit_index >= ctx.num_units - self.protect:
            return True

        u = ctx.state.user
        prev = u.get("prev")
        u["prev"] = ctx.e0.clone()
        if prev is None:
            return True
        return relative_l1(ctx.e0, prev) >= self.rel_l1_thresh
