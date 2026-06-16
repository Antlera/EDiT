# TeaCache policy — port of official ali-vilab/TeaCache for Wan.
#
# Signal: the timestep embedding. use_ret_steps=True uses e0 (timestep_proj),
# False uses e (base time embedding). A per-model polynomial rescales the per-step
# relative-L1 of that signal; the rescaled value is accumulated and the unit is
# skipped while the accumulator stays below rel_l1_thresh. First/last steps are
# force-computed. See README / framework.py for how this maps onto the mechanism.
#
# Per-branch counters here count one denoise step (the official code counts twice
# per step via a doubled cnt; nano keeps one counter per CFG branch, so the
# forced-compute windows are the halved equivalents — see _windows()).

import numpy as np

from edit.cache.framework import CachePolicy, register_policy, relative_l1


# Official ali-vilab/TeaCache Wan2.1 coefficients, keyed by (task, variant, use_ret_steps).
WAN_TEACACHE_COEFFICIENTS = {
    ("t2v", "1.3B", True): [-5.21862437e04, 9.23041404e03, -5.28275948e02, 1.36987616e01, -4.99875664e-02],
    ("t2v", "14B", True): [-3.03318725e05, 4.90537029e04, -2.65530556e03, 5.87365115e01, -3.15583525e-01],
    ("t2v", "1.3B", False): [2.39676752e03, -1.31110545e03, 2.01331979e02, -8.29855975e00, 1.37887774e-01],
    ("t2v", "14B", False): [-5784.54975374, 5449.50911966, -1811.16591783, 256.27178429, -13.02252404],
    ("i2v", "480p", True): [2.57151496e05, -3.54229917e04, 1.40286849e03, -1.35890334e01, 1.32517977e-01],
    ("i2v", "720p", True): [8.10705460e03, 2.13393892e03, -3.72934672e02, 1.66203073e01, -4.17769401e-02],
    ("i2v", "480p", False): [-3.02331670e02, 2.23948934e02, -5.25463970e01, 5.87348440e00, -2.01973289e-01],
    ("i2v", "720p", False): [-114.36346466, 65.26524496, -18.82220707, 4.91518089, -0.23412683],
}


def _parse_variant(wan_variant: str):
    task, _, variant = wan_variant.lower().partition("-")
    if task not in ("t2v", "i2v") or not variant:
        raise ValueError(
            f"Unrecognized wan_variant {wan_variant!r}; expected 't2v-14B', 't2v-1.3B', "
            "'i2v-480p', or 'i2v-720p'."
        )
    variant = {"14b": "14B", "1.3b": "1.3B", "480p": "480p", "720p": "720p"}.get(variant, variant)
    return task, variant


@register_policy("teacache")
class TeaCachePolicy(CachePolicy):
    def __init__(self, *, rel_l1_thresh=0.2, wan_variant=None, coefficients=None, use_ret_steps=True):
        if coefficients is None:
            if wan_variant is None:
                raise ValueError("TeaCache needs either `coefficients` or `wan_variant`.")
            task, variant = _parse_variant(wan_variant)
            key = (task, variant, use_ret_steps)
            if key not in WAN_TEACACHE_COEFFICIENTS:
                raise ValueError(f"No official coefficients for {key}.")
            coefficients = WAN_TEACACHE_COEFFICIENTS[key]
        self.rel_l1_thresh = float(rel_l1_thresh)
        self.coefficients = list(coefficients)
        self.rescale = np.poly1d(self.coefficients)
        self.use_ret_steps = bool(use_ret_steps)

    def reset(self, state):
        state.user["accumulated"] = 0.0
        state.user["prev"] = None

    def _windows(self, num_steps):
        # forced-compute window, per-branch (see module docstring)
        if self.use_ret_steps:
            return 5, num_steps          # ret_steps, cutoff (cutoff never fires)
        return 1, num_steps - 1

    def should_compute(self, ctx):
        modulated = ctx.e0 if self.use_ret_steps else ctx.e
        if modulated is None:
            raise RuntimeError(
                "use_ret_steps=False needs the base time embedding `e`, captured by "
                "apply_cache_on_transformer."
            )

        u = ctx.state.user
        ret_steps, cutoff_steps = self._windows(ctx.num_steps)
        forced = ctx.step < ret_steps or ctx.step >= cutoff_steps
        prev = u.get("prev")

        if forced or prev is None:
            compute = True
            u["accumulated"] = 0.0
        else:
            u["accumulated"] += float(self.rescale(relative_l1(modulated, prev)))
            if u["accumulated"] < self.rel_l1_thresh:
                compute = False
            else:
                compute = True
                u["accumulated"] = 0.0

        # refresh the reference every step (incl. skipped), as the official code does
        u["prev"] = modulated.clone()
        return compute
