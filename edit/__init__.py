"""EDiT: a lightweight, editable inference engine for diffusion transformers (DiTs).

Imported as ``edit``. It provides an explicit Wan denoising loop plus pluggable
TeaCache / First-Block-Cache policies for cache-strategy research.
"""

from edit.cache import (
    apply_cache_on_transformer,
    remove_cache_from_transformer,
    CachePolicy,
    register_policy,
)
from edit.pipeline import EditWanPipeline

__all__ = [
    "apply_cache_on_transformer",
    "remove_cache_from_transformer",
    "CachePolicy",
    "register_policy",
    "EditWanPipeline",
]
