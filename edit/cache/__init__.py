"""Pluggable feature-cache (TeaCache / First-Block-Cache / custom) step-skipping
for single-GPU diffusers Wan inference.

Three independent axes:
  - POLICY      : subclass CachePolicy + @register_policy("name") (when to skip).
  - GRANULARITY : granularity="stack" | "per_block" | int k (unit size).
  - STORAGE     : store=GPUStore() | CPUOffloadStore() | your CacheStore subclass
                  (where/how cached tensors live).

`apply_cache_on_transformer(transformer, policy="name", num_inference_steps=N,
granularity="stack", store=GPUStore())`.
"""

from edit.cache.framework import (
    apply_cache_on_transformer,
    remove_cache_from_transformer,
    CachePolicy,
    register_policy,
    POLICY_REGISTRY,
    CachedUnit,
    CacheController,
    StepContext,
    UnitState,
    HistoryView,
    relative_l1,
)
from edit.cache.store import (
    CacheStore, KeyedTensorStore, GPUStore, CPUOffloadStore, DiskOffloadStore, TieredStore,
)

# Importing the policies package registers the built-in policies.
from edit.cache import policies  # noqa: F401
from edit.cache.policies import WAN_TEACACHE_COEFFICIENTS

__all__ = [
    "apply_cache_on_transformer",
    "remove_cache_from_transformer",
    "CachePolicy",
    "register_policy",
    "POLICY_REGISTRY",
    "CachedUnit",
    "CacheController",
    "StepContext",
    "UnitState",
    "HistoryView",
    "relative_l1",
    "CacheStore",
    "KeyedTensorStore",
    "GPUStore",
    "CPUOffloadStore",
    "DiskOffloadStore",
    "TieredStore",
    "WAN_TEACACHE_COEFFICIENTS",
]
