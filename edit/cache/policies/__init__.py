"""Built-in cache policies. Importing this package registers them in
POLICY_REGISTRY. Add a new policy by dropping a module here that defines a
CachePolicy subclass decorated with @register_policy("name")."""

from edit.cache.policies import teacache, fbcache, perblock  # noqa: F401  (register on import)

from edit.cache.policies.teacache import TeaCachePolicy, WAN_TEACACHE_COEFFICIENTS
from edit.cache.policies.fbcache import FBCachePolicy
from edit.cache.policies.perblock import PerBlockDemoPolicy

__all__ = ["TeaCachePolicy", "FBCachePolicy", "PerBlockDemoPolicy", "WAN_TEACACHE_COEFFICIENTS"]
