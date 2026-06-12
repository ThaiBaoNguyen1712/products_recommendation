import json
import time
from typing import Any


class RecommendationCache:
    def __init__(self):
        self._memory_cache: dict[str, tuple[float | None, str]] = {}
        self._memory_version = "1"
        self._version_key = "rec:version"

    def get_version(self) -> str:
        return self._memory_version

    def invalidate_all(self) -> str:
        new_version = str(int(time.time()))
        self._memory_version = new_version
        self._memory_cache.clear()
        return new_version

    def build_key(self, scene: str, *, limit: int, user_id: int | None = None, product_sys_id: str | None = None) -> str:
        version = self.get_version()
        key_parts = [f"rec:v{version}", scene]
        if user_id is not None:
            key_parts.append(f"user:{int(user_id)}")
        if product_sys_id is not None:
            key_parts.append(f"product:{str(product_sys_id).strip()}")
        key_parts.append(f"limit:{int(limit)}")
        return ":".join(key_parts)

    def get_json(self, key: str) -> dict[str, Any] | None:
        cached = self._memory_cache.get(key)
        if not cached:
            return None

        expires_at, payload = cached
        if expires_at is not None and expires_at <= time.time():
            self._memory_cache.pop(key, None)
            return None
        return json.loads(payload)

    def set_json(self, key: str, value: dict[str, Any], ttl_seconds: int) -> None:
        payload = json.dumps(value, ensure_ascii=True)
        expires_at = time.time() + max(int(ttl_seconds), 1)
        self._memory_cache[key] = (expires_at, payload)


recommendation_cache = RecommendationCache()


SCENE_CACHE_TTLS = {
    "similar": 6 * 60 * 60,
    "detail": 10 * 60,
    "wishlist": 10 * 60,
    "cart": 10 * 60,
    "homepage": 20 * 60,
}


def get_scene_cache_ttl(scene: str) -> int:
    return int(SCENE_CACHE_TTLS.get(scene, 10 * 60))
