import json
import os
import time
from typing import Any

import upstash_redis as redis


class RecommendationCache:
    def __init__(self):
        cache_url = (os.getenv("UPSTASH_URL") or "").strip()
        cache_token = (os.getenv("UPSTASH_TOKEN") or "").strip()
        self._has_remote_cache = bool(cache_url and cache_token)
        self._memory_cache: dict[str, tuple[float | None, str]] = {}
        self._memory_version = "1"
        self._version_key = "rec:version"
        self._redis_client = None
        if self._has_remote_cache:
            self._redis_client = redis.Redis(url=cache_url, token=cache_token)

    def get_version(self) -> str:
        if self._redis_client is not None:
            try:
                version = self._redis_client.get(self._version_key)
                if version is None:
                    self._redis_client.set(self._version_key, self._memory_version)
                    return self._memory_version
                return str(version)
            except Exception:
                pass
        return self._memory_version

    def invalidate_all(self) -> str:
        new_version = str(int(time.time()))
        self._memory_version = new_version
        self._memory_cache.clear()
        if self._redis_client is not None:
            try:
                self._redis_client.set(self._version_key, new_version)
            except Exception:
                pass
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
        if self._redis_client is not None:
            try:
                payload = self._redis_client.get(key)
                if payload is not None:
                    return json.loads(str(payload))
            except Exception:
                pass

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
        if self._redis_client is not None:
            try:
                self._redis_client.set(key, payload, ex=ttl_seconds)
                return
            except Exception:
                pass

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
