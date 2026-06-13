import json
import os
import time
from typing import Any

from upstash_redis import Redis


class RecommendationCache:
    def __init__(self):
        self._memory_cache: dict[str, tuple[float | None, str]] = {}
        self._memory_version = "1"
        self._version_key = "rec:version"
        self._version_cache_ttl_seconds = 5.0
        self._last_version_sync_at = 0.0
        self._redis = self._build_redis_client()

    def _build_redis_client(self) -> Redis | None:
        redis_url = (
            os.getenv("UPSTASH_REDIS_REST_URL", "").strip()
            or os.getenv("UPSTASH_URL", "").strip()
        )
        redis_token = (
            os.getenv("UPSTASH_REDIS_REST_TOKEN", "").strip()
            or os.getenv("UPSTASH_TOKEN", "").strip()
        )
        if not redis_url or not redis_token:
            return None
        try:
            return Redis(
                url=redis_url,
                token=redis_token,
                allow_telemetry=False,
            )
        except Exception:
            return None

    def _sync_version_from_redis(self, force: bool = False) -> None:
        if self._redis is None:
            return

        now = time.time()
        if not force and (now - self._last_version_sync_at) < self._version_cache_ttl_seconds:
            return

        try:
            remote_version = self._redis.get(self._version_key)
            if remote_version is None:
                self._redis.set(self._version_key, self._memory_version)
            else:
                normalized_version = str(remote_version).strip()
                if normalized_version:
                    self._memory_version = normalized_version
        except Exception:
            pass
        finally:
            self._last_version_sync_at = now

    def get_version(self) -> str:
        self._sync_version_from_redis()
        return self._memory_version

    def invalidate_all(self) -> str:
        new_version = str(int(time.time()))
        self._memory_version = new_version
        self._last_version_sync_at = time.time()
        self._memory_cache.clear()
        if self._redis is not None:
            try:
                self._redis.set(self._version_key, new_version)
            except Exception:
                pass
        return new_version

    def build_key(
        self,
        scene: str,
        *,
        limit: int,
        user_id: int | None = None,
        product_sys_id: str | None = None,
        state_token: str | None = None,
    ) -> str:
        version = self.get_version()
        key_parts = [f"rec:v{version}", scene]
        if user_id is not None:
            key_parts.append(f"user:{int(user_id)}")
        if product_sys_id is not None:
            key_parts.append(f"product:{str(product_sys_id).strip()}")
        if state_token is not None:
            normalized_state_token = str(state_token).strip()
            if normalized_state_token:
                key_parts.append(f"state:{normalized_state_token}")
        key_parts.append(f"limit:{int(limit)}")
        return ":".join(key_parts)

    def get_json(self, key: str) -> dict[str, Any] | None:
        if self._redis is not None:
            try:
                cached_payload = self._redis.get(key)
                if cached_payload is not None:
                    return json.loads(str(cached_payload))
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
        expires_at = time.time() + max(int(ttl_seconds), 1)
        self._memory_cache[key] = (expires_at, payload)
        if self._redis is not None:
            try:
                self._redis.set(key, payload, ex=max(int(ttl_seconds), 1))
            except Exception:
                pass


recommendation_cache = RecommendationCache()


SCENE_CACHE_TTLS = {
    "similar": 6 * 60 * 60,
    "detail": 2 * 60,
    "wishlist": 60,
    "cart": 30,
    "homepage": 60,
}


def get_scene_cache_ttl(scene: str) -> int:
    return int(SCENE_CACHE_TTLS.get(scene, 10 * 60))
