import hashlib
from typing import Optional
import redis
from src.config import settings


class ExactCache:
    """
    Tier-1 Exact Match Cache powered by Redis.

    Why:
    - Provides sub-millisecond lookups for repeated queries.
    - Prevents unnecessary vector embedding API calls for identical prompts.
    - Features TTL (Time-To-Live) expiration to prevent stale data.
    """

    def __init__(self, default_ttl_seconds: int = 86400):
        self.default_ttl = default_ttl_seconds
        self.client = redis.Redis.from_url(
            settings.redis_url,
            decode_responses=True,
            protocol=2,  # Forces RESP2 protocol; prevents 'HELLO 3' incompatibility with Redis 5.x
            socket_connect_timeout=2.0,
        )

    def _generate_key(self, prompt: str) -> str:
        """
        Normalizes the prompt (strips leading/trailing spaces, converts to lowercase)
        and computes a deterministic SHA-256 hash.
        """
        normalized = prompt.strip().lower()
        hash_digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return f"warmstart:exact:{hash_digest}"

    def get(self, prompt: str) -> Optional[str]:
        """
        Fetches cached response for an exact prompt.
        Returns None on cache miss or connection error.
        """
        key = self._generate_key(prompt)
        try:
            return self.client.get(key)
        except redis.RedisError as e:
            print(f"[REDIS WARNING] Failed to read from cache: {e}")
            return None

    def set(self, prompt: str, response: str, ttl: Optional[int] = None) -> bool:
        """
        Persists a prompt-response pair with an expiration TTL.
        """
        key = self._generate_key(prompt)
        target_ttl = ttl or self.default_ttl
        try:
            return bool(self.client.setex(name=key, time=target_ttl, value=response))
        except redis.RedisError as e:
            print(f"[REDIS WARNING] Failed to write to cache: {e}")
            return False

    def flush(self) -> None:
        """Flushes all keys in the current Redis database."""
        try:
            self.client.flushdb()
        except redis.RedisError as e:
            print(f"[REDIS WARNING] Failed to flush cache: {e}")
