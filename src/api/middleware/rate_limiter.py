"""Rate Limiter — token bucket algorithm, per-tenant isolation.

Uses the token bucket algorithm: each tenant has a bucket that refills at
a configurable rate. Requests consume tokens. When the bucket is empty,
the request is rejected with HTTP 429.

Thread-safe for use with async FastAPI.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from fastapi import Request


@dataclass
class TokenBucket:
    """Token bucket for a single tenant's rate limit."""

    tokens: float
    max_tokens: float
    refill_rate: float  # tokens per second
    last_refill: float = 0.0

    def __post_init__(self):
        self.last_refill = time.monotonic()
        self.tokens = self.max_tokens


class RateLimiter:
    """Per-tenant token bucket rate limiter.

    Usage:
        limiter = RateLimiter(default_tokens_per_minute=60)

        # In FastAPI middleware/dependency:
        if not await limiter.acquire(tenant_id):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
    """

    def __init__(
        self,
        default_tokens_per_minute: int = 60,
        burst_multiplier: float = 1.5,
    ):
        self.default_rate = default_tokens_per_minute / 60.0  # tokens per second
        self.burst_multiplier = burst_multiplier
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = asyncio.Lock()

    def _get_or_create_bucket(self, tenant_id: str) -> TokenBucket:
        """Get or create a token bucket for a tenant."""
        if tenant_id not in self._buckets:
            max_tokens = self.default_rate * 60 * self.burst_multiplier
            self._buckets[tenant_id] = TokenBucket(
                tokens=max_tokens,
                max_tokens=max_tokens,
                refill_rate=self.default_rate,
            )
        return self._buckets[tenant_id]

    def _refill(self, bucket: TokenBucket) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - bucket.last_refill
        bucket.tokens = min(bucket.max_tokens, bucket.tokens + elapsed * bucket.refill_rate)
        bucket.last_refill = now

    async def acquire(self, tenant_id: str, tokens: float = 1.0) -> bool:
        """Try to acquire tokens for a request.

        Returns True if the request should be allowed, False if rate limited.
        """
        async with self._lock:
            bucket = self._get_or_create_bucket(tenant_id)
            self._refill(bucket)

            if bucket.tokens >= tokens:
                bucket.tokens -= tokens
                return True
            return False

    async def get_remaining(self, tenant_id: str) -> float:
        """Get remaining tokens for a tenant."""
        async with self._lock:
            bucket = self._buckets.get(tenant_id)
            if bucket is None:
                return self.default_rate * 60
            self._refill(bucket)
            return bucket.tokens

    def set_rate(self, tenant_id: str, tokens_per_minute: int) -> None:
        """Override the rate limit for a specific tenant."""
        bucket = self._buckets.get(tenant_id)
        new_rate = tokens_per_minute / 60.0
        if bucket:
            bucket.refill_rate = new_rate
            bucket.max_tokens = new_rate * 60 * self.burst_multiplier

    def cleanup(self, max_age_seconds: float = 3600.0) -> int:
        """Remove buckets inactive for longer than max_age_seconds. Returns count removed."""
        now = time.monotonic()
        to_remove = [
            tid for tid, b in self._buckets.items() if now - b.last_refill > max_age_seconds
        ]
        for tid in to_remove:
            del self._buckets[tid]
        return len(to_remove)


class RateLimitMiddleware:
    """FastAPI middleware for rate limiting.

    Usage:
        app.add_middleware(RateLimitMiddleware, limiter=RateLimiter())
    """

    def __init__(self, app, limiter: RateLimiter | None = None):
        self.app = app
        self.limiter = limiter or RateLimiter()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        tenant_id = getattr(request.state, "tenant_id", None) or "anonymous"

        if not await self.limiter.acquire(tenant_id):
            response = self._rate_limit_response()
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)

    @staticmethod
    def _rate_limit_response():
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=429,
            content={
                "error": {
                    "type": "https://api.contractintel.com/errors/rate-limit-exceeded",
                    "title": "Rate Limit Exceeded",
                    "status": 429,
                    "detail": "Too many requests. Please wait before retrying.",
                }
            },
            headers={"Retry-After": "60"},
        )
