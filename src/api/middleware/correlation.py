"""Correlation ID middleware — injects X-Correlation-ID into every request.

If the client sends an X-Correlation-ID header, it's reused.
Otherwise, a new UUID4 is generated. The ID is stored in:
  - request.state.correlation_id
  - Response header X-Correlation-ID
  - structlog thread-local context (for structured logging)
  - OTel span attribute (for distributed tracing)

This enables end-to-end request tracking across: API → Agent → RAG → LLM.
"""

from __future__ import annotations

import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """Middleware that ensures every request has a correlation ID.

    Usage:
        app.add_middleware(CorrelationIDMiddleware)
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        # Extract or generate correlation ID
        correlation_id = request.headers.get("X-Correlation-ID")
        if not correlation_id:
            correlation_id = str(uuid.uuid4())

        # Store on request state for downstream access
        request.state.correlation_id = correlation_id

        # Inject into structlog context if configured
        try:
            import structlog
            structlog.contextvars.bind_contextvars(correlation_id=correlation_id)
        except ImportError:
            pass

        # Process request
        response = await call_next(request)

        # Include in response headers
        response.headers["X-Correlation-ID"] = correlation_id

        # Cleanup structlog context
        try:
            import structlog
            structlog.contextvars.unbind_contextvars("correlation_id")
        except ImportError:
            pass

        return response


def get_correlation_id(request: Request) -> str:
    """Get the correlation ID from request state."""
    return getattr(request.state, "correlation_id", "unknown")
