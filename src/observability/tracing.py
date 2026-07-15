"""OpenTelemetry 分布式追踪配置。"""

from __future__ import annotations

import contextlib
import functools
from typing import Any, Callable


class _NoopSpan:
    """空操作 Span（当 OTel 未启用时使用）。"""
    def set_attribute(self, *args, **kwargs): pass
    def set_status(self, *args, **kwargs): pass
    def end(self, *args, **kwargs): pass


_NOOP = _NoopSpan()


def setup_tracing(
    service_name: str = "rag-research-agent",
    otlp_endpoint: str | None = None,
) -> None:
    """初始化 OpenTelemetry 追踪。

    Args:
        service_name: 服务名称
        otlp_endpoint: OTLP collector 地址，None 则只用 console exporter
    """
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource(attributes={SERVICE_NAME: service_name})
        provider = TracerProvider(resource=resource)

        # Console exporter（开发时方便调试）
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

        # OTLP exporter（生产环境）
        if otlp_endpoint:
            otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

        trace.set_tracer_provider(provider)
    except ImportError:
        pass


def get_tracer(name: str = "rag-agent"):
    """获取 tracer 实例。"""
    try:
        from opentelemetry import trace
        return trace.get_tracer(name)
    except ImportError:
        return None


@contextlib.contextmanager
def traced_span(name: str, **attributes: Any):
    """上下文管理器：创建一个追踪 Span。

    Usage:
        with traced_span("agent.search", strategy="hybrid", query="..."):
            results = search(...)
    """
    tracer = get_tracer()
    if tracer is None:
        yield _NOOP
        return

    with tracer.start_as_current_span(name) as span:
        for k, v in attributes.items():
            span.set_attribute(k, str(v)[:256])
        try:
            yield span
        except Exception as exc:
            span.set_status({"status_code": 2, "description": str(exc)[:256]})
            raise


def trace(name_template: str) -> Callable:
    """装饰器：为函数调用自动创建 Span。

    Usage:
        @trace("search.{strategy}")
        async def search(query: str, strategy: str = "hybrid"):
            ...
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def async_wrapper(*args, **kwargs):
            span_name = name_template.format(**kwargs)
            with traced_span(span_name, **{k: v for k, v in kwargs.items() if isinstance(v, (str, int, float))}):
                return await fn(*args, **kwargs)

        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            span_name = name_template.format(**kwargs)
            with traced_span(span_name, **{k: v for k, v in kwargs.items() if isinstance(v, (str, int, float))}):
                return fn(*args, **kwargs)

        import asyncio
        if asyncio.iscoroutinefunction(fn):
            return async_wrapper
        return sync_wrapper
    return decorator
