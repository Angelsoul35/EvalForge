"""Optional OpenTelemetry tracing for eval runs.

When enabled, each suite run and each case is wrapped in a span, so an eval can
be observed in the same tracing backend as the system under test. When
disabled (the default), the context managers are cheap no-ops.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

_TRACER = None


def _get_tracer():
    global _TRACER
    if _TRACER is not None:
        return _TRACER
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    _TRACER = trace.get_tracer("evalforge")
    return _TRACER


@contextmanager
def span(name: str, enabled: bool, **attributes) -> Iterator[None]:
    if not enabled:
        yield
        return
    tracer = _get_tracer()
    with tracer.start_as_current_span(name) as current:
        for key, value in attributes.items():
            current.set_attribute(key, value)
        yield
