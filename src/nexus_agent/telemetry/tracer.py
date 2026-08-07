import time
import uuid
import logging
from contextlib import contextmanager
from typing import Dict, Any, Generator

logger = logging.getLogger("nexus_agent.telemetry")


class SimpleSpan:
    def __init__(self, name: str, trace_id: str):
        self.name = name
        self.trace_id = trace_id
        self.attributes: Dict[str, Any] = {}
        self.start_time: float = time.perf_counter()
        self.duration_ms: float = 0.0

    def set_attribute(self, key: str, value: Any):
        self.attributes[key] = value

    def end(self):
        self.duration_ms = (time.perf_counter() - self.start_time) * 1000.0
        logger.debug("Span [%s] (TraceID: %s) completed in %.2fms. Attributes: %s",
                     self.name, self.trace_id, self.duration_ms, self.attributes)


class NexusTracer:
    """
    OpenTelemetry-compatible distributed tracer for NexusAgent spans.
    """
    def __init__(self, service_name: str = "nexus-agent-core"):
        self.service_name = service_name

    def generate_trace_id(self) -> str:
        return uuid.uuid4().hex

    @contextmanager
    def start_span(self, name: str, trace_id: str = None) -> Generator[SimpleSpan, None, None]:
        tid = trace_id or self.generate_trace_id()
        span = SimpleSpan(name=name, trace_id=tid)
        try:
            yield span
        finally:
            span.end()


tracer = NexusTracer()
