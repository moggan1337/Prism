"""
Tracing Manager - Distributed tracing for request monitoring.

Provides distributed tracing capabilities using OpenTelemetry for:
- Request tracing
- Span management
- Trace context propagation
- Custom span creation
"""

from __future__ import annotations

import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

import structlog

from prism.core.config import ObservabilityConfig


logger = structlog.get_logger(__name__)


# Context variable for current span
current_span: ContextVar["Span | None"] = ContextVar("current_span", default=None)


@dataclass
class Span:
    """Represents a trace span."""
    span_id: str
    trace_id: str
    name: str
    start_time: float
    end_time: float | None = None
    parent_span_id: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    status: str = "OK"
    status_message: str = ""
    
    def add_attribute(self, key: str, value: Any) -> None:
        """Add attribute to span."""
        self.attributes[key] = value
    
    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Add event to span."""
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {}
        })
    
    def set_status(self, status: str, message: str = "") -> None:
        """Set span status."""
        self.status = status
        self.status_message = message
    
    def end(self) -> None:
        """End the span."""
        self.end_time = time.time()
    
    @property
    def duration_ms(self) -> float:
        """Get span duration in milliseconds."""
        if self.end_time is None:
            return (time.time() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000
    
    def to_dict(self) -> dict[str, Any]:
        """Convert span to dictionary."""
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "parent_span_id": self.parent_span_id,
            "attributes": self.attributes,
            "events": self.events,
            "status": self.status,
            "status_message": self.status_message,
        }


class TracingManager:
    """
    Tracing manager for distributed tracing.
    
    Provides span management and trace context propagation.
    Can use OpenTelemetry for production or simple in-memory tracing for development.
    
    Example:
        >>> config = ObservabilityConfig(tracing_enabled=True)
        >>> tracer = TracingManager(config)
        >>> 
        >>> # Create span
        >>> span_id = await tracer.start_span("process_request", {"user_id": "123"})
        >>> 
        >>> # Do work...
        >>> 
        >>> # End span
        >>> await tracer.end_span(span_id)
    """
    
    def __init__(self, config: ObservabilityConfig):
        """
        Initialize tracing manager.
        
        Args:
            config: Observability configuration
        """
        self.config = config
        self._spans: dict[str, Span] = {}
        self._traces: dict[str, list[str]] = {}  # trace_id -> span_ids
        self._otel_tracer = None
        self._otel_provider = None
        
        # Initialize OpenTelemetry if configured
        if config.tracing_enabled and config.otlp_endpoint:
            self._init_opentelemetry()
    
    def _init_opentelemetry(self) -> None:
        """Initialize OpenTelemetry tracing."""
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            
            # Create provider
            self._otel_provider = TracerProvider()
            
            # Create exporter
            exporter = OTLPSpanExporter(endpoint=self.config.otlp_endpoint)
            
            # Add processor
            self._otel_provider.add_span_processor(BatchSpanProcessor(exporter))
            
            # Set as provider
            trace.set_tracer_provider(self._otel_provider)
            
            # Get tracer
            self._otel_tracer = trace.get_tracer(
                self.config.service_name,
                "1.0.0"
            )
            
            logger.info("opentelemetry_initialized", endpoint=self.config.otlp_endpoint)
        
        except ImportError:
            logger.warning("opentelemetry_not_installed")
        except Exception as e:
            logger.error("opentelemetry_init_error", error=str(e))
    
    async def start_span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
        parent_name: str | None = None,
        parent_span_id: str | None = None
    ) -> str:
        """
        Start a new trace span.
        
        Args:
            name: Span name
            attributes: Span attributes
            parent_name: Parent span name (for nested spans)
            parent_span_id: Parent span ID (for context propagation)
            
        Returns:
            Span ID
        """
        span_id = str(uuid.uuid4())[:16]
        trace_id = str(uuid.uuid4())[:16]
        
        # Get parent span if not provided
        if parent_span_id is None:
            parent_span = current_span.get()
            if parent_span:
                parent_span_id = parent_span.span_id
                trace_id = parent_span.trace_id
        
        span = Span(
            span_id=span_id,
            trace_id=trace_id,
            name=name,
            start_time=time.time(),
            parent_span_id=parent_span_id,
            attributes=attributes or {}
        )
        
        # Store span
        self._spans[span_id] = span
        if trace_id not in self._traces:
            self._traces[trace_id] = []
        self._traces[trace_id].append(span_id)
        
        # Set current span
        token = current_span.set(span)
        span._token = token  # Store for restoration
        
        # Log span start
        logger.debug(
            "span_started",
            span_id=span_id,
            trace_id=trace_id,
            name=name,
            parent_id=parent_span_id
        )
        
        return span_id
    
    async def end_span(
        self,
        span_id: str,
        status: str = "OK",
        status_message: str = ""
    ) -> dict[str, Any]:
        """
        End a trace span.
        
        Args:
            span_id: Span ID to end
            status: Span status (OK, ERROR, etc.)
            status_message: Status message
            
        Returns:
            Span data dictionary
        """
        span = self._spans.get(span_id)
        
        if span is None:
            logger.warning("span_not_found", span_id=span_id)
            return {}
        
        # End span
        span.end()
        span.set_status(status, status_message)
        
        # Log span end
        logger.debug(
            "span_ended",
            span_id=span_id,
            trace_id=span.trace_id,
            duration_ms=span.duration_ms,
            status=status
        )
        
        # Clear current span if this is the current span
        current = current_span.get()
        if current and current.span_id == span_id:
            current_span.set(None)
        
        return span.to_dict()
    
    async def record_exception(
        self,
        span_id: str,
        exception: Exception,
        attributes: dict[str, Any] | None = None
    ) -> None:
        """
        Record an exception in a span.
        
        Args:
            span_id: Span ID
            exception: Exception to record
            attributes: Additional attributes
        """
        span = self._spans.get(span_id)
        
        if span is None:
            return
        
        span.add_event(
            "exception",
            {
                "exception.type": type(exception).__name__,
                "exception.message": str(exception),
                "exception.stacktrace": str(exception),
                **(attributes or {})
            }
        )
        span.set_status("ERROR", str(exception))
    
    def get_span(self, span_id: str) -> Span | None:
        """Get span by ID."""
        return self._spans.get(span_id)
    
    def get_trace(self, trace_id: str) -> list[Span]:
        """Get all spans for a trace."""
        span_ids = self._traces.get(trace_id, [])
        return [self._spans[ sid] for sid in span_ids if sid in self._spans]
    
    def get_current_span(self) -> Span | None:
        """Get current active span."""
        return current_span.get()
    
    def get_trace_tree(self, trace_id: str) -> dict[str, Any]:
        """
        Get trace as a tree structure.
        
        Args:
            trace_id: Trace ID
            
        Returns:
            Nested tree of spans
        """
        spans = self.get_trace(trace_id)
        
        if not spans:
            return {}
        
        # Build tree
        span_map = {s.span_id: s.to_dict() for s in spans}
        
        for span_dict in span_map.values():
            parent_id = span_dict.get("parent_span_id")
            if parent_id and parent_id in span_map:
                if "children" not in span_map[parent_id]:
                    span_map[parent_id]["children"] = []
                span_map[parent_id]["children"].append(span_dict)
        
        # Find root spans
        roots = [s for s in spans if s.parent_span_id is None]
        
        return {
            "trace_id": trace_id,
            "root_spans": [s.to_dict() for s in roots],
            "total_spans": len(spans)
        }
    
    async def flush(self) -> None:
        """Flush any pending spans."""
        # End any open spans
        for span_id, span in list(self._spans.items()):
            if span.end_time is None:
                await self.end_span(span_id, status="CANCELLED")
        
        # Clear old spans (keep last 1000)
        if len(self._spans) > 1000:
            sorted_spans = sorted(
                self._spans.items(),
                key=lambda x: x[1].start_time
            )
            to_remove = sorted_spans[:-1000]
            for span_id, _ in to_remove:
                del self._spans[span_id]
    
    def get_stats(self) -> dict[str, Any]:
        """Get tracing statistics."""
        total_spans = len(self._spans)
        active_spans = sum(1 for s in self._spans.values() if s.end_time is None)
        total_traces = len(self._traces)
        
        durations = [
            s.duration_ms for s in self._spans.values() if s.end_time is not None
        ]
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        return {
            "total_spans": total_spans,
            "active_spans": active_spans,
            "total_traces": total_traces,
            "avg_span_duration_ms": avg_duration,
        }
    
    def shutdown(self) -> None:
        """Shutdown tracing."""
        if self._otel_provider:
            try:
                self._otel_provider.shutdown()
            except Exception as e:
                logger.error("tracer_shutdown_error", error=str(e))
