"""
Metrics Collector - Prometheus-compatible metrics collection.

Provides metrics collection for:
- Counters
- Gauges
- Histograms
- Labels
- Export to Prometheus
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import structlog

from prism.core.config import ObservabilityConfig


logger = structlog.get_logger(__name__)


@dataclass
class MetricValue:
    """Represents a metric value with labels."""
    labels: tuple[tuple[str, str], ...]
    value: float
    timestamp: float
    

class Counter:
    """Counter metric - only increments."""
    
    def __init__(self, name: str, description: str, labels: list[str]):
        self.name = name
        self.description = description
        self.labels = labels
        self._values: dict[tuple[tuple[str, str], ...], float] = defaultdict(float)
        self._total = 0.0
    
    def increment(self, value: float = 1, **label_kwargs) -> None:
        """Increment counter."""
        label_tuple = self._make_label_tuple(label_kwargs)
        self._values[label_tuple] += value
        self._total += value
    
    def get_value(self, **label_kwargs) -> float:
        """Get counter value for labels."""
        label_tuple = self._make_label_tuple(label_kwargs)
        return self._values.get(label_tuple, 0.0)
    
    def _make_label_tuple(self, label_kwargs: dict[str, str]) -> tuple[tuple[str, str], ...]:
        """Convert label kwargs to sorted tuple."""
        return tuple(sorted(label_kwargs.items()))
    
    def collect(self) -> list[MetricValue]:
        """Collect all values."""
        return [
            MetricValue(labels=k, value=v, timestamp=time.time())
            for k, v in self._values.items()
        ]


class Gauge:
    """Gauge metric - can go up and down."""
    
    def __init__(self, name: str, description: str, labels: list[str]):
        self.name = name
        self.description = description
        self.labels = labels
        self._values: dict[tuple[tuple[str, str], ...], float] = {}
        self._default_value = 0.0
    
    def set(self, value: float, **label_kwargs) -> None:
        """Set gauge value."""
        label_tuple = self._make_label_tuple(label_kwargs)
        self._values[label_tuple] = value
    
    def increment(self, value: float = 1, **label_kwargs) -> None:
        """Increment gauge."""
        label_tuple = self._make_label_tuple(label_kwargs)
        self._values[label_tuple] = self._values.get(label_tuple, 0.0) + value
    
    def decrement(self, value: float = 1, **label_kwargs) -> None:
        """Decrement gauge."""
        label_tuple = self._make_label_tuple(label_kwargs)
        self._values[label_tuple] = self._values.get(label_tuple, 0.0) - value
    
    def get_value(self, **label_kwargs) -> float:
        """Get gauge value."""
        label_tuple = self._make_label_tuple(label_kwargs)
        return self._values.get(label_tuple, self._default_value)
    
    def _make_label_tuple(self, label_kwargs: dict[str, str]) -> tuple[tuple[str, str], ...]:
        """Convert label kwargs to sorted tuple."""
        return tuple(sorted(label_kwargs.items()))
    
    def collect(self) -> list[MetricValue]:
        """Collect all values."""
        return [
            MetricValue(labels=k, value=v, timestamp=time.time())
            for k, v in self._values.items()
        ]


class Histogram:
    """Histogram metric - tracks distributions."""
    
    def __init__(
        self,
        name: str,
        description: str,
        labels: list[str],
        buckets: list[float] | None = None
    ):
        self.name = name
        self.description = description
        self.labels = labels
        self.buckets = buckets or [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]
        self._values: dict[tuple[tuple[str, str], ...], list[float]] = defaultdict(list)
        self._sums: dict[tuple[tuple[str, str], ...], float] = defaultdict(float)
        self._counts: dict[tuple[tuple[str, str], ...], int] = defaultdict(int)
    
    def observe(self, value: float, **label_kwargs) -> None:
        """Observe a value."""
        label_tuple = self._make_label_tuple(label_kwargs)
        self._values[label_tuple].append(value)
        self._sums[label_tuple] += value
        self._counts[label_tuple] += 1
    
    def get_stats(self, **label_kwargs) -> dict[str, Any]:
        """Get histogram statistics."""
        label_tuple = self._make_label_tuple(label_kwargs)
        values = self._values.get(label_tuple, [])
        
        if not values:
            return {"count": 0, "sum": 0, "avg": 0}
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        return {
            "count": n,
            "sum": self._sums[label_tuple],
            "avg": self._sums[label_tuple] / n,
            "min": sorted_values[0],
            "max": sorted_values[-1],
            "p50": sorted_values[n // 2],
            "p90": sorted_values[int(n * 0.9)],
            "p95": sorted_values[int(n * 0.95)],
            "p99": sorted_values[int(n * 0.99)],
        }
    
    def _make_label_tuple(self, label_kwargs: dict[str, str]) -> tuple[tuple[str, str], ...]:
        """Convert label kwargs to sorted tuple."""
        return tuple(sorted(label_kwargs.items()))
    
    def collect(self) -> list[MetricValue]:
        """Collect all bucket values."""
        results = []
        
        for label_tuple, values in self._values.items():
            sorted_values = sorted(values)
            n = len(sorted_values)
            
            # Count values in each bucket
            cumulative = 0
            for bucket in self.buckets:
                cumulative += sum(1 for v in sorted_values if v <= bucket)
                results.append(MetricValue(
                    labels=label_tuple + (("le", str(bucket)),),
                    value=float(cumulative),
                    timestamp=time.time()
                ))
            
            # +Inf bucket
            results.append(MetricValue(
                labels=label_tuple + (("le", "+Inf"),),
                value=float(n),
                timestamp=time.time()
            ))
        
        return results


class Summary:
    """Summary metric - tracks quantiles."""
    
    def __init__(
        self,
        name: str,
        description: str,
        labels: list[str],
        quantiles: list[float] | None = None
    ):
        self.name = name
        self.description = description
        self.labels = labels
        self.quantiles = quantiles or [0.5, 0.9, 0.95, 0.99]
        self._values: dict[tuple[tuple[str, str], ...], list[float]] = defaultdict(list)
        self._counts: dict[tuple[tuple[str, str], ...], int] = defaultdict(int)
        self._sums: dict[tuple[tuple[str, str], ...], float] = defaultdict(float)
        self._max_size = 1000  # Keep last 1000 values per label set
    
    def observe(self, value: float, **label_kwargs) -> None:
        """Observe a value."""
        label_tuple = self._make_label_tuple(label_kwargs)
        values = self._values[label_tuple]
        values.append(value)
        
        # Keep only last N values
        if len(values) > self._max_size:
            values.pop(0)
        
        self._sums[label_tuple] += value
        self._counts[label_tuple] += 1
    
    def get_quantiles(self, **label_kwargs) -> dict[str, float]:
        """Get quantile values."""
        label_tuple = self._make_label_tuple(label_kwargs)
        values = sorted(self._values.get(label_tuple, []))
        
        if not values:
            return {}
        
        result = {}
        for q in self.quantiles:
            idx = int(len(values) * q)
            idx = min(idx, len(values) - 1)
            result[f"q{int(q * 100)}"] = values[idx]
        
        return result
    
    def _make_label_tuple(self, label_kwargs: dict[str, str]) -> tuple[tuple[str, str], ...]:
        """Convert label kwargs to sorted tuple."""
        return tuple(sorted(label_kwargs.items()))
    
    def collect(self) -> list[MetricValue]:
        """Collect all quantile values."""
        results = []
        
        for label_tuple, values in self._values.items():
            sorted_values = sorted(values)
            
            for q in self.quantiles:
                idx = int(len(sorted_values) * q)
                idx = min(idx, len(sorted_values) - 1)
                results.append(MetricValue(
                    labels=label_tuple + (("quantile", str(q)),),
                    value=sorted_values[idx],
                    timestamp=time.time()
                ))
        
        return results


class MetricsCollector:
    """
    Prometheus-compatible metrics collector.
    
    Provides metrics collection with support for:
    - Counters
    - Gauges
    - Histograms
    - Labels
    - Export to Prometheus format
    
    Example:
        >>> config = ObservabilityConfig(metrics_enabled=True)
        >>> collector = MetricsCollector(config)
        >>> 
        >>> # Create metrics
        >>> collector.create_counter("requests_total", "Total requests", ["method", "path"])
        >>> collector.create_histogram("request_duration_ms", "Request duration", ["method"])
        >>> 
        >>> # Record metrics
        >>> collector.increment_counter("requests_total", method="GET", path="/api")
        >>> collector.record_histogram("request_duration_ms", 150.5, method="GET")
        >>> 
        >>> # Export
        >>> print(collector.export_prometheus())
    """
    
    def __init__(self, config: ObservabilityConfig):
        """
        Initialize metrics collector.
        
        Args:
            config: Observability configuration
        """
        self.config = config
        self._counters: dict[str, Counter] = {}
        self._gauges: dict[str, Gauge] = {}
        self._histograms: dict[str, Histogram] = {}
        self._summaries: dict[str, Summary] = {}
        
        # Create default metrics
        self._create_default_metrics()
    
    def _create_default_metrics(self) -> None:
        """Create default metrics."""
        # Request metrics
        self.create_counter(
            "prism_requests_total",
            "Total number of requests",
            ["content_type", "task_type", "model"]
        )
        
        self.create_histogram(
            "prism_request_duration_seconds",
            "Request duration in seconds",
            ["content_type", "task_type", "model"]
        )
        
        self.create_counter(
            "prism_requests_failed_total",
            "Total number of failed requests",
            ["content_type", "error_type"]
        )
        
        # Cache metrics
        self.create_counter(
            "prism_cache_hits_total",
            "Total cache hits",
            ["content_type"]
        )
        
        self.create_counter(
            "prism_cache_misses_total",
            "Total cache misses",
            ["content_type"]
        )
        
        # Cost metrics
        self.create_counter(
            "prism_cost_total",
            "Total cost in dollars",
            ["model", "content_type"]
        )
        
        self.create_histogram(
            "prism_request_cost_dollars",
            "Cost per request in dollars",
            ["model"]
        )
        
        # Model metrics
        self.create_gauge(
            "prism_model_requests_in_flight",
            "Number of requests in flight",
            ["model"]
        )
        
        self.create_histogram(
            "prism_model_latency_ms",
            "Model latency in milliseconds",
            ["model", "provider"]
        )
        
        # Routing metrics
        self.create_counter(
            "prism_routing_decisions_total",
            "Total routing decisions",
            ["strategy", "content_type"]
        )
        
        self.create_histogram(
            "prism_routing_score",
            "Routing score distribution",
            ["content_type"]
        )
    
    def create_counter(
        self,
        name: str,
        description: str,
        label_names: list[str] | None = None
    ) -> Counter:
        """Create a counter metric."""
        labels = label_names or []
        counter = Counter(name, description, labels)
        self._counters[name] = counter
        return counter
    
    def create_gauge(
        self,
        name: str,
        description: str,
        label_names: list[str] | None = None
    ) -> Gauge:
        """Create a gauge metric."""
        labels = label_names or []
        gauge = Gauge(name, description, labels)
        self._gauges[name] = gauge
        return gauge
    
    def create_histogram(
        self,
        name: str,
        description: str,
        label_names: list[str] | None = None,
        buckets: list[float] | None = None
    ) -> Histogram:
        """Create a histogram metric."""
        labels = label_names or []
        histogram = Histogram(name, description, labels, buckets)
        self._histograms[name] = histogram
        return histogram
    
    def create_summary(
        self,
        name: str,
        description: str,
        label_names: list[str] | None = None,
        quantiles: list[float] | None = None
    ) -> Summary:
        """Create a summary metric."""
        labels = label_names or []
        summary = Summary(name, description, labels, quantiles)
        self._summaries[name] = summary
        return summary
    
    # Convenience methods
    def increment_counter(
        self,
        name: str,
        value: float = 1,
        **labels
    ) -> None:
        """Increment a counter."""
        if name in self._counters:
            self._counters[name].increment(value, **labels)
    
    def set_gauge(
        self,
        name: str,
        value: float,
        **labels
    ) -> None:
        """Set a gauge value."""
        if name in self._gauges:
            self._gauges[name].set(value, **labels)
    
    def record_histogram(
        self,
        name: str,
        value: float,
        **labels
    ) -> None:
        """Record a histogram value."""
        if name in self._histograms:
            self._histograms[name].observe(value, **labels)
    
    def record_summary(
        self,
        name: str,
        value: float,
        **labels
    ) -> None:
        """Record a summary value."""
        if name in self._summaries:
            self._summaries[name].observe(value, **labels)
    
    def get_counter_value(self, name: str, **labels) -> float:
        """Get counter value."""
        if name in self._counters:
            return self._counters[name].get_value(**labels)
        return 0.0
    
    def get_gauge_value(self, name: str, **labels) -> float:
        """Get gauge value."""
        if name in self._gauges:
            return self._gauges[name].get_value(**labels)
        return 0.0
    
    def get_histogram_stats(self, name: str, **labels) -> dict[str, Any]:
        """Get histogram statistics."""
        if name in self._histograms:
            return self._histograms[name].get_stats(**labels)
        return {}
    
    def export_prometheus(self) -> str:
        """
        Export all metrics in Prometheus format.
        
        Returns:
            Metrics in Prometheus exposition format
        """
        lines = []
        
        # Export counters
        for name, counter in self._counters.items():
            lines.append(f"# HELP {name} {counter.description}")
            lines.append(f"# TYPE {name} counter")
            for mv in counter.collect():
                label_str = self._format_labels(mv.labels)
                lines.append(f"{name}{label_str} {mv.value}")
        
        # Export gauges
        for name, gauge in self._gauges.items():
            lines.append(f"# HELP {name} {gauge.description}")
            lines.append(f"# TYPE {name} gauge")
            for mv in gauge.collect():
                label_str = self._format_labels(mv.labels)
                lines.append(f"{name}{label_str} {mv.value}")
        
        # Export histograms
        for name, histogram in self._histograms.items():
            lines.append(f"# HELP {name} {histogram.description}")
            lines.append(f"# TYPE {name} histogram")
            for mv in histogram.collect():
                label_str = self._format_labels(mv.labels)
                lines.append(f"{name}{label_str} {mv.value}")
        
        # Export summaries
        for name, summary in self._summaries.items():
            lines.append(f"# HELP {name} {summary.description}")
            lines.append(f"# TYPE {name} summary")
            for mv in summary.collect():
                label_str = self._format_labels(mv.labels)
                lines.append(f"{name}{label_str} {mv.value}")
        
        return "\n".join(lines) + "\n"
    
    def _format_labels(
        self,
        labels: tuple[tuple[str, str], ...]
    ) -> str:
        """Format labels for Prometheus export."""
        if not labels:
            return ""
        
        label_parts = [f'{k}="{v}"' for k, v in labels]
        return "{" + ",".join(label_parts) + "}"
    
    def get_all_metrics(self) -> dict[str, Any]:
        """Get all metrics as dictionary."""
        metrics = {
            "counters": {},
            "gauges": {},
            "histograms": {},
            "summaries": {}
        }
        
        for name, counter in self._counters.items():
            metrics["counters"][name] = {
                "description": counter.description,
                "values": [
                    {
                        "labels": dict(mv.labels),
                        "value": mv.value
                    }
                    for mv in counter.collect()
                ]
            }
        
        for name, gauge in self._gauges.items():
            metrics["gauges"][name] = {
                "description": gauge.description,
                "values": [
                    {
                        "labels": dict(mv.labels),
                        "value": mv.value
                    }
                    for mv in gauge.collect()
                ]
            }
        
        for name, histogram in self._histograms.items():
            metrics["histograms"][name] = {
                "description": histogram.description,
                "buckets": histogram.buckets
            }
        
        for name, summary in self._summaries.items():
            metrics["summaries"][name] = {
                "description": summary.description,
                "quantiles": summary.quantiles
            }
        
        return metrics
    
    def reset(self) -> None:
        """Reset all metrics."""
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()
        self._summaries.clear()
        self._create_default_metrics()
