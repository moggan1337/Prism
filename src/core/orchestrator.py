"""
Prism Orchestrator - Main entry point for multi-modal AI routing.

Orchestrates the entire pipeline including routing, caching, retries,
fallbacks, aggregation, and observability.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable

import structlog

from prism.core.config import (
    PrismConfig, ContentType, TaskType, RoutingStrategy,
    ModelConfig, CacheConfig, BudgetConfig, ObservabilityConfig
)
from prism.core.router import Router, RouteRequest, RouteResult, UsageStats
from prism.cache.manager import CacheManager
from prism.observability.tracer import TracingManager
from prism.observability.metrics import MetricsCollector
from prism.adapters.base import ModelAdapter, AdapterResponse


logger = structlog.get_logger(__name__)


@dataclass
class RequestContext:
    """Context for a request being processed."""
    request_id: str
    start_time: float
    content: Any
    content_type: ContentType | None
    task_type: TaskType | None
    route_result: RouteResult | None = None
    adapter_response: AdapterResponse | None = None
    cache_hit: bool = False
    retry_count: int = 0
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestratedResponse:
    """Response from orchestrated request."""
    request_id: str
    content: Any
    model: str
    provider: str
    cost: float
    latency_ms: float
    cache_hit: bool
    content_type: ContentType
    task_type: TaskType | None
    success: bool
    error: str | None = None
    alternatives_used: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "content": str(self.content)[:1000] if isinstance(self.content, str) else self.content,
            "model": self.model,
            "provider": self.provider,
            "cost": self.cost,
            "latency_ms": self.latency_ms,
            "cache_hit": self.cache_hit,
            "content_type": self.content_type.value,
            "task_type": self.task_type.value if self.task_type else None,
            "success": self.success,
            "error": self.error,
            "alternatives_used": self.alternatives_used,
            "metadata": self.metadata,
        }


class PrismOrchestrator:
    """
    Main orchestrator for multi-modal AI routing.
    
    Coordinates all components to provide a unified interface for
    routing requests to optimal AI models.
    
    Features:
    - Intelligent content classification
    - Model selection based on cost/latency/accuracy
    - Request caching with TTL
    - Automatic retry with backoff
    - Fallback to alternative models
    - Multi-model aggregation
    - Budget enforcement
    - Distributed tracing
    - Metrics collection
    
    Example:
        >>> config = PrismConfig.from_env()
        >>> orchestrator = PrismOrchestrator(config)
        >>> 
        >>> # Process a text request
        >>> response = await orchestrator.process(
        ...     content="Hello, world!",
        ...     task_type=TaskType.TEXT_GENERATION
        ... )
        >>> print(f"Result: {response.content}")
    """
    
    def __init__(
        self,
        config: PrismConfig | None = None,
        adapters: dict[str, ModelAdapter] | None = None,
        enable_observability: bool = True
    ):
        """
        Initialize the orchestrator.
        
        Args:
            config: Prism configuration (uses defaults if not provided)
            adapters: Map of model name to adapter
            enable_observability: Enable tracing and metrics
        """
        self.config = config or PrismConfig()
        self.models = self.config.models or PrismConfig.load_default_models()
        self.adapters = adapters or {}
        
        # Initialize router
        self.router = Router(self.config, self.models)
        
        # Initialize cache
        self.cache = CacheManager(self.config.cache)
        
        # Initialize observability
        self._enable_observability = enable_observability
        if enable_observability:
            self.tracer = TracingManager(self.config.observability)
            self.metrics = MetricsCollector(self.config.observability)
        else:
            self.tracer = None
            self.metrics = None
        
        # Request tracking
        self._active_requests: dict[str, RequestContext] = {}
        self._request_queue: asyncio.PriorityQueue | None = None
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        logger.info(
            "prism_orchestrator_initialized",
            models=len(self.models),
            adapters=len(self.adapters),
            routing_strategy=self.config.default_routing_strategy.value
        )
    
    async def process(
        self,
        content: Any,
        content_type: ContentType | None = None,
        task_type: TaskType | None = None,
        user_id: str | None = None,
        api_key: str | None = None,
        options: dict[str, Any] | None = None,
        **kwargs
    ) -> OrchestratedResponse:
        """
        Process a request through the routing pipeline.
        
        Args:
            content: The content to process (text, image bytes, audio, etc.)
            content_type: Optional content type hint
            task_type: Optional task type hint
            user_id: Optional user identifier for tracking
            api_key: Optional API key for tracking
            options: Additional processing options
            **kwargs: Additional parameters passed to route request
            
        Returns:
            OrchestratedResponse with results
        """
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Create context
        context = RequestContext(
            request_id=request_id,
            start_time=start_time,
            content=content,
            content_type=content_type,
            task_type=task_type
        )
        
        # Track request
        async with self._lock:
            self._active_requests[request_id] = context
        
        try:
            # Start trace span
            if self.tracer:
                await self.tracer.start_span(request_id, {
                    "content_type": content_type.value if content_type else "unknown",
                    "task_type": task_type.value if task_type else "unknown",
                    "user_id": user_id,
                })
            
            # Record metrics
            if self.metrics:
                self.metrics.increment_counter("prism_requests_total")
            
            # Build route request
            route_request = RouteRequest(
                content=content,
                content_type=content_type,
                task_type=task_type,
                user_id=user_id,
                api_key=api_key,
                **(options or {}),
                **kwargs
            )
            
            # Check cache first
            cache_key = route_request.get_cache_key()
            cached_response = await self.cache.get(cache_key)
            
            if cached_response:
                context.cache_hit = True
                self.router.record_cache_hit()
                
                if self.metrics:
                    self.metrics.increment_counter("prism_cache_hits_total")
                
                logger.info("cache_hit", request_id=request_id, cache_key=cache_key)
                
                return OrchestratedResponse(
                    request_id=request_id,
                    content=cached_response["content"],
                    model=cached_response.get("model", "cached"),
                    provider=cached_response.get("provider", "cache"),
                    cost=0.0,
                    latency_ms=(time.time() - start_time) * 1000,
                    cache_hit=True,
                    content_type=content_type or ContentType.UNKNOWN,
                    task_type=task_type,
                    success=True,
                    metadata=cached_response.get("metadata", {})
                )
            
            self.router.record_cache_miss()
            if self.metrics:
                self.metrics.increment_counter("prism_cache_misses_total")
            
            # Route request
            route_result = self.router.route(route_request)
            context.route_result = route_result
            
            # Check budget
            if not self.router.can_afford(route_result.estimated_cost):
                if self.config.budget.auto_fallback_to_cheaper:
                    # Try cheaper alternative
                    for alt in route_result.alternatives:
                        alt_cost = route_result.estimated_cost * 0.5  # Simplified
                        if self.router.can_afford(alt_cost):
                            route_result.selected_model = alt.name
                            route_result.provider = alt.provider
                            break
                else:
                    raise BudgetExceededError(
                        f"Budget exceeded. Remaining: ${self.router.get_usage_stats().budget_remaining:.2f}"
                    )
            
            # Get or create adapter
            adapter = await self._get_adapter(route_result.selected_model)
            
            # Execute with retries
            response = await self._execute_with_retry(
                adapter=adapter,
                content=content,
                context=context,
                route_result=route_result
            )
            
            # Cache successful response
            if response.success and route_result.should_cache:
                await self.cache.set(
                    key=cache_key,
                    value={
                        "content": response.content,
                        "model": response.model,
                        "provider": response.provider,
                        "metadata": response.metadata
                    },
                    ttl=self._get_cache_ttl(content_type)
                )
            
            # Record success
            self.router.record_success(
                model_name=response.model,
                cost=route_result.estimated_cost,
                latency_ms=response.latency_ms
            )
            self.router.update_budget(route_result.estimated_cost)
            
            # Record metrics
            if self.metrics:
                self.metrics.record_histogram(
                    "prism_request_latency_ms",
                    response.latency_ms,
                    tags={"model": response.model, "content_type": content_type.value if content_type else "unknown"}
                )
                self.metrics.record_histogram(
                    "prism_request_cost",
                    route_result.estimated_cost,
                    tags={"model": response.model}
                )
            
            return response
            
        except Exception as e:
            logger.error("request_failed", request_id=request_id, error=str(e))
            context.errors.append(str(e))
            
            if self.metrics:
                self.metrics.increment_counter("prism_requests_failed_total")
            
            return OrchestratedResponse(
                request_id=request_id,
                content=None,
                model=context.route_result.selected_model if context.route_result else "unknown",
                provider=context.route_result.provider if context.route_result else "unknown",
                cost=0.0,
                latency_ms=(time.time() - start_time) * 1000,
                cache_hit=False,
                content_type=content_type or ContentType.UNKNOWN,
                task_type=task_type,
                success=False,
                error=str(e)
            )
        
        finally:
            # End trace span
            if self.tracer:
                await self.tracer.end_span(request_id)
            
            # Cleanup
            async with self._lock:
                self._active_requests.pop(request_id, None)
    
    async def process_batch(
        self,
        requests: list[dict[str, Any]],
        parallel: bool = False,
        max_concurrent: int = 5
    ) -> list[OrchestratedResponse]:
        """
        Process multiple requests in batch.
        
        Args:
            requests: List of request dictionaries
            parallel: Process in parallel if True, sequential if False
            max_concurrent: Maximum concurrent requests (for parallel mode)
            
        Returns:
            List of OrchestratedResponse
        """
        if parallel:
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def bounded_process(req: dict) -> OrchestratedResponse:
                async with semaphore:
                    return await self.process(**req)
            
            return await asyncio.gather(*[bounded_process(req) for req in requests])
        else:
            results = []
            for req in requests:
                result = await self.process(**req)
                results.append(result)
            return results
    
    async def _get_adapter(self, model_name: str) -> ModelAdapter:
        """Get or create adapter for model."""
        if model_name in self.adapters:
            return self.adapters[model_name]
        
        # Import adapters here to avoid circular imports
        from prism.adapters.openai_adapter import OpenAIAdapter
        from prism.adapters.anthropic_adapter import AnthropicAdapter
        from prism.adapters.local_adapter import LocalAdapter
        
        model = self.models.get(model_name)
        if not model:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Create appropriate adapter based on provider
        provider = model.provider.value
        
        if provider == "openai":
            adapter = OpenAIAdapter(model_name, self.config.providers.get("openai"))
        elif provider == "anthropic":
            adapter = AnthropicAdapter(model_name, self.config.providers.get("anthropic"))
        else:
            # Fall back to local adapter for other providers
            adapter = LocalAdapter(model_name)
        
        self.adapters[model_name] = adapter
        return adapter
    
    async def _execute_with_retry(
        self,
        adapter: ModelAdapter,
        content: Any,
        context: RequestContext,
        route_result: RouteResult
    ) -> OrchestratedResponse:
        """Execute request with retry logic."""
        retry_config = self.config.retry
        max_attempts = retry_config.max_attempts
        current_attempt = 0
        last_error = None
        
        while current_attempt < max_attempts:
            try:
                response = await adapter.execute(
                    content=content,
                    task_type=context.task_type,
                    metadata=context.metadata
                )
                
                if response.success:
                    return OrchestratedResponse(
                        request_id=context.request_id,
                        content=response.content,
                        model=route_result.selected_model,
                        provider=route_result.provider,
                        cost=route_result.estimated_cost,
                        latency_ms=response.latency_ms,
                        cache_hit=False,
                        content_type=context.content_type or ContentType.UNKNOWN,
                        task_type=context.task_type,
                        success=True,
                        metadata=response.metadata
                    )
                else:
                    last_error = response.error
                    context.errors.append(f"Attempt {current_attempt + 1}: {response.error}")
            
            except Exception as e:
                last_error = str(e)
                context.errors.append(f"Attempt {current_attempt + 1}: {str(e)}")
                logger.warning(
                    "retry_attempt",
                    request_id=context.request_id,
                    attempt=current_attempt + 1,
                    error=str(e)
                )
            
            current_attempt += 1
            context.retry_count = current_attempt
            
            # Check if we should retry
            if current_attempt < max_attempts:
                delay = self._calculate_retry_delay(current_attempt)
                await asyncio.sleep(delay / 1000)
                
                # Try fallback model if available
                if route_result.alternatives and self.config.fallback.enabled:
                    alt = route_result.alternatives[0]
                    logger.info(
                        "trying_fallback_model",
                        original=route_result.selected_model,
                        fallback=alt.name
                    )
                    route_result.selected_model = alt.name
                    route_result.provider = alt.provider
                    route_result.estimated_cost = alt.estimated_cost
                    route_result.estimated_latency_ms = alt.estimated_latency_ms
                    context.alternatives_used.append(alt.name)
                    
                    adapter = await self._get_adapter(alt.name)
        
        # All retries failed
        self.router.record_failure(route_result.selected_model)
        
        return OrchestratedResponse(
            request_id=context.request_id,
            content=None,
            model=route_result.selected_model,
            provider=route_result.provider,
            cost=route_result.estimated_cost,
            latency_ms=(time.time() - context.start_time) * 1000,
            cache_hit=False,
            content_type=context.content_type or ContentType.UNKNOWN,
            task_type=context.task_type,
            success=False,
            error=last_error or "Max retries exceeded",
            alternatives_used=context.alternatives_used
        )
    
    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate retry delay with exponential backoff."""
        retry_config = self.config.retry
        
        delay = retry_config.initial_delay_ms * (retry_config.backoff_factor ** (attempt - 1))
        delay = min(delay, retry_config.max_delay_ms)
        
        if retry_config.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)
        
        return delay
    
    def _get_cache_ttl(self, content_type: ContentType | None) -> int:
        """Get cache TTL based on content type."""
        cache_config = self.config.cache
        
        if content_type == ContentType.TEXT:
            return cache_config.text_ttl_seconds
        elif content_type == ContentType.IMAGE:
            return cache_config.image_ttl_seconds
        elif content_type == ContentType.AUDIO:
            return cache_config.audio_ttl_seconds
        elif content_type == ContentType.VIDEO:
            return cache_config.video_ttl_seconds
        elif content_type == ContentType.DOCUMENT:
            return cache_config.document_ttl_seconds
        
        return cache_config.ttl_seconds
    
    async def aggregate(
        self,
        content: Any,
        task_type: TaskType | None = None,
        models: list[str] | None = None
    ) -> OrchestratedResponse:
        """
        Aggregate responses from multiple models.
        
        Args:
            content: Content to process
            task_type: Task type
            models: Specific models to use (uses default if not provided)
            
        Returns:
            Aggregated response
        """
        if not self.config.aggregation.enabled:
            raise ValueError("Aggregation is not enabled in configuration")
        
        # Default models for aggregation
        if models is None:
            models = list(self.models.keys())[:self.config.aggregation.models_per_request]
        
        # Process with all models
        results = await asyncio.gather(*[
            self.process(content, task_type=task_type)
            for model in models
        ])
        
        # Aggregate responses
        aggregated = self._aggregate_responses(results)
        
        return aggregated
    
    def _aggregate_responses(
        self,
        responses: list[OrchestratedResponse]
    ) -> OrchestratedResponse:
        """Aggregate multiple responses into one."""
        strategy = self.config.aggregation.aggregation_strategy
        
        if strategy == "weighted_average":
            return self._weighted_average_aggregate(responses)
        elif strategy == "majority_vote":
            return self._majority_vote_aggregate(responses)
        elif strategy == "hierarchical":
            return self._hierarchical_aggregate(responses)
        else:
            # Default: use first successful response
            for resp in responses:
                if resp.success:
                    return resp
            return responses[0] if responses else responses[0]
    
    def _weighted_average_aggregate(
        self,
        responses: list[OrchestratedResponse]
    ) -> OrchestratedResponse:
        """Aggregate using weighted average based on model accuracy."""
        successful = [r for r in responses if r.success]
        
        if not successful:
            return responses[0]
        
        # Weight by inverse cost * accuracy
        weights = []
        for resp in successful:
            weight = 1.0 / (resp.cost + 0.001)  # Avoid division by zero
            weights.append(weight)
        
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Combine text responses
        if all(isinstance(r.content, str) for r in successful):
            combined = " ".join([
                f"[Model: {r.model}] {r.content}"
                for r, w in zip(successful, weights)
            ])
        else:
            combined = successful[0].content
        
        avg_latency = sum(r.latency_ms * w for r, w in zip(successful, weights))
        total_cost = sum(r.cost for r in successful)
        
        return OrchestratedResponse(
            request_id=successful[0].request_id,
            content=combined,
            model="aggregated",
            provider="prism",
            cost=total_cost,
            latency_ms=avg_latency,
            cache_hit=False,
            content_type=successful[0].content_type,
            task_type=successful[0].task_type,
            success=True,
            alternatives_used=[r.model for r in successful],
            metadata={"aggregation_strategy": "weighted_average"}
        )
    
    def _majority_vote_aggregate(
        self,
        responses: list[OrchestratedResponse]
    ) -> OrchestratedResponse:
        """Aggregate using majority voting."""
        successful = [r for r in responses if r.success]
        
        if not successful:
            return responses[0]
        
        # For text, simple vote by content hash
        if all(isinstance(r.content, str) for r in successful):
            from collections import Counter
            content_hashes = [hash(r.content) for r in successful]
            vote_counts = Counter(content_hashes)
            majority_hash, _ = vote_counts.most_common(1)[0]
            
            for resp in successful:
                if hash(resp.content) == majority_hash:
                    return OrchestratedResponse(
                        request_id=resp.request_id,
                        content=resp.content,
                        model=f"majority_vote({', '.join(r.model for r in successful)})",
                        provider="prism",
                        cost=sum(r.cost for r in successful),
                        latency_ms=sum(r.latency_ms for r in successful) / len(successful),
                        cache_hit=False,
                        content_type=resp.content_type,
                        task_type=resp.task_type,
                        success=True,
                        alternatives_used=[r.model for r in successful],
                        metadata={"aggregation_strategy": "majority_vote"}
                    )
        
        return successful[0]
    
    def _hierarchical_aggregate(
        self,
        responses: list[OrchestratedResponse]
    ) -> OrchestratedResponse:
        """Aggregate using hierarchical approach."""
        successful = sorted(
            [r for r in responses if r.success],
            key=lambda r: r.latency_ms
        )
        
        if not successful:
            return responses[0]
        
        # Use fastest as base, validate with others
        base = successful[0]
        
        return OrchestratedResponse(
            request_id=base.request_id,
            content=base.content,
            model=f"hierarchical({', '.join(r.model for r in successful)})",
            provider="prism",
            cost=sum(r.cost for r in successful),
            latency_ms=max(r.latency_ms for r in successful),
            cache_hit=False,
            content_type=base.content_type,
            task_type=base.task_type,
            success=True,
            alternatives_used=[r.model for r in successful],
            metadata={"aggregation_strategy": "hierarchical"}
        )
    
    def get_stats(self) -> dict[str, Any]:
        """Get current statistics."""
        usage_stats = self.router.get_usage_stats()
        
        return {
            "usage": usage_stats.__dict__,
            "cache": self.cache.get_stats(),
            "models": {
                name: {
                    "requests": usage_stats.by_model.get(name, 0),
                    "enabled": model.enabled,
                    "provider": model.provider.value
                }
                for name, model in self.models.items()
            },
            "active_requests": len(self._active_requests)
        }
    
    async def health_check(self) -> dict[str, Any]:
        """Check health of all components."""
        health = {
            "status": "healthy",
            "components": {}
        }
        
        # Check cache
        try:
            await self.cache.set("health_check", {"status": "ok"}, ttl=1)
            health["components"]["cache"] = {"status": "healthy"}
        except Exception as e:
            health["components"]["cache"] = {"status": "unhealthy", "error": str(e)}
            health["status"] = "degraded"
        
        # Check adapters
        for name, adapter in self.adapters.items():
            try:
                is_healthy = await adapter.health_check()
                health["components"][f"adapter_{name}"] = {
                    "status": "healthy" if is_healthy else "unhealthy"
                }
            except Exception as e:
                health["components"][f"adapter_{name}"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health["status"] = "degraded"
        
        return health
    
    @asynccontextmanager
    async def span(self, name: str, attributes: dict[str, Any] | None = None) -> AsyncIterator[None]:
        """Create a trace span for custom operations."""
        span_id = str(uuid.uuid4())
        
        if self.tracer:
            await self.tracer.start_span(span_id, attributes or {}, parent_name=name)
        
        try:
            yield
        finally:
            if self.tracer:
                await self.tracer.end_span(span_id)


class BudgetExceededError(Exception):
    """Raised when budget limit is exceeded."""
    pass
