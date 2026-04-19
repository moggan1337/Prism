"""
Router module for intelligent request routing.

Handles content classification, model selection, and request routing
based on cost, latency, accuracy, and other factors.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable
from enum import Enum

from prism.core.config import (
    PrismConfig, ContentType, TaskType, RoutingStrategy,
    ModelConfig, CacheConfig, BudgetConfig
)


@dataclass
class RouteRequest:
    """Request to be routed to a model."""
    content: Any
    content_type: ContentType | None = None
    task_type: TaskType | None = None
    user_id: str | None = None
    api_key: str | None = None
    priority: int = 0  # Higher = more priority
    metadata: dict[str, Any] = field(default_factory=dict)
    preferred_models: list[str] | None = None
    blocked_models: list[str] | None = None
    max_cost: float | None = None
    max_latency_ms: float | None = None
    force_model: str | None = None
    
    def get_cache_key(self) -> str:
        """Generate cache key for this request."""
        content_hash = hashlib.sha256(
            json.dumps(self.content, sort_keys=True).encode()
        ).hexdigest()[:16]
        metadata_hash = hashlib.sha256(
            json.dumps({
                "task": self.task_type.value if self.task_type else None,
                "user": self.user_id,
                **self.metadata
            }, sort_keys=True).encode()
        ).hexdigest()[:8]
        return f"{content_hash}:{metadata_hash}"


@dataclass
class RouteResult:
    """Result of routing decision."""
    selected_model: str
    provider: str
    estimated_cost: float
    estimated_latency_ms: float
    confidence: float
    alternatives: list[AlternativeModel] = field(default_factory=list)
    routing_reason: str = ""
    cache_key: str | None = None
    should_cache: bool = True
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_model": self.selected_model,
            "provider": self.provider,
            "estimated_cost": self.estimated_cost,
            "estimated_latency_ms": self.estimated_latency_ms,
            "confidence": self.confidence,
            "alternatives": [a.to_dict() for a in self.alternatives],
            "routing_reason": self.routing_reason,
            "cache_key": self.cache_key,
            "should_cache": self.should_cache,
        }


@dataclass
class AlternativeModel:
    """Alternative model option."""
    name: str
    provider: str
    estimated_cost: float
    estimated_latency_ms: float
    score: float
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "provider": self.provider,
            "estimated_cost": self.estimated_cost,
            "estimated_latency_ms": self.estimated_latency_ms,
            "score": self.score,
        }


@dataclass 
class UsageStats:
    """Usage statistics for tracking."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_cost: float = 0.0
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    budget_remaining: float = 0.0
    by_model: dict[str, int] = field(default_factory=dict)
    by_content_type: dict[str, int] = field(default_factory=dict)
    by_task_type: dict[str, int] = field(default_factory=dict)
    
    @property
    def cache_hit_rate(self) -> float:
        if self.cache_hits + self.cache_misses == 0:
            return 0.0
        return self.cache_hits / (self.cache_hits + self.cache_misses)
    
    @property
    def avg_latency_ms(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_latency_ms / self.total_requests
    
    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests


class Router:
    """
    Intelligent router for multi-modal AI requests.
    
    Routes requests to optimal models based on:
    - Content type detection
    - Task type requirements
    - Cost optimization
    - Latency requirements
    - Accuracy requirements
    - Model availability and rate limits
    
    Example:
        >>> config = PrismConfig()
        >>> router = Router(config)
        >>> request = RouteRequest(
        ...     content="Hello, world!",
        ...     task_type=TaskType.TEXT_GENERATION
        ... )
        >>> result = router.route(request)
        >>> print(f"Selected model: {result.selected_model}")
    """
    
    def __init__(
        self,
        config: PrismConfig,
        models: dict[str, ModelConfig] | None = None,
        custom_scorer: Callable[[ModelConfig, RouteRequest], float] | None = None
    ):
        """
        Initialize the router.
        
        Args:
            config: Prism configuration
            models: Override model configurations
            custom_scorer: Custom scoring function for model selection
        """
        self.config = config
        self.models = models or PrismConfig.load_default_models()
        self.custom_scorer = custom_scorer
        self._usage_stats = UsageStats(budget_remaining=config.budget.monthly_limit)
        self._model_circuit_breakers: dict[str, CircuitBreakerState] = {}
        
        # Routing weights based on strategy
        self._routing_weights = self._get_routing_weights(config.default_routing_strategy)
    
    def _get_routing_weights(self, strategy: RoutingStrategy) -> dict[str, float]:
        """Get routing weights based on strategy."""
        weights = {
            "cost": 0.33,
            "latency": 0.33,
            "accuracy": 0.34
        }
        
        if strategy == RoutingStrategy.COST_OPTIMIZED:
            weights = {"cost": 0.7, "latency": 0.2, "accuracy": 0.1}
        elif strategy == RoutingStrategy.LATENCY_OPTIMIZED:
            weights = {"cost": 0.1, "latency": 0.7, "accuracy": 0.2}
        elif strategy == RoutingStrategy.ACCURACY_OPTIMIZED:
            weights = {"cost": 0.1, "latency": 0.2, "accuracy": 0.7}
        elif strategy == RoutingStrategy.BALANCED:
            weights = {"cost": 0.33, "latency": 0.33, "accuracy": 0.34}
        
        return weights
    
    def route(self, request: RouteRequest) -> RouteResult:
        """
        Route a request to the optimal model.
        
        Args:
            request: The routing request
            
        Returns:
            RouteResult with selected model and alternatives
        """
        # Generate cache key
        cache_key = request.get_cache_key()
        
        # Determine content type if not provided
        content_type = request.content_type
        if content_type is None:
            content_type = self._classify_content(request.content)
        
        # Determine task type if not provided
        task_type = request.task_type
        
        # Filter eligible models
        eligible_models = self._filter_models(
            content_type=content_type,
            task_type=task_type,
            request=request
        )
        
        if not eligible_models:
            raise NoEligibleModelError(
                f"No eligible models found for content_type={content_type}, "
                f"task_type={task_type}"
            )
        
        # Score and rank models
        scored_models = self._score_models(eligible_models, request)
        
        if not scored_models:
            raise NoEligibleModelError("All models failed scoring")
        
        # Select best model
        best_model_name, best_score = scored_models[0]
        best_model = self.models[best_model_name]
        
        # Generate alternatives (up to 3)
        alternatives = []
        for model_name, score in scored_models[1:4]:
            model = self.models[model_name]
            alternatives.append(AlternativeModel(
                name=model.name,
                provider=model.provider.value,
                estimated_cost=self._estimate_cost(model, request),
                estimated_latency_ms=model.avg_latency_ms,
                score=score
            ))
        
        # Build result
        result = RouteResult(
            selected_model=best_model_name,
            provider=best_model.provider.value,
            estimated_cost=self._estimate_cost(best_model, request),
            estimated_latency_ms=best_model.avg_latency_ms,
            confidence=min(best_score, 1.0),
            alternatives=alternatives,
            routing_reason=self._explain_routing(best_model, request),
            cache_key=cache_key,
            should_cache=self._should_cache(content_type)
        )
        
        # Update stats
        self._usage_stats.total_requests += 1
        self._usage_stats.by_content_type[content_type.value] = \
            self._usage_stats.by_content_type.get(content_type.value, 0) + 1
        if task_type:
            self._usage_stats.by_task_type[task_type.value] = \
                self._usage_stats.by_task_type.get(task_type.value, 0) + 1
        
        return result
    
    def _classify_content(self, content: Any) -> ContentType:
        """Classify content type from content."""
        # Import here to avoid circular imports
        from prism.routing.classifier import ContentClassifier
        
        classifier = ContentClassifier()
        return classifier.classify(content)
    
    def _filter_models(
        self,
        content_type: ContentType,
        task_type: TaskType | None,
        request: RouteRequest
    ) -> list[str]:
        """Filter models to eligible candidates."""
        eligible = []
        
        for model_name, model in self.models.items():
            # Check if model is enabled
            if not model.enabled:
                continue
            
            # Check circuit breaker
            if self._is_circuit_open(model_name):
                continue
            
            # Check content type support
            if content_type not in model.content_types:
                continue
            
            # Check task type support
            if task_type and task_type not in model.task_types:
                continue
            
            # Check preferred models
            if request.preferred_models and model_name not in request.preferred_models:
                continue
            
            # Check blocked models
            if request.blocked_models and model_name in request.blocked_models:
                continue
            
            # Check cost constraint
            if request.max_cost is not None:
                estimated_cost = self._estimate_cost(model, request)
                if estimated_cost > request.max_cost:
                    continue
            
            # Check latency constraint
            if request.max_latency_ms is not None:
                if model.avg_latency_ms > request.max_latency_ms:
                    continue
            
            # Check rate limits
            if not self._check_rate_limit(model):
                continue
            
            eligible.append(model_name)
        
        return eligible
    
    def _score_models(
        self,
        model_names: list[str],
        request: RouteRequest
    ) -> list[tuple[str, float]]:
        """Score models and return sorted by score."""
        scored = []
        
        for model_name in model_names:
            model = self.models[model_name]
            
            if self.custom_scorer:
                score = self.custom_scorer(model, request)
            else:
                score = self._calculate_score(model, request)
            
            scored.append((model_name, score))
        
        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
    
    def _calculate_score(self, model: ModelConfig, request: RouteRequest) -> float:
        """Calculate routing score for a model."""
        weights = self._routing_weights
        
        # Cost score (lower cost = higher score)
        cost_score = self._score_cost(model, request)
        
        # Latency score (lower latency = higher score)
        latency_score = self._score_latency(model)
        
        # Accuracy score (higher accuracy = higher score)
        accuracy_score = model.accuracy_score
        
        # Weighted sum
        total_score = (
            weights["cost"] * cost_score +
            weights["latency"] * latency_score +
            weights["accuracy"] * accuracy_score
        )
        
        # Apply circuit breaker penalty
        if model.name in self._model_circuit_breakers:
            breaker = self._model_circuit_breakers[model.name]
            total_score *= (1.0 - breaker.failure_rate * 0.5)
        
        return total_score
    
    def _score_cost(self, model: ModelConfig, request: RouteRequest) -> float:
        """Score based on cost (inverse - lower is better)."""
        estimated_cost = self._estimate_cost(model, request)
        
        # Normalize to 0-1 range (assuming max cost of $0.50)
        max_cost = 0.50
        if estimated_cost >= max_cost:
            return 0.0
        
        return 1.0 - (estimated_cost / max_cost)
    
    def _score_latency(self, model: ModelConfig) -> float:
        """Score based on latency (inverse - lower is better)."""
        # Normalize to 0-1 range (assuming max latency of 30s)
        max_latency = 30000
        if model.avg_latency_ms >= max_latency:
            return 0.0
        
        return 1.0 - (model.avg_latency_ms / max_latency)
    
    def _estimate_cost(self, model: ModelConfig, request: RouteRequest) -> float:
        """Estimate cost for a request."""
        content = request.content
        
        # Estimate based on content type
        if isinstance(content, str):
            # Estimate tokens (rough: 4 chars per token)
            tokens = len(content) / 4
            tokens_1k = tokens / 1000
            return (model.cost_per_1k_input_tokens * tokens_1k * 0.5 +
                   model.cost_per_1k_output_tokens * tokens_1k * 0.5)
        
        elif isinstance(content, dict):
            # For structured data, estimate based on serialization
            import json
            content_str = json.dumps(content)
            tokens = len(content_str) / 4
            tokens_1k = tokens / 1000
            return (model.cost_per_1k_input_tokens * tokens_1k * 0.7 +
                   model.cost_per_1k_output_tokens * tokens_1k * 0.3)
        
        elif isinstance(content, (bytes, bytearray)):
            # Binary content (image, audio, video)
            size_kb = len(content) / 1024
            if model.cost_per_image > 0:
                return model.cost_per_image
            return size_kb * 0.0001
        
        return 0.01  # Default minimal cost
    
    def _should_cache(self, content_type: ContentType) -> bool:
        """Determine if content type should be cached."""
        cache_config = self.config.cache
        if not cache_config.enabled:
            return False
        
        # Check content-type specific TTLs
        if content_type == ContentType.TEXT:
            return cache_config.text_ttl_seconds > 0
        elif content_type == ContentType.IMAGE:
            return cache_config.image_ttl_seconds > 0
        elif content_type == ContentType.AUDIO:
            return cache_config.audio_ttl_seconds > 0
        elif content_type == ContentType.VIDEO:
            return cache_config.video_ttl_seconds > 0
        elif content_type == ContentType.DOCUMENT:
            return cache_config.document_ttl_seconds > 0
        
        return True
    
    def _explain_routing(self, model: ModelConfig, request: RouteRequest) -> str:
        """Generate explanation for routing decision."""
        content_type = request.content_type or self._classify_content(request.content)
        strategy = self.config.default_routing_strategy.value
        
        reasons = [
            f"Strategy: {strategy}",
            f"Content type: {content_type.value}",
            f"Estimated cost: ${self._estimate_cost(model, request):.4f}",
            f"Estimated latency: {model.avg_latency_ms}ms",
            f"Model accuracy: {model.accuracy_score:.2%}",
        ]
        
        if request.task_type:
            reasons.append(f"Task type: {request.task_type.value}")
        
        return "; ".join(reasons)
    
    def _is_circuit_open(self, model_name: str) -> bool:
        """Check if circuit breaker is open for a model."""
        if model_name not in self._model_circuit_breakers:
            return False
        
        breaker = self._model_circuit_breakers[model_name]
        if breaker.state == CircuitState.OPEN:
            # Check if timeout has passed
            if time.time() - breaker.last_failure_time > self.config.circuit_breaker_timeout_seconds:
                # Transition to half-open
                breaker.state = CircuitState.HALF_OPEN
                return False
            return True
        
        return False
    
    def _check_rate_limit(self, model: ModelConfig) -> bool:
        """Check if model has available rate limit capacity."""
        # Simple check - in production, use token bucket or sliding window
        model_requests = self._usage_stats.by_model.get(model.name, 0)
        return model_requests < model.rate_limit_rpm * 10  # Conservative buffer
    
    def record_success(self, model_name: str, cost: float, latency_ms: float) -> None:
        """Record successful request."""
        self._usage_stats.successful_requests += 1
        self._usage_stats.total_cost += cost
        self._usage_stats.total_latency_ms += latency_ms
        self._usage_stats.by_model[model_name] = \
            self._usage_stats.by_model.get(model_name, 0) + 1
        
        # Reset circuit breaker on success
        if model_name in self._model_circuit_breakers:
            breaker = self._model_circuit_breakers[model_name]
            breaker.consecutive_failures = 0
            breaker.state = CircuitState.CLOSED
    
    def record_failure(self, model_name: str) -> None:
        """Record failed request."""
        self._usage_stats.failed_requests += 1
        
        # Update circuit breaker
        if model_name not in self._model_circuit_breakers:
            self._model_circuit_breakers[model_name] = CircuitBreakerState()
        
        breaker = self._model_circuit_breakers[model_name]
        breaker.consecutive_failures += 1
        breaker.last_failure_time = time.time()
        breaker.failure_rate = min(breaker.consecutive_failures / 10, 1.0)
        
        # Open circuit if too many failures
        if breaker.consecutive_failures >= self.config.circuit_breaker_threshold:
            breaker.state = CircuitState.OPEN
    
    def record_cache_hit(self) -> None:
        """Record cache hit."""
        self._usage_stats.cache_hits += 1
    
    def record_cache_miss(self) -> None:
        """Record cache miss."""
        self._usage_stats.cache_misses += 1
    
    def get_usage_stats(self) -> UsageStats:
        """Get current usage statistics."""
        self._usage_stats.budget_remaining = (
            self.config.budget.monthly_limit - self._usage_stats.total_cost
        )
        return self._usage_stats
    
    def can_afford(self, cost: float) -> bool:
        """Check if request can be afforded within budget."""
        if not self.config.budget.enabled:
            return True
        
        remaining = self.config.budget.monthly_limit - self._usage_stats.total_cost
        return remaining >= cost
    
    def update_budget(self, cost: float) -> None:
        """Update budget after request."""
        self._usage_stats.total_cost += cost
        self._usage_stats.budget_remaining = (
            self.config.budget.monthly_limit - self._usage_stats.total_cost
        )


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerState:
    """State for circuit breaker."""
    state: CircuitState = CircuitState.CLOSED
    consecutive_failures: int = 0
    last_failure_time: float = 0.0
    failure_rate: float = 0.0


class NoEligibleModelError(Exception):
    """Raised when no eligible model is found for a request."""
    pass


class BudgetExceededError(Exception):
    """Raised when budget limit is exceeded."""
    pass
