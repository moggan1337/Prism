"""
Model Selector - Advanced model selection based on multiple criteria.

Provides sophisticated model selection capabilities including:
- Multi-criteria optimization (cost, latency, accuracy)
- Constraint-based selection
- Model capability matching
- Dynamic model weighting
- Custom selection strategies
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
from enum import Enum

import structlog

from prism.core.config import (
    ModelConfig, ContentType, TaskType, RoutingStrategy
)


logger = structlog.get_logger(__name__)


@dataclass
class SelectionCriteria:
    """
    Criteria for model selection.
    
    Attributes:
        max_cost: Maximum cost per request
        max_latency_ms: Maximum acceptable latency in milliseconds
        min_accuracy: Minimum required accuracy score (0-1)
        required_capabilities: Required model capabilities
        preferred_providers: Preferred model providers (in order)
        exclude_providers: Providers to exclude
        context_window_needed: Required context window size
        streaming_required: Whether streaming is required
        batching_required: Whether request batching is required
    """
    max_cost: float | None = None
    max_latency_ms: float | None = None
    min_accuracy: float | None = None
    required_capabilities: list[str] = field(default_factory=list)
    preferred_providers: list[str] = field(default_factory=list)
    exclude_providers: list[str] = field(default_factory=list)
    context_window_needed: int | None = None
    streaming_required: bool = False
    batching_required: bool = False
    
    # Custom weights for scoring
    cost_weight: float = 1.0
    latency_weight: float = 1.0
    accuracy_weight: float = 1.0
    provider_preference_weight: float = 0.5
    
    def merge(self, other: "SelectionCriteria") -> "SelectionCriteria":
        """Merge with another criteria, taking minimum/maximum as appropriate."""
        return SelectionCriteria(
            max_cost=min(self.max_cost or float('inf'), other.max_cost or float('inf')),
            max_latency_ms=min(self.max_latency_ms or float('inf'), 
                              other.max_latency_ms or float('inf')),
            min_accuracy=max(self.min_accuracy or 0, other.min_accuracy or 0),
            required_capabilities=list(set(self.required_capabilities + other.required_capabilities)),
            preferred_providers=self.preferred_providers or other.preferred_providers,
            exclude_providers=list(set(self.exclude_providers + other.exclude_providers)),
            context_window_needed=max(self.context_window_needed or 0, 
                                     other.context_window_needed or 0),
            streaming_required=self.streaming_required or other.streaming_required,
            batching_required=self.batching_required or other.batching_required,
            cost_weight=self.cost_weight,
            latency_weight=self.latency_weight,
            accuracy_weight=self.accuracy_weight,
            provider_preference_weight=self.provider_preference_weight,
        )


@dataclass
class ModelScore:
    """Score breakdown for a model."""
    model_name: str
    total_score: float
    cost_score: float
    latency_score: float
    accuracy_score: float
    capability_score: float
    provider_score: float
    constraint_failures: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "total_score": self.total_score,
            "cost_score": self.cost_score,
            "latency_score": self.latency_score,
            "accuracy_score": self.accuracy_score,
            "capability_score": self.capability_score,
            "provider_score": self.provider_score,
            "constraint_failures": self.constraint_failures,
        }


class SelectionStrategy(Enum):
    """Predefined selection strategies."""
    CHEAPEST = "cheapest"
    FASTEST = "fastest"
    MOST_ACCURATE = "most_accurate"
    BEST_VALUE = "best_value"
    CUSTOM = "custom"


class ModelSelector:
    """
    Advanced model selector with multi-criteria optimization.
    
    Selects optimal models based on:
    - Cost efficiency
    - Latency requirements
    - Accuracy requirements
    - Model capabilities
    - Provider preferences
    - Custom constraints
    
    Example:
        >>> selector = ModelSelector()
        >>> criteria = SelectionCriteria(
        ...     max_cost=0.01,
        ...     max_latency_ms=2000,
        ...     min_accuracy=0.9
        ... )
        >>> models = selector.get_available_models(TaskType.TEXT_GENERATION, ContentType.TEXT)
        >>> selected = selector.select(models, criteria)
        >>> print(f"Selected: {selected.model_name}")
    """
    
    def __init__(
        self,
        models: dict[str, ModelConfig] | None = None,
        default_strategy: RoutingStrategy = RoutingStrategy.BALANCED
    ):
        """
        Initialize the model selector.
        
        Args:
            models: Dictionary of model configurations
            default_strategy: Default routing strategy
        """
        self.models = models or {}
        self.default_strategy = default_strategy
    
    def select(
        self,
        candidates: list[str | ModelConfig],
        criteria: SelectionCriteria,
        strategy: RoutingStrategy | None = None
    ) -> ModelScore | None:
        """
        Select the best model from candidates.
        
        Args:
            candidates: List of model names or ModelConfig objects
            criteria: Selection criteria
            strategy: Optional routing strategy override
            
        Returns:
            ModelScore for the selected model, or None if no suitable model found
        """
        # Convert names to configs if needed
        configs = []
        for candidate in candidates:
            if isinstance(candidate, str):
                if candidate in self.models:
                    configs.append(self.models[candidate])
                else:
                    logger.warning("model_not_found", model_name=candidate)
            else:
                configs.append(candidate)
        
        if not configs:
            return None
        
        # Score all candidates
        scored = []
        for config in configs:
            score = self._score_model(config, criteria, strategy)
            if score:
                # Check if all constraints are met
                if not score.constraint_failures:
                    scored.append(score)
        
        if not scored:
            return None
        
        # Return highest scoring model
        scored.sort(key=lambda x: x.total_score, reverse=True)
        return scored[0]
    
    def select_top_n(
        self,
        candidates: list[str | ModelConfig],
        criteria: SelectionCriteria,
        n: int = 3,
        strategy: RoutingStrategy | None = None
    ) -> list[ModelScore]:
        """
        Select top N models from candidates.
        
        Args:
            candidates: List of model names or ModelConfig objects
            criteria: Selection criteria
            n: Number of models to return
            strategy: Optional routing strategy override
            
        Returns:
            List of top N ModelScore objects
        """
        # Convert names to configs if needed
        configs = []
        for candidate in candidates:
            if isinstance(candidate, str):
                if candidate in self.models:
                    configs.append(self.models[candidate])
            else:
                configs.append(candidate)
        
        # Score all candidates
        scored = []
        for config in configs:
            score = self._score_model(config, criteria, strategy)
            if score:
                scored.append(score)
        
        # Sort by total score and return top N
        scored.sort(key=lambda x: x.total_score, reverse=True)
        return scored[:n]
    
    def get_available_models(
        self,
        task_type: TaskType | None = None,
        content_type: ContentType | None = None,
        criteria: SelectionCriteria | None = None
    ) -> list[ModelConfig]:
        """
        Get available models matching criteria.
        
        Args:
            task_type: Required task type
            content_type: Required content type
            criteria: Optional selection criteria
            
        Returns:
            List of matching ModelConfig objects
        """
        candidates = []
        
        for model in self.models.values():
            if not model.enabled:
                continue
            
            # Check content type
            if content_type and content_type not in model.content_types:
                continue
            
            # Check task type
            if task_type and task_type not in model.task_types:
                continue
            
            # Check criteria constraints
            if criteria:
                failures = self._check_constraints(model, criteria)
                if failures:
                    continue
            
            candidates.append(model)
        
        return candidates
    
    def _score_model(
        self,
        model: ModelConfig,
        criteria: SelectionCriteria,
        strategy: RoutingStrategy | None
    ) -> ModelScore | None:
        """Score a single model against criteria."""
        failures = self._check_constraints(model, criteria)
        
        # Calculate component scores
        cost_score = self._score_cost(model, criteria)
        latency_score = self._score_latency(model, criteria)
        accuracy_score = self._score_accuracy(model, criteria)
        capability_score = self._score_capabilities(model, criteria)
        provider_score = self._score_provider(model.provider.value, criteria)
        
        # Apply weights
        weights = self._get_weights(strategy, criteria)
        
        total_score = (
            cost_score * weights["cost"] +
            latency_score * weights["latency"] +
            accuracy_score * weights["accuracy"] +
            capability_score * weights.get("capability", 0.1) +
            provider_score * weights.get("provider", 0.1)
        )
        
        return ModelScore(
            model_name=model.name,
            total_score=total_score,
            cost_score=cost_score,
            latency_score=latency_score,
            accuracy_score=accuracy_score,
            capability_score=capability_score,
            provider_score=provider_score,
            constraint_failures=failures
        )
    
    def _check_constraints(
        self,
        model: ModelConfig,
        criteria: SelectionCriteria
    ) -> list[str]:
        """Check if model meets constraints."""
        failures = []
        
        if criteria.max_cost and model.total_cost_per_1k_tokens > criteria.max_cost:
            failures.append(f"cost_exceeds_max: {model.total_cost_per_1k_tokens} > {criteria.max_cost}")
        
        if criteria.max_latency_ms and model.avg_latency_ms > criteria.max_latency_ms:
            failures.append(f"latency_exceeds_max: {model.avg_latency_ms}ms > {criteria.max_latency_ms}ms")
        
        if criteria.min_accuracy and model.accuracy_score < criteria.min_accuracy:
            failures.append(f"accuracy_below_min: {model.accuracy_score} < {criteria.min_accuracy}")
        
        if criteria.context_window_needed and model.context_window < criteria.context_window_needed:
            failures.append(f"context_window_too_small: {model.context_window} < {criteria.context_window_needed}")
        
        if criteria.streaming_required and not model.supports_streaming:
            failures.append("streaming_required_but_not_supported")
        
        if criteria.batching_required and not model.supports_batching:
            failures.append("batching_required_but_not_supported")
        
        if criteria.exclude_providers and model.provider.value in criteria.exclude_providers:
            failures.append(f"provider_excluded: {model.provider.value}")
        
        return failures
    
    def _score_cost(self, model: ModelConfig, criteria: SelectionCriteria) -> float:
        """Score based on cost (lower is better)."""
        if not criteria.max_cost:
            # Normalize against max possible cost
            max_cost = 0.50
            cost = model.total_cost_per_1k_tokens
            return max(0, 1 - (cost / max_cost))
        
        # Score relative to max allowed cost
        cost = model.total_cost_per_1k_tokens
        if cost > criteria.max_cost:
            return 0.0
        
        return 1 - (cost / criteria.max_cost)
    
    def _score_latency(self, model: ModelConfig, criteria: SelectionCriteria) -> float:
        """Score based on latency (lower is better)."""
        if not criteria.max_latency_ms:
            # Normalize against max possible latency
            max_latency = 30000  # 30 seconds
            latency = model.avg_latency_ms
            return max(0, 1 - (latency / max_latency))
        
        # Score relative to max allowed latency
        latency = model.avg_latency_ms
        if latency > criteria.max_latency_ms:
            return 0.0
        
        return 1 - (latency / criteria.max_latency_ms)
    
    def _score_accuracy(self, model: ModelConfig, criteria: SelectionCriteria) -> float:
        """Score based on accuracy."""
        if not criteria.min_accuracy:
            return model.accuracy_score
        
        # If model meets minimum, score based on how much better it is
        if model.accuracy_score >= criteria.min_accuracy:
            return model.accuracy_score
        
        return 0.0
    
    def _score_capabilities(self, model: ModelConfig, criteria: SelectionCriteria) -> float:
        """Score based on capabilities."""
        if not criteria.required_capabilities:
            return 1.0
        
        # Check required capabilities in metadata
        model_caps = set(model.metadata.get("capabilities", []))
        required = set(criteria.required_capabilities)
        
        matched = len(model_caps & required)
        total = len(required)
        
        return matched / total if total > 0 else 1.0
    
    def _score_provider(self, provider: str, criteria: SelectionCriteria) -> float:
        """Score based on provider preference."""
        if not criteria.preferred_providers:
            return 0.5  # Neutral score
        
        try:
            index = criteria.preferred_providers.index(provider)
            # Higher preference = higher score
            return 1 - (index / len(criteria.preferred_providers))
        except ValueError:
            return 0.0
    
    def _get_weights(
        self,
        strategy: RoutingStrategy | None,
        criteria: SelectionCriteria
    ) -> dict[str, float]:
        """Get scoring weights based on strategy and criteria."""
        # Base weights
        weights = {
            "cost": criteria.cost_weight,
            "latency": criteria.latency_weight,
            "accuracy": criteria.accuracy_weight,
        }
        
        # Override with strategy weights
        if strategy:
            if strategy == RoutingStrategy.COST_OPTIMIZED:
                weights = {"cost": 0.7, "latency": 0.2, "accuracy": 0.1}
            elif strategy == RoutingStrategy.LATENCY_OPTIMIZED:
                weights = {"cost": 0.1, "latency": 0.7, "accuracy": 0.2}
            elif strategy == RoutingStrategy.ACCURACY_OPTIMIZED:
                weights = {"cost": 0.1, "latency": 0.2, "accuracy": 0.7}
            elif strategy == RoutingStrategy.BALANCED:
                weights = {"cost": 0.33, "latency": 0.33, "accuracy": 0.34}
        
        # Normalize weights
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        # Add capability and provider weights
        weights["capability"] = 0.1
        weights["provider"] = criteria.provider_preference_weight
        
        return weights
    
    def create_criteria_from_task(
        self,
        task_type: TaskType,
        content_type: ContentType | None = None
    ) -> SelectionCriteria:
        """
        Create default selection criteria for a task type.
        
        Args:
            task_type: Type of task
            content_type: Type of content
            
        Returns:
            SelectionCriteria with sensible defaults
        """
        # Define sensible defaults per task type
        task_defaults = {
            TaskType.TEXT_GENERATION: SelectionCriteria(
                max_latency_ms=5000,
                min_accuracy=0.85,
                latency_weight=1.5,  # Latency matters for generation
            ),
            TaskType.TEXT_CLASSIFICATION: SelectionCriteria(
                max_latency_ms=2000,
                min_accuracy=0.90,
                accuracy_weight=1.5,  # Accuracy matters for classification
            ),
            TaskType.TEXT_QA: SelectionCriteria(
                max_latency_ms=3000,
                min_accuracy=0.88,
                cost_weight=1.2,  # Balance cost and accuracy
            ),
            TaskType.TEXT_SUMMARIZATION: SelectionCriteria(
                max_latency_ms=4000,
                min_accuracy=0.85,
                latency_weight=1.2,
            ),
            TaskType.IMAGE_GENERATION: SelectionCriteria(
                max_latency_ms=30000,
                min_accuracy=0.80,
                max_cost=0.10,  # Image gen can be expensive
            ),
            TaskType.IMAGE_CLASSIFICATION: SelectionCriteria(
                max_latency_ms=2000,
                min_accuracy=0.90,
                accuracy_weight=1.5,
            ),
            TaskType.SPEECH_TO_TEXT: SelectionCriteria(
                max_latency_ms=10000,
                min_accuracy=0.85,
                latency_weight=1.3,
            ),
            TaskType.TEXT_TO_SPEECH: SelectionCriteria(
                max_latency_ms=5000,
                min_accuracy=0.90,
                latency_weight=1.4,  # Latency critical for TTS
            ),
        }
        
        criteria = task_defaults.get(task_type, SelectionCriteria())
        
        # Add content type if specified
        if content_type == ContentType.IMAGE:
            criteria.required_capabilities.append("vision")
        elif content_type == ContentType.AUDIO:
            criteria.required_capabilities.append("audio")
        
        return criteria
    
    def compare_models(
        self,
        model_a: str,
        model_b: str,
        criteria: SelectionCriteria | None = None
    ) -> dict[str, Any]:
        """
        Compare two models against each other.
        
        Args:
            model_a: First model name
            model_b: Second model name
            criteria: Selection criteria
            
        Returns:
            Dictionary with comparison results
        """
        if criteria is None:
            criteria = SelectionCriteria()
        
        config_a = self.models.get(model_a)
        config_b = self.models.get(model_b)
        
        if not config_a or not config_b:
            return {"error": "One or both models not found"}
        
        score_a = self._score_model(config_a, criteria, None)
        score_b = self._score_model(config_b, criteria, None)
        
        if not score_a or not score_b:
            return {"error": "Could not score one or both models"}
        
        return {
            "model_a": model_a,
            "model_b": model_b,
            "scores": {
                model_a: score_a.to_dict(),
                model_b: score_b.to_dict(),
            },
            "winner": model_a if score_a.total_score > score_b.total_score else model_b,
            "score_difference": abs(score_a.total_score - score_b.total_score),
        }
    
    def get_model_recommendations(
        self,
        task_type: TaskType,
        content_type: ContentType | None = None,
        top_n: int = 5
    ) -> list[dict[str, Any]]:
        """
        Get model recommendations for a task.
        
        Args:
            task_type: Type of task
            content_type: Type of content
            top_n: Number of recommendations
            
        Returns:
            List of model recommendations with scores
        """
        criteria = self.create_criteria_from_task(task_type, content_type)
        available = self.get_available_models(task_type, content_type, criteria)
        
        # Score all available models
        scored = []
        for model in available:
            score = self._score_model(model, criteria, None)
            if score and not score.constraint_failures:
                scored.append({
                    "model_name": model.name,
                    "provider": model.provider.value,
                    "total_score": score.total_score,
                    "cost_score": score.cost_score,
                    "latency_score": score.latency_score,
                    "accuracy_score": score.accuracy_score,
                    "estimated_cost": model.total_cost_per_1k_tokens,
                    "estimated_latency_ms": model.avg_latency_ms,
                    "accuracy": model.accuracy_score,
                })
        
        # Sort and return top N
        scored.sort(key=lambda x: x["total_score"], reverse=True)
        return scored[:top_n]
