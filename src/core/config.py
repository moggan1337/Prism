"""
Configuration models for Prism.

Defines all configuration options for the orchestrator including
model providers, routing strategies, caching, and budget settings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ContentType(str, Enum):
    """Supported content types for routing."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"
    UNKNOWN = "unknown"


class TaskType(str, Enum):
    """Task types for model selection."""
    # Text tasks
    TEXT_GENERATION = "text_generation"
    TEXT_CLASSIFICATION = "text_classification"
    TEXT_EXTRACTION = "text_extraction"
    TEXT_SUMMARIZATION = "text_summarization"
    TEXT_TRANSLATION = "text_translation"
    TEXT_QUESTION_ANSWERING = "text_qa"
    
    # Image tasks
    IMAGE_CLASSIFICATION = "image_classification"
    IMAGE_GENERATION = "image_generation"
    IMAGE_EDITING = "image_editing"
    OBJECT_DETECTION = "object_detection"
    OCR = "ocr"
    
    # Audio tasks
    SPEECH_TO_TEXT = "speech_to_text"
    TEXT_TO_SPEECH = "text_to_speech"
    AUDIO_CLASSIFICATION = "audio_classification"
    
    # Video tasks
    VIDEO_CLASSIFICATION = "video_classification"
    VIDEO_GENERATION = "video_generation"
    
    # Document tasks
    DOCUMENT_PARSING = "document_parsing"
    DOCUMENT_QA = "document_qa"
    KEY_VALUE_EXTRACTION = "key_value_extraction"
    
    # Multi-modal tasks
    MULTIMODAL_QA = "multimodal_qa"
    VISION_LANGUAGE = "vision_language"


class RoutingStrategy(str, Enum):
    """Routing strategies for model selection."""
    COST_OPTIMIZED = "cost_optimized"
    LATENCY_OPTIMIZED = "latency_optimized"
    ACCURACY_OPTIMIZED = "accuracy_optimized"
    BALANCED = "balanced"
    CUSTOM = "custom"


class ModelProvider(str, Enum):
    """Supported AI model providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    MISTRAL = "mistral"
    COHERE = "cohere"
    STABILITY = "stability"
    ELEVEN_LABS = "eleven_labs"
    WHISPER = "whisper"
    LOCAL = "local"
    CUSTOM = "custom"


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    name: str
    provider: ModelProvider
    task_types: list[TaskType]
    content_types: list[ContentType]
    cost_per_1k_input_tokens: float = 0.0
    cost_per_1k_output_tokens: float = 0.0
    cost_per_image: float = 0.0
    cost_per_second: float = 0.0
    avg_latency_ms: float = 1000.0
    max_tokens: int = 4096
    context_window: int = 8192
    accuracy_score: float = 0.9
    supports_streaming: bool = False
    supports_batching: bool = False
    rate_limit_rpm: int = 60
    rate_limit_tpm: int = 100000
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_cost_per_1k_tokens(self) -> float:
        """Calculate total cost per 1k tokens (input + output)."""
        return self.cost_per_1k_input_tokens + self.cost_per_1k_output_tokens


@dataclass
class ProviderConfig:
    """Configuration for a model provider."""
    name: ModelProvider
    api_key: str | None = None
    base_url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    timeout_seconds: int = 60
    max_retries: int = 3
    retry_backoff_factor: float = 2.0
    enabled: bool = True


@dataclass
class CacheConfig:
    """Configuration for caching."""
    enabled: bool = True
    backend: str = "memory"  # "memory", "redis", "disk"
    redis_url: str | None = None
    ttl_seconds: int = 3600
    max_size_mb: int = 512
    key_prefix: str = "prism:"
    
    # Cache strategies by content type
    text_ttl_seconds: int = 3600
    image_ttl_seconds: int = 86400
    audio_ttl_seconds: int = 86400
    video_ttl_seconds: int = 604800
    document_ttl_seconds: int = 604800


@dataclass
class BudgetConfig:
    """Configuration for budget enforcement."""
    enabled: bool = True
    monthly_limit: float = 1000.0
    daily_limit: float | None = None
    per_request_limit: float | None = None
    track_by_user: bool = False
    track_by_api_key: bool = False
    alert_threshold_percent: float = 80.0
    auto_fallback_to_cheaper: bool = True
    hard_limit_enabled: bool = True


@dataclass
class ObservabilityConfig:
    """Configuration for observability features."""
    tracing_enabled: bool = True
    metrics_enabled: bool = True
    log_level: str = "INFO"
    
    # OpenTelemetry settings
    otlp_endpoint: str | None = None
    service_name: str = "prism-router"
    trace_sample_rate: float = 1.0
    
    # Prometheus settings
    prometheus_port: int = 9090
    prometheus_prefix: str = "prism_"
    
    # Custom metrics
    enable_cost_tracking: bool = True
    enable_latency_tracking: bool = True
    enable_accuracy_tracking: bool = True
    enable_cache_hit_tracking: bool = True


@dataclass
class BatchingConfig:
    """Configuration for request batching."""
    enabled: bool = True
    max_batch_size: int = 10
    max_wait_time_ms: int = 100
    max_batch_delay_ms: int = 500
    auto_flush_on_size: bool = True
    priority_bypass: bool = False


@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    max_attempts: int = 3
    initial_delay_ms: int = 100
    max_delay_ms: int = 10000
    backoff_factor: float = 2.0
    jitter: bool = True
    retry_on_status_codes: list[int] = field(
        default_factory=lambda: [429, 500, 502, 503, 504]
    )
    retry_on_errors: list[str] = field(
        default_factory=lambda: ["RateLimitError", "TimeoutError", "ServiceUnavailable"]
    )


@dataclass
class FallbackConfig:
    """Configuration for fallback behavior."""
    enabled: bool = True
    fallback_models: dict[TaskType, list[str]] = field(default_factory=dict)
    fallback_on_timeout: bool = True
    fallback_on_error: bool = True
    fallback_on_cost_exceeded: bool = True
    aggregate_fallbacks: bool = False
    min_fallback_success_rate: float = 0.5


@dataclass
class AggregationConfig:
    """Configuration for multi-model aggregation."""
    enabled: bool = False
    models_per_request: int = 2
    aggregation_strategy: str = "weighted_average"  # "weighted_average", "majority_vote", "hierarchical"
    weight_by_accuracy: bool = True
    weight_by_cost: bool = False
    require_consensus: bool = False
    consensus_threshold: float = 0.7


class PrismConfig(BaseModel):
    """Main configuration for Prism orchestrator."""
    
    # Model configurations
    models: dict[str, ModelConfig] = Field(default_factory=dict)
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    
    # Routing settings
    default_routing_strategy: RoutingStrategy = RoutingStrategy.BALANCED
    routing_prompt: str | None = None
    
    # Component configurations
    cache: CacheConfig = Field(default_factory=CacheConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    batching: BatchingConfig = Field(default_factory=BatchingConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    fallback: FallbackConfig = Field(default_factory=FallbackConfig)
    aggregation: AggregationConfig = Field(default_factory=AggregationConfig)
    
    # General settings
    enable_circuit_breaker: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout_seconds: int = 60
    enable_request_id: bool = True
    custom_headers: dict[str, str] = Field(default_factory=dict)
    
    @classmethod
    def from_env(cls) -> "PrismConfig":
        """Load configuration from environment variables."""
        config = cls()
        
        # Load API keys from environment
        for provider in ModelProvider:
            env_key = f"{provider.value.upper()}_API_KEY"
            if api_key := os.getenv(env_key):
                config.providers[provider.value] = ProviderConfig(
                    name=provider,
                    api_key=api_key,
                    base_url=os.getenv(f"{provider.value.upper()}_BASE_URL")
                )
        
        return config
    
    @classmethod
    def load_default_models(cls) -> dict[str, ModelConfig]:
        """Load default model configurations."""
        return {
            # Text Models
            "gpt-4-turbo": ModelConfig(
                name="gpt-4-turbo",
                provider=ModelProvider.OPENAI,
                task_types=[TaskType.TEXT_GENERATION, TaskType.TEXT_CLASSIFICATION, 
                           TaskType.TEXT_EXTRACTION, TaskType.TEXT_QA, TaskType.MULTIMODAL_QA],
                content_types=[ContentType.TEXT, ContentType.IMAGE],
                cost_per_1k_input_tokens=0.01,
                cost_per_1k_output_tokens=0.03,
                avg_latency_ms=2000,
                max_tokens=128000,
                context_window=128000,
                accuracy_score=0.95,
                supports_streaming=True,
                supports_batching=True,
                rate_limit_rpm=500,
                rate_limit_tpm=2000000
            ),
            "gpt-3.5-turbo": ModelConfig(
                name="gpt-3.5-turbo",
                provider=ModelProvider.OPENAI,
                task_types=[TaskType.TEXT_GENERATION, TaskType.TEXT_CLASSIFICATION,
                           TaskType.TEXT_EXTRACTION, TaskType.TEXT_QA],
                content_types=[ContentType.TEXT],
                cost_per_1k_input_tokens=0.0005,
                cost_per_1k_output_tokens=0.0015,
                avg_latency_ms=500,
                max_tokens=16385,
                context_window=16385,
                accuracy_score=0.88,
                supports_streaming=True,
                supports_batching=True,
                rate_limit_rpm=3500,
                rate_limit_tpm=150000
            ),
            "claude-3-opus": ModelConfig(
                name="claude-3-opus",
                provider=ModelProvider.ANTHROPIC,
                task_types=[TaskType.TEXT_GENERATION, TaskType.TEXT_CLASSIFICATION,
                           TaskType.TEXT_EXTRACTION, TaskType.TEXT_SUMMARIZATION, 
                           TaskType.TEXT_QA, TaskType.MULTIMODAL_QA],
                content_types=[ContentType.TEXT, ContentType.IMAGE],
                cost_per_1k_input_tokens=0.015,
                cost_per_1k_output_tokens=0.075,
                avg_latency_ms=3000,
                max_tokens=4096,
                context_window=200000,
                accuracy_score=0.97,
                supports_streaming=True,
                supports_batching=False,
                rate_limit_rpm=50,
                rate_limit_tpm=200000
            ),
            "claude-3-sonnet": ModelConfig(
                name="claude-3-sonnet",
                provider=ModelProvider.ANTHROPIC,
                task_types=[TaskType.TEXT_GENERATION, TaskType.TEXT_CLASSIFICATION,
                           TaskType.TEXT_EXTRACTION, TaskType.TEXT_QA, TaskType.MULTIMODAL_QA],
                content_types=[ContentType.TEXT, ContentType.IMAGE],
                cost_per_1k_input_tokens=0.003,
                cost_per_1k_output_tokens=0.015,
                avg_latency_ms=1500,
                max_tokens=4096,
                context_window=200000,
                accuracy_score=0.94,
                supports_streaming=True,
                supports_batching=False,
                rate_limit_rpm=80,
                rate_limit_tpm=400000
            ),
            "gemini-pro": ModelConfig(
                name="gemini-pro",
                provider=ModelProvider.GOOGLE,
                task_types=[TaskType.TEXT_GENERATION, TaskType.TEXT_CLASSIFICATION,
                           TaskType.TEXT_QA, TaskType.MULTIMODAL_QA],
                content_types=[ContentType.TEXT, ContentType.IMAGE],
                cost_per_1k_input_tokens=0.000125,
                cost_per_1k_output_tokens=0.000375,
                avg_latency_ms=1000,
                max_tokens=8192,
                context_window=32768,
                accuracy_score=0.92,
                supports_streaming=True,
                supports_batching=True,
                rate_limit_rpm=60,
                rate_limit_tpm=1000000
            ),
            "mistral-large": ModelConfig(
                name="mistral-large",
                provider=ModelProvider.MISTRAL,
                task_types=[TaskType.TEXT_GENERATION, TaskType.TEXT_CLASSIFICATION,
                           TaskType.TEXT_EXTRACTION, TaskType.TEXT_QA],
                content_types=[ContentType.TEXT],
                cost_per_1k_input_tokens=0.008,
                cost_per_1k_output_tokens=0.024,
                avg_latency_ms=1500,
                max_tokens=32000,
                context_window=128000,
                accuracy_score=0.93,
                supports_streaming=True,
                supports_batching=False,
                rate_limit_rpm=100,
                rate_limit_tpm=500000
            ),
            # Vision Models
            "dall-e-3": ModelConfig(
                name="dall-e-3",
                provider=ModelProvider.OPENAI,
                task_types=[TaskType.IMAGE_GENERATION],
                content_types=[ContentType.IMAGE],
                cost_per_image=0.04,
                avg_latency_ms=10000,
                accuracy_score=0.9,
                supports_streaming=False,
                supports_batching=False,
                rate_limit_rpm=5,
                rate_limit_tpm=50
            ),
            "gpt-4-vision": ModelConfig(
                name="gpt-4-vision",
                provider=ModelProvider.OPENAI,
                task_types=[TaskType.IMAGE_CLASSIFICATION, TaskType.OBJECT_DETECTION,
                           TaskType.OCR, TaskType.MULTIMODAL_QA],
                content_types=[ContentType.IMAGE],
                cost_per_1k_input_tokens=0.01,
                cost_per_1k_output_tokens=0.03,
                avg_latency_ms=2500,
                max_tokens=4096,
                context_window=128000,
                accuracy_score=0.94,
                supports_streaming=False,
                supports_batching=False,
                rate_limit_rpm=500,
                rate_limit_tpm=2000000
            ),
            # Audio Models
            "whisper-1": ModelConfig(
                name="whisper-1",
                provider=ModelProvider.WHISPER,
                task_types=[TaskType.SPEECH_TO_TEXT],
                content_types=[ContentType.AUDIO],
                cost_per_minute=0.006,
                avg_latency_ms=30000,
                accuracy_score=0.92,
                supports_streaming=False,
                supports_batching=True,
                rate_limit_rpm=50,
                rate_limit_tpm=500000
            ),
            "eleven-multilingual-v2": ModelConfig(
                name="eleven-multilingual-v2",
                provider=ModelProvider.ELEVEN_LABS,
                task_types=[TaskType.TEXT_TO_SPEECH],
                content_types=[ContentType.AUDIO],
                cost_per_character=0.00018,
                avg_latency_ms=5000,
                accuracy_score=0.95,
                supports_streaming=True,
                supports_batching=False,
                rate_limit_rpm=100,
                rate_limit_tpm=1000000
            ),
        }
