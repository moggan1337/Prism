"""
Base Adapter - Abstract interface for model adapters.

Defines the contract that all model adapters must implement
for interacting with different AI model providers.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import structlog


logger = structlog.get_logger(__name__)


@dataclass
class AdapterResponse:
    """Response from a model adapter."""
    content: Any
    model: str
    provider: str
    success: bool
    latency_ms: float
    tokens_used: int | None = None
    cost: float | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "model": self.model,
            "provider": self.provider,
            "success": self.success,
            "latency_ms": self.latency_ms,
            "tokens_used": self.tokens_used,
            "cost": self.cost,
            "error": self.error,
            "metadata": self.metadata,
            "finish_reason": self.finish_reason,
        }


class ModelAdapter(ABC):
    """
    Abstract base class for model adapters.
    
    All model provider adapters must inherit from this class
    and implement the required methods.
    
    Example:
        >>> class MyModelAdapter(ModelAdapter):
        ...     async def execute(self, content, task_type, metadata):
        ...         # Implement model execution
        ...         return AdapterResponse(...)
        ...     
        ...     async def execute_stream(self, content, task_type, metadata):
        ...         # Implement streaming execution
        ...         yield partial_response
    """
    
    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int = 60,
        max_retries: int = 3
    ):
        """
        Initialize the adapter.
        
        Args:
            model_name: Name of the model to use
            api_key: API key for authentication
            base_url: Base URL for API requests
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
        """
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self._health_check_cache = True
        self._last_health_check = 0.0
    
    @abstractmethod
    async def execute(
        self,
        content: Any,
        task_type: str | None = None,
        metadata: dict[str, Any] | None = None
    ) -> AdapterResponse:
        """
        Execute a request to the model.
        
        Args:
            content: Input content (text, image, etc.)
            task_type: Type of task (e.g., "text_generation", "image_classification")
            metadata: Additional metadata for the request
            
        Returns:
            AdapterResponse with model output
        """
        pass
    
    async def execute_stream(
        self,
        content: Any,
        task_type: str | None = None,
        metadata: dict[str, Any] | None = None
    ) -> AsyncIterator[AdapterResponse]:
        """
        Execute a streaming request to the model.
        
        Override this method for streaming support.
        
        Args:
            content: Input content
            task_type: Type of task
            metadata: Additional metadata
            
        Yields:
            AdapterResponse for each chunk
        """
        # Default implementation: execute normally and yield single response
        response = await self.execute(content, task_type, metadata)
        yield response
    
    async def health_check(self) -> bool:
        """
        Check if the model adapter is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        # Cache health check for 30 seconds
        if self._health_check_cache and time.time() - self._last_health_check < 30:
            return True
        
        try:
            # Simple health check: try to get a very cheap response
            response = await self.execute(
                content="health_check",
                task_type="text_generation",
                metadata={"max_tokens": 1}
            )
            
            is_healthy = response.success
            self._last_health_check = time.time()
            
            return is_healthy
        
        except Exception as e:
            logger.warning("health_check_failed", model=self.model_name, error=str(e))
            return False
    
    def get_capabilities(self) -> dict[str, bool]:
        """
        Get adapter capabilities.
        
        Returns:
            Dictionary of capability names to supported status
        """
        return {
            "streaming": self.supports_streaming(),
            "batching": self.supports_batching(),
            "vision": self.supports_vision(),
            "audio": self.supports_audio(),
            "function_calling": self.supports_function_calling(),
        }
    
    def supports_streaming(self) -> bool:
        """Check if streaming is supported."""
        return False
    
    def supports_batching(self) -> bool:
        """Check if request batching is supported."""
        return False
    
    def supports_vision(self) -> bool:
        """Check if vision/image input is supported."""
        return False
    
    def supports_audio(self) -> bool:
        """Check if audio input/output is supported."""
        return False
    
    def supports_function_calling(self) -> bool:
        """Check if function calling is supported."""
        return False
    
    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int | None = None
    ) -> float:
        """
        Estimate cost for a request.
        
        Override this method with actual pricing.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens (if known)
            
        Returns:
            Estimated cost in dollars
        """
        # Default estimate based on generic pricing
        input_cost = input_tokens * 0.00001  # $0.01 per 1k tokens
        output_cost = (output_tokens or input_tokens) * 0.00003  # $0.03 per 1k tokens
        return input_cost + output_cost
    
    async def close(self) -> None:
        """
        Close any resources held by the adapter.
        
        Override this method for cleanup.
        """
        pass
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} model={self.model_name}>"


class RetryableError(Exception):
    """Error that can be retried."""
    pass


class RateLimitError(RetryableError):
    """Rate limit exceeded error."""
    pass


class AuthenticationError(Exception):
    """Authentication failed error."""
    pass


class ModelError(Exception):
    """Generic model error."""
    pass
