"""
OpenAI Adapter - Adapter for OpenAI models.

Supports GPT-4, GPT-3.5, DALL-E, Whisper, and other OpenAI models.
"""

from __future__ import annotations

import time
from typing import Any

import structlog

from prism.adapters.base import (
    ModelAdapter, AdapterResponse, RateLimitError, AuthenticationError, ModelError
)
from prism.core.config import ProviderConfig


logger = structlog.get_logger(__name__)


class OpenAIAdapter(ModelAdapter):
    """
    Adapter for OpenAI API models.
    
    Supports:
    - GPT-4 and GPT-3.5 for text
    - DALL-E for image generation
    - Whisper for speech-to-text
    - GPT-4 Vision for image understanding
    """
    
    def __init__(
        self,
        model_name: str,
        provider_config: ProviderConfig | None = None,
        base_url: str | None = None
    ):
        super().__init__(
            model_name=model_name,
            api_key=provider_config.api_key if provider_config else None,
            base_url=base_url or "https://api.openai.com/v1",
            timeout=provider_config.timeout_seconds if provider_config else 60,
            max_retries=provider_config.max_retries if provider_config else 3
        )
        self.provider_config = provider_config
    
    async def execute(
        self,
        content: Any,
        task_type: str | None = None,
        metadata: dict[str, Any] | None = None
    ) -> AdapterResponse:
        """
        Execute request to OpenAI API.
        
        Args:
            content: Input content (string for text, dict with image_url/base64 for vision)
            task_type: Task type (text_generation, image_generation, etc.)
            metadata: Additional parameters (temperature, max_tokens, etc.)
        """
        start_time = time.time()
        metadata = metadata or {}
        
        try:
            # Determine endpoint based on model and task
            if "dall" in self.model_name.lower():
                return await self._execute_image_generation(content, metadata, start_time)
            elif "whisper" in self.model_name.lower():
                return await self._execute_speech_to_text(content, metadata, start_time)
            elif "gpt-4-vision" in self.model_name.lower():
                return await self._execute_vision(content, metadata, start_time)
            else:
                return await self._execute_chat_completion(content, metadata, start_time)
        
        except Exception as e:
            return AdapterResponse(
                content=None,
                model=self.model_name,
                provider="openai",
                success=False,
                latency_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )
    
    async def _execute_chat_completion(
        self,
        content: Any,
        metadata: dict[str, Any],
        start_time: float
    ) -> AdapterResponse:
        """Execute chat completion request."""
        import httpx
        
        # Format messages
        messages = self._format_messages(content)
        
        # Build request
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": metadata.get("temperature", 0.7),
            "max_tokens": metadata.get("max_tokens", 2048),
            "stream": False,
        }
        
        # Add optional parameters
        if "top_p" in metadata:
            payload["top_p"] = metadata["top_p"]
        if "frequency_penalty" in metadata:
            payload["frequency_penalty"] = metadata["frequency_penalty"]
        if "presence_penalty" in metadata:
            payload["presence_penalty"] = metadata["presence_penalty"]
        if "functions" in metadata:
            payload["functions"] = metadata["functions"]
        
        # Make request
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            )
            
            if response.status_code == 401:
                raise AuthenticationError("Invalid API key")
            elif response.status_code == 429:
                raise RateLimitError("Rate limit exceeded")
            elif response.status_code >= 400:
                raise ModelError(f"API error: {response.text}")
            
            data = response.json()
            
            return AdapterResponse(
                content=data["choices"][0]["message"]["content"],
                model=self.model_name,
                provider="openai",
                success=True,
                latency_ms=(time.time() - start_time) * 1000,
                tokens_used=data.get("usage", {}).get("total_tokens", 0),
                cost=self._calculate_cost(data),
                metadata=data,
                finish_reason=data["choices"][0].get("finish_reason")
            )
    
    async def _execute_vision(
        self,
        content: Any,
        metadata: dict[str, Any],
        start_time: float
    ) -> AdapterResponse:
        """Execute vision request."""
        import httpx
        
        # Format messages with image
        if isinstance(content, dict):
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": content.get("text", "Describe this image.")},
                    {
                        "type": "image_url",
                        "image_url": content.get("image_url", {})
                    }
                ]
            }]
        else:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": str(content)},
                ]
            }]
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": metadata.get("max_tokens", 2048),
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            )
            
            if response.status_code >= 400:
                raise ModelError(f"API error: {response.text}")
            
            data = response.json()
            
            return AdapterResponse(
                content=data["choices"][0]["message"]["content"],
                model=self.model_name,
                provider="openai",
                success=True,
                latency_ms=(time.time() - start_time) * 1000,
                tokens_used=data.get("usage", {}).get("total_tokens", 0),
                cost=self._calculate_cost(data),
                metadata=data
            )
    
    async def _execute_image_generation(
        self,
        content: Any,
        metadata: dict[str, Any],
        start_time: float
    ) -> AdapterResponse:
        """Execute DALL-E image generation."""
        import httpx
        
        payload = {
            "model": self.model_name,
            "prompt": str(content),
            "n": metadata.get("n", 1),
            "size": metadata.get("size", "1024x1024"),
            "response_format": metadata.get("response_format", "url"),
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/images/generations",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            )
            
            if response.status_code >= 400:
                raise ModelError(f"API error: {response.text}")
            
            data = response.json()
            
            # Calculate cost
            num_images = len(data.get("data", []))
            cost = num_images * 0.04  # DALL-E 3 is $0.04 per image
            
            return AdapterResponse(
                content=data["data"],
                model=self.model_name,
                provider="openai",
                success=True,
                latency_ms=(time.time() - start_time) * 1000,
                cost=cost,
                metadata=data
            )
    
    async def _execute_speech_to_text(
        self,
        content: Any,
        metadata: dict[str, Any],
        start_time: float
    ) -> AdapterResponse:
        """Execute Whisper speech-to-text."""
        import httpx
        
        # Handle different input formats
        if isinstance(content, bytes):
            files = {"file": ("audio.wav", content, "audio/wav")}
        elif isinstance(content, str) and content.startswith("http"):
            # URL to audio file
            async with httpx.AsyncClient() as client:
                response = await client.get(content)
                files = {"file": ("audio.wav", response.content, "audio/wav")}
        else:
            raise ValueError("Content must be bytes or URL string for Whisper")
        
        data = {"model": self.model_name}
        if "language" in metadata:
            data["language"] = metadata["language"]
        if "prompt" in metadata:
            data["prompt"] = metadata["prompt"]
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/audio/transcriptions",
                files=files,
                data=data,
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            
            if response.status_code >= 400:
                raise ModelError(f"API error: {response.text}")
            
            result = response.json()
            
            # Estimate cost based on duration
            duration_seconds = metadata.get("duration", 60)
            cost = (duration_seconds / 60) * 0.006  # $0.006 per minute
            
            return AdapterResponse(
                content=result["text"],
                model=self.model_name,
                provider="openai",
                success=True,
                latency_ms=(time.time() - start_time) * 1000,
                cost=cost,
                metadata=result
            )
    
    def _format_messages(self, content: Any) -> list[dict[str, str]]:
        """Format content into messages."""
        if isinstance(content, str):
            return [{"role": "user", "content": content}]
        elif isinstance(content, list):
            return content  # Assume already formatted
        elif isinstance(content, dict):
            return [{"role": content.get("role", "user"), "content": str(content)}]
        else:
            return [{"role": "user", "content": str(content)}]
    
    def _calculate_cost(self, data: dict[str, Any]) -> float:
        """Calculate cost from response."""
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        
        # Pricing (can be configured)
        if "gpt-4" in self.model_name:
            cost = (prompt_tokens * 0.00001 + completion_tokens * 0.00003)
        elif "gpt-3.5" in self.model_name:
            cost = (prompt_tokens * 0.0000005 + completion_tokens * 0.0000015)
        else:
            cost = (prompt_tokens * 0.00001 + completion_tokens * 0.00003)
        
        return cost
    
    def supports_streaming(self) -> bool:
        """OpenAI supports streaming."""
        return True
    
    def supports_vision(self) -> bool:
        """GPT-4 Vision supports images."""
        return "vision" in self.model_name.lower()
    
    def supports_batching(self) -> bool:
        """OpenAI supports batch requests."""
        return True
