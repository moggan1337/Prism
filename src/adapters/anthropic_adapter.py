"""
Anthropic Adapter - Adapter for Anthropic Claude models.

Supports Claude 3 Opus, Sonnet, and Haiku.
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


class AnthropicAdapter(ModelAdapter):
    """
    Adapter for Anthropic Claude API.
    
    Supports:
    - Claude 3 Opus
    - Claude 3 Sonnet
    - Claude 3 Haiku
    """
    
    BASE_URL = "https://api.anthropic.com/v1"
    
    def __init__(
        self,
        model_name: str,
        provider_config: ProviderConfig | None = None,
        base_url: str | None = None
    ):
        super().__init__(
            model_name=model_name,
            api_key=provider_config.api_key if provider_config else None,
            base_url=base_url or self.BASE_URL,
            timeout=provider_config.timeout_seconds if provider_config else 120,
            max_retries=provider_config.max_retries if provider_config else 3
        )
        self.provider_config = provider_config
    
    async def execute(
        self,
        content: Any,
        task_type: str | None = None,
        metadata: dict[str, Any] | None = None
    ) -> AdapterResponse:
        """Execute request to Anthropic API."""
        start_time = time.time()
        metadata = metadata or {}
        
        try:
            # Check if vision request
            if self._is_vision_request(content):
                return await self._execute_vision(content, metadata, start_time)
            
            return await self._execute_completion(content, metadata, start_time)
        
        except Exception as e:
            return AdapterResponse(
                content=None,
                model=self.model_name,
                provider="anthropic",
                success=False,
                latency_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )
    
    async def _execute_completion(
        self,
        content: Any,
        metadata: dict[str, Any],
        start_time: float
    ) -> AdapterResponse:
        """Execute text completion request."""
        import httpx
        
        # Format messages
        messages = self._format_messages(content)
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": metadata.get("max_tokens", 4096),
            "temperature": metadata.get("temperature", 1.0),
            "top_p": metadata.get("top_p", 0.999),
            "top_k": metadata.get("top_k", 250),
        }
        
        # Add streaming
        if metadata.get("stream", False):
            payload["stream"] = True
        
        # Add system prompt
        if "system" in metadata:
            payload["system"] = metadata["system"]
        
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/messages",
                json=payload,
                headers=headers
            )
            
            if response.status_code == 401:
                raise AuthenticationError("Invalid API key")
            elif response.status_code == 429:
                raise RateLimitError("Rate limit exceeded")
            elif response.status_code >= 400:
                raise ModelError(f"API error: {response.text}")
            
            data = response.json()
            
            return AdapterResponse(
                content=data["content"][0]["text"],
                model=self.model_name,
                provider="anthropic",
                success=True,
                latency_ms=(time.time() - start_time) * 1000,
                tokens_used=data.get("usage", {}).get("input_tokens", 0) + 
                           data.get("usage", {}).get("output_tokens", 0),
                cost=self._calculate_cost(data),
                metadata=data,
                finish_reason=data.get("stop_reason")
            )
    
    async def _execute_vision(
        self,
        content: Any,
        metadata: dict[str, Any],
        start_time: float
    ) -> AdapterResponse:
        """Execute vision request."""
        import httpx
        import base64
        
        # Format content with image
        content_blocks = []
        
        if isinstance(content, dict):
            # Text content
            if "text" in content:
                content_blocks.append({
                    "type": "text",
                    "text": content["text"]
                })
            
            # Image content
            if "image" in content:
                image_data = content["image"]
                if isinstance(image_data, bytes):
                    encoded = base64.b64encode(image_data).decode()
                    media_type = self._detect_image_type(image_data)
                    content_blocks.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": encoded
                        }
                    })
        else:
            content_blocks.append({
                "type": "text",
                "text": str(content)
            })
        
        messages = [{
            "role": "user",
            "content": content_blocks
        }]
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": metadata.get("max_tokens", 4096),
        }
        
        if "system" in metadata:
            payload["system"] = metadata["system"]
        
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/messages",
                json=payload,
                headers=headers
            )
            
            if response.status_code >= 400:
                raise ModelError(f"API error: {response.text}")
            
            data = response.json()
            
            return AdapterResponse(
                content=data["content"][0]["text"],
                model=self.model_name,
                provider="anthropic",
                success=True,
                latency_ms=(time.time() - start_time) * 1000,
                tokens_used=data.get("usage", {}).get("input_tokens", 0) + 
                           data.get("usage", {}).get("output_tokens", 0),
                cost=self._calculate_cost(data),
                metadata=data
            )
    
    def _is_vision_request(self, content: Any) -> bool:
        """Check if this is a vision request."""
        if isinstance(content, dict) and "image" in content:
            return True
        if isinstance(content, bytes):
            return True
        return False
    
    def _detect_image_type(self, data: bytes) -> str:
        """Detect image MIME type from bytes."""
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            return "image/png"
        elif data[:3] == b"\xff\xd8\xff":
            return "image/jpeg"
        elif data[:6] in (b"GIF87a", b"GIF89a"):
            return "image/gif"
        elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return "image/webp"
        return "image/png"
    
    def _format_messages(self, content: Any) -> list[dict[str, str]]:
        """Format content into messages."""
        if isinstance(content, str):
            return [{"role": "user", "content": content}]
        elif isinstance(content, list):
            return content
        elif isinstance(content, dict):
            return [{"role": content.get("role", "user"), "content": str(content)}]
        else:
            return [{"role": "user", "content": str(content)}]
    
    def _calculate_cost(self, data: dict[str, Any]) -> float:
        """Calculate cost from response."""
        usage = data.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        
        # Pricing for Claude 3
        if "opus" in self.model_name:
            cost = (input_tokens * 0.000015 + output_tokens * 0.000075)
        elif "sonnet" in self.model_name:
            cost = (input_tokens * 0.000003 + output_tokens * 0.000015)
        elif "haiku" in self.model_name:
            cost = (input_tokens * 0.0000008 + output_tokens * 0.000004)
        else:
            cost = (input_tokens * 0.000003 + output_tokens * 0.000015)
        
        return cost
    
    def supports_streaming(self) -> bool:
        """Anthropic supports streaming."""
        return True
    
    def supports_vision(self) -> bool:
        """Claude 3 supports vision."""
        return True
