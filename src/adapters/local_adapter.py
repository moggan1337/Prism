"""
Local Adapter - Adapter for local model servers.

Supports local LLM servers like:
- Ollama
- LM Studio
- LocalAI
- vLLM
- llama.cpp server
"""

from __future__ import annotations

import time
from typing import Any

import structlog

from prism.adapters.base import (
    ModelAdapter, AdapterResponse, ModelError
)
from prism.core.config import ProviderConfig


logger = structlog.get_logger(__name__)


class LocalAdapter(ModelAdapter):
    """
    Adapter for local model servers.
    
    Supports various local inference servers with OpenAI-compatible APIs.
    """
    
    def __init__(
        self,
        model_name: str,
        provider_config: ProviderConfig | None = None,
        base_url: str | None = None
    ):
        super().__init__(
            model_name=model_name,
            api_key=provider_config.api_key if provider_config else "not-required",
            base_url=base_url or "http://localhost:11434/v1",
            timeout=provider_config.timeout_seconds if provider_config else 300,
            max_retries=provider_config.max_retries if provider_config else 1
        )
        self.provider_config = provider_config
        self._server_type = self._detect_server_type()
    
    def _detect_server_type(self) -> str:
        """Detect local server type from base URL."""
        url = self.base_url.lower()
        
        if "11434" in url or "ollama" in url:
            return "ollama"
        elif "lm-studio" in url or "lmstudio" in url:
            return "lmstudio"
        elif "localai" in url:
            return "localai"
        elif "vllm" in url:
            return "vllm"
        elif "llama.cpp" in url:
            return "llamacpp"
        else:
            return "generic"
    
    async def execute(
        self,
        content: Any,
        task_type: str | None = None,
        metadata: dict[str, Any] | None = None
    ) -> AdapterResponse:
        """Execute request to local model server."""
        start_time = time.time()
        metadata = metadata or {}
        
        try:
            if self._server_type == "ollama":
                return await self._execute_ollama(content, metadata, start_time)
            else:
                return await self._execute_openai_compatible(content, metadata, start_time)
        
        except Exception as e:
            return AdapterResponse(
                content=None,
                model=self.model_name,
                provider="local",
                success=False,
                latency_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )
    
    async def _execute_ollama(
        self,
        content: Any,
        metadata: dict[str, Any],
        start_time: float
    ) -> AdapterResponse:
        """Execute request to Ollama."""
        import httpx
        
        # Format prompt
        prompt = self._format_prompt(content)
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": metadata.get("temperature", 0.7),
                "num_predict": metadata.get("max_tokens", 2048),
                "top_p": metadata.get("top_p", 0.9),
                "top_k": metadata.get("top_k", 40),
            }
        }
        
        # Add context for continuation
        if "context" in metadata:
            payload["context"] = metadata["context"]
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # Generate endpoint
            response = await client.post(
                "http://localhost:11434/api/generate",
                json=payload
            )
            
            if response.status_code >= 400:
                raise ModelError(f"Ollama error: {response.text}")
            
            data = response.json()
            
            return AdapterResponse(
                content=data.get("response", ""),
                model=self.model_name,
                provider="ollama",
                success=True,
                latency_ms=(time.time() - start_time) * 1000,
                tokens_used=data.get("eval_count", 0),
                metadata=data
            )
    
    async def _execute_openai_compatible(
        self,
        content: Any,
        metadata: dict[str, Any],
        start_time: float
    ) -> AdapterResponse:
        """Execute request to OpenAI-compatible server."""
        import httpx
        
        # Format messages
        messages = self._format_messages(content)
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": metadata.get("temperature", 0.7),
            "max_tokens": metadata.get("max_tokens", 2048),
            "stream": False,
        }
        
        if "top_p" in metadata:
            payload["top_p"] = metadata["top_p"]
        
        # Add auth header if provided
        headers = {}
        if self.api_key and self.api_key != "not-required":
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers
            )
            
            if response.status_code >= 400:
                raise ModelError(f"Server error: {response.text}")
            
            data = response.json()
            
            return AdapterResponse(
                content=data["choices"][0]["message"]["content"],
                model=self.model_name,
                provider="local",
                success=True,
                latency_ms=(time.time() - start_time) * 1000,
                tokens_used=data.get("usage", {}).get("total_tokens", 0),
                metadata=data
            )
    
    def _format_prompt(self, content: Any) -> str:
        """Format content as prompt."""
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            # Format as chat messages
            prompt_parts = []
            for msg in content:
                role = msg.get("role", "user")
                text = msg.get("content", "")
                prompt_parts.append(f"{role.capitalize()}: {text}")
            return "\n".join(prompt_parts)
        elif isinstance(content, dict):
            return str(content.get("content", content))
        else:
            return str(content)
    
    def _format_messages(self, content: Any) -> list[dict[str, str]]:
        """Format content as messages."""
        if isinstance(content, str):
            return [{"role": "user", "content": content}]
        elif isinstance(content, list):
            return content
        elif isinstance(content, dict):
            return [{"role": content.get("role", "user"), "content": str(content)}]
        else:
            return [{"role": "user", "content": str(content)}]
    
    async def health_check(self) -> bool:
        """Check if local server is healthy."""
        try:
            import httpx
            
            if self._server_type == "ollama":
                async with httpx.AsyncClient(timeout=5) as client:
                    response = await client.get("http://localhost:11434/api/tags")
                    return response.status_code == 200
            else:
                async with httpx.AsyncClient(timeout=5) as client:
                    response = await client.get(f"{self.base_url}/models")
                    return response.status_code == 200
        
        except Exception as e:
            logger.debug("local_server_health_check_failed", error=str(e))
            return False
    
    def supports_streaming(self) -> bool:
        """Local models typically support streaming."""
        return True
    
    def supports_batching(self) -> bool:
        """Local models may support batching."""
        return self._server_type in ("vllm", "lmstudio")
    
    def estimate_cost(self, input_tokens: int, output_tokens: int | None = None) -> float:
        """Local models have no API cost."""
        return 0.0
