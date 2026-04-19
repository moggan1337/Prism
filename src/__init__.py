"""
Prism - Multi-Modal AI Router & Orchestrator
=============================================

A comprehensive framework for intelligent routing of AI requests across
multiple modalities including text, image, audio, video, and documents.

Architecture Overview:
    - Content Classification: Automatically detect input type
    - Model Selection: Choose optimal model based on cost/latency/accuracy
    - Request Batching: Aggregate requests for efficiency
    - Caching: Redis-based caching with TTL support
    - Fallback & Retry: Automatic retry with exponential backoff
    - Multi-Model Aggregation: Combine outputs from multiple models
    - Budget Enforcement: Track and limit spending
    - Observability: Distributed tracing and metrics
"""

__version__ = "0.1.0"
__author__ = "moggan1337"

from prism.core.orchestrator import PrismOrchestrator
from prism.core.router import Router
from prism.routing.classifier import ContentClassifier
from prism.routing.model_selector import ModelSelector

__all__ = [
    "PrismOrchestrator",
    "Router", 
    "ContentClassifier",
    "ModelSelector",
]
