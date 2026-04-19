# Prism - Multi-Modal AI Router & Orchestrator

<p align="center">
  <img src="docs/prism-logo.png" alt="Prism Logo" width="200"/>
</p>

<p align="center">
  <strong>Intelligent routing for text, image, audio, video, and document AI processing</strong>
</p>

<p align="center">
  <a href="https://github.com/moggan1337/Prism/actions">
    <img src="https://img.shields.io/github/actions/workflow/status/moggan1337/Prism/test.yml?style=flat-square" alt="CI">
  </a>
  <a href="https://pypi.org/project/prism-ai-router/">
    <img src="https://img.shields.io/pypi/v/prism-ai-router?style=flat-square" alt="PyPI">
  </a>
  <a href="https://pypi.org/project/prism-ai-router/">
    <img src="https://img.shields.io/pypi/pyversions/prism-ai-router?style=flat-square" alt="Python">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" alt="License">
  </a>
</p>

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Core Concepts](#core-concepts)
- [Usage Examples](#usage-examples)
- [Routing Logic](#routing-logic)
- [Caching](#caching)
- [Observability](#observability)
- [API Reference](#api-reference)
- [Supported Models](#supported-models)
- [Performance](#performance)
- [Contributing](#contributing)
- [License](#license)

---

## Features

Prism is a comprehensive multi-modal AI routing framework that provides:

### 🚀 Intelligent Routing
- **Content Classification**: Automatically detect input type (text, image, audio, video, document)
- **Model Selection**: Choose optimal model based on cost, latency, and accuracy trade-offs
- **Multi-Criteria Optimization**: Balance competing priorities with configurable weights
- **Custom Strategies**: Implement your own routing logic with custom scorers

### 💾 Request Caching
- **Multi-Backend Support**: Memory, Redis, and disk caching
- **TTL Management**: Content-type specific expiration policies
- **LRU Eviction**: Automatic cleanup of least recently used entries
- **Distributed Cache**: Redis-based caching for multi-instance deployments

### 🔄 Reliability
- **Automatic Retries**: Configurable retry with exponential backoff
- **Circuit Breakers**: Prevent cascading failures with automatic model disabling
- **Fallback Chains**: Define fallback models for graceful degradation
- **Rate Limiting**: Built-in rate limit awareness

### 📊 Observability
- **Distributed Tracing**: OpenTelemetry-compatible span tracking
- **Metrics Collection**: Prometheus-compatible counters, gauges, histograms
- **Request Tracking**: Full request lifecycle visibility
- **Cost Tracking**: Monitor and budget AI spending

### 🔗 Multi-Model Aggregation
- **Weighted Averaging**: Combine responses based on accuracy/cost
- **Majority Voting**: Ensemble decisions from multiple models
- **Hierarchical Selection**: Use fastest as base, validate with others

### 💰 Budget Enforcement
- **Monthly/Daily Limits**: Control AI spending at multiple granularities
- **Per-Request Limits**: Prevent single requests from breaking the bank
- **Automatic Fallback**: Switch to cheaper models when budget is tight

---

## Architecture

Prism follows a modular architecture designed for scalability and extensibility:

```
┌─────────────────────────────────────────────────────────────────┐
│                        PrismOrchestrator                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │    Router   │  │    Cache    │  │ Observability│            │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ ContentClassifier│ │  ModelSelector  │ │   RouteRequest  │
└─────────────────┘ └─────────────────┘ └─────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Model Adapters                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   OpenAI    │  │  Anthropic  │  │    Local    │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. PrismOrchestrator
The main entry point that coordinates all components:
- Routes incoming requests through the routing pipeline
- Manages caching and retries
- Collects metrics and traces
- Enforces budgets

#### 2. Router
Handles intelligent request routing:
- Classifies content type automatically
- Selects optimal model based on criteria
- Manages circuit breakers
- Tracks usage statistics

#### 3. ContentClassifier
Automatically detects content types:
- MIME type detection for files
- Magic bytes analysis for binary content
- Heuristic analysis for text
- Base64 detection and decoding

#### 4. ModelSelector
Selects optimal models using multi-criteria optimization:
- Cost efficiency scoring
- Latency scoring
- Accuracy scoring
- Capability matching
- Provider preferences

#### 5. CacheManager
Multi-backend caching system:
- Memory cache (LRU)
- Redis cache (distributed)
- Disk cache (large objects)
- TTL-based expiration

#### 6. Observability
Distributed tracing and metrics:
- OpenTelemetry integration
- Prometheus-compatible metrics
- Request tracing
- Cost tracking

---

## Quick Start

### Basic Usage

```python
import asyncio
from prism import PrismOrchestrator
from prism.core.config import TaskType

async def main():
    # Initialize orchestrator
    orchestrator = PrismOrchestrator()
    
    # Process a text request
    response = await orchestrator.process(
        content="Hello, world!",
        task_type=TaskType.TEXT_GENERATION
    )
    
    print(f"Result: {response.content}")
    print(f"Model: {response.model}")
    print(f"Cost: ${response.cost:.4f}")

asyncio.run(main())
```

### With Configuration

```python
from prism import PrismOrchestrator
from prism.core.config import PrismConfig, RoutingStrategy

# Create custom configuration
config = PrismConfig(
    default_routing_strategy=RoutingStrategy.COST_OPTIMIZED,
    cache=CacheConfig(enabled=True, ttl_seconds=3600),
    budget=BudgetConfig(monthly_limit=100.0)
)

# Initialize with config
orchestrator = PrismOrchestrator(config=config)

# Process request
response = await orchestrator.process(
    content="Explain quantum computing",
    task_type=TaskType.TEXT_GENERATION
)
```

---

## Installation

### Using pip

```bash
pip install prism-ai-router
```

### From Source

```bash
git clone https://github.com/moggan1337/Prism.git
cd Prism
pip install -e .
```

### Development Installation

```bash
pip install -e ".[dev]"
```

### Dependencies

Prism requires Python 3.10+ and the following dependencies:

- `aiohttp>=3.9.0` - Async HTTP client
- `redis>=5.0.0` - Redis client (optional, for distributed caching)
- `pydantic>=2.5.0` - Data validation
- `structlog>=24.0.0` - Structured logging
- `tenacity>=8.2.0` - Retry logic
- `prometheus-client>=0.19.0` - Prometheus metrics
- `opentelemetry-api>=1.21.0` - OpenTelemetry API
- `httpx>=0.26.0` - HTTP client

---

## Configuration

### Environment Variables

Prism can be configured via environment variables:

```bash
# API Keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Base URLs (for proxies/custom endpoints)
export OPENAI_BASE_URL="https://api.openai.com/v1"
export ANTHROPIC_BASE_URL="https://api.anthropic.com"

# Redis (for distributed caching)
export REDIS_URL="redis://localhost:6379/0"

# OpenTelemetry
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"
```

### Configuration File

Create a `prism.yaml` configuration file:

```yaml
# Prism Configuration

# Routing settings
routing:
  strategy: balanced  # cost_optimized, latency_optimized, accuracy_optimized, balanced

# Model configurations
models:
  gpt-4-turbo:
    enabled: true
    cost_per_1k_input_tokens: 0.01
    cost_per_1k_output_tokens: 0.03
    avg_latency_ms: 2000
    accuracy_score: 0.95

# Cache settings
cache:
  enabled: true
  backend: memory  # memory, redis, disk
  ttl_seconds: 3600
  text_ttl_seconds: 3600
  image_ttl_seconds: 86400

# Budget settings
budget:
  enabled: true
  monthly_limit: 1000.0
  daily_limit: 100.0
  auto_fallback_to_cheaper: true

# Observability
observability:
  tracing_enabled: true
  metrics_enabled: true
  log_level: INFO
  prometheus_port: 9090

# Retry settings
retry:
  max_attempts: 3
  initial_delay_ms: 100
  max_delay_ms: 10000
  backoff_factor: 2.0
```

---

## Core Concepts

### Content Types

Prism supports the following content types:

| Type | Description | Examples |
|------|-------------|----------|
| `TEXT` | Text content | Plain text, code, JSON, XML |
| `IMAGE` | Image data | PNG, JPEG, GIF, WebP |
| `AUDIO` | Audio data | MP3, WAV, OGG, FLAC |
| `VIDEO` | Video data | MP4, AVI, MKV, WebM |
| `DOCUMENT` | Document files | PDF, DOC, XLS, PPT |

### Task Types

Prism supports various task types for model selection:

**Text Tasks:**
- `TEXT_GENERATION` - Text completion and generation
- `TEXT_CLASSIFICATION` - Classification tasks
- `TEXT_EXTRACTION` - Information extraction
- `TEXT_SUMMARIZATION` - Summarization
- `TEXT_TRANSLATION` - Translation
- `TEXT_QUESTION_ANSWERING` - Question answering

**Image Tasks:**
- `IMAGE_CLASSIFICATION` - Image classification
- `IMAGE_GENERATION` - Image generation (DALL-E, etc.)
- `IMAGE_EDITING` - Image editing
- `OBJECT_DETECTION` - Object detection
- `OCR` - Optical character recognition

**Audio Tasks:**
- `SPEECH_TO_TEXT` - Speech recognition
- `TEXT_TO_SPEECH` - Text to speech
- `AUDIO_CLASSIFICATION` - Audio classification

**Video Tasks:**
- `VIDEO_CLASSIFICATION` - Video classification
- `VIDEO_GENERATION` - Video generation

**Document Tasks:**
- `DOCUMENT_PARSING` - Document parsing
- `DOCUMENT_QA` - Document question answering
- `KEY_VALUE_EXTRACTION` - Key-value extraction

**Multi-Modal Tasks:**
- `MULTIMODAL_QA` - Multi-modal question answering
- `VISION_LANGUAGE` - Vision-language tasks

### Routing Strategies

Prism supports multiple routing strategies:

1. **COST_OPTIMIZED** - Prioritizes cost efficiency (70% cost, 20% latency, 10% accuracy)
2. **LATENCY_OPTIMIZED** - Prioritizes speed (10% cost, 70% latency, 20% accuracy)
3. **ACCURACY_OPTIMIZED** - Prioritizes quality (10% cost, 20% latency, 70% accuracy)
4. **BALANCED** - Equal weight to all factors (33% each)

---

## Usage Examples

### Example 1: Text Generation

```python
from prism import PrismOrchestrator
from prism.core.config import TaskType

orchestrator = PrismOrchestrator()

# Simple text generation
response = await orchestrator.process(
    content="Write a haiku about artificial intelligence",
    task_type=TaskType.TEXT_GENERATION
)

print(f"Haiku:\n{response.content}")
print(f"Model used: {response.model}")
print(f"Cost: ${response.cost:.4f}")
```

### Example 2: Image Classification

```python
from prism import PrismOrchestrator
from prism.core.config import TaskType, ContentType

orchestrator = PrismOrchestrator()

# Read image
with open("image.jpg", "rb") as f:
    image_data = f.read()

# Classify image
response = await orchestrator.process(
    content=image_data,
    content_type=ContentType.IMAGE,
    task_type=TaskType.IMAGE_CLASSIFICATION
)

print(f"Classification: {response.content}")
print(f"Model: {response.model}")
```

### Example 3: Speech to Text

```python
from prism import PrismOrchestrator
from prism.core.config import TaskType

orchestrator = PrismOrchestrator()

# Transcribe audio
with open("audio.mp3", "rb") as f:
    audio_data = f.read()

response = await orchestrator.process(
    content=audio_data,
    task_type=TaskType.SPEECH_TO_TEXT
)

print(f"Transcription: {response.content}")
```

### Example 4: Batch Processing

```python
from prism import PrismOrchestrator
from prism.core.config import TaskType

orchestrator = PrismOrchestrator()

# Prepare batch requests
requests = [
    {"content": f"Review #{i}", "task_type": TaskType.TEXT_CLASSIFICATION}
    for i in range(10)
]

# Process in parallel
responses = await orchestrator.process_batch(
    requests=requests,
    parallel=True,
    max_concurrent=5
)

for i, response in enumerate(responses):
    print(f"Review {i}: {response.content}")
```

### Example 5: Multi-Model Aggregation

```python
from prism import PrismOrchestrator
from prism.core.config import TaskType

orchestrator = PrismOrchestrator(
    config=PrismConfig(
        aggregation=AggregationConfig(
            enabled=True,
            models_per_request=2,
            aggregation_strategy="weighted_average"
        )
    )
)

# Aggregate responses from multiple models
response = await orchestrator.aggregate(
    content="What is the capital of France?",
    task_type=TaskType.TEXT_QUESTION_ANSWERING,
    models=["gpt-4", "claude-3-sonnet"]
)

print(f"Aggregated answer: {response.content}")
print(f"Models used: {response.alternatives_used}")
```

### Example 6: Custom Model Selection

```python
from prism import PrismOrchestrator
from prism.core.config import ModelConfig, TaskType

# Add custom model
custom_model = ModelConfig(
    name="my-custom-model",
    provider=ModelProvider.LOCAL,
    task_types=[TaskType.TEXT_GENERATION],
    content_types=[ContentType.TEXT],
    cost_per_1k_input_tokens=0.0,
    cost_per_1k_output_tokens=0.0,
    avg_latency_ms=100,
    accuracy_score=0.75
)

orchestrator = PrismOrchestrator()
orchestrator.models["my-custom-model"] = custom_model

# Force use of custom model
response = await orchestrator.process(
    content="Hello",
    task_type=TaskType.TEXT_GENERATION,
    options={"force_model": "my-custom-model"}
)
```

---

## Routing Logic

### How Routing Works

1. **Content Classification**
   ```
   Input Content → ContentClassifier → ContentType
   ```

2. **Model Filtering**
   ```
   ContentType + TaskType → Filter eligible models
   ```

3. **Scoring**
   ```
   Eligible Models + Criteria → Score each model
   ```

4. **Selection**
   ```
   Scored Models → Select best + alternatives
   ```

### Scoring Algorithm

```python
def calculate_score(model, criteria):
    # Cost score (inverse - lower cost = higher score)
    cost_score = 1 - (model.cost / max_cost)
    
    # Latency score (inverse - lower latency = higher score)
    latency_score = 1 - (model.latency / max_latency)
    
    # Accuracy score
    accuracy_score = model.accuracy
    
    # Weighted sum
    total = (
        cost_score * weights["cost"] +
        latency_score * weights["latency"] +
        accuracy_score * weights["accuracy"]
    )
    
    # Apply circuit breaker penalty
    if model.circuit_breaker_failing:
        total *= 0.5
    
    return total
```

### Circuit Breaker

Prism implements a circuit breaker pattern to prevent cascading failures:

```
CLOSED → (5 failures) → OPEN → (30s timeout) → HALF_OPEN → (success) → CLOSED
                                                        ↓
                                                     (failure)
                                                        ↓
                                                        OPEN
```

---

## Caching

### Cache Backends

#### Memory Cache
- In-memory LRU cache
- Fast, single-instance
- Configurable max size

```python
config = PrismConfig(
    cache=CacheConfig(
        backend="memory",
        ttl_seconds=3600,
        max_size_mb=512
    )
)
```

#### Redis Cache
- Distributed cache
- Shared across instances
- TTL-based expiration

```python
config = PrismConfig(
    cache=CacheConfig(
        backend="redis",
        redis_url="redis://localhost:6379/0",
        ttl_seconds=3600
    )
)
```

#### Disk Cache
- File-based cache
- Good for large objects
- Automatic cleanup

```python
config = PrismConfig(
    cache=CacheConfig(
        backend="disk",
        ttl_seconds=604800  # 1 week
    )
)
```

### Cache Key Generation

Cache keys are generated from content hash + metadata:

```python
def get_cache_key(request):
    content_hash = hash(content)[:16]
    metadata_hash = hash(task_type + user_id + metadata)[:8]
    return f"{content_hash}:{metadata_hash}"
```

---

## Observability

### Metrics

Prism exports Prometheus-compatible metrics:

```
# Counters
prism_requests_total{content_type, task_type, model}
prism_requests_failed_total{content_type, error_type}
prism_cache_hits_total{content_type}
prism_cache_misses_total{content_type}
prism_cost_total{model, content_type}

# Histograms
prism_request_duration_seconds{content_type, task_type, model}
prism_model_latency_ms{model, provider}
prism_routing_score{content_type}

# Gauges
prism_model_requests_in_flight{model}
```

### Tracing

Prism supports OpenTelemetry tracing:

```python
config = PrismConfig(
    observability=ObservabilityConfig(
        tracing_enabled=True,
        otlp_endpoint="http://localhost:4317",
        service_name="prism-router"
    )
)
```

### Prometheus Endpoint

Access metrics at `http://localhost:9090/metrics`:

```bash
# Start Prometheus scraping
prometheus -config.file=prism.yaml

# Query metrics
curl http://localhost:9090/metrics | grep prism_
```

---

## API Reference

### PrismOrchestrator

```python
class PrismOrchestrator:
    def __init__(
        self,
        config: PrismConfig | None = None,
        adapters: dict[str, ModelAdapter] | None = None,
        enable_observability: bool = True
    )
    
    async def process(
        self,
        content: Any,
        content_type: ContentType | None = None,
        task_type: TaskType | None = None,
        user_id: str | None = None,
        api_key: str | None = None,
        options: dict[str, Any] | None = None
    ) -> OrchestratedResponse
    
    async def process_batch(
        self,
        requests: list[dict[str, Any]],
        parallel: bool = False,
        max_concurrent: int = 5
    ) -> list[OrchestratedResponse]
    
    async def aggregate(
        self,
        content: Any,
        task_type: TaskType | None = None,
        models: list[str] | None = None
    ) -> OrchestratedResponse
    
    def get_stats(self) -> dict[str, Any]
    
    async def health_check(self) -> dict[str, Any]
```

### Router

```python
class Router:
    def __init__(
        self,
        config: PrismConfig,
        models: dict[str, ModelConfig] | None = None,
        custom_scorer: Callable[[ModelConfig, RouteRequest], float] | None = None
    )
    
    def route(self, request: RouteRequest) -> RouteResult
    
    def record_success(self, model_name: str, cost: float, latency_ms: float)
    
    def record_failure(self, model_name: str)
    
    def can_afford(self, cost: float) -> bool
    
    def get_usage_stats(self) -> UsageStats
```

### ContentClassifier

```python
class ContentClassifier:
    def classify(self, content: Any) -> ContentType
    
    def classify_with_confidence(self, content: Any) -> tuple[ContentType, float]
    
    def get_supported_types(self) -> list[ContentType]
```

### ModelSelector

```python
class ModelSelector:
    def select(
        self,
        candidates: list[str | ModelConfig],
        criteria: SelectionCriteria
    ) -> ModelScore | None
    
    def select_top_n(
        self,
        candidates: list[str | ModelConfig],
        criteria: SelectionCriteria,
        n: int = 3
    ) -> list[ModelScore]
    
    def get_available_models(
        self,
        task_type: TaskType | None = None,
        content_type: ContentType | None = None,
        criteria: SelectionCriteria | None = None
    ) -> list[ModelConfig]
```

---

## Supported Models

### Text Models

| Model | Provider | Context | Cost (1K in/out) | Latency | Accuracy |
|-------|----------|---------|------------------|---------|----------|
| GPT-4 Turbo | OpenAI | 128K | $0.01/$0.03 | ~2s | 95% |
| GPT-3.5 Turbo | OpenAI | 16K | $0.0005/$0.0015 | ~500ms | 88% |
| Claude 3 Opus | Anthropic | 200K | $0.015/$0.075 | ~3s | 97% |
| Claude 3 Sonnet | Anthropic | 200K | $0.003/$0.015 | ~1.5s | 94% |
| Gemini Pro | Google | 32K | $0.000125/$0.000375 | ~1s | 92% |
| Mistral Large | Mistral | 128K | $0.008/$0.024 | ~1.5s | 93% |

### Vision Models

| Model | Provider | Cost | Latency | Accuracy |
|-------|----------|------|---------|----------|
| GPT-4 Vision | OpenAI | $0.01/$0.03 | ~2.5s | 94% |
| DALL-E 3 | OpenAI | $0.04/image | ~10s | 90% |

### Audio Models

| Model | Provider | Cost | Latency | Accuracy |
|-------|----------|------|---------|----------|
| Whisper | OpenAI | $0.006/min | ~30s | 92% |
| Eleven Multilingual | ElevenLabs | $0.00018/char | ~5s | 95% |

---

## Performance

### Benchmarks

| Operation | Latency (p50) | Latency (p95) | Throughput |
|-----------|--------------|---------------|------------|
| Text classification | 150ms | 400ms | 100 req/s |
| Text generation | 800ms | 2000ms | 20 req/s |
| Image classification | 300ms | 800ms | 30 req/s |
| Speech-to-text | 2000ms | 5000ms | 5 req/s |

### Optimization Tips

1. **Enable Caching**: Reduces redundant API calls
2. **Use Batching**: Process multiple requests together
3. **Select Cheaper Models**: For non-critical tasks
4. **Monitor Metrics**: Track and optimize bottlenecks

---

## Contributing

Contributions are welcome! Please read our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/`
5. Submit a pull request

### Development Setup

```bash
git clone https://github.com/moggan1337/Prism.git
cd Prism
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>Built with ❤️ by moggan1337</strong>
</p>
