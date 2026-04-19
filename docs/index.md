# Prism Documentation

Welcome to Prism documentation!

## Quick Links

- [README](../README.md) - Main documentation
- [API Reference](../src/) - Source code documentation

## Architecture

Prism is built with the following components:

```
src/
├── core/           # Core orchestrator and router
├── routing/        # Content classification and model selection
├── cache/          # Multi-backend caching
├── observability/ # Tracing and metrics
└── adapters/       # Model provider adapters
```

## Examples

See `examples/` directory for usage examples:

- `basic_usage.py` - Basic examples for all features
