"""Tests for Prism AI Router."""

import pytest
from prism.core.config import PrismConfig, ContentType, TaskType, RoutingStrategy
from prism.core.router import Router, RouteRequest, UsageStats
from prism.routing.classifier import ContentClassifier
from prism.routing.model_selector import ModelSelector, SelectionCriteria
from prism.cache.manager import CacheManager, MemoryCache
from prism.observability.metrics import MetricsCollector
from prism.core.orchestrator import PrismOrchestrator


class TestContentClassifier:
    """Tests for ContentClassifier."""
    
    def test_classify_text_string(self):
        """Test classifying text strings."""
        classifier = ContentClassifier()
        
        assert classifier.classify("Hello, world!") == ContentType.TEXT
        assert classifier.classify("Some text content") == ContentType.TEXT
        assert classifier.classify("") == ContentType.TEXT
    
    def test_classify_json(self):
        """Test classifying JSON content."""
        classifier = ContentClassifier()
        
        json_str = '{"key": "value", "number": 42}'
        assert classifier.classify(json_str) == ContentType.TEXT
    
    def test_classify_dict(self):
        """Test classifying dictionary content."""
        classifier = ContentClassifier()
        
        assert classifier.classify({"key": "value"}) == ContentType.TEXT
        assert classifier.classify([1, 2, 3]) == ContentType.TEXT
    
    def test_classify_with_confidence(self):
        """Test classification with confidence."""
        classifier = ContentClassifier()
        
        content_type, confidence = classifier.classify_with_confidence("Hello")
        assert content_type == ContentType.TEXT
        assert 0 <= confidence <= 1.0
    
    def test_classify_code(self):
        """Test detecting code content."""
        classifier = ContentClassifier()
        
        code = '''
        def hello():
            print("Hello, world!")
            return 42
        '''
        assert classifier.classify(code) == ContentType.TEXT
    
    def test_classify_binary_image(self):
        """Test classifying binary image data."""
        classifier = ContentClassifier()
        
        # PNG magic bytes
        png_data = b'\x89PNG\r\n\x1a\n' + b'\x00' * 100
        assert classifier.classify(png_data) == ContentType.IMAGE
        
        # JPEG magic bytes
        jpeg_data = b'\xff\xd8\xff\xe0' + b'\x00' * 100
        assert classifier.classify(jpeg_data) == ContentType.IMAGE


class TestRouter:
    """Tests for Router."""
    
    def test_router_initialization(self):
        """Test router initialization."""
        config = PrismConfig()
        router = Router(config)
        
        assert router is not None
        assert len(router.models) > 0
    
    def test_route_text_request(self):
        """Test routing a text request."""
        config = PrismConfig()
        router = Router(config)
        
        request = RouteRequest(
            content="Hello, world!",
            task_type=TaskType.TEXT_GENERATION
        )
        
        result = router.route(request)
        
        assert result is not None
        assert result.selected_model is not None
        assert result.provider is not None
        assert result.estimated_cost >= 0
    
    def test_route_with_cost_constraint(self):
        """Test routing with cost constraint."""
        config = PrismConfig()
        router = Router(config)
        
        request = RouteRequest(
            content="Hello",
            task_type=TaskType.TEXT_GENERATION,
            max_cost=0.001  # Very low cost
        )
        
        result = router.route(request)
        
        assert result.estimated_cost <= 0.001
    
    def test_route_with_latency_constraint(self):
        """Test routing with latency constraint."""
        config = PrismConfig()
        router = Router(config)
        
        request = RouteRequest(
            content="Hello",
            task_type=TaskType.TEXT_GENERATION,
            max_latency_ms=500
        )
        
        result = router.route(request)
        
        assert result.estimated_latency_ms <= 500
    
    def test_route_cache_key(self):
        """Test cache key generation."""
        request1 = RouteRequest(content="Hello")
        request2 = RouteRequest(content="Hello")
        request3 = RouteRequest(content="World")
        
        assert request1.get_cache_key() == request2.get_cache_key()
        assert request1.get_cache_key() != request3.get_cache_key()
    
    def test_usage_stats(self):
        """Test usage statistics tracking."""
        config = PrismConfig()
        router = Router(config)
        
        stats = router.get_usage_stats()
        
        assert isinstance(stats, UsageStats)
        assert stats.total_requests == 0


class TestModelSelector:
    """Tests for ModelSelector."""
    
    def test_selector_initialization(self):
        """Test model selector initialization."""
        selector = ModelSelector()
        
        assert selector is not None
    
    def test_select_with_criteria(self):
        """Test model selection with criteria."""
        selector = ModelSelector()
        
        criteria = SelectionCriteria(
            max_cost=0.01,
            min_accuracy=0.9
        )
        
        available = selector.get_available_models(
            task_type=TaskType.TEXT_GENERATION,
            content_type=ContentType.TEXT
        )
        
        assert len(available) > 0
    
    def test_get_available_models(self):
        """Test getting available models."""
        selector = ModelSelector()
        
        models = selector.get_available_models()
        
        assert len(models) > 0
    
    def test_create_criteria_from_task(self):
        """Test creating criteria from task type."""
        selector = ModelSelector()
        
        criteria = selector.create_criteria_from_task(TaskType.TEXT_GENERATION)
        
        assert criteria.max_latency_ms is not None
        assert criteria.min_accuracy is not None


class TestMemoryCache:
    """Tests for MemoryCache."""
    
    @pytest.mark.asyncio
    async def test_cache_set_and_get(self):
        """Test setting and getting cache values."""
        cache = MemoryCache()
        
        await cache.set("key1", {"data": "value1"}, ttl=3600)
        value = await cache.get("key1")
        
        assert value == {"data": "value1"}
    
    @pytest.mark.asyncio
    async def test_cache_miss(self):
        """Test cache miss."""
        cache = MemoryCache()
        
        value = await cache.get("nonexistent")
        
        assert value is None
    
    @pytest.mark.asyncio
    async def test_cache_expiration(self):
        """Test cache expiration."""
        import asyncio
        cache = MemoryCache()
        
        await cache.set("key1", "value1", ttl=1)
        
        # Should exist immediately
        value = await cache.get("key1")
        assert value == "value1"
        
        # Wait for expiration
        await asyncio.sleep(1.1)
        
        # Should be expired
        value = await cache.get("key1")
        assert value is None
    
    @pytest.mark.asyncio
    async def test_cache_delete(self):
        """Test deleting cache values."""
        cache = MemoryCache()
        
        await cache.set("key1", "value1", ttl=3600)
        deleted = await cache.delete("key1")
        
        assert deleted is True
        
        value = await cache.get("key1")
        assert value is None
    
    @pytest.mark.asyncio
    async def test_cache_stats(self):
        """Test cache statistics."""
        cache = MemoryCache()
        
        await cache.set("key1", "value1", ttl=3600)
        await cache.get("key1")  # Hit
        await cache.get("key2")  # Miss
        
        stats = cache.get_stats()
        
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["entries"] == 1


class TestMetricsCollector:
    """Tests for MetricsCollector."""
    
    def test_counter_increment(self):
        """Test counter increment."""
        collector = MetricsCollector(PrismConfig().observability)
        
        collector.increment_counter("test_counter", value=5)
        value = collector.get_counter_value("test_counter")
        
        assert value == 5
    
    def test_gauge_set(self):
        """Test gauge set."""
        collector = MetricsCollector(PrismConfig().observability)
        
        collector.set_gauge("test_gauge", value=42)
        value = collector.get_gauge_value("test_gauge")
        
        assert value == 42
    
    def test_histogram_record(self):
        """Test histogram recording."""
        collector = MetricsCollector(PrismConfig().observability)
        
        collector.record_histogram("test_histogram", 100)
        collector.record_histogram("test_histogram", 200)
        
        stats = collector.get_histogram_stats("test_histogram")
        
        assert stats["count"] == 2
        assert stats["sum"] == 300
    
    def test_export_prometheus(self):
        """Test Prometheus export."""
        collector = MetricsCollector(PrismConfig().observability)
        
        collector.increment_counter("test_counter")
        
        output = collector.export_prometheus()
        
        assert "test_counter" in output
        assert "# HELP" in output
        assert "# TYPE" in output


class TestPrismOrchestrator:
    """Tests for PrismOrchestrator."""
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        orchestrator = PrismOrchestrator()
        
        assert orchestrator is not None
        assert orchestrator.router is not None
        assert orchestrator.cache is not None
    
    @pytest.mark.asyncio
    async def test_process_text_request(self):
        """Test processing a text request."""
        orchestrator = PrismOrchestrator()
        
        response = await orchestrator.process(
            content="Hello, world!",
            task_type=TaskType.TEXT_GENERATION
        )
        
        assert response is not None
        assert response.request_id is not None
    
    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Test getting statistics."""
        orchestrator = PrismOrchestrator()
        
        stats = orchestrator.get_stats()
        
        assert "usage" in stats
        assert "cache" in stats
        assert "models" in stats
    
    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health check."""
        orchestrator = PrismOrchestrator()
        
        health = await orchestrator.health_check()
        
        assert "status" in health
        assert "components" in health


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
