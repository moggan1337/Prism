"""
Basic Usage Examples for Prism AI Router

This module demonstrates the core functionality of Prism
for various use cases.
"""

import asyncio
from prism import PrismOrchestrator
from prism.core.config import (
    TaskType, ContentType, PrismConfig, RoutingStrategy,
    CacheConfig, BudgetConfig, ObservabilityConfig
)


async def example_text_generation():
    """Example: Basic text generation."""
    print("=" * 50)
    print("Example 1: Text Generation")
    print("=" * 50)
    
    orchestrator = PrismOrchestrator()
    
    response = await orchestrator.process(
        content="Write a short poem about artificial intelligence",
        task_type=TaskType.TEXT_GENERATION
    )
    
    print(f"\nResult:\n{response.content}")
    print(f"\nModel: {response.model}")
    print(f"Provider: {response.provider}")
    print(f"Cost: ${response.cost:.6f}")
    print(f"Latency: {response.latency_ms:.2f}ms")
    print(f"Cache Hit: {response.cache_hit}")


async def example_text_classification():
    """Example: Text classification."""
    print("\n" + "=" * 50)
    print("Example 2: Text Classification")
    print("=" * 50)
    
    orchestrator = PrismOrchestrator()
    
    reviews = [
        "This product is amazing! Highly recommend it.",
        "Terrible experience. Would never buy again.",
        "It's okay, nothing special about it.",
        "Best purchase I've ever made!",
        "Complete waste of money."
    ]
    
    for review in reviews:
        response = await orchestrator.process(
            content=f"Classify this review as positive, negative, or neutral: {review}",
            task_type=TaskType.TEXT_CLASSIFICATION
        )
        print(f"Review: '{review[:30]}...' → {response.content[:20]}...")


async def example_cost_optimized():
    """Example: Cost-optimized routing."""
    print("\n" + "=" * 50)
    print("Example 3: Cost-Optimized Routing")
    print("=" * 50)
    
    config = PrismConfig(
        default_routing_strategy=RoutingStrategy.COST_OPTIMIZED,
        budget=BudgetConfig(monthly_limit=10.0)
    )
    
    orchestrator = PrismOrchestrator(config=config)
    
    response = await orchestrator.process(
        content="Summarize the key points of this text: Lorem ipsum dolor sit amet.",
        task_type=TaskType.TEXT_SUMMARIZATION
    )
    
    print(f"\nResult: {response.content[:100]}...")
    print(f"Model: {response.model} (selected for cost efficiency)")
    print(f"Cost: ${response.cost:.6f}")


async def example_accuracy_optimized():
    """Example: Accuracy-optimized routing."""
    print("\n" + "=" * 50)
    print("Example 4: Accuracy-Optimized Routing")
    print("=" * 50)
    
    config = PrismConfig(
        default_routing_strategy=RoutingStrategy.ACCURACY_OPTIMIZED
    )
    
    orchestrator = PrismOrchestrator(config=config)
    
    response = await orchestrator.process(
        content="Analyze this legal document and identify all clauses related to liability: " +
                "The parties agree to indemnify and hold harmless...",
        task_type=TaskType.TEXT_EXTRACTION
    )
    
    print(f"\nResult: {response.content[:100]}...")
    print(f"Model: {response.model} (selected for accuracy)")


async def example_with_caching():
    """Example: Request with caching enabled."""
    print("\n" + "=" * 50)
    print("Example 5: Request with Caching")
    print("=" * 50)
    
    config = PrismConfig(
        cache=CacheConfig(
            enabled=True,
            backend="memory",
            ttl_seconds=3600
        )
    )
    
    orchestrator = PrismOrchestrator(config=config)
    query = "What is the capital of France?"
    
    # First request - cache miss
    response1 = await orchestrator.process(
        content=query,
        task_type=TaskType.TEXT_QUESTION_ANSWERING
    )
    print(f"\nFirst Request:")
    print(f"  Result: {response1.content}")
    print(f"  Cache Hit: {response1.cache_hit}")
    print(f"  Cost: ${response1.cost:.6f}")
    
    # Second request - should be cache hit
    response2 = await orchestrator.process(
        content=query,
        task_type=TaskType.TEXT_QUESTION_ANSWERING
    )
    print(f"\nSecond Request:")
    print(f"  Result: {response2.content}")
    print(f"  Cache Hit: {response2.cache_hit}")
    print(f"  Cost: ${response2.cost:.6f}")


async def example_batch_processing():
    """Example: Batch processing multiple requests."""
    print("\n" + "=" * 50)
    print("Example 6: Batch Processing")
    print("=" * 50)
    
    orchestrator = PrismOrchestrator()
    
    requests = [
        {"content": f"Translate to French: Hello world {i}", "task_type": TaskType.TEXT_TRANSLATION}
        for i in range(5)
    ]
    
    print("\nProcessing 5 translation requests in parallel...")
    
    responses = await orchestrator.process_batch(
        requests=requests,
        parallel=True,
        max_concurrent=3
    )
    
    for i, response in enumerate(responses):
        print(f"\n  Request {i+1}:")
        print(f"    Result: {response.content}")
        print(f"    Model: {response.model}")


async def example_stats():
    """Example: Getting usage statistics."""
    print("\n" + "=" * 50)
    print("Example 7: Usage Statistics")
    print("=" * 50)
    
    orchestrator = PrismOrchestrator()
    
    # Make some requests
    for _ in range(3):
        await orchestrator.process(
            content="Hello!",
            task_type=TaskType.TEXT_GENERATION
        )
    
    stats = orchestrator.get_stats()
    
    print("\nUsage Statistics:")
    print(f"  Total Requests: {stats['usage']['total_requests']}")
    print(f"  Successful: {stats['usage']['successful_requests']}")
    print(f"  Failed: {stats['usage']['failed_requests']}")
    print(f"  Total Cost: ${stats['usage']['total_cost']:.6f}")
    print(f"  Cache Hit Rate: {stats['usage']['cache_hit_rate']:.2%}")
    print(f"  Avg Latency: {stats['usage']['avg_latency_ms']:.2f}ms")


async def example_health_check():
    """Example: Health check."""
    print("\n" + "=" * 50)
    print("Example 8: Health Check")
    print("=" * 50)
    
    orchestrator = PrismOrchestrator()
    
    health = await orchestrator.health_check()
    
    print(f"\nSystem Status: {health['status'].upper()}")
    print("\nComponents:")
    for component, status in health['components'].items():
        status_str = status.get('status', 'unknown')
        print(f"  {component}: {status_str.upper()}")


async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("  Prism AI Router - Usage Examples")
    print("=" * 60)
    
    await example_text_generation()
    await example_text_classification()
    await example_cost_optimized()
    await example_accuracy_optimized()
    await example_with_caching()
    await example_batch_processing()
    await example_stats()
    await example_health_check()
    
    print("\n" + "=" * 60)
    print("  All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
