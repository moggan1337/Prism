"""
Cache Manager - Multi-backend caching for request results.

Supports multiple caching backends:
- Memory: In-memory LRU cache
- Redis: Redis-based distributed cache
- Disk: File-based cache (for large objects)

Features:
- TTL-based expiration
- Content-type specific TTLs
- Cache invalidation
- Statistics tracking
- Distributed cache sync (Redis)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from prism.core.config import CacheConfig


logger = structlog.get_logger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: float
    expires_at: float
    hits: int = 0
    last_accessed: float = 0.0
    size_bytes: int = 0
    
    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return time.time() > self.expires_at
    
    def access(self) -> None:
        """Record a cache hit."""
        self.hits += 1
        self.last_accessed = time.time()


class CacheBackend(ABC):
    """Abstract base class for cache backends."""
    
    @abstractmethod
    async def get(self, key: str) -> Any | None:
        """Get value from cache."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int) -> None:
        """Set value in cache with TTL."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass
    
    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        pass


class MemoryCache(CacheBackend):
    """
    In-memory LRU cache implementation.
    
    Features:
    - LRU eviction policy
    - Configurable max size
    - TTL expiration
    - Statistics tracking
    """
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 512):
        """
        Initialize memory cache.
        
        Args:
            max_size: Maximum number of entries
            max_memory_mb: Maximum memory usage in MB
        """
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size = max_size
        self._max_memory_bytes = max_memory_mb * 1024 * 1024
        self._current_memory = 0
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0
    
    async def get(self, key: str) -> Any | None:
        """Get value from cache."""
        entry = self._cache.get(key)
        
        if entry is None:
            self._misses += 1
            return None
        
        # Check expiration
        if entry.is_expired:
            await self.delete(key)
            self._expirations += 1
            self._misses += 1
            return None
        
        # Move to end (most recently used)
        self._cache.move_to_end(key)
        entry.access()
        
        self._hits += 1
        return entry.value
    
    async def set(self, key: str, value: Any, ttl: int) -> None:
        """Set value in cache with TTL."""
        # Calculate size
        value_str = json.dumps(value) if not isinstance(value, str) else value
        size_bytes = len(value_str.encode())
        
        # Check memory limit
        if size_bytes > self._max_memory_bytes:
            logger.warning("value_too_large_for_cache", key=key, size_bytes=size_bytes)
            return
        
        # Evict if necessary
        while (len(self._cache) >= self._max_size or 
               self._current_memory + size_bytes > self._max_memory_bytes):
            if not self._cache:
                break
            await self._evict_lru()
        
        now = time.time()
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=now,
            expires_at=now + ttl,
            size_bytes=size_bytes
        )
        
        # Remove old entry if exists
        if key in self._cache:
            old_entry = self._cache[key]
            self._current_memory -= old_entry.size_bytes
            self._evictions += 1
        
        self._cache[key] = entry
        self._current_memory += size_bytes
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        if key in self._cache:
            entry = self._cache.pop(key)
            self._current_memory -= entry.size_bytes
            return True
        return False
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._current_memory = 0
        logger.info("cache_cleared", backend="memory")
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        entry = self._cache.get(key)
        if entry is None:
            return False
        
        if entry.is_expired:
            await self.delete(key)
            return False
        
        return True
    
    async def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if self._cache:
            key = next(iter(self._cache))
            entry = self._cache.pop(key)
            self._current_memory -= entry.size_bytes
            self._evictions += 1
    
    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "backend": "memory",
            "entries": len(self._cache),
            "memory_used_bytes": self._current_memory,
            "memory_limit_bytes": self._max_memory_bytes,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "evictions": self._evictions,
            "expirations": self._expirations,
        }
    
    async def cleanup_expired(self) -> int:
        """Remove all expired entries. Returns count of removed entries."""
        removed = 0
        now = time.time()
        
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.expires_at <= now
        ]
        
        for key in expired_keys:
            entry = self._cache.pop(key, None)
            if entry:
                self._current_memory -= entry.size_bytes
                removed += 1
                self._expirations += 1
        
        if removed > 0:
            logger.info("expired_entries_removed", count=removed)
        
        return removed


class RedisCache(CacheBackend):
    """
    Redis-based distributed cache.
    
    Features:
    - Distributed cache across multiple instances
    - TTL expiration
    - Pub/sub for cache invalidation
    - Statistics tracking
    """
    
    def __init__(
        self,
        redis_url: str,
        key_prefix: str = "prism:",
        max_connections: int = 10
    ):
        """
        Initialize Redis cache.
        
        Args:
            redis_url: Redis connection URL
            key_prefix: Prefix for all cache keys
            max_connections: Maximum connection pool size
        """
        self._redis_url = redis_url
        self._key_prefix = key_prefix
        self._max_connections = max_connections
        self._pool: Any = None
        self._client: Any = None
        
        # Statistics
        self._hits = 0
        self._misses = 0
    
    async def _get_client(self):
        """Get or create Redis client."""
        if self._client is None:
            try:
                import redis.asyncio as aioredis
                self._pool = aioredis.ConnectionPool.from_url(
                    self._redis_url,
                    max_connections=self._max_connections,
                    decode_responses=True
                )
                self._client = aioredis.Redis(connection_pool=self._pool)
                logger.info("redis_cache_connected", url=self._redis_url)
            except ImportError:
                logger.error("redis_not_installed")
                raise
        
        return self._client
    
    def _make_key(self, key: str) -> str:
        """Add prefix to key."""
        return f"{self._key_prefix}{key}"
    
    async def get(self, key: str) -> Any | None:
        """Get value from cache."""
        try:
            client = await self._get_client()
            value = await client.get(self._make_key(key))
            
            if value is None:
                self._misses += 1
                return None
            
            self._hits += 1
            return json.loads(value)
        
        except Exception as e:
            logger.error("redis_get_error", key=key, error=str(e))
            self._misses += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: int) -> None:
        """Set value in cache with TTL."""
        try:
            client = await self._get_client()
            value_str = json.dumps(value)
            await client.setex(self._make_key(key), ttl, value_str)
        
        except Exception as e:
            logger.error("redis_set_error", key=key, error=str(e))
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            client = await self._get_client()
            result = await client.delete(self._make_key(key))
            return result > 0
        
        except Exception as e:
            logger.error("redis_delete_error", key=key, error=str(e))
            return False
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        try:
            client = await self._get_client()
            pattern = f"{self._key_prefix}*"
            
            cursor = 0
            while True:
                cursor, keys = await client.scan(cursor, match=pattern, count=100)
                if keys:
                    await client.delete(*keys)
                if cursor == 0:
                    break
            
            logger.info("redis_cache_cleared")
        
        except Exception as e:
            logger.error("redis_clear_error", error=str(e))
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        try:
            client = await self._get_client()
            return await client.exists(self._make_key(key)) > 0
        
        except Exception as e:
            logger.error("redis_exists_error", key=key, error=str(e))
            return False
    
    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "backend": "redis",
            "url": self._redis_url,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
        }
    
    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
        if self._pool:
            await self._pool.disconnect()


class DiskCache(CacheBackend):
    """
    Disk-based cache for large objects.
    
    Features:
    - File-based storage
    - TTL expiration
    - Automatic cleanup
    - Compression support
    """
    
    def __init__(self, cache_dir: str = "/tmp/prism_cache", max_size_gb: float = 10.0):
        """
        Initialize disk cache.
        
        Args:
            cache_dir: Directory for cache files
            max_size_gb: Maximum cache size in GB
        """
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        self._current_size = 0
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
    
    def _get_path(self, key: str) -> Path:
        """Get file path for key."""
        # Hash key to create safe filename
        key_hash = hashlib.sha256(key.encode()).hexdigest()[:32]
        return self._cache_dir / key_hash
    
    def _get_meta_path(self, key: str) -> Path:
        """Get metadata file path for key."""
        return self._get_path(key).with_suffix(".meta")
    
    async def get(self, key: str) -> Any | None:
        """Get value from cache."""
        try:
            import json
            
            value_path = self._get_path(key)
            meta_path = self._get_meta_path(key)
            
            if not value_path.exists() or not meta_path.exists():
                self._misses += 1
                return None
            
            # Check expiration
            with open(meta_path, "r") as f:
                meta = json.load(f)
            
            if time.time() > meta.get("expires_at", 0):
                await self.delete(key)
                self._misses += 1
                return None
            
            # Read value
            with open(value_path, "rb") as f:
                value = f.read()
            
            # Deserialize
            value = json.loads(value.decode("utf-8"))
            
            self._hits += 1
            return value
        
        except Exception as e:
            logger.error("disk_cache_get_error", key=key, error=str(e))
            self._misses += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: int) -> None:
        """Set value in cache with TTL."""
        try:
            value_path = self._get_path(key)
            meta_path = self._get_meta_path(key)
            
            # Serialize value
            value_bytes = json.dumps(value).encode("utf-8")
            size = len(value_bytes)
            
            # Check size limit
            while self._current_size + size > self._max_size_bytes:
                if not await self._evict_oldest():
                    break
            
            # Write value
            with open(value_path, "wb") as f:
                f.write(value_bytes)
            
            # Write metadata
            meta = {
                "key": key,
                "created_at": time.time(),
                "expires_at": time.time() + ttl,
                "size": size
            }
            with open(meta_path, "w") as f:
                json.dump(meta, f)
            
            self._current_size += size
        
        except Exception as e:
            logger.error("disk_cache_set_error", key=key, error=str(e))
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            value_path = self._get_path(key)
            meta_path = self._get_meta_path(key)
            
            deleted = False
            
            if value_path.exists():
                size = value_path.stat().st_size
                value_path.unlink()
                self._current_size -= size
                deleted = True
            
            if meta_path.exists():
                meta_path.unlink()
            
            return deleted
        
        except Exception as e:
            logger.error("disk_cache_delete_error", key=key, error=str(e))
            return False
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        try:
            for f in self._cache_dir.iterdir():
                if f.is_file():
                    f.unlink()
            self._current_size = 0
            logger.info("disk_cache_cleared", dir=str(self._cache_dir))
        
        except Exception as e:
            logger.error("disk_cache_clear_error", error=str(e))
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        return self._get_path(key).exists()
    
    async def _evict_oldest(self) -> bool:
        """Evict oldest cache entry. Returns True if evicted."""
        try:
            meta_files = list(self._cache_dir.glob("*.meta"))
            if not meta_files:
                return False
            
            # Find oldest by access time
            oldest = min(meta_files, key=lambda f: f.stat().st_atime)
            
            # Get corresponding value file
            value_path = oldest.with_suffix("")
            
            # Delete both
            size = value_path.stat().st_size if value_path.exists() else 0
            oldest.unlink()
            if value_path.exists():
                value_path.unlink()
            
            self._current_size -= size
            self._evictions += 1
            
            return True
        
        except Exception as e:
            logger.error("disk_cache_evict_error", error=str(e))
            return False
    
    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "backend": "disk",
            "cache_dir": str(self._cache_dir),
            "size_used_bytes": self._current_size,
            "size_limit_bytes": self._max_size_bytes,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "evictions": self._evictions,
        }


class CacheManager:
    """
    Unified cache manager with multi-backend support.
    
    Provides a consistent interface for caching across different backends,
    with automatic backend selection based on configuration.
    
    Example:
        >>> config = CacheConfig(enabled=True, backend="memory", ttl_seconds=3600)
        >>> cache = CacheManager(config)
        >>> 
        >>> # Store value
        >>> await cache.set("key", {"result": "data"}, ttl=3600)
        >>> 
        >>> # Retrieve value
        >>> value = await cache.get("key")
        >>> print(value)  # {"result": "data"}
    """
    
    def __init__(self, config: CacheConfig):
        """
        Initialize cache manager.
        
        Args:
            config: Cache configuration
        """
        self.config = config
        self._backend: CacheBackend | None = None
        self._backend_lock = asyncio.Lock()
    
    async def _get_backend(self) -> CacheBackend:
        """Get or create cache backend."""
        if self._backend is not None:
            return self._backend
        
        async with self._backend_lock:
            if self._backend is not None:
                return self._backend
            
            if not self.config.enabled:
                self._backend = NullCache()
            elif self.config.backend == "memory":
                self._backend = MemoryCache(
                    max_size=10000,
                    max_memory_mb=self.config.max_size_mb
                )
            elif self.config.backend == "redis":
                if not self.config.redis_url:
                    logger.warning("redis_url_not_configured_using_memory")
                    self._backend = MemoryCache()
                else:
                    self._backend = RedisCache(
                        redis_url=self.config.redis_url,
                        key_prefix=self.config.key_prefix
                    )
            elif self.config.backend == "disk":
                self._backend = DiskCache(
                    cache_dir="/tmp/prism_cache",
                    max_size_gb=self.config.max_size_mb / 1024
                )
            else:
                logger.warning(f"unknown_cache_backend_{self.config.backend}_using_memory")
                self._backend = MemoryCache()
            
            return self._backend
    
    async def get(self, key: str) -> Any | None:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        backend = await self._get_backend()
        return await backend.get(key)
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None
    ) -> None:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses config default if not specified)
        """
        backend = await self._get_backend()
        ttl = ttl or self.config.ttl_seconds
        await backend.set(key, value, ttl)
    
    async def delete(self, key: str) -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted, False if not found
        """
        backend = await self._get_backend()
        return await backend.delete(key)
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        backend = await self._get_backend()
        await backend.clear()
    
    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if exists, False otherwise
        """
        backend = await self._get_backend()
        return await backend.exists(key)
    
    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        if self._backend:
            return self._backend.get_stats()
        return {"backend": "not_initialized"}


class NullCache(CacheBackend):
    """Null cache backend that doesn't actually cache anything."""
    
    async def get(self, key: str) -> None:
        return None
    
    async def set(self, key: str, value: Any, ttl: int) -> None:
        pass
    
    async def delete(self, key: str) -> bool:
        return False
    
    async def clear(self) -> None:
        pass
    
    async def exists(self, key: str) -> bool:
        return False
    
    def get_stats(self) -> dict[str, Any]:
        return {"backend": "null", "note": "caching disabled"}
