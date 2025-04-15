"""
Caching utilities for the RAG system.
Adapted for Google Colab environment.
"""

from typing import Dict, Any, List, Optional, Union, Callable, TypeVar, Generic
import os
import json
import pickle
import time
import hashlib
from pathlib import Path
from collections import OrderedDict

from .debug_utils import debug_logger, debug_function

T = TypeVar('T')  # Generic type for cache values

class LRUCache(Generic[T]):
    """LRU (Least Recently Used) cache implementation."""
    
    def __init__(self, capacity: int = 1000):
        """
        Initialize the LRU cache.
        
        Args:
            capacity: Maximum number of items to store
        """
        self.cache = OrderedDict()
        self.capacity = capacity
    
    def get(self, key: str) -> Optional[T]:
        """
        Get an item from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        if key not in self.cache:
            return None
        
        # Move the key to the end to mark it as recently used
        value = self.cache.pop(key)
        self.cache[key] = value
        
        return value
    
    def put(self, key: str, value: T) -> None:
        """
        Put an item in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        if key in self.cache:
            # Remove the key to update its position
            self.cache.pop(key)
        elif len(self.cache) >= self.capacity:
            # Remove the least recently used item (first item)
            self.cache.popitem(last=False)
        
        # Add the key-value pair
        self.cache[key] = value
    
    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
    
    def __len__(self) -> int:
        """Get the number of items in the cache."""
        return len(self.cache)
    
    def __contains__(self, key: str) -> bool:
        """Check if a key is in the cache."""
        return key in self.cache


class EmbeddingCache:
    """Cache for embeddings with disk persistence."""
    
    def __init__(self, capacity: int = 10000, cache_dir: Optional[str] = "cache/embeddings", 
                ttl: Optional[float] = None, debug: bool = True):
        """
        Initialize the embedding cache.
        
        Args:
            capacity: Maximum number of items to store in memory
            cache_dir: Directory to store persistent cache
            ttl: Time-to-live in seconds (None for no expiration)
            debug: Whether to enable debugging
        """
        self.memory_cache = LRUCache[Dict[str, Any]](capacity)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.ttl = ttl
        self.debug = debug
        
        # Create cache directory if it doesn't exist
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            
            if self.debug:
                debug_logger.logger.info(f"Initialized EmbeddingCache with disk cache at {self.cache_dir}")
        elif self.debug:
            debug_logger.logger.info("Initialized EmbeddingCache with memory cache only")
    
    def _get_cache_key(self, text: str) -> str:
        """
        Generate a cache key for a text.
        
        Args:
            text: Text to generate key for
            
        Returns:
            Cache key
        """
        # Use MD5 hash of the text as the cache key
        return hashlib.md5(text.encode()).hexdigest()
    
    def _get_cache_path(self, key: str) -> Optional[Path]:
        """
        Get the cache file path for a key.
        
        Args:
            key: Cache key
            
        Returns:
            Cache file path or None if disk cache is disabled
        """
        if not self.cache_dir:
            return None
        
        # Use the first two characters of the key as a subdirectory
        # to avoid having too many files in a single directory
        subdir = self.cache_dir / key[:2]
        os.makedirs(subdir, exist_ok=True)
        
        return subdir / f"{key}.json"
    
    @debug_function()
    def get(self, text: str) -> Optional[List[float]]:
        """
        Get an embedding from the cache.
        
        Args:
            text: Text to get embedding for
            
        Returns:
            Cached embedding or None if not found
        """
        key = self._get_cache_key(text)
        
        # Check memory cache first
        cached_item = self.memory_cache.get(key)
        
        if cached_item:
            # Check if the item has expired
            if self.ttl and time.time() - cached_item["timestamp"] > self.ttl:
                if self.debug:
                    debug_logger.logger.debug(f"Cache item expired for key: {key[:8]}...")
                return None
            
            if self.debug:
                debug_logger.logger.debug(f"Memory cache hit for key: {key[:8]}...")
            
            return cached_item["embedding"]
        
        # If not in memory cache, check disk cache
        cache_path = self._get_cache_path(key)
        
        if cache_path and cache_path.exists():
            try:
                with open(cache_path, "r") as f:
                    cached_item = json.load(f)
                
                # Check if the item has expired
                if self.ttl and time.time() - cached_item["timestamp"] > self.ttl:
                    if self.debug:
                        debug_logger.logger.debug(f"Disk cache item expired for key: {key[:8]}...")
                    return None
                
                # Add to memory cache
                self.memory_cache.put(key, cached_item)
                
                if self.debug:
                    debug_logger.logger.debug(f"Disk cache hit for key: {key[:8]}...")
                
                return cached_item["embedding"]
            except Exception as e:
                if self.debug:
                    debug_logger.logger.warning(f"Error loading from disk cache: {str(e)}")
        
        return None
    
    @debug_function()
    def put(self, text: str, embedding: List[float]) -> None:
        """
        Put an embedding in the cache.
        
        Args:
            text: Text to cache embedding for
            embedding: Embedding to cache
        """
        key = self._get_cache_key(text)
        
        # Create cache item
        cache_item = {
            "embedding": embedding,
            "timestamp": time.time(),
            "text_hash": key
        }
        
        # Add to memory cache
        self.memory_cache.put(key, cache_item)
        
        # Add to disk cache if enabled
        cache_path = self._get_cache_path(key)
        
        if cache_path:
            try:
                with open(cache_path, "w") as f:
                    json.dump(cache_item, f)
                
                if self.debug:
                    debug_logger.logger.debug(f"Saved embedding to disk cache: {key[:8]}...")
            except Exception as e:
                if self.debug:
                    debug_logger.logger.warning(f"Error saving to disk cache: {str(e)}")
    
    def clear(self) -> None:
        """Clear the cache."""
        # Clear memory cache
        self.memory_cache.clear()
        
        # Clear disk cache if enabled
        if self.cache_dir and self.cache_dir.exists():
            try:
                import shutil
                shutil.rmtree(self.cache_dir)
                os.makedirs(self.cache_dir, exist_ok=True)
                
                if self.debug:
                    debug_logger.logger.info(f"Cleared disk cache at {self.cache_dir}")
            except Exception as e:
                if self.debug:
                    debug_logger.logger.warning(f"Error clearing disk cache: {str(e)}")


class QueryCache:
    """Cache for query results."""
    
    def __init__(self, capacity: int = 100, ttl: Optional[float] = None, debug: bool = True):
        """
        Initialize the query cache.
        
        Args:
            capacity: Maximum number of items to store
            ttl: Time-to-live in seconds (None for no expiration)
            debug: Whether to enable debugging
        """
        self.cache = LRUCache[Dict[str, Any]](capacity)
        self.ttl = ttl
        self.debug = debug
        
        if self.debug:
            debug_logger.logger.info(f"Initialized QueryCache with capacity: {capacity}")
    
    def _get_cache_key(self, query: str, **kwargs) -> str:
        """
        Generate a cache key for a query.
        
        Args:
            query: Query string
            **kwargs: Additional query parameters
            
        Returns:
            Cache key
        """
        # Create a string representation of the kwargs
        kwargs_str = json.dumps(kwargs, sort_keys=True)
        
        # Combine query and kwargs
        key_str = f"{query}|{kwargs_str}"
        
        # Use MD5 hash as the cache key
        return hashlib.md5(key_str.encode()).hexdigest()
    
    @debug_function()
    def get(self, query: str, **kwargs) -> Optional[List[Dict[str, Any]]]:
        """
        Get query results from the cache.
        
        Args:
            query: Query string
            **kwargs: Additional query parameters
            
        Returns:
            Cached query results or None if not found
        """
        key = self._get_cache_key(query, **kwargs)
        
        # Check cache
        cached_item = self.cache.get(key)
        
        if cached_item:
            # Check if the item has expired
            if self.ttl and time.time() - cached_item["timestamp"] > self.ttl:
                if self.debug:
                    debug_logger.logger.debug(f"Cache item expired for query: {query[:50]}...")
                return None
            
            if self.debug:
                debug_logger.logger.debug(f"Cache hit for query: {query[:50]}...")
            
            return cached_item["results"]
        
        return None
    
    @debug_function()
    def put(self, query: str, results: List[Dict[str, Any]], **kwargs) -> None:
        """
        Put query results in the cache.
        
        Args:
            query: Query string
            results: Query results to cache
            **kwargs: Additional query parameters
        """
        key = self._get_cache_key(query, **kwargs)
        
        # Create cache item
        cache_item = {
            "results": results,
            "timestamp": time.time(),
            "query": query,
            "params": kwargs
        }
        
        # Add to cache
        self.cache.put(key, cache_item)
        
        if self.debug:
            debug_logger.logger.debug(f"Cached results for query: {query[:50]}...")
    
    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        
        if self.debug:
            debug_logger.logger.info("Cleared query cache")
    
    def __len__(self) -> int:
        """Get the number of items in the cache."""
        return len(self.cache)
