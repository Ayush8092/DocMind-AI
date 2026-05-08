"""
Simple LRU disk-based cache for embeddings and query results.
Falls back to in-memory dict when disk writes fail.
"""

import hashlib
import json
import logging
import os
import pickle
import time
from collections import OrderedDict
from typing import Any, Optional

from config import config

logger = logging.getLogger(__name__)

os.makedirs(config.CACHE_DIR, exist_ok=True)


# =============================================================================
# IN-MEMORY LRU CACHE
# =============================================================================

class LRUCache:
    """Thread-unsafe but fast in-memory LRU cache."""

    def __init__(self, max_size: int = 256):
        self._cache: OrderedDict = OrderedDict()
        self._max_size = max_size

    def get(self, key: str) -> Optional[Any]:
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def set(self, key: str, value: Any):
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def clear(self):
        self._cache.clear()

    def __len__(self):
        return len(self._cache)


# =============================================================================
# DISK CACHE (pickle-based)
# =============================================================================

class DiskCache:
    """Key-value store backed by individual pickle files on disk."""

    def __init__(self, directory: str = config.CACHE_DIR, ttl_seconds: int = 86400):
        self._dir = directory
        self._ttl = ttl_seconds
        os.makedirs(self._dir, exist_ok=True)

    def _path(self, key: str) -> str:
        hashed = hashlib.sha256(key.encode()).hexdigest()
        return os.path.join(self._dir, f"{hashed}.pkl")

    def get(self, key: str) -> Optional[Any]:
        path = self._path(key)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "rb") as fh:
                entry = pickle.load(fh)
            if time.time() - entry["ts"] > self._ttl:
                os.remove(path)
                return None
            return entry["value"]
        except Exception as e:
            logger.warning(f"DiskCache read error for key {key!r}: {e}")
            return None

    def set(self, key: str, value: Any):
        path = self._path(key)
        try:
            with open(path, "wb") as fh:
                pickle.dump({"ts": time.time(), "value": value}, fh)
        except Exception as e:
            logger.warning(f"DiskCache write error for key {key!r}: {e}")

    def delete(self, key: str):
        path = self._path(key)
        if os.path.exists(path):
            os.remove(path)

    def clear(self):
        for fname in os.listdir(self._dir):
            if fname.endswith(".pkl"):
                try:
                    os.remove(os.path.join(self._dir, fname))
                except Exception:
                    pass


# =============================================================================
# TWO-LAYER CACHE
# =============================================================================

class TwoLayerCache:
    """
    Checks in-memory LRU first, then falls through to disk.
    On a disk hit the value is promoted to LRU.
    """

    def __init__(self, lru_size: int = 128):
        self._lru = LRUCache(max_size=lru_size)
        self._disk = DiskCache()

    def get(self, key: str) -> Optional[Any]:
        value = self._lru.get(key)
        if value is not None:
            return value
        value = self._disk.get(key)
        if value is not None:
            self._lru.set(key, value)
        return value

    def set(self, key: str, value: Any):
        self._lru.set(key, value)
        self._disk.set(key, value)

    def clear(self):
        self._lru.clear()
        self._disk.clear()


# Singleton
_cache_instance: Optional[TwoLayerCache] = None


def get_cache() -> TwoLayerCache:
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = TwoLayerCache()
    return _cache_instance


def make_cache_key(*parts) -> str:
    """Create a deterministic cache key from arbitrary string parts."""
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode()).hexdigest()