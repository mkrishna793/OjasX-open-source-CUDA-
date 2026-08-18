"""
OpenCL Memory Manager — High-performance Slab Allocator & Unified Memory Pool.
Handles GPU buffer allocation, deallocation, and zero-copy host-pinned transfers.
"""

from __future__ import annotations

import math
import threading
import weakref
from collections import defaultdict
from typing import Any

import numpy as np
import pyopencl as cl

from .context import get_context, get_queue


def _next_power_of_two(n: int) -> int:
    """Return smallest power of two >= n, minimum 64 bytes."""
    if n <= 64:
        return 64
    return 1 << (n - 1).bit_length()


class CLBuffer:
    """Wrapper around a pyopencl.Buffer with slab metadata."""

    __slots__ = ("buffer", "nbytes", "capacity", "dtype", "shape", "_id", "is_pinned", "__weakref__")

    _counter = 0
    _lock = threading.Lock()

    def __init__(
        self,
        buffer: cl.Buffer,
        nbytes: int,
        capacity: int,
        dtype: np.dtype | None = None,
        shape: tuple | None = None,
        is_pinned: bool = False,
    ):
        self.buffer = buffer
        self.nbytes = nbytes
        self.capacity = capacity
        self.dtype = dtype
        self.shape = shape
        self.is_pinned = is_pinned
        with CLBuffer._lock:
            CLBuffer._counter += 1
            self._id = CLBuffer._counter

    def __repr__(self) -> str:
        return (
            f"CLBuffer(id={self._id}, nbytes={self.nbytes}, capacity={self.capacity}, "
            f"dtype={self.dtype}, shape={self.shape}, pinned={self.is_pinned})"
        )


class CLBufferPool:
    """High-performance Slab & Zero-Copy Allocator for OpenCL/Vulkan/Unified Memory.

    Features:
      1. Power-of-two slab binning (eliminates driver allocation churn).
      2. Zero-copy host-pinned memory allocations for integrated/unified GPUs.
      3. Real-time VRAM telemetry (active, cached, peak memory tracking).
      4. Auto-eviction under memory pressure.
    """

    def __init__(self) -> None:
        self._slabs: dict[int, list[cl.Buffer]] = defaultdict(list)
        self._active: dict[int, CLBuffer] = weakref.WeakValueDictionary()
        self._lock = threading.Lock()
        self._stats = {
            "alloc_count": 0,
            "reuse_count": 0,
            "free_count": 0,
            "active_bytes": 0,
            "cached_bytes": 0,
            "peak_bytes": 0,
            "total_allocated_bytes": 0,
        }

    def allocate(
        self,
        nbytes: int,
        dtype: np.dtype | None = None,
        shape: tuple | None = None,
        pin_memory: bool = False,
    ) -> CLBuffer:
        """Allocate a GPU buffer using slab binning."""
        if nbytes <= 0:
            nbytes = 64

        slab_size = _next_power_of_two(nbytes)
        ctx = get_context()

        with self._lock:
            # Check slab pool for available buffer
            if self._slabs[slab_size]:
                raw_buf = self._slabs[slab_size].pop()
                self._stats["reuse_count"] += 1
                self._stats["cached_bytes"] -= slab_size
            else:
                flags = cl.mem_flags.READ_WRITE
                if pin_memory:
                    flags |= cl.mem_flags.ALLOC_HOST_PTR
                try:
                    raw_buf = cl.Buffer(ctx, flags, size=slab_size)
                except cl.MemoryError:
                    # Clear cache and retry
                    self._slabs.clear()
                    self._stats["cached_bytes"] = 0
                    raw_buf = cl.Buffer(ctx, flags, size=slab_size)

                self._stats["alloc_count"] += 1
                self._stats["total_allocated_bytes"] += slab_size

            self._stats["active_bytes"] += slab_size
            if self._stats["active_bytes"] > self._stats["peak_bytes"]:
                self._stats["peak_bytes"] = self._stats["active_bytes"]

            cl_buf = CLBuffer(raw_buf, nbytes, slab_size, dtype, shape, is_pinned=pin_memory)
            self._active[cl_buf._id] = cl_buf

        return cl_buf

    def free(self, cl_buf: CLBuffer) -> None:
        """Return a buffer to its slab pool."""
        with self._lock:
            self._active.pop(cl_buf._id, None)
            slab_size = cl_buf.capacity
            self._slabs[slab_size].append(cl_buf.buffer)
            self._stats["free_count"] += 1
            self._stats["active_bytes"] = max(0, self._stats["active_bytes"] - slab_size)
            self._stats["cached_bytes"] += slab_size

    def empty_cache(self) -> None:
        """Release all pooled cached buffers."""
        with self._lock:
            self._slabs.clear()
            self._stats["cached_bytes"] = 0

    def host_to_device(
        self,
        host_array: np.ndarray,
        cl_buf: CLBuffer | None = None,
        non_blocking: bool = False,
    ) -> CLBuffer:
        """Copy numpy host array to GPU buffer."""
        queue = get_queue()
        host_array = np.ascontiguousarray(host_array)
        nbytes = host_array.nbytes

        if cl_buf is None:
            cl_buf = self.allocate(nbytes, host_array.dtype, host_array.shape)

        cl.enqueue_copy(queue, cl_buf.buffer, host_array, is_blocking=not non_blocking)
        return cl_buf

    def device_to_host(
        self,
        cl_buf: CLBuffer,
        dtype: np.dtype = np.float32,
        shape: tuple | None = None,
    ) -> np.ndarray:
        """Copy GPU buffer contents back to CPU numpy array."""
        queue = get_queue()
        if shape is not None:
            host_array = np.empty(shape, dtype=dtype)
        else:
            numel = cl_buf.nbytes // np.dtype(dtype).itemsize
            host_array = np.empty(numel, dtype=dtype)

        cl.enqueue_copy(queue, host_array, cl_buf.buffer, is_blocking=True)
        return host_array

    def device_to_device(
        self,
        src: CLBuffer,
        dst: CLBuffer | None = None,
    ) -> CLBuffer:
        """Copy GPU buffer to another GPU buffer."""
        queue = get_queue()
        if dst is None:
            dst = self.allocate(src.nbytes, src.dtype, src.shape)

        cl.enqueue_copy(queue, dst.buffer, src.buffer, is_blocking=False)
        return dst

    def zero_fill(self, cl_buf: CLBuffer) -> None:
        """Fill buffer with zeros."""
        queue = get_queue()
        pattern = np.zeros(1, dtype=np.uint8)
        cl.enqueue_fill_buffer(queue, cl_buf.buffer, pattern, 0, cl_buf.nbytes)

    def get_stats(self) -> dict[str, Any]:
        """Return memory statistics."""
        with self._lock:
            stats = dict(self._stats)
            stats["active_buffers"] = len(self._active)
            return stats

    def active_count(self) -> int:
        with self._lock:
            return len(self._active)


_global_pool: CLBufferPool | None = None


def get_buffer_pool() -> CLBufferPool:
    global _global_pool
    if _global_pool is None:
        _global_pool = CLBufferPool()
    return _global_pool
