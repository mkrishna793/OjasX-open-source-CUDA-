"""
OpenCL Memory Manager — Handles GPU buffer allocation, deallocation,
and CPU↔GPU data transfers with a caching buffer pool.

The buffer pool reuses freed buffers to avoid expensive clCreateBuffer calls.
"""

from __future__ import annotations

import threading
import weakref
from collections import defaultdict

import numpy as np
import pyopencl as cl

from .context import get_context, get_queue


class CLBuffer:
    """Wrapper around a pyopencl.Buffer with metadata."""

    __slots__ = ("buffer", "nbytes", "dtype", "shape", "_id", "__weakref__")

    _counter = 0
    _lock = threading.Lock()

    def __init__(
        self,
        buffer: cl.Buffer,
        nbytes: int,
        dtype: np.dtype | None = None,
        shape: tuple | None = None,
    ):
        self.buffer = buffer
        self.nbytes = nbytes
        self.dtype = dtype
        self.shape = shape
        with CLBuffer._lock:
            CLBuffer._counter += 1
            self._id = CLBuffer._counter

    def __repr__(self) -> str:
        return (
            f"CLBuffer(id={self._id}, nbytes={self.nbytes}, "
            f"dtype={self.dtype}, shape={self.shape})"
        )


class CLBufferPool:
    """Caching allocator that reuses OpenCL buffers of the same size.

    When a buffer is freed, it goes back into a per-size pool rather than
    being released to the driver.  The next allocation of the same size
    can then reuse it instantly.
    """

    def __init__(self) -> None:
        self._pool: dict[int, list[cl.Buffer]] = defaultdict(list)
        self._active = weakref.WeakValueDictionary()
        self._lock = threading.Lock()
        self._stats = {
            "alloc_count": 0,
            "reuse_count": 0,
            "free_count": 0,
            "total_bytes_allocated": 0,
        }

    # ── Allocation ───────────────────────────────────────────────

    def allocate(
        self,
        nbytes: int,
        dtype: np.dtype | None = None,
        shape: tuple | None = None,
    ) -> CLBuffer:
        """Allocate a GPU buffer, reusing from pool if possible."""
        if nbytes <= 0:
            nbytes = 4  # Minimum allocation (OpenCL doesn't allow 0)

        ctx = get_context()

        with self._lock:
            # Try to reuse an existing buffer of the same size
            if self._pool[nbytes]:
                raw_buf = self._pool[nbytes].pop()
                self._stats["reuse_count"] += 1
            else:
                raw_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, size=nbytes)
                self._stats["alloc_count"] += 1
                self._stats["total_bytes_allocated"] += nbytes

            cl_buf = CLBuffer(raw_buf, nbytes, dtype, shape)
            self._active[cl_buf._id] = cl_buf

        return cl_buf

    def free(self, cl_buf: CLBuffer) -> None:
        """Return a buffer to the pool for reuse."""
        with self._lock:
            self._active.pop(cl_buf._id, None)
            self._pool[cl_buf.nbytes].append(cl_buf.buffer)
            self._stats["free_count"] += 1

    def empty_cache(self) -> None:
        """Release all pooled (unused) buffers back to the driver."""
        with self._lock:
            self._pool.clear()

    # ── Data transfer ────────────────────────────────────────────

    def host_to_device(
        self,
        host_array: np.ndarray,
        cl_buf: CLBuffer | None = None,
    ) -> CLBuffer:
        """Copy a numpy array to GPU memory.

        If *cl_buf* is None, allocates a new buffer.
        """
        queue = get_queue()
        host_array = np.ascontiguousarray(host_array)
        nbytes = host_array.nbytes

        if cl_buf is None:
            cl_buf = self.allocate(nbytes, host_array.dtype, host_array.shape)

        cl.enqueue_copy(queue, cl_buf.buffer, host_array, is_blocking=True)
        return cl_buf

    def device_to_host(
        self,
        cl_buf: CLBuffer,
        dtype: np.dtype = np.float32,
        shape: tuple | None = None,
    ) -> np.ndarray:
        """Copy GPU buffer contents back to a numpy array."""
        queue = get_queue()
        if shape is not None:
            host_array = np.empty(shape, dtype=dtype)
        else:
            numel = cl_buf.nbytes // np.dtype(dtype).itemsize
            host_array = np.empty(numel, dtype=dtype)

        cl.enqueue_copy(queue, host_array, cl_buf.buffer)
        queue.finish()
        return host_array

    def device_to_device(
        self,
        src: CLBuffer,
        dst: CLBuffer | None = None,
    ) -> CLBuffer:
        """Copy one GPU buffer to another GPU buffer."""
        queue = get_queue()
        if dst is None:
            dst = self.allocate(src.nbytes, src.dtype, src.shape)

        cl.enqueue_copy(queue, dst.buffer, src.buffer)
        return dst

    def zero_fill(self, cl_buf: CLBuffer) -> None:
        """Fill a GPU buffer with zeros."""
        queue = get_queue()
        pattern = np.zeros(1, dtype=np.uint8)
        cl.enqueue_fill_buffer(queue, cl_buf.buffer, pattern, 0, cl_buf.nbytes)

    # ── Info ─────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return allocation statistics."""
        with self._lock:
            return dict(self._stats)

    def active_count(self) -> int:
        """Return number of currently active (not freed) buffers."""
        with self._lock:
            return len(self._active)


# ── Module-level singleton ───────────────────────────────────────────
_global_pool: CLBufferPool | None = None


def get_buffer_pool() -> CLBufferPool:
    """Return the global buffer pool, creating it if needed."""
    global _global_pool
    if _global_pool is None:
        _global_pool = CLBufferPool()
    return _global_pool
