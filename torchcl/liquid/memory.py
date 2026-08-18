"""
Continuous Memory (CMEM) — Ring-buffer allocator and dynamic-shape LiquidTensor.
"""

from __future__ import annotations
import numpy as np
import pyopencl as cl

from torchcl.runtime.context import get_context, get_queue
from torchcl.runtime.memory import CLBuffer, get_buffer_pool


class LiquidMemoryPool:
    """Memory pool specializing in large continuous allocations for time-varying tensors."""

    def __init__(self, size_mb: int = 128) -> None:
        self.size_bytes = size_mb * 1024 * 1024
        self.ctx = get_context()
        self.raw_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, size=self.size_bytes)
        self.offset = 0
        self.active_ranges: dict[int, tuple[int, int]] = {}

    def allocate(self, nbytes: int, dtype: np.dtype | None = None, shape: tuple | None = None) -> CLBuffer:
        """Contiguous allocation with active range overlap collision checking."""
        aligned_offset = ((self.offset + 127) // 128) * 128
        if aligned_offset + nbytes > self.size_bytes:
            aligned_offset = 0

        end_offset = aligned_offset + nbytes
        
        # Verify no overlap with active ranges
        for start, end in self.active_ranges.values():
            if max(aligned_offset, start) < min(end_offset, end):
                raise MemoryError("LiquidMemoryPool OutOfMemory: collision with active buffer ranges.")

        sub_raw = self.raw_buf.get_sub_region(aligned_offset, nbytes, cl.mem_flags.READ_WRITE)
        cl_buf = CLBuffer(sub_raw, nbytes, nbytes, dtype, shape)
        
        self.active_ranges[cl_buf._id] = (aligned_offset, end_offset)
        self.offset = end_offset

        return cl_buf

    def free(self, cl_buf: CLBuffer) -> None:
        """Release allocated block range."""
        self.active_ranges.pop(cl_buf._id, None)


class LiquidTensor:
    """A tensor with a dynamic shape that can grow or shrink along an axis without re-allocation."""

    def __init__(self, max_shape: tuple, dtype=np.float32) -> None:
        self.max_shape = max_shape
        self.dtype = np.dtype(dtype)
        self.itemsize = self.dtype.itemsize
        self.max_numel = int(np.prod(max_shape))
        self.nbytes = self.max_numel * self.itemsize

        # Allocate from global buffer pool
        self.cl_buf = get_buffer_pool().allocate(self.nbytes, self.dtype, max_shape)
        
        # Initially empty along the dynamic axis (axis 0)
        self.current_shape = list(max_shape)
        self.current_shape[0] = 0
        self.write_ptr_elements = 0  # Number of elements written along dynamic axis

    @property
    def shape(self) -> tuple:
        return tuple(self.current_shape)

    @property
    def numel(self) -> int:
        return int(np.prod(self.current_shape))

    def append(self, data: CLBuffer, axis: int = 0):
        """Append a new chunk of data along the dynamic axis without reallocating."""
        if axis != 0:
            raise NotImplementedError("Dynamic appending is currently only supported along axis 0.")

        queue = get_queue()
        chunk_numel = data.nbytes // self.itemsize
        slice_shape = data.shape or (chunk_numel,)
        
        # Verify shape compatibility along other dimensions
        if len(self.max_shape) > 1:
            if list(self.max_shape[1:]) != list(slice_shape[1:]):
                raise ValueError(f"Shape mismatch: cannot append {slice_shape} to max shape {self.max_shape}")

        chunk_size_axis0 = slice_shape[0] if len(slice_shape) > 0 else chunk_numel
        
        # Calculate byte offset
        offset_bytes = self.write_ptr_elements * self.itemsize
        
        # Check if we exceed max buffer capacity (wrap around as ring buffer)
        if offset_bytes + data.nbytes > self.nbytes:
            # Ring buffer wrap around
            self.write_ptr_elements = 0
            offset_bytes = 0

        # Copy data into our pre-allocated continuous buffer
        cl.enqueue_copy(
            queue,
            self.cl_buf.buffer,
            data.buffer,
            src_offset=0,
            dest_offset=offset_bytes,
            byte_count=data.nbytes
        )
        
        self.write_ptr_elements += chunk_numel
        
        # Update current shape (cap to max_shape)
        new_size_axis0 = min(self.max_shape[0], self.current_shape[0] + chunk_size_axis0)
        self.current_shape[0] = new_size_axis0

    def view(self, start_idx: int, end_idx: int, axis: int = 0) -> CLBuffer:
        """Return a zero-copy sub-buffer view of a slice along the dynamic axis."""
        if axis != 0:
            raise NotImplementedError("Slicing views are currently only supported along axis 0.")

        # Compute elements per slice along axis 0
        elements_per_slice = 1
        if len(self.max_shape) > 1:
            elements_per_slice = int(np.prod(self.max_shape[1:]))

        start_element = start_idx * elements_per_slice
        end_element = end_idx * elements_per_slice
        num_elements = end_element - start_element

        offset_bytes = start_element * self.itemsize
        size_bytes = num_elements * self.itemsize

        if offset_bytes + size_bytes > self.nbytes:
            raise IndexError("Slice view out of bounds of physical buffer.")

        sub_raw = self.cl_buf.buffer.get_sub_region(
            offset_bytes, size_bytes, cl.mem_flags.READ_WRITE
        )
        
        view_shape = list(self.max_shape)
        view_shape[0] = end_idx - start_idx
        
        return CLBuffer(sub_raw, size_bytes, size_bytes, self.dtype, tuple(view_shape))

    def release(self):
        """Free the underlying buffer."""
        get_buffer_pool().free(self.cl_buf)
