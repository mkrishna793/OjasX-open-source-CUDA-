"""
OpenCL HAL Backend — Implements the HALBackend interface for OpenCL 3.0 GPUs.
Consolidates device init, memory, and kernel execution into the unified HAL contract.
Delegates to existing runtime modules for backward compatibility.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
import pyopencl as cl

from torchcl.hal.interface import (
    HALBackend,
    BufferHandle,
    DeviceInfo,
    KernelHandle,
)


def _next_power_of_two(n: int) -> int:
    if n <= 64:
        return 64
    return 1 << (n - 1).bit_length()


class OpenCLHAL(HALBackend):
    """Production OpenCL 3.0 implementation of the HAL interface."""

    def __init__(self, device_index: int = 0) -> None:
        self._platform: cl.Platform | None = None
        self._device: cl.Device | None = None
        self._context: cl.Context | None = None
        self._queue: cl.CommandQueue | None = None
        self._kernel_cache: dict[str, cl.Kernel] = {}
        self._program_cache: dict[str, cl.Program] = {}
        self._buf_counter = 0
        self._lock = threading.Lock()
        self.device_info = self.init(device_index)

    # ── 1. Initialization ────────────────────────────────────────

    def init(self, device_index: int = 0) -> DeviceInfo:
        platforms = cl.get_platforms()
        if not platforms:
            raise RuntimeError("HAL: No OpenCL platforms found.")

        self._platform = platforms[0]

        try:
            devices = self._platform.get_devices(device_type=cl.device_type.GPU)
        except cl.RuntimeError:
            devices = []
        if not devices:
            devices = self._platform.get_devices(device_type=cl.device_type.ALL)
        if not devices:
            raise RuntimeError(f"HAL: No OpenCL devices on '{self._platform.name}'.")

        idx = min(device_index, len(devices) - 1)
        self._device = devices[idx]
        self._context = cl.Context([self._device])
        self._queue = cl.CommandQueue(
            self._context, self._device,
            properties=cl.command_queue_properties.PROFILING_ENABLE,
        )

        # Detect fp16 support
        extensions = self._device.extensions.lower() if hasattr(self._device, 'extensions') else ""
        supports_fp16 = "cl_khr_fp16" in extensions
        supports_subgroups = "cl_khr_subgroups" in extensions or "cl_intel_subgroups" in extensions

        return DeviceInfo(
            name=self._device.name.strip(),
            vendor=self._device.vendor.strip(),
            compute_units=self._device.max_compute_units,
            global_mem_mb=self._device.global_mem_size // (1024 * 1024),
            local_mem_kb=self._device.local_mem_size // 1024,
            max_workgroup=self._device.max_work_group_size,
            supports_fp16=supports_fp16,
            supports_bf16=False,  # OpenCL doesn't natively support bf16
            supports_int8=True,   # All OpenCL devices support char/uchar
            supports_subgroups=supports_subgroups,
            preferred_vector_width=self._device.preferred_vector_width_float,
            max_bandwidth_gbps=0.0,  # Would need benchmarking to determine
            driver_version=self._device.driver_version.strip(),
        )

    # ── 2. Memory Management ─────────────────────────────────────

    def allocate(self, nbytes: int, dtype: np.dtype | None = None,
                 shape: tuple | None = None) -> BufferHandle:
        if nbytes <= 0:
            nbytes = 64
        capacity = _next_power_of_two(nbytes)
        raw = cl.Buffer(self._context, cl.mem_flags.READ_WRITE, size=capacity)
        with self._lock:
            self._buf_counter += 1
            bid = self._buf_counter
        return BufferHandle(raw=raw, nbytes=nbytes, capacity=capacity,
                            dtype=dtype, shape=shape, _id=bid)

    def free(self, handle: BufferHandle) -> None:
        # OpenCL buffers are GC'd by pyopencl; explicit free is a no-op
        handle.raw = None

    def host_to_device(self, host_array: np.ndarray,
                       handle: BufferHandle) -> None:
        host_array = np.ascontiguousarray(host_array)
        cl.enqueue_copy(self._queue, handle.raw, host_array, is_blocking=True)

    def device_to_host(self, handle: BufferHandle,
                       shape: tuple, dtype: np.dtype) -> np.ndarray:
        host_array = np.empty(shape, dtype=dtype)
        cl.enqueue_copy(self._queue, host_array, handle.raw, is_blocking=True)
        return host_array

    # ── 3. Kernel Execution ──────────────────────────────────────

    def compile_kernel(self, source: str,
                       kernel_name: str) -> KernelHandle:
        cache_key = f"{hash(source)}:{kernel_name}"
        if cache_key in self._kernel_cache:
            return KernelHandle(
                raw=self._kernel_cache[cache_key],
                name=kernel_name,
                is_synthesized=True,
            )

        program = cl.Program(self._context, source).build()
        kernel = getattr(program, kernel_name)
        self._kernel_cache[cache_key] = kernel
        return KernelHandle(raw=kernel, name=kernel_name, is_synthesized=True)

    def enqueue_kernel(self, kernel: KernelHandle,
                       global_size: tuple, local_size: tuple | None,
                       *args: Any) -> None:
        kernel.raw(self._queue, global_size, local_size, *args)

    # ── 4. Synchronization ───────────────────────────────────────

    def synchronize(self) -> None:
        if self._queue is not None:
            self._queue.finish()

    # ── Accessors for backward compatibility ─────────────────────

    @property
    def context(self) -> cl.Context:
        return self._context

    @property
    def queue(self) -> cl.CommandQueue:
        return self._queue

    @property
    def device(self) -> cl.Device:
        return self._device
