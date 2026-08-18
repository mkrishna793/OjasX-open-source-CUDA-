"""
OpenCL Runtime Context — Discovers and manages the OpenCL platform, device,
context, and command queue. This is the lowest layer of TorchCL.

Usage:
    from torchcl.runtime.context import init_opencl, get_device_info
    init_opencl()
    print(get_device_info())
"""

from __future__ import annotations

import pyopencl as cl

# ── Module-level singletons ──────────────────────────────────────────
_platform: cl.Platform | None = None
_device: cl.Device | None = None
_context: cl.Context | None = None
_queue: cl.CommandQueue | None = None
_device_info: dict | None = None
_initialized: bool = False


# ── Public API ───────────────────────────────────────────────────────

def init_opencl(
    platform_index: int | None = None,
    device_index: int | None = None,
    device_type: int = cl.device_type.GPU,
) -> None:
    """Initialize the OpenCL runtime.

    Discovers available platforms and devices, then creates a context and
    command queue on the best available GPU.  Falls back to CPU if no GPU
    is found.

    Args:
        platform_index: Force a specific platform (0-based). Auto-select if None.
        device_index:   Force a specific device   (0-based). Auto-select if None.
        device_type:    Preferred device type (GPU, CPU, ALL). Default GPU.
    """
    global _platform, _device, _context, _queue, _device_info, _initialized

    if _initialized:
        return  # Already initialized — idempotent

    platforms = cl.get_platforms()
    if not platforms:
        raise RuntimeError(
            "TorchCL: No OpenCL platforms found. "
            "Make sure your GPU drivers are installed."
        )

    # ── Select platform ──────────────────────────────────────────
    if platform_index is not None:
        if platform_index >= len(platforms):
            raise ValueError(
                f"Platform index {platform_index} out of range "
                f"(found {len(platforms)} platform(s))."
            )
        _platform = platforms[platform_index]
    else:
        _platform = platforms[0]  # First platform is usually the best

    # ── Select device ────────────────────────────────────────────
    try:
        devices = _platform.get_devices(device_type=device_type)
    except cl.RuntimeError:
        devices = []

    # Fallback to any device type if preferred type not found
    if not devices:
        devices = _platform.get_devices(device_type=cl.device_type.ALL)

    if not devices:
        raise RuntimeError(
            f"TorchCL: No OpenCL devices found on platform '{_platform.name}'."
        )

    if device_index is not None:
        if device_index >= len(devices):
            raise ValueError(
                f"Device index {device_index} out of range "
                f"(found {len(devices)} device(s))."
            )
        _device = devices[device_index]
    else:
        _device = devices[0]

    # ── Create context & queue ───────────────────────────────────
    _context = cl.Context([_device])
    _queue = cl.CommandQueue(
        _context,
        _device,
        properties=cl.command_queue_properties.PROFILING_ENABLE,
    )

    # ── Cache device info ────────────────────────────────────────
    _device_info = _query_device_info(_device)
    _initialized = True


import threading
_thread_local = threading.local()


def get_context() -> cl.Context:
    """Return the active OpenCL context. Initializes if needed."""
    _ensure_initialized()
    return _context  # type: ignore[return-value]


def get_queue() -> cl.CommandQueue:
    """Return the active command queue (thread-local or default). Initializes if needed."""
    _ensure_initialized()
    q = getattr(_thread_local, "current_queue", None)
    if q is not None:
        return q
    return _queue  # type: ignore[return-value]


def _set_current_queue(q: cl.CommandQueue | None) -> None:
    _thread_local.current_queue = q


def _get_default_queue() -> cl.CommandQueue:
    _ensure_initialized()
    return _queue  # type: ignore[return-value]


def get_device() -> cl.Device:
    """Return the active OpenCL device. Initializes if needed."""
    _ensure_initialized()
    return _device  # type: ignore[return-value]


def get_device_info() -> dict:
    """Return a dict of device properties (cached). Initializes if needed."""
    _ensure_initialized()
    return dict(_device_info)  # type: ignore[arg-type]


def synchronize() -> None:
    """Block until all enqueued commands on the device finish."""
    _ensure_initialized()
    get_queue().finish()


def is_available() -> bool:
    """Return True if at least one OpenCL device exists."""
    try:
        platforms = cl.get_platforms()
        for p in platforms:
            if p.get_devices():
                return True
    except Exception:
        pass
    return False


# ── Internal helpers ─────────────────────────────────────────────────

def _ensure_initialized() -> None:
    """Lazy-init: automatically initialize on first access."""
    if not _initialized:
        init_opencl()


def _query_device_info(device: cl.Device) -> dict:
    """Extract useful device properties into a plain dict."""
    return {
        "name": device.name.strip(),
        "vendor": device.vendor.strip(),
        "version": device.version.strip(),
        "driver_version": device.driver_version.strip(),
        "device_type": cl.device_type.to_string(device.type),
        "max_compute_units": device.max_compute_units,
        "max_clock_frequency_mhz": device.max_clock_frequency,
        "max_work_group_size": device.max_work_group_size,
        "max_work_item_sizes": list(device.max_work_item_sizes),
        "global_mem_size_mb": device.global_mem_size // (1024 * 1024),
        "local_mem_size_kb": device.local_mem_size // 1024,
        "max_mem_alloc_size_mb": device.max_mem_alloc_size // (1024 * 1024),
        "preferred_vector_width_float": device.preferred_vector_width_float,
        "image_support": bool(device.image_support),
    }
