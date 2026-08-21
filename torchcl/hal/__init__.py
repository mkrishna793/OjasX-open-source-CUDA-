"""
OjasX HAL — Hardware Abstraction Layer.
The universal bridge between PyTorch and ANY GPU hardware.

Usage:
    from torchcl.hal import get_hal, get_device_info
    hal = get_hal()
    buf = hal.allocate(1024)
    hal.host_to_device(np_array, buf)
"""

from torchcl.hal.interface import (
    HALBackend,
    DeviceInfo,
    BufferHandle,
    PrecisionPolicy,
)

# ── Singleton HAL instance ───────────────────────────────────────────
_hal_instance: HALBackend | None = None


def get_hal() -> HALBackend:
    """Get the active HAL backend (lazy-initialized)."""
    global _hal_instance
    if _hal_instance is None:
        from torchcl.hal.opencl_backend import OpenCLHAL
        _hal_instance = OpenCLHAL()
    return _hal_instance


def get_hal_device_info() -> DeviceInfo:
    """Shorthand for get_hal().device_info."""
    return get_hal().device_info


__all__ = [
    "HALBackend",
    "DeviceInfo",
    "BufferHandle",
    "PrecisionPolicy",
    "get_hal",
    "get_hal_device_info",
]
