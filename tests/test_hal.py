"""
HAL (Hardware Abstraction Layer) Test Suite.
Verifies the universal GPU interface works correctly with the OpenCL backend.
"""

import pytest
import numpy as np
import torch


def test_hal_init():
    """HAL discovers and reports GPU capabilities."""
    from torchcl.hal import get_hal, get_hal_device_info
    hal = get_hal()
    info = hal.device_info

    assert info.name, "Device name must not be empty"
    assert info.vendor, "Vendor must not be empty"
    assert info.compute_units > 0
    assert info.global_mem_mb > 0
    assert info.local_mem_kb > 0
    assert info.max_workgroup > 0
    assert info.vendor_short in ("intel", "amd", "apple", "qualcomm", "arm", "nvidia", "generic")

    # get_hal_device_info shorthand
    info2 = get_hal_device_info()
    assert info2.name == info.name


def test_hal_allocate_free():
    """HAL allocates and frees GPU buffers."""
    from torchcl.hal import get_hal
    hal = get_hal()

    buf = hal.allocate(1024, dtype=np.dtype(np.float32), shape=(256,))
    assert buf.nbytes == 1024
    assert buf.capacity >= 1024
    assert buf._id > 0

    hal.free(buf)
    assert buf.raw is None  # Freed


def test_hal_host_device_roundtrip():
    """Data survives CPU → GPU → CPU roundtrip at byte-level precision."""
    from torchcl.hal import get_hal
    hal = get_hal()

    original = np.array([1.0, 2.5, -3.14, 0.0, 42.0], dtype=np.float32)
    buf = hal.allocate(original.nbytes, dtype=original.dtype, shape=original.shape)
    hal.host_to_device(original, buf)
    recovered = hal.device_to_host(buf, shape=original.shape, dtype=original.dtype)

    np.testing.assert_array_almost_equal(recovered, original, decimal=6)


def test_hal_compile_and_run_kernel():
    """HAL compiles and executes a custom OpenCL kernel."""
    from torchcl.hal import get_hal
    hal = get_hal()

    source = """
    __kernel void test_add_one(__global float* data, const int n) {
        int i = get_global_id(0);
        if (i < n) data[i] += 1.0f;
    }
    """
    kernel = hal.compile_kernel(source, "test_add_one")
    assert kernel.name == "test_add_one"
    assert kernel.is_synthesized  # Compiled at runtime

    data = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    buf = hal.allocate(data.nbytes, dtype=data.dtype, shape=data.shape)
    hal.host_to_device(data, buf)

    hal.enqueue_kernel(kernel, (4,), None, buf.raw, np.int32(4))
    hal.synchronize()

    result = hal.device_to_host(buf, shape=data.shape, dtype=data.dtype)
    np.testing.assert_array_almost_equal(result, [11.0, 21.0, 31.0, 41.0])


def test_hal_device_info_properties():
    """DeviceInfo vendor_short normalization works."""
    from torchcl.hal.interface import DeviceInfo
    
    assert DeviceInfo(name="x", vendor="Intel Corporation", compute_units=1,
                      global_mem_mb=1, local_mem_kb=1, max_workgroup=1).vendor_short == "intel"
    assert DeviceInfo(name="x", vendor="Advanced Micro Devices", compute_units=1,
                      global_mem_mb=1, local_mem_kb=1, max_workgroup=1).vendor_short == "amd"
    assert DeviceInfo(name="x", vendor="Apple", compute_units=1,
                      global_mem_mb=1, local_mem_kb=1, max_workgroup=1).vendor_short == "apple"
    assert DeviceInfo(name="x", vendor="Unknown Corp", compute_units=1,
                      global_mem_mb=1, local_mem_kb=1, max_workgroup=1).vendor_short == "generic"


def test_hal_buffer_handle_repr():
    """BufferHandle has a useful repr."""
    from torchcl.hal.interface import BufferHandle
    h = BufferHandle(raw=None, nbytes=4096, capacity=4096, _id=42,
                     dtype=np.dtype(np.float32), shape=(1024,))
    assert "42" in repr(h)
    assert "4096" in repr(h)


def test_hal_synchronize():
    """Synchronize doesn't crash."""
    from torchcl.hal import get_hal
    hal = get_hal()
    hal.synchronize()  # Should not raise
