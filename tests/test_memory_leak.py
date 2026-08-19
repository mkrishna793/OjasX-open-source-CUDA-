"""
Test suite for shadow tensor garbage collection and memory leak prevention.
"""

import gc
import torch
import torchcl


def test_shadow_buffer_gc():
    print("\n--- Test Shadow Buffer Garbage Collection ---")
    
    # Establish clean baseline
    gc.collect()
    initial_count = len(torchcl.api._opencl_buffers)
    print(f"Initial shadow buffers: {initial_count}")
    
    def allocate_tensors():
        for _ in range(50):
            t = torchcl.randn(10, 10)
            tid = getattr(t, "_torchcl_id")
            assert tid in torchcl.api._opencl_buffers

    allocate_tensors()
    
    # Forces garbage collection
    gc.collect()
    
    # Check that temporary tensors were removed
    final_count = len(torchcl.api._opencl_buffers)
    print(f"Final shadow buffers (after GC): {final_count}")
    
    # The temporary buffers should be fully cleaned up
    assert final_count == initial_count
    print("  [PASS] Shadow buffer garbage collection")


if __name__ == "__main__":
    test_shadow_buffer_gc()
    print("\nAll memory leak tests completed successfully!")
