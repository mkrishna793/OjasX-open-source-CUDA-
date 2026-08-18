"""
JIT Kernel Compiler — Generates fused OpenCL kernel source code at runtime.

This is the "brain" of TorchCL. Instead of launching separate kernels for
each operation (add, relu, mul...), the JIT compiler fuses multiple
operations into a SINGLE kernel — massively reducing GPU launch overhead.

Example:
    # Without fusion: 3 kernel launches
    temp1 = matmul(A, B)    # launch 1
    temp2 = add(temp1, C)   # launch 2
    result = relu(temp2)    # launch 3

    # With fusion: 1 kernel launch
    result = fused_matmul_add_relu(A, B, C)  # launch 1 — does ALL three!
"""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np
import pyopencl as cl

from torchcl.runtime.context import get_context, get_queue
from torchcl.jit.cache import get_kernel_cache
from torchcl.jit.tuner import get_auto_tuner


# ── Supported fuse-able operations ───────────────────────────────────
# Use __VAL__ as placeholder to avoid corrupting function names like exp/fabs
_UNARY_OPS = {
    "relu":       "fmax(__VAL__, 0.0f)",
    "sigmoid":    "(1.0f / (1.0f + exp(-(__VAL__))))",
    "tanh":       "tanh(__VAL__)",
    "neg":        "(-(__VAL__))",
    "abs":        "fabs(__VAL__)",
    "exp":        "exp(__VAL__)",
    "log":        "log(__VAL__)",
    "sqrt":       "sqrt(__VAL__)",
    "silu":       "((__VAL__) / (1.0f + exp(-(__VAL__))))",
    "gelu":       "((__VAL__) * 0.5f * (1.0f + tanh(0.7978845608f * ((__VAL__) + 0.044715f * (__VAL__) * (__VAL__) * (__VAL__)))))",
    "square":     "((__VAL__) * (__VAL__))",
}

_BINARY_OPS = {
    "add": "(a_val + b_val)",
    "sub": "(a_val - b_val)",
    "mul": "(a_val * b_val)",
    "div": "(a_val / b_val)",
}


class JITCompiler:
    """Generates and compiles fused OpenCL kernels at runtime."""

    def __init__(self) -> None:
        self._cache = get_kernel_cache()
        self._tuner = get_auto_tuner()

    # ── Fused element-wise chain ─────────────────────────────────

    def fuse_elementwise_chain(
        self,
        ops: Sequence[str],
        n: int,
        input_buffers: list[cl.Buffer],
        output_buffer: cl.Buffer,
    ) -> None:
        """Fuse a chain of element-wise unary ops into one kernel.

        Example:
            fuse_elementwise_chain(["relu", "sigmoid"], n, [a_buf], out_buf)
            # Generates: out[i] = sigmoid(relu(a[i]))
        """
        cache_key = f"fused_unary_{'_'.join(ops)}_{n}"

        # Check cache first
        kernel = self._cache.get(cache_key)
        if kernel is None:
            source = self._generate_fused_unary_source(ops)
            kernel = self._compile(cache_key, source, "fused_unary")
            self._cache.put(cache_key, kernel)

        queue = get_queue()
        workgroup = self._tuner.optimal_workgroup_1d(n)
        global_size = (self._round_up(n, workgroup),)
        local_size = (workgroup,)

        from torchcl.jit.graph_runner import get_active_capture_graph
        cg = get_active_capture_graph()
        if cg is not None:
            cg.record_kernel_launch(kernel, global_size, local_size, input_buffers[0], output_buffer, np.int32(n))

        kernel(queue, global_size, local_size,
               input_buffers[0], output_buffer, np.int32(n))

    def fuse_binary_then_unary(
        self,
        binary_op: str,
        unary_ops: Sequence[str],
        n: int,
        a_buf: cl.Buffer,
        b_buf: cl.Buffer,
        out_buf: cl.Buffer,
    ) -> None:
        """Fuse a binary op followed by unary ops.

        Example:
            fuse_binary_then_unary("add", ["relu"], n, a, b, out)
            # Generates: out[i] = relu(a[i] + b[i])
        """
        cache_key = f"fused_{binary_op}_{'_'.join(unary_ops)}_{n}"

        kernel = self._cache.get(cache_key)
        if kernel is None:
            source = self._generate_fused_binary_unary_source(binary_op, unary_ops)
            kernel = self._compile(cache_key, source, "fused_binary_unary")
            self._cache.put(cache_key, kernel)

        queue = get_queue()
        workgroup = self._tuner.optimal_workgroup_1d(n)
        global_size = (self._round_up(n, workgroup),)
        local_size = (workgroup,)

        from torchcl.jit.graph_runner import get_active_capture_graph
        cg = get_active_capture_graph()
        if cg is not None:
            cg.record_kernel_launch(kernel, global_size, local_size, a_buf, b_buf, out_buf, np.int32(n))

        kernel(queue, global_size, local_size,
               a_buf, b_buf, out_buf, np.int32(n))

    # ── Code generation ──────────────────────────────────────────

    def _generate_fused_unary_source(self, ops: Sequence[str]) -> str:
        """Generate fused unary kernel source code."""
        # Build the nested expression using __VAL__ placeholder
        expr = "__VAL__"
        for op in ops:
            if op not in _UNARY_OPS:
                raise ValueError(f"Unknown unary op for fusion: {op}")
            template = _UNARY_OPS[op]
            expr = template.replace("__VAL__", f"({expr})")

        # Final substitution: replace __VAL__ with the actual variable
        final_expr = expr.replace("__VAL__", "x")

        return f"""
__kernel void fused_unary(__global const float* input,
                          __global float* output,
                          const int n) {{
    int gid = get_global_id(0);
    if (gid < n) {{
        float x = input[gid];
        output[gid] = {final_expr};
    }}
}}
"""

    def _generate_fused_binary_unary_source(
        self,
        binary_op: str,
        unary_ops: Sequence[str],
    ) -> str:
        """Generate fused binary + unary chain kernel source."""
        if binary_op not in _BINARY_OPS:
            raise ValueError(f"Unknown binary op for fusion: {binary_op}")

        # Start with binary op result as __VAL__
        expr = _BINARY_OPS[binary_op]

        # Chain unary ops using __VAL__ placeholder
        for op in unary_ops:
            if op not in _UNARY_OPS:
                raise ValueError(f"Unknown unary op for fusion: {op}")
            template = _UNARY_OPS[op]
            expr = template.replace("__VAL__", f"({expr})")

        # Clean up any remaining __VAL__ placeholders
        final_expr = expr.replace("__VAL__", "x")

        return f"""
__kernel void fused_binary_unary(__global const float* a,
                                 __global const float* b,
                                 __global float* output,
                                 const int n) {{
    int gid = get_global_id(0);
    if (gid < n) {{
        float a_val = a[gid];
        float b_val = b[gid];
        output[gid] = {final_expr};
    }}
}}
"""

    def generate_custom_kernel(self, name: str, source: str) -> cl.Kernel:
        """Compile a fully custom kernel source string.

        For advanced users who want to write their own OpenCL C code.
        """
        kernel = self._cache.get(name)
        if kernel is None:
            kernel = self._compile(name, source, name)
            self._cache.put(name, kernel)
        return kernel

    # ── Internal helpers ─────────────────────────────────────────

    def _compile(self, cache_key: str, source: str, kernel_name: str) -> cl.Kernel:
        """Compile source code to an OpenCL kernel."""
        ctx = get_context()
        program = cl.Program(ctx, source).build()
        return cl.Kernel(program, kernel_name)

    @staticmethod
    def _round_up(n: int, multiple: int) -> int:
        return ((n + multiple - 1) // multiple) * multiple


# ── Module-level singleton ───────────────────────────────────────────
_global_jit: JITCompiler | None = None


def get_jit_compiler() -> JITCompiler:
    """Return the global JIT compiler."""
    global _global_jit
    if _global_jit is None:
        _global_jit = JITCompiler()
    return _global_jit
