"""
OjasX Minimal Graph IR & JIT Vectorized Kernel Synthesizer.
Transforms computation DAGs into fused, vectorized OpenCL C / SPIR-V kernels.

Usage:
    from torchcl.jit.graph_ir import IRGraph, IRNode

    g = IRGraph()
    x = g.placeholder("x", shape=(1024, 1024))
    w = g.placeholder("w", shape=(1024, 1024))
    b = g.placeholder("b", shape=(1024, 1024))

    # Build graph
    prod = g.mul(x, w)
    added = g.add(prod, b)
    out = g.gelu(added)
    g.output(out)

    # Synthesize fused kernel
    kernel_source = g.synthesize_fused_kernel()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Tuple, Optional, Any, Set
import numpy as np
import pyopencl as cl

from torchcl.runtime.context import get_context, get_queue
from torchcl.runtime.memory import CLBuffer, get_buffer_pool
from torchcl.jit.cache import get_kernel_cache


class IROp(Enum):
    PLACEHOLDER = auto()
    CONSTANT = auto()
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    NEG = auto()
    ABS = auto()
    EXP = auto()
    LOG = auto()
    SQRT = auto()
    RELU = auto()
    SIGMOID = auto()
    TANH = auto()
    GELU = auto()
    SILU = auto()
    MATMUL = auto()
    LAYERNORM = auto()
    SOFTMAX = auto()
    OUTPUT = auto()


_OP_CODEGEN_MAP = {
    IROp.ADD: "({0} + {1})",
    IROp.SUB: "({0} - {1})",
    IROp.MUL: "({0} * {1})",
    IROp.DIV: "({0} / {1})",
    IROp.NEG: "(-({0}))",
    IROp.ABS: "fabs({0})",
    IROp.EXP: "exp({0})",
    IROp.LOG: "log({0})",
    IROp.SQRT: "sqrt({0})",
    IROp.RELU: "fmax({0}, 0.0f)",
    IROp.SIGMOID: "(1.0f / (1.0f + exp(-({0}))))",
    IROp.TANH: "tanh({0})",
    IROp.SILU: "(({0}) / (1.0f + exp(-({0}))))",
    IROp.GELU: "(({0}) * 0.5f * (1.0f + tanh(0.7978845608f * (({0}) + 0.044715f * ({0}) * ({0}) * ({0})))))",
}


@dataclass
class IRNode:
    name: str
    op: IROp
    args: List[IRNode] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    shape: Optional[Tuple[int, ...]] = None
    dtype: np.dtype = np.float32
    cl_buffer: Optional[CLBuffer] = None

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, IRNode):
            return False
        return self.name == other.name


class IRGraph:
    """Represents a computation graph for operator fusion and compilation."""

    def __init__(self, name: str = "ojasx_graph") -> None:
        self.name = name
        self.nodes: List[IRNode] = []
        self.inputs: List[IRNode] = []
        self.outputs: List[IRNode] = []
        self._node_count: int = 0

    def _gen_name(self, prefix: str) -> str:
        self._node_count += 1
        return f"{prefix}_{self._node_count}"

    def placeholder(self, name: Optional[str] = None, shape: Optional[Tuple[int, ...]] = None, dtype: np.dtype = np.float32) -> IRNode:
        n = IRNode(
            name=name or self._gen_name("in"),
            op=IROp.PLACEHOLDER,
            shape=shape,
            dtype=dtype,
        )
        self.nodes.append(n)
        self.inputs.append(n)
        return n

    def add(self, a: IRNode, b: IRNode) -> IRNode:
        n = IRNode(name=self._gen_name("add"), op=IROp.ADD, args=[a, b], shape=a.shape or b.shape)
        self.nodes.append(n)
        return n

    def sub(self, a: IRNode, b: IRNode) -> IRNode:
        n = IRNode(name=self._gen_name("sub"), op=IROp.SUB, args=[a, b], shape=a.shape or b.shape)
        self.nodes.append(n)
        return n

    def mul(self, a: IRNode, b: IRNode) -> IRNode:
        n = IRNode(name=self._gen_name("mul"), op=IROp.MUL, args=[a, b], shape=a.shape or b.shape)
        self.nodes.append(n)
        return n

    def div(self, a: IRNode, b: IRNode) -> IRNode:
        n = IRNode(name=self._gen_name("div"), op=IROp.DIV, args=[a, b], shape=a.shape or b.shape)
        self.nodes.append(n)
        return n

    def relu(self, a: IRNode) -> IRNode:
        n = IRNode(name=self._gen_name("relu"), op=IROp.RELU, args=[a], shape=a.shape)
        self.nodes.append(n)
        return n

    def gelu(self, a: IRNode) -> IRNode:
        n = IRNode(name=self._gen_name("gelu"), op=IROp.GELU, args=[a], shape=a.shape)
        self.nodes.append(n)
        return n

    def silu(self, a: IRNode) -> IRNode:
        n = IRNode(name=self._gen_name("silu"), op=IROp.SILU, args=[a], shape=a.shape)
        self.nodes.append(n)
        return n

    def output(self, a: IRNode) -> IRNode:
        n = IRNode(name=self._gen_name("out"), op=IROp.OUTPUT, args=[a], shape=a.shape)
        self.nodes.append(n)
        self.outputs.append(n)
        return n

    def topological_sort(self) -> List[IRNode]:
        """Return nodes in valid topological dependency order."""
        visited: Set[IRNode] = set()
        order: List[IRNode] = []

        def dfs(node: IRNode) -> None:
            if node in visited:
                return
            visited.add(node)
            for pred in node.args:
                dfs(pred)
            order.append(node)

        for out in self.outputs:
            dfs(out)
        return order

    def synthesize_fused_kernel(self) -> Tuple[str, str]:
        """Synthesize a unified OpenCL C kernel compiling the elementwise graph into 1 kernel."""
        sorted_nodes = self.topological_sort()
        input_nodes = [n for n in sorted_nodes if n.op == IROp.PLACEHOLDER]
        output_nodes = [n.args[0] for n in sorted_nodes if n.op == IROp.OUTPUT]

        kernel_name = f"fused_graph_{self.name}_{len(sorted_nodes)}"

        # Parameter list
        param_list = []
        for i, inp in enumerate(input_nodes):
            param_list.append(f"__global const float* {inp.name}")
        for i, out in enumerate(output_nodes):
            param_list.append(f"__global float* out_{out.name}")
        param_list.append("const int total_elements")

        # Body expressions
        expr_map: Dict[IRNode, str] = {}
        lines = []

        for node in sorted_nodes:
            if node.op == IROp.PLACEHOLDER:
                expr_map[node] = f"{node.name}[gid]"
            elif node.op in _OP_CODEGEN_MAP:
                arg_exprs = [expr_map[a] for a in node.args]
                op_fmt = _OP_CODEGEN_MAP[node.op]
                expr_map[node] = f"var_{node.name}"
                sub_expr = op_fmt.format(*arg_exprs)
                lines.append(f"    float var_{node.name} = {sub_expr};")
            elif node.op == IROp.OUTPUT:
                out_target = node.args[0]
                lines.append(f"    out_{out_target.name}[gid] = {expr_map[out_target]};")

        kernel_body = "\n".join(lines)
        source = f"""
__kernel void {kernel_name}(
    {', '.join(param_list)}
) {{
    int gid = get_global_id(0);
    if (gid >= total_elements) return;

{kernel_body}
}}
"""
        return kernel_name, source
