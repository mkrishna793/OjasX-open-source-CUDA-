"""
OjasX V3 — Native Tensor Subclass and PyTorch Integration
Hooks into PyTorch's __torch_dispatch__ and monkeypatches nn.Linear / nn.Module
to provide a seamless 'device="opencl"' and 'device="ojasx"' native interface.
"""

import torch
import torchcl
import numpy as np
import torchcl.autograd as autograd

# ── ATen Op Dispatcher ────────────────────────────────────────────────

def _sdpa_dispatch(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    def _unwrap(x):
        return x._elem if isinstance(x, OjasXTensor) else x
    q, k, v = _unwrap(query), _unwrap(key), _unwrap(value)
    out = autograd.fused_attention(q, k, v, scale=scale)
    return OjasXTensor(out) if not isinstance(out, OjasXTensor) else out

def _layer_norm_dispatch(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    def _unwrap(x):
        return x._elem if isinstance(x, OjasXTensor) else x
    inp = _unwrap(input)
    w = _unwrap(weight) if weight is not None else None
    b = _unwrap(bias) if bias is not None else None
    out = autograd.LayerNormFunction.apply(inp, w, b, normalized_shape[-1], eps)
    return OjasXTensor(out)

def _embedding_dispatch(weight, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False):
    def _unwrap(x):
        return x._elem if isinstance(x, OjasXTensor) else x
    w = _unwrap(weight)
    idx = _unwrap(indices)
    out = autograd.EmbeddingFunction.apply(w, idx)
    return OjasXTensor(out)

# Mapping from PyTorch ATen ops to torchcl autograd functions
DISPATCH_TABLE = {
    torch.ops.aten.add.Tensor: autograd.AddFunction.apply,
    torch.ops.aten.sub.Tensor: autograd.SubFunction.apply,
    torch.ops.aten.mul.Tensor: autograd.MulFunction.apply,
    torch.ops.aten.div.Tensor: autograd.DivFunction.apply,
    torch.ops.aten.neg.default: torchcl.neg,
    torch.ops.aten.abs.default: torchcl.abs_,
    torch.ops.aten.exp.default: torchcl.exp,
    torch.ops.aten.log.default: torchcl.log,
    torch.ops.aten.sqrt.default: torchcl.sqrt,
    torch.ops.aten.relu.default: autograd.ReluFunction.apply,
    torch.ops.aten.sigmoid.default: autograd.SigmoidFunction.apply,
    torch.ops.aten.tanh.default: autograd.TanhFunction.apply,
    torch.ops.aten.gelu.default: autograd.GeluFunction.apply,
    torch.ops.aten.silu.default: autograd.SiluFunction.apply,
    torch.ops.aten.mm.default: autograd.MatmulFunction.apply,
    torch.ops.aten.native_layer_norm.default: _layer_norm_dispatch,
    torch.ops.aten.scaled_dot_product_attention.default: _sdpa_dispatch,
    torch.ops.aten.embedding.default: _embedding_dispatch,
}


def execute_on_cpu(func, *args, **kwargs):
    """Seamless CPU fallback: moves tensors to CPU, runs PyTorch native, moves back."""
    def unwrap_to_cpu(x):
        if isinstance(x, OjasXTensor):
            return torchcl.to_cpu(x._elem)
        return x

    cpu_args = torch.utils._pytree.tree_map(unwrap_to_cpu, args)
    cpu_kwargs = torch.utils._pytree.tree_map(unwrap_to_cpu, kwargs)

    res = func(*cpu_args, **cpu_kwargs)

    def wrap_to_opencl(x):
        if isinstance(x, torch.Tensor):
            return OjasXTensor(torchcl.to_opencl(x.contiguous()))
        return x

    return torch.utils._pytree.tree_map(wrap_to_opencl, res)


def dispatch_op(func, *args, **kwargs):
    """Routes the ATen operation to OpenCL or falls back to CPU."""
    if func in DISPATCH_TABLE:
        try:
            res = DISPATCH_TABLE[func](*args, **kwargs)
            return res
        except Exception:
            return execute_on_cpu(func, *args, **kwargs)

    return execute_on_cpu(func, *args, **kwargs)


# ── Wrapper Tensor Subclass ───────────────────────────────────────────

class OjasXTensor(torch.Tensor):
    """A PyTorch Tensor subclass that intercepts all operations."""
    @staticmethod
    def __new__(cls, elem):
        if isinstance(elem, cls):
            return elem
        real_size = getattr(elem, "_torchcl_shape", elem.size())
        dummy = torch.empty(real_size)
        real_strides = dummy.stride()

        r = torch.Tensor._make_wrapper_subclass(
            cls,
            size=real_size,
            strides=real_strides,
            storage_offset=0,
            dtype=elem.dtype,
            layout=elem.layout,
            device=torch.device("cpu"),
            requires_grad=elem.requires_grad
        )
        r._elem = elem
        return r

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        return dispatch_op(func, *args, **kwargs)

    @property
    def device(self):
        class FakeDevice:
            def __init__(self):
                self.type = 'opencl'
            def __str__(self):
                return "opencl"
            def __repr__(self):
                return "device(type='opencl')"
            def __eq__(self, other):
                return str(other) in ('opencl', 'ojasx') or (hasattr(other, 'type') and other.type in ('opencl', 'ojasx', 'privateuseone'))
        return FakeDevice()

    def __repr__(self):
        return f"OjasXTensor({self._elem}, device='opencl')"

    def __getattr__(self, name):
        if name in ("_torchcl_id", "_torchcl_shape", "_torchcl_dtype",
                    "_lazy_inputs", "_lazy_op_type", "_lazy_binary_op", "_lazy_ops"):
            if hasattr(self, "_elem"):
                return getattr(self._elem, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


# ── Monkeypatches for PyTorch ──────────────────────────────────────────

_original_tensor_to = torch.Tensor.to
_original_module_to = torch.nn.Module.to
_original_linear_forward = torch.nn.Linear.forward

_TARGET_DEVICES = {"opencl", "ojasx", "device(type='opencl')", "device(type='ojasx')"}


def _custom_tensor_to(self, *args, **kwargs):
    """Intercepts tensor.to('opencl') and tensor.to('ojasx')"""
    target_device = None
    if len(args) > 0 and isinstance(args[0], str):
        target_device = args[0].lower()
    elif kwargs.get("device") is not None:
        target_device = str(kwargs.get("device")).lower()

    if target_device in _TARGET_DEVICES:
        if isinstance(self, OjasXTensor):
            return self
        if torchcl.is_opencl_tensor(self):
            return OjasXTensor(self)
        ocl_tensor = torchcl.to_opencl(self)
        return OjasXTensor(ocl_tensor)

    if (target_device == "cpu" or target_device == "device(type='cpu')") and isinstance(self, OjasXTensor):
        return torchcl.to_cpu(self._elem)

    return _original_tensor_to(self, *args, **kwargs)


def _custom_module_to(self, *args, **kwargs):
    """Intercepts model.to('opencl') and model.to('ojasx')"""
    target_device = None
    if len(args) > 0 and isinstance(args[0], str):
        target_device = args[0].lower()
    elif kwargs.get("device") is not None:
        target_device = str(kwargs.get("device")).lower()

    if target_device in _TARGET_DEVICES:
        def _convert(t):
            if isinstance(t, OjasXTensor):
                return t
            return OjasXTensor(torchcl.to_opencl(t.contiguous()))

        for name, param in self.named_parameters(recurse=False):
            if param is not None:
                new_param = torch.nn.Parameter(_convert(param.data), requires_grad=param.requires_grad)
                self.register_parameter(name, new_param)

        for name, buf in self.named_buffers(recurse=False):
            if buf is not None:
                self.register_buffer(name, _convert(buf.data))

        for child in self.children():
            child.to("opencl")

        return self

    return _original_module_to(self, *args, **kwargs)


def _custom_linear_forward(self, input: torch.Tensor) -> torch.Tensor:
    """Intercepts Linear forward to avoid ATen addmm, routing through fused LinearFunction."""
    is_opencl = False
    if isinstance(input, OjasXTensor):
        is_opencl = True
    elif input.device.type in ("opencl", "ojasx", "privateuseone"):
        is_opencl = True
    elif getattr(input, "device", None) in ("opencl", "ojasx"):
        is_opencl = True

    if is_opencl:
        from torchcl.autograd import LinearFunction
        res = LinearFunction.apply(input, self.weight, self.bias)
        return res

    return _original_linear_forward(self, input)


def apply_monkeypatches():
    """Applies the patches. Called automatically in __init__.py"""
    torch.Tensor.to = _custom_tensor_to
    torch.nn.Module.to = _custom_module_to
    torch.nn.Linear.forward = _custom_linear_forward
