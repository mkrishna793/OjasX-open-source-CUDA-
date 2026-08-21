"""
PrecisionSelector — Auto-selects optimal numerical precision (fp32/fp16/int8)
based on hardware capability and data statistics.

Policies:
    SAFE:  fp32 for training, auto fp16 for inference when data fits
    AUTO:  Full AMP — fp16 forward, fp32 gradients  
    EXACT: fp32 everywhere, bit-perfect reproducibility
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy as np
import torch

from torchcl.hal.interface import DeviceInfo, PrecisionPolicy


@dataclass
class PrecisionDecision:
    """The result of precision analysis."""
    compute_dtype: str       # "fp32", "fp16", "bf16", "int8"
    accumulate_dtype: str    # Always "fp32" for numerical safety
    reason: str              # Human-readable explanation
    bytes_per_element: int   # 4, 2, or 1

    @property
    def numpy_dtype(self) -> np.dtype:
        return {
            "fp32": np.dtype(np.float32),
            "fp16": np.dtype(np.float16),
            "int8": np.dtype(np.int8),
        }.get(self.compute_dtype, np.dtype(np.float32))


class PrecisionSelector:
    """Automatically selects the best numerical precision for each operation."""

    def __init__(self, policy: PrecisionPolicy = PrecisionPolicy.SAFE,
                 device_info: DeviceInfo | None = None) -> None:
        self.policy = policy
        self._device_info = device_info

    @property
    def device_info(self) -> DeviceInfo:
        if self._device_info is None:
            from torchcl.hal import get_hal
            self._device_info = get_hal().device_info
        return self._device_info

    def select(self, op: str, tensor: torch.Tensor | None = None,
               is_training: bool = True) -> PrecisionDecision:
        """Select optimal precision for an operation."""
        
        # EXACT policy: always fp32
        if self.policy == PrecisionPolicy.EXACT:
            return PrecisionDecision(
                compute_dtype="fp32", accumulate_dtype="fp32",
                reason="EXACT policy: bit-perfect fp32",
                bytes_per_element=4,
            )

        # Check hardware capability
        hw_fp16 = self.device_info.supports_fp16

        # SAFE policy: fp32 for training, auto fp16 for inference
        if self.policy == PrecisionPolicy.SAFE:
            if is_training:
                return PrecisionDecision(
                    compute_dtype="fp32", accumulate_dtype="fp32",
                    reason="SAFE policy: fp32 during training",
                    bytes_per_element=4,
                )
            if hw_fp16 and tensor is not None and self._data_fits_fp16(tensor):
                return PrecisionDecision(
                    compute_dtype="fp16", accumulate_dtype="fp32",
                    reason="SAFE policy: data fits fp16, inference mode",
                    bytes_per_element=2,
                )
            return PrecisionDecision(
                compute_dtype="fp32", accumulate_dtype="fp32",
                reason="SAFE policy: fp32 fallback",
                bytes_per_element=4,
            )

        # AUTO policy: full AMP
        if self.policy == PrecisionPolicy.AUTO:
            # Ops safe for fp16: matmul, convolution, attention
            fp16_safe_ops = {"gemm", "matmul", "attention", "conv2d",
                             "relu", "sigmoid", "gelu", "silu", "tanh",
                             "softmax", "layer_norm", "rms_norm"}
            # Ops that need fp32: loss, reduction, normalization accumulation
            fp32_required = {"cross_entropy", "mse_loss", "sum", "mean",
                             "log", "exp", "sqrt"}

            if op in fp32_required or not hw_fp16:
                return PrecisionDecision(
                    compute_dtype="fp32", accumulate_dtype="fp32",
                    reason=f"AUTO: {op} requires fp32 precision",
                    bytes_per_element=4,
                )
            if op in fp16_safe_ops:
                return PrecisionDecision(
                    compute_dtype="fp16", accumulate_dtype="fp32",
                    reason=f"AUTO AMP: {op} safe for fp16",
                    bytes_per_element=2,
                )
            return PrecisionDecision(
                compute_dtype="fp32", accumulate_dtype="fp32",
                reason=f"AUTO: unknown op {op}, defaulting fp32",
                bytes_per_element=4,
            )

        # Default fallback
        return PrecisionDecision(
            compute_dtype="fp32", accumulate_dtype="fp32",
            reason="Default fp32",
            bytes_per_element=4,
        )

    def _data_fits_fp16(self, tensor: torch.Tensor,
                        max_samples: int = 256) -> bool:
        """Check if tensor values fit in fp16 range without overflow."""
        FP16_MAX = 65504.0
        try:
            flat = tensor.detach().cpu().flatten()
            n = min(max_samples, flat.numel())
            if n == 0:
                return True
            sample = flat[:n].float()
            max_val = sample.abs().max().item()
            return max_val < FP16_MAX * 0.9  # 10% safety margin
        except Exception:
            return False
