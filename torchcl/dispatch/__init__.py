"""
OjasX Dispatch — Intelligent Cost-Aware Operation Dispatch.

Usage:
    from torchcl.dispatch import get_dispatcher
    dispatcher = get_dispatcher()
    result = dispatcher.dispatch("gemm", a, b)
"""

from torchcl.dispatch.cost_dispatcher import CostAwareDispatcher, MorphismCost
from torchcl.dispatch.kernel_synthesizer import KernelSynthesizer
from torchcl.dispatch.precision import PrecisionSelector

_dispatcher: CostAwareDispatcher | None = None


def get_dispatcher() -> CostAwareDispatcher:
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = CostAwareDispatcher()
    return _dispatcher


__all__ = [
    "CostAwareDispatcher",
    "MorphismCost",
    "KernelSynthesizer",
    "PrecisionSelector",
    "get_dispatcher",
]
