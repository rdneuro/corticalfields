"""
Deep learning architectures for brain point cloud analysis.

Provides PyTorch wrappers for geometric deep learning architectures
operating natively on cortical and subcortical point clouds. All
modules use explicit VRAM management, gradient checkpointing for
large surfaces, and automatic CPU fallback.

Submodules
----------
diffusion_net   : DiffusionNet wrapper (discretization-agnostic)
egnn            : E(n) Equivariant Graph Neural Networks
"""

from __future__ import annotations

__all__ = ["diffusion_net", "egnn"]


def __getattr__(name: str):
    if name in __all__:
        import importlib
        mod = importlib.import_module(f"corticalfields.pointcloud.deep.{name}")
        globals()[name] = mod
        return mod
    raise AttributeError(
        f"module 'corticalfields.pointcloud.deep' has no attribute {name!r}"
    )
