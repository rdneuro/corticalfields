"""
Point cloud cortical geometry — unified mesh-free and mesh-based analysis.

This subpackage provides a **complete FreeSurfer-free path** for cortical
morphometry via point clouds, plus GPU-accelerated spectral analysis,
functional maps, optimal transport, registration, and deep learning
architectures operating natively on brain point clouds.

Submodules
----------
core            : CorticalPointCloud dataclass and basic geometry
io              : FreeSurfer → PCD, NIfTI → PCD, T1w extraction
laplacian       : LBO computation (mesh and point cloud via robust_laplacian)
spectral        : GPU-accelerated HKS / WKS / GPS on point clouds
functional_maps : Functional maps and correspondence on point clouds
transport       : GPU-accelerated optimal transport (Wasserstein, Sinkhorn)
registration    : Point cloud registration (ICP, CPD, learned)
morphometrics   : Surface area, volume, curvature, thickness from PCD
viz             : Publication-quality point cloud brain visualization
deep            : Deep learning architectures (DiffusionNet, EGNN, PointNet++)
"""

from __future__ import annotations

# ── Backward-compatible re-exports from the legacy monolithic module ─────
# These ensure that `from corticalfields.pointcloud import X` still works
# for every symbol that was previously in the flat pointcloud.py.
from corticalfields._pointcloud_legacy import (
    CorticalPointCloud,
    T1wExtractionResult,
    from_freesurfer_surface,
    from_cortical_surface,
    from_nifti_mask,
    from_t1w,
    compute_mesh_laplacian,
    compute_mesh_eigenpairs,
    compute_pointcloud_laplacian,
    compute_pointcloud_eigenpairs,
    estimate_normals,
    compute_point_areas,
    to_feature_matrix,
)

# ── New submodule lazy imports ───────────────────────────────────────────
# Heavy GPU modules are loaded on first access to keep top-level import fast.

_LAZY_SUBMODULES = {
    "spectral",
    "functional_maps",
    "transport",
    "registration",
    "morphometrics",
    "viz",
    "deep",
}


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        import importlib
        mod = importlib.import_module(f"corticalfields.pointcloud.{name}")
        globals()[name] = mod
        return mod
    raise AttributeError(f"module 'corticalfields.pointcloud' has no attribute {name!r}")


__all__ = [
    # Legacy re-exports
    "CorticalPointCloud",
    "T1wExtractionResult",
    "from_freesurfer_surface",
    "from_cortical_surface",
    "from_nifti_mask",
    "from_t1w",
    "compute_mesh_laplacian",
    "compute_mesh_eigenpairs",
    "compute_pointcloud_laplacian",
    "compute_pointcloud_eigenpairs",
    "estimate_normals",
    "compute_point_areas",
    "to_feature_matrix",
    # New submodules
    "spectral",
    "functional_maps",
    "transport",
    "registration",
    "morphometrics",
    "viz",
    "deep",
]
