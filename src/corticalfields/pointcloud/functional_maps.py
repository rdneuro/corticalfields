"""
Functional maps and correspondence on point clouds.

Computes the functional map C matrix between pairs of brain surface
point clouds (e.g., left ↔ mirrored-right for asymmetry analysis)
using point cloud LBO eigenpairs. Supports both classical descriptor-
based optimization and DiffusionNet-learned features.

The functional map framework operates identically on meshes and point
clouds — the only difference is how the LBO eigenbasis is computed
(cotangent Laplacian vs ``robust_laplacian.point_cloud_laplacian``).
All algorithms in this module are discretization-agnostic.

Functions
---------
compute_functional_map        : C matrix from descriptor correspondences
convert_to_pointwise_map      : C → vertex-to-vertex map (nearest-neighbor)
zoomout_refinement            : Spectral upsampling of the C matrix
descriptor_preservation_loss  : Descriptor-based loss for optimization
compute_descriptors           : Extract descriptors for functional maps

References
----------
Ovsjanikov et al. (2012). Functional Maps: A Flexible Representation of
    Maps Between Shapes. ACM SIGGRAPH.
Melzi et al. (2019). ZoomOut: Spectral Upsampling for Efficient Shape
    Correspondence. ACM SIGGRAPH Asia.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class FunctionalMapResult:
    """
    Result of a functional map computation between two point clouds.

    Attributes
    ----------
    C : np.ndarray, shape (K2, K1)
        Functional map matrix in the spectral domain.
    pointwise_map : np.ndarray or None, shape (N2,)
        Vertex indices mapping source → target.
    eigenvalues_source : np.ndarray, shape (K1,)
    eigenvalues_target : np.ndarray, shape (K2,)
    eigenvectors_source : np.ndarray, shape (N1, K1)
    eigenvectors_target : np.ndarray, shape (N2, K2)
    """

    C: np.ndarray
    pointwise_map: Optional[np.ndarray] = None
    eigenvalues_source: Optional[np.ndarray] = None
    eigenvalues_target: Optional[np.ndarray] = None
    eigenvectors_source: Optional[np.ndarray] = None
    eigenvectors_target: Optional[np.ndarray] = None


# ═══════════════════════════════════════════════════════════════════════════
# Descriptor extraction
# ═══════════════════════════════════════════════════════════════════════════


def compute_descriptors(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    descriptor_types: Optional[List[str]] = None,
    hks_scales: int = 16,
    wks_scales: int = 100,
) -> np.ndarray:
    """
    Extract spectral descriptors for functional map optimization.

    Parameters
    ----------
    eigenvalues : np.ndarray, shape (K,)
    eigenvectors : np.ndarray, shape (N, K)
    descriptor_types : list of str or None
        Which descriptors to use. Default: ``['hks', 'wks']``.
    hks_scales : int
    wks_scales : int

    Returns
    -------
    descriptors : np.ndarray, shape (N, D)
        Concatenated descriptor matrix.
    """
    from corticalfields.pointcloud.spectral import (
        hks_pointcloud,
        wks_pointcloud,
    )

    if descriptor_types is None:
        descriptor_types = ["hks", "wks"]

    parts = []
    for dtype in descriptor_types:
        if dtype == "hks":
            parts.append(hks_pointcloud(
                eigenvalues, eigenvectors, hks_scales, use_gpu=True,
            ))
        elif dtype == "wks":
            parts.append(wks_pointcloud(
                eigenvalues, eigenvectors, wks_scales, use_gpu=True,
            ))
        else:
            raise ValueError(f"Unknown descriptor type: {dtype!r}")

    desc = np.hstack(parts)
    # Normalize columns to unit variance for balanced optimization
    std = desc.std(axis=0, keepdims=True)
    std[std < 1e-12] = 1.0
    return desc / std


# ═══════════════════════════════════════════════════════════════════════════
# Functional map computation
# ═══════════════════════════════════════════════════════════════════════════


def compute_functional_map(
    eigenvectors_source: np.ndarray,
    eigenvectors_target: np.ndarray,
    descriptors_source: np.ndarray,
    descriptors_target: np.ndarray,
    eigenvalues_source: Optional[np.ndarray] = None,
    eigenvalues_target: Optional[np.ndarray] = None,
    n_basis: int = 50,
    lambda_reg: float = 1e-3,
    regularization: str = "laplacian_commutativity",
) -> FunctionalMapResult:
    """
    Compute the functional map C between two point clouds.

    Solves: min_C ‖C A_s − A_t ‖² + λ R(C)

    where A_s, A_t are the spectral projections of descriptors, and
    R(C) is a regularization term.

    Parameters
    ----------
    eigenvectors_source : np.ndarray, shape (N1, K1)
        Source LBO eigenvectors (from point cloud or mesh).
    eigenvectors_target : np.ndarray, shape (N2, K2)
        Target LBO eigenvectors.
    descriptors_source : np.ndarray, shape (N1, D)
        Source descriptor matrix.
    descriptors_target : np.ndarray, shape (N2, D)
        Target descriptor matrix.
    eigenvalues_source : np.ndarray or None, shape (K1,)
        Required for Laplacian commutativity regularization.
    eigenvalues_target : np.ndarray or None, shape (K2,)
        Required for Laplacian commutativity regularization.
    n_basis : int
        Number of spectral basis functions to use.
    lambda_reg : float
        Regularization weight.
    regularization : str
        ``'laplacian_commutativity'``, ``'operator_commutativity'``,
        or ``'none'``.

    Returns
    -------
    FunctionalMapResult
        Contains the C matrix and metadata.
    """
    K1 = min(n_basis, eigenvectors_source.shape[1])
    K2 = min(n_basis, eigenvectors_target.shape[1])

    Phi_s = eigenvectors_source[:, :K1]  # (N1, K1)
    Phi_t = eigenvectors_target[:, :K2]  # (N2, K2)

    # Project descriptors into spectral domain
    # A_s = Φ_s^T F_s, shape (K1, D)
    A_s = Phi_s.T @ descriptors_source
    A_t = Phi_t.T @ descriptors_target   # (K2, D)

    # Build the linear system for C: min_C ‖C A_s − A_t‖² + λ R(C)
    # This is solved column-by-column as a regularized least-squares problem.
    #
    # For each descriptor column d: min_c ‖A_s^T c − a_t‖² + λ ‖Rc‖²
    # where c is a row of C.
    #
    # But more efficiently: solve C = A_t A_s^T (A_s A_s^T + λ Reg)^{-1}

    AAt = A_s @ A_s.T                    # (K1, K1)

    # Regularization matrix
    if regularization == "laplacian_commutativity":
        if eigenvalues_source is None or eigenvalues_target is None:
            raise ValueError(
                "Eigenvalues required for Laplacian commutativity "
                "regularization"
            )
        lam_s = eigenvalues_source[:K1]
        lam_t = eigenvalues_target[:K2]
        # ΔΛ[i,j] = (λ_s_j − λ_t_i)² — penalizes C entries that mix
        # eigenspaces with different eigenvalues
        delta_lam = (lam_s[np.newaxis, :] - lam_t[:, np.newaxis]) ** 2
        # This acts element-wise on C; approximate with diagonal
        reg_diag = delta_lam.mean(axis=0)
        Reg = np.diag(reg_diag)
    elif regularization == "operator_commutativity":
        Reg = np.eye(K1)
    elif regularization == "none":
        Reg = np.zeros((K1, K1))
    else:
        raise ValueError(f"Unknown regularization: {regularization!r}")

    # Solve: C = A_t A_s^T (A_s A_s^T + λ Reg)^{-1}
    lhs = AAt + lambda_reg * Reg         # (K1, K1)
    rhs = A_t @ A_s.T                    # (K2, K1)

    try:
        C = np.linalg.solve(lhs.T, rhs.T).T  # (K2, K1)
    except np.linalg.LinAlgError:
        logger.warning("Singular system; using pseudoinverse")
        C = rhs @ np.linalg.pinv(lhs)

    logger.info(
        "Functional map C: %s, Frobenius norm=%.4f",
        C.shape, np.linalg.norm(C, "fro"),
    )

    return FunctionalMapResult(
        C=C,
        eigenvalues_source=eigenvalues_source,
        eigenvalues_target=eigenvalues_target,
        eigenvectors_source=eigenvectors_source,
        eigenvectors_target=eigenvectors_target,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Pointwise map conversion
# ═══════════════════════════════════════════════════════════════════════════


def convert_to_pointwise_map(
    fmap: FunctionalMapResult,
    use_gpu: bool = True,
) -> np.ndarray:
    """
    Convert a spectral functional map to a point-to-point correspondence.

    For each target vertex j, finds the source vertex i whose spectral
    embedding Φ_s[i, :] C^T is closest to Φ_t[j, :].

    Parameters
    ----------
    fmap : FunctionalMapResult
        Must contain C, eigenvectors_source, eigenvectors_target.
    use_gpu : bool
        Use CUDA for the nearest-neighbor search.

    Returns
    -------
    pointwise_map : np.ndarray, shape (N_target,)
        For each target vertex, the index of the corresponding
        source vertex.
    """
    C = fmap.C                                    # (K2, K1)
    Phi_s = fmap.eigenvectors_source[:, :C.shape[1]]  # (N1, K1)
    Phi_t = fmap.eigenvectors_target[:, :C.shape[0]]  # (N2, K2)

    if use_gpu:
        try:
            return _pointwise_map_torch(C, Phi_s, Phi_t)
        except (ImportError, RuntimeError) as exc:
            logger.info("GPU pointwise map failed (%s); using CPU", exc)

    return _pointwise_map_numpy(C, Phi_s, Phi_t)


def _pointwise_map_torch(
    C: np.ndarray,
    Phi_s: np.ndarray,
    Phi_t: np.ndarray,
) -> np.ndarray:
    """GPU-accelerated nearest-neighbor in spectral embedding space."""
    import torch
    from corticalfields.pointcloud.spectral import _get_device, _cleanup_gpu

    device = _get_device(prefer_cuda=True)

    C_t = torch.as_tensor(C, dtype=torch.float32, device=device)
    Phi_s_t = torch.as_tensor(Phi_s, dtype=torch.float32, device=device)
    Phi_t_t = torch.as_tensor(Phi_t, dtype=torch.float32, device=device)

    # Embed source into target spectral space: Φ_s C^T, shape (N1, K2)
    source_embed = Phi_s_t @ C_t.T

    # Batched nearest-neighbor to manage VRAM
    N2 = Phi_t_t.shape[0]
    batch_size = min(8192, N2)
    pmap = np.empty(N2, dtype=np.int64)

    for start in range(0, N2, batch_size):
        end = min(start + batch_size, N2)
        target_batch = Phi_t_t[start:end]         # (B, K2)
        # Pairwise distance: (B, N1)
        dists = torch.cdist(target_batch, source_embed)
        pmap[start:end] = dists.argmin(dim=1).cpu().numpy()
        del target_batch, dists

    _cleanup_gpu(C_t, Phi_s_t, Phi_t_t, source_embed)
    return pmap


def _pointwise_map_numpy(
    C: np.ndarray,
    Phi_s: np.ndarray,
    Phi_t: np.ndarray,
) -> np.ndarray:
    """CPU nearest-neighbor using scipy KDTree."""
    from scipy.spatial import cKDTree

    source_embed = Phi_s @ C.T                    # (N1, K2)
    tree = cKDTree(source_embed)
    _, pmap = tree.query(Phi_t)
    return pmap.astype(np.int64)


# ═══════════════════════════════════════════════════════════════════════════
# ZoomOut spectral upsampling
# ═══════════════════════════════════════════════════════════════════════════


def zoomout_refinement(
    fmap: FunctionalMapResult,
    k_start: Optional[int] = None,
    k_final: Optional[int] = None,
    n_iters: int = 10,
    use_gpu: bool = True,
) -> FunctionalMapResult:
    """
    Refine a functional map via ZoomOut spectral upsampling.

    Starting from a low-rank C matrix, iteratively:
    1. Convert C to pointwise map (nearest-neighbor).
    2. Recompute C at increased spectral resolution (more eigenpairs).
    3. Repeat until k_final basis functions are reached.

    Parameters
    ----------
    fmap : FunctionalMapResult
        Initial (low-rank) functional map.
    k_start : int or None
        Starting spectral resolution. Default: current C size.
    k_final : int or None
        Target spectral resolution. Default: min(K1, K2).
    n_iters : int
        Number of refinement iterations.
    use_gpu : bool
        Use CUDA for nearest-neighbor search.

    Returns
    -------
    FunctionalMapResult
        Refined functional map at higher spectral resolution.

    References
    ----------
    Melzi et al. (2019). ZoomOut: Spectral Upsampling for Efficient
    Shape Correspondence. ACM SIGGRAPH Asia.
    """
    C = fmap.C
    Phi_s = fmap.eigenvectors_source
    Phi_t = fmap.eigenvectors_target

    if k_start is None:
        k_start = C.shape[0]
    if k_final is None:
        k_final = min(Phi_s.shape[1], Phi_t.shape[1])

    k_schedule = np.linspace(k_start, k_final, n_iters + 1, dtype=int)

    for step, k in enumerate(k_schedule[1:], 1):
        # Step 1: Convert current C to pointwise
        current_fmap = FunctionalMapResult(
            C=C,
            eigenvectors_source=Phi_s[:, :C.shape[1]],
            eigenvectors_target=Phi_t[:, :C.shape[0]],
        )
        pmap = convert_to_pointwise_map(current_fmap, use_gpu=use_gpu)

        # Step 2: Recompute C at higher resolution from pointwise map
        # C_new = Φ_t[:, :k]^T  P  Φ_s[:, :k]
        # where P is the permutation matrix defined by pmap
        Phi_s_k = Phi_s[:, :k]                    # (N1, k)
        Phi_t_k = Phi_t[:, :k]                    # (N2, k)

        # P Φ_s = Φ_s[pmap, :], shape (N2, k)
        C = Phi_t_k.T @ Phi_s_k[pmap, :]          # (k, k)

        if step % max(1, n_iters // 4) == 0:
            logger.info(
                "  ZoomOut step %d/%d: k=%d, ‖C‖_F=%.4f",
                step, n_iters, k, np.linalg.norm(C, "fro"),
            )

    # Final pointwise map at full resolution
    final_fmap = FunctionalMapResult(
        C=C,
        eigenvectors_source=Phi_s[:, :k_final],
        eigenvectors_target=Phi_t[:, :k_final],
    )
    pmap = convert_to_pointwise_map(final_fmap, use_gpu=use_gpu)

    return FunctionalMapResult(
        C=C,
        pointwise_map=pmap,
        eigenvalues_source=fmap.eigenvalues_source,
        eigenvalues_target=fmap.eigenvalues_target,
        eigenvectors_source=fmap.eigenvectors_source,
        eigenvectors_target=fmap.eigenvectors_target,
    )
