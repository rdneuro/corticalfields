"""
Point cloud registration for brain surfaces.

Provides rigid and non-rigid registration methods for aligning cortical
point clouds across subjects or between hemispheres. Includes classical
ICP, probabilistic (CPD/BCPD), and spectral correspondence approaches
with GPU acceleration and VRAM-safe batching.

Functions
---------
icp_registration         : Iterative Closest Point (rigid/affine)
cpd_registration         : Coherent Point Drift (non-rigid)
spectral_registration    : Eigenbasis alignment via functional maps
procrustes_alignment     : Rigid Procrustes alignment
compute_registration_error : Registration quality metrics

References
----------
Besl & McKay (1992). A Method for Registration of 3-D Shapes. IEEE TPAMI.
Myronenko & Song (2010). Point Set Registration: Coherent Point Drift.
    IEEE TPAMI.
Hirose (2021). A Bayesian Formulation of Coherent Point Drift. IEEE TPAMI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class RegistrationResult:
    """
    Result of a point cloud registration.

    Attributes
    ----------
    transformed : np.ndarray, shape (N, 3)
        Registered (transformed) source points.
    rotation : np.ndarray or None, shape (3, 3)
        Rotation matrix (rigid/affine only).
    translation : np.ndarray or None, shape (3,)
        Translation vector (rigid/affine only).
    scale : float
        Uniform scale factor (1.0 if no scaling).
    correspondence : np.ndarray or None, shape (N,)
        Per-point correspondence to target (indices).
    rmse : float
        Root mean square error after alignment.
    n_iterations : int
        Number of iterations used.
    method : str
        Registration method name.
    """

    transformed: np.ndarray
    rotation: Optional[np.ndarray] = None
    translation: Optional[np.ndarray] = None
    scale: float = 1.0
    correspondence: Optional[np.ndarray] = None
    rmse: float = float("inf")
    n_iterations: int = 0
    method: str = "unknown"


# ═══════════════════════════════════════════════════════════════════════════
# Procrustes alignment (rigid)
# ═══════════════════════════════════════════════════════════════════════════


def procrustes_alignment(
    source: np.ndarray,
    target: np.ndarray,
    allow_scale: bool = False,
) -> RegistrationResult:
    """
    Rigid Procrustes alignment of source onto target.

    Finds R, t (and optionally s) minimising ‖s R @ source.T + t − target.T‖.

    Parameters
    ----------
    source : np.ndarray, shape (N, 3)
        Source point cloud. Must have same N as target.
    target : np.ndarray, shape (N, 3)
        Target point cloud.
    allow_scale : bool
        If True, also optimise a uniform scale factor.

    Returns
    -------
    RegistrationResult
        Aligned source, rotation, translation, scale.
    """
    assert source.shape == target.shape, (
        f"Shape mismatch: source {source.shape} vs target {target.shape}"
    )

    # Center both
    mu_s = source.mean(axis=0)
    mu_t = target.mean(axis=0)
    S = source - mu_s
    T = target - mu_t

    # Cross-covariance
    H = S.T @ T  # (3, 3)
    U, Sigma, Vt = np.linalg.svd(H)

    # Optimal rotation (handle reflection)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, np.sign(d)])
    R = Vt.T @ D @ U.T

    # Scale
    if allow_scale:
        s = np.sum(Sigma * np.diag(D)) / np.sum(S ** 2)
    else:
        s = 1.0

    # Translation
    t = mu_t - s * R @ mu_s

    transformed = s * (source @ R.T) + t
    rmse = np.sqrt(np.mean(np.sum((transformed - target) ** 2, axis=1)))

    logger.info(
        "Procrustes: RMSE=%.4f mm, scale=%.4f",
        rmse, s,
    )
    return RegistrationResult(
        transformed=transformed,
        rotation=R,
        translation=t,
        scale=s,
        rmse=rmse,
        n_iterations=1,
        method="procrustes",
    )


# ═══════════════════════════════════════════════════════════════════════════
# ICP registration
# ═══════════════════════════════════════════════════════════════════════════


def icp_registration(
    source: np.ndarray,
    target: np.ndarray,
    max_iterations: int = 50,
    tolerance: float = 1e-5,
    max_distance: Optional[float] = None,
    use_open3d: bool = True,
) -> RegistrationResult:
    """
    Iterative Closest Point registration.

    Parameters
    ----------
    source : np.ndarray, shape (N, 3)
    target : np.ndarray, shape (M, 3)
    max_iterations : int
    tolerance : float
        Convergence threshold on RMSE change.
    max_distance : float or None
        Maximum correspondence distance (mm). Rejects outlier pairs.
        Default: 10 × median nearest-neighbor distance.
    use_open3d : bool
        Use Open3D's optimised ICP if available.

    Returns
    -------
    RegistrationResult
    """
    if use_open3d:
        try:
            return _icp_open3d(source, target, max_iterations, tolerance, max_distance)
        except ImportError:
            logger.info("Open3D not available; using built-in ICP")

    return _icp_builtin(source, target, max_iterations, tolerance, max_distance)


def _icp_open3d(
    source: np.ndarray,
    target: np.ndarray,
    max_iterations: int,
    tolerance: float,
    max_distance: Optional[float],
) -> RegistrationResult:
    """ICP via Open3D (optimised C++ backend)."""
    import open3d as o3d

    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(source.astype(np.float64))
    tgt_pcd = o3d.geometry.PointCloud()
    tgt_pcd.points = o3d.utility.Vector3dVector(target.astype(np.float64))

    if max_distance is None:
        # Estimate from median NN distance
        tree = o3d.geometry.KDTreeFlann(tgt_pcd)
        dists = []
        for i in range(min(1000, len(source))):
            _, _, d = tree.search_knn_vector_3d(src_pcd.points[i], 1)
            dists.append(d[0])
        max_distance = 10.0 * np.sqrt(np.median(dists))

    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=max_iterations,
        relative_fitness=tolerance,
        relative_rmse=tolerance,
    )

    result = o3d.pipelines.registration.registration_icp(
        src_pcd, tgt_pcd,
        max_correspondence_distance=max_distance,
        criteria=criteria,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )

    T = np.asarray(result.transformation)
    R = T[:3, :3]
    t = T[:3, 3]
    transformed = (source @ R.T) + t
    rmse = result.inlier_rmse

    logger.info(
        "ICP (Open3D): RMSE=%.4f mm, fitness=%.4f, %d inliers",
        rmse, result.fitness,
        len(result.correspondence_set),
    )

    # Extract correspondence
    corr = np.asarray(result.correspondence_set)
    correspondence = np.full(source.shape[0], -1, dtype=np.int64)
    if len(corr) > 0:
        correspondence[corr[:, 0]] = corr[:, 1]

    return RegistrationResult(
        transformed=transformed,
        rotation=R,
        translation=t,
        correspondence=correspondence,
        rmse=rmse,
        n_iterations=max_iterations,
        method="icp_open3d",
    )


def _icp_builtin(
    source: np.ndarray,
    target: np.ndarray,
    max_iterations: int,
    tolerance: float,
    max_distance: Optional[float],
) -> RegistrationResult:
    """Pure NumPy/SciPy ICP fallback."""
    from scipy.spatial import cKDTree

    tree = cKDTree(target)

    if max_distance is None:
        dists, _ = tree.query(source[:min(1000, len(source))])
        max_distance = 10.0 * np.median(dists)

    current = source.copy()
    prev_rmse = float("inf")

    for it in range(max_iterations):
        dists, idx = tree.query(current)

        # Reject outlier pairs
        mask = dists < max_distance
        if mask.sum() < 10:
            logger.warning("ICP: too few inliers (%d); stopping", mask.sum())
            break

        src_matched = current[mask]
        tgt_matched = target[idx[mask]]

        # Solve for rigid transform
        result = procrustes_alignment(src_matched, tgt_matched)
        R, t = result.rotation, result.translation

        current = (current @ R.T) + t
        rmse = np.sqrt(np.mean(dists[mask] ** 2))

        if abs(prev_rmse - rmse) < tolerance:
            logger.info("ICP converged at iteration %d, RMSE=%.4f", it + 1, rmse)
            break
        prev_rmse = rmse

    # Final correspondence
    dists, idx = tree.query(current)

    logger.info("ICP (built-in): RMSE=%.4f mm after %d iterations", rmse, it + 1)

    return RegistrationResult(
        transformed=current,
        rotation=R,
        translation=t,
        correspondence=idx,
        rmse=rmse,
        n_iterations=it + 1,
        method="icp_builtin",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Coherent Point Drift (non-rigid)
# ═══════════════════════════════════════════════════════════════════════════


def cpd_registration(
    source: np.ndarray,
    target: np.ndarray,
    registration_type: str = "nonrigid",
    max_iterations: int = 100,
    tolerance: float = 1e-5,
    w: float = 0.0,
    beta: float = 2.0,
    lmbda: float = 2.0,
) -> RegistrationResult:
    """
    Coherent Point Drift registration.

    Uses the pycpd library for Gaussian mixture model-based probabilistic
    registration. Supports rigid, affine, and non-rigid deformations.

    Parameters
    ----------
    source : np.ndarray, shape (N, 3)
    target : np.ndarray, shape (M, 3)
    registration_type : ``'rigid'``, ``'affine'``, or ``'nonrigid'``
    max_iterations : int
    tolerance : float
        EM convergence threshold.
    w : float
        Noise weight (0 = no noise, 1 = all noise). Typically 0.0–0.1.
    beta : float
        Width of Gaussian kernel (non-rigid only). Larger = smoother.
    lmbda : float
        Regularization weight (non-rigid only). Larger = more rigid.

    Returns
    -------
    RegistrationResult
    """
    try:
        from pycpd import RigidRegistration, AffineRegistration, DeformableRegistration
    except ImportError:
        raise ImportError(
            "pycpd is required for CPD registration. "
            "Install with: pip install pycpd"
        )

    kwargs = dict(
        X=target.astype(np.float64),
        Y=source.astype(np.float64),
        max_iterations=max_iterations,
        tolerance=tolerance,
        w=w,
    )

    if registration_type == "rigid":
        reg = RigidRegistration(**kwargs)
    elif registration_type == "affine":
        reg = AffineRegistration(**kwargs)
    elif registration_type == "nonrigid":
        reg = DeformableRegistration(**kwargs, beta=beta, alpha=lmbda)
    else:
        raise ValueError(
            f"Unknown registration_type: {registration_type!r}. "
            "Use 'rigid', 'affine', or 'nonrigid'."
        )

    transformed, params = reg.register()

    from scipy.spatial import cKDTree
    tree = cKDTree(target)
    dists, idx = tree.query(transformed)
    rmse = np.sqrt(np.mean(dists ** 2))

    R = params.get("R") if isinstance(params, dict) else getattr(params, "R", None)
    t = params.get("t") if isinstance(params, dict) else getattr(params, "t", None)
    s = params.get("s") if isinstance(params, dict) else getattr(params, "s", 1.0)

    logger.info(
        "CPD (%s): RMSE=%.4f mm after registration",
        registration_type, rmse,
    )

    return RegistrationResult(
        transformed=transformed,
        rotation=R,
        translation=t.flatten() if t is not None else None,
        scale=float(s) if s is not None else 1.0,
        correspondence=idx,
        rmse=rmse,
        n_iterations=max_iterations,
        method=f"cpd_{registration_type}",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Spectral registration via functional maps
# ═══════════════════════════════════════════════════════════════════════════


def spectral_registration(
    eigenvalues_source: np.ndarray,
    eigenvectors_source: np.ndarray,
    eigenvalues_target: np.ndarray,
    eigenvectors_target: np.ndarray,
    source_points: np.ndarray,
    target_points: np.ndarray,
    n_basis: int = 50,
    zoomout_iters: int = 10,
    use_gpu: bool = True,
) -> RegistrationResult:
    """
    Point cloud registration via spectral functional maps.

    Computes a functional map between the two surfaces, converts it
    to a pointwise correspondence, then uses Procrustes to find the
    rigid alignment. This naturally handles the atlas-free interhemispheric
    correspondence problem in CorticalFields.

    Parameters
    ----------
    eigenvalues_source : np.ndarray, shape (K1,)
    eigenvectors_source : np.ndarray, shape (N1, K1)
    eigenvalues_target : np.ndarray, shape (K2,)
    eigenvectors_target : np.ndarray, shape (N2, K2)
    source_points : np.ndarray, shape (N1, 3)
    target_points : np.ndarray, shape (N2, 3)
    n_basis : int
    zoomout_iters : int
    use_gpu : bool

    Returns
    -------
    RegistrationResult
    """
    from corticalfields.pointcloud.functional_maps import (
        compute_descriptors,
        compute_functional_map,
        zoomout_refinement,
        convert_to_pointwise_map,
    )

    logger.info(
        "Spectral registration: %d → %d points, k=%d",
        source_points.shape[0], target_points.shape[0], n_basis,
    )

    # Compute descriptors
    desc_s = compute_descriptors(eigenvalues_source, eigenvectors_source)
    desc_t = compute_descriptors(eigenvalues_target, eigenvectors_target)

    # Functional map
    fmap = compute_functional_map(
        eigenvectors_source, eigenvectors_target,
        desc_s, desc_t,
        eigenvalues_source, eigenvalues_target,
        n_basis=n_basis,
    )

    # ZoomOut refinement
    if zoomout_iters > 0:
        fmap = zoomout_refinement(fmap, n_iters=zoomout_iters, use_gpu=use_gpu)

    pmap = fmap.pointwise_map
    if pmap is None:
        pmap = convert_to_pointwise_map(fmap, use_gpu=use_gpu)

    # Procrustes on corresponding points
    source_matched = source_points[pmap]  # (N_target, 3)
    result = procrustes_alignment(source_matched, target_points)

    # Apply transform to ALL source points
    transformed = result.scale * (source_points @ result.rotation.T) + result.translation

    from scipy.spatial import cKDTree
    tree = cKDTree(target_points)
    dists, _ = tree.query(transformed)
    rmse = np.sqrt(np.mean(dists ** 2))

    logger.info("Spectral registration: final RMSE=%.4f mm", rmse)

    return RegistrationResult(
        transformed=transformed,
        rotation=result.rotation,
        translation=result.translation,
        scale=result.scale,
        correspondence=pmap,
        rmse=rmse,
        n_iterations=zoomout_iters,
        method="spectral_fmap",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Registration quality metrics
# ═══════════════════════════════════════════════════════════════════════════


def compute_registration_error(
    source_registered: np.ndarray,
    target: np.ndarray,
    percentiles: Optional[list] = None,
) -> Dict[str, float]:
    """
    Compute registration quality metrics.

    Parameters
    ----------
    source_registered : np.ndarray, shape (N, 3)
        Registered source points.
    target : np.ndarray, shape (M, 3)
        Target points.
    percentiles : list of float or None
        Distance percentiles to report. Default: [50, 90, 95, 99].

    Returns
    -------
    dict
        ``'rmse'``, ``'mean_distance'``, ``'hausdorff'``,
        ``'p50'``, ``'p90'``, ``'p95'``, ``'p99'``.
    """
    from scipy.spatial import cKDTree

    if percentiles is None:
        percentiles = [50, 90, 95, 99]

    tree = cKDTree(target)
    dists_s2t, _ = tree.query(source_registered)

    tree_s = cKDTree(source_registered)
    dists_t2s, _ = tree_s.query(target)

    # Symmetric distances
    all_dists = np.concatenate([dists_s2t, dists_t2s])

    result = {
        "rmse": float(np.sqrt(np.mean(dists_s2t ** 2))),
        "mean_distance": float(all_dists.mean()),
        "hausdorff": float(max(dists_s2t.max(), dists_t2s.max())),
        "assd": float(all_dists.mean()),  # Average Symmetric Surface Distance
    }

    for p in percentiles:
        result[f"p{int(p)}"] = float(np.percentile(all_dists, p))

    return result
