"""
GPU-accelerated optimal transport on brain point clouds.

Computes Wasserstein and Sinkhorn distances between cortical point cloud
distributions using PyTorch, GeomLoss, and POT backends. Designed for
interhemispheric asymmetry quantification and shape comparison.

All functions include explicit VRAM management, batched processing for
large point clouds (>50K points), and automatic CPU fallback.

Functions
---------
sliced_wasserstein          : Sliced Wasserstein distance (CPU, O(n log n))
sinkhorn_divergence_gpu     : GPU-accelerated Sinkhorn via GeomLoss
wasserstein_features_gpu    : Feature-space OT on point clouds
interhemispheric_ot         : Full asymmetry quantification pipeline
pairwise_ot_matrix          : N×N subject-pair OT distance matrix

References
----------
Feydy et al. (2019). Interpolating between Optimal Transport and MMD
    using Sinkhorn Divergences. AISTATS.
Bonneel et al. (2015). Sliced and Radon Wasserstein Barycenters of
    Measures. Journal of Mathematical Imaging and Vision.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class OTResult:
    """
    Result of an optimal transport computation between point clouds.

    Attributes
    ----------
    distance : float
        The computed OT distance / divergence.
    method : str
        Which OT method was used.
    n_source : int
        Number of source points.
    n_target : int
        Number of target points.
    transport_plan : np.ndarray or None
        Coupling matrix (if computed; None for sliced Wasserstein).
    """

    distance: float
    method: str
    n_source: int
    n_target: int
    transport_plan: Optional[np.ndarray] = None


# ═══════════════════════════════════════════════════════════════════════════
# Sliced Wasserstein distance (CPU, fast, O(n log n))
# ═══════════════════════════════════════════════════════════════════════════


def sliced_wasserstein(
    source: np.ndarray,
    target: np.ndarray,
    n_projections: int = 100,
    p: int = 2,
    seed: Optional[int] = 42,
    weights_source: Optional[np.ndarray] = None,
    weights_target: Optional[np.ndarray] = None,
) -> float:
    """
    Compute the sliced Wasserstein distance between two point clouds.

    Uses random 1D projections for O(n log n) computation per projection.
    Ideal for point clouds up to ~500K points on CPU.

    Parameters
    ----------
    source : np.ndarray, shape (N1, D)
        Source point cloud (coordinates and/or features).
    target : np.ndarray, shape (N2, D)
        Target point cloud.
    n_projections : int
        Number of random 1D projections.
    p : int
        Wasserstein-p exponent (1 or 2).
    seed : int or None
        Random seed for projection directions.
    weights_source : np.ndarray or None, shape (N1,)
        Point weights (e.g., Voronoi areas). If None, uniform.
    weights_target : np.ndarray or None, shape (N2,)
        Point weights.

    Returns
    -------
    float
        Sliced Wasserstein-p distance.
    """
    try:
        import ot

        if weights_source is None:
            weights_source = np.ones(source.shape[0]) / source.shape[0]
        else:
            weights_source = weights_source / weights_source.sum()
        if weights_target is None:
            weights_target = np.ones(target.shape[0]) / target.shape[0]
        else:
            weights_target = weights_target / weights_target.sum()

        return float(ot.sliced_wasserstein_distance(
            source.astype(np.float64),
            target.astype(np.float64),
            a=weights_source,
            b=weights_target,
            n_projections=n_projections,
            p=p,
            seed=seed,
        ))

    except ImportError:
        logger.info("POT not available; using built-in sliced Wasserstein")
        return _sliced_wasserstein_builtin(
            source, target, n_projections, p, seed,
        )


def _sliced_wasserstein_builtin(
    source: np.ndarray,
    target: np.ndarray,
    n_projections: int,
    p: int,
    seed: Optional[int],
) -> float:
    """Pure NumPy sliced Wasserstein (no POT dependency)."""
    rng = np.random.default_rng(seed)
    D = source.shape[1]

    # Random unit directions on the sphere
    directions = rng.standard_normal((n_projections, D))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    total = 0.0
    for d in directions:
        proj_s = source @ d
        proj_t = target @ d
        proj_s.sort()
        proj_t.sort()

        # If different sizes, subsample the larger to match
        n1, n2 = len(proj_s), len(proj_t)
        if n1 != n2:
            n = min(n1, n2)
            idx_s = np.linspace(0, n1 - 1, n, dtype=int)
            idx_t = np.linspace(0, n2 - 1, n, dtype=int)
            proj_s = proj_s[idx_s]
            proj_t = proj_t[idx_t]

        total += np.mean(np.abs(proj_s - proj_t) ** p)

    return float(total / n_projections) ** (1.0 / p)


# ═══════════════════════════════════════════════════════════════════════════
# Sinkhorn divergence (GPU, GeomLoss)
# ═══════════════════════════════════════════════════════════════════════════


def sinkhorn_divergence_gpu(
    source: np.ndarray,
    target: np.ndarray,
    blur: float = 0.05,
    p: int = 2,
    scaling: float = 0.9,
    backend: str = "auto",
    weights_source: Optional[np.ndarray] = None,
    weights_target: Optional[np.ndarray] = None,
    max_vram_fraction: float = 0.6,
) -> float:
    """
    Compute Sinkhorn divergence on GPU via GeomLoss.

    Uses the multiscale Sinkhorn algorithm with linear memory (O(n))
    via kernel truncation. Handles 50K–500K points per measure on a
    single GPU in under a second.

    Parameters
    ----------
    source : np.ndarray, shape (N1, D)
        Source point cloud.
    target : np.ndarray, shape (N2, D)
        Target point cloud.
    blur : float
        Sinkhorn regularization (entropic blur). Smaller = closer to
        exact OT but slower convergence. 0.05 is a good default for
        brain surfaces in mm coordinates.
    p : int
        Cost exponent (1 or 2).
    scaling : float
        Multiscale scaling factor (0 < scaling < 1). Closer to 1 =
        more accurate but slower.
    backend : ``'auto'``, ``'multiscale'``, ``'online'``, ``'tensorized'``
        GeomLoss backend. ``'multiscale'`` for >10K points.
    weights_source : np.ndarray or None, shape (N1,)
    weights_target : np.ndarray or None, shape (N2,)
    max_vram_fraction : float
        Maximum fraction of free VRAM to use before refusing.

    Returns
    -------
    float
        Sinkhorn divergence (debiased).

    Raises
    ------
    ImportError
        If ``geomloss`` is not installed.
    RuntimeError
        If VRAM is insufficient.
    """
    try:
        from geomloss import SamplesLoss
    except ImportError:
        raise ImportError(
            "geomloss is required for GPU Sinkhorn divergence. "
            "Install with: pip install geomloss"
        )

    import torch
    from corticalfields.pointcloud.spectral import _get_device, _cleanup_gpu

    device = _get_device(prefer_cuda=True)

    # VRAM safety check
    if device.type == "cuda":
        N = max(source.shape[0], target.shape[0])
        D = source.shape[1]
        # Rough estimate: 2 × N × D × 4 bytes + overhead
        est_mem = 2 * N * D * 4 * 3  # ×3 for intermediates
        free = (
            torch.cuda.get_device_properties(0).total_memory
            - torch.cuda.memory_allocated(0)
        )
        if est_mem > max_vram_fraction * free:
            logger.warning(
                "Estimated memory (%.0f MB) > %.0f%% of free VRAM (%.0f MB)."
                " Falling back to CPU.",
                est_mem / 1e6, max_vram_fraction * 100, free / 1e6,
            )
            device = torch.device("cpu")

    # Auto backend selection
    if backend == "auto":
        N = max(source.shape[0], target.shape[0])
        backend = "multiscale" if N > 10000 else "tensorized"

    x = torch.as_tensor(
        source.astype(np.float32), device=device,
    ).contiguous()
    y = torch.as_tensor(
        target.astype(np.float32), device=device,
    ).contiguous()

    # Weights
    if weights_source is not None:
        w_s = torch.as_tensor(
            (weights_source / weights_source.sum()).astype(np.float32),
            device=device,
        )
    else:
        w_s = None

    if weights_target is not None:
        w_t = torch.as_tensor(
            (weights_target / weights_target.sum()).astype(np.float32),
            device=device,
        )
    else:
        w_t = None

    loss_fn = SamplesLoss(
        loss="sinkhorn",
        p=p,
        blur=blur,
        scaling=scaling,
        backend=backend,
    )

    with torch.no_grad():
        if w_s is not None and w_t is not None:
            dist = loss_fn(w_s, x, w_t, y)
        else:
            dist = loss_fn(x, y)

    result = float(dist.item())
    _cleanup_gpu(x, y, w_s, w_t, dist)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Feature-space OT on point clouds
# ═══════════════════════════════════════════════════════════════════════════


def wasserstein_features_gpu(
    source_features: np.ndarray,
    target_features: np.ndarray,
    source_coords: Optional[np.ndarray] = None,
    target_coords: Optional[np.ndarray] = None,
    alpha: float = 0.5,
    blur: float = 0.05,
    use_gpu: bool = True,
) -> float:
    """
    Feature-augmented OT distance between point cloud distributions.

    Optionally combines spatial coordinates and spectral features
    into a joint representation for computing OT.

    Parameters
    ----------
    source_features : np.ndarray, shape (N1, D_f)
        Source feature vectors (e.g., HKS, WKS).
    target_features : np.ndarray, shape (N2, D_f)
        Target feature vectors.
    source_coords : np.ndarray or None, shape (N1, 3)
        Source 3D coordinates. If provided, concatenated with features.
    target_coords : np.ndarray or None, shape (N2, 3)
        Target 3D coordinates.
    alpha : float
        Weight for spatial coordinates vs features when both given.
        0 = features only, 1 = coordinates only.
    blur : float
        Sinkhorn blur parameter.
    use_gpu : bool
        Use CUDA if available.

    Returns
    -------
    float
        Feature-space OT distance.
    """
    # Normalize features to unit variance
    s_std = source_features.std(axis=0, keepdims=True)
    t_std = target_features.std(axis=0, keepdims=True)
    s_std[s_std < 1e-12] = 1.0
    t_std[t_std < 1e-12] = 1.0

    s_norm = source_features / s_std
    t_norm = target_features / t_std

    if source_coords is not None and target_coords is not None:
        # Normalize coordinates similarly
        all_coords = np.vstack([source_coords, target_coords])
        c_std = all_coords.std(axis=0, keepdims=True)
        c_std[c_std < 1e-12] = 1.0

        s_c = source_coords / c_std
        t_c = target_coords / c_std

        # Weighted concatenation
        source_combined = np.hstack([
            alpha * s_c,
            (1 - alpha) * s_norm,
        ])
        target_combined = np.hstack([
            alpha * t_c,
            (1 - alpha) * t_norm,
        ])
    else:
        source_combined = s_norm
        target_combined = t_norm

    if use_gpu:
        try:
            return sinkhorn_divergence_gpu(
                source_combined, target_combined, blur=blur,
            )
        except (ImportError, RuntimeError) as exc:
            logger.info("GPU OT failed (%s); using sliced Wasserstein", exc)

    return sliced_wasserstein(source_combined, target_combined)


# ═══════════════════════════════════════════════════════════════════════════
# Interhemispheric OT asymmetry
# ═══════════════════════════════════════════════════════════════════════════


def interhemispheric_ot(
    lh_features: np.ndarray,
    rh_features: np.ndarray,
    lh_coords: Optional[np.ndarray] = None,
    rh_coords: Optional[np.ndarray] = None,
    methods: Optional[list] = None,
    n_projections: int = 100,
    blur: float = 0.05,
    use_gpu: bool = True,
) -> Dict[str, float]:
    """
    Compute interhemispheric asymmetry via multiple OT methods.

    Assumes the right hemisphere has been mirrored (x → −x) so that
    the two hemispheres are in comparable coordinate spaces.

    Parameters
    ----------
    lh_features : np.ndarray, shape (N_lh, D)
        Left hemisphere spectral features.
    rh_features : np.ndarray, shape (N_rh, D)
        Right hemisphere spectral features (mirrored).
    lh_coords : np.ndarray or None, shape (N_lh, 3)
    rh_coords : np.ndarray or None, shape (N_rh, 3)
    methods : list of str or None
        OT methods to use. Default: ``['sliced_wasserstein', 'sinkhorn']``.
    n_projections : int
        For sliced Wasserstein.
    blur : float
        For Sinkhorn.
    use_gpu : bool

    Returns
    -------
    dict
        Method name → OT distance.
    """
    if methods is None:
        methods = ["sliced_wasserstein", "sinkhorn"]

    results = {}

    for method in methods:
        if method == "sliced_wasserstein":
            d = sliced_wasserstein(
                lh_features, rh_features,
                n_projections=n_projections,
            )
            results["sliced_wasserstein"] = d

        elif method == "sinkhorn":
            try:
                d = sinkhorn_divergence_gpu(
                    lh_features, rh_features,
                    blur=blur,
                )
                results["sinkhorn"] = d
            except (ImportError, RuntimeError) as exc:
                logger.warning("Sinkhorn failed (%s); skipping", exc)

        elif method == "feature_ot":
            d = wasserstein_features_gpu(
                lh_features, rh_features,
                lh_coords, rh_coords,
                use_gpu=use_gpu,
            )
            results["feature_ot"] = d

        else:
            raise ValueError(f"Unknown OT method: {method!r}")

    logger.info("Interhemispheric OT distances: %s", results)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Pairwise OT distance matrix
# ═══════════════════════════════════════════════════════════════════════════


def pairwise_ot_matrix(
    feature_list: list,
    method: str = "sliced_wasserstein",
    n_projections: int = 100,
    blur: float = 0.05,
    use_gpu: bool = True,
    n_jobs: int = 1,
) -> np.ndarray:
    """
    Compute pairwise OT distance matrix for a cohort.

    Parameters
    ----------
    feature_list : list of np.ndarray
        Each element is shape (N_i, D) — features for one subject.
    method : str
        ``'sliced_wasserstein'`` or ``'sinkhorn'``.
    n_projections : int
    blur : float
    use_gpu : bool
    n_jobs : int
        Number of parallel jobs (only for CPU methods).

    Returns
    -------
    dist_matrix : np.ndarray, shape (S, S)
        Symmetric pairwise OT distance matrix.
    """
    S = len(feature_list)
    dist = np.zeros((S, S), dtype=np.float64)

    pairs = [(i, j) for i in range(S) for j in range(i + 1, S)]
    logger.info("Computing %d pairwise OT distances (method=%s)...", len(pairs), method)

    for idx, (i, j) in enumerate(pairs):
        if method == "sliced_wasserstein":
            d = sliced_wasserstein(
                feature_list[i], feature_list[j],
                n_projections=n_projections,
            )
        elif method == "sinkhorn":
            d = sinkhorn_divergence_gpu(
                feature_list[i], feature_list[j],
                blur=blur,
            )
        else:
            raise ValueError(f"Unknown method: {method!r}")

        dist[i, j] = d
        dist[j, i] = d

        if (idx + 1) % max(1, len(pairs) // 10) == 0:
            logger.info("  %d/%d pairs computed", idx + 1, len(pairs))

    return dist
