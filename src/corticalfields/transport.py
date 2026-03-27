"""
Optimal transport for cortical morphometry and asymmetry.

Provides scalable Wasserstein-type distances between cortical point clouds,
enabling atlas-free comparison of hemispheric geometry across subjects. Three
computational backends are supported, ranging from fast-and-approximate to
structure-preserving:

  1. **Sliced Wasserstein** (via POT) — O(N log N), CPU-friendly, provably
     PD kernel available. Best for pairwise distance matrices.
  2. **Sinkhorn divergence** (via GeomLoss/KeOps) — GPU-accelerated,
     linear memory, tunable spatial resolution via ``blur``. Best for
     subject-pair analysis at full resolution.
  3. **FUGW** (Fused Unbalanced Gromov-Wasserstein) — jointly matches
     anatomy + function, coarse-to-fine. Best for detailed vertex-level
     alignment, too slow for batch distance matrices.

The sliced Wasserstein Gaussian kernel is provably positive definite
(Kolouri et al., CVPR 2016), making it valid for kernel ridge regression,
HSIC, and SVM — connecting geometry directly to clinical outcomes.

Dependencies
------------
- ``pot`` — Python Optimal Transport (required for sliced Wasserstein)
- ``geomloss`` + ``pykeops`` — Sinkhorn on GPU (optional)
- ``fugw`` — Fused Unbalanced GW (optional)
- ``torch`` — for GPU-accelerated backends

References
----------
Kolouri, S., et al. (2016). Sliced Wasserstein Kernels for Probability
    Distributions. CVPR.
Feydy, J., et al. (2019). Interpolating between Optimal Transport and MMD
    using Sinkhorn Divergences. AISTATS.
Thual, A., et al. (2022). Aligning individual brains with Fused Unbalanced
    Gromov-Wasserstein. NeurIPS.
Gerber, S., et al. (2023). Unbalanced Transport Morphometry. Medical Image
    Analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class TransportResult:
    """
    Result of an optimal transport computation between two point sets.

    Parameters
    ----------
    distance : float
        The OT distance/divergence value.
    method : str
        Which OT method was used.
    metadata : dict
        Computation parameters and timing.
    transport_plan : np.ndarray or None
        Explicit transport plan (only for exact OT / FUGW).
    dual_potentials : tuple or None
        Dual (Kantorovich) potentials if available.
    """

    distance: float
    method: str
    metadata: Dict = field(default_factory=dict)
    transport_plan: Optional[np.ndarray] = None
    dual_potentials: Optional[Tuple[np.ndarray, np.ndarray]] = None


# ═══════════════════════════════════════════════════════════════════════════
# Sliced Wasserstein distance (CPU, scalable)
# ═══════════════════════════════════════════════════════════════════════════


def sliced_wasserstein_distance(
    X: np.ndarray,
    Y: np.ndarray,
    n_projections: int = 200,
    p: int = 2,
    weights_X: Optional[np.ndarray] = None,
    weights_Y: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> float:
    """
    Compute the sliced Wasserstein distance between two point clouds.

    Projects both clouds onto random 1D directions, computes exact
    1D Wasserstein on each projection, and averages. Runs in
    O(N log N × n_projections) time with negligible memory overhead.

    For 150k-point cortical surfaces with 200 projections, this takes
    approximately 1–5 seconds on CPU.

    Parameters
    ----------
    X : np.ndarray, shape (N, D)
        Source point cloud (e.g. left hemisphere coordinates + features).
    Y : np.ndarray, shape (M, D)
        Target point cloud (e.g. mirrored right hemisphere).
    n_projections : int
        Number of random 1D projections (200 is stable for 3D data).
    p : int
        Order of the Wasserstein distance (1 or 2).
    weights_X : np.ndarray or None, shape (N,)
        Point weights (e.g. area elements). Must sum to 1.
    weights_Y : np.ndarray or None, shape (M,)
        Point weights. Must sum to 1.
    seed : int or None
        Random seed for reproducible projections.

    Returns
    -------
    float
        Sliced Wasserstein distance SW_p(X, Y).

    Examples
    --------
    >>> lh_coords = ...  # (N, 3) left hemisphere pial coordinates
    >>> rh_mirrored = ...  # (M, 3) right hemisphere mirrored across YZ
    >>> d = sliced_wasserstein_distance(lh_coords, rh_mirrored)
    """
    try:
        import ot
    except ImportError:
        raise ImportError(
            "POT (Python Optimal Transport) is required. "
            "Install with: pip install POT"
        )

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    if weights_X is not None:
        weights_X = np.asarray(weights_X, dtype=np.float64)
        weights_X = weights_X / weights_X.sum()
    if weights_Y is not None:
        weights_Y = np.asarray(weights_Y, dtype=np.float64)
        weights_Y = weights_Y / weights_Y.sum()

    return float(ot.sliced_wasserstein_distance(
        X, Y,
        a=weights_X,
        b=weights_Y,
        n_projections=n_projections,
        p=p,
        seed=seed,
    ))


def sliced_wasserstein_with_features(
    coords_X: np.ndarray,
    coords_Y: np.ndarray,
    features_X: Optional[np.ndarray] = None,
    features_Y: Optional[np.ndarray] = None,
    feature_weight: float = 1.0,
    n_projections: int = 200,
    p: int = 2,
    weights_X: Optional[np.ndarray] = None,
    weights_Y: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> float:
    """
    Sliced Wasserstein distance with optional morphometric features.

    Concatenates 3D coordinates with per-vertex features (e.g. thickness,
    curvature) to compute OT distance in the joint geometry-feature space.
    Features are scaled by ``feature_weight`` relative to coordinates.

    Parameters
    ----------
    coords_X : np.ndarray, shape (N, 3)
        Source 3D coordinates.
    coords_Y : np.ndarray, shape (M, 3)
        Target 3D coordinates.
    features_X : np.ndarray or None, shape (N, F)
        Source morphometric features.
    features_Y : np.ndarray or None, shape (M, F)
        Target morphometric features.
    feature_weight : float
        Relative weight of features vs. coordinates. A weight of 1.0
        treats them equally; higher values emphasise morphometric
        differences over spatial displacement.
    n_projections : int
        Number of random projections.
    p : int
        Wasserstein order.
    weights_X, weights_Y : np.ndarray or None
        Point weights.
    seed : int or None
        Random seed.

    Returns
    -------
    float
        Sliced Wasserstein distance in the joint space.
    """
    X = np.asarray(coords_X, dtype=np.float64)
    Y = np.asarray(coords_Y, dtype=np.float64)

    if features_X is not None and features_Y is not None:
        fX = np.asarray(features_X, dtype=np.float64)
        fY = np.asarray(features_Y, dtype=np.float64)
        if fX.ndim == 1:
            fX = fX[:, np.newaxis]
        if fY.ndim == 1:
            fY = fY[:, np.newaxis]

        # Scale features relative to coordinate range
        coord_scale = max(
            X.std(axis=0).mean(), Y.std(axis=0).mean(), 1e-8,
        )
        feat_scale = max(
            fX.std(axis=0).mean(), fY.std(axis=0).mean(), 1e-8,
        )
        scaling = feature_weight * (coord_scale / feat_scale)

        X = np.hstack([X, fX * scaling])
        Y = np.hstack([Y, fY * scaling])

    return sliced_wasserstein_distance(
        X, Y,
        n_projections=n_projections,
        p=p,
        weights_X=weights_X,
        weights_Y=weights_Y,
        seed=seed,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Sinkhorn divergence (GPU, high resolution)
# ═══════════════════════════════════════════════════════════════════════════


def sinkhorn_divergence(
    X: np.ndarray,
    Y: np.ndarray,
    blur: float = 5.0,
    p: int = 2,
    scaling: float = 0.5,
    reach: Optional[float] = None,
    weights_X: Optional[np.ndarray] = None,
    weights_Y: Optional[np.ndarray] = None,
    backend: str = "online",
    device: str = "auto",
) -> TransportResult:
    """
    Compute the Sinkhorn divergence between two point clouds using GeomLoss.

    Uses the KeOps backend for GPU computation with **linear memory**,
    enabling 150k+ point clouds on a single GPU. The ``blur`` parameter
    controls spatial resolution in millimetres.

    Parameters
    ----------
    X : np.ndarray, shape (N, D)
        Source point cloud.
    Y : np.ndarray, shape (M, D)
        Target point cloud.
    blur : float
        Spatial resolution in mm. 5 mm resolves gyral-scale structure;
        1 mm captures sulcal detail but requires more iterations.
        Default 0.05 in GeomLoss assumes data in [0,1]; for cortical
        surfaces in mm, use 1–10.
    p : int
        Wasserstein order (1 or 2).
    scaling : float
        Epsilon-scaling ratio (0.5 = default speed/accuracy).
    reach : float or None
        Marginal relaxation for unbalanced OT. None = balanced.
        Set ~20 mm for cortical surfaces with unequal point counts.
    weights_X, weights_Y : np.ndarray or None
        Point weights.
    backend : ``'online'``, ``'multiscale'``, or ``'tensorized'``
        GeomLoss backend. ``'online'`` uses linear memory (recommended).
        ``'tensorized'`` materialises N×M cost matrix (OOM for >10k pts).
    device : str
        ``'auto'``, ``'cuda'``, or ``'cpu'``.

    Returns
    -------
    TransportResult
        Sinkhorn divergence value and metadata.
    """
    try:
        import torch
        from geomloss import SamplesLoss
    except ImportError:
        raise ImportError(
            "GeomLoss is required for Sinkhorn divergence. "
            "Install with: pip install geomloss pykeops"
        )

    # Resolve device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    logger.info(
        "Sinkhorn divergence: %d vs %d points, blur=%.1f mm, backend=%s, device=%s",
        X.shape[0], Y.shape[0], blur, backend, device,
    )

    x = torch.tensor(X, dtype=torch.float32, device=dev)
    y = torch.tensor(Y, dtype=torch.float32, device=dev)

    loss_fn = SamplesLoss(
        loss="sinkhorn",
        p=p,
        blur=blur,
        scaling=scaling,
        reach=reach,
        debias=True,
        backend=backend,
    )

    if weights_X is not None and weights_Y is not None:
        alpha = torch.tensor(weights_X, dtype=torch.float32, device=dev)
        beta = torch.tensor(weights_Y, dtype=torch.float32, device=dev)
        value = loss_fn(alpha, x, beta, y)
    else:
        value = loss_fn(x, y)

    return TransportResult(
        distance=float(value.item()),
        method="sinkhorn",
        metadata={
            "blur": blur,
            "p": p,
            "scaling": scaling,
            "reach": reach,
            "backend": backend,
            "device": device,
            "n_source": X.shape[0],
            "n_target": Y.shape[0],
        },
    )


# ═══════════════════════════════════════════════════════════════════════════
# Pairwise distance matrices
# ═══════════════════════════════════════════════════════════════════════════


def pairwise_wasserstein_matrix(
    point_clouds: List[np.ndarray],
    method: str = "sliced",
    n_projections: int = 200,
    p: int = 2,
    weights: Optional[List[np.ndarray]] = None,
    seed: int = 42,
    n_jobs: int = 1,
    **kwargs,
) -> np.ndarray:
    """
    Compute a pairwise Wasserstein distance matrix across a cohort.

    For N subjects, computes the N(N-1)/2 pairwise distances and returns
    a symmetric distance matrix. For N=46 with 150k points, this takes
    ~1–2 hours on CPU using sliced Wasserstein.

    Parameters
    ----------
    point_clouds : list of np.ndarray
        List of point clouds, each shape (N_i, D).
    method : ``'sliced'`` or ``'sinkhorn'``
        OT method. ``'sliced'`` is recommended for batch computation.
    n_projections : int
        Number of random projections (for sliced Wasserstein).
    p : int
        Wasserstein order.
    weights : list of np.ndarray or None
        Per-point weights for each cloud.
    seed : int
        Random seed.
    n_jobs : int
        Number of parallel workers (requires joblib). 1 = sequential.
    **kwargs
        Additional arguments passed to the distance function.

    Returns
    -------
    D : np.ndarray, shape (N, N)
        Symmetric distance matrix.
    """
    N = len(point_clouds)
    D = np.zeros((N, N), dtype=np.float64)

    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    logger.info(
        "Computing %d pairwise %s Wasserstein distances...",
        len(pairs), method,
    )

    def _compute_pair(pair):
        i, j = pair
        w_i = weights[i] if weights else None
        w_j = weights[j] if weights else None

        if method == "sliced":
            d = sliced_wasserstein_distance(
                point_clouds[i], point_clouds[j],
                n_projections=n_projections, p=p,
                weights_X=w_i, weights_Y=w_j,
                seed=seed,
            )
        elif method == "sinkhorn":
            result = sinkhorn_divergence(
                point_clouds[i], point_clouds[j],
                p=p, weights_X=w_i, weights_Y=w_j,
                **kwargs,
            )
            d = result.distance
        else:
            raise ValueError(f"Unknown method: {method!r}")
        return i, j, d

    if n_jobs == 1:
        for idx, pair in enumerate(pairs):
            i, j, d = _compute_pair(pair)
            D[i, j] = d
            D[j, i] = d
            if (idx + 1) % 100 == 0:
                logger.info("  %d/%d pairs computed", idx + 1, len(pairs))
    else:
        try:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=n_jobs)(
                delayed(_compute_pair)(pair) for pair in pairs
            )
            for i, j, d in results:
                D[i, j] = d
                D[j, i] = d
        except ImportError:
            logger.warning("joblib not available; falling back to sequential")
            for pair in pairs:
                i, j, d = _compute_pair(pair)
                D[i, j] = d
                D[j, i] = d

    return D


def interhemispheric_wasserstein_distances(
    lh_clouds: List[np.ndarray],
    rh_clouds: List[np.ndarray],
    method: str = "sliced",
    mirror_rh: bool = True,
    n_projections: int = 200,
    p: int = 2,
    seed: int = 42,
    **kwargs,
) -> np.ndarray:
    """
    Compute per-subject inter-hemispheric Wasserstein distances.

    For each subject, computes the OT distance between the left
    hemisphere point cloud and the (mirrored) right hemisphere.

    Parameters
    ----------
    lh_clouds : list of np.ndarray
        Left hemisphere point clouds, each shape (N_i, D).
    rh_clouds : list of np.ndarray
        Right hemisphere point clouds, each shape (M_i, D).
    method : ``'sliced'`` or ``'sinkhorn'``
        OT method.
    mirror_rh : bool
        If True, mirror the RH coordinates across YZ (x → −x)
        before computing the distance.
    n_projections : int
        Number of projections (sliced Wasserstein).
    p : int
        Wasserstein order.
    seed : int
        Random seed.
    **kwargs
        Additional arguments.

    Returns
    -------
    distances : np.ndarray, shape (N_subjects,)
        Per-subject inter-hemispheric distance.
    """
    assert len(lh_clouds) == len(rh_clouds), \
        "Number of LH and RH point clouds must match"

    N = len(lh_clouds)
    distances = np.zeros(N, dtype=np.float64)

    for i in range(N):
        rh = rh_clouds[i].copy()
        if mirror_rh:
            rh[:, 0] *= -1  # Mirror x-coordinate

        if method == "sliced":
            distances[i] = sliced_wasserstein_distance(
                lh_clouds[i], rh,
                n_projections=n_projections, p=p, seed=seed,
            )
        elif method == "sinkhorn":
            result = sinkhorn_divergence(
                lh_clouds[i], rh, p=p, **kwargs,
            )
            distances[i] = result.distance
        else:
            raise ValueError(f"Unknown method: {method!r}")

        if (i + 1) % 10 == 0:
            logger.info("  %d/%d subjects processed", i + 1, N)

    return distances


# ═══════════════════════════════════════════════════════════════════════════
# Wasserstein kernels
# ═══════════════════════════════════════════════════════════════════════════


def wasserstein_kernel(
    D: np.ndarray,
    gamma: Optional[float] = None,
    kernel_type: str = "gaussian",
    backend: str = "numpy",
) -> np.ndarray:
    """
    Compute a positive-definite kernel from a Wasserstein distance matrix.

    The Gaussian kernel K(x,y) = exp(−γ · D²(x,y)) on sliced Wasserstein
    distances is provably positive definite for all γ > 0, because SW²
    is a conditionally negative definite (Hilbertian) metric (Kolouri
    et al., CVPR 2016).

    Parameters
    ----------
    D : np.ndarray, shape (N, N)
        Pairwise Wasserstein distance matrix.
    gamma : float or None
        Kernel bandwidth. If None, uses the median heuristic:
        γ = 1 / median(D²).
    kernel_type : ``'gaussian'`` or ``'laplacian'``
        Kernel function.
    backend : ``'numpy'``, ``'torch'``, or ``'cupy'``
        Compute backend. GPU backends are faster for large N.

    Returns
    -------
    K : np.ndarray, shape (N, N)
        Symmetric positive semi-definite kernel matrix.
    """
    D = np.asarray(D, dtype=np.float64)

    if gamma is None:
        D_flat = D[np.triu_indices_from(D, k=1)]
        if len(D_flat) > 0 and D_flat.max() > 0:
            if kernel_type == "gaussian":
                gamma = 1.0 / max(np.median(D_flat ** 2), 1e-12)
            else:
                gamma = 1.0 / max(np.median(D_flat), 1e-12)
        else:
            gamma = 1.0

    if backend == "torch":
        return _wasserstein_kernel_torch(D, gamma, kernel_type)
    elif backend == "cupy":
        return _wasserstein_kernel_cupy(D, gamma, kernel_type)

    # NumPy default
    if kernel_type == "gaussian":
        K = np.exp(-gamma * D ** 2)
    elif kernel_type == "laplacian":
        K = np.exp(-gamma * D)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type!r}")

    return K


def functional_map_kernel(
    functional_maps: List["FunctionalMap"],
    gamma: Optional[float] = None,
    metric: str = "frobenius",
) -> np.ndarray:
    """
    Build a kernel matrix from functional map C matrices.

    Computes pairwise distances between C matrices and converts to a
    PD kernel via the Gaussian RBF.

    Parameters
    ----------
    functional_maps : list of FunctionalMap
        One per subject.
    gamma : float or None
        Kernel bandwidth (median heuristic if None).
    metric : ``'frobenius'`` or ``'geodesic'``
        Distance metric on C matrices.

    Returns
    -------
    K : np.ndarray, shape (N, N)
        Kernel matrix.
    """
    from corticalfields.functional_maps import functional_map_distance

    N = len(functional_maps)
    D = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(i + 1, N):
            d = functional_map_distance(functional_maps[i], functional_maps[j], metric)
            D[i, j] = d
            D[j, i] = d

    return wasserstein_kernel(D, gamma=gamma, kernel_type="gaussian")


# ═══════════════════════════════════════════════════════════════════════════
# FUGW (optional, for detailed alignment)
# ═══════════════════════════════════════════════════════════════════════════


def fugw_alignment(
    source_features: np.ndarray,
    target_features: np.ndarray,
    source_geometry: np.ndarray,
    target_geometry: np.ndarray,
    alpha: float = 0.5,
    rho: float = 1e-2,
    eps: float = 1e-4,
    n_landmarks: int = 1000,
    seed: int = 42,
) -> TransportResult:
    """
    Fused Unbalanced Gromov-Wasserstein alignment (Thual et al., NeurIPS 2022).

    Jointly matches functional features and anatomical geometry between
    two surfaces. Uses coarse-to-fine strategy with landmark sampling.

    .. warning::

        FUGW is **not suitable for pairwise distance matrices** (10 min/pair
        × 1000 pairs = ~7 GPU-days). Use for detailed vertex-level alignment
        on specific subject pairs only.

    Parameters
    ----------
    source_features : np.ndarray, shape (n_contrasts, N_src)
        Functional features on source (e.g. thickness, curvature).
    target_features : np.ndarray, shape (n_contrasts, N_tgt)
        Functional features on target.
    source_geometry : np.ndarray, shape (N_src, K)
        Geometry embeddings (e.g. first K LMD eigenvectors).
    target_geometry : np.ndarray, shape (N_tgt, K)
        Geometry embeddings.
    alpha : float
        Balance: 0 = pure feature Wasserstein, 1 = pure geometry GW.
    rho : float
        Marginal relaxation (unbalanced parameter).
    eps : float
        Entropic regularisation.
    n_landmarks : int
        Number of landmark vertices for coarse mapping.
    seed : int
        Random seed for landmark selection.

    Returns
    -------
    TransportResult
        FUGW distance and transport plan.
    """
    try:
        import torch
        from fugw.mappings import FUGWSparse
    except ImportError:
        raise ImportError(
            "FUGW is required for this function. "
            "Install with: pip install fugw"
        )

    logger.info(
        "FUGW alignment: %d → %d vertices, alpha=%.2f, %d landmarks",
        source_geometry.shape[0], target_geometry.shape[0], alpha, n_landmarks,
    )

    # Select landmarks via farthest-point sampling
    from corticalfields.pointcloud import _farthest_point_sampling
    src_idx = _farthest_point_sampling(source_geometry, n_landmarks, seed=seed)
    tgt_idx = _farthest_point_sampling(target_geometry, n_landmarks, seed=seed)

    # Compute radius for local neighborhoods
    from scipy.spatial import cKDTree
    src_tree = cKDTree(source_geometry)
    dists_src, _ = src_tree.query(source_geometry[src_idx], k=2)
    radius = float(np.median(dists_src[:, 1]) * 5)

    mapping = FUGWSparse(alpha=alpha, rho=rho, eps=eps)

    sf = torch.tensor(source_features, dtype=torch.float32)
    tf = torch.tensor(target_features, dtype=torch.float32)
    sg = torch.tensor(source_geometry, dtype=torch.float32)
    tg = torch.tensor(target_geometry, dtype=torch.float32)

    mapping.fit(
        source_features=sf,
        target_features=tf,
        source_geometry_embeddings=sg,
        target_geometry_embeddings=tg,
        solver="sinkhorn",
        solver_params={"nits_bcd": 5, "tol_uot": 1e-10},
        source_sample=src_idx,
        target_sample=tgt_idx,
        source_selection_radius=radius,
        target_selection_radius=radius,
    )

    return TransportResult(
        distance=float(mapping.loss),
        method="fugw",
        metadata={
            "alpha": alpha,
            "rho": rho,
            "eps": eps,
            "n_landmarks": n_landmarks,
            "radius": radius,
        },
    )


# ═══════════════════════════════════════════════════════════════════════════
# GPU-accelerated kernel computation
# ═══════════════════════════════════════════════════════════════════════════


def _wasserstein_kernel_torch(
    D: np.ndarray, gamma: float, kernel_type: str,
) -> np.ndarray:
    """Compute Wasserstein kernel on GPU via PyTorch (element-wise exp)."""
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D_t = torch.tensor(D, dtype=torch.float64, device=device)
    if kernel_type == "gaussian":
        K_t = torch.exp(-gamma * D_t ** 2)
    elif kernel_type == "laplacian":
        K_t = torch.exp(-gamma * D_t)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type!r}")
    return K_t.cpu().numpy()


def _wasserstein_kernel_cupy(
    D: np.ndarray, gamma: float, kernel_type: str,
) -> np.ndarray:
    """Compute Wasserstein kernel on GPU via CuPy."""
    import cupy as cp
    D_c = cp.asarray(D)
    if kernel_type == "gaussian":
        K_c = cp.exp(-gamma * D_c ** 2)
    elif kernel_type == "laplacian":
        K_c = cp.exp(-gamma * D_c)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type!r}")
    return cp.asnumpy(K_c)
