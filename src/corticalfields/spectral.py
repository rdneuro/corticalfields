"""
Laplace–Beltrami spectral decomposition and shape descriptors.

This module is the mathematical core of CorticalFields. It computes:

    1. The discrete Laplace–Beltrami (LB) operator on a triangle mesh
       via the cotangent-weight scheme (Meyer et al. 2003), or via
       ``robust-laplacian`` (Sharp & Crane, SGP 2020) when available.

    2. The leading eigenpairs (λ_i, φ_i) of the generalised eigenvalue
       problem  L φ = λ M φ ,  where L is the stiffness (cotangent)
       matrix and M is the lumped mass (area) matrix.

    3. Spectral shape descriptors computed from the eigenpairs:
       • Heat Kernel Signature   — HKS(x, t)
       • Wave Kernel Signature   — WKS(x, e)
       • Global Point Signature  — GPS(x)

GPU acceleration
----------------
All compute-intensive functions accept a ``backend`` parameter:
  - ``"auto"``  — selects the best available (cupy → torch → scipy)
  - ``"scipy"`` — CPU-only, ARPACK shift-invert (most robust)
  - ``"cupy"``  — GPU via CuPy LOBPCG + CuPy array ops
  - ``"torch"`` — GPU via PyTorch LOBPCG + tensor ops

The eigensolver is the bottleneck (~30–120s on CPU for 300 eigenpairs
on a 150K-vertex mesh). GPU backends achieve 3–10× speedup. The
spectral descriptors are pure GEMM operations completing in <1ms on
GPU regardless of backend.

Mathematical background
-----------------------
On a compact Riemannian manifold (Σ, g), the LB operator Δ_g has a
discrete spectrum  0 = λ_0 ≤ λ_1 ≤ λ_2 ≤ … with orthonormal
eigenfunctions  Δ_g φ_i = λ_i φ_i.

The **heat kernel** is  K_t(x, y) = Σ_i exp(−λ_i t) φ_i(x) φ_i(y),
and its diagonal  h(x, t) = K_t(x, x) = Σ_i exp(−λ_i t) φ_i(x)²
is the **Heat Kernel Signature** (Sun, Ovsjanikov & Guibas, 2009).

The **Wave Kernel Signature** (Aubry, Schlickewei & Cremers, 2011)
replaces the exponential decay with a log-normal energy filter:
  w(x, e) = Σ_i exp(−(e − log λ_i)² / 2σ²) φ_i(x)²  /  C(e)

References
----------
[1] M. Meyer et al., "Discrete differential-geometry operators for
    triangulated 2-manifolds", Springer, 2003.
[2] N. Sharp & K. Crane, "A Laplacian for Nonmanifold Triangle Meshes",
    SGP 2020 (best paper).
[3] J. Sun, M. Ovsjanikov & L. Guibas, "A Concise and Provably
    Informative Multi-Scale Signature Based on Heat Diffusion",
    SGP 2009.
[4] M. Aubry, U. Schlickewei & D. Cremers, "The Wave Kernel
    Signature: A Quantum Mechanical Approach to Shape Analysis",
    ICCV Workshops 2011.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp

from corticalfields.backends import (
    ArrayBackend,
    Backend,
    compute_laplacian as _backends_compute_laplacian,
    eigsh_solve,
    resolve_backend,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Laplace–Beltrami operator
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class LaplaceBeltrami:
    """
    Discrete Laplace–Beltrami operator on a triangle mesh.

    After construction, access ``.eigenvalues`` and ``.eigenvectors``
    for the leading spectral components.

    Parameters
    ----------
    stiffness : scipy.sparse.csc_matrix, shape (N, N)
        Cotangent stiffness matrix L.
    mass : scipy.sparse.csc_matrix, shape (N, N)
        Lumped (diagonal) mass matrix M.
    eigenvalues : np.ndarray, shape (K,)
        Leading eigenvalues λ_0 ≤ λ_1 ≤ … ≤ λ_{K-1}.
    eigenvectors : np.ndarray, shape (N, K)
        Corresponding M-orthonormal eigenvectors.
    """

    stiffness: sp.csc_matrix
    mass: sp.csc_matrix
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray

    @property
    def n_vertices(self) -> int:
        return self.eigenvectors.shape[0]

    @property
    def n_eigenpairs(self) -> int:
        return len(self.eigenvalues)


def compute_laplacian(
    vertices: np.ndarray,
    faces: np.ndarray,
    use_robust: bool = True,
    method: str = "auto",
) -> Tuple[sp.csc_matrix, sp.csc_matrix]:
    """
    Build the discrete Laplace–Beltrami stiffness and mass matrices.

    Delegates to ``backends.compute_laplacian`` with a 3-level fallback:
      1. robust-laplacian (intrinsic Delaunay, non-manifold safe)
      2. LaPy (vectorized FEM cotangent, optional CHOLMOD)
      3. built-in vectorized cotangent (pure NumPy)

    Parameters
    ----------
    vertices : (N, 3) float64
    faces    : (F, 3) int64
    use_robust : bool
        Backward-compatible flag. If True and method="auto", prefers
        robust-laplacian. If False, skips robust and uses lapy/builtin.
    method : str
        Explicit Laplacian method: ``"auto"``, ``"robust"``, ``"lapy"``,
        or ``"builtin"``. Overrides ``use_robust`` when not ``"auto"``.

    Returns
    -------
    L : scipy.sparse.csc_matrix (N, N)
        Stiffness matrix (positive semi-definite).
    M : scipy.sparse.csc_matrix (N, N)
        Diagonal lumped mass matrix.
    """
    # Backward compatibility: if method is explicitly set, use it;
    # otherwise, use_robust controls whether robust-laplacian is tried
    if method != "auto":
        return _backends_compute_laplacian(vertices, faces, method=method)
    if not use_robust:
        # Skip robust-laplacian, try lapy → builtin
        return _backends_compute_laplacian(vertices, faces, method="lapy")
    return _backends_compute_laplacian(vertices, faces, method="auto")


def compute_eigenpairs(
    vertices: np.ndarray,
    faces: np.ndarray,
    n_eigenpairs: int = 300,
    use_robust: bool = True,
    sigma: float = -0.01,
    backend: str = "auto",
    dtype: str = "float64",
    laplacian_method: str = "auto",
) -> LaplaceBeltrami:
    """
    Compute the leading eigenpairs of the LB operator on a mesh.

    This is the main entry point for spectral analysis. It builds the
    Laplacian, solves the generalised eigenvalue problem, and returns
    a :class:`LaplaceBeltrami` object with all spectral data.

    Parameters
    ----------
    vertices : (N, 3) array
    faces : (F, 3) array
    n_eigenpairs : int
        Number of eigenpairs to compute (including λ_0 = 0).
    use_robust : bool
        Backward-compatible flag for Laplacian assembly.
    sigma : float
        Shift-invert parameter for scipy backend. Ignored by GPU backends.
    backend : str
        Eigensolver backend: ``'auto'``, ``'scipy'``, ``'cupy'``, ``'torch'``.
    dtype : str
        Internal precision: ``'float64'`` (default) or ``'float32'``.
    laplacian_method : str
        Laplacian assembly method: ``'auto'``, ``'robust'``, ``'lapy'``,
        or ``'builtin'``. When ``'auto'``, uses the 3-level fallback chain.

    Returns
    -------
    LaplaceBeltrami

    Examples
    --------
    >>> # Default: best Laplacian + scipy ARPACK (most robust)
    >>> lb = compute_eigenpairs(vertices, faces)

    >>> # Explicit LaPy Laplacian + scipy eigensolver
    >>> lb = compute_eigenpairs(vertices, faces, laplacian_method="lapy")

    >>> # GPU eigensolver (CuPy LOBPCG)
    >>> lb = compute_eigenpairs(vertices, faces, backend="cupy")

    >>> # LaPy Laplacian + GPU descriptors (mix-and-match)
    >>> lb = compute_eigenpairs(vertices, faces, laplacian_method="lapy")
    >>> features = spectral_feature_matrix(lb, backend="cupy")

    Notes
    -----
    The Laplacian is always assembled on CPU (the assembly is fast
    and produces SciPy sparse matrices). Only the eigensolver and
    descriptor computation benefit from GPU acceleration.

    For a typical FreeSurfer mesh (~150k vertices), computing 300
    eigenpairs takes 30–120s on CPU, 5–30s on GPU.
    """
    logger.info(
        "Computing %d LB eigenpairs for mesh with %d vertices (backend=%s)…",
        n_eigenpairs, vertices.shape[0], backend,
    )

    # Step 1: Build Laplacian (always on CPU — fast, ~100ms)
    # Uses the 3-level fallback: robust → lapy → builtin
    L, M = compute_laplacian(
        vertices, faces,
        use_robust=use_robust, method=laplacian_method,
    )

    # Step 2: Solve eigenvalue problem (dispatched to backend)
    eigenvalues, eigenvectors = eigsh_solve(
        L, M,
        k=n_eigenpairs,
        backend=backend,
        sigma=sigma,
        dtype=dtype,
    )

    # Clamp tiny/negative eigenvalues to zero (numerical noise)
    eigenvalues = np.maximum(eigenvalues, 0.0)

    logger.info(
        "  eigenvalue range: [%.6f, %.4f], sum area = %.1f mm²",
        eigenvalues[0], eigenvalues[-1],
        np.array(M.diagonal()).sum(),
    )

    return LaplaceBeltrami(
        stiffness=L,
        mass=M,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Spectral shape descriptors
# ═══════════════════════════════════════════════════════════════════════════


def heat_kernel_signature(
    lb: LaplaceBeltrami,
    time_scales: Optional[np.ndarray] = None,
    n_scales: int = 16,
    t_min: Optional[float] = None,
    t_max: Optional[float] = None,
    backend: str = "scipy",
) -> np.ndarray:
    """
    Heat Kernel Signature (Sun, Ovsjanikov & Guibas, 2009).

    For each vertex x and diffusion time t:
        HKS(x, t) = Σ_{i=1}^{K} exp(−λ_i · t) · φ_i(x)²

    The HKS at small t captures local curvature; at large t it
    captures global shape features (e.g., which lobe a vertex belongs to).
    This multi-scale behaviour is what makes HKS ideal for anomaly
    detection: atrophy in MTLE-HS will alter the HKS across scales.

    Parameters
    ----------
    lb : LaplaceBeltrami
        Pre-computed eigenpairs.
    time_scales : np.ndarray or None
        Explicit array of diffusion times. If None, ``n_scales``
        log-spaced times are generated from eigenvalue range.
    n_scales : int
        Number of log-spaced time scales (used if ``time_scales`` is None).
    t_min, t_max : float or None
        Bounds for auto time scales. Defaults:
        ``t_min = 4 ln(10) / λ_max``, ``t_max = 4 ln(10) / λ_1``.
    backend : str
        Compute backend for the dense GEMM. Descriptors are pure
        matrix operations: ``phi_sq @ weights``. On GPU, this is a
        single cuBLAS/cuDNN call completing in <1ms for 150K × 300.

    Returns
    -------
    hks : np.ndarray, shape (N, T)
        HKS values at each vertex and time scale.

    Notes
    -----
    The first eigenvalue λ_0 ≈ 0 is excluded from the sum to avoid
    a constant offset that carries no geometric information.
    """
    ab = ArrayBackend.create(backend)

    # Exclude λ_0 ≈ 0 (constant eigenfunction)
    evals_np = lb.eigenvalues[1:]
    evecs_np = lb.eigenvectors[:, 1:]

    if time_scales is None:
        lam_min = max(evals_np[0], 1e-8)
        lam_max = evals_np[-1]
        if t_min is None:
            t_min = 4.0 * np.log(10) / lam_max
        if t_max is None:
            t_max = 4.0 * np.log(10) / lam_min
        time_scales = np.logspace(np.log10(t_min), np.log10(t_max), n_scales)

    # Transfer to backend
    evals = ab.asarray(evals_np)
    evecs = ab.asarray(evecs_np)
    t_arr = ab.asarray(time_scales)

    # HKS: phi_sq @ exp_weights → (N, T) GEMM
    weights = ab.exp(-evals[:, None] * t_arr[None, :])  # (K, T)
    phi_sq = ab.square(evecs)  # (N, K)
    hks = ab.matmul(phi_sq, weights)  # (N, T)

    return ab.to_numpy(hks)


def wave_kernel_signature(
    lb: LaplaceBeltrami,
    n_energies: int = 16,
    sigma: Optional[float] = None,
    e_min: Optional[float] = None,
    e_max: Optional[float] = None,
    backend: str = "scipy",
) -> np.ndarray:
    """
    Wave Kernel Signature (Aubry, Schlickewei & Cremers, 2011).

    Uses a log-normal energy distribution centred at different energy
    levels in the eigenvalue spectrum:

        WKS(x, e) = Σ_i exp(−(e − log λ_i)² / 2σ²) · φ_i(x)² / C(e)

    where C(e) is a normalising constant.

    The WKS is scale-invariant (unlike HKS) and provides a finer
    frequency decomposition, making it sensitive to local folding
    patterns and gyrification changes.

    Parameters
    ----------
    lb : LaplaceBeltrami
    n_energies : int
    sigma : float or None
    e_min, e_max : float or None
    backend : str
        Compute backend for dense operations.

    Returns
    -------
    wks : np.ndarray, shape (N, E)
    """
    ab = ArrayBackend.create(backend)

    evals_np = lb.eigenvalues[1:]
    evecs_np = lb.eigenvectors[:, 1:]

    log_evals_np = np.log(np.maximum(evals_np, 1e-12))
    if e_min is None:
        e_min = log_evals_np[0]
    if e_max is None:
        e_max = log_evals_np[-1]

    energies_np = np.linspace(e_min, e_max, n_energies)
    delta_e = energies_np[1] - energies_np[0] if n_energies > 1 else 1.0
    if sigma is None:
        sigma = 7.0 * delta_e

    # Transfer to backend
    log_evals = ab.asarray(log_evals_np)
    energies = ab.asarray(energies_np)
    evecs = ab.asarray(evecs_np)

    # WKS weights: log-normal filter
    diff = energies[None, :] - log_evals[:, None]  # (K, E)
    weights = ab.exp(-ab.square(diff) / (2.0 * sigma ** 2))  # (K, E)

    # Normalisation per energy level
    if ab._is_torch:
        import torch
        norm = weights.sum(dim=0, keepdim=True)
        norm = torch.clamp(norm, min=1e-16)
    else:
        norm = weights.sum(axis=0, keepdims=True)
        norm_np = ab.to_numpy(norm)
        norm_np[norm_np < 1e-16] = 1e-16
        norm = ab.asarray(norm_np)

    phi_sq = ab.square(evecs)  # (N, K)
    wks = ab.matmul(phi_sq, weights) / norm  # (N, E)

    return ab.to_numpy(wks)


def global_point_signature(
    lb: LaplaceBeltrami,
    n_components: Optional[int] = None,
    backend: str = "scipy",
) -> np.ndarray:
    """
    Global Point Signature (Rustamov, 2007).

    GPS(x) = ( φ_1(x)/√λ_1 , φ_2(x)/√λ_2 , … , φ_K(x)/√λ_K )

    This embedding maps each vertex to a point in ℝ^K such that the
    Euclidean distance in the embedding approximates the Green's
    function distance on the manifold. It provides a global coordinate
    system on the cortex that respects intrinsic geometry.

    Parameters
    ----------
    lb : LaplaceBeltrami
    n_components : int or None
    backend : str
        Compute backend for the scaling operation.

    Returns
    -------
    gps : np.ndarray, shape (N, K)
    """
    ab = ArrayBackend.create(backend)

    evals_np = lb.eigenvalues[1:]
    evecs_np = lb.eigenvectors[:, 1:]

    if n_components is not None:
        evals_np = evals_np[:n_components]
        evecs_np = evecs_np[:, :n_components]

    evals = ab.asarray(evals_np)
    evecs = ab.asarray(evecs_np)

    inv_sqrt_evals = 1.0 / ab.sqrt(ab.maximum(evals, 1e-12))
    gps = evecs * inv_sqrt_evals[None, :]

    return ab.to_numpy(gps)


# ═══════════════════════════════════════════════════════════════════════════
# Multi-scale spectral feature matrix
# ═══════════════════════════════════════════════════════════════════════════


def spectral_feature_matrix(
    lb: LaplaceBeltrami,
    hks_scales: int = 16,
    wks_energies: int = 16,
    gps_components: int = 10,
    include_hks: bool = True,
    include_wks: bool = True,
    include_gps: bool = True,
    backend: str = "scipy",
) -> np.ndarray:
    """
    Build a combined per-vertex spectral feature matrix.

    Concatenates HKS, WKS, and GPS descriptors column-wise,
    producing a rich multi-scale feature vector per vertex.

    Parameters
    ----------
    lb : LaplaceBeltrami
    hks_scales, wks_energies, gps_components : int
        Dimensionality of each descriptor block.
    include_hks, include_wks, include_gps : bool
        Which descriptors to include.
    backend : str
        Compute backend for dense operations. Descriptors are pure
        GEMM and element-wise ops — GPU gives 10–30× speedup.

    Returns
    -------
    features : np.ndarray, shape (N, D)
        D = sum of included descriptor dimensions.

    Examples
    --------
    >>> # CPU (default)
    >>> features = spectral_feature_matrix(lb)

    >>> # GPU via CuPy (10–30× faster for 150K vertices)
    >>> features = spectral_feature_matrix(lb, backend="cupy")
    """
    blocks = []
    if include_hks:
        blocks.append(heat_kernel_signature(lb, n_scales=hks_scales, backend=backend))
    if include_wks:
        blocks.append(wave_kernel_signature(lb, n_energies=wks_energies, backend=backend))
    if include_gps:
        blocks.append(global_point_signature(lb, n_components=gps_components, backend=backend))

    if not blocks:
        raise ValueError("At least one descriptor must be included.")

    return np.hstack(blocks)


# ═══════════════════════════════════════════════════════════════════════════
# Internals (kept for backward compatibility)
# ═══════════════════════════════════════════════════════════════════════════


def _safe_cot(u: np.ndarray, v: np.ndarray) -> float:
    """Cotangent of the angle between vectors u and v, clamped to avoid NaN."""
    cos_angle = np.dot(u, v)
    sin_angle = np.linalg.norm(np.cross(u, v))
    if sin_angle < 1e-12:
        return 0.0
    return cos_angle / sin_angle
