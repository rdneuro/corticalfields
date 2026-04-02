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

These descriptors capture multi-scale geometric information on the cortical
surface and serve as features for the downstream GP normative model.

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
from scipy.sparse.linalg import eigsh

logger = logging.getLogger(__name__)

# Try to import robust-laplacian (preferred); fall back to our own
# cotangent Laplacian implementation if unavailable.
try:
    import robust_laplacian

    _HAS_ROBUST_LAP = True
    logger.debug("Using robust-laplacian (Sharp & Crane, SGP 2020).")
except ImportError:
    _HAS_ROBUST_LAP = False
    logger.debug("robust-laplacian not found; using cotangent Laplacian.")


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
    stiffness : scipy.sparse.csc_matrix or None, shape (N, N)
        Cotangent stiffness matrix L.  May be ``None`` when loading
        from a cache that stored only eigenvalues/eigenvectors.
    mass : scipy.sparse.csc_matrix or None, shape (N, N)
        Lumped (diagonal) mass matrix M.  May be ``None`` when loading
        from cache.  Some downstream operations (proper L² projection
        in functional maps) require mass; if unavailable, uniform
        weighting is used as fallback.
    eigenvalues : np.ndarray, shape (K,)
        Leading eigenvalues λ_0 ≤ λ_1 ≤ … ≤ λ_{K-1}.
    eigenvectors : np.ndarray, shape (N, K)
        Corresponding M-orthonormal eigenvectors.
    """

    stiffness: Optional[sp.csc_matrix]
    mass: Optional[sp.csc_matrix]
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray

    @property
    def n_vertices(self) -> int:
        """Number of mesh vertices (rows of the eigenvector matrix)."""
        return self.eigenvectors.shape[0]

    @property
    def n_eigenpairs(self) -> int:
        """Number of computed eigenpairs."""
        return len(self.eigenvalues)

    @property
    def has_matrices(self) -> bool:
        """Whether stiffness (L) and mass (M) matrices are available."""
        return self.stiffness is not None and self.mass is not None


def compute_laplacian(
    vertices: np.ndarray,
    faces: np.ndarray,
    use_robust: bool = True,
) -> Tuple[sp.csc_matrix, sp.csc_matrix]:
    """
    Build the discrete Laplace–Beltrami stiffness and mass matrices.

    When ``robust-laplacian`` is available and ``use_robust=True``, uses
    the intrinsic Delaunay refinement method (Sharp & Crane 2020), which
    is numerically stable even on poor-quality or non-manifold meshes.

    Otherwise, falls back to the standard cotangent scheme.

    Parameters
    ----------
    vertices : (N, 3) float64
    faces    : (F, 3) int64
    use_robust : bool
        Prefer robust-laplacian if installed.

    Returns
    -------
    L : scipy.sparse.csc_matrix (N, N)
        Stiffness matrix (positive semi-definite).
    M : scipy.sparse.csc_matrix (N, N)
        Diagonal lumped mass matrix.
    """
    if use_robust and _HAS_ROBUST_LAP:
        L, M = robust_laplacian.mesh_laplacian(
            np.asarray(vertices, dtype=np.float64),
            np.asarray(faces, dtype=np.int64),
        )
        return sp.csc_matrix(L), sp.csc_matrix(M)

    # ── Cotangent Laplacian (Meyer et al. 2003) ────────────────────────
    N = vertices.shape[0]
    v = vertices.astype(np.float64)
    f = faces.astype(np.int64)

    # For each triangle (i, j, k), compute cotangent of the angle at
    # vertex k for the edge (i, j), and accumulate into L.
    rows, cols, vals = [], [], []
    areas = np.zeros(N, dtype=np.float64)

    for tri in f:
        i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
        vi, vj, vk = v[i], v[j], v[k]

        # Edge vectors
        eij = vj - vi  # edge from i to j
        eik = vk - vi  # edge from i to k
        ejk = vk - vj  # edge from j to k

        # Cotangent weights for each angle
        # Angle at i  → weight for edge (j, k)
        cot_i = _safe_cot(eij, eik)
        # Angle at j  → weight for edge (i, k)
        cot_j = _safe_cot(-eij, ejk)
        # Angle at k  → weight for edge (i, j)
        cot_k = _safe_cot(-eik, -ejk)

        # Stiffness contributions (off-diagonal: −½ cotangent)
        for (a, b, w) in [(j, k, cot_i), (i, k, cot_j), (i, j, cot_k)]:
            rows.extend([a, b])
            cols.extend([b, a])
            vals.extend([-0.5 * w, -0.5 * w])

        # Triangle area (for mass matrix)
        tri_area = 0.5 * np.linalg.norm(np.cross(eij, eik))
        # Distribute ⅓ of area to each vertex (lumped)
        areas[i] += tri_area / 3.0
        areas[j] += tri_area / 3.0
        areas[k] += tri_area / 3.0

    # Assemble stiffness
    L = sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsc()
    # Make diagonal = −Σ off-diagonal (ensures row-sum = 0)
    diag = -np.array(L.sum(axis=1)).ravel()
    L = L + sp.diags(diag, format="csc")

    # Lumped mass matrix
    areas[areas < 1e-16] = 1e-16  # avoid zero mass
    M = sp.diags(areas, format="csc")

    return L, M


def compute_eigenpairs(
    vertices: np.ndarray,
    faces: np.ndarray,
    n_eigenpairs: int = 300,
    use_robust: bool = True,
    sigma: float = -0.01,
    backend: str = "auto",
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
        Prefer robust-laplacian if available.
    sigma : float
        Shift-invert parameter for the eigensolver.
        A small negative value targets the smallest eigenvalues.
    backend : ``'auto'``, ``'scipy'``, ``'cupy'``, or ``'torch'``
        Compute backend for the eigendecomposition.
        ``'auto'`` selects cupy → torch → scipy in order of availability.
        ``'cupy'`` is recommended (fastest, no CSR warnings).

    Returns
    -------
    LaplaceBeltrami
        Object with stiffness, mass, eigenvalues, eigenvectors.

    Notes
    -----
    The generalised eigenproblem ``L φ = λ M φ`` is solved using
    ARPACK's shift-invert mode (via ``eigsh``) for scipy, or the
    equivalent GPU-accelerated solver for cupy/torch backends.

    For a typical FreeSurfer mesh (~150k vertices), computing 300
    eigenpairs takes 1–3 minutes on CPU, ~30s on GPU (cupy).
    """
    from corticalfields.backends import eigsh_solve, resolve_backend

    logger.info(
        "Computing %d LB eigenpairs for mesh with %d vertices…",
        n_eigenpairs, vertices.shape[0],
    )

    L, M = compute_laplacian(vertices, faces, use_robust=use_robust)

    be = resolve_backend(backend)
    logger.info("  Eigensolver backend: %s", be.value)

    eigenvalues, eigenvectors = eigsh_solve(
        L, M, k=n_eigenpairs, sigma=sigma, backend=be.value,
    )

    # Sort explicitly to be safe
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

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

    Returns
    -------
    hks : np.ndarray, shape (N, T)
        HKS values at each vertex and time scale.

    Notes
    -----
    The first eigenvalue λ_0 ≈ 0 is excluded from the sum to avoid
    a constant offset that carries no geometric information.
    """
    # Exclude λ_0 ≈ 0 (constant eigenfunction)
    evals = lb.eigenvalues[1:]
    evecs = lb.eigenvectors[:, 1:]

    if time_scales is None:
        # Heuristic time range from Sun et al. 2009
        lam_min = max(evals[0], 1e-8)
        lam_max = evals[-1]
        if t_min is None:
            t_min = 4.0 * np.log(10) / lam_max
        if t_max is None:
            t_max = 4.0 * np.log(10) / lam_min
        time_scales = np.logspace(np.log10(t_min), np.log10(t_max), n_scales)

    # Compute HKS: for each time t, h(x,t) = Σ_i exp(-λ_i t) φ_i(x)²
    # Shape: (K,) × (T,) → (K, T) exponential weights
    weights = np.exp(-evals[:, None] * time_scales[None, :])  # (K, T)
    phi_sq = evecs ** 2  # (N, K)
    hks = phi_sq @ weights  # (N, T)

    return hks


def wave_kernel_signature(
    lb: LaplaceBeltrami,
    n_energies: int = 16,
    sigma: Optional[float] = None,
    e_min: Optional[float] = None,
    e_max: Optional[float] = None,
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
        Pre-computed eigenpairs.
    n_energies : int
        Number of energy levels to sample.
    sigma : float or None
        Bandwidth of the log-normal filter. Default: 7× the energy spacing.
    e_min, e_max : float or None
        Bounds for energy levels in log-eigenvalue space.

    Returns
    -------
    wks : np.ndarray, shape (N, E)
        WKS values at each vertex and energy level.
    """
    # Exclude λ_0 ≈ 0; work in log-space
    evals = lb.eigenvalues[1:]
    evecs = lb.eigenvectors[:, 1:]

    log_evals = np.log(np.maximum(evals, 1e-12))

    if e_min is None:
        e_min = log_evals[0]
    if e_max is None:
        e_max = log_evals[-1]

    energies = np.linspace(e_min, e_max, n_energies)
    delta_e = energies[1] - energies[0] if n_energies > 1 else 1.0

    if sigma is None:
        sigma = 7.0 * delta_e

    # Weights: (K, E) — log-normal filter for each eigenvalue at each energy
    diff = energies[None, :] - log_evals[:, None]  # (K, E)
    weights = np.exp(-diff ** 2 / (2.0 * sigma ** 2))  # (K, E)

    # Normalisation per energy level
    norm = weights.sum(axis=0, keepdims=True)  # (1, E)
    norm[norm < 1e-16] = 1e-16

    phi_sq = evecs ** 2  # (N, K)
    wks = (phi_sq @ weights) / norm  # (N, E)

    return wks


def global_point_signature(
    lb: LaplaceBeltrami,
    n_components: Optional[int] = None,
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
        How many components to use (default: all non-zero).

    Returns
    -------
    gps : np.ndarray, shape (N, K)
    """
    evals = lb.eigenvalues[1:]  # skip λ_0 = 0
    evecs = lb.eigenvectors[:, 1:]

    if n_components is not None:
        evals = evals[:n_components]
        evecs = evecs[:, :n_components]

    inv_sqrt_evals = 1.0 / np.sqrt(np.maximum(evals, 1e-12))
    gps = evecs * inv_sqrt_evals[None, :]

    return gps


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

    Returns
    -------
    features : np.ndarray, shape (N, D)
        D = sum of included descriptor dimensions.
    """
    blocks = []
    if include_hks:
        blocks.append(heat_kernel_signature(lb, n_scales=hks_scales))
    if include_wks:
        blocks.append(wave_kernel_signature(lb, n_energies=wks_energies))
    if include_gps:
        blocks.append(global_point_signature(lb, n_components=gps_components))

    if not blocks:
        raise ValueError("At least one descriptor must be included.")

    return np.hstack(blocks)


# ═══════════════════════════════════════════════════════════════════════════
# Internals
# ═══════════════════════════════════════════════════════════════════════════


def _safe_cot(u: np.ndarray, v: np.ndarray) -> float:
    """Cotangent of the angle between vectors u and v, clamped to avoid NaN."""
    cos_angle = np.dot(u, v)
    sin_angle = np.linalg.norm(np.cross(u, v))
    if sin_angle < 1e-12:
        return 0.0
    return cos_angle / sin_angle
