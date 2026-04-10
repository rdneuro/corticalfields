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
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

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
# Batch eigenpair computation with memory-aware parallelism
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class MemoryEstimate:
    """Per-subject memory estimate (in bytes) for eigenpair computation.

    Attributes
    ----------
    ram_bytes : int
        Predicted host RAM consumption.
    vram_bytes : int
        Predicted GPU VRAM consumption (0 for CPU backends).
    n_vertices : int
        Vertex count used for the estimate.
    n_eigenpairs : int
        Eigenpair count used for the estimate.
    backend : str
        Backend string (``"scipy"``, ``"torch"``, ``"cupy"``).
    """

    ram_bytes: int
    vram_bytes: int
    n_vertices: int
    n_eigenpairs: int
    backend: str

    @property
    def ram_gb(self) -> float:
        return self.ram_bytes / (1024 ** 3)

    @property
    def vram_gb(self) -> float:
        return self.vram_bytes / (1024 ** 3)

    def __repr__(self) -> str:
        return (
            f"MemoryEstimate(RAM={self.ram_gb:.2f} GB, "
            f"VRAM={self.vram_gb:.2f} GB, "
            f"V={self.n_vertices}, K={self.n_eigenpairs}, "
            f"backend={self.backend!r})"
        )


def estimate_memory_per_subject(
    n_vertices: int,
    n_eigenpairs: int = 300,
    backend: str = "scipy",
) -> MemoryEstimate:
    """
    Predict RAM and VRAM usage for a single eigenpair computation.

    The model accounts for:

    - **Sparse matrices** (L, M): stiffness has ~7 non-zeros per row
      for a triangle mesh (cotangent weights); mass is diagonal.
      With CSC storage overhead: ``(7 · N · 16 + N · 16)`` bytes
      ≈ ``128 · N``.
    - **Eigenvectors**: ``N × K × 8`` bytes (float64).
    - **Solver workspace**: ARPACK needs ≈ ``3 × N × K × 8`` bytes for
      Lanczos vectors; LOBPCG (torch/cupy) needs ≈ ``5 × N × K × 8``
      for X, W, P blocks plus temporaries.
    - **Safety factor**: 1.3× to cover Python object overhead,
      temporaries during Laplacian assembly, and backend-specific
      allocations.

    For GPU backends the VRAM estimate includes eigenvectors + workspace
    on device, plus 0.5× sparse cost for partial dense transfers that
    LOBPCG may perform internally.

    Parameters
    ----------
    n_vertices : int
        Number of mesh vertices.
    n_eigenpairs : int
        Number of eigenpairs to compute.
    backend : str
        ``"scipy"``, ``"torch"``, or ``"cupy"``.

    Returns
    -------
    MemoryEstimate
    """
    N, K = n_vertices, n_eigenpairs
    DTYPE_SIZE = 8  # float64

    sparse_bytes = int(128 * N)
    evec_bytes = N * K * DTYPE_SIZE

    workspace_multiplier = 3 if backend == "scipy" else 5
    workspace_bytes = workspace_multiplier * N * K * DTYPE_SIZE

    SAFETY = 1.3
    ram_total = int((sparse_bytes + evec_bytes + workspace_bytes) * SAFETY)

    if backend in ("torch", "cupy"):
        vram_total = int(
            (evec_bytes + workspace_bytes + 0.5 * sparse_bytes) * SAFETY
        )
    else:
        vram_total = 0

    return MemoryEstimate(
        ram_bytes=ram_total,
        vram_bytes=vram_total,
        n_vertices=N,
        n_eigenpairs=K,
        backend=backend,
    )


def _get_system_free_ram_bytes() -> int:
    """Return free RAM in bytes (psutil → /proc/meminfo → 8 GB fallback)."""
    try:
        import psutil
        return psutil.virtual_memory().available
    except ImportError:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except FileNotFoundError:
        pass
    warnings.warn(
        "Cannot determine free RAM; assuming 8 GB. "
        "Install psutil for accuracy.",
        RuntimeWarning,
        stacklevel=2,
    )
    return 8 * (1024 ** 3)


def _get_gpu_free_vram_bytes() -> int:
    """Return free VRAM in bytes (torch → cupy → 0)."""
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total = props.total_memory  # NOT .total_mem
            allocated = torch.cuda.memory_allocated(0)
            return total - allocated
    except ImportError:
        pass
    try:
        import cupy as cp
        free, _total = cp.cuda.Device(0).mem_info
        return free
    except (ImportError, Exception):
        pass
    return 0


def compute_safe_parallelism(
    n_vertices: int,
    n_eigenpairs: int = 300,
    backend: str = "scipy",
    requested_n_jobs: int = 1,
    requested_batch_size: int = 1,
    ram_reserve_gb: float = 2.0,
    vram_reserve_gb: float = 1.0,
) -> Tuple[int, int, MemoryEstimate]:
    """
    Compute safe ``n_jobs`` and ``batch_size`` for batch eigenpairs.

    Estimates per-subject memory, queries system free RAM / VRAM, and
    clamps the user-requested parallelism to avoid OOM.

    Parameters
    ----------
    n_vertices : int
        Representative vertex count (use the largest subject).
    n_eigenpairs : int
        Number of eigenpairs per subject.
    backend : str
        Eigensolver backend string.
    requested_n_jobs : int
        Desired joblib parallelism (CPU backend only).
    requested_batch_size : int
        Desired GPU batch size (torch/cupy only).
    ram_reserve_gb : float
        RAM headroom to leave untouched.
    vram_reserve_gb : float
        VRAM headroom to leave untouched.

    Returns
    -------
    safe_n_jobs : int
        Clamped ``n_jobs`` (always 1 for GPU backends).
    safe_batch_size : int
        Clamped ``batch_size`` (always 1 for scipy backend).
    estimate : MemoryEstimate
        Per-subject memory estimate used for the calculation.
    """
    est = estimate_memory_per_subject(n_vertices, n_eigenpairs, backend)

    if backend == "scipy":
        free_ram = _get_system_free_ram_bytes()
        usable_ram = max(
            free_ram - int(ram_reserve_gb * (1024 ** 3)), est.ram_bytes,
        )
        max_parallel = max(1, int(usable_ram / est.ram_bytes))
        safe_jobs = min(requested_n_jobs, max_parallel)

        if safe_jobs < requested_n_jobs:
            logger.warning(
                "Clamped n_jobs %d → %d (free RAM: %.1f GB, "
                "per-subject: %.2f GB, reserve: %.1f GB)",
                requested_n_jobs, safe_jobs,
                free_ram / (1024 ** 3), est.ram_gb, ram_reserve_gb,
            )
        return safe_jobs, 1, est

    # GPU backends: limit batch_size by VRAM
    free_vram = _get_gpu_free_vram_bytes()
    if free_vram == 0:
        logger.warning(
            "No GPU VRAM detected for backend=%r; falling back to "
            "sequential processing (batch_size=1).",
            backend,
        )
        return 1, 1, est

    usable_vram = max(
        free_vram - int(vram_reserve_gb * (1024 ** 3)), est.vram_bytes,
    )
    max_parallel = max(1, int(usable_vram / est.vram_bytes))
    safe_batch = min(requested_batch_size, max_parallel)

    if safe_batch < requested_batch_size:
        logger.warning(
            "Clamped batch_size %d → %d (free VRAM: %.1f GB, "
            "per-subject: %.2f GB, reserve: %.1f GB)",
            requested_batch_size, safe_batch,
            free_vram / (1024 ** 3), est.vram_gb, vram_reserve_gb,
        )

    return 1, safe_batch, est


@dataclass
class SubjectMesh:
    """
    Lightweight container for one subject's mesh data.

    Parameters
    ----------
    subject_id : str
        Identifier (used for cache filenames and logging).
    vertices : np.ndarray, shape (N, 3)
        Mesh vertex coordinates.
    faces : np.ndarray, shape (F, 3)
        Triangle connectivity.
    """

    subject_id: str
    vertices: np.ndarray
    faces: np.ndarray

    @property
    def n_vertices(self) -> int:
        return self.vertices.shape[0]


def _process_single_subject(
    subject: SubjectMesh,
    n_eigenpairs: int,
    use_robust: bool,
    sigma: float,
    backend: str,
    cache_dir: Optional[Path],
    save_matrices: bool,
) -> Tuple[str, Optional[LaplaceBeltrami]]:
    """
    Compute eigenpairs for one subject with optional disk caching.

    Returns ``(subject_id, LaplaceBeltrami | None)``.
    Returns ``None`` when a cache file already exists (skip).
    """
    from corticalfields.utils import gc_gpu

    sid = subject.subject_id

    if cache_dir is not None:
        cache_file = cache_dir / f"{sid}.npz"
        if cache_file.exists():
            logger.debug("Cache hit for %s, skipping.", sid)
            return sid, None

    lb = compute_eigenpairs(
        subject.vertices,
        subject.faces,
        n_eigenpairs=n_eigenpairs,
        use_robust=use_robust,
        sigma=sigma,
        backend=backend,
    )

    if cache_dir is not None:
        cache_file = cache_dir / f"{sid}.npz"
        save_dict: Dict[str, Any] = {
            "eigenvalues": lb.eigenvalues,
            "eigenvectors": lb.eigenvectors,
        }
        if save_matrices and lb.has_matrices:
            save_dict["stiffness_data"] = lb.stiffness.data
            save_dict["stiffness_indices"] = lb.stiffness.indices
            save_dict["stiffness_indptr"] = lb.stiffness.indptr
            save_dict["stiffness_shape"] = np.array(lb.stiffness.shape)
            save_dict["mass_data"] = lb.mass.data
            save_dict["mass_indices"] = lb.mass.indices
            save_dict["mass_indptr"] = lb.mass.indptr
            save_dict["mass_shape"] = np.array(lb.mass.shape)
        np.savez_compressed(str(cache_file), **save_dict)

    gc_gpu()
    return sid, lb


@dataclass
class BatchResult:
    """
    Container for batch eigenpair processing results.

    Attributes
    ----------
    results : dict[str, LaplaceBeltrami | None]
        Mapping ``subject_id → LaplaceBeltrami``.  Value is ``None``
        for cache-hit subjects when ``return_results=False``.
    cache_dir : Path or None
        Directory where ``.npz`` caches were written.
    n_computed : int
        Subjects actually computed (excluding cache hits).
    n_cached : int
        Subjects skipped due to existing cache.
    elapsed_seconds : float
        Wall-clock time for the entire batch.
    memory_estimate : MemoryEstimate
        Per-subject memory estimate that was used for resource planning.
    """

    results: Dict[str, Any]
    cache_dir: Optional[Path]
    n_computed: int
    n_cached: int
    elapsed_seconds: float
    memory_estimate: MemoryEstimate


def batch_compute_eigenpairs(
    subjects: Sequence[SubjectMesh],
    n_eigenpairs: int = 300,
    use_robust: bool = True,
    sigma: float = -0.01,
    backend: str = "auto",
    n_jobs: int = 4,
    batch_size: int = 1,
    cache_dir: Optional[Union[str, Path]] = None,
    save_matrices: bool = False,
    return_results: bool = True,
    ram_reserve_gb: float = 2.0,
    vram_reserve_gb: float = 1.0,
    progress: bool = True,
) -> BatchResult:
    """
    Compute LBO eigenpairs for multiple subjects with smart parallelism.

    Strategy per backend
    --------------------
    - **scipy** (CPU): :mod:`joblib` with ``n_jobs`` parallel workers.
      ``n_jobs`` is clamped if predicted per-subject RAM exceeds
      available system memory.
    - **torch** (GPU): Sequential on GPU in batches of ``batch_size``.
      :func:`~corticalfields.utils.gc_gpu` is called between batches.
      ``batch_size`` is clamped by free VRAM.
    - **cupy** (GPU): Same strategy as torch; CuPy memory pool is freed
      between batches.

    The function calls :func:`compute_safe_parallelism` to estimate
    per-subject memory and adjust ``n_jobs`` / ``batch_size`` to fit
    within system resources before starting.

    Parameters
    ----------
    subjects : sequence of SubjectMesh
        Each element carries ``subject_id``, ``vertices``, ``faces``.
    n_eigenpairs : int
        Number of LBO eigenpairs per subject.
    use_robust : bool
        Use ``robust-laplacian`` (Sharp & Crane 2020) when available.
    sigma : float
        Shift-invert parameter for the eigensolver.
    backend : str
        ``"auto"``, ``"scipy"``, ``"torch"``, or ``"cupy"``.
    n_jobs : int
        Maximum joblib workers (CPU only).  Automatically clamped by
        available RAM.  Ignored for GPU backends.
    batch_size : int
        Maximum subjects per GPU batch before ``gc_gpu()``.  Clamped
        by available VRAM.  Ignored for CPU.
    cache_dir : str or Path or None
        If set, each subject is saved as ``{subject_id}.npz`` and
        skipped on re-run.  Created automatically if missing.
    save_matrices : bool
        Also cache the sparse stiffness / mass matrices (large).
    return_results : bool
        If ``False``, result dict values are ``None`` (saves RAM when
        only caching to disk).
    ram_reserve_gb : float
        RAM headroom to leave for the OS / other processes.
    vram_reserve_gb : float
        VRAM headroom for other GPU consumers.
    progress : bool
        Show Rich progress bars (two-level: total + batch).

    Returns
    -------
    BatchResult

    Examples
    --------
    >>> meshes = [SubjectMesh("sub-01", v1, f1), SubjectMesh("sub-02", v2, f2)]
    >>> res = batch_compute_eigenpairs(meshes, n_eigenpairs=200, n_jobs=4,
    ...                                cache_dir="./cache")
    >>> res.results["sub-01"].eigenvalues.shape
    (200,)
    """
    from corticalfields.backends import resolve_backend

    if len(subjects) == 0:
        raise ValueError("subjects list is empty.")

    resolved = resolve_backend(backend)
    backend_str = resolved.value

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Memory-aware parallelism ──
    max_verts = max(s.n_vertices for s in subjects)
    safe_n_jobs, safe_batch_size, mem_est = compute_safe_parallelism(
        n_vertices=max_verts,
        n_eigenpairs=n_eigenpairs,
        backend=backend_str,
        requested_n_jobs=n_jobs,
        requested_batch_size=batch_size,
        ram_reserve_gb=ram_reserve_gb,
        vram_reserve_gb=vram_reserve_gb,
    )

    n_total = len(subjects)

    logger.info(
        "Batch eigenpairs: %d subjects, backend=%s, n_jobs=%d, "
        "batch_size=%d, per-subject RAM=%.2f GB, VRAM=%.2f GB",
        n_total, backend_str, safe_n_jobs, safe_batch_size,
        mem_est.ram_gb, mem_est.vram_gb,
    )

    t0 = time.perf_counter()

    if backend_str == "scipy":
        results, n_computed, n_cached = _batch_cpu(
            subjects, n_eigenpairs, use_robust, sigma, backend_str,
            safe_n_jobs, cache_dir, save_matrices, return_results,
            progress,
        )
    else:
        results, n_computed, n_cached = _batch_gpu(
            subjects, n_eigenpairs, use_robust, sigma, backend_str,
            safe_batch_size, cache_dir, save_matrices, return_results,
            progress,
        )

    elapsed = time.perf_counter() - t0
    logger.info(
        "Batch complete: %d computed, %d cached, %.1fs elapsed.",
        n_computed, n_cached, elapsed,
    )

    return BatchResult(
        results=results,
        cache_dir=cache_dir,
        n_computed=n_computed,
        n_cached=n_cached,
        elapsed_seconds=elapsed,
        memory_estimate=mem_est,
    )


# ── Rich progress column spec (shared) ──────────────────────────────────


def _rich_columns():
    """Return the Rich progress column list for batch bars."""
    from rich.progress import (
        BarColumn, MofNCompleteColumn, SpinnerColumn, TextColumn,
        TimeElapsedColumn, TimeRemainingColumn,
    )
    return [
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    ]


# ── CPU batch (joblib) ──────────────────────────────────────────────────


def _batch_cpu(
    subjects: Sequence[SubjectMesh],
    n_eigenpairs: int,
    use_robust: bool,
    sigma: float,
    backend: str,
    n_jobs: int,
    cache_dir: Optional[Path],
    save_matrices: bool,
    return_results: bool,
    progress: bool,
) -> Tuple[Dict[str, Any], int, int]:
    """CPU-parallel batch via joblib with Rich progress."""
    from joblib import Parallel, delayed

    n_total = len(subjects)
    to_compute: List[SubjectMesh] = []
    cached_ids: List[str] = []

    for s in subjects:
        if cache_dir is not None and (cache_dir / f"{s.subject_id}.npz").exists():
            cached_ids.append(s.subject_id)
        else:
            to_compute.append(s)

    results: Dict[str, Any] = {sid: None for sid in cached_ids}
    n_cached = len(cached_ids)
    n_to_do = len(to_compute)

    if n_to_do == 0:
        if progress:
            _print_all_cached(n_total)
        return results, 0, n_cached

    if not progress:
        pairs = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_process_single_subject)(
                s, n_eigenpairs, use_robust, sigma, backend,
                cache_dir, save_matrices,
            )
            for s in to_compute
        )
        for sid, lb in pairs:
            results[sid] = lb if return_results else None
        return results, n_to_do, n_cached

    # Rich progress with joblib generator (≥ 1.3) or chunk fallback
    from rich.progress import Progress

    with Progress(*_rich_columns(), transient=False) as prog:
        total_task = prog.add_task(
            "Subjects", total=n_total, completed=n_cached,
        )
        try:
            gen = Parallel(
                n_jobs=n_jobs, prefer="processes",
                return_as="generator",
            )(
                delayed(_process_single_subject)(
                    s, n_eigenpairs, use_robust, sigma, backend,
                    cache_dir, save_matrices,
                )
                for s in to_compute
            )
            for sid, lb in gen:
                results[sid] = lb if return_results else None
                prog.advance(total_task, 1)
        except TypeError:
            chunk_sz = max(1, n_jobs)
            for i in range(0, n_to_do, chunk_sz):
                chunk = to_compute[i : i + chunk_sz]
                pairs = Parallel(n_jobs=n_jobs, prefer="processes")(
                    delayed(_process_single_subject)(
                        s, n_eigenpairs, use_robust, sigma, backend,
                        cache_dir, save_matrices,
                    )
                    for s in chunk
                )
                for sid, lb in pairs:
                    results[sid] = lb if return_results else None
                    prog.advance(total_task, 1)

    return results, n_to_do, n_cached


# ── GPU batch (torch / cupy — sequential, gc between batches) ───────────


def _batch_gpu(
    subjects: Sequence[SubjectMesh],
    n_eigenpairs: int,
    use_robust: bool,
    sigma: float,
    backend: str,
    batch_size: int,
    cache_dir: Optional[Path],
    save_matrices: bool,
    return_results: bool,
    progress: bool,
) -> Tuple[Dict[str, Any], int, int]:
    """GPU sequential batch with two-level Rich progress."""
    from corticalfields.utils import gc_gpu

    n_total = len(subjects)
    to_compute: List[SubjectMesh] = []
    cached_ids: List[str] = []

    for s in subjects:
        if cache_dir is not None and (cache_dir / f"{s.subject_id}.npz").exists():
            cached_ids.append(s.subject_id)
        else:
            to_compute.append(s)

    results: Dict[str, Any] = {sid: None for sid in cached_ids}
    n_cached = len(cached_ids)
    n_to_do = len(to_compute)

    if n_to_do == 0:
        if progress:
            _print_all_cached(n_total)
        return results, 0, n_cached

    batches: List[List[SubjectMesh]] = [
        to_compute[i : i + batch_size]
        for i in range(0, n_to_do, batch_size)
    ]
    n_batches = len(batches)
    show_batch_bar = n_batches > 1

    if not progress:
        for batch in batches:
            for s in batch:
                sid, lb = _process_single_subject(
                    s, n_eigenpairs, use_robust, sigma, backend,
                    cache_dir, save_matrices,
                )
                results[sid] = lb if return_results else None
            gc_gpu()
        return results, n_to_do, n_cached

    from rich.progress import Progress

    with Progress(*_rich_columns(), transient=False) as prog:
        total_task = prog.add_task(
            "Subjects", total=n_total, completed=n_cached,
        )
        batch_task = None
        if show_batch_bar:
            batch_task = prog.add_task("  Batch", total=0, visible=True)

        for b_idx, batch in enumerate(batches):
            if show_batch_bar and batch_task is not None:
                prog.reset(batch_task, total=len(batch), completed=0)
                prog.update(
                    batch_task,
                    description=f"  Batch {b_idx + 1}/{n_batches}",
                )

            for s in batch:
                sid, lb = _process_single_subject(
                    s, n_eigenpairs, use_robust, sigma, backend,
                    cache_dir, save_matrices,
                )
                results[sid] = lb if return_results else None
                prog.advance(total_task, 1)
                if show_batch_bar and batch_task is not None:
                    prog.advance(batch_task, 1)

            gc_gpu()

        if batch_task is not None:
            prog.update(batch_task, visible=False)

    return results, n_to_do, n_cached


def _print_all_cached(n: int) -> None:
    """Quick Rich print when everything is cached."""
    try:
        from rich.console import Console
        Console().print(
            f"[green]✓[/green] All {n} subjects found in cache "
            "— nothing to compute."
        )
    except ImportError:
        print(f"All {n} subjects found in cache — nothing to compute.")


def load_cached_eigenpairs(
    cache_dir: Union[str, Path],
    subject_id: str,
) -> LaplaceBeltrami:
    """
    Load a cached eigenpair result from disk.

    Parameters
    ----------
    cache_dir : str or Path
        Directory passed to :func:`batch_compute_eigenpairs`.
    subject_id : str
        Subject identifier.

    Returns
    -------
    LaplaceBeltrami
        Reconstructed object.  ``stiffness`` / ``mass`` are ``None``
        unless ``save_matrices=True`` was used during batch processing.
    """
    cache_file = Path(cache_dir) / f"{subject_id}.npz"
    if not cache_file.exists():
        raise FileNotFoundError(
            f"No cache for subject {subject_id!r} at {cache_file}"
        )

    data = np.load(str(cache_file), allow_pickle=False)

    stiffness = None
    mass = None
    if "stiffness_data" in data:
        stiffness = sp.csc_matrix(
            (data["stiffness_data"], data["stiffness_indices"],
             data["stiffness_indptr"]),
            shape=tuple(data["stiffness_shape"]),
        )
    if "mass_data" in data:
        mass = sp.csc_matrix(
            (data["mass_data"], data["mass_indices"],
             data["mass_indptr"]),
            shape=tuple(data["mass_shape"]),
        )

    return LaplaceBeltrami(
        stiffness=stiffness,
        mass=mass,
        eigenvalues=data["eigenvalues"],
        eigenvectors=data["eigenvectors"],
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
    backend: str = "auto",
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
        Accepted for API consistency.  Descriptor computation uses NumPy
        (matrix multiplications on the precomputed eigenvector basis).

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
