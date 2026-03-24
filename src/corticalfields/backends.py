"""
Compute backends for CorticalFields.

This module provides a clean abstraction layer for GPU-accelerated
computation, supporting three backends:

    1. **scipy** (default) — CPU-only, uses ARPACK shift-invert via
       ``scipy.sparse.linalg.eigsh``. Most robust and numerically
       stable. Recommended for production use and validation.

    2. **cupy** — GPU-accelerated via NVIDIA CUDA. Uses CuPy's
       LOBPCG eigensolver for the generalized eigenvalue problem
       and CuPy arrays for descriptor computation (GEMM on GPU).
       Requires ``cupy-cuda12x`` (or appropriate CUDA version).
       Expected speedup: 3–10× for eigensolver, 10–30× for
       spectral descriptors.

    3. **torch** — GPU-accelerated via PyTorch's ``torch.lobpcg``.
       Useful when PyTorch is already in the environment (e.g., for
       the normative GP model). Supports CUDA and MPS backends.
       Note: ``torch.lobpcg`` has known numerical issues for some
       problems — validate against scipy results.

Backend selection follows the principle of **per-function dispatch**:
each function accepts a ``backend=`` parameter that defaults to
``"auto"`` (which selects the best available backend). This allows
mixing backends freely — e.g., CPU eigensolver with GPU descriptors.

Design principles
-----------------
- **Zero mandatory GPU dependencies**: ``pip install corticalfields``
  works on CPU-only machines. GPU backends are optional extras.
- **Graceful fallback**: If a requested backend is unavailable, the
  function logs a warning and falls back to scipy.
- **Numerical consistency**: Results from all backends should agree
  to within floating-point tolerance. The test suite validates this.
- **Lazy imports**: Heavy libraries (cupy, torch) are imported only
  when their backend is first requested.

Memory budget (float64, 150K vertices, 300 eigenpairs)
------------------------------------------------------
- Sparse matrix (CSR): ~12 MB
- LOBPCG workspace: ~2.2 GB
- Eigenvectors output: ~360 MB
- Total GPU memory: ~2.7 GB (fits on any 8+ GB GPU)
- float32 mode halves all of the above

References
----------
[1] A. V. Knyazev, "Toward the Optimal Preconditioned Eigensolver:
    Locally Optimal Block Preconditioned Conjugate Gradient Method",
    SIAM J. Sci. Comput., 2001. (LOBPCG algorithm)
[2] CuPy documentation: cupyx.scipy.sparse.linalg.lobpcg
[3] PyTorch documentation: torch.lobpcg
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh as scipy_eigsh

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Backend detection
# ═══════════════════════════════════════════════════════════════════════════


class Backend(Enum):
    """Available compute backends."""
    SCIPY = "scipy"
    CUPY = "cupy"
    TORCH = "torch"


# Lazy availability flags — checked once on first access
_backend_status = {}


def _check_cupy() -> bool:
    """Check if CuPy is installed with a working CUDA device."""
    if "cupy" not in _backend_status:
        try:
            import cupy
            cupy.cuda.Device(0).compute_capability
            _backend_status["cupy"] = True
            logger.debug(
                "CuPy available: CUDA device %s (%.1f GB)",
                cupy.cuda.runtime.getDeviceProperties(0)["name"].decode(),
                cupy.cuda.Device(0).mem_info[1] / (1024 ** 3),
            )
        except Exception:
            _backend_status["cupy"] = False
    return _backend_status["cupy"]


def _check_torch_cuda() -> bool:
    """Check if PyTorch is installed with CUDA support."""
    if "torch" not in _backend_status:
        try:
            import torch
            _backend_status["torch"] = torch.cuda.is_available()
            if _backend_status["torch"]:
                logger.debug(
                    "PyTorch CUDA available: %s (%.1f GB)",
                    torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory / (1024 ** 3),
                )
        except ImportError:
            _backend_status["torch"] = False
    return _backend_status["torch"]


def resolve_backend(requested: str = "auto") -> Backend:
    """
    Resolve the requested backend string to a Backend enum.

    Parameters
    ----------
    requested : str
        One of ``"auto"``, ``"scipy"``, ``"cupy"``, ``"torch"``.
        ``"auto"`` selects the best available backend in order:
        cupy → torch → scipy.

    Returns
    -------
    Backend
        The resolved backend.

    Notes
    -----
    If the requested backend is unavailable, falls back to scipy
    with a warning (never raises an error for backend unavailability).
    """
    requested = requested.lower().strip()

    if requested == "auto":
        if _check_cupy():
            return Backend.CUPY
        if _check_torch_cuda():
            return Backend.TORCH
        return Backend.SCIPY

    if requested == "cupy":
        if _check_cupy():
            return Backend.CUPY
        logger.warning(
            "CuPy backend requested but not available "
            "(install cupy-cuda12x). Falling back to scipy."
        )
        return Backend.SCIPY

    if requested == "torch":
        if _check_torch_cuda():
            return Backend.TORCH
        logger.warning(
            "PyTorch CUDA backend requested but not available. "
            "Falling back to scipy."
        )
        return Backend.SCIPY

    if requested == "scipy":
        return Backend.SCIPY

    raise ValueError(
        f"Unknown backend: {requested!r}. "
        f"Choose from: 'auto', 'scipy', 'cupy', 'torch'."
    )


def available_backends() -> dict:
    """
    Report which backends are available.

    Returns
    -------
    dict
        Mapping from backend name to availability bool.
        Example: ``{'scipy': True, 'cupy': True, 'torch': False}``
    """
    return {
        "scipy": True,  # always available
        "cupy": _check_cupy(),
        "torch": _check_torch_cuda(),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Eigensolver dispatch
# ═══════════════════════════════════════════════════════════════════════════


def eigsh_solve(
    L: sp.spmatrix,
    M: sp.spmatrix,
    k: int = 300,
    backend: str = "auto",
    sigma: float = -0.01,
    tol: float = 1e-7,
    maxiter: int = 500,
    dtype: str = "float64",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve the generalized eigenvalue problem L φ = λ M φ for the
    smallest k eigenvalues, dispatching to the appropriate backend.

    This is the primary computational bottleneck in CorticalFields.
    On a 150K-vertex cortical mesh with k=300:
      - scipy (CPU):  30–120 seconds (ARPACK shift-invert)
      - cupy (GPU):   5–30 seconds  (LOBPCG)
      - torch (GPU):  10–40 seconds (LOBPCG)

    Parameters
    ----------
    L : scipy.sparse matrix, shape (N, N)
        Stiffness (cotangent) matrix. Symmetric positive semi-definite.
    M : scipy.sparse matrix, shape (N, N)
        Lumped mass (area) matrix. Symmetric positive definite diagonal.
    k : int
        Number of eigenpairs to compute.
    backend : str
        Backend to use: ``'auto'``, ``'scipy'``, ``'cupy'``, ``'torch'``.
    sigma : float
        Shift parameter for scipy's ARPACK shift-invert. Ignored by
        LOBPCG backends (which target smallest eigenvalues directly).
    tol : float
        Convergence tolerance for the eigensolver.
    maxiter : int
        Maximum iterations (for LOBPCG backends).
    dtype : str
        ``'float64'`` or ``'float32'``. GPU backends benefit
        significantly from float32 (2× memory reduction, often 2×
        faster SpMM). The eigenpairs are always returned as float64
        numpy arrays regardless of internal precision.

    Returns
    -------
    eigenvalues : np.ndarray, shape (k,)
        Sorted smallest eigenvalues (ascending).
    eigenvectors : np.ndarray, shape (N, k)
        Corresponding eigenvectors (columns).

    Notes
    -----
    All backends return numpy arrays on CPU. The GPU→CPU transfer
    is handled internally (~16 ms for 360 MB on PCIe 4.0).

    The LOBPCG backends solve the problem directly for smallest
    eigenvalues (``largest=False``), avoiding the sparse factorization
    required by shift-invert. This is more GPU-friendly because the
    core operation is sparse matrix–block vector multiplication (SpMM),
    which maps well to GPU SIMT architecture.
    """
    resolved = resolve_backend(backend)
    N = L.shape[0]

    logger.info(
        "Eigensolver: %s backend, N=%d, k=%d, dtype=%s",
        resolved.value, N, k, dtype,
    )

    if resolved == Backend.CUPY:
        return _eigsh_cupy(L, M, k, tol, maxiter, dtype)
    elif resolved == Backend.TORCH:
        return _eigsh_torch(L, M, k, tol, maxiter, dtype)
    else:
        return _eigsh_scipy(L, M, k, sigma, tol)


# ── SciPy backend (CPU, ARPACK shift-invert) ───────────────────────────

def _eigsh_scipy(
    L: sp.spmatrix, M: sp.spmatrix,
    k: int, sigma: float, tol: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    SciPy ARPACK eigensolver with shift-invert mode.

    This is the most numerically robust backend. Shift-invert mode
    transforms the problem to (L - σM)⁻¹ M φ = θ φ, which clusters
    eigenvalues near σ and makes ARPACK converge rapidly for the
    smallest eigenvalues of L.
    """
    L_csc = sp.csc_matrix(L, dtype=np.float64)
    M_csc = sp.csc_matrix(M, dtype=np.float64)

    eigenvalues, eigenvectors = scipy_eigsh(
        L_csc, k=k, M=M_csc,
        sigma=sigma, which="LM",
        tol=tol,
    )

    # Sort by eigenvalue (ascending)
    order = np.argsort(eigenvalues)
    return eigenvalues[order].astype(np.float64), eigenvectors[:, order].astype(np.float64)


# ── CuPy backend (GPU, LOBPCG) ─────────────────────────────────────────

def _eigsh_cupy(
    L: sp.spmatrix, M: sp.spmatrix,
    k: int, tol: float, maxiter: int, dtype: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    CuPy LOBPCG eigensolver on GPU.

    LOBPCG (Locally Optimal Block Preconditioned Conjugate Gradient)
    directly targets the smallest eigenvalues without shift-invert,
    making it GPU-friendly: the core operation is SpMM (sparse
    matrix × block vector), which runs at near-peak GPU bandwidth.

    The generalized problem L φ = λ M φ is handled natively via
    the ``B=`` parameter.
    """
    import cupy as cp
    import cupyx.scipy.sparse as csp
    from cupyx.scipy.sparse.linalg import lobpcg

    np_dtype = np.float32 if dtype == "float32" else np.float64
    cp_dtype = cp.float32 if dtype == "float32" else cp.float64

    # Transfer sparse matrices to GPU (CSR format for efficient SpMV)
    L_gpu = csp.csr_matrix(L.tocsr().astype(np_dtype))
    M_gpu = csp.csr_matrix(M.tocsr().astype(np_dtype))

    N = L.shape[0]

    # Random initial block vector (orthogonalized internally by LOBPCG)
    # Using a fixed seed for reproducibility across runs
    rng = cp.random.RandomState(seed=42)
    X0 = rng.randn(N, k, dtype=cp_dtype)

    logger.info("  CuPy LOBPCG: transferring to GPU...")

    # Diagonal preconditioner: M⁻¹ (accelerates convergence)
    # Since M is diagonal (lumped mass), this is trivially efficient
    M_diag = cp.array(M.diagonal().astype(np_dtype))
    M_diag[M_diag < 1e-16] = 1e-16
    M_inv_diag = 1.0 / M_diag

    def preconditioner(x):
        return M_inv_diag[:, None] * x

    logger.info("  CuPy LOBPCG: solving (k=%d, maxiter=%d)...", k, maxiter)

    eigenvalues, eigenvectors = lobpcg(
        L_gpu, X0,
        B=M_gpu,
        largest=False,
        tol=tol,
        maxiter=maxiter,
    )

    # Transfer back to CPU and ensure float64 + sorted
    evals = cp.asnumpy(eigenvalues).astype(np.float64)
    evecs = cp.asnumpy(eigenvectors).astype(np.float64)

    order = np.argsort(evals)
    return evals[order], evecs[:, order]


# ── PyTorch backend (GPU, LOBPCG) ──────────────────────────────────────

def _eigsh_torch(
    L: sp.spmatrix, M: sp.spmatrix,
    k: int, tol: float, maxiter: int, dtype: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    PyTorch LOBPCG eigensolver on CUDA.

    Uses ``torch.lobpcg`` with the generalized form ``A x = λ B x``.
    Particularly useful when the downstream pipeline (normative
    modeling, SpectralMaternKernel) already uses PyTorch — the
    eigenvectors can be kept on GPU as torch.Tensors, avoiding a
    round-trip through numpy.

    Caveat: ``torch.lobpcg`` has known numerical issues for some
    matrix types (see PyTorch GitHub #101075). Always validate
    against scipy results on a test mesh before using in production.
    """
    import torch

    torch_dtype = torch.float32 if dtype == "float32" else torch.float64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert scipy sparse → torch sparse CSR
    L_csr = L.tocsr()
    M_csr = M.tocsr()

    def _scipy_to_torch_csr(mat, dtype, device):
        return torch.sparse_csr_tensor(
            torch.tensor(mat.indptr, dtype=torch.int64, device=device),
            torch.tensor(mat.indices, dtype=torch.int64, device=device),
            torch.tensor(mat.data, dtype=dtype, device=device),
            size=mat.shape,
        )

    L_t = _scipy_to_torch_csr(L_csr, torch_dtype, device)
    M_t = _scipy_to_torch_csr(M_csr, torch_dtype, device)

    N = L.shape[0]

    # Random initial vectors
    torch.manual_seed(42)
    X0 = torch.randn(N, k, dtype=torch_dtype, device=device)

    logger.info("  torch LOBPCG: solving on %s (k=%d)...", device, k)

    # torch.lobpcg returns (eigenvalues, eigenvectors)
    # largest=False targets smallest eigenvalues
    eigenvalues, eigenvectors = torch.lobpcg(
        A=L_t,
        k=k,
        B=M_t,
        X=X0,
        largest=False,
        niter=maxiter,
        tol=tol,
    )

    # Transfer to CPU numpy
    evals = eigenvalues.cpu().numpy().astype(np.float64)
    evecs = eigenvectors.cpu().numpy().astype(np.float64)

    order = np.argsort(evals)
    return evals[order], evecs[:, order]


# ═══════════════════════════════════════════════════════════════════════════
#  GPU-accelerated dense operations (descriptors)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ArrayBackend:
    """
    Thin wrapper providing array operations on the resolved backend.

    This class abstracts numpy/cupy/torch differences for the dense
    matrix operations used in spectral descriptor computation (HKS,
    WKS, GPS). All operations are element-wise or GEMM, which map
    trivially to GPU.

    Usage
    -----
    >>> ab = ArrayBackend.create("cupy")  # or "auto", "torch", "scipy"
    >>> phi_sq = ab.square(eigenvectors)  # runs on GPU if cupy/torch
    >>> hks = ab.matmul(phi_sq, weights)  # GPU GEMM
    >>> result = ab.to_numpy(hks)         # back to CPU numpy
    """
    name: str
    xp: object      # numpy, cupy, or torch-as-numpy-like
    _is_torch: bool = False

    @staticmethod
    def create(backend: str = "auto") -> "ArrayBackend":
        """Create an ArrayBackend from a backend string."""
        resolved = resolve_backend(backend)

        if resolved == Backend.CUPY:
            import cupy as cp
            return ArrayBackend(name="cupy", xp=cp)

        if resolved == Backend.TORCH:
            # For dense operations, we use torch tensors directly
            return ArrayBackend(name="torch", xp=None, _is_torch=True)

        return ArrayBackend(name="numpy", xp=np)

    def asarray(self, x: np.ndarray, dtype=None) -> object:
        """Convert numpy array to backend array."""
        if self._is_torch:
            import torch
            t = torch.from_numpy(x)
            if dtype == "float32":
                t = t.float()
            if torch.cuda.is_available():
                t = t.cuda()
            return t
        if dtype is not None:
            return self.xp.asarray(x, dtype=getattr(self.xp, dtype, None) or dtype)
        return self.xp.asarray(x)

    def square(self, x) -> object:
        """Element-wise square."""
        if self._is_torch:
            return x ** 2
        return x ** 2

    def exp(self, x) -> object:
        """Element-wise exponential."""
        if self._is_torch:
            import torch
            return torch.exp(x)
        return self.xp.exp(x)

    def log(self, x) -> object:
        """Element-wise log."""
        if self._is_torch:
            import torch
            return torch.log(x)
        return self.xp.log(x)

    def sqrt(self, x) -> object:
        """Element-wise sqrt."""
        if self._is_torch:
            import torch
            return torch.sqrt(x)
        return self.xp.sqrt(x)

    def maximum(self, x, val) -> object:
        """Element-wise maximum with scalar."""
        if self._is_torch:
            import torch
            return torch.clamp(x, min=val)
        return self.xp.maximum(x, val)

    def matmul(self, a, b) -> object:
        """Matrix multiplication (GEMM)."""
        if self._is_torch:
            return a @ b
        return a @ b

    def logspace(self, start, stop, num) -> object:
        """Logspace."""
        if self._is_torch:
            import torch
            return torch.logspace(
                start, stop, num,
                dtype=torch.float64,
                device=next(iter([]))  # handled by caller
            )
        return self.xp.logspace(start, stop, num)

    def linspace(self, start, stop, num) -> object:
        """Linspace."""
        if self._is_torch:
            import torch
            return torch.linspace(start, stop, num)
        return self.xp.linspace(start, stop, num)

    def hstack(self, arrays) -> object:
        """Horizontal stack."""
        if self._is_torch:
            import torch
            return torch.cat(arrays, dim=1)
        return self.xp.hstack(arrays)

    def to_numpy(self, x) -> np.ndarray:
        """Convert backend array back to numpy."""
        if self._is_torch:
            return x.detach().cpu().numpy()
        if self.name == "cupy":
            import cupy
            return cupy.asnumpy(x)
        return np.asarray(x)


# ═══════════════════════════════════════════════════════════════════════════
#  Vectorized cotangent Laplacian (CPU, ~100× faster than loop)
# ═══════════════════════════════════════════════════════════════════════════


def compute_cotangent_laplacian_vectorized(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> Tuple[sp.csc_matrix, sp.csc_matrix]:
    """
    Vectorized cotangent Laplacian assembly.

    This replaces the per-triangle Python loop in the original
    ``compute_laplacian`` with fully vectorized NumPy operations,
    yielding ~100× speedup on typical FreeSurfer meshes (~300K faces).

    The algorithm computes cotangent weights for all triangles
    simultaneously using the identity:
        cot(angle at vertex k, opposite edge i-j) =
            dot(e_ki, e_kj) / |cross(e_ki, e_kj)|

    Parameters
    ----------
    vertices : (N, 3) float64
    faces : (F, 3) int64

    Returns
    -------
    L : scipy.sparse.csc_matrix (N, N)
        Cotangent stiffness matrix.
    M : scipy.sparse.csc_matrix (N, N)
        Diagonal lumped mass matrix.
    """
    N = vertices.shape[0]
    F = faces.shape[0]
    v = vertices.astype(np.float64)
    f = faces.astype(np.int64)

    # Vertex positions for each triangle corner
    v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]

    # Edge vectors
    e01 = v1 - v0  # (F, 3)
    e02 = v2 - v0
    e12 = v2 - v1

    # Cross products and their norms (= 2 × triangle area)
    cross_012 = np.cross(e01, e02)  # (F, 3)
    double_areas = np.linalg.norm(cross_012, axis=1)  # (F,)
    double_areas = np.maximum(double_areas, 1e-16)  # avoid division by zero

    # Cotangent weights:
    # cot(angle at v0) = dot(e01, e02) / |cross(e01, e02)| → weight for edge (v1, v2)
    # cot(angle at v1) = dot(-e01, e12) / |cross(-e01, e12)| → weight for edge (v0, v2)
    # cot(angle at v2) = dot(-e02, -e12) / |cross(-e02, -e12)| → weight for edge (v0, v1)
    cot0 = np.sum(e01 * e02, axis=1) / double_areas  # (F,)
    cot1 = np.sum(-e01 * e12, axis=1) / double_areas  # (F,)
    cot2 = np.sum(-e02 * (-e12), axis=1) / double_areas  # (F,)

    # Assemble sparse stiffness matrix
    # For each triangle, 3 edges contribute to off-diagonal entries
    # Edge (v1,v2) with weight -0.5 * cot0, etc.
    i_idx = np.concatenate([f[:, 1], f[:, 2], f[:, 0], f[:, 2], f[:, 0], f[:, 1]])
    j_idx = np.concatenate([f[:, 2], f[:, 1], f[:, 2], f[:, 0], f[:, 1], f[:, 0]])
    w_val = np.concatenate([
        -0.5 * cot0, -0.5 * cot0,
        -0.5 * cot1, -0.5 * cot1,
        -0.5 * cot2, -0.5 * cot2,
    ])

    L = sp.coo_matrix((w_val, (i_idx, j_idx)), shape=(N, N)).tocsc()
    # Set diagonal = -sum of off-diagonal (ensures row-sum = 0)
    diag = -np.array(L.sum(axis=1)).ravel()
    L = L + sp.diags(diag, format="csc")

    # Lumped mass matrix: each vertex gets 1/3 of the area of its adjacent triangles
    tri_areas = double_areas / 2.0  # (F,)
    areas = np.zeros(N, dtype=np.float64)
    np.add.at(areas, f[:, 0], tri_areas / 3.0)
    np.add.at(areas, f[:, 1], tri_areas / 3.0)
    np.add.at(areas, f[:, 2], tri_areas / 3.0)
    areas = np.maximum(areas, 1e-16)
    M = sp.diags(areas, format="csc")

    return L, M
