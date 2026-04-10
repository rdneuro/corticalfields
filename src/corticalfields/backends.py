"""
Compute backends for CorticalFields.

This module provides:

  1. **Laplacian assembly backends** — Three options in fallback order:
       • robust-laplacian (Sharp & Crane, SGP 2020) — intrinsic Delaunay
       • lapy (Deep-MI / Reuter lab) — vectorized FEM cotangent + CHOLMOD
       • built-in vectorized cotangent — pure NumPy, no extra deps

  2. **Eigensolver backends** — Three options:
       • scipy  (default) — ARPACK shift-invert, gold standard
       • cupy   — GPU LOBPCG via CuPy
       • torch  — GPU LOBPCG via PyTorch

  3. **Dense array backends** — For spectral descriptor computation:
       • numpy  (default)
       • cupy   — GPU GEMM via cuBLAS
       • torch  — GPU GEMM via PyTorch

  4. **Graph analysis backends** — For MSN/SSN graph metrics:
       • igraph   (recommended, 10-100× faster than NetworkX)
       • networkx (fallback)

Design principles
-----------------
- **Per-function dispatch**: each function accepts a ``backend=``
  parameter. This allows CPU eigensolver + GPU descriptors.
- **Zero mandatory GPU deps**: ``pip install corticalfields`` works
  on CPU-only machines. GPU/igraph/lapy are optional.
- **Graceful fallback**: Unavailable backends → warning + fallback.
- **Lazy imports**: Heavy libraries loaded only when first requested.

References
----------
[1] Sharp & Crane, "A Laplacian for Nonmanifold Triangle Meshes",
    SGP 2020.
[2] Reuter et al., LaPy — FEM on Triangle Meshes, Deep-MI group.
[3] Knyazev, "Toward the Optimal Preconditioned Eigensolver: LOBPCG",
    SIAM J. Sci. Comput., 2001.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh as scipy_eigsh

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Laplacian backend detection
# ═══════════════════════════════════════════════════════════════════════════

_laplacian_status: Dict[str, bool] = {}


def _check_robust_laplacian() -> bool:
    if "robust" not in _laplacian_status:
        try:
            import robust_laplacian
            _laplacian_status["robust"] = True
        except ImportError:
            _laplacian_status["robust"] = False
    return _laplacian_status["robust"]


def _check_lapy() -> bool:
    if "lapy" not in _laplacian_status:
        try:
            from lapy import TriaMesh, Solver
            _laplacian_status["lapy"] = True
        except ImportError:
            _laplacian_status["lapy"] = False
    return _laplacian_status["lapy"]


def available_laplacian_backends() -> Dict[str, bool]:
    """Report which Laplacian assembly backends are available."""
    return {
        "robust-laplacian": _check_robust_laplacian(),
        "lapy": _check_lapy(),
        "builtin": True,  # always available (pure NumPy)
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Eigensolver backend detection
# ═══════════════════════════════════════════════════════════════════════════


class Backend(Enum):
    """Available compute backends."""
    SCIPY = "scipy"
    CUPY = "cupy"
    TORCH = "torch"


_backend_status: Dict[str, bool] = {}


def _check_cupy() -> bool:
    if "cupy" not in _backend_status:
        try:
            import cupy
            cupy.cuda.Device(0).compute_capability
            _backend_status["cupy"] = True
            logger.debug("CuPy CUDA available.")
        except Exception:
            _backend_status["cupy"] = False
    return _backend_status["cupy"]


def _check_torch_cuda() -> bool:
    if "torch" not in _backend_status:
        try:
            import torch
            _backend_status["torch"] = torch.cuda.is_available()
        except ImportError:
            _backend_status["torch"] = False
    return _backend_status["torch"]


def resolve_backend(requested: str = "auto") -> Backend:
    """
    Resolve backend string to Backend enum.

    ``"auto"`` selects: cupy → torch → scipy.
    Falls back to scipy if the requested backend is unavailable.
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
        logger.warning("CuPy unavailable. Falling back to scipy.")
        return Backend.SCIPY

    if requested == "torch":
        if _check_torch_cuda():
            return Backend.TORCH
        logger.warning("PyTorch CUDA unavailable. Falling back to scipy.")
        return Backend.SCIPY

    if requested == "scipy":
        return Backend.SCIPY

    raise ValueError(f"Unknown backend: {requested!r}")


def available_backends() -> Dict[str, bool]:
    """Report which compute backends are available."""
    return {
        "scipy": True,
        "cupy": _check_cupy(),
        "torch": _check_torch_cuda(),
        **{f"laplacian_{k}": v for k, v in available_laplacian_backends().items()},
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Laplacian assembly — 3-level fallback chain
# ═══════════════════════════════════════════════════════════════════════════


def compute_laplacian(
    vertices: np.ndarray,
    faces: np.ndarray,
    method: str = "auto",
    use_cholmod: bool = True,
) -> Tuple[sp.csc_matrix, sp.csc_matrix]:
    """
    Build the Laplace-Beltrami stiffness and mass matrices.

    Fallback chain (when method="auto"):
      1. robust-laplacian — intrinsic Delaunay, handles non-manifold
      2. lapy — vectorized FEM cotangent, optional CHOLMOD
      3. builtin — pure NumPy vectorized cotangent

    Parameters
    ----------
    vertices : (N, 3) float64
    faces : (F, 3) int64
    method : str
        ``"auto"``, ``"robust"``, ``"lapy"``, or ``"builtin"``.
    use_cholmod : bool
        Use CHOLMOD acceleration in LaPy (requires scikit-sparse).

    Returns
    -------
    L : scipy.sparse.csc_matrix (N, N) — stiffness matrix
    M : scipy.sparse.csc_matrix (N, N) — mass matrix (diagonal lumped)
    """
    method = method.lower().strip()

    if method == "auto":
        if _check_robust_laplacian():
            method = "robust"
        elif _check_lapy():
            method = "lapy"
        else:
            method = "builtin"

    if method == "robust":
        if not _check_robust_laplacian():
            logger.warning("robust-laplacian not available; falling back.")
            return compute_laplacian(vertices, faces, method="lapy",
                                     use_cholmod=use_cholmod)
        import robust_laplacian
        logger.debug("Laplacian assembly: robust-laplacian (intrinsic Delaunay)")
        L, M = robust_laplacian.mesh_laplacian(
            np.asarray(vertices, dtype=np.float64),
            np.asarray(faces, dtype=np.int64),
        )
        return sp.csc_matrix(L), sp.csc_matrix(M)

    if method == "lapy":
        if not _check_lapy():
            logger.warning("LaPy not available; falling back to builtin.")
            return compute_laplacian(vertices, faces, method="builtin")
        logger.debug("Laplacian assembly: LaPy (FEM cotangent)")
        return _laplacian_lapy(vertices, faces, use_cholmod=use_cholmod)

    if method == "builtin":
        logger.debug("Laplacian assembly: builtin vectorized cotangent")
        return compute_cotangent_laplacian_vectorized(vertices, faces)

    raise ValueError(f"Unknown Laplacian method: {method!r}")


def _laplacian_lapy(
    vertices: np.ndarray,
    faces: np.ndarray,
    use_cholmod: bool = True,
) -> Tuple[sp.csc_matrix, sp.csc_matrix]:
    """
    Laplacian assembly via LaPy (Deep-MI / Reuter lab).

    LaPy uses a vectorized FEM cotangent scheme internally. With
    ``use_cholmod=True`` and scikit-sparse installed, the Solver
    uses CHOLMOD Cholesky factorization, which significantly
    accelerates the shift-invert step in eigsh.

    The mass matrix is returned in lumped (diagonal) form to match
    the convention used by the rest of CorticalFields.
    """
    from lapy import TriaMesh, Solver

    mesh = TriaMesh(
        np.asarray(vertices, dtype=np.float64),
        np.asarray(faces, dtype=np.int32),
    )

    # lump=True → diagonal mass matrix (consistent with CF convention)
    fem = Solver(mesh, lump=True, use_cholmod=use_cholmod)

    L = sp.csc_matrix(fem.stiffness)
    M = sp.csc_matrix(fem.mass)

    return L, M


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
    Solve L phi = lambda M phi for the smallest k eigenvalues.

    Performance (150K vertices, k=300):
      scipy (CPU, ARPACK):     30-120s  <-- most robust
      cupy  (GPU, Lanczos):     5-30s   <-- fastest, needs CuPy
      torch (GPU, ChFSI):      10-25s   <-- pure PyTorch, no lobpcg

    The torch backend uses Chebyshev-Filtered Subspace Iteration with
    mixed-precision (float32 SpMV + float64 Rayleigh-Ritz).  Peak VRAM
    usage is ~600 MB for N=150k, k=300 (5× less than the previous
    torch.lobpcg implementation).

    Returns
    -------
    eigenvalues : (k,) float64 — sorted ascending
    eigenvectors : (N, k) float64
    """
    resolved = resolve_backend(backend)

    logger.info(
        "Eigensolver: %s, N=%d, k=%d, dtype=%s",
        resolved.value, L.shape[0], k, dtype,
    )

    if resolved == Backend.CUPY:
        return _eigsh_cupy(L, M, k, tol, maxiter, dtype)
    elif resolved == Backend.TORCH:
        return _eigsh_torch(L, M, k, tol, maxiter, dtype)
    else:
        return _eigsh_scipy(L, M, k, sigma, tol)


def _eigsh_scipy(
    L: sp.spmatrix, M: sp.spmatrix,
    k: int, sigma: float, tol: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """SciPy ARPACK shift-invert — most robust, recommended."""
    L_csc = sp.csc_matrix(L, dtype=np.float64)
    M_csc = sp.csc_matrix(M, dtype=np.float64)

    eigenvalues, eigenvectors = scipy_eigsh(
        L_csc, k=k, M=M_csc,
        sigma=sigma, which="LM", tol=tol,
    )
    order = np.argsort(eigenvalues)
    return eigenvalues[order].astype(np.float64), \
           eigenvectors[:, order].astype(np.float64)


def _eigsh_cupy(
    L: sp.spmatrix, M: sp.spmatrix,
    k: int, tol: float, maxiter: int, dtype: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    CuPy GPU eigensolver for the generalised problem Lφ = λMφ.

    CuPy eigsh API (from official docs, CuPy 14.0.1):
        eigsh(a, k, *, which, v0, ncv, maxiter, tol, return_eigenvectors)
        which: 'LM', 'LA', 'SA'
        NOT supported: M= (no generalised), sigma= (no shift-invert)

    Strategy — spectral complement:
        1. Transform to standard problem: A = M^{-1/2} L M^{-1/2}
           (exact, M is diagonal lumped mass)
        2. Find λ_max of A  (eigsh, k=1, which='LM' — <1 sec)
        3. Form B = λ_max·I − A
           Largest eigenvalues of B = smallest eigenvalues of A
        4. eigsh(B, k, which='LM') — Lanczos converges fastest
           at spectral extremes, so this is robust for k=300
        5. Convert back: λ_i = λ_max − μ_i,  φ_i = M^{-1/2} y_i

    VRAM optimisation
    -----------------
    - A_gpu is deleted BEFORE B_gpu is allocated to halve peak
      sparse-matrix VRAM usage.
    - Memory pool freed after computation.
    """
    import cupy as cp
    import cupyx.scipy.sparse as csp
    from cupyx.scipy.sparse.linalg import eigsh as cupy_eigsh

    np_dtype = np.float32 if dtype == "float32" else np.float64

    # ── Step 1: Generalised → standard via M^{-1/2} ─────────────────
    M_diag = np.array(M.diagonal()).ravel().astype(np_dtype)
    M_diag = np.maximum(M_diag, 1e-16)
    M_inv_sqrt = (1.0 / np.sqrt(M_diag)).astype(np_dtype)

    D = sp.diags(M_inv_sqrt, format="csc", dtype=np_dtype)
    A_cpu = (D @ L.tocsc().astype(np_dtype) @ D).tocsc()

    try:
        # ── Step 2: Find λ_max (k=1, which='LM') ────────────────────
        A_gpu = csp.csc_matrix(A_cpu)
        lm_vals, _ = cupy_eigsh(A_gpu, k=1, which='LM')
        lambda_max = float(cp.asnumpy(lm_vals)[0]) * 1.01   # 1% buffer
        del lm_vals, _

        # ── Step 3: FREE A_gpu, then allocate B_gpu ─────────────────
        del A_gpu
        cp.get_default_memory_pool().free_all_blocks()

        N = A_cpu.shape[0]
        B_cpu = (sp.eye(N, format="csc", dtype=np_dtype) * np_dtype(lambda_max)
                 - A_cpu).tocsc()
        del A_cpu
        B_gpu = csp.csc_matrix(B_cpu)
        del B_cpu

        # ── Step 4: k largest of B = k smallest of A ────────────────
        mu, Y = cupy_eigsh(B_gpu, k=k, which='LM', maxiter=maxiter, tol=tol)
        del B_gpu

        # ── Step 5: Convert eigenvalues back, recover eigenvectors ───
        evals = (lambda_max - cp.asnumpy(mu)).astype(np.float64)
        evecs = (cp.asnumpy(Y) * M_inv_sqrt[:, None]).astype(np.float64)
        del mu, Y

    finally:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

    order = np.argsort(evals)
    return evals[order], evecs[:, order]


def _eigsh_torch(
    L: sp.spmatrix, M: sp.spmatrix,
    k: int, tol: float, maxiter: int, dtype: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    PyTorch GPU eigensolver — ChFSI with in-place VRAM management.

    Uses **Chebyshev-Filtered Subspace Iteration** (ChFSI) with:

    - **COO sparse format**: bypasses the buggy cuSPARSE CSR
      ``load_balancing_kernel`` (CUDA 13.0 bugs CUSPARSE-2380/2764/2612,
      PyTorch #188669) that causes out-of-bounds GPU memory reads after
      ~90-100 repeated SpMM calls, leading to unrecoverable PCIe hangs.
    - **In-place Chebyshev recurrence**: ``Tensor.add_(X, alpha=s)``
      and ``Tensor.mul_()`` eliminate ALL intermediate tensor allocations
      in the filter loop.
    - **Eager deallocation + ``empty_cache()`` every outer iteration**.
    - **VRAM watermark monitoring** with leak detection.
    - **``torch.no_grad()``** to prevent autograd graph leak.
    - **Periodic ``synchronize()``** to prevent driver watchdog timeout.
    - **``PYTORCH_CUDA_ALLOC_CONF``** with expandable segments for
      batch stability across hundreds of subjects.

    Per-subject VRAM budget (N=150k, k=100, m=120):
        Sparse A:     ~14 MB (CSR, f32, ~7 nnz/row)
        Subspace V:   N × m × 4 = ~72 MB
        SpMV temp:    N × m × 4 = ~72 MB  (freed each step)
        Ritz f64:   2 × N × m × 8 = ~288 MB (freed after Ritz)
        **Peak: ~446 MB** — leaves >23 GB free on RTX 3090.

    The critical constraint for batch stability is not peak usage but
    **fragmentation over subjects**.  In-place operations reduce the
    number of alloc/free cycles from ~30 per outer iteration (old) to
    ~3 (new), dramatically reducing caching-allocator fragmentation.

    Parameters
    ----------
    L, M, k, tol, maxiter, dtype : see ``eigsh_solve``

    Returns
    -------
    eigenvalues : (k,) float64 — sorted ascending
    eigenvectors : (N, k) float64

    References
    ----------
    [1] Y. Zhou, Y. Saad et al., "Chebyshev-filtered subspace iteration",
        J. Comput. Phys. 219 (2006) 172–184.
    """
    import gc
    import torch

    spmv_torch_dtype = torch.float32
    ritz_torch_dtype = torch.float64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_cuda = device.type == "cuda"
    N = L.shape[0]

    EXTRA = min(30, max(10, k // 10))
    m = k + EXTRA
    CHEB_DEGREE = 12
    POWER_ITERS = 30

    # ── VRAM watermark (start) ──────────────────────────────────────
    vram_start = 0
    if is_cuda:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        vram_start = torch.cuda.memory_allocated(0)

    logger.info(
        "  torch ChFSI: N=%d, k=%d, m=%d, deg=%d, "
        "VRAM_start=%.0f MB",
        N, k, m, CHEB_DEGREE, vram_start / 1e6,
    )

    # ── Step 1: Generalised → standard via M^{−½} (CPU) ────────────
    M_diag = np.array(M.diagonal()).ravel().astype(np.float64)
    M_diag = np.maximum(M_diag, 1e-16)
    M_inv_sqrt_np = 1.0 / np.sqrt(M_diag)

    D_sp = sp.diags(M_inv_sqrt_np.astype(np.float32), format="csc")
    A_cpu = (D_sp @ L.tocsc().astype(np.float32) @ D_sp).tocoo()
    del D_sp

    # ── Configure PyTorch CUDA allocator for batch stability ────────
    # expandable_segments avoids fragmentation from hundreds of
    # alloc/free cycles across subjects; max_split_size_mb prevents
    # the allocator from splitting large blocks into small slivers;
    # garbage_collection_threshold triggers proactive reclamation.
    import os
    _alloc_conf = (
        "expandable_segments:True,"
        "max_split_size_mb:128,"
        "garbage_collection_threshold:0.6"
    )
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = _alloc_conf
        # Only takes effect if called before first CUDA allocation,
        # so we also set it programmatically where possible:
        try:
            torch.cuda.memory._set_allocator_settings(_alloc_conf)
        except Exception:
            pass  # older PyTorch — env var is the fallback

    # ── scipy COO → torch sparse COO on device ─────────────────────
    # COO format bypasses the buggy cuSPARSE CSR load-balancing kernel
    # (cusparse::load_balancing_kernel<CsrMMOpAlg1>) that causes
    # out-of-bounds reads after ~90-100 repeated SpMM calls
    # (CUDA 13.0 bugs CUSPARSE-2380, CUSPARSE-2764, CUSPARSE-2612;
    # confirmed in PyTorch issue #188669; fixed in CUDA 13.2).
    # COO routes through an older, more stable cuSPARSE code path.
    def _to_coo(m_coo):
        indices = np.vstack([m_coo.row.astype(np.int64),
                             m_coo.col.astype(np.int64)])
        return torch.sparse_coo_tensor(
            torch.from_numpy(indices).to(device),
            torch.from_numpy(m_coo.data.astype(np.float32)).to(device),
            size=m_coo.shape, dtype=spmv_torch_dtype,
        ).coalesce()

    try:
        A_t = _to_coo(A_cpu)
        del A_cpu

        with torch.no_grad():

            # ── Step 2: λ_max via power iteration ───────────────────
            torch.manual_seed(42)
            v = torch.randn(N, 1, dtype=spmv_torch_dtype, device=device)
            v.div_(v.norm())
            for pi in range(POWER_ITERS):
                v = torch.sparse.mm(A_t, v)
                v.div_(v.norm())
                if is_cuda and pi % 10 == 9:
                    torch.cuda.synchronize()

            Av = torch.sparse.mm(A_t, v)
            lambda_max = float((v.T @ Av).item()) * 1.05
            del v, Av
            if is_cuda:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            logger.info("    λ_max ≈ %.4f", lambda_max)

            # ── Step 3: ChFSI outer loop ────────────────────────────
            torch.manual_seed(42)
            V = torch.randn(N, m, dtype=spmv_torch_dtype, device=device)
            V, _ = torch.linalg.qr(V)

            lambda_cut = lambda_max * (2.0 * m / N)
            lambda_cut = max(lambda_cut, lambda_max * 0.01)

            converged = False
            max_res = float("inf")

            for outer in range(maxiter):

                # ── Chebyshev filter (IN-PLACE) ─────────────────────
                # All arithmetic uses .add_(), .mul_() to avoid temps.
                # Only torch.sparse.mm allocates (no out= support).
                e = (lambda_max - lambda_cut) / 2.0
                cc = (lambda_max + lambda_cut) / 2.0
                if e < 1e-10:
                    e = lambda_max * 0.5
                    cc = lambda_max * 0.5

                sigma = e / cc if abs(cc) > 1e-12 else 1.0
                sigma1 = sigma

                # Y₁ = (σ₁/e) · (A·V − c·V)
                # In-place: AV = sparse.mm(A, V); AV -= c*V; AV *= σ₁/e
                Y_curr = torch.sparse.mm(A_t, V)     # (N,m) NEW alloc
                Y_curr.add_(V, alpha=-cc)             # in-place
                Y_curr.mul_(sigma1 / e)               # in-place
                Y_prev = V.clone()                    # need a copy (V reused)

                for d in range(2, CHEB_DEGREE + 1):
                    sigma_new = 1.0 / (2.0 / sigma1 - sigma)

                    # Y_next = (2σ_new/e)(A·Y_curr − c·Y_curr) − σ·σ_new·Y_prev
                    # In-place on the SpMV output:
                    Y_next = torch.sparse.mm(A_t, Y_curr)  # NEW alloc
                    Y_next.add_(Y_curr, alpha=-cc)          # -= c * Y_curr
                    Y_next.mul_(2.0 * sigma_new / e)        # *= 2σ/e
                    Y_next.add_(Y_prev, alpha=-(sigma * sigma_new))

                    # Rotate buffers — reuse memory
                    Y_prev = Y_curr    # old Y_curr becomes Y_prev
                    Y_curr = Y_next    # new result becomes Y_curr
                    sigma = sigma_new
                    # Y_next ref dropped; old Y_prev eligible for GC

                    if is_cuda and d % 4 == 0:
                        torch.cuda.synchronize()

                del Y_prev  # free last-gen buffer
                if is_cuda:
                    torch.cuda.synchronize()

                # ── QR ──────────────────────────────────────────────
                V, _ = torch.linalg.qr(Y_curr)
                del Y_curr

                # ── Rayleigh–Ritz (f64 for accuracy) ────────────────
                AV_f32 = torch.sparse.mm(A_t, V)            # (N,m) f32
                V64 = V.to(ritz_torch_dtype)                 # (N,m) f64
                AV64 = AV_f32.to(ritz_torch_dtype)           # (N,m) f64
                del AV_f32  # free f32 copy NOW

                H = V64.T @ AV64                             # (m,m) f64
                H = 0.5 * (H + H.T)                          # symmetrise (safe)
                ritz_vals, ritz_vecs = torch.linalg.eigh(H)
                del H

                # ── Convergence check ───────────────────────────────
                # Compute residual norms without large (N,k) temporaries:
                # res_i = ||AV64 @ s_i - λ_i * V64 @ s_i||
                S_k = ritz_vecs[:, :k]                       # (m,k) f64 — view
                Z_k = V64 @ S_k                              # (N,k) f64
                AZ_k = AV64 @ S_k                            # (N,k) f64
                del V64, AV64  # free the two big f64 blocks NOW

                # In-place: scale Z_k columns by eigenvalues, then subtract
                Z_k.mul_(ritz_vals[:k].unsqueeze(0))  # Z_k[:,i] *= λ_i
                AZ_k.sub_(Z_k)                         # AZ_k -= λ·Z_k
                max_res = float(AZ_k.norm(dim=0).max().item())
                del Z_k, AZ_k, S_k

                if is_cuda:
                    torch.cuda.synchronize()

                if outer % 5 == 0 or max_res < tol:
                    logger.info(
                        "    ChFSI iter %2d: res=%.2e, λ_cut=%.4f",
                        outer, max_res, lambda_cut,
                    )

                if max_res < tol:
                    converged = True
                    break

                # Rotate V into Ritz basis
                V = V @ ritz_vecs[:, :m].to(spmv_torch_dtype)

                # Refine λ_cut
                if ritz_vals.shape[0] > k:
                    lambda_cut = float(ritz_vals[m - 1].item()) * 1.5
                    lambda_cut = min(lambda_cut, lambda_max * 0.95)

                # ── Aggressive VRAM cleanup EVERY iteration ─────────
                if is_cuda:
                    torch.cuda.empty_cache()

            # ── end outer loop ──────────────────────────────────────

            if not converged:
                logger.warning(
                    "  ChFSI did not converge in %d iters "
                    "(res=%.2e > tol=%.1e).",
                    maxiter, max_res, tol,
                )

            # ── Extract eigenpairs ──────────────────────────────────
            evals_t = ritz_vals[:k]                          # (k,) f64
            evecs_t = V.to(ritz_torch_dtype) @ ritz_vecs[:, :k]  # (N,k) f64
            del V, ritz_vals, ritz_vecs

            M_inv_sqrt_t = torch.from_numpy(
                M_inv_sqrt_np
            ).to(dtype=ritz_torch_dtype, device=device).unsqueeze(1)
            evecs_t.mul_(M_inv_sqrt_t)                       # in-place
            del M_inv_sqrt_t

            if is_cuda:
                torch.cuda.synchronize()
            evals = evals_t.cpu().numpy().astype(np.float64)
            evecs = evecs_t.cpu().numpy().astype(np.float64)
            del evals_t, evecs_t

    finally:
        if is_cuda:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()  # double-tap after gc frees python refs
            vram_end = torch.cuda.memory_allocated(0)
            delta = vram_end - vram_start
            if delta > 1e6:  # > 1 MB leak
                logger.warning(
                    "  VRAM leak detected: +%.1f MB (start=%.0f, end=%.0f)",
                    delta / 1e6, vram_start / 1e6, vram_end / 1e6,
                )

    order = np.argsort(evals)
    return evals[order], evecs[:, order]


# ═══════════════════════════════════════════════════════════════════════════
#  Dense array backend (for spectral descriptors)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ArrayBackend:
    """
    Thin wrapper abstracting numpy/cupy/torch for dense GEMM ops
    used in spectral descriptor computation (HKS, WKS, GPS).

    Usage
    -----
    >>> ab = ArrayBackend.create("cupy")
    >>> phi_sq = ab.square(eigenvectors)   # GPU
    >>> hks = ab.matmul(phi_sq, weights)   # cuBLAS GEMM
    >>> result = ab.to_numpy(hks)          # back to CPU
    """
    name: str
    xp: object
    _is_torch: bool = False

    @staticmethod
    def create(backend: str = "auto") -> "ArrayBackend":
        resolved = resolve_backend(backend)
        if resolved == Backend.CUPY:
            import cupy as cp
            return ArrayBackend(name="cupy", xp=cp)
        if resolved == Backend.TORCH:
            return ArrayBackend(name="torch", xp=None, _is_torch=True)
        return ArrayBackend(name="numpy", xp=np)

    def asarray(self, x: np.ndarray, dtype=None) -> object:
        if self._is_torch:
            import torch
            t = torch.from_numpy(np.ascontiguousarray(x))
            if dtype == "float32":
                t = t.float()
            if torch.cuda.is_available():
                t = t.cuda()
            return t
        if dtype is not None:
            return self.xp.asarray(x, dtype=getattr(self.xp, dtype, None) or dtype)
        return self.xp.asarray(x)

    def square(self, x):
        return x ** 2

    def exp(self, x):
        if self._is_torch:
            import torch
            return torch.exp(x)
        return self.xp.exp(x)

    def log(self, x):
        if self._is_torch:
            import torch
            return torch.log(x)
        return self.xp.log(x)

    def sqrt(self, x):
        if self._is_torch:
            import torch
            return torch.sqrt(x)
        return self.xp.sqrt(x)

    def maximum(self, x, val):
        if self._is_torch:
            import torch
            return torch.clamp(x, min=val)
        return self.xp.maximum(x, val)

    def matmul(self, a, b):
        return a @ b

    def to_numpy(self, x) -> np.ndarray:
        if self._is_torch:
            return x.detach().cpu().numpy()
        if self.name == "cupy":
            import cupy
            return cupy.asnumpy(x)
        return np.asarray(x)

    def cleanup(self) -> None:
        """Free GPU memory held by this backend."""
        if self._is_torch:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        elif self.name == "cupy":
            import cupy
            cupy.get_default_memory_pool().free_all_blocks()


# ═══════════════════════════════════════════════════════════════════════════
#  Vectorized cotangent Laplacian (builtin, ~100x faster than loop)
# ═══════════════════════════════════════════════════════════════════════════


def compute_cotangent_laplacian_vectorized(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> Tuple[sp.csc_matrix, sp.csc_matrix]:
    """
    Fully vectorized cotangent Laplacian — pure NumPy, no extra deps.

    ~100x faster than the per-triangle Python loop. Used as the final
    fallback when neither robust-laplacian nor LaPy is installed.
    """
    N = vertices.shape[0]
    v = vertices.astype(np.float64)
    f = faces.astype(np.int64)

    v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    e01, e02, e12 = v1 - v0, v2 - v0, v2 - v1

    cross_012 = np.cross(e01, e02)
    double_areas = np.maximum(np.linalg.norm(cross_012, axis=1), 1e-16)

    cot0 = np.sum(e01 * e02, axis=1) / double_areas
    cot1 = np.sum(-e01 * e12, axis=1) / double_areas
    cot2 = np.sum(-e02 * (-e12), axis=1) / double_areas

    i_idx = np.concatenate([f[:, 1], f[:, 2], f[:, 0], f[:, 2], f[:, 0], f[:, 1]])
    j_idx = np.concatenate([f[:, 2], f[:, 1], f[:, 2], f[:, 0], f[:, 1], f[:, 0]])
    w_val = np.concatenate([
        -0.5 * cot0, -0.5 * cot0,
        -0.5 * cot1, -0.5 * cot1,
        -0.5 * cot2, -0.5 * cot2,
    ])

    L = sp.coo_matrix((w_val, (i_idx, j_idx)), shape=(N, N)).tocsc()
    diag = -np.array(L.sum(axis=1)).ravel()
    L = L + sp.diags(diag, format="csc")

    tri_areas = double_areas / 2.0
    areas = np.zeros(N, dtype=np.float64)
    np.add.at(areas, f[:, 0], tri_areas / 3.0)
    np.add.at(areas, f[:, 1], tri_areas / 3.0)
    np.add.at(areas, f[:, 2], tri_areas / 3.0)
    areas = np.maximum(areas, 1e-16)
    M = sp.diags(areas, format="csc")

    return L, M


# ═══════════════════════════════════════════════════════════════════════════
#  Vectorized MSN/SSN construction (replaces scipy pearsonr loop)
# ═══════════════════════════════════════════════════════════════════════════


def vectorized_correlation_matrix(
    profiles: np.ndarray,
    method: str = "pearson",
    fisher_z: bool = True,
) -> np.ndarray:
    """
    Vectorized inter-ROI correlation matrix.

    Replaces the O(R^2) loop over ``scipy.stats.pearsonr`` calls with
    a single ``np.corrcoef`` call (~100x faster for R=200).

    Parameters
    ----------
    profiles : (R, F) array — mean feature profile per ROI
    method : 'pearson' or 'spearman'
    fisher_z : bool — apply Fisher z-transform

    Returns
    -------
    corr : (R, R) symmetric correlation matrix
    """
    if method == "spearman":
        from scipy.stats import rankdata
        profiles = np.apply_along_axis(rankdata, 1, profiles)

    corr = np.corrcoef(profiles)
    np.fill_diagonal(corr, 0.0)

    if fisher_z:
        corr = np.arctanh(np.clip(corr, -0.9999, 0.9999))
        np.fill_diagonal(corr, 0.0)

    return corr


# ═══════════════════════════════════════════════════════════════════════════
#  Graph metrics — igraph (fast) -> networkx (fallback)
# ═══════════════════════════════════════════════════════════════════════════


def _check_igraph() -> bool:
    if "igraph" not in _backend_status:
        try:
            import igraph
            _backend_status["igraph"] = True
        except ImportError:
            _backend_status["igraph"] = False
    return _backend_status["igraph"]


def compute_graph_metrics(
    adjacency: np.ndarray,
    threshold: Optional[float] = None,
    density: Optional[float] = None,
    backend: str = "auto",
) -> Dict[str, object]:
    """
    Graph metrics from a similarity matrix.

    Uses igraph (10-100x faster) when available, falls back to NetworkX.

    Parameters
    ----------
    adjacency : (R, R) symmetric weight matrix
    threshold : float — hard threshold on edge weights
    density : float — target density (overrides threshold)
    backend : 'auto', 'igraph', or 'networkx'

    Returns
    -------
    dict with: degree, clustering, efficiency, betweenness,
               modularity, strength, assortativity
    """
    W = adjacency.copy()

    if density is not None:
        flat = np.sort(W[np.triu_indices_from(W, k=1)])[::-1]
        n_edges = int(density * len(flat))
        if 0 < n_edges <= len(flat):
            threshold = flat[n_edges - 1]

    if threshold is not None:
        W[W < threshold] = 0.0

    if backend == "auto":
        backend = "igraph" if _check_igraph() else "networkx"

    if backend == "igraph" and _check_igraph():
        return _graph_metrics_igraph(W)
    return _graph_metrics_networkx(W)


def _graph_metrics_igraph(W: np.ndarray) -> Dict[str, object]:
    """Graph metrics via igraph (10-100x faster than NetworkX)."""
    import igraph as ig

    R = W.shape[0]
    sources, targets = np.where(np.triu(W, k=1) > 0)
    weights = W[sources, targets]

    G = ig.Graph(n=R, edges=list(zip(sources.tolist(), targets.tolist())),
                 directed=False)
    G.es["weight"] = weights.tolist()

    result = {
        "n_nodes": G.vcount(),
        "n_edges": G.ecount(),
        "density": G.density(),
        "strength": dict(enumerate(G.strength(weights="weight"))),
        "degree": dict(enumerate(G.degree())),
        "clustering": dict(enumerate(G.transitivity_local_undirected(
            weights="weight"))),
    }

    btw = G.betweenness(weights="weight")
    norm = max((R - 1) * (R - 2) / 2, 1)
    result["betweenness"] = dict(enumerate([b / norm for b in btw]))

    try:
        sp_mat = np.array(G.shortest_paths(weights="weight"))
        sp_mat[sp_mat == 0] = np.inf
        np.fill_diagonal(sp_mat, np.inf)
        inv_sp = 1.0 / sp_mat
        inv_sp[~np.isfinite(inv_sp)] = 0.0
        result["global_efficiency"] = inv_sp.sum() / (R * (R - 1))
    except Exception:
        result["global_efficiency"] = np.nan

    try:
        partition = G.community_multilevel(weights="weight")
        result["modularity"] = partition.modularity
        result["communities"] = partition.membership
    except Exception:
        result["modularity"] = np.nan
        result["communities"] = []

    try:
        result["assortativity"] = G.assortativity_degree(directed=False)
    except Exception:
        result["assortativity"] = np.nan

    return result


def _graph_metrics_networkx(W: np.ndarray) -> Dict[str, object]:
    """Graph metrics via NetworkX (fallback)."""
    import networkx as nx
    from networkx.algorithms.community import greedy_modularity_communities

    G = nx.from_numpy_array(W)
    zero_edges = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] <= 0]
    G.remove_edges_from(zero_edges)

    result = {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "density": nx.density(G),
        "strength": dict(G.degree(weight="weight")),
        "degree": dict(G.degree()),
        "clustering": nx.clustering(G, weight="weight"),
    }

    try:
        result["global_efficiency"] = nx.global_efficiency(G)
    except Exception:
        result["global_efficiency"] = np.nan

    result["betweenness"] = nx.betweenness_centrality(G, weight="weight")

    try:
        communities = list(greedy_modularity_communities(G, weight="weight"))
        result["modularity"] = nx.community.modularity(G, communities, weight="weight")
        result["communities"] = communities
    except Exception:
        result["modularity"] = np.nan
        result["communities"] = []

    try:
        result["assortativity"] = nx.degree_assortativity_coefficient(G, weight="weight")
    except Exception:
        result["assortativity"] = np.nan

    return result
