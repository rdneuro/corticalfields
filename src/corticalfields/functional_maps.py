"""
Functional maps for cortical correspondence and asymmetry analysis.

Implements the functional maps framework (Ovsjanikov et al., 2012) applied
to cortical surfaces. A **functional map** between two surfaces is a small
matrix **C** (k × k) that maps functions expressed in the Laplace–Beltrami
eigenbasis of one surface to the eigenbasis of the other.

For cortical analysis, the key application is **inter-hemispheric
correspondence**: the C matrix between left and right hemisphere encodes
how each spatial frequency of one hemisphere maps to each frequency of
the other. In a perfectly symmetric brain, C would be diagonal (each
eigenfunction maps to its counterpart). The **off-diagonal energy** of C
is therefore a principled, continuous, atlas-free measure of asymmetry
that decomposes by spatial frequency.

Pipeline
--------
1. Compute Laplace–Beltrami eigenpairs on both surfaces (via ``spectral.py``
   or ``pointcloud.py``).
2. Compute descriptor matrices (HKS/WKS) in the spectral basis.
3. Solve for the initial C matrix via least-squares.
4. Refine with ZoomOut (Melzi et al., 2019) for higher-resolution maps.
5. Extract asymmetry metrics from the refined C matrix.

Dependencies
------------
- Core: numpy, scipy (always available)
- Optional: ``pyFM`` for advanced functional map operations

References
----------
Ovsjanikov, M., Ben-Chen, M., Solomon, J., Butscher, A., & Guibas, L.
    (2012). Functional maps: a flexible representation of maps between
    shapes. ACM Trans. Graphics, 31(4).
Melzi, S., Ren, J., Rodolà, E., Sharma, A., Wonka, P., & Ovsjanikov, M.
    (2019). ZoomOut: Spectral Upsampling for Efficient Shape Correspondence.
    ACM Trans. Graphics, 38(6).
Lombaert, H., et al. (2015). Brain Transfer: Spectral analysis of cortical
    surfaces and functional maps. IPMI 2015.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import orthogonal_procrustes

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class FunctionalMap:
    """
    A functional map between two surfaces.

    Parameters
    ----------
    C : np.ndarray, shape (k2, k1)
        Functional map matrix mapping functions from source eigenbasis
        to target eigenbasis. ``C @ f_coeffs_source = f_coeffs_target``.
    k_source : int
        Number of eigenfunctions used on source surface.
    k_target : int
        Number of eigenfunctions used on target surface.
    source_eigenvalues : np.ndarray, shape (k1,)
        Eigenvalues of the source surface LBO.
    target_eigenvalues : np.ndarray, shape (k2,)
        Eigenvalues of the target surface LBO.
    metadata : dict
        Computation metadata (method, descriptors used, etc.).
    """

    C: np.ndarray
    k_source: int
    k_target: int
    source_eigenvalues: np.ndarray
    target_eigenvalues: np.ndarray
    metadata: Dict = field(default_factory=dict)

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of the C matrix."""
        return self.C.shape

    @property
    def off_diagonal_energy(self) -> float:
        """
        Frobenius norm of the off-diagonal elements of C.

        For a square C matrix, this measures deviation from perfect
        correspondence (where C would be diagonal).
        """
        k = min(self.C.shape)
        C_sq = self.C[:k, :k]
        diag_energy = np.sum(np.diag(C_sq) ** 2)
        total_energy = np.sum(C_sq ** 2)
        return float(np.sqrt(max(total_energy - diag_energy, 0.0)))

    @property
    def diagonal_dominance(self) -> float:
        """
        Ratio of diagonal energy to total energy.

        A value of 1.0 indicates perfect correspondence (identity map);
        lower values indicate greater asymmetry.
        """
        k = min(self.C.shape)
        C_sq = self.C[:k, :k]
        diag_energy = np.sum(np.diag(C_sq) ** 2)
        total_energy = np.sum(C_sq ** 2)
        if total_energy < 1e-12:
            return 1.0
        return float(diag_energy / total_energy)

    def frequency_band_energy(
        self,
        bands: Optional[List[Tuple[int, int]]] = None,
    ) -> Dict[str, float]:
        """
        Decompose off-diagonal energy by spectral frequency band.

        Parameters
        ----------
        bands : list of (start, end) tuples or None
            Eigenfunction index ranges for each band. If None, uses
            default bands: low (0–20), mid (20–100), high (100+).

        Returns
        -------
        dict
            Band name → off-diagonal Frobenius norm for that block.
        """
        k = min(self.C.shape)
        C_sq = self.C[:k, :k]

        if bands is None:
            bands_dict = {
                "low_freq": (0, min(20, k)),
                "mid_freq": (min(20, k), min(100, k)),
                "high_freq": (min(100, k), k),
            }
        else:
            bands_dict = {f"band_{i}": b for i, b in enumerate(bands)}

        result = {}
        for name, (start, end) in bands_dict.items():
            if start >= end:
                result[name] = 0.0
                continue
            block = C_sq[start:end, start:end]
            diag_e = np.sum(np.diag(block) ** 2)
            total_e = np.sum(block ** 2)
            result[name] = float(np.sqrt(max(total_e - diag_e, 0.0)))

        return result

    def dominant_asymmetry_modes(
        self,
        n_modes: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract dominant asymmetry modes via SVD of the off-diagonal part.

        Parameters
        ----------
        n_modes : int
            Number of modes to extract.

        Returns
        -------
        U : np.ndarray, shape (k, n_modes)
            Left singular vectors (target basis).
        S : np.ndarray, shape (n_modes,)
            Singular values (asymmetry strength per mode).
        Vt : np.ndarray, shape (n_modes, k)
            Right singular vectors (source basis).
        """
        k = min(self.C.shape)
        C_sq = self.C[:k, :k].copy()
        # Remove diagonal to isolate off-diagonal structure
        np.fill_diagonal(C_sq, 0.0)

        U, S, Vt = np.linalg.svd(C_sq, full_matrices=False)
        n = min(n_modes, len(S))
        return U[:, :n], S[:n], Vt[:n, :]


# ═══════════════════════════════════════════════════════════════════════════
# Descriptor computation in spectral basis
# ═══════════════════════════════════════════════════════════════════════════


def _project_to_basis(
    descriptors: np.ndarray,
    eigenvectors: np.ndarray,
    mass: Optional["sp.csc_matrix"] = None,
) -> np.ndarray:
    """
    Project vertex-wise descriptors onto the LB eigenbasis.

    Parameters
    ----------
    descriptors : np.ndarray, shape (N, D)
        Per-vertex descriptors (HKS, WKS, etc.).
    eigenvectors : np.ndarray, shape (N, K)
        LB eigenvectors.
    mass : scipy.sparse or None
        Mass matrix for proper L²-inner product. If None, uses
        uniform weighting.

    Returns
    -------
    coeffs : np.ndarray, shape (K, D)
        Descriptor coefficients in the eigenbasis.
    """
    if mass is not None:
        # Proper L² projection: c_k = φ_k^T M f
        import scipy.sparse as sp
        M_desc = mass @ descriptors  # (N, D)
        return eigenvectors.T @ M_desc  # (K, D)
    else:
        return eigenvectors.T @ descriptors  # (K, D)


def compute_descriptor_matrix(
    lb: "LaplaceBeltrami",
    descriptor_type: str = "hks",
    n_descriptors: int = 100,
    **kwargs,
) -> np.ndarray:
    """
    Compute spectral descriptors and project to the eigenbasis.

    This produces the matrix A such that A[k, d] is the coefficient
    of the d-th descriptor function in the k-th eigenfunction.

    Parameters
    ----------
    lb : LaplaceBeltrami
        Spectral decomposition of a surface.
    descriptor_type : ``'hks'``, ``'wks'``, or ``'both'``
        Which descriptors to compute.
    n_descriptors : int
        Number of descriptor scales/energies.
    **kwargs
        Additional arguments passed to HKS/WKS computation.

    Returns
    -------
    A : np.ndarray, shape (K, D)
        Projected descriptor coefficients.
    """
    from corticalfields.spectral import (
        heat_kernel_signature,
        wave_kernel_signature,
    )

    descriptors_list = []

    if descriptor_type in ("hks", "both"):
        n_hks = n_descriptors if descriptor_type == "hks" else n_descriptors // 2
        hks = heat_kernel_signature(lb, n_scales=n_hks, **kwargs)
        descriptors_list.append(hks)

    if descriptor_type in ("wks", "both"):
        n_wks = n_descriptors if descriptor_type == "wks" else n_descriptors // 2
        wks = wave_kernel_signature(lb, n_energies=n_wks, **kwargs)
        descriptors_list.append(wks)

    descriptors = np.hstack(descriptors_list)  # (N, D)
    return _project_to_basis(descriptors, lb.eigenvectors, lb.mass)


# ═══════════════════════════════════════════════════════════════════════════
# Functional map computation
# ═══════════════════════════════════════════════════════════════════════════


def compute_functional_map(
    lb_source: "LaplaceBeltrami",
    lb_target: "LaplaceBeltrami",
    k: int = 50,
    descriptor_type: str = "hks",
    n_descriptors: int = 100,
    alpha_desc: float = 1.0,
    alpha_lap: float = 1e-3,
    alpha_orient: float = 0.0,
    A_source: Optional[np.ndarray] = None,
    A_target: Optional[np.ndarray] = None,
    backend: str = "numpy",
) -> FunctionalMap:
    """
    Compute a functional map between two surfaces.

    Solves the optimisation problem:

    .. math::

        C^* = \\arg\\min_C \\| C A_{\\text{src}} - A_{\\text{tgt}} \\|_F^2
              + \\alpha_{\\text{lap}} \\| \\Delta_{\\text{tgt}} C
                - C \\Delta_{\\text{src}} \\|_F^2

    where A are descriptor matrices projected to the eigenbasis, and the
    Laplacian commutativity term encourages C to be near-diagonal.

    Parameters
    ----------
    lb_source : LaplaceBeltrami
        Spectral decomposition of the source surface.
    lb_target : LaplaceBeltrami
        Spectral decomposition of the target surface.
    k : int
        Number of eigenfunctions to use for the functional map.
    descriptor_type : ``'hks'``, ``'wks'``, or ``'both'``
        Which spectral descriptors to use for correspondence.
    n_descriptors : int
        Number of descriptor scales.
    alpha_desc : float
        Weight for the descriptor preservation term.
    alpha_lap : float
        Weight for the Laplacian commutativity regularisation.
        Higher values enforce more diagonal C (smoother maps).
    alpha_orient : float
        Weight for orientation-preserving regularisation (experimental).
    A_source : np.ndarray or None
        Pre-computed source descriptor matrix, shape (K_src, D).
        If None, computed from ``lb_source``.
    A_target : np.ndarray or None
        Pre-computed target descriptor matrix, shape (K_tgt, D).

    Returns
    -------
    FunctionalMap
        The computed functional map with metadata.

    Examples
    --------
    >>> from corticalfields.spectral import compute_eigenpairs
    >>> from corticalfields.functional_maps import compute_functional_map
    >>> lb_lh = compute_eigenpairs(verts_lh, faces_lh, n_eigenpairs=100)
    >>> lb_rh = compute_eigenpairs(verts_rh, faces_rh, n_eigenpairs=100)
    >>> fm = compute_functional_map(lb_lh, lb_rh, k=50)
    >>> print(f"Off-diagonal energy: {fm.off_diagonal_energy:.4f}")
    >>> print(f"Diagonal dominance: {fm.diagonal_dominance:.4f}")
    """
    k_src = min(k, lb_source.n_eigenpairs)
    k_tgt = min(k, lb_target.n_eigenpairs)

    # Compute or truncate descriptor matrices
    if A_source is None:
        A_source = compute_descriptor_matrix(lb_source, descriptor_type, n_descriptors)
    if A_target is None:
        A_target = compute_descriptor_matrix(lb_target, descriptor_type, n_descriptors)

    A_s = A_source[:k_src]  # (k_src, D)
    A_t = A_target[:k_tgt]  # (k_tgt, D)

    # Eigenvalue diagonal matrices for commutativity term
    ev_src = lb_source.eigenvalues[:k_src]
    ev_tgt = lb_target.eigenvalues[:k_tgt]

    logger.info(
        "Computing functional map C (%d×%d) with %s descriptors...",
        k_tgt, k_src, descriptor_type,
    )

    # Solve: min_C ||C @ A_s - A_t||² + alpha_lap * ||Λ_tgt C - C Λ_src||²
    # This is a linear least-squares problem in the entries of C.
    # Vectorise: c = vec(C), build the normal equation (A^T A) c = A^T b.

    D = A_s.shape[1]

    # Descriptor term: ||C A_s - A_t||²_F
    # = sum_d ||C a_s^d - a_t^d||² where a^d are columns of A
    # Rearranging as linear system in vec(C):
    # Kronecker: (A_s^T ⊗ I_{k_tgt}) vec(C) = vec(A_t)
    # But it's more efficient to solve column-by-column or via normal equations.

    # Direct solution via normal equations:
    # C* = (A_t @ A_s^T + alpha_lap * Λ_tgt^{-1} * correction) @ (A_s @ A_s^T + ...)^{-1}
    # For simplicity and numerical stability, use per-column solve:

    # Combined: C (alpha_desc * A_s @ A_s^T + alpha_lap * Λ_commutativity) = alpha_desc * A_t @ A_s^T

    # Commutativity term: ||diag(ev_tgt) C - C diag(ev_src)||²_F
    # The (i,j) entry contributes (ev_tgt[i] - ev_src[j])² * C[i,j]²
    # This means the normal equations decouple by column of C if we use
    # the Tikhonov-style approach.

    # Build per-element weight for commutativity
    ev_diff_sq = (ev_tgt[:, None] - ev_src[None, :]) ** 2  # (k_tgt, k_src)

    # Descriptor gram matrix
    G = alpha_desc * (A_s @ A_s.T)  # (k_src, k_src)
    rhs = alpha_desc * (A_t @ A_s.T)  # (k_tgt, k_src)

    # Solve row by row of C (each row i of C is independent due to diagonal structure)
    # GPU backends: batch all k_tgt linear systems into a single batched solve
    if backend == "torch":
        C = _solve_functional_map_torch(G, rhs, ev_diff_sq, alpha_desc, alpha_lap, k_tgt, k_src)
    elif backend == "cupy":
        C = _solve_functional_map_cupy(G, rhs, ev_diff_sq, alpha_desc, alpha_lap, k_tgt, k_src)
    else:
        # NumPy fallback (row-by-row)
        C = np.zeros((k_tgt, k_src), dtype=np.float64)
        for i in range(k_tgt):
            lhs = G + alpha_lap * np.diag(ev_diff_sq[i])
            C[i] = np.linalg.solve(lhs, rhs[i])

    logger.info("  Initial C computed. Off-diagonal energy: %.4f",
                _off_diagonal_frobenius(C))

    return FunctionalMap(
        C=C,
        k_source=k_src,
        k_target=k_tgt,
        source_eigenvalues=ev_src,
        target_eigenvalues=ev_tgt,
        metadata={
            "method": "least_squares",
            "descriptor_type": descriptor_type,
            "n_descriptors": n_descriptors,
            "alpha_desc": alpha_desc,
            "alpha_lap": alpha_lap,
        },
    )


# ═══════════════════════════════════════════════════════════════════════════
# ZoomOut refinement
# ═══════════════════════════════════════════════════════════════════════════


def zoomout_refine(
    fm: FunctionalMap,
    lb_source: "LaplaceBeltrami",
    lb_target: "LaplaceBeltrami",
    k_final: int = 200,
    n_iterations: int = 10,
    step: Optional[int] = None,
    backend: str = "numpy",
) -> FunctionalMap:
    """
    Refine a functional map using the ZoomOut algorithm (Melzi et al., 2019).

    ZoomOut iteratively increases the spectral resolution of the map:
    at each step, it converts C to a pointwise map via nearest-neighbour
    in the eigenbasis, then re-estimates C at a higher resolution.

    Parameters
    ----------
    fm : FunctionalMap
        Initial functional map (e.g. from :func:`compute_functional_map`).
    lb_source : LaplaceBeltrami
        Source spectral decomposition (must have ≥ k_final eigenpairs).
    lb_target : LaplaceBeltrami
        Target spectral decomposition.
    k_final : int
        Target number of eigenfunctions for the refined map.
    n_iterations : int
        Number of refinement iterations.
    step : int or None
        Increment per iteration. If None, computed automatically.

    Returns
    -------
    FunctionalMap
        Refined functional map at resolution k_final.
    """
    k_init = fm.k_source
    k_max = min(k_final, lb_source.n_eigenpairs, lb_target.n_eigenpairs)

    if step is None:
        step = max(1, (k_max - k_init) // n_iterations)

    logger.info(
        "ZoomOut refinement: k=%d → %d (step=%d, %d iterations)",
        k_init, k_max, step, n_iterations,
    )

    C = fm.C.copy()
    phi_src = lb_source.eigenvectors
    phi_tgt = lb_target.eigenvectors
    M_tgt = lb_target.mass

    for it in range(n_iterations):
        k_current = min(k_init + (it + 1) * step, k_max)

        # 1. Convert C to pointwise map via nearest-neighbour
        #    Project target vertices into source basis: φ_tgt[:, :k] @ C → (N, k_src)
        k_old_tgt = C.shape[0]
        k_old_src = C.shape[1]
        emb_tgt = phi_tgt[:, :k_old_tgt] @ C  # (N_tgt, k_old_src)
        emb_src = phi_src[:, :k_old_src]       # (N_src, k_old_src)

        # Nearest neighbour: for each target vertex, find closest source vertex
        # Use chunked computation to avoid memory issues
        p2p = _nearest_neighbour_map(emb_tgt, emb_src, chunk_size=5000)

        # 2. Re-estimate C at higher resolution via Procrustes
        phi_src_k = phi_src[:, :k_current]  # (N_src, k_current)
        phi_tgt_k = phi_tgt[:, :k_current]  # (N_tgt, k_current)

        # Pull-back: select source basis rows according to p2p map
        phi_src_pulled = phi_src_k[p2p]  # (N_tgt, k_current)

        # Weighted least-squares: C_new = (φ_tgt^T M φ_src_pulled)
        if M_tgt is not None:
            M_phi_tgt = M_tgt @ phi_tgt_k  # (N_tgt, k_current)
        else:
            M_phi_tgt = phi_tgt_k
        C = M_phi_tgt.T @ phi_src_pulled  # (k_current, k_current)

        # Nearest orthogonal matrix (ICP-style)
        U, _, Vt = np.linalg.svd(C, full_matrices=False)
        C = U @ Vt

        logger.debug("  ZoomOut iter %d: k=%d, off-diag=%.4f",
                      it + 1, k_current, _off_diagonal_frobenius(C))

    return FunctionalMap(
        C=C,
        k_source=k_current,
        k_target=k_current,
        source_eigenvalues=lb_source.eigenvalues[:k_current],
        target_eigenvalues=lb_target.eigenvalues[:k_current],
        metadata={
            **fm.metadata,
            "refinement": "zoomout",
            "k_init": k_init,
            "k_final": k_current,
            "n_iterations": n_iterations,
        },
    )


# ═══════════════════════════════════════════════════════════════════════════
# Pointwise map extraction
# ═══════════════════════════════════════════════════════════════════════════


def functional_map_to_pointwise(
    fm: FunctionalMap,
    lb_source: "LaplaceBeltrami",
    lb_target: "LaplaceBeltrami",
) -> np.ndarray:
    """
    Convert a functional map to a pointwise vertex correspondence.

    For each target vertex, finds the source vertex whose spectral
    embedding (under the map C) is nearest.

    Parameters
    ----------
    fm : FunctionalMap
        The functional map C.
    lb_source : LaplaceBeltrami
        Source spectral decomposition.
    lb_target : LaplaceBeltrami
        Target spectral decomposition.

    Returns
    -------
    p2p : np.ndarray, shape (N_target,)
        Index of the corresponding source vertex for each target vertex.
    """
    k = min(fm.k_source, fm.k_target)
    emb_tgt = lb_target.eigenvectors[:, :k] @ fm.C[:k, :k]  # (N_tgt, k)
    emb_src = lb_source.eigenvectors[:, :k]                  # (N_src, k)
    return _nearest_neighbour_map(emb_tgt, emb_src)


def transfer_function(
    fm: FunctionalMap,
    lb_source: "LaplaceBeltrami",
    lb_target: "LaplaceBeltrami",
    f_source: np.ndarray,
) -> np.ndarray:
    """
    Transfer a scalar function from source to target surface via the
    functional map.

    Parameters
    ----------
    fm : FunctionalMap
        The functional map C.
    lb_source : LaplaceBeltrami
        Source spectral decomposition.
    lb_target : LaplaceBeltrami
        Target spectral decomposition.
    f_source : np.ndarray, shape (N_source,)
        Scalar function on the source surface (e.g. thickness).

    Returns
    -------
    f_target : np.ndarray, shape (N_target,)
        Transferred function on the target surface.
    """
    k = min(fm.k_source, fm.k_target)

    # Project f into source eigenbasis
    if lb_source.mass is not None:
        coeffs_src = lb_source.eigenvectors[:, :k].T @ (lb_source.mass @ f_source)
    else:
        coeffs_src = lb_source.eigenvectors[:, :k].T @ f_source

    # Map coefficients through C
    coeffs_tgt = fm.C[:k, :k] @ coeffs_src

    # Reconstruct in target basis
    f_target = lb_target.eigenvectors[:, :k] @ coeffs_tgt
    return f_target


# ═══════════════════════════════════════════════════════════════════════════
# Inter-hemispheric functional map (convenience)
# ═══════════════════════════════════════════════════════════════════════════


def compute_interhemispheric_map(
    lb_lh: "LaplaceBeltrami",
    lb_rh: "LaplaceBeltrami",
    k: int = 50,
    k_final: int = 200,
    descriptor_type: str = "hks",
    n_descriptors: int = 100,
    alpha_lap: float = 1e-3,
    zoomout: bool = True,
    n_zoomout_iters: int = 10,
    backend: str = "numpy",
) -> FunctionalMap:
    """
    Compute an inter-hemispheric functional map.

    Convenience function that computes the functional map from the left
    hemisphere to the right hemisphere, including optional ZoomOut
    refinement. The resulting C matrix encodes the correspondence
    between hemispheres in the spectral domain.

    Parameters
    ----------
    lb_lh : LaplaceBeltrami
        Left hemisphere spectral decomposition.
    lb_rh : LaplaceBeltrami
        Right hemisphere spectral decomposition.
    k : int
        Initial spectral resolution.
    k_final : int
        Final spectral resolution after ZoomOut.
    descriptor_type : str
        Descriptor type for initial correspondence.
    n_descriptors : int
        Number of descriptor scales.
    alpha_lap : float
        Laplacian commutativity weight.
    zoomout : bool
        Whether to apply ZoomOut refinement.
    n_zoomout_iters : int
        Number of ZoomOut iterations.

    Returns
    -------
    FunctionalMap
        Inter-hemispheric functional map (source=LH, target=RH).
    """
    fm = compute_functional_map(
        lb_source=lb_lh,
        lb_target=lb_rh,
        k=k,
        descriptor_type=descriptor_type,
        n_descriptors=n_descriptors,
        alpha_lap=alpha_lap,
        backend=backend,
    )

    if zoomout and k_final > k:
        fm = zoomout_refine(
            fm, lb_lh, lb_rh,
            k_final=k_final,
            n_iterations=n_zoomout_iters,
            backend=backend,
        )

    fm.metadata["type"] = "interhemispheric"
    logger.info(
        "Inter-hemispheric map: off-diag=%.4f, diag_dom=%.4f",
        fm.off_diagonal_energy, fm.diagonal_dominance,
    )
    return fm


# ═══════════════════════════════════════════════════════════════════════════
# Batch processing
# ═══════════════════════════════════════════════════════════════════════════


def compute_cohort_functional_maps(
    lb_pairs: List[Tuple["LaplaceBeltrami", "LaplaceBeltrami"]],
    subject_ids: Optional[List[str]] = None,
    k: int = 50,
    k_final: int = 200,
    descriptor_type: str = "hks",
    n_descriptors: int = 100,
    alpha_lap: float = 1e-3,
    zoomout: bool = True,
) -> List[FunctionalMap]:
    """
    Compute inter-hemispheric functional maps for a cohort.

    Parameters
    ----------
    lb_pairs : list of (LaplaceBeltrami, LaplaceBeltrami)
        Pairs of (LH, RH) spectral decompositions.
    subject_ids : list of str or None
        Subject identifiers for logging.
    k : int
        Initial spectral resolution.
    k_final : int
        Final resolution after ZoomOut.
    descriptor_type : str
        Descriptor type.
    n_descriptors : int
        Number of descriptor scales.
    alpha_lap : float
        Commutativity weight.
    zoomout : bool
        Whether to apply ZoomOut refinement.

    Returns
    -------
    list of FunctionalMap
        One functional map per subject.
    """
    n_subjects = len(lb_pairs)
    results = []

    for i, (lb_lh, lb_rh) in enumerate(lb_pairs):
        sid = subject_ids[i] if subject_ids else f"sub-{i:03d}"
        logger.info("Processing %s (%d/%d)...", sid, i + 1, n_subjects)

        fm = compute_interhemispheric_map(
            lb_lh, lb_rh,
            k=k,
            k_final=k_final,
            descriptor_type=descriptor_type,
            n_descriptors=n_descriptors,
            alpha_lap=alpha_lap,
            zoomout=zoomout,
        )
        fm.metadata["subject_id"] = sid
        results.append(fm)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════


def _nearest_neighbour_map(
    emb_target: np.ndarray,
    emb_source: np.ndarray,
    chunk_size: int = 5000,
) -> np.ndarray:
    """
    Compute nearest-neighbour map from target to source embeddings.

    Uses chunked computation to handle large point sets.

    Parameters
    ----------
    emb_target : np.ndarray, shape (N_tgt, D)
    emb_source : np.ndarray, shape (N_src, D)
    chunk_size : int
        Process this many target points at once.

    Returns
    -------
    p2p : np.ndarray, shape (N_tgt,)
        Source index for each target point.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(emb_source)
    N = emb_target.shape[0]
    p2p = np.zeros(N, dtype=np.int64)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        _, idx = tree.query(emb_target[start:end])
        p2p[start:end] = idx

    return p2p


def _off_diagonal_frobenius(C: np.ndarray) -> float:
    """Frobenius norm of off-diagonal elements of a square matrix."""
    k = min(C.shape)
    C_sq = C[:k, :k]
    diag_e = np.sum(np.diag(C_sq) ** 2)
    total_e = np.sum(C_sq ** 2)
    return float(np.sqrt(max(total_e - diag_e, 0.0)))


def functional_map_distance(
    fm1: FunctionalMap,
    fm2: FunctionalMap,
    metric: str = "frobenius",
) -> float:
    """
    Compute distance between two functional maps.

    Useful for building subject-level distance matrices from C matrices.

    Parameters
    ----------
    fm1 : FunctionalMap
    fm2 : FunctionalMap
    metric : ``'frobenius'`` or ``'geodesic'``
        Distance metric. ``'frobenius'`` uses ||C1 - C2||_F.
        ``'geodesic'`` uses the geodesic distance on SO(k) if both
        C matrices are orthogonal.

    Returns
    -------
    float
        Distance between the two maps.
    """
    k = min(fm1.C.shape[0], fm1.C.shape[1], fm2.C.shape[0], fm2.C.shape[1])
    C1 = fm1.C[:k, :k]
    C2 = fm2.C[:k, :k]

    if metric == "frobenius":
        return float(np.linalg.norm(C1 - C2, "fro"))
    elif metric == "geodesic":
        # Geodesic distance on SO(k): ||log(C1^T C2)||_F
        R = C1.T @ C2
        # Clip eigenvalues for numerical stability
        U, S, Vt = np.linalg.svd(R)
        R_proj = U @ Vt  # Project to SO(k)
        # log of rotation matrix
        cos_theta = np.clip((np.trace(R_proj) - 1) / 2, -1, 1)
        theta = np.arccos(cos_theta)
        return float(theta)
    else:
        raise ValueError(f"Unknown metric: {metric!r}")


# ═══════════════════════════════════════════════════════════════════════════
# GPU-accelerated solvers for functional map computation
# ═══════════════════════════════════════════════════════════════════════════


def _solve_functional_map_torch(G, rhs, ev_diff_sq, alpha_desc, alpha_lap, k_tgt, k_src):
    """Batched linear solve on GPU via PyTorch (torch.linalg.solve)."""
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("  Functional map solve on %s via torch.linalg.solve", device)

    G_t = torch.tensor(G, dtype=torch.float64, device=device)
    rhs_t = torch.tensor(rhs, dtype=torch.float64, device=device)
    ev_t = torch.tensor(ev_diff_sq, dtype=torch.float64, device=device)

    # Build batch of (k_tgt, k_src, k_src) LHS matrices
    # lhs[i] = G + alpha_lap * diag(ev_diff_sq[i])
    diag_batch = torch.zeros(k_tgt, k_src, k_src, dtype=torch.float64, device=device)
    idx = torch.arange(k_src, device=device)
    diag_batch[:, idx, idx] = alpha_lap * ev_t
    lhs_batch = G_t.unsqueeze(0).expand(k_tgt, -1, -1) + diag_batch

    # Batched solve: lhs_batch @ C_rows = rhs
    C_t = torch.linalg.solve(lhs_batch, rhs_t.unsqueeze(-1)).squeeze(-1)
    return C_t.cpu().numpy()


def _solve_functional_map_cupy(G, rhs, ev_diff_sq, alpha_desc, alpha_lap, k_tgt, k_src):
    """Batched linear solve on GPU via CuPy."""
    import cupy as cp
    logger.info("  Functional map solve via CuPy batched solve")

    G_c = cp.asarray(G)
    rhs_c = cp.asarray(rhs)
    ev_c = cp.asarray(ev_diff_sq)

    C = cp.zeros((k_tgt, k_src), dtype=cp.float64)
    for i in range(k_tgt):
        lhs = G_c + alpha_lap * cp.diag(ev_c[i])
        C[i] = cp.linalg.solve(lhs, rhs_c[i])

    return cp.asnumpy(C)

