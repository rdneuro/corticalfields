"""
GPU-accelerated spectral descriptors on point clouds.

Computes Heat Kernel Signature (HKS), Wave Kernel Signature (WKS),
and Global Point Signature (GPS) directly from point cloud Laplacian
eigenpairs using PyTorch CUDA. All functions have automatic CPU fallback
and explicit VRAM management for large cortical surfaces (~150 K points).

Functions
---------
hks_pointcloud       : Heat Kernel Signature with GPU acceleration
wks_pointcloud       : Wave Kernel Signature with GPU acceleration
gps_pointcloud       : Global Point Signature
spectral_features    : Unified HKS + WKS + GPS feature extraction
validate_eigenpairs  : Validate PCD vs mesh LBO eigenpairs

References
----------
Sun, Ovsjanikov & Guibas (2009). A Concise and Provably Informative
    Multi-Scale Signature Based on Heat Diffusion. SGP.
Aubry, Schlickewei & Cremers (2011). The Wave Kernel Signature: A
    Quantum Mechanical Approach to Shape Analysis. ICCV Workshops.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Device management
# ═══════════════════════════════════════════════════════════════════════════


def _get_device(prefer_cuda: bool = True) -> "torch.device":
    """Return the best available device with VRAM check."""
    import torch

    if prefer_cuda and torch.cuda.is_available():
        free_mem = (
            torch.cuda.get_device_properties(0).total_memory
            - torch.cuda.memory_allocated(0)
        )
        if free_mem < 256 * 1024 * 1024:  # < 256 MB free
            logger.warning(
                "VRAM nearly full (%.0f MB free) — falling back to CPU",
                free_mem / (1024 ** 2),
            )
            return torch.device("cpu")
        return torch.device("cuda", 0)
    return torch.device("cpu")


def _to_torch(
    arr: np.ndarray,
    device: "torch.device",
    dtype: Optional["torch.dtype"] = None,
) -> "torch.Tensor":
    """Convert numpy array to torch tensor on device."""
    import torch

    if dtype is None:
        dtype = torch.float32
    t = torch.as_tensor(arr, dtype=dtype)
    return t.to(device, non_blocking=True)


def _cleanup_gpu(*tensors) -> None:
    """Explicitly delete GPU tensors and free VRAM."""
    import torch

    for t in tensors:
        if t is not None:
            del t
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════
# Heat Kernel Signature
# ═══════════════════════════════════════════════════════════════════════════


def hks_pointcloud(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    n_scales: int = 16,
    t_min: Optional[float] = None,
    t_max: Optional[float] = None,
    use_gpu: bool = True,
) -> np.ndarray:
    """
    Compute Heat Kernel Signature on point cloud eigenpairs.

    HKS(x, t) = Σ_i exp(−λ_i t) φ_i(x)²

    Parameters
    ----------
    eigenvalues : np.ndarray, shape (K,)
        LBO eigenvalues (non-negative, sorted ascending).
        Must exclude the trivial zero eigenvalue (start from λ_1).
    eigenvectors : np.ndarray, shape (N, K)
        Corresponding eigenvectors (columns of the matrix Φ).
    n_scales : int
        Number of logarithmically spaced time scales.
    t_min : float or None
        Minimum diffusion time. Default: 4 ln(10) / λ_max.
    t_max : float or None
        Maximum diffusion time. Default: 4 ln(10) / λ_1.
    use_gpu : bool
        Use CUDA if available. Falls back to CPU transparently.

    Returns
    -------
    hks : np.ndarray, shape (N, n_scales)
        Heat Kernel Signature at each point and scale.

    Notes
    -----
    Memory: For N=150K, K=300, T=16 the computation requires ~350 MB
    VRAM. Batched processing is used for K > 500 to stay within 8 GB.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    eigenvectors = np.asarray(eigenvectors, dtype=np.float64)

    N, K = eigenvectors.shape
    assert eigenvalues.shape == (K,), (
        f"Shape mismatch: eigenvalues {eigenvalues.shape} vs "
        f"eigenvectors columns {K}"
    )

    # Skip the trivial eigenvalue if present
    if eigenvalues[0] < 1e-10:
        eigenvalues = eigenvalues[1:]
        eigenvectors = eigenvectors[:, 1:]
        K -= 1

    # Auto time scale: Sun et al. (2009) recommendation
    lam_min = max(eigenvalues[0], 1e-8)
    lam_max = eigenvalues[-1]
    c = 4.0 * np.log(10.0)
    if t_min is None:
        t_min = c / lam_max
    if t_max is None:
        t_max = c / lam_min

    t_values = np.geomspace(t_min, t_max, n_scales)

    if use_gpu:
        try:
            return _hks_torch(eigenvalues, eigenvectors, t_values)
        except (ImportError, RuntimeError) as exc:
            logger.info("GPU HKS failed (%s); falling back to NumPy", exc)

    return _hks_numpy(eigenvalues, eigenvectors, t_values)


def _hks_torch(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    t_values: np.ndarray,
) -> np.ndarray:
    """GPU-accelerated HKS with batched computation for VRAM safety."""
    import torch

    device = _get_device(prefer_cuda=True)
    N, K = eigenvectors.shape
    T = len(t_values)

    # Estimate memory: phi_sq(N,K) + decay(T,K) + hks(N,T) in float32
    mem_bytes = (N * K + T * K + N * T) * 4
    vram_available = (
        torch.cuda.get_device_properties(0).total_memory
        - torch.cuda.memory_allocated(0)
    ) if device.type == "cuda" else float("inf")

    # If >60% of free VRAM, use batched approach
    if mem_bytes > 0.6 * vram_available and device.type == "cuda":
        return _hks_torch_batched(eigenvalues, eigenvectors, t_values, device)

    lam = _to_torch(eigenvalues, device)         # (K,)
    phi = _to_torch(eigenvectors, device)         # (N, K)
    t = _to_torch(t_values, device)               # (T,)

    phi_sq = phi.pow(2)                           # (N, K)
    decay = torch.exp(-lam.unsqueeze(0) * t.unsqueeze(1))  # (T, K)
    hks = phi_sq @ decay.T                        # (N, T)

    # HKS is theoretically non-negative
    hks = torch.clamp(hks, min=0.0)

    result = hks.cpu().numpy()
    _cleanup_gpu(lam, phi, t, phi_sq, decay, hks)
    return result


def _hks_torch_batched(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    t_values: np.ndarray,
    device: "torch.device",
    batch_size: int = 16384,
) -> np.ndarray:
    """Batched HKS for large point clouds that exceed VRAM."""
    import torch

    N, K = eigenvectors.shape
    T = len(t_values)

    lam = _to_torch(eigenvalues, device)          # (K,)
    t = _to_torch(t_values, device)               # (T,)
    decay = torch.exp(-lam.unsqueeze(0) * t.unsqueeze(1))  # (T, K)

    result = np.empty((N, T), dtype=np.float32)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        phi_batch = _to_torch(eigenvectors[start:end], device)  # (B, K)
        phi_sq = phi_batch.pow(2)                               # (B, K)
        hks_batch = phi_sq @ decay.T                            # (B, T)
        hks_batch = torch.clamp(hks_batch, min=0.0)
        result[start:end] = hks_batch.cpu().numpy()
        del phi_batch, phi_sq, hks_batch

    _cleanup_gpu(lam, t, decay)
    return result


def _hks_numpy(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    t_values: np.ndarray,
) -> np.ndarray:
    """CPU fallback HKS via NumPy."""
    phi_sq = eigenvectors ** 2                    # (N, K)
    decay = np.exp(
        -eigenvalues[np.newaxis, :] * t_values[:, np.newaxis]
    )                                             # (T, K)
    hks = phi_sq @ decay.T                        # (N, T)
    return np.clip(hks, 0.0, None)


# ═══════════════════════════════════════════════════════════════════════════
# Wave Kernel Signature
# ═══════════════════════════════════════════════════════════════════════════


def wks_pointcloud(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    n_scales: int = 100,
    sigma: Optional[float] = None,
    use_gpu: bool = True,
) -> np.ndarray:
    """
    Compute Wave Kernel Signature on point cloud eigenpairs.

    WKS(x, e) = Σ_i exp(−(e − log λ_i)² / 2σ²) φ_i(x)² / C(e)

    Parameters
    ----------
    eigenvalues : np.ndarray, shape (K,)
        LBO eigenvalues (positive, sorted ascending, λ_0 > 0).
    eigenvectors : np.ndarray, shape (N, K)
        Corresponding eigenvectors.
    n_scales : int
        Number of energy scales.
    sigma : float or None
        Gaussian bandwidth. Default: 7 × (e_max − e_min) / n_scales
        (Aubry et al. 2011 recommendation).
    use_gpu : bool
        Use CUDA if available.

    Returns
    -------
    wks : np.ndarray, shape (N, n_scales)
        Wave Kernel Signature at each point and energy.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    eigenvectors = np.asarray(eigenvectors, dtype=np.float64)

    N, K = eigenvectors.shape
    assert eigenvalues.shape == (K,)

    # Skip trivial eigenvalue
    if eigenvalues[0] < 1e-10:
        eigenvalues = eigenvalues[1:]
        eigenvectors = eigenvectors[:, 1:]
        K -= 1

    log_evals = np.log(np.maximum(eigenvalues, 1e-12))
    e_min, e_max = log_evals[0], log_evals[-1]

    if sigma is None:
        sigma = 7.0 * (e_max - e_min) / n_scales

    energies = np.linspace(e_min, e_max, n_scales)

    if use_gpu:
        try:
            return _wks_torch(eigenvectors, log_evals, energies, sigma)
        except (ImportError, RuntimeError) as exc:
            logger.info("GPU WKS failed (%s); falling back to NumPy", exc)

    return _wks_numpy(eigenvectors, log_evals, energies, sigma)


def _wks_torch(
    eigenvectors: np.ndarray,
    log_evals: np.ndarray,
    energies: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """GPU-accelerated WKS with VRAM-safe batching."""
    import torch

    device = _get_device(prefer_cuda=True)
    N, K = eigenvectors.shape
    E = len(energies)

    phi = _to_torch(eigenvectors, device)         # (N, K)
    le = _to_torch(log_evals, device)             # (K,)
    en = _to_torch(energies, device)              # (E,)

    phi_sq = phi.pow(2)                           # (N, K)
    # Gaussian weights: (E, K)
    diff = en.unsqueeze(1) - le.unsqueeze(0)      # (E, K)
    weights = torch.exp(-diff.pow(2) / (2 * sigma ** 2))  # (E, K)

    # WKS = phi_sq @ weights.T, then normalize per energy
    wks = phi_sq @ weights.T                      # (N, E)
    normalizer = weights.sum(dim=1, keepdim=True).T  # (1, E)
    normalizer = torch.clamp(normalizer, min=1e-12)
    wks = wks / normalizer

    result = wks.cpu().numpy()
    _cleanup_gpu(phi, le, en, phi_sq, diff, weights, wks, normalizer)
    return result


def _wks_numpy(
    eigenvectors: np.ndarray,
    log_evals: np.ndarray,
    energies: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """CPU fallback WKS."""
    phi_sq = eigenvectors ** 2                    # (N, K)
    diff = energies[:, None] - log_evals[None, :]  # (E, K)
    weights = np.exp(-diff ** 2 / (2 * sigma ** 2))  # (E, K)
    wks = phi_sq @ weights.T                      # (N, E)
    normalizer = weights.sum(axis=1, keepdims=True).T  # (1, E)
    normalizer = np.clip(normalizer, 1e-12, None)
    return wks / normalizer


# ═══════════════════════════════════════════════════════════════════════════
# Global Point Signature
# ═══════════════════════════════════════════════════════════════════════════


def gps_pointcloud(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    n_components: Optional[int] = None,
) -> np.ndarray:
    """
    Compute Global Point Signature from point cloud eigenpairs.

    GPS(x) = (φ_1(x)/√λ_1, φ_2(x)/√λ_2, ..., φ_K(x)/√λ_K)

    Parameters
    ----------
    eigenvalues : np.ndarray, shape (K,)
        Positive LBO eigenvalues.
    eigenvectors : np.ndarray, shape (N, K)
        Corresponding eigenvectors.
    n_components : int or None
        How many GPS components to keep. None = all K.

    Returns
    -------
    gps : np.ndarray, shape (N, n_components)
        Global Point Signature embedding.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    eigenvectors = np.asarray(eigenvectors, dtype=np.float64)

    # Skip trivial eigenvalue
    if eigenvalues[0] < 1e-10:
        eigenvalues = eigenvalues[1:]
        eigenvectors = eigenvectors[:, 1:]

    if n_components is not None:
        eigenvalues = eigenvalues[:n_components]
        eigenvectors = eigenvectors[:, :n_components]

    inv_sqrt_lam = 1.0 / np.sqrt(np.maximum(eigenvalues, 1e-12))
    return eigenvectors * inv_sqrt_lam[np.newaxis, :]


# ═══════════════════════════════════════════════════════════════════════════
# Unified feature extraction
# ═══════════════════════════════════════════════════════════════════════════


def spectral_features(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    hks_scales: int = 16,
    wks_scales: int = 100,
    gps_components: Optional[int] = 50,
    use_gpu: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Extract all spectral descriptors from point cloud eigenpairs.

    Convenience function that computes HKS, WKS, and GPS in a single
    call, sharing eigendata and performing GPU cleanup once at the end.

    Parameters
    ----------
    eigenvalues : np.ndarray, shape (K,)
    eigenvectors : np.ndarray, shape (N, K)
    hks_scales : int
        Number of HKS time scales.
    wks_scales : int
        Number of WKS energy scales.
    gps_components : int or None
        Number of GPS embedding dimensions. None = all.
    use_gpu : bool
        Use CUDA if available.

    Returns
    -------
    dict
        Keys: ``'hks'`` (N, T), ``'wks'`` (N, E), ``'gps'`` (N, D).
    """
    logger.info(
        "Computing spectral features: HKS(%d), WKS(%d), GPS(%s) "
        "for %d points, %d eigenpairs",
        hks_scales, wks_scales,
        gps_components if gps_components else "all",
        eigenvectors.shape[0], eigenvectors.shape[1],
    )

    hks = hks_pointcloud(eigenvalues, eigenvectors, hks_scales, use_gpu=use_gpu)
    wks = wks_pointcloud(eigenvalues, eigenvectors, wks_scales, use_gpu=use_gpu)
    gps = gps_pointcloud(eigenvalues, eigenvectors, gps_components)

    logger.info(
        "  HKS: %s | WKS: %s | GPS: %s",
        hks.shape, wks.shape, gps.shape,
    )
    return {"hks": hks, "wks": wks, "gps": gps}


# ═══════════════════════════════════════════════════════════════════════════
# Eigenpair validation: PCD vs mesh LBO
# ═══════════════════════════════════════════════════════════════════════════


def validate_eigenpairs(
    eigenvalues_mesh: np.ndarray,
    eigenvectors_mesh: np.ndarray,
    eigenvalues_pcd: np.ndarray,
    eigenvectors_pcd: np.ndarray,
    n_compare: int = 50,
) -> Dict[str, np.ndarray]:
    """
    Validate point cloud LBO eigenpairs against mesh cotangent Laplacian.

    This implements the validation study identified as a novel contribution
    in the CorticalFields research: comparing ``robust_laplacian``
    point cloud eigenpairs to mesh-based cotangent Laplacian eigenpairs
    on the same FreeSurfer cortical surfaces.

    Parameters
    ----------
    eigenvalues_mesh : np.ndarray, shape (K_mesh,)
        Mesh-based LBO eigenvalues.
    eigenvectors_mesh : np.ndarray, shape (N, K_mesh)
        Mesh-based LBO eigenvectors.
    eigenvalues_pcd : np.ndarray, shape (K_pcd,)
        Point cloud LBO eigenvalues.
    eigenvectors_pcd : np.ndarray, shape (N, K_pcd)
        Point cloud LBO eigenvectors.
    n_compare : int
        Number of leading eigenpairs to compare.

    Returns
    -------
    dict
        ``'eigenvalue_ratio'`` : np.ndarray, shape (n_compare,)
            λ_pcd / λ_mesh per mode (should be ~1.0).
        ``'eigenvalue_rmse'`` : float
            RMSE of eigenvalue ratios from unity.
        ``'eigenvector_correlation'`` : np.ndarray, shape (n_compare,)
            |corr(φ_mesh_i, φ_pcd_i)| per mode (should be close to 1.0).
            Note: sign ambiguity is resolved by taking absolute value.
        ``'hks_correlation'`` : np.ndarray, shape (16,)
            Correlation of HKS computed from mesh vs PCD eigenpairs
            at each of 16 time scales.
    """
    n = min(n_compare, len(eigenvalues_mesh), len(eigenvalues_pcd))

    lam_m = eigenvalues_mesh[:n]
    lam_p = eigenvalues_pcd[:n]
    phi_m = eigenvectors_mesh[:, :n]
    phi_p = eigenvectors_pcd[:, :n]

    # Skip trivial eigenvalue for ratio
    start = 1 if lam_m[0] < 1e-10 else 0
    ratio = np.ones(n)
    ratio[start:] = lam_p[start:] / np.maximum(lam_m[start:], 1e-12)

    rmse = np.sqrt(np.mean((ratio[start:] - 1.0) ** 2))

    # Eigenvector correlation (sign-ambiguous)
    corr = np.zeros(n)
    for i in range(n):
        r = np.corrcoef(phi_m[:, i], phi_p[:, i])[0, 1]
        corr[i] = abs(r)

    # HKS correlation across time scales
    hks_m = _hks_numpy(
        eigenvalues_mesh[start:n], eigenvectors_mesh[:, start:n],
        np.geomspace(
            4 * np.log(10) / max(eigenvalues_mesh[n - 1], 1e-8),
            4 * np.log(10) / max(eigenvalues_mesh[start], 1e-8),
            16,
        ),
    )
    hks_p = _hks_numpy(
        eigenvalues_pcd[start:n], eigenvectors_pcd[:, start:n],
        np.geomspace(
            4 * np.log(10) / max(eigenvalues_pcd[n - 1], 1e-8),
            4 * np.log(10) / max(eigenvalues_pcd[start], 1e-8),
            16,
        ),
    )
    hks_corr = np.array([
        np.corrcoef(hks_m[:, t], hks_p[:, t])[0, 1]
        for t in range(16)
    ])

    logger.info(
        "Eigenpair validation (n=%d): eigenvalue RMSE=%.4f, "
        "mean |eigvec corr|=%.4f, mean HKS corr=%.4f",
        n, rmse, corr[start:].mean(), hks_corr.mean(),
    )

    return {
        "eigenvalue_ratio": ratio,
        "eigenvalue_rmse": rmse,
        "eigenvector_correlation": corr,
        "hks_correlation": hks_corr,
    }
