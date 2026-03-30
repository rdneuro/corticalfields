"""
Utility functions for CorticalFields.

Provides helpers for parcellation loading, timing, GPU detection,
and data validation.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Timing
# ═══════════════════════════════════════════════════════════════════════════


@contextmanager
def timer(label: str = ""):
    """Context manager that logs elapsed time."""
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    if elapsed < 60:
        logger.info("[%s] %.2f seconds", label, elapsed)
    else:
        logger.info("[%s] %.1f minutes", label, elapsed / 60)


# ═══════════════════════════════════════════════════════════════════════════
# GPU detection
# ═══════════════════════════════════════════════════════════════════════════


def get_device(prefer_cuda: bool = True) -> "torch.device":
    """
    Return the best available PyTorch device.

    Parameters
    ----------
    prefer_cuda : bool
        If True and CUDA is available, return a CUDA device.

    Returns
    -------
    torch.device
    """
    import torch

    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info("Using CUDA device: %s (%.1f GB)", name, mem)
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device.")

    return device


# ═══════════════════════════════════════════════════════════════════════════
# Parcellation helpers
# ═══════════════════════════════════════════════════════════════════════════


def load_fsaverage_parcellation(
    atlas: str = "schaefer_200",
    hemi: str = "lh",
    resolution: str = "fsaverage",
) -> Tuple[np.ndarray, List[str]]:
    """
    Load a standard cortical parcellation on fsaverage.

    Requires ``neuromaps`` or ``nibabel`` with FreeSurfer data.

    Parameters
    ----------
    atlas : str
        Atlas name. Supported: ``'schaefer_200'``, ``'schaefer_400'``,
        ``'desikan'``, ``'destrieux'``, ``'yeo_7'``, ``'yeo_17'``.
    hemi : ``'lh'`` or ``'rh'``
    resolution : str
        Surface template resolution.

    Returns
    -------
    labels : np.ndarray, shape (N,)
        Integer label per vertex.
    names : list[str]
        Region/network names.
    """
    try:
        import nibabel as nib
        from nibabel.freesurfer import read_annot
    except ImportError:
        raise ImportError("nibabel is required for parcellation loading.")

    # Try neuromaps first for Schaefer atlases
    if "schaefer" in atlas.lower():
        try:
            from neuromaps.datasets import fetch_schaefer2018
            n_parcels = int(atlas.split("_")[-1])
            # neuromaps returns GIfTI label files
            parc = fetch_schaefer2018(n_parcels=n_parcels)
            side = "L" if hemi == "lh" else "R"
            gii = nib.load(parc[side])
            labels = gii.darrays[0].data.astype(np.int64)
            # Generate names
            names = [f"Region_{i}" for i in range(n_parcels // 2 + 1)]
            return labels, names
        except Exception as e:
            logger.debug("neuromaps Schaefer load failed: %s", e)

    # Fallback: FreeSurfer annot files
    fs_home = Path("/usr/local/freesurfer") / "subjects" / resolution
    if not fs_home.exists():
        # Try environment variable
        import os
        fsh = os.environ.get("FREESURFER_HOME", "")
        if fsh:
            fs_home = Path(fsh) / "subjects" / resolution

    annot_map = {
        "desikan": "aparc",
        "destrieux": "aparc.a2009s",
        "yeo_7": "Yeo2011_7Networks_N1000",
        "yeo_17": "Yeo2011_17Networks_N1000",
    }

    annot_name = annot_map.get(atlas)
    if annot_name is None:
        raise ValueError(
            f"Unknown atlas: {atlas}. Supported: {list(annot_map.keys())}"
        )

    annot_path = fs_home / "label" / f"{hemi}.{annot_name}.annot"
    if not annot_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {annot_path}")

    labels, ctab, names = read_annot(str(annot_path))
    names = [n.decode() if isinstance(n, bytes) else n for n in names]
    return labels.astype(np.int64), names


# ═══════════════════════════════════════════════════════════════════════════
# Data validation
# ═══════════════════════════════════════════════════════════════════════════


def validate_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    strict: bool = False,
) -> Dict[str, object]:
    """
    Validate a triangle mesh for common issues.

    Returns
    -------
    report : dict
        Keys: ``n_vertices``, ``n_faces``, ``n_edges``,
        ``has_degenerate_faces``, ``euler_characteristic``,
        ``is_closed``, ``bbox``.
    """
    report = {
        "n_vertices": vertices.shape[0],
        "n_faces": faces.shape[0],
    }

    # Check for degenerate faces (area ≈ 0)
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    report["has_degenerate_faces"] = bool(np.any(areas < 1e-10))
    report["min_face_area"] = float(areas.min())
    report["mean_face_area"] = float(areas.mean())
    report["total_area"] = float(areas.sum())

    # Euler characteristic: V - E + F
    edges = set()
    for f in faces:
        for i in range(3):
            a, b = int(f[i]), int(f[(i + 1) % 3])
            edges.add((min(a, b), max(a, b)))
    n_edges = len(edges)
    report["n_edges"] = n_edges
    euler = vertices.shape[0] - n_edges + faces.shape[0]
    report["euler_characteristic"] = euler
    report["is_closed"] = (euler == 2)  # genus-0 closed surface

    # Bounding box
    report["bbox"] = {
        "min": vertices.min(axis=0).tolist(),
        "max": vertices.max(axis=0).tolist(),
        "extent": (vertices.max(axis=0) - vertices.min(axis=0)).tolist(),
    }

    if strict and report["has_degenerate_faces"]:
        raise ValueError(
            f"Mesh has degenerate faces (min area: {report['min_face_area']:.2e})"
        )

    return report


# ═══════════════════════════════════════════════════════════════════════════
# Logging setup
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# Eigenpair estimation
# ═══════════════════════════════════════════════════════════════════════════


def estimate_n_eigenpairs(
    n_vertices: int,
    *,
    surface_area_mm2: float | None = None,
    mode: str = "auto",
) -> int:
    """
    Estimate the optimal number of LBO eigenpairs for a given mesh.

    The useful spectral bandwidth of a triangle mesh is fundamentally
    limited by vertex density.  For N vertices, at most ~N/5 to N/10
    eigenpairs carry geometric (rather than aliasing) information.
    We use a piecewise heuristic calibrated to neuroimaging meshes:

        cortical  (~150 k vertices) →  200–300 eigenpairs
        subcortical (~2–5 k vertices) →  50–150 eigenpairs
        tiny       (<1 k vertices)   →  20–50  eigenpairs

    An optional Weyl-law refinement is available when surface area
    is known: λ_k ~ 4πk / A, so k_max ~ A·λ_max / (4π), where
    λ_max is taken to be the Nyquist-like limit set by mean edge
    length ℓ:  λ_Nyquist ~ (2π/ℓ)².

    Parameters
    ----------
    n_vertices : int
        Number of mesh vertices.
    surface_area_mm2 : float or None
        Total surface area in mm².  If provided *and* ``mode='weyl'``,
        uses the Weyl-law estimate.  Otherwise ignored.
    mode : ``'auto'``, ``'conservative'``, ``'aggressive'``, ``'weyl'``
        ``'auto'``         – piecewise heuristic (recommended default).
        ``'conservative'`` – ~N/10, safe for noisy marching-cubes meshes.
        ``'aggressive'``   – ~N/5, for high-quality smoothed meshes.
        ``'weyl'``         – Weyl-law estimate (needs ``surface_area_mm2``).

    Returns
    -------
    k : int
        Recommended number of eigenpairs (including λ_0 = 0).

    Examples
    --------
    >>> estimate_n_eigenpairs(163842)           # cortical mesh
    300
    >>> estimate_n_eigenpairs(3000)             # hippocampus from aseg
    100
    >>> estimate_n_eigenpairs(800, mode='conservative')
    50
    """
    # Hard floor / ceiling
    K_MIN, K_MAX = 10, 500

    if mode == "weyl" and surface_area_mm2 is not None:
        # Mean vertex spacing  ℓ ≈ √(2A / (√3 · N))  for equilateral tris
        mean_edge = np.sqrt(2.0 * surface_area_mm2 / (np.sqrt(3) * n_vertices))
        # Nyquist wavenumber → eigenvalue
        lam_nyquist = (2.0 * np.pi / mean_edge) ** 2
        # Weyl:  k ≈ A · λ / (4π)
        k_weyl = int(surface_area_mm2 * lam_nyquist / (4.0 * np.pi))
        # Apply a safety factor of 0.5 (we don't want aliased modes)
        k = max(K_MIN, min(int(k_weyl * 0.5), K_MAX))
        logger.debug(
            "Weyl estimate: ℓ=%.2f mm, λ_Nyq=%.1f, k_raw=%d → k=%d",
            mean_edge, lam_nyquist, k_weyl, k,
        )
        return k

    if mode == "conservative":
        ratio = 10
    elif mode == "aggressive":
        ratio = 5
    else:
        # 'auto' — piecewise heuristic
        ratio = 7  # middle ground

    k_raw = n_vertices / ratio

    # Piecewise clamping with smooth transitions
    if n_vertices >= 100_000:
        # Cortical: cap at 300 (beyond this the spectral gap is tiny)
        k = min(int(k_raw), 300)
    elif n_vertices >= 5_000:
        # Large subcortical (thalamus, caudate) or decimated cortex
        k = min(int(k_raw), 200)
    elif n_vertices >= 1_000:
        # Typical subcortical from marching cubes
        k = min(int(k_raw), 150)
    else:
        # Very small mesh (<1 k verts)
        k = min(int(k_raw), 80)

    k = max(K_MIN, min(k, K_MAX))

    logger.debug(
        "estimate_n_eigenpairs: N=%d, mode=%s → k=%d (ratio=1:%d)",
        n_vertices, mode, k, ratio,
    )
    return k


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for CorticalFields."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="[%(asctime)s] %(name)s — %(levelname)s — %(message)s",
        datefmt="%H:%M:%S",
    )
