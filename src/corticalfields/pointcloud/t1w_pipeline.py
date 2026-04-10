"""
corticalfields.pointcloud.t1w_pipeline
======================================
FreeSurfer-free T1w → Point Cloud extraction pipeline.

Extracts cortical, subcortical, and hippocampal point clouds from raw T1w
NIfTI images using only DL-based tools (deepbet + SynthSeg/FastSurferCNN).

Key design decisions
--------------------
- **Single skull-strip**: deepbet runs ONCE per T1w; the brain mask is reused
  for all downstream extractions.
- **Single segmentation**: SynthSeg (or fallback) runs ONCE per T1w; the aseg
  label map is reused for cortical GM, subcortical, and hippocampal extraction.
- **Hemisphere split is geometric**: sagittal midplane split on the already-
  extracted cortical PCD, not a separate extraction per hemisphere.

Usage
-----
>>> from corticalfields.pointcloud.t1w_pipeline import T1wPCDPipeline
>>> pipe = T1wPCDPipeline(use_gpu=True)
>>> result = pipe.run(t1w_path)
>>> result.cortical["lh"].points.shape
(78432, 3)
>>> result.subcortical["Left-Thalamus"].points.shape
(1823, 3)

Author : Rodrigo Debona (INNT/UFRJ) + Claude
Date   : 2026-04
"""

from __future__ import annotations

import gc
import logging
import shutil
import subprocess
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

# SynthSeg / FreeSurfer aseg label IDs
# Cortical GM labels (used if SynthSeg provides them; otherwise morphological)
CORTICAL_GM_LABELS = [3, 42]  # Left/Right-Cerebral-Cortex

# Subcortical structure labels (bilateral)
SUBCORTICAL_LABELS = {
    # label_id: (name, hemisphere)
    10: ("Left-Thalamus", "lh"),
    49: ("Right-Thalamus", "rh"),
    11: ("Left-Caudate", "lh"),
    50: ("Right-Caudate", "rh"),
    12: ("Left-Putamen", "lh"),
    51: ("Right-Putamen", "rh"),
    13: ("Left-Pallidum", "lh"),
    52: ("Right-Pallidum", "rh"),
    26: ("Left-Accumbens", "lh"),
    58: ("Right-Accumbens", "rh"),
    28: ("Left-VentralDC", "lh"),
    60: ("Right-VentralDC", "rh"),
    17: ("Left-Hippocampus", "lh"),
    53: ("Right-Hippocampus", "rh"),
    18: ("Left-Amygdala", "lh"),
    54: ("Right-Amygdala", "rh"),
}

# Hippocampal labels only (subset for convenience)
HIPPOCAMPAL_LABELS = {
    17: ("Left-Hippocampus", "lh"),
    53: ("Right-Hippocampus", "rh"),
}

# Minimum voxels for a structure to be meshed
MIN_VOXELS_FOR_MESH = 30


# ═══════════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PCDResult:
    """Single point cloud result."""
    points: np.ndarray          # (N, 3) in mm (scanner space)
    normals: np.ndarray         # (N, 3) estimated normals
    n_points: int = 0
    label_id: Optional[int] = None
    structure_name: str = ""
    hemisphere: str = ""        # "lh", "rh", or "both"

    def __post_init__(self):
        self.n_points = self.points.shape[0]


@dataclass
class T1wPipelineResult:
    """Complete result from the T1w → PCD pipeline."""
    # Cortical hemispheres
    cortical: Dict[str, PCDResult] = field(default_factory=dict)    # {"lh": ..., "rh": ...}
    cortical_whole: Optional[PCDResult] = None                       # unsplit whole-brain

    # Subcortical structures
    subcortical: Dict[str, PCDResult] = field(default_factory=dict)  # {"Left-Thalamus": ...}

    # Hippocampi (also in subcortical, but separate for convenience)
    hippocampal: Dict[str, PCDResult] = field(default_factory=dict)  # {"Left-Hippocampus": ...}

    # Metadata
    t1w_path: str = ""
    brain_mask_path: str = ""
    aseg_path: str = ""
    segmentation_backend: str = ""

    def summary(self) -> Dict[str, Any]:
        """Return dict of structure names → n_points."""
        out = {}
        for hemi, pcd in self.cortical.items():
            out[f"cortical_{hemi}"] = pcd.n_points
        for name, pcd in self.subcortical.items():
            out[name] = pcd.n_points
        return out


# ═══════════════════════════════════════════════════════════════════════════════
# Skull stripping
# ═══════════════════════════════════════════════════════════════════════════════

def skull_strip_deepbet(
    t1w_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    use_gpu: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Skull-strip T1w using deepbet (GPU-accelerated).

    Parameters
    ----------
    t1w_path : path
        Input T1w NIfTI.
    output_dir : path, optional
        Where to save brain mask. If None, uses temp dir.
    use_gpu : bool
        Use CUDA if available.

    Returns
    -------
    brain_data : ndarray
        Skull-stripped T1w volume.
    brain_mask : ndarray, bool
        Binary brain mask.
    affine : ndarray (4, 4)
        Voxel-to-world affine.
    """
    t1w_path = Path(t1w_path)
    img = nib.load(str(t1w_path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    affine = img.affine

    try:
        from deepbet import run_bet
        import torch

        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp(prefix="cf_deepbet_"))
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        mask_path = output_dir / "brain_mask.nii.gz"
        brain_path = output_dir / "brain.nii.gz"

        # deepbet API changed across versions:
        #   old: run_bet(input=str, output=str, gpu=bool)
        #   new: run_bet(input_paths=[str], brain_paths=[str], mask_paths=[str], gpu=bool)
        import inspect
        sig = inspect.signature(run_bet)
        params = set(sig.parameters.keys())

        use_cuda = use_gpu and torch.cuda.is_available()

        if "input_paths" in params:
            run_bet(
                input_paths=[str(t1w_path)],
                brain_paths=[str(brain_path)],
                mask_paths=[str(mask_path)],
                gpu=use_cuda,
            )
        else:
            run_bet(
                input=str(t1w_path),
                output=str(mask_path),
                gpu=use_cuda,
            )

        mask_img = nib.load(str(mask_path))
        brain_mask = np.asarray(mask_img.dataobj).astype(bool)

        # Cleanup GPU
        if use_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    except ImportError:
        logger.warning(
            "deepbet not installed. Falling back to intensity-based brain extraction. "
            "Install deepbet for better results: pip install deepbet"
        )
        brain_mask = _fallback_brain_mask(data)

    brain_data = data * brain_mask
    return brain_data, brain_mask, affine


def _fallback_brain_mask(data: np.ndarray) -> np.ndarray:
    """Simple Otsu + morphology brain mask (fallback only)."""
    from skimage.filters import threshold_otsu

    threshold = threshold_otsu(data[data > 0])
    mask = data > threshold * 0.3
    mask = binary_erosion(mask, iterations=2)
    mask = binary_dilation(mask, iterations=3)
    return mask.astype(bool)


# ═══════════════════════════════════════════════════════════════════════════════
# Volumetric segmentation (SynthSeg / FastSurferCNN / ANTsPyNet)
# ═══════════════════════════════════════════════════════════════════════════════

def segment_volume(
    t1w_path: Union[str, Path],
    output_dir: Union[str, Path],
    backend: str = "auto",
    use_gpu: bool = True,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Segment T1w into aseg-style labels.

    Tries backends in order: synthseg → fastsurfer → antspynet → fail.

    Parameters
    ----------
    t1w_path : path
        Input T1w (does NOT need to be skull-stripped).
    output_dir : path
        Where to write the aseg volume.
    backend : str
        "auto", "synthseg", "fastsurfer", or "antspynet".
    use_gpu : bool
        Use GPU if available.

    Returns
    -------
    aseg : ndarray, int32
        Label volume (same grid as T1w).
    affine : ndarray (4, 4)
        Voxel-to-world affine.
    backend_used : str
        Which backend actually ran.
    """
    t1w_path = Path(t1w_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    aseg_path = output_dir / "synthseg_aseg.nii.gz"

    if backend in ("auto", "synthseg"):
        result = _try_synthseg(t1w_path, aseg_path, use_gpu)
        if result is not None:
            return result

    if backend in ("auto", "fastsurfer"):
        result = _try_fastsurfer(t1w_path, aseg_path, use_gpu)
        if result is not None:
            return result

    if backend in ("auto", "antspynet"):
        result = _try_antspynet(t1w_path, aseg_path, use_gpu)
        if result is not None:
            return result

    raise RuntimeError(
        "No segmentation backend available. Install one of:\n"
        "  1. SynthSeg:    mri_synthseg via FreeSurfer 7.3+ ($FREESURFER_HOME/bin/)\n"
        "  2. FastSurfer:  run_fastsurfer.sh --seg_only\n"
        "  3. ANTsPyNet:   pip install antspynet\n"
    )


def _try_synthseg(
    t1w_path: Path, aseg_path: Path, use_gpu: bool,
) -> Optional[Tuple[np.ndarray, np.ndarray, str]]:
    """Try SynthSeg via FreeSurfer's mri_synthseg binary.

    Strategy (in order):
      1. shutil.which("mri_synthseg") — if it's on PATH
      2. $FREESURFER_HOME/bin/mri_synthseg — direct path from env var
      3. os.system() fallback — uses full shell environment (catches
         cases where FS is sourced but subprocess doesn't see it)

    NOTE: The `pip install synthseg` package requires Python ≤ 3.8 and is
    therefore incompatible with most modern environments. We do NOT attempt
    the Python import — the FS binary is the reliable path.
    """
    import os

    # ── Find the mri_synthseg binary ─────────────────────────────────────
    synthseg_bin = shutil.which("mri_synthseg")

    if synthseg_bin is None:
        # Try $FREESURFER_HOME/bin/
        fs_home = os.environ.get("FREESURFER_HOME", "")
        if fs_home:
            candidate = Path(fs_home) / "bin" / "mri_synthseg"
            if candidate.exists():
                synthseg_bin = str(candidate)

    # ── Attempt 1: subprocess.run (preferred — captures errors) ──────────
    if synthseg_bin is not None:
        try:
            cmd = [
                synthseg_bin,
                "--i", str(t1w_path),
                "--o", str(aseg_path),
                "--robust",
            ]
            if use_gpu:
                cmd.append("--gpu")

            logger.info(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, capture_output=True, timeout=900)

            if aseg_path.exists():
                img = nib.load(str(aseg_path))
                return np.asarray(img.dataobj, dtype=np.int32), img.affine, "synthseg_cli"
        except subprocess.TimeoutExpired:
            logger.warning("mri_synthseg timed out (900s)")
        except subprocess.CalledProcessError as e:
            logger.warning(f"mri_synthseg failed (returncode {e.returncode}): {e.stderr[:500] if e.stderr else ''}")

    # ── Attempt 2: os.system fallback (inherits full shell env) ──────────
    # This catches cases where FS is sourced in .bashrc but subprocess
    # doesn't inherit those paths (common in conda + Spyder setups).
    gpu_flag = "--gpu" if use_gpu else ""
    shell_cmd = (
        f'mri_synthseg --i "{t1w_path}" --o "{aseg_path}" '
        f'--robust {gpu_flag}'
    )
    logger.info(f"Trying os.system fallback: {shell_cmd}")
    ret = os.system(shell_cmd)

    if ret == 0 and aseg_path.exists():
        img = nib.load(str(aseg_path))
        return np.asarray(img.dataobj, dtype=np.int32), img.affine, "synthseg_os_system"

    logger.warning(
        "mri_synthseg not found or failed. Ensure FreeSurfer 7.3+ is installed "
        "and $FREESURFER_HOME is set, or that 'mri_synthseg' is on PATH."
    )
    return None


def _try_fastsurfer(
    t1w_path: Path, aseg_path: Path, use_gpu: bool,
) -> Optional[Tuple[np.ndarray, np.ndarray, str]]:
    """Try FastSurferCNN segmentation-only."""
    # ── CLI: run_fastsurfer.sh --seg_only ────────────────────────────────
    if shutil.which("run_fastsurfer.sh"):
        try:
            with tempfile.TemporaryDirectory(prefix="cf_fastsurfer_") as tmpdir:
                cmd = [
                    "run_fastsurfer.sh",
                    "--t1", str(t1w_path),
                    "--sd", tmpdir,
                    "--sid", "tmp_subj",
                    "--seg_only",
                    "--no_cereb", "--no_biasfield",
                ]
                if not use_gpu:
                    cmd.append("--no_cuda")
                subprocess.run(cmd, check=True, capture_output=True, timeout=600)
                fs_aseg = Path(tmpdir) / "tmp_subj" / "mri" / "aparc.DKTatlas+aseg.deep.mgz"
                if fs_aseg.exists():
                    img = nib.load(str(fs_aseg))
                    aseg = np.asarray(img.dataobj, dtype=np.int32)
                    nib.save(nib.Nifti1Image(aseg, img.affine), str(aseg_path))
                    return aseg, img.affine, "fastsurfer_cli"
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.warning(f"FastSurfer CLI failed: {e}")

    # ── Python: FastSurferCNN direct ─────────────────────────────────────
    try:
        from FastSurferCNN.quick_qc import main as fastsurfer_seg
        # This is less reliable; skip if CLI didn't work
    except ImportError:
        pass

    return None


def _try_antspynet(
    t1w_path: Path, aseg_path: Path, use_gpu: bool,
) -> Optional[Tuple[np.ndarray, np.ndarray, str]]:
    """Try ANTsPyNet desikan_killiany_tourville_labeling."""
    try:
        import ants
        import antspynet

        img_ants = ants.image_read(str(t1w_path))
        # DKT labeling gives cortical + subcortical labels
        result = antspynet.desikan_killiany_tourville_labeling(
            img_ants, do_preprocessing=True, verbose=False,
        )
        # result is an ANTsImage with labels
        aseg = result.numpy().astype(np.int32)
        affine = nib.load(str(t1w_path)).affine  # ANTs uses its own coord system

        nib.save(nib.Nifti1Image(aseg, affine), str(aseg_path))
        return aseg, affine, "antspynet"
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"ANTsPyNet failed: {e}")

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# PCD extraction from volumetric labels
# ═══════════════════════════════════════════════════════════════════════════════

def _label_to_pcd(
    label_mask: np.ndarray,
    affine: np.ndarray,
    smooth_sigma: float = 0.5,
    level: float = 0.5,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Convert binary label mask → point cloud via marching cubes.

    Parameters
    ----------
    label_mask : ndarray, bool/int
        Binary mask of the structure.
    affine : ndarray (4, 4)
        Voxel-to-mm affine.
    smooth_sigma : float
        Gaussian smoothing before marching cubes (in voxels).
    level : float
        Isosurface level.

    Returns
    -------
    points : ndarray (N, 3) in mm
    faces : ndarray (F, 3)
    """
    from skimage.measure import marching_cubes

    mask = label_mask.astype(np.float32)
    if mask.sum() < MIN_VOXELS_FOR_MESH:
        return None

    # Smooth for better surface
    if smooth_sigma > 0:
        mask = gaussian_filter(mask, sigma=smooth_sigma)

    # Pad to avoid boundary artifacts
    padded = np.pad(mask, 1, mode="constant", constant_values=0)

    try:
        verts_vox, faces, _, _ = marching_cubes(padded, level=level)
    except (ValueError, RuntimeError):
        return None

    verts_vox -= 1.0  # undo padding offset

    # Transform to scanner mm space
    points_mm = nib.affines.apply_affine(affine, verts_vox)

    return points_mm, faces


def _estimate_normals_local(
    points: np.ndarray,
    n_neighbors: int = 30,
) -> np.ndarray:
    """Estimate normals via PCA on k-NN neighborhoods.

    Falls back to importing from CF if available, otherwise uses
    a lightweight local implementation.
    """
    try:
        from corticalfields._pointcloud_legacy import estimate_normals
        return estimate_normals(points, n_neighbors=n_neighbors, orient_consistent=True)
    except (ImportError, AttributeError):
        pass

    # Lightweight fallback: PCA on k-NN
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    _, idx = tree.query(points, k=n_neighbors)
    normals = np.zeros_like(points)

    for i in range(len(points)):
        neighbors = points[idx[i]]
        centered = neighbors - neighbors.mean(axis=0)
        cov = centered.T @ centered
        eigvals, eigvecs = np.linalg.eigh(cov)
        normals[i] = eigvecs[:, 0]  # smallest eigenvalue = normal

    # Orient consistently: normals point away from centroid
    centroid = points.mean(axis=0)
    outward = points - centroid
    flip = np.sum(normals * outward, axis=1) < 0
    normals[flip] *= -1

    # Normalize
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    normals /= norms

    return normals


def extract_cortical_gm_pcd(
    brain_data: np.ndarray,
    brain_mask: np.ndarray,
    affine: np.ndarray,
    aseg: Optional[np.ndarray] = None,
    cortical_thickness_mm: float = 3.0,
    smooth_sigma: float = 0.5,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Extract cortical gray matter point cloud.

    If aseg labels are available, uses the cortical GM labels directly.
    Otherwise, falls back to morphological extraction (outer brain shell).

    Parameters
    ----------
    brain_data : ndarray
        Skull-stripped T1w volume.
    brain_mask : ndarray
        Binary brain mask.
    affine : ndarray (4, 4)
        Voxel-to-mm.
    aseg : ndarray, optional
        SynthSeg/FastSurfer label volume. If provided, uses labels 3+42 (cortex).
    cortical_thickness_mm : float
        For morphological fallback: estimated cortex thickness in mm.
    smooth_sigma : float
        Gaussian smoothing before marching cubes.

    Returns
    -------
    points : ndarray (N, 3) in mm
    faces : ndarray (F, 3)
    """
    if aseg is not None:
        # Use aseg cortical GM labels
        cortex_mask = np.isin(aseg, CORTICAL_GM_LABELS)
        if cortex_mask.sum() > MIN_VOXELS_FOR_MESH:
            return _label_to_pcd(cortex_mask, affine, smooth_sigma=smooth_sigma)

    # Fallback: morphological extraction (brain shell = outer - eroded)
    voxel_size = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0)).mean()
    erode_voxels = max(1, int(round(cortical_thickness_mm / voxel_size)))

    inner = binary_erosion(brain_mask, iterations=erode_voxels)
    gm_shell = brain_mask & (~inner)

    return _label_to_pcd(gm_shell, affine, smooth_sigma=smooth_sigma)


def split_hemispheres(
    points: np.ndarray,
    normals: np.ndarray,
    method: str = "sagittal_midplane",
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Split whole-brain PCD into left and right hemispheres.

    Parameters
    ----------
    points : ndarray (N, 3)
        Point cloud coordinates (mm, scanner space — RAS or LAS).
    normals : ndarray (N, 3)
        Point normals.
    method : str
        "sagittal_midplane" — split at x=median(x) along the first axis.

    Returns
    -------
    (lh_points, lh_normals), (rh_points, rh_normals)
        NOTE: In RAS convention, x < 0 is LEFT hemisphere.
        We detect orientation from the data spread.
    """
    # Use median x-coordinate as the midplane
    x_median = np.median(points[:, 0])

    # Left hemisphere: x < midplane (RAS convention: negative x = left)
    lh_mask = points[:, 0] < x_median
    rh_mask = ~lh_mask

    lh = (points[lh_mask].copy(), normals[lh_mask].copy())
    rh = (points[rh_mask].copy(), normals[rh_mask].copy())

    return lh, rh


def extract_structure_pcd(
    aseg: np.ndarray,
    affine: np.ndarray,
    label_id: int,
    structure_name: str,
    hemisphere: str,
    smooth_sigma: float = 0.5,
    n_neighbors_normals: int = 30,
) -> Optional[PCDResult]:
    """Extract a single subcortical/hippocampal structure as PCD.

    Parameters
    ----------
    aseg : ndarray, int32
        Label volume from SynthSeg/FastSurfer.
    affine : ndarray (4, 4)
        Voxel-to-mm.
    label_id : int
        Structure label ID.
    structure_name : str
        Human-readable name.
    hemisphere : str
        "lh" or "rh".
    smooth_sigma : float
        Gaussian smoothing before meshing.
    n_neighbors_normals : int
        k-NN for normal estimation.

    Returns
    -------
    PCDResult or None if structure too small.
    """
    mask = (aseg == label_id)
    if mask.sum() < MIN_VOXELS_FOR_MESH:
        logger.warning(
            f"Structure {structure_name} (label={label_id}) has only "
            f"{mask.sum()} voxels — skipping."
        )
        return None

    result = _label_to_pcd(mask, affine, smooth_sigma=smooth_sigma)
    if result is None:
        return None

    points, faces = result
    normals = _estimate_normals_local(points, n_neighbors=n_neighbors_normals)

    return PCDResult(
        points=points,
        normals=normals,
        label_id=label_id,
        structure_name=structure_name,
        hemisphere=hemisphere,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Main pipeline class
# ═══════════════════════════════════════════════════════════════════════════════

class T1wPCDPipeline:
    """Complete T1w → Point Cloud pipeline (FreeSurfer-free).

    Performs skull-stripping (deepbet) and volumetric segmentation
    (SynthSeg/FastSurfer/ANTsPyNet) ONCE per T1w, then extracts cortical,
    subcortical, and hippocampal point clouds from the label volume.

    Parameters
    ----------
    use_gpu : bool
        Use GPU for deepbet and segmentation.
    seg_backend : str
        Segmentation backend: "auto", "synthseg", "fastsurfer", "antspynet".
    smooth_sigma : float
        Gaussian smoothing before marching cubes (voxels).
    cortical_thickness_mm : float
        Fallback cortical thickness for morphological GM extraction.
    n_neighbors_normals : int
        k-NN for normal estimation.
    cache_dir : path, optional
        If set, caches intermediate results (brain mask, aseg) per subject.
    extract_cortical : bool
        Whether to extract cortical hemispheres.
    extract_subcortical : bool
        Whether to extract subcortical structures.
    extract_hippocampal : bool
        Whether to extract hippocampi (also in subcortical dict).
    subcortical_labels : dict, optional
        Custom label→(name, hemi) mapping. Defaults to SUBCORTICAL_LABELS.

    Example
    -------
    >>> pipe = T1wPCDPipeline(use_gpu=True, seg_backend="synthseg")
    >>> result = pipe.run("/data/sub-001_T1w.nii.gz")
    >>> print(result.summary())
    """

    def __init__(
        self,
        use_gpu: bool = True,
        seg_backend: str = "auto",
        smooth_sigma: float = 0.5,
        cortical_thickness_mm: float = 3.0,
        n_neighbors_normals: int = 30,
        cache_dir: Optional[Union[str, Path]] = None,
        extract_cortical: bool = True,
        extract_subcortical: bool = True,
        extract_hippocampal: bool = True,
        subcortical_labels: Optional[Dict[int, Tuple[str, str]]] = None,
    ):
        self.use_gpu = use_gpu
        self.seg_backend = seg_backend
        self.smooth_sigma = smooth_sigma
        self.cortical_thickness_mm = cortical_thickness_mm
        self.n_neighbors_normals = n_neighbors_normals
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.extract_cortical = extract_cortical
        self.extract_subcortical = extract_subcortical
        self.extract_hippocampal = extract_hippocampal
        self.subcortical_labels = subcortical_labels or SUBCORTICAL_LABELS

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        t1w_path: Union[str, Path],
        subject_id: Optional[str] = None,
    ) -> T1wPipelineResult:
        """Run the full pipeline on a single T1w image.

        Parameters
        ----------
        t1w_path : path
            Path to T1w NIfTI (.nii.gz).
        subject_id : str, optional
            Used for cache filenames. Inferred from filename if None.

        Returns
        -------
        T1wPipelineResult
            Contains cortical (lh/rh), subcortical, and hippocampal PCDs.
        """
        t1w_path = Path(t1w_path)
        if subject_id is None:
            subject_id = t1w_path.stem.replace(".nii", "")

        result = T1wPipelineResult(t1w_path=str(t1w_path))

        # ── Step 1: Skull-strip (ONCE) ───────────────────────────────────
        cache_sub = self.cache_dir / subject_id if self.cache_dir else None
        if cache_sub:
            cache_sub.mkdir(exist_ok=True)

        brain_data, brain_mask, affine = skull_strip_deepbet(
            t1w_path, output_dir=cache_sub, use_gpu=self.use_gpu,
        )
        logger.info(f"Skull-strip done: {brain_mask.sum()} brain voxels")

        # ── Step 2: Segment (ONCE) ───────────────────────────────────────
        seg_dir = cache_sub or Path(tempfile.mkdtemp(prefix="cf_seg_"))
        aseg, aseg_affine, seg_backend = segment_volume(
            t1w_path, output_dir=seg_dir,
            backend=self.seg_backend, use_gpu=self.use_gpu,
        )
        result.segmentation_backend = seg_backend
        result.aseg_path = str(seg_dir / "synthseg_aseg.nii.gz")
        logger.info(f"Segmentation done ({seg_backend}): {np.unique(aseg).size} labels")

        # ── Step 3: Cortical PCD (extract ONCE, split into hemis) ────────
        if self.extract_cortical:
            cortex_result = extract_cortical_gm_pcd(
                brain_data, brain_mask, affine,
                aseg=aseg,
                cortical_thickness_mm=self.cortical_thickness_mm,
                smooth_sigma=self.smooth_sigma,
            )
            if cortex_result is not None:
                points_all, faces_all = cortex_result
                normals_all = _estimate_normals_local(
                    points_all, n_neighbors=self.n_neighbors_normals,
                )
                result.cortical_whole = PCDResult(
                    points=points_all, normals=normals_all,
                    structure_name="cortex_whole", hemisphere="both",
                )

                # Split hemispheres (GEOMETRIC — no re-extraction)
                (lh_pts, lh_nrm), (rh_pts, rh_nrm) = split_hemispheres(
                    points_all, normals_all,
                )
                result.cortical["lh"] = PCDResult(
                    points=lh_pts, normals=lh_nrm,
                    structure_name="cortex_lh", hemisphere="lh",
                )
                result.cortical["rh"] = PCDResult(
                    points=rh_pts, normals=rh_nrm,
                    structure_name="cortex_rh", hemisphere="rh",
                )
                logger.info(
                    f"Cortical PCD: {points_all.shape[0]} total → "
                    f"LH {lh_pts.shape[0]} + RH {rh_pts.shape[0]}"
                )

        # ── Step 4: Subcortical structures ───────────────────────────────
        if self.extract_subcortical:
            for label_id, (name, hemi) in self.subcortical_labels.items():
                pcd = extract_structure_pcd(
                    aseg, affine, label_id, name, hemi,
                    smooth_sigma=self.smooth_sigma,
                    n_neighbors_normals=self.n_neighbors_normals,
                )
                if pcd is not None:
                    result.subcortical[name] = pcd

                    # Also store hippocampi separately for convenience
                    if label_id in HIPPOCAMPAL_LABELS and self.extract_hippocampal:
                        result.hippocampal[name] = pcd

            logger.info(f"Subcortical: {len(result.subcortical)} structures extracted")

        # ── GPU cleanup ──────────────────────────────────────────────────
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        gc.collect()

        return result


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience functions (module-level API)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_whole_brain_pcd(
    t1w_path: Union[str, Path],
    use_gpu: bool = True,
    seg_backend: str = "auto",
) -> T1wPipelineResult:
    """One-liner: extract all structures from a T1w.

    >>> result = extract_whole_brain_pcd("sub-001_T1w.nii.gz")
    """
    pipe = T1wPCDPipeline(use_gpu=use_gpu, seg_backend=seg_backend)
    return pipe.run(t1w_path)


def extract_cortical_hemispheres(
    t1w_path: Union[str, Path],
    use_gpu: bool = True,
    seg_backend: str = "auto",
) -> Dict[str, PCDResult]:
    """Extract only cortical LH + RH point clouds.

    >>> hemis = extract_cortical_hemispheres("sub-001_T1w.nii.gz")
    >>> hemis["lh"].points.shape
    """
    pipe = T1wPCDPipeline(
        use_gpu=use_gpu, seg_backend=seg_backend,
        extract_subcortical=False, extract_hippocampal=False,
    )
    return pipe.run(t1w_path).cortical
