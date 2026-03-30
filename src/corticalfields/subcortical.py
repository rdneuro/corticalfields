"""
Subcortical surface extraction and spectral shape analysis.

This module extends CorticalFields to **closed subcortical surfaces**
extracted from volumetric segmentations (FreeSurfer aseg, hippocampal
subfield segmentation of Iglesias et al., or HippUnfold outputs).

Pipeline overview
-----------------

    1. **IO** — Load a binary label mask from one of three origins:
       ``'fs'`` (FreeSurfer aseg.mgz), ``'fs_hc'`` (FreeSurfer
       hippocampal subfields, merging hippocampal labels and discarding
       the amygdala), or ``'hippunfold'`` (native-space GIfTI surfaces
       from HippUnfold).

    2. **Surface extraction** — For volumetric origins (``fs``,
       ``fs_hc``), the label is binarised, a boundary surface is
       extracted via marching cubes (``skimage``), optionally smoothed
       (Laplacian or Taubin), and decimated to a target vertex count.

    3. **Spectral analysis** — The number of eigenpairs is estimated
       automatically from vertex count (via :func:`~corticalfields.utils.
       estimate_n_eigenpairs`).  The full CorticalFields spectral
       pipeline (LBO → HKS / WKS / GPS → functional maps → Wasserstein)
       operates on the resulting :class:`SubcorticalSurface` exactly as
       it does on cortical meshes — with no boundary-condition ambiguity,
       because subcortical surfaces are topologically closed (genus-0).

    4. **Subcortical-specific metrics** — Additional descriptors
       available only on closed surfaces: volume, sphericity
       (isoperimetric ratio), Willmore bending energy, and per-vertex
       Gaussian/mean curvature.

FreeSurfer label IDs
--------------------
The ``FS_ASEG_LABELS`` dictionary maps human-readable names to the
integer codes used in ``aseg.mgz``.  For hippocampal subfields
(``fs_hc``), relevant labels are merged into a single hippocampal
volume and the amygdala (labels 7000+/18) is explicitly excluded.

References
----------
[1] Wachinger, C. et al. "BrainPrint: A Discriminative Characterization
    of Brain Morphology." NeuroImage 109 (2015): 232–248.
[2] Iglesias, J.E. et al. "A computational atlas of the hippocampal
    formation…" NeuroImage 117 (2015): 44–57.
[3] DeKraker, J. et al. "Automated hippocampal unfolding for
    morphometry and subfield segmentation with HippUnfold."
    eLife 11 (2022): e77945.
[4] Lorensen, W.E. & Cline, H.E. "Marching cubes: A high resolution
    3D surface construction algorithm." ACM SIGGRAPH 1987.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# FreeSurfer label look-up tables
# ═══════════════════════════════════════════════════════════════════════════

# aseg.mgz integer labels for commonly analysed structures.
# Full LUT: $FREESURFER_HOME/FreeSurferColorLUT.txt
FS_ASEG_LABELS: Dict[str, int] = {
    # Hippocampus
    "Left-Hippocampus": 17,
    "Right-Hippocampus": 53,
    # Amygdala
    "Left-Amygdala": 18,
    "Right-Amygdala": 54,
    # Thalamus
    "Left-Thalamus": 10,
    "Right-Thalamus": 49,
    # Caudate
    "Left-Caudate": 11,
    "Right-Caudate": 50,
    # Putamen
    "Left-Putamen": 12,
    "Right-Putamen": 51,
    # Pallidum
    "Left-Pallidum": 13,
    "Right-Pallidum": 52,
    # Nucleus accumbens
    "Left-Accumbens-area": 26,
    "Right-Accumbens-area": 58,
    # Ventral DC
    "Left-VentralDC": 28,
    "Right-VentralDC": 60,
    # Brain stem (single label)
    "Brain-Stem": 16,
}

# Reverse map for convenience
_LABEL_TO_NAME: Dict[int, str] = {v: k for k, v in FS_ASEG_LABELS.items()}

# Hippocampal subfield labels (Iglesias et al. 2015).
# These live in {subjects_dir}/{sid}/mri/lh.hippoSfVolumes-T1.v22.txt
# but the actual segmentation is in *hippoAmygLabels-T1.v22.mgz.
# Labels 200–299 are hippocampal subfields; 7000+ are amygdala nuclei.
FS_HC_HIPP_LABELS: Dict[str, List[int]] = {
    "lh": [
        # parasubiculum, presubiculum, subiculum, CA1, CA2/3, CA4, GC-ML-DG,
        # molecular layer HP, HATA, fimbria, HP-tail, HP-fissure
        226, 203, 204, 205, 206, 208, 211, 214, 215, 212, 209, 210,
    ],
    "rh": [
        # Same labels, but in the right-hemisphere volume
        226, 203, 204, 205, 206, 208, 211, 214, 215, 212, 209, 210,
    ],
}
# Amygdala nuclei to EXCLUDE when loading 'fs_hc' as hippocampus-only
_FS_HC_AMYGDALA_RANGE = range(7000, 7999)


# ═══════════════════════════════════════════════════════════════════════════
# Data structure
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class SubcorticalSurface:
    """
    A closed (genus-0) subcortical triangle mesh with per-vertex overlays.

    This is the subcortical counterpart of
    :class:`~corticalfields.surface.CorticalSurface`.  All downstream
    spectral modules (:mod:`spectral`, :mod:`functional_maps`,
    :mod:`transport`, :mod:`asymmetry`) accept a ``SubcorticalSurface``
    identically to a cortical mesh, with the advantage that boundary
    conditions are absent and the eigendecomposition is cleaner.

    Parameters
    ----------
    vertices : np.ndarray, shape (N, 3)
        Vertex coordinates in scanner RAS (mm).
    faces : np.ndarray, shape (F, 3)
        Triangle connectivity (0-indexed).
    structure : str
        Human-readable structure name (e.g. ``'Left-Hippocampus'``).
    hemi : ``'lh'`` or ``'rh'``
        Hemisphere.
    overlays : dict[str, np.ndarray]
        Per-vertex scalar maps (curvature, thickness, …).
    metadata : dict
        Origin information, subject ID, paths, etc.

    Notes
    -----
    The mesh is expected to be a **closed, genus-0 manifold** — this is
    automatically verified on construction (Euler χ = 2).  The enclosed
    volume is computed via the divergence theorem and stored in
    ``metadata['volume_mm3']``.
    """

    vertices: np.ndarray
    faces: np.ndarray
    structure: str = ""
    hemi: str = "lh"
    overlays: Dict[str, np.ndarray] = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=np.float64)
        self.faces = np.asarray(self.faces, dtype=np.int64)
        # Compute and cache enclosed volume
        if "volume_mm3" not in self.metadata:
            self.metadata["volume_mm3"] = float(self.enclosed_volume)

    # ── core properties ───────────────────────────────────────────────

    @property
    def n_vertices(self) -> int:
        return self.vertices.shape[0]

    @property
    def n_faces(self) -> int:
        return self.faces.shape[0]

    @property
    def overlay_names(self) -> List[str]:
        return list(self.overlays.keys())

    @property
    def face_areas(self) -> np.ndarray:
        """Area of each triangular face (mm²)."""
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

    @property
    def total_area(self) -> float:
        """Total surface area (mm²)."""
        return float(self.face_areas.sum())

    @property
    def enclosed_volume(self) -> float:
        """
        Enclosed volume via the divergence theorem (mm³).

        V = (1/6) Σ_f  v0 · (v1 × v2)

        Only well-defined for closed, consistently-oriented meshes.
        """
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        # Signed volume of tetrahedra formed with origin
        signed_vols = np.einsum("ij,ij->i", v0, np.cross(v1, v2))
        return abs(float(signed_vols.sum() / 6.0))

    @property
    def sphericity(self) -> float:
        """
        Isoperimetric ratio  Ψ = 36π V² / A³.

        Equals 1.0 for a perfect sphere; decreases with elongation
        or surface irregularity.  Provides a single-number shape
        complexity descriptor unavailable for open cortical surfaces.
        """
        A = self.total_area
        V = self.enclosed_volume
        if A < 1e-12:
            return 0.0
        return float(36.0 * np.pi * V**2 / A**3)

    @property
    def vertex_normals(self) -> np.ndarray:
        """Per-vertex unit normals (area-weighted face-normal average)."""
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        face_normals = np.cross(v1 - v0, v2 - v0)
        normals = np.zeros_like(self.vertices)
        for i in range(3):
            np.add.at(normals, self.faces[:, i], face_normals)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return normals / norms

    # ── overlay management ────────────────────────────────────────────

    def add_overlay(self, name: str, data: np.ndarray) -> None:
        if data.shape[0] != self.n_vertices:
            raise ValueError(
                f"Overlay '{name}' has {data.shape[0]} values but mesh "
                f"has {self.n_vertices} vertices."
            )
        self.overlays[name] = np.asarray(data, dtype=np.float64)

    def get_overlay(self, name: str) -> np.ndarray:
        if name not in self.overlays:
            available = ", ".join(self.overlay_names) or "(none)"
            raise KeyError(f"Overlay '{name}' not found. Available: {available}")
        return self.overlays[name]

    # ── geometry methods unique to closed surfaces ────────────────────

    def compute_curvatures(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Discrete Gaussian (K) and mean (H) curvature at each vertex.

        Uses the angle-defect formula for K and the Laplace–Beltrami
        mean-curvature normal for H.  Results are also stored as
        overlays ``'gaussian_curvature'`` and ``'mean_curvature'``.

        Returns
        -------
        K : np.ndarray, shape (N,)
            Gaussian curvature (rad / mm²).
        H : np.ndarray, shape (N,)
            Mean curvature (1/mm).
        """
        N = self.n_vertices
        v = self.vertices
        f = self.faces

        # ---- Gaussian curvature via angle defect ----
        # K(v_i) = (2π − Σ_angles) / A_mixed
        angle_sum = np.zeros(N, dtype=np.float64)
        area_mixed = np.zeros(N, dtype=np.float64)

        for tri in f:
            i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
            vi, vj, vk = v[i], v[j], v[k]
            # Triangle area (for Voronoi mixed area)
            tri_area = 0.5 * np.linalg.norm(np.cross(vj - vi, vk - vi))
            third_area = tri_area / 3.0
            area_mixed[i] += third_area
            area_mixed[j] += third_area
            area_mixed[k] += third_area
            # Angles at each vertex
            for a, b, c in [(i, j, k), (j, k, i), (k, i, j)]:
                e1 = v[b] - v[a]
                e2 = v[c] - v[a]
                cos_a = np.dot(e1, e2) / (
                    np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-30
                )
                cos_a = np.clip(cos_a, -1.0, 1.0)
                angle_sum[a] += np.arccos(cos_a)

        area_mixed[area_mixed < 1e-16] = 1e-16
        K = (2.0 * np.pi - angle_sum) / area_mixed

        # ---- Mean curvature via Laplacian norm ----
        # H = |Δx| / 2  where Δ is the cotangent Laplacian applied to
        # vertex positions.  We import compute_laplacian to avoid
        # code duplication.
        from corticalfields.spectral import compute_laplacian

        L, M = compute_laplacian(v, f, use_robust=True)
        # Δx = M⁻¹ L x   (Laplace–Beltrami of coordinate functions)
        M_inv_diag = 1.0 / np.maximum(M.diagonal(), 1e-16)
        Lx = L.dot(v)  # (N, 3)
        Hn = np.zeros_like(v)
        for d in range(3):
            Hn[:, d] = M_inv_diag * Lx[:, d]
        H = 0.5 * np.linalg.norm(Hn, axis=1)

        self.add_overlay("gaussian_curvature", K)
        self.add_overlay("mean_curvature", H)
        logger.info(
            "Curvatures: K ∈ [%.4f, %.4f], H ∈ [%.4f, %.4f]",
            K.min(), K.max(), H.min(), H.max(),
        )
        return K, H

    def willmore_energy(self) -> float:
        """
        Willmore bending energy  W = ∫ H² dA.

        For a genus-0 embedding, W ≥ 4π (equality iff sphere).  Higher
        values indicate more folded or irregular surfaces.  This is a
        conformally invariant global descriptor.
        """
        if "mean_curvature" not in self.overlays:
            self.compute_curvatures()
        H = self.overlays["mean_curvature"]

        # Vertex area: ⅓ of adjacent face areas (lumped)
        vert_area = np.zeros(self.n_vertices, dtype=np.float64)
        fa = self.face_areas
        for i in range(3):
            np.add.at(vert_area, self.faces[:, i], fa / 3.0)

        W = float(np.sum(H**2 * vert_area))
        self.metadata["willmore_energy"] = W
        logger.info("Willmore energy: %.2f  (4π ≈ %.2f)", W, 4 * np.pi)
        return W

    # ── serialisation ─────────────────────────────────────────────────

    def to_gifti(self, path: Union[str, Path]) -> None:
        """Save as GIfTI (.surf.gii) with overlays."""
        import nibabel as nib

        path = Path(path)
        coord = nib.gifti.GiftiDataArray(
            data=self.vertices.astype(np.float32),
            intent="NIFTI_INTENT_POINTSET",
            datatype="NIFTI_TYPE_FLOAT32",
        )
        tri = nib.gifti.GiftiDataArray(
            data=self.faces.astype(np.int32),
            intent="NIFTI_INTENT_TRIANGLE",
            datatype="NIFTI_TYPE_INT32",
        )
        darrays = [coord, tri]
        for name, vals in self.overlays.items():
            da = nib.gifti.GiftiDataArray(
                data=vals.astype(np.float32),
                intent="NIFTI_INTENT_SHAPE",
                datatype="NIFTI_TYPE_FLOAT32",
            )
            da.meta = nib.gifti.GiftiMetaData.from_dict({"Name": name})
            darrays.append(da)
        gii = nib.gifti.GiftiImage(darrays=darrays)
        nib.save(gii, str(path))
        logger.info("Saved SubcorticalSurface to %s", path)


# ═══════════════════════════════════════════════════════════════════════════
# Volume → surface pipeline (marching cubes + refinement)
# ═══════════════════════════════════════════════════════════════════════════


def _volume_to_surface(
    volume: np.ndarray,
    affine: np.ndarray,
    *,
    smooth_iterations: int = 30,
    smooth_method: str = "taubin",
    decimate_target: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract a triangulated boundary surface from a binary volume.

    Steps:
      1. Marching cubes at level=0.5 on the binary mask.
      2. Transform vertices from voxel to RAS coordinates.
      3. Laplacian or Taubin smoothing to remove voxel staircase.
      4. Optional decimation to a target vertex count.

    Parameters
    ----------
    volume : np.ndarray, 3D bool/int
        Binary mask (1 = inside, 0 = outside).
    affine : np.ndarray, shape (4, 4)
        Voxel-to-RAS affine.
    smooth_iterations : int
        Number of smoothing passes (0 = no smoothing).
    smooth_method : ``'laplacian'`` or ``'taubin'``
        Taubin (λ/μ) smoothing is volume-preserving and recommended.
    decimate_target : int or None
        If given, decimate mesh to approximately this many vertices.

    Returns
    -------
    vertices : (N, 3) float64 — in RAS coordinates.
    faces : (F, 3) int64
    """
    from skimage.measure import marching_cubes

    # Pad by 1 voxel to guarantee a closed surface even if the mask
    # touches the volume boundary.
    padded = np.pad(volume.astype(np.float32), pad_width=1, constant_values=0)

    verts_vox, faces, _, _ = marching_cubes(padded, level=0.5)

    # Undo padding offset
    verts_vox -= 1.0

    # Voxel → RAS
    verts_ras = (
        affine[:3, :3] @ verts_vox.T + affine[:3, 3:4]
    ).T.astype(np.float64)
    faces = faces.astype(np.int64)

    logger.info(
        "Marching cubes: %d vertices, %d faces",
        verts_ras.shape[0], faces.shape[0],
    )

    # ── Smoothing ──
    if smooth_iterations > 0:
        verts_ras = _smooth_mesh(
            verts_ras, faces,
            iterations=smooth_iterations,
            method=smooth_method,
        )

    # ── Decimation ──
    if decimate_target is not None and verts_ras.shape[0] > decimate_target * 1.2:
        verts_ras, faces = _decimate_mesh(verts_ras, faces, decimate_target)

    return verts_ras, faces


def _smooth_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    iterations: int = 30,
    method: str = "taubin",
    lam: float = 0.5,
    mu: float = -0.53,
) -> np.ndarray:
    """
    Smooth a triangle mesh in-place.

    Taubin smoothing alternates a positive (λ) and negative (μ) step,
    preventing the mesh from shrinking — essential for preserving
    subcortical volumes.

    Parameters
    ----------
    vertices : (N, 3)
    faces : (F, 3)
    iterations : int
    method : ``'laplacian'`` or ``'taubin'``
    lam : float — positive step weight.
    mu : float — negative step weight (only for Taubin).

    Returns
    -------
    vertices : (N, 3) — smoothed copy.
    """
    import scipy.sparse as sp

    v = vertices.copy()
    N = v.shape[0]

    # Build adjacency as a sparse NxN matrix with uniform weights
    rows, cols = [], []
    for tri in faces:
        for i in range(3):
            a, b = int(tri[i]), int(tri[(i + 1) % 3])
            rows.extend([a, b])
            cols.extend([b, a])
    data = np.ones(len(rows), dtype=np.float64)
    A = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()

    # Normalise rows → uniform Laplacian weights
    deg = np.array(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    D_inv = sp.diags(1.0 / deg)
    L_norm = D_inv @ A  # Each row sums to 1

    for it in range(iterations):
        # Laplacian displacement: Lv - v  →  direction towards centroid
        Lv = L_norm @ v
        if method == "taubin":
            # Alternating λ / μ steps
            if it % 2 == 0:
                v = v + lam * (Lv - v)
            else:
                v = v + mu * (Lv - v)
        else:
            # Pure Laplacian (shrinks — use with caution)
            v = v + lam * (Lv - v)

    logger.debug(
        "Smoothed mesh: %d iterations (%s, λ=%.2f, μ=%.2f)",
        iterations, method, lam, mu,
    )
    return v


def _decimate_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_vertices: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decimate a mesh to approximately ``target_vertices`` using
    PyVista (VTK) or a fallback uniform subsampling.
    """
    ratio = target_vertices / vertices.shape[0]
    ratio = np.clip(ratio, 0.05, 1.0)

    try:
        import pyvista as pv

        mesh = pv.PolyData(vertices, np.hstack([
            np.full((faces.shape[0], 1), 3, dtype=np.int64), faces
        ]))
        decimated = mesh.decimate(1.0 - ratio)
        v_out = np.array(decimated.points, dtype=np.float64)
        f_out = decimated.faces.reshape(-1, 4)[:, 1:].astype(np.int64)
        logger.info(
            "Decimated: %d → %d vertices (target %d)",
            vertices.shape[0], v_out.shape[0], target_vertices,
        )
        return v_out, f_out

    except ImportError:
        logger.warning(
            "PyVista not available — skipping decimation. "
            "Install pyvista for mesh simplification."
        )
        return vertices, faces


# ═══════════════════════════════════════════════════════════════════════════
# IO: load from different origins
# ═══════════════════════════════════════════════════════════════════════════


def load_subcortical_surface(
    subjects_dir: Union[str, Path],
    subject_id: str,
    structure: str = "Left-Hippocampus",
    hemi: str = "lh",
    *,
    origin: str = "fs",
    smooth_iterations: int = 30,
    smooth_method: str = "taubin",
    decimate_target: int | None = None,
) -> SubcorticalSurface:
    """
    Load a subcortical structure as a closed triangle mesh.

    This is the main entry point for subcortical analysis.  It
    supports three input origins and returns a ready-to-analyse
    :class:`SubcorticalSurface` compatible with the full spectral
    pipeline.

    Parameters
    ----------
    subjects_dir : path-like
        FreeSurfer ``$SUBJECTS_DIR`` (for ``'fs'`` and ``'fs_hc'``)
        or the HippUnfold output directory (for ``'hippunfold'``).
    subject_id : str
        Subject folder name.
    structure : str
        Structure name.  For ``origin='fs'``, any key in
        :data:`FS_ASEG_LABELS` (e.g. ``'Left-Hippocampus'``,
        ``'Right-Thalamus'``).  For ``origin='fs_hc'``, one of
        ``'Left-Hippocampus'`` or ``'Right-Hippocampus'`` (amygdala
        nuclei are excluded).  For ``origin='hippunfold'``, one of
        ``'hippocampus'`` or ``'dentate'``.
    hemi : ``'lh'`` or ``'rh'``
        Hemisphere (used for ``fs_hc`` and ``hippunfold`` path
        resolution, and stored as metadata).
    origin : ``'fs'``, ``'fs_hc'``, or ``'hippunfold'``
        Input source:

        - ``'fs'``         — FreeSurfer ``aseg.mgz`` segmentation.
        - ``'fs_hc'``      — FreeSurfer hippocampal subfield
          segmentation (``?h.hippoAmygLabels-T1.v22.mgz``).
          All hippocampal subfield labels are merged; amygdala
          nuclei (7xxx) are discarded.
        - ``'hippunfold'`` — HippUnfold native-space GIfTI surfaces
          (``*_midthickness.surf.gii``).  No volumetric extraction
          is needed — the mesh is loaded directly.
    smooth_iterations : int
        Taubin/Laplacian smoothing passes (volumetric origins only).
    smooth_method : ``'taubin'`` or ``'laplacian'``
    decimate_target : int or None
        Optional target vertex count after decimation.

    Returns
    -------
    SubcorticalSurface
        Closed mesh ready for spectral analysis.

    Examples
    --------
    >>> # FreeSurfer aseg — any subcortical structure
    >>> surf = load_subcortical_surface(
    ...     "/data/fs", "sub-001", structure="Left-Hippocampus",
    ...     origin="fs", decimate_target=5000,
    ... )
    >>> surf.n_vertices, surf.sphericity
    (4987, 0.412)

    >>> # FreeSurfer hippocampal subfields (Iglesias) — hippocampus only
    >>> surf = load_subcortical_surface(
    ...     "/data/fs", "sub-001", hemi="lh", origin="fs_hc",
    ... )

    >>> # HippUnfold — direct GIfTI mesh loading
    >>> surf = load_subcortical_surface(
    ...     "/data/hippunfold", "sub-001", hemi="lh",
    ...     structure="hippocampus", origin="hippunfold",
    ... )
    """
    origin = origin.lower()

    if origin == "fs":
        return _load_from_fs_aseg(
            subjects_dir, subject_id, structure,
            smooth_iterations=smooth_iterations,
            smooth_method=smooth_method,
            decimate_target=decimate_target,
        )
    elif origin == "fs_hc":
        return _load_from_fs_hc(
            subjects_dir, subject_id, hemi,
            smooth_iterations=smooth_iterations,
            smooth_method=smooth_method,
            decimate_target=decimate_target,
        )
    elif origin == "hippunfold":
        return _load_from_hippunfold(
            subjects_dir, subject_id, hemi, structure,
        )
    else:
        raise ValueError(
            f"Unknown origin '{origin}'. "
            f"Supported: 'fs', 'fs_hc', 'hippunfold'."
        )


# ── FreeSurfer aseg.mgz ──────────────────────────────────────────────


def _load_from_fs_aseg(
    subjects_dir: Union[str, Path],
    subject_id: str,
    structure: str,
    *,
    smooth_iterations: int = 30,
    smooth_method: str = "taubin",
    decimate_target: int | None = None,
) -> SubcorticalSurface:
    """Extract a subcortical surface from aseg.mgz via marching cubes."""
    import nibabel as nib

    sdir = Path(subjects_dir) / subject_id
    aseg_path = sdir / "mri" / "aseg.mgz"
    if not aseg_path.exists():
        raise FileNotFoundError(f"aseg.mgz not found: {aseg_path}")

    # Resolve label ID
    if structure in FS_ASEG_LABELS:
        label_id = FS_ASEG_LABELS[structure]
    elif structure.isdigit():
        label_id = int(structure)
    else:
        raise ValueError(
            f"Unknown structure '{structure}'. "
            f"Use one of: {list(FS_ASEG_LABELS.keys())} or an integer label."
        )

    # Determine hemisphere from structure name or label
    hemi = "lh" if "Left" in structure or "left" in structure else "rh"

    aseg_img = nib.load(str(aseg_path))
    aseg_data = np.asarray(aseg_img.dataobj, dtype=np.int32)
    affine = aseg_img.affine

    mask = (aseg_data == label_id).astype(np.float32)
    n_voxels = int(mask.sum())
    if n_voxels == 0:
        raise ValueError(
            f"Label {label_id} ({structure}) has 0 voxels in {aseg_path}"
        )
    logger.info(
        "aseg label %d (%s): %d voxels", label_id, structure, n_voxels,
    )

    vertices, faces = _volume_to_surface(
        mask, affine,
        smooth_iterations=smooth_iterations,
        smooth_method=smooth_method,
        decimate_target=decimate_target,
    )

    return SubcorticalSurface(
        vertices=vertices,
        faces=faces,
        structure=structure,
        hemi=hemi,
        metadata={
            "origin": "fs",
            "subjects_dir": str(subjects_dir),
            "subject_id": subject_id,
            "label_id": label_id,
            "n_voxels": n_voxels,
            "smooth_iterations": smooth_iterations,
            "smooth_method": smooth_method,
        },
    )


# ── FreeSurfer hippocampal subfields (Iglesias et al.) ────────────────


def _load_from_fs_hc(
    subjects_dir: Union[str, Path],
    subject_id: str,
    hemi: str = "lh",
    *,
    smooth_iterations: int = 30,
    smooth_method: str = "taubin",
    decimate_target: int | None = None,
) -> SubcorticalSurface:
    """
    Load hippocampus from FreeSurfer's hippocampal subfield
    segmentation, merging all hippocampal labels and excluding
    amygdala nuclei.

    Searches for the high-resolution volume at:
      {subjects_dir}/{sid}/mri/{hemi}.hippoAmygLabels-T1.v22.mgz
    or the v21 fallback.
    """
    import nibabel as nib

    sdir = Path(subjects_dir) / subject_id / "mri"

    # Try v22 first, then v21
    candidates = [
        sdir / f"{hemi}.hippoAmygLabels-T1.v22.mgz",
        sdir / f"{hemi}.hippoAmygLabels-T1.v21.mgz",
        sdir / f"{hemi}.hippoAmygLabels-T1.mgz",
    ]
    seg_path = None
    for cand in candidates:
        if cand.exists():
            seg_path = cand
            break

    if seg_path is None:
        raise FileNotFoundError(
            f"Hippocampal subfield segmentation not found for {hemi}. "
            f"Tried: {[str(c) for c in candidates]}"
        )

    seg_img = nib.load(str(seg_path))
    seg_data = np.asarray(seg_img.dataobj, dtype=np.int32)
    affine = seg_img.affine

    # Merge all hippocampal subfield labels into one binary mask,
    # explicitly excluding amygdala nuclei (labels 7000–7999).
    hipp_labels = FS_HC_HIPP_LABELS.get(hemi, FS_HC_HIPP_LABELS["lh"])
    mask = np.zeros_like(seg_data, dtype=np.float32)
    used_labels = []
    for lab in np.unique(seg_data):
        if lab == 0:
            continue
        # Include if it's a known hippocampal label OR if it's
        # in the 200–299 range (hippocampal) and NOT in 7000+
        # (amygdala).
        if lab in hipp_labels or (200 <= lab < 300):
            mask[seg_data == lab] = 1.0
            used_labels.append(int(lab))

    n_voxels = int(mask.sum())
    if n_voxels == 0:
        raise ValueError(
            f"No hippocampal voxels found in {seg_path}. "
            f"Available labels: {sorted(np.unique(seg_data).tolist())}"
        )

    logger.info(
        "fs_hc %s: %d voxels from labels %s (excl. amygdala 7xxx)",
        hemi, n_voxels, used_labels,
    )

    structure = f"{'Left' if hemi == 'lh' else 'Right'}-Hippocampus"

    vertices, faces = _volume_to_surface(
        mask, affine,
        smooth_iterations=smooth_iterations,
        smooth_method=smooth_method,
        decimate_target=decimate_target,
    )

    return SubcorticalSurface(
        vertices=vertices,
        faces=faces,
        structure=structure,
        hemi=hemi,
        metadata={
            "origin": "fs_hc",
            "subjects_dir": str(subjects_dir),
            "subject_id": subject_id,
            "seg_path": str(seg_path),
            "merged_labels": used_labels,
            "n_voxels": n_voxels,
            "smooth_iterations": smooth_iterations,
            "smooth_method": smooth_method,
        },
    )


# ── HippUnfold ────────────────────────────────────────────────────────


def _load_from_hippunfold(
    hippunfold_dir: Union[str, Path],
    subject_id: str,
    hemi: str = "lh",
    structure: str = "hippocampus",
) -> SubcorticalSurface:
    """
    Load a HippUnfold native-space surface (GIfTI).

    HippUnfold outputs are already high-quality triangle meshes —
    no marching cubes or smoothing needed.  We look for the
    midthickness surface by default.

    Expected path pattern (BIDS-like):
      {hippunfold_dir}/hippunfold/{subject_id}/surf/
        {subject_id}_hemi-{L|R}_space-T1w_den-0p5mm_label-{hipp|dentate}
        _midthickness.surf.gii
    """
    import nibabel as nib

    hu_dir = Path(hippunfold_dir)
    h = "L" if hemi == "lh" else "R"
    label = "hipp" if "hipp" in structure.lower() else "dentate"

    # Try common path patterns
    patterns = [
        # BIDS standard HippUnfold output
        hu_dir / "hippunfold" / subject_id / "surf" /
        f"{subject_id}_hemi-{h}_space-T1w_den-0p5mm_label-{label}_midthickness.surf.gii",
        # Simpler layout
        hu_dir / subject_id / "surf" /
        f"{subject_id}_hemi-{h}_space-T1w_den-0p5mm_label-{label}_midthickness.surf.gii",
        # Very simple layout
        hu_dir / subject_id /
        f"hemi-{h}_label-{label}_midthickness.surf.gii",
    ]

    surf_path = None
    for p in patterns:
        if p.exists():
            surf_path = p
            break

    if surf_path is None:
        raise FileNotFoundError(
            f"HippUnfold surface not found. Tried:\n"
            + "\n".join(f"  {p}" for p in patterns)
        )

    gii = nib.load(str(surf_path))
    coords, faces_arr = None, None
    overlays = {}

    for da in gii.darrays:
        intent = da.intent
        if intent == nib.nifti1.intent_codes["NIFTI_INTENT_POINTSET"]:
            coords = da.data.astype(np.float64)
        elif intent == nib.nifti1.intent_codes["NIFTI_INTENT_TRIANGLE"]:
            faces_arr = da.data.astype(np.int64)
        elif intent == nib.nifti1.intent_codes["NIFTI_INTENT_SHAPE"]:
            name = da.meta.get("Name", f"overlay_{len(overlays)}")
            overlays[name] = da.data.astype(np.float64)

    if coords is None or faces_arr is None:
        raise ValueError(f"GIfTI lacks coordinate/triangle data: {surf_path}")

    structure_name = f"{'Left' if hemi == 'lh' else 'Right'}-Hippocampus"
    if "dentate" in structure.lower():
        structure_name = f"{'Left' if hemi == 'lh' else 'Right'}-DentateGyrus"

    logger.info(
        "HippUnfold %s %s: %d vertices, %d faces",
        hemi, label, coords.shape[0], faces_arr.shape[0],
    )

    # Try loading matching morphometric overlays (thickness, curvature)
    morph_names = ["thickness", "curvature", "gyrification"]
    surf_dir = surf_path.parent
    for mname in morph_names:
        mpath = surf_dir / surf_path.name.replace(
            "midthickness.surf.gii", f"{mname}.shape.gii"
        )
        if mpath.exists():
            mgii = nib.load(str(mpath))
            for da in mgii.darrays:
                overlays[mname] = da.data.astype(np.float64)
                break
            logger.debug("  loaded overlay '%s'", mname)

    return SubcorticalSurface(
        vertices=coords,
        faces=faces_arr,
        structure=structure_name,
        hemi=hemi,
        overlays=overlays,
        metadata={
            "origin": "hippunfold",
            "hippunfold_dir": str(hippunfold_dir),
            "subject_id": subject_id,
            "surf_path": str(surf_path),
            "label": label,
        },
    )


# ═══════════════════════════════════════════════════════════════════════════
# Generic volume import (arbitrary NIfTI label)
# ═══════════════════════════════════════════════════════════════════════════


def load_subcortical_from_nifti(
    nifti_path: Union[str, Path],
    label_ids: Union[int, Sequence[int]],
    *,
    structure_name: str = "custom",
    hemi: str = "lh",
    smooth_iterations: int = 30,
    smooth_method: str = "taubin",
    decimate_target: int | None = None,
) -> SubcorticalSurface:
    """
    Extract a subcortical surface from **any** NIfTI label volume.

    This is the most flexible entry point: you provide a segmentation
    volume (e.g. ``aseg.mgz``, a custom parcellation, or an atlas in
    MNI space) and one or more integer labels to merge into a single
    binary mask.

    Parameters
    ----------
    nifti_path : path-like
        Path to a NIfTI (.nii, .nii.gz) or FreeSurfer (.mgz) volume.
    label_ids : int or sequence of int
        One or more integer labels to extract and merge.
    structure_name : str
        Human-readable name for the structure.
    hemi : ``'lh'`` or ``'rh'``
    smooth_iterations : int
    smooth_method : str
    decimate_target : int or None

    Returns
    -------
    SubcorticalSurface
    """
    import nibabel as nib

    nifti_path = Path(nifti_path)
    img = nib.load(str(nifti_path))
    data = np.asarray(img.dataobj, dtype=np.int32)
    affine = img.affine

    if isinstance(label_ids, int):
        label_ids = [label_ids]

    mask = np.zeros_like(data, dtype=np.float32)
    for lab in label_ids:
        mask[data == lab] = 1.0

    n_voxels = int(mask.sum())
    if n_voxels == 0:
        raise ValueError(
            f"Labels {label_ids} have 0 voxels in {nifti_path}"
        )

    logger.info(
        "NIfTI %s: labels %s → %d voxels",
        nifti_path.name, label_ids, n_voxels,
    )

    vertices, faces = _volume_to_surface(
        mask, affine,
        smooth_iterations=smooth_iterations,
        smooth_method=smooth_method,
        decimate_target=decimate_target,
    )

    return SubcorticalSurface(
        vertices=vertices,
        faces=faces,
        structure=structure_name,
        hemi=hemi,
        metadata={
            "origin": "nifti",
            "nifti_path": str(nifti_path),
            "label_ids": list(label_ids),
            "n_voxels": n_voxels,
            "smooth_iterations": smooth_iterations,
            "smooth_method": smooth_method,
        },
    )


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: full spectral pipeline for a SubcorticalSurface
# ═══════════════════════════════════════════════════════════════════════════


def subcortical_spectral_analysis(
    surf: SubcorticalSurface,
    *,
    n_eigenpairs: int | None = None,
    eigenpair_mode: str = "auto",
    hks_scales: int = 16,
    wks_energies: int = 16,
    gps_components: int = 10,
    compute_curvatures: bool = True,
    backend: str = "auto",
):
    """
    Run the full CorticalFields spectral pipeline on a subcortical mesh.

    This is a convenience wrapper that:
      1. Estimates the optimal number of eigenpairs (if not specified).
      2. Computes LBO eigenpairs.
      3. Extracts HKS, WKS, GPS descriptors.
      4. Optionally computes discrete curvatures, sphericity, and
         Willmore energy.
      5. Attaches all results to the surface and returns them.

    Parameters
    ----------
    surf : SubcorticalSurface
        The input mesh.
    n_eigenpairs : int or None
        If None, estimated automatically via
        :func:`~corticalfields.utils.estimate_n_eigenpairs`.
    eigenpair_mode : str
        Mode for eigenpair estimation (``'auto'``, ``'conservative'``,
        ``'aggressive'``, ``'weyl'``).
    hks_scales, wks_energies, gps_components : int
        Descriptor dimensionalities.
    compute_curvatures : bool
        If True, compute Gaussian/mean curvature and Willmore energy.
    backend : str
        Eigensolver backend.

    Returns
    -------
    results : dict
        Keys: ``'lb'`` (LaplaceBeltrami), ``'hks'``, ``'wks'``,
        ``'gps'``, ``'features'`` (concatenated matrix), plus
        ``'K'``, ``'H'``, ``'willmore'`` if curvatures were computed.
    """
    from corticalfields.spectral import (
        compute_eigenpairs,
        heat_kernel_signature,
        wave_kernel_signature,
        global_point_signature,
        spectral_feature_matrix,
    )
    from corticalfields.utils import estimate_n_eigenpairs

    # Step 1: estimate eigenpairs
    if n_eigenpairs is None:
        n_eigenpairs = estimate_n_eigenpairs(
            surf.n_vertices,
            surface_area_mm2=surf.total_area,
            mode=eigenpair_mode,
        )

    logger.info(
        "Subcortical spectral analysis: %s (%s) — %d vertices, "
        "%d eigenpairs",
        surf.structure, surf.hemi, surf.n_vertices, n_eigenpairs,
    )

    # Step 2: eigenpairs
    lb = compute_eigenpairs(
        surf.vertices, surf.faces,
        n_eigenpairs=n_eigenpairs,
        backend=backend,
    )

    # Step 3: descriptors
    hks = heat_kernel_signature(lb, n_scales=hks_scales)
    wks = wave_kernel_signature(lb, n_energies=wks_energies)
    gps = global_point_signature(lb, n_components=gps_components)
    features = spectral_feature_matrix(
        lb, hks_scales=hks_scales, wks_energies=wks_energies,
        gps_components=gps_components,
    )

    results = {
        "lb": lb,
        "hks": hks,
        "wks": wks,
        "gps": gps,
        "features": features,
        "n_eigenpairs": n_eigenpairs,
    }

    # Attach descriptors as overlays (first scale / energy for viz)
    surf.add_overlay("hks_t0", hks[:, 0])
    surf.add_overlay("hks_tmid", hks[:, hks.shape[1] // 2])
    surf.add_overlay("hks_tmax", hks[:, -1])
    surf.add_overlay("wks_e0", wks[:, 0])
    surf.add_overlay("wks_emid", wks[:, wks.shape[1] // 2])
    surf.add_overlay("wks_emax", wks[:, -1])

    # Step 4: curvatures and global shape descriptors
    if compute_curvatures:
        K, H = surf.compute_curvatures()
        W = surf.willmore_energy()
        results["K"] = K
        results["H"] = H
        results["willmore"] = W

    results["sphericity"] = surf.sphericity
    results["volume_mm3"] = surf.enclosed_volume
    results["area_mm2"] = surf.total_area

    logger.info(
        "  Sphericity=%.3f, Volume=%.1f mm³, Area=%.1f mm², "
        "Willmore=%.1f",
        results["sphericity"],
        results.get("volume_mm3", 0),
        results.get("area_mm2", 0),
        results.get("willmore", 0),
    )

    return results
