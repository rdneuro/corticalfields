"""
Surface I/O and mesh utilities for cortical surfaces.

Handles loading FreeSurfer surfaces (pial, white, inflated), GIfTI files,
and morphometric overlays (thickness, curvature, sulcal depth, etc.).
Provides a unified CorticalSurface object used by all downstream modules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FreeSurfer surface names and their standard overlay files
# ---------------------------------------------------------------------------
FREESURFER_OVERLAYS = {
    "thickness": "{hemi}.thickness",
    "curv": "{hemi}.curv",
    "sulc": "{hemi}.sulc",
    "area": "{hemi}.area",
    "volume": "{hemi}.volume",
    "pial_lgi": "{hemi}.pial_lgi",  # local gyrification index
}

# Hemispheres
LH, RH = "lh", "rh"


@dataclass
class CorticalSurface:
    """
    A cortical triangle mesh with per-vertex scalar overlays.

    This is the central data structure in CorticalFields. Every downstream
    module (spectral decomposition, GP inference, surprise maps) operates
    on a CorticalSurface instance.

    Parameters
    ----------
    vertices : np.ndarray, shape (N, 3)
        Vertex coordinates in RAS space (millimetres).
    faces : np.ndarray, shape (F, 3)
        Triangle connectivity (0-indexed vertex indices).
    hemi : str
        Hemisphere identifier — ``'lh'`` or ``'rh'``.
    overlays : dict[str, np.ndarray]
        Named per-vertex scalar maps. Each value has shape ``(N,)``.
    metadata : dict
        Arbitrary metadata (subject ID, FreeSurfer directory, etc.).

    Notes
    -----
    Vertices and faces are stored as float64 / int64 numpy arrays.
    The mesh is assumed to be a closed, manifold triangle mesh as produced
    by FreeSurfer's ``recon-all``.
    """

    vertices: np.ndarray
    faces: np.ndarray
    hemi: str = "lh"
    overlays: Dict[str, np.ndarray] = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)

    # ---- properties -------------------------------------------------------

    @property
    def n_vertices(self) -> int:
        """Number of mesh vertices."""
        return self.vertices.shape[0]

    @property
    def n_faces(self) -> int:
        """Number of triangular faces."""
        return self.faces.shape[0]

    @property
    def overlay_names(self) -> List[str]:
        """Names of available per-vertex overlays."""
        return list(self.overlays.keys())

    @property
    def vertex_normals(self) -> np.ndarray:
        """
        Compute per-vertex normals via area-weighted face normal averaging.

        Returns
        -------
        normals : np.ndarray, shape (N, 3)
            Unit normals at each vertex.
        """
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        # Face normals (not normalised — magnitude encodes area)
        face_normals = np.cross(v1 - v0, v2 - v0)
        # Accumulate onto vertices
        normals = np.zeros_like(self.vertices)
        for i in range(3):
            np.add.at(normals, self.faces[:, i], face_normals)
        # Normalise to unit length
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return normals / norms

    @property
    def face_areas(self) -> np.ndarray:
        """
        Area of each triangular face.

        Returns
        -------
        areas : np.ndarray, shape (F,)
        """
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

    @property
    def total_area(self) -> float:
        """Total surface area in mm²."""
        return float(self.face_areas.sum())

    # ---- overlay management -----------------------------------------------

    def add_overlay(self, name: str, data: np.ndarray) -> None:
        """Attach a per-vertex scalar overlay to the surface."""
        if data.shape[0] != self.n_vertices:
            raise ValueError(
                f"Overlay '{name}' has {data.shape[0]} values but surface "
                f"has {self.n_vertices} vertices."
            )
        self.overlays[name] = np.asarray(data, dtype=np.float64)

    def get_overlay(self, name: str) -> np.ndarray:
        """Retrieve a per-vertex overlay by name."""
        if name not in self.overlays:
            available = ", ".join(self.overlay_names) or "(none)"
            raise KeyError(
                f"Overlay '{name}' not found. Available: {available}"
            )
        return self.overlays[name]

    # ---- mesh topology helpers --------------------------------------------

    def vertex_adjacency(self) -> List[np.ndarray]:
        """
        Build an adjacency list: for each vertex, a sorted array of neighbour
        indices connected by a mesh edge.

        Returns
        -------
        adj : list[np.ndarray]
            ``adj[i]`` is the 1D array of vertex indices adjacent to vertex i.
        """
        from collections import defaultdict

        neighbours: Dict[int, set] = defaultdict(set)
        for f in self.faces:
            for i in range(3):
                a, b = int(f[i]), int(f[(i + 1) % 3])
                neighbours[a].add(b)
                neighbours[b].add(a)
        return [
            np.sort(np.array(list(neighbours.get(i, set())), dtype=np.int64))
            for i in range(self.n_vertices)
        ]

    def edge_list(self) -> np.ndarray:
        """
        Unique undirected edges as an (E, 2) int64 array, sorted per row.
        """
        edges = set()
        for f in self.faces:
            for i in range(3):
                a, b = int(f[i]), int(f[(i + 1) % 3])
                edges.add((min(a, b), max(a, b)))
        return np.array(sorted(edges), dtype=np.int64)

    # ---- serialisation ----------------------------------------------------

    def to_gifti(self, path: Union[str, Path]) -> None:
        """
        Save as a GIfTI file with coordinate and topology arrays, plus
        overlays as extra data arrays.
        """
        path = Path(path)
        coord_darray = nib.gifti.GiftiDataArray(
            data=self.vertices.astype(np.float32),
            intent="NIFTI_INTENT_POINTSET",
            datatype="NIFTI_TYPE_FLOAT32",
        )
        tri_darray = nib.gifti.GiftiDataArray(
            data=self.faces.astype(np.int32),
            intent="NIFTI_INTENT_TRIANGLE",
            datatype="NIFTI_TYPE_INT32",
        )
        darrays = [coord_darray, tri_darray]
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
        logger.info("Saved GIfTI surface to %s", path)


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def load_freesurfer_surface(
    subjects_dir: Union[str, Path],
    subject_id: str,
    hemi: str = "lh",
    surface: str = "midthickness",
    overlays: Optional[List[str]] = None,
) -> CorticalSurface:
    """
    Load a FreeSurfer surface with morphometric overlays.

    Parameters
    ----------
    subjects_dir : path-like
        FreeSurfer ``$SUBJECTS_DIR``.
    subject_id : str
        Subject folder name (e.g. ``'sub-001'``).
    hemi : ``'lh'`` or ``'rh'``
        Hemisphere.
    surface : str
        Surface name: ``'midthickness'`` (default, recommended for LBO/HKS),
        ``'pial'``, ``'white'``, ``'inflated'``, etc.
        If ``'midthickness'`` is not found, attempts ``'graymid'``, then
        auto-generates from white+pial average (HCP convention).
    overlays : list[str] or None
        Overlay names to load (keys of ``FREESURFER_OVERLAYS``).
        If *None*, loads all available overlays.

    Returns
    -------
    CorticalSurface
        Mesh with attached overlays.

    Examples
    --------
    >>> surf = load_freesurfer_surface(
    ...     "/data/freesurfer", "sub-001", hemi="lh",
    ...     surface="pial", overlays=["thickness", "curv", "sulc"]
    ... )
    >>> surf.n_vertices
    163842
    """
    base = Path(subjects_dir) / subject_id / "surf"
    surf_path = base / f"{hemi}.{surface}"
    if not surf_path.exists():
        # Midthickness fallback chain
        if surface in ("midthickness", "graymid"):
            alt_name = "graymid" if surface == "midthickness" else "midthickness"
            alt_path = base / f"{hemi}.{alt_name}"
            if alt_path.exists():
                surf_path = alt_path
                surface = alt_name
                logger.info("Using %s.%s (fallback)", hemi, alt_name)
            else:
                # Auto-generate from white + pial average (HCP convention)
                white_path = base / f"{hemi}.white"
                pial_path = base / f"{hemi}.pial"
                if white_path.exists() and pial_path.exists():
                    w_coords, w_faces = nib.freesurfer.read_geometry(str(white_path))
                    p_coords, _ = nib.freesurfer.read_geometry(str(pial_path))
                    mid_coords = (w_coords + p_coords) / 2.0
                    nib.freesurfer.write_geometry(str(surf_path), mid_coords, w_faces)
                    logger.info(
                        "Auto-generated %s.midthickness from white+pial average",
                        hemi,
                    )
                else:
                    raise FileNotFoundError(
                        f"Surface not found: {surf_path}. "
                        f"Cannot auto-generate: need both {hemi}.white and {hemi}.pial"
                    )
        else:
            raise FileNotFoundError(f"Surface not found: {surf_path}")

    # nibabel reads FreeSurfer binary surfaces
    coords, faces = nib.freesurfer.read_geometry(str(surf_path))
    logger.info(
        "Loaded %s.%s: %d vertices, %d faces",
        hemi, surface, coords.shape[0], faces.shape[0],
    )

    cs = CorticalSurface(
        vertices=coords.astype(np.float64),
        faces=faces.astype(np.int64),
        hemi=hemi,
        metadata={
            "subjects_dir": str(subjects_dir),
            "subject_id": subject_id,
            "surface": surface,
        },
    )

    # Load overlays
    if overlays is None:
        overlays = list(FREESURFER_OVERLAYS.keys())

    for name in overlays:
        template = FREESURFER_OVERLAYS.get(name)
        if template is None:
            logger.warning("Unknown overlay '%s', skipping.", name)
            continue
        opath = base / template.format(hemi=hemi)
        if opath.exists():
            data = nib.freesurfer.read_morph_data(str(opath))
            cs.add_overlay(name, data)
            logger.debug("  loaded overlay '%s' (%d values)", name, len(data))
        else:
            logger.debug("  overlay '%s' not found at %s", name, opath)

    return cs


def load_gifti_surface(
    surf_path: Union[str, Path],
    hemi: str = "lh",
) -> CorticalSurface:
    """
    Load a GIfTI surface file (.surf.gii or .gii).

    Parameters
    ----------
    surf_path : path-like
        Path to the GIfTI file.
    hemi : str
        Hemisphere tag for metadata.

    Returns
    -------
    CorticalSurface
    """
    gii = nib.load(str(surf_path))
    coords = None
    faces = None
    overlays = {}

    for da in gii.darrays:
        intent = da.intent
        if intent == nib.nifti1.intent_codes["NIFTI_INTENT_POINTSET"]:
            coords = da.data.astype(np.float64)
        elif intent == nib.nifti1.intent_codes["NIFTI_INTENT_TRIANGLE"]:
            faces = da.data.astype(np.int64)
        elif intent == nib.nifti1.intent_codes["NIFTI_INTENT_SHAPE"]:
            name = da.meta.get("Name", f"overlay_{len(overlays)}")
            overlays[name] = da.data.astype(np.float64)

    if coords is None or faces is None:
        raise ValueError(f"GIfTI file lacks coordinate/triangle arrays: {surf_path}")

    return CorticalSurface(
        vertices=coords, faces=faces, hemi=hemi, overlays=overlays,
    )


def load_annot(
    subjects_dir: Union[str, Path],
    subject_id: str,
    hemi: str = "lh",
    annot: str = "aparc",
) -> Tuple[np.ndarray, List[str]]:
    """
    Load a FreeSurfer annotation (parcellation) file.

    Returns
    -------
    labels : np.ndarray, shape (N,)
        Integer label per vertex.
    names : list[str]
        Region names corresponding to each label value.
    """
    apath = (
        Path(subjects_dir) / subject_id / "label" / f"{hemi}.{annot}.annot"
    )
    labels, ctab, names = nib.freesurfer.read_annot(str(apath))
    names = [n.decode() if isinstance(n, bytes) else n for n in names]
    return labels, names
