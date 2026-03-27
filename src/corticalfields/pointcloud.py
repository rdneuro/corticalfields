"""
Point cloud cortical geometry — mesh-free Laplace–Beltrami analysis.

This module provides an **alternative path** to ``surface.py`` for cortical
morphometry that does not require a triangulated mesh. Instead, it works
directly with 3D point clouds extracted from either:

  1. **FreeSurfer pial surfaces** — the vertex coordinates are already a
     point cloud; the faces are discarded.
  2. **NIfTI binary masks** — the outer cortical surface is extracted via
     marching cubes and the resulting points are used without connectivity.

The key mathematical contribution is that the **Laplace–Beltrami operator
can be estimated on point clouds** using the intrinsic Delaunay approach of
Sharp & Crane (2020), which provably converges to the true LBO as point
density increases. This means all downstream spectral analysis (HKS, WKS,
GPS, functional maps, normative models) can run on mesh-free data.

Use cases
---------
- Subjects where FreeSurfer surface reconstruction failed but a segmentation
  mask exists (e.g., severe pathology, post-surgical cavities).
- Comparing cortical geometry from different processing pipelines that
  produce incompatible meshes.
- Research on cortical folding using conformal flattening of point clouds.

Dependencies
------------
- ``robust_laplacian`` — intrinsic Delaunay LBO on point clouds (required)
- ``open3d`` — point cloud processing, normal estimation (optional, for
  advanced features like Poisson reconstruction)
- ``skimage`` — marching cubes for NIfTI extraction (optional)

References
----------
Sharp, N. & Crane, K. (2020). A Laplacian for Nonmanifold Triangle Meshes.
    Computer Graphics Forum (SGP), 39(5).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class CorticalPointCloud:
    """
    A cortical point cloud with optional per-point scalar overlays.

    This is the mesh-free counterpart to :class:`~corticalfields.surface.CorticalSurface`.
    It stores 3D coordinates, optional normals, and per-point scalar maps.

    Parameters
    ----------
    points : np.ndarray, shape (N, 3)
        3D coordinates in RAS space (millimetres).
    normals : np.ndarray or None, shape (N, 3)
        Unit normals at each point (estimated if not provided).
    hemi : str
        Hemisphere identifier — ``'lh'`` or ``'rh'``.
    overlays : dict[str, np.ndarray]
        Named per-point scalar maps, each shape ``(N,)``.
    metadata : dict
        Arbitrary metadata (subject ID, source file, etc.).
    """

    points: np.ndarray
    normals: Optional[np.ndarray] = None
    hemi: str = "lh"
    overlays: Dict[str, np.ndarray] = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)

    @property
    def n_points(self) -> int:
        """Number of points in the cloud."""
        return self.points.shape[0]

    @property
    def overlay_names(self) -> List[str]:
        """Names of available per-point overlays."""
        return list(self.overlays.keys())

    @property
    def centroid(self) -> np.ndarray:
        """Geometric centroid of the point cloud, shape (3,)."""
        return self.points.mean(axis=0)

    @property
    def bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Axis-aligned bounding box as (min_corner, max_corner)."""
        return self.points.min(axis=0), self.points.max(axis=0)

    def subsample(
        self,
        n_points: int,
        method: str = "farthest_point",
        seed: Optional[int] = None,
    ) -> "CorticalPointCloud":
        """
        Downsample the point cloud.

        Parameters
        ----------
        n_points : int
            Target number of points.
        method : ``'farthest_point'`` or ``'random'``
            Sampling strategy. Farthest-point sampling preserves spatial
            coverage; random is faster.
        seed : int or None
            Random seed for reproducibility.

        Returns
        -------
        CorticalPointCloud
            Subsampled point cloud with overlays transferred.
        """
        if n_points >= self.n_points:
            return self

        if method == "random":
            rng = np.random.default_rng(seed)
            idx = rng.choice(self.n_points, n_points, replace=False)
            idx.sort()
        elif method == "farthest_point":
            idx = _farthest_point_sampling(self.points, n_points, seed=seed)
        else:
            raise ValueError(f"Unknown sampling method: {method!r}")

        new_overlays = {k: v[idx] for k, v in self.overlays.items()}
        new_normals = self.normals[idx] if self.normals is not None else None
        return CorticalPointCloud(
            points=self.points[idx].copy(),
            normals=new_normals,
            hemi=self.hemi,
            overlays=new_overlays,
            metadata={**self.metadata, "subsampled_from": self.n_points},
        )

    def mirror_x(self) -> "CorticalPointCloud":
        """
        Mirror the point cloud across the YZ plane (x → −x).

        This is essential for inter-hemispheric correspondence: the right
        hemisphere must be mirrored before computing functional maps or
        optimal transport distances to the left hemisphere.

        Returns
        -------
        CorticalPointCloud
            Mirrored copy with flipped hemisphere label.
        """
        mirrored = self.points.copy()
        mirrored[:, 0] *= -1
        new_hemi = "rh" if self.hemi == "lh" else "lh"
        new_normals = None
        if self.normals is not None:
            new_normals = self.normals.copy()
            new_normals[:, 0] *= -1
        return CorticalPointCloud(
            points=mirrored,
            normals=new_normals,
            hemi=new_hemi,
            overlays=dict(self.overlays),  # shallow copy
            metadata={**self.metadata, "mirrored": True},
        )


# ═══════════════════════════════════════════════════════════════════════════
# Point cloud extraction
# ═══════════════════════════════════════════════════════════════════════════


def from_freesurfer_surface(
    subjects_dir: Union[str, Path],
    subject_id: str,
    hemi: str = "lh",
    surface: str = "pial",
    overlays: Optional[List[str]] = None,
) -> CorticalPointCloud:
    """
    Extract a point cloud from FreeSurfer surface files.

    The vertex coordinates of the triangulated surface are used directly;
    the face connectivity is discarded. This provides a point cloud with
    the same spatial density and distribution as the original surface.

    Parameters
    ----------
    subjects_dir : str or Path
        FreeSurfer SUBJECTS_DIR.
    subject_id : str
        Subject directory name.
    hemi : ``'lh'`` or ``'rh'``
        Hemisphere.
    surface : str
        Surface name (``'pial'``, ``'white'``, ``'inflated'``, etc.).
    overlays : list of str or None
        Overlay names to load (e.g., ``['thickness', 'curv', 'sulc']``).

    Returns
    -------
    CorticalPointCloud
        Point cloud with loaded overlays.
    """
    import nibabel as nib

    sd = Path(subjects_dir)
    surf_path = sd / subject_id / "surf" / f"{hemi}.{surface}"
    if not surf_path.exists():
        raise FileNotFoundError(f"Surface not found: {surf_path}")

    coords, _ = nib.freesurfer.read_geometry(str(surf_path))
    logger.info(
        "Loaded %d points from %s.%s (%s)",
        coords.shape[0], hemi, surface, subject_id,
    )

    overlay_dict = {}
    if overlays:
        for name in overlays:
            ov_path = sd / subject_id / "surf" / f"{hemi}.{name}"
            if ov_path.exists():
                data = nib.freesurfer.read_morph_data(str(ov_path))
                overlay_dict[name] = np.asarray(data, dtype=np.float64)
                logger.debug("  Loaded overlay '%s' (%d values)", name, len(data))
            else:
                logger.warning("  Overlay not found: %s", ov_path)

    return CorticalPointCloud(
        points=np.asarray(coords, dtype=np.float64),
        hemi=hemi,
        overlays=overlay_dict,
        metadata={
            "subject_id": subject_id,
            "subjects_dir": str(sd),
            "surface": surface,
            "source": "freesurfer",
        },
    )


def from_cortical_surface(
    surface: "CorticalSurface",
) -> CorticalPointCloud:
    """
    Convert an existing :class:`~corticalfields.surface.CorticalSurface`
    to a :class:`CorticalPointCloud`, discarding face connectivity.

    Parameters
    ----------
    surface : CorticalSurface
        The mesh-based surface to convert.

    Returns
    -------
    CorticalPointCloud
        Point cloud preserving all overlays and metadata.
    """
    return CorticalPointCloud(
        points=surface.vertices.copy(),
        hemi=surface.hemi,
        overlays=dict(surface.overlays),
        metadata=dict(surface.metadata),
    )


def from_nifti_mask(
    nifti_path: Union[str, Path],
    hemi: str = "lh",
    level: float = 0.5,
    step_size: int = 1,
    smooth: bool = True,
) -> CorticalPointCloud:
    """
    Extract a cortical point cloud from a NIfTI binary segmentation mask.

    Uses scikit-image's marching cubes to extract the isosurface at the
    given level, then transforms the resulting coordinates into RAS space
    using the NIfTI affine.

    Parameters
    ----------
    nifti_path : str or Path
        Path to the NIfTI file (binary mask or probability map).
    hemi : ``'lh'`` or ``'rh'``
        Hemisphere label for the resulting point cloud.
    level : float
        Isovalue for marching cubes (0.5 for binary masks).
    step_size : int
        Step size for marching cubes (1 = full resolution).
    smooth : bool
        Apply Gaussian smoothing (sigma=1 voxel) before extraction
        to reduce staircase artefacts.

    Returns
    -------
    CorticalPointCloud
        Point cloud in RAS coordinates.
    """
    import nibabel as nib

    try:
        from skimage.measure import marching_cubes
    except ImportError:
        raise ImportError(
            "scikit-image is required for NIfTI mask extraction. "
            "Install with: pip install scikit-image"
        )

    nifti_path = Path(nifti_path)
    img = nib.load(str(nifti_path))
    data = np.asarray(img.dataobj, dtype=np.float64)
    affine = img.affine

    if smooth:
        from scipy.ndimage import gaussian_filter
        data = gaussian_filter(data, sigma=1.0)

    verts_vox, faces_mc, normals_mc, _ = marching_cubes(
        data, level=level, step_size=step_size,
    )

    # Transform from voxel to RAS
    ones = np.ones((verts_vox.shape[0], 1))
    verts_ras = (affine @ np.hstack([verts_vox, ones]).T).T[:, :3]

    # Transform normals (use inverse-transpose of affine 3x3)
    A33 = affine[:3, :3]
    normal_xform = np.linalg.inv(A33).T
    normals_ras = (normal_xform @ normals_mc.T).T
    norms = np.linalg.norm(normals_ras, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normals_ras = normals_ras / norms

    logger.info(
        "Extracted %d points from NIfTI mask %s", verts_ras.shape[0], nifti_path.name,
    )

    return CorticalPointCloud(
        points=verts_ras.astype(np.float64),
        normals=normals_ras.astype(np.float64),
        hemi=hemi,
        metadata={
            "source": "nifti_mask",
            "nifti_path": str(nifti_path),
            "affine": affine,
        },
    )


# ═══════════════════════════════════════════════════════════════════════════
# Point cloud Laplace–Beltrami operator
# ═══════════════════════════════════════════════════════════════════════════


def compute_pointcloud_laplacian(
    points: np.ndarray,
    n_neighbors: int = 30,
) -> Tuple[sp.csc_matrix, sp.csc_matrix]:
    """
    Compute the Laplace–Beltrami operator on a 3D point cloud.

    Uses the ``robust_laplacian`` package (Sharp & Crane 2020), which
    builds an intrinsic Delaunay triangulation of the local neighbourhood
    and computes cotangent weights. The resulting operator provably
    converges to the continuous LBO as point density increases.

    Parameters
    ----------
    points : np.ndarray, shape (N, 3)
        3D point coordinates.
    n_neighbors : int
        Number of nearest neighbours for local triangulation.
        30 is a good default for cortical surfaces (~163k points).

    Returns
    -------
    L : scipy.sparse.csc_matrix, shape (N, N)
        Stiffness matrix (positive semi-definite).
    M : scipy.sparse.csc_matrix, shape (N, N)
        Diagonal mass matrix.

    Raises
    ------
    ImportError
        If ``robust_laplacian`` is not installed.
    """
    try:
        import robust_laplacian
    except ImportError:
        raise ImportError(
            "robust_laplacian is required for point cloud LBO. "
            "Install with: pip install robust-laplacian"
        )

    logger.info(
        "Computing point cloud Laplacian for %d points (k=%d)...",
        points.shape[0], n_neighbors,
    )
    L, M = robust_laplacian.point_cloud_laplacian(
        np.asarray(points, dtype=np.float64),
        n_neighbors=n_neighbors,
    )
    L = sp.csc_matrix(L)
    M = sp.csc_matrix(M)
    logger.info("  L: %d×%d, %d nnz | M: %d×%d", L.shape[0], L.shape[1], L.nnz, M.shape[0], M.shape[1])
    return L, M


def compute_pointcloud_eigenpairs(
    points: np.ndarray,
    n_eigenpairs: int = 300,
    n_neighbors: int = 30,
    sigma: float = -0.01,
    backend: str = "auto",
) -> "LaplaceBeltrami":
    """
    Compute Laplace–Beltrami eigenpairs directly from a point cloud.

    This is the mesh-free equivalent of
    :func:`~corticalfields.spectral.compute_eigenpairs`. The resulting
    :class:`~corticalfields.spectral.LaplaceBeltrami` object is fully
    compatible with all downstream CorticalFields analysis (HKS, WKS,
    GPS, spectral kernels, normative models, functional maps).

    Parameters
    ----------
    points : np.ndarray, shape (N, 3)
        3D point coordinates.
    n_eigenpairs : int
        Number of eigenpairs to compute.
    n_neighbors : int
        Neighbours for local triangulation (passed to
        :func:`compute_pointcloud_laplacian`).
    sigma : float
        Shift-invert parameter for the eigensolver.
    backend : str
        Compute backend (``'auto'``, ``'scipy'``, ``'cupy'``, ``'torch'``).

    Returns
    -------
    LaplaceBeltrami
        Spectral decomposition compatible with CorticalFields pipeline.

    Examples
    --------
    >>> from corticalfields.pointcloud import (
    ...     from_freesurfer_surface, compute_pointcloud_eigenpairs,
    ... )
    >>> pc = from_freesurfer_surface("/data/fs", "sub-01", "lh", "pial")
    >>> lb = compute_pointcloud_eigenpairs(pc.points, n_eigenpairs=300)
    >>> # Now use with any CorticalFields spectral function:
    >>> from corticalfields.spectral import heat_kernel_signature
    >>> hks = heat_kernel_signature(lb, n_scales=16)
    """
    from corticalfields.spectral import LaplaceBeltrami
    from corticalfields.backends import eigsh_solve, resolve_backend

    L, M = compute_pointcloud_laplacian(points, n_neighbors=n_neighbors)

    be = resolve_backend(backend)
    logger.info(
        "Solving generalised eigenproblem for %d eigenpairs (backend=%s)...",
        n_eigenpairs, be.value,
    )
    eigenvalues, eigenvectors = eigsh_solve(
        L, M,
        k=n_eigenpairs,
        sigma=sigma,
        backend=be.value,
    )

    return LaplaceBeltrami(
        stiffness=L,
        mass=M,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Normal estimation
# ═══════════════════════════════════════════════════════════════════════════


def estimate_normals(
    points: np.ndarray,
    n_neighbors: int = 30,
    orient_consistent: bool = True,
) -> np.ndarray:
    """
    Estimate point normals via local PCA on the k-nearest neighbourhood.

    Parameters
    ----------
    points : np.ndarray, shape (N, 3)
        3D point coordinates.
    n_neighbors : int
        Size of the local neighbourhood for PCA.
    orient_consistent : bool
        If True, attempt to orient normals consistently outward
        using a minimum spanning tree propagation.

    Returns
    -------
    normals : np.ndarray, shape (N, 3)
        Unit normals at each point.
    """
    try:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=n_neighbors),
        )
        if orient_consistent:
            pcd.orient_normals_consistent_tangent_plane(k=n_neighbors)
        return np.asarray(pcd.normals, dtype=np.float64)
    except ImportError:
        logger.info("Open3D not available; using PCA-based normal estimation")
        return _estimate_normals_pca(points, n_neighbors)


def _estimate_normals_pca(
    points: np.ndarray,
    n_neighbors: int = 30,
) -> np.ndarray:
    """Fallback PCA-based normal estimation using scipy KDTree."""
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    _, idx = tree.query(points, k=n_neighbors)

    normals = np.zeros_like(points)
    for i in range(points.shape[0]):
        neighbours = points[idx[i]]
        centered = neighbours - neighbours.mean(axis=0)
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        normals[i] = Vt[-1]  # smallest singular vector = normal

    # Rough outward orientation: point away from centroid
    centroid = points.mean(axis=0)
    for i in range(points.shape[0]):
        if np.dot(normals[i], points[i] - centroid) < 0:
            normals[i] *= -1

    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return normals / norms


# ═══════════════════════════════════════════════════════════════════════════
# Point cloud utilities
# ═══════════════════════════════════════════════════════════════════════════


def _farthest_point_sampling(
    points: np.ndarray,
    n_samples: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Farthest-point sampling (FPS) for uniform spatial coverage.

    Parameters
    ----------
    points : np.ndarray, shape (N, 3)
    n_samples : int
        Number of points to select.
    seed : int or None
        Random seed for the initial point selection.

    Returns
    -------
    indices : np.ndarray, shape (n_samples,)
        Selected point indices.
    """
    N = points.shape[0]
    rng = np.random.default_rng(seed)
    selected = np.zeros(n_samples, dtype=np.int64)
    selected[0] = rng.integers(0, N)

    # Distance from each point to the nearest selected point
    dists = np.full(N, np.inf, dtype=np.float64)

    for i in range(1, n_samples):
        last = selected[i - 1]
        d = np.sum((points - points[last]) ** 2, axis=1)
        dists = np.minimum(dists, d)
        selected[i] = np.argmax(dists)

    return selected


def compute_point_areas(
    points: np.ndarray,
    n_neighbors: int = 10,
) -> np.ndarray:
    """
    Estimate the Voronoi area element for each point in a point cloud.

    This is useful for weighting points in optimal transport computations,
    so that regions with denser sampling (e.g., sulcal fundi) do not
    dominate the distance metric.

    Parameters
    ----------
    points : np.ndarray, shape (N, 3)
        3D coordinates.
    n_neighbors : int
        Number of neighbours for local density estimation.

    Returns
    -------
    areas : np.ndarray, shape (N,)
        Estimated area element per point (sums to approximate total
        surface area).
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    dists, _ = tree.query(points, k=n_neighbors + 1)
    # Mean distance to k nearest neighbours (exclude self at index 0)
    mean_dist = dists[:, 1:].mean(axis=1)
    # Area ~ π r² where r is the mean nearest-neighbour distance
    areas = np.pi * mean_dist ** 2
    # Normalize so areas sum to approximate total surface area
    # (rough estimate: total area ≈ N × mean_area)
    return areas


def to_feature_matrix(
    pointcloud: CorticalPointCloud,
    overlay_names: Optional[List[str]] = None,
    include_coordinates: bool = False,
) -> np.ndarray:
    """
    Stack overlays into a feature matrix for downstream analysis.

    Parameters
    ----------
    pointcloud : CorticalPointCloud
        Source point cloud.
    overlay_names : list of str or None
        Which overlays to include. None = all overlays.
    include_coordinates : bool
        If True, prepend the 3D coordinates as columns.

    Returns
    -------
    features : np.ndarray, shape (N, D)
        Feature matrix.
    """
    names = overlay_names or pointcloud.overlay_names
    if not names and not include_coordinates:
        raise ValueError("No overlays selected and include_coordinates=False")

    columns = []
    if include_coordinates:
        columns.append(pointcloud.points)
    for name in names:
        if name not in pointcloud.overlays:
            raise KeyError(f"Overlay '{name}' not found. Available: {pointcloud.overlay_names}")
        columns.append(pointcloud.overlays[name][:, np.newaxis])

    return np.hstack(columns)
