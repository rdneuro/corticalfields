"""
Publication-quality brain point cloud visualization.

Renders cortical and subcortical point clouds as 3D brain figures with
spectral descriptor overlays, eigenpair visualization, curvature maps,
interhemispheric asymmetry panels, and hippomaps-style folded + unfolded
hippocampal surface figures.

All functions follow the Jiang et al. 2024 (Nature Communications)
figure style with colorblind-safe palettes, per-view Plotter instances
(to fix PyVista multi-subplot background bugs), and publication DPI.

Functions
---------
plot_pointcloud_brain        : 4-view brain render of a scalar overlay
plot_eigenpairs              : Multi-panel eigenvector visualization
plot_spectral_descriptors    : HKS/WKS/GPS at selected scales
plot_curvature_map           : Mean/Gaussian curvature on point cloud
plot_asymmetry_pointcloud    : Left vs right hemisphere comparison
plot_registration_result     : Before/after registration overlay
plot_morphometric_summary    : Per-ROI morphometric bar/spider charts
reconstruct_mesh_from_pcd    : Triangulated mesh from point cloud (Open3D)
spectral_unfold              : 2D parameterization via LBO eigenvectors
plot_hippo_folded_unfolded   : Single-scalar folded 3D + unfolded 2D panel
plot_hippo_multi_scalar      : Multi-row folded + unfolded (hippomaps style)
plot_hippo_hks_panel         : One-liner HKS multi-scale hippomaps panel
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# ── Publication defaults ─────────────────────────────────────────────────
_DEFAULT_DPI = 300
_DEFAULT_CMAP = "viridis"
_DEFAULT_BG = "white"
_DEFAULT_POINT_SIZE = 2.0
_FIGSIZE_4VIEW = (12, 6)


# ═══════════════════════════════════════════════════════════════════════════
# 4-view brain render
# ═══════════════════════════════════════════════════════════════════════════


def plot_pointcloud_brain(
    points: np.ndarray,
    scalars: Optional[np.ndarray] = None,
    cmap: str = _DEFAULT_CMAP,
    clim: Optional[Tuple[float, float]] = None,
    title: str = "",
    point_size: float = _DEFAULT_POINT_SIZE,
    views: Optional[List[str]] = None,
    bg_color: str = _DEFAULT_BG,
    show_colorbar: bool = True,
    scalar_name: str = "overlay",
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = _DEFAULT_DPI,
    off_screen: bool = True,
    show: bool = True,
) -> Optional["pyvista.Plotter"]:
    """
    Render a point cloud brain in 4 canonical views.

    Creates lateral-left, medial-left, medial-right, lateral-right views
    arranged horizontally, following the BrainSpace / Jiang et al. style.

    Parameters
    ----------
    points : np.ndarray, shape (N, 3)
        Point cloud coordinates in RAS space.
    scalars : np.ndarray or None, shape (N,)
        Per-point scalar values for coloring (e.g., HKS, curvature).
        If None, renders uniform color.
    cmap : str
        Matplotlib/PyVista colormap name.
    clim : tuple of (float, float) or None
        Color range. None = auto from data.
    title : str
        Figure title.
    point_size : float
        Rendered point size in pixels.
    views : list of str or None
        View names. Default: lateral_lh, medial_lh, medial_rh, lateral_rh.
    bg_color : str
        Background color.
    show_colorbar : bool
    scalar_name : str
        Label for the colorbar.
    save_path : str, Path, or None
        If provided, saves the figure (PNG, PDF, or SVG).
    dpi : int
        Resolution for saved figures.
    off_screen : bool
        Render off-screen (for batch processing / servers).
    show : bool
        If True (default), call ``plt.show()`` so the figure renders in
        interactive environments like Spyder.  Set False for batch mode.

    Returns
    -------
    matplotlib.figure.Figure or None
        The figure object, or None if *show* is True (already displayed).
    """
    try:
        import pyvista as pv
    except ImportError:
        raise ImportError(
            "PyVista is required for point cloud brain visualization. "
            "Install with: pip install pyvista"
        )

    if views is None:
        views = ["lateral_lh", "medial_lh", "medial_rh", "lateral_rh"]

    # Camera positions for each canonical view (azimuth, elevation, roll)
    _CAMERAS = {
        "lateral_lh":  {"position": (-200, 0, 0), "focal_point": (0, 0, 0), "viewup": (0, 0, 1)},
        "medial_lh":   {"position": (200, 0, 0),  "focal_point": (0, 0, 0), "viewup": (0, 0, 1)},
        "lateral_rh":  {"position": (200, 0, 0),  "focal_point": (0, 0, 0), "viewup": (0, 0, 1)},
        "medial_rh":   {"position": (-200, 0, 0), "focal_point": (0, 0, 0), "viewup": (0, 0, 1)},
        "dorsal":      {"position": (0, 0, 200),  "focal_point": (0, 0, 0), "viewup": (0, 1, 0)},
        "ventral":     {"position": (0, 0, -200), "focal_point": (0, 0, 0), "viewup": (0, 1, 0)},
        "anterior":    {"position": (0, 200, 0),  "focal_point": (0, 0, 0), "viewup": (0, 0, 1)},
        "posterior":   {"position": (0, -200, 0), "focal_point": (0, 0, 0), "viewup": (0, 0, 1)},
    }

    n_views = len(views)

    # Create a PolyData point cloud
    cloud = pv.PolyData(points.astype(np.float32))
    if scalars is not None:
        cloud[scalar_name] = scalars.astype(np.float32)

    # Use per-view Plotter instances to avoid PyVista background color bug
    # in multi-subplot mode, then composite into a single image.
    images = []
    window_size = (600, 500)

    for view_name in views:
        p = pv.Plotter(off_screen=True, window_size=window_size)
        p.background_color = bg_color

        add_kwargs = dict(
            point_size=point_size,
            render_points_as_spheres=True,
            show_scalar_bar=False,
        )
        if scalars is not None:
            add_kwargs["scalars"] = scalar_name
            add_kwargs["cmap"] = cmap
            if clim is not None:
                add_kwargs["clim"] = clim

        p.add_mesh(cloud, **add_kwargs)

        cam = _CAMERAS.get(view_name)
        if cam:
            p.camera_position = [
                cam["position"], cam["focal_point"], cam["viewup"],
            ]

        images.append(p.screenshot(return_img=True))
        p.close()

    # Composite into a single matplotlib figure
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1, n_views,
        figsize=(4 * n_views, 4),
        dpi=dpi,
    )
    if n_views == 1:
        axes = [axes]

    for ax, img, vname in zip(axes, images, views):
        ax.imshow(img)
        ax.set_title(vname.replace("_", " ").title(), fontsize=10)
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    # Add colorbar
    if show_colorbar and scalars is not None:
        import matplotlib.cm as mcm
        from matplotlib.colors import Normalize

        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        norm = Normalize(
            vmin=clim[0] if clim else scalars.min(),
            vmax=clim[1] if clim else scalars.max(),
        )
        sm = mcm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax, label=scalar_name)

    plt.tight_layout()

    if save_path:
        fig.savefig(
            str(save_path), dpi=dpi, bbox_inches="tight",
            facecolor="white", transparent=False,
        )
        logger.info("Saved brain figure to %s", save_path)

    if show:
        plt.show()
        return None
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Eigenpair visualization
# ═══════════════════════════════════════════════════════════════════════════


def plot_eigenpairs(
    points: np.ndarray,
    eigenvectors: np.ndarray,
    eigenvalues: Optional[np.ndarray] = None,
    modes: Optional[List[int]] = None,
    cmap: str = "RdBu_r",
    point_size: float = _DEFAULT_POINT_SIZE,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = _DEFAULT_DPI,
    show: bool = True,
) -> Optional["matplotlib.figure.Figure"]:
    """
    Visualize selected LBO eigenvectors on a point cloud brain.

    Creates a multi-panel figure with one brain render per eigenmode,
    using diverging colormap (RdBu_r) centered at zero.

    Parameters
    ----------
    points : np.ndarray, shape (N, 3)
    eigenvectors : np.ndarray, shape (N, K)
    eigenvalues : np.ndarray or None, shape (K,)
        If provided, shown in subplot titles.
    modes : list of int or None
        Which eigenvector indices to plot. Default: [1, 2, 5, 10, 20, 50].
    cmap : str
    point_size : float
    save_path : str, Path, or None
    dpi : int
    show : bool
        If True (default), call ``plt.show()``.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    try:
        import pyvista as pv
    except ImportError:
        raise ImportError("PyVista required for eigenpair visualization")

    if modes is None:
        K = eigenvectors.shape[1]
        modes = [i for i in [1, 2, 5, 10, 20, 50] if i < K]

    n_modes = len(modes)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1, n_modes,
        figsize=(3.5 * n_modes, 3.5),
        dpi=dpi,
    )
    if n_modes == 1:
        axes = [axes]

    for ax, mode_idx in zip(axes, modes):
        phi = eigenvectors[:, mode_idx]
        vmax = max(abs(phi.min()), abs(phi.max()))

        cloud = pv.PolyData(points.astype(np.float32))
        cloud["phi"] = phi.astype(np.float32)

        p = pv.Plotter(off_screen=True, window_size=(500, 450))
        p.background_color = "white"
        p.add_mesh(
            cloud,
            scalars="phi",
            cmap=cmap,
            clim=(-vmax, vmax),
            point_size=point_size,
            render_points_as_spheres=True,
            show_scalar_bar=False,
        )
        p.camera_position = [(-200, 0, 0), (0, 0, 0), (0, 0, 1)]
        img = p.screenshot(return_img=True)
        p.close()

        ax.imshow(img)
        lam_str = f" (λ={eigenvalues[mode_idx]:.2f})" if eigenvalues is not None else ""
        ax.set_title(f"φ_{{{mode_idx}}}{lam_str}", fontsize=10)
        ax.axis("off")

    fig.suptitle("LBO Eigenvectors on Point Cloud", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
        logger.info("Saved eigenpair figure to %s", save_path)

    if show:
        plt.show()
        return None
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Spectral descriptor visualization
# ═══════════════════════════════════════════════════════════════════════════


def plot_spectral_descriptors(
    points: np.ndarray,
    hks: Optional[np.ndarray] = None,
    wks: Optional[np.ndarray] = None,
    scales: Optional[List[int]] = None,
    point_size: float = _DEFAULT_POINT_SIZE,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = _DEFAULT_DPI,
    show: bool = True,
) -> Optional["matplotlib.figure.Figure"]:
    """
    Visualize HKS and/or WKS at selected scales on a brain point cloud.

    Parameters
    ----------
    points : np.ndarray, shape (N, 3)
    hks : np.ndarray or None, shape (N, T_hks)
    wks : np.ndarray or None, shape (N, T_wks)
    scales : list of int or None
        Which scale indices to plot. Default: 4 evenly spaced.
    point_size : float
    save_path : str, Path, or None
    dpi : int
    show : bool
        If True (default), call ``plt.show()``.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    try:
        import pyvista as pv
    except ImportError:
        raise ImportError("PyVista required for descriptor visualization")

    import matplotlib.pyplot as plt

    descriptors = []
    labels = []

    if hks is not None:
        T = hks.shape[1]
        sel = scales if scales else np.linspace(0, T - 1, min(4, T), dtype=int).tolist()
        for s in sel:
            descriptors.append(hks[:, s])
            labels.append(f"HKS t={s}")

    if wks is not None:
        E = wks.shape[1]
        sel = scales if scales else np.linspace(0, E - 1, min(4, E), dtype=int).tolist()
        for s in sel:
            descriptors.append(wks[:, s])
            labels.append(f"WKS e={s}")

    if not descriptors:
        raise ValueError("At least one of hks or wks must be provided")

    n_panels = len(descriptors)
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(3.5 * n_panels, 3.5),
        dpi=dpi,
    )
    if n_panels == 1:
        axes = [axes]

    for ax, data, label in zip(axes, descriptors, labels):
        cloud = pv.PolyData(points.astype(np.float32))
        cloud["desc"] = data.astype(np.float32)

        p = pv.Plotter(off_screen=True, window_size=(500, 450))
        p.background_color = "white"
        p.add_mesh(
            cloud,
            scalars="desc",
            cmap="inferno",
            point_size=point_size,
            render_points_as_spheres=True,
            show_scalar_bar=False,
        )
        p.camera_position = [(-200, 0, 0), (0, 0, 0), (0, 0, 1)]
        img = p.screenshot(return_img=True)
        p.close()

        ax.imshow(img)
        ax.set_title(label, fontsize=10)
        ax.axis("off")

    fig.suptitle("Spectral Descriptors", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
        return None
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Curvature map
# ═══════════════════════════════════════════════════════════════════════════


def plot_curvature_map(
    points: np.ndarray,
    mean_curvature: np.ndarray,
    gaussian_curvature: Optional[np.ndarray] = None,
    point_size: float = _DEFAULT_POINT_SIZE,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = _DEFAULT_DPI,
    show: bool = True,
) -> Optional["matplotlib.figure.Figure"]:
    """
    Render mean and Gaussian curvature on brain point cloud.

    Parameters
    ----------
    points : np.ndarray, shape (N, 3)
    mean_curvature : np.ndarray, shape (N,)
    gaussian_curvature : np.ndarray or None, shape (N,)
    point_size : float
    save_path : str, Path, or None
    dpi : int
    show : bool
        If True (default), call ``plt.show()``.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    panels = [("Mean curvature (H)", mean_curvature, "RdBu_r")]
    if gaussian_curvature is not None:
        panels.append(("Gaussian curvature (K)", gaussian_curvature, "PiYG_r"))

    return _render_multi_scalar(
        points, panels, point_size, save_path, dpi,
        suptitle="Curvature",
        show=show,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Asymmetry visualization
# ═══════════════════════════════════════════════════════════════════════════


def plot_asymmetry_pointcloud(
    lh_points: np.ndarray,
    rh_points_mirrored: np.ndarray,
    lh_scalars: np.ndarray,
    rh_scalars: np.ndarray,
    scalar_name: str = "HKS",
    point_size: float = _DEFAULT_POINT_SIZE,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = _DEFAULT_DPI,
    show: bool = True,
) -> Optional["matplotlib.figure.Figure"]:
    """
    Side-by-side left vs mirrored-right hemisphere comparison.

    Parameters
    ----------
    lh_points : np.ndarray, shape (N_lh, 3)
    rh_points_mirrored : np.ndarray, shape (N_rh, 3)
    lh_scalars : np.ndarray, shape (N_lh,)
    rh_scalars : np.ndarray, shape (N_rh,)
    scalar_name : str
    point_size : float
    save_path : str, Path, or None
    dpi : int
    show : bool
        If True (default), call ``plt.show()``.
    """
    panels = [
        (f"Left — {scalar_name}", lh_points, lh_scalars, "inferno"),
        (f"Right (mirrored) — {scalar_name}", rh_points_mirrored, rh_scalars, "inferno"),
    ]

    import matplotlib.pyplot as plt
    try:
        import pyvista as pv
    except ImportError:
        raise ImportError("PyVista required")

    # Shared clim
    vmin = min(lh_scalars.min(), rh_scalars.min())
    vmax = max(lh_scalars.max(), rh_scalars.max())

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=dpi)

    for ax, (title, pts, vals, cmap) in zip(axes, panels):
        cloud = pv.PolyData(pts.astype(np.float32))
        cloud["s"] = vals.astype(np.float32)

        p = pv.Plotter(off_screen=True, window_size=(600, 500))
        p.background_color = "white"
        p.add_mesh(
            cloud,
            scalars="s",
            cmap=cmap,
            clim=(vmin, vmax),
            point_size=point_size,
            render_points_as_spheres=True,
            show_scalar_bar=False,
        )
        p.camera_position = [(-200, 0, 0), (0, 0, 0), (0, 0, 1)]
        img = p.screenshot(return_img=True)
        p.close()

        ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    fig.suptitle("Interhemispheric Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
        return None
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers (original)
# ═══════════════════════════════════════════════════════════════════════════


def _render_multi_scalar(
    points: np.ndarray,
    panels: list,
    point_size: float,
    save_path,
    dpi: int,
    suptitle: str = "",
    show: bool = True,
):
    """Render multiple scalar overlays as a horizontal panel figure."""
    try:
        import pyvista as pv
    except ImportError:
        raise ImportError("PyVista required")

    import matplotlib.pyplot as plt

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), dpi=dpi)
    if n == 1:
        axes = [axes]

    for ax, (title, scalars, cmap) in zip(axes, panels):
        vmax = max(abs(scalars.min()), abs(scalars.max()))

        cloud = pv.PolyData(points.astype(np.float32))
        cloud["s"] = scalars.astype(np.float32)

        p = pv.Plotter(off_screen=True, window_size=(500, 450))
        p.background_color = "white"
        p.add_mesh(
            cloud,
            scalars="s",
            cmap=cmap,
            clim=(-vmax, vmax),
            point_size=point_size,
            render_points_as_spheres=True,
            show_scalar_bar=False,
        )
        p.camera_position = [(-200, 0, 0), (0, 0, 0), (0, 0, 1)]
        img = p.screenshot(return_img=True)
        p.close()

        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
        return None
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#
# HIPPOMAPS-STYLE VISUALIZATION
# Folded 3D surface + spectrally-unfolded 2D flatmap
#
# The 2D "unfolded" view uses the first two non-trivial LBO eigenvectors
# (φ₁, φ₂) as a spectral parameterization of the surface.  For the
# hippocampus: φ₁ ≈ anterior–posterior, φ₂ ≈ proximal–distal (subfield).
# This is an intrinsic conformal unfolding — no template needed.
#
# The 3D "folded" view reconstructs a triangulated mesh from the PCD via
# Open3D (Poisson or Ball-Pivoting) for smooth surface rendering.
#
# ═══════════════════════════════════════════════════════════════════════════


# ── Mesh reconstruction from point cloud ─────────────────────────────────

def reconstruct_mesh_from_pcd(
    points: np.ndarray,
    normals: Optional[np.ndarray] = None,
    method: str = "poisson",
    depth: int = 9,
    n_neighbors_normal: int = 30,
    radii: Optional[List[float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruct a triangulated mesh from a point cloud via Open3D.

    Parameters
    ----------
    points : ndarray, shape (N, 3)
        Point cloud coordinates.
    normals : ndarray, shape (N, 3), optional
        Pre-computed oriented normals.  Estimated via PCA if None.
    method : {"poisson", "ball_pivoting"}
        ``"poisson"`` — smoother, fills holes, may extrapolate slightly.
        ``"ball_pivoting"`` — faithful to point density, may leave holes.
    depth : int
        Octree depth for Poisson reconstruction (higher = more detail).
    n_neighbors_normal : int
        KNN for normal estimation when *normals* is None.
    radii : list of float, optional
        Ball radii for ball-pivoting.  Auto-estimated if None.

    Returns
    -------
    vertices : ndarray, shape (V, 3)
    faces : ndarray, shape (F, 3), dtype int64
    """
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError(
            "Open3D is required for mesh reconstruction from point clouds. "
            "Install with: pip install open3d"
        )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
    else:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=n_neighbors_normal)
        )
        pcd.orient_normals_consistent_tangent_plane(k=n_neighbors_normal)

    if method == "poisson":
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth, linear_fit=True
        )
        # Trim low-density vertices extrapolated beyond the PCD boundary
        densities_arr = np.asarray(densities)
        density_thresh = np.quantile(densities_arr, 0.01)
        mesh.remove_vertices_by_mask(densities_arr < density_thresh)

    elif method == "ball_pivoting":
        if radii is None:
            dists = pcd.compute_nearest_neighbor_distance()
            avg_dist = float(np.mean(dists))
            radii = [avg_dist * f for f in (1.0, 1.5, 2.0, 3.0)]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )
    else:
        raise ValueError(
            f"Unknown reconstruction method '{method}'. "
            "Use 'poisson' or 'ball_pivoting'."
        )

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles).astype(np.int64)
    return vertices, faces


# ── Scalar transfer PCD → mesh ───────────────────────────────────────────

def _transfer_scalars(
    src_points: np.ndarray,
    src_data: np.ndarray,
    dst_points: np.ndarray,
) -> np.ndarray:
    """Transfer per-point data via nearest-neighbor lookup.

    Parameters
    ----------
    src_points : (N, 3)
    src_data : (N,) or (N, C)
    dst_points : (V, 3)

    Returns
    -------
    dst_data : (V,) or (V, C)
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(src_points)
    _, idx = tree.query(dst_points, k=1)
    return src_data[idx]


# ── Spectral unfolding ───────────────────────────────────────────────────

def spectral_unfold(
    eigenvectors: np.ndarray,
    ev_indices: Tuple[int, int] = (1, 2),
    flip_ap: bool = False,
    flip_pd: bool = False,
) -> np.ndarray:
    """Build 2D unfolded coordinates from LBO eigenvectors.

    The first non-trivial eigenvector (φ₁, index 1) typically aligns with
    the longest axis of the structure (anterior–posterior for hippocampus).
    The second (φ₂, index 2) aligns with the proximal–distal / subfield
    direction.  Together they define an intrinsic spectral flattening.

    Parameters
    ----------
    eigenvectors : ndarray, shape (V, K)
        Mass-orthonormal LBO eigenvectors.
    ev_indices : tuple of int
        Which eigenvectors to use as (x, y).  Default ``(1, 2)`` = first
        two non-trivial.  Try ``(1, 3)`` or ``(2, 3)`` if the default
        produces a folded-back map.
    flip_ap, flip_pd : bool
        Flip horizontal / vertical axis to match anatomical convention.

    Returns
    -------
    uv : ndarray, shape (V, 2)
    """
    i, j = ev_indices
    if max(i, j) >= eigenvectors.shape[1]:
        raise ValueError(
            f"ev_indices={ev_indices} but eigenvectors only has "
            f"{eigenvectors.shape[1]} columns.  Increase n_eigenpairs."
        )
    u = eigenvectors[:, i].copy()
    v = eigenvectors[:, j].copy()
    if flip_ap:
        u *= -1
    if flip_pd:
        v *= -1
    return np.column_stack([u, v])


# ── Internal: render folded 3D surface via PyVista (off-screen) ──────────

def _render_folded_3d(
    vertices: np.ndarray,
    faces: np.ndarray,
    scalars: np.ndarray,
    cmap: str = _DEFAULT_CMAP,
    clim: Optional[Tuple[float, float]] = None,
    view_angle: str = "lateral",
    window_size: Tuple[int, int] = (900, 700),
    smooth_shading: bool = True,
) -> np.ndarray:
    """Render the folded 3D hippocampal surface off-screen.

    Returns
    -------
    image : ndarray, shape (H, W, 3), dtype uint8
    """
    import pyvista as pv

    pv.OFF_SCREEN = True

    n_f = faces.shape[0]
    pv_faces = np.column_stack([np.full(n_f, 3), faces]).ravel()
    mesh = pv.PolyData(vertices, pv_faces)
    mesh["overlay"] = scalars

    _cam = {
        "lateral":   (np.array([-1, 0, 0], dtype=float), np.array([0, 0, 1], dtype=float)),
        "medial":    (np.array([1, 0, 0], dtype=float),  np.array([0, 0, 1], dtype=float)),
        "anterior":  (np.array([0, 1, 0], dtype=float),  np.array([0, 0, 1], dtype=float)),
        "posterior": (np.array([0, -1, 0], dtype=float), np.array([0, 0, 1], dtype=float)),
        "dorsal":    (np.array([0, 0, 1], dtype=float),  np.array([0, 1, 0], dtype=float)),
        "ventral":   (np.array([0, 0, -1], dtype=float), np.array([0, 1, 0], dtype=float)),
    }

    pl = pv.Plotter(off_screen=True, window_size=window_size)
    pl.set_background("white")

    if clim is None:
        lo = float(np.percentile(scalars, 2))
        hi = float(np.percentile(scalars, 98))
        clim = (lo, hi)

    pl.add_mesh(
        mesh,
        scalars="overlay",
        cmap=cmap,
        clim=clim,
        smooth_shading=smooth_shading,
        show_scalar_bar=False,
    )

    center = vertices.mean(axis=0)
    extent = float(np.ptp(vertices, axis=0).max())

    if isinstance(view_angle, str) and view_angle in _cam:
        direction, up = _cam[view_angle]
        cam_pos = center + direction * extent * 2.5
        pl.camera.position = cam_pos.tolist()
        pl.camera.focal_point = center.tolist()
        pl.camera.up = up.tolist()
    elif isinstance(view_angle, (tuple, list)) and len(view_angle) == 2:
        pl.camera.azimuth = view_angle[0]
        pl.camera.elevation = view_angle[1]
    else:
        pl.reset_camera()

    pl.camera.zoom(1.3)
    img = pl.screenshot(return_img=True)
    pl.close()
    return img


# ── Internal: render unfolded 2D flatmap via matplotlib ──────────────────

def _render_unfolded_2d(
    uv: np.ndarray,
    scalars: np.ndarray,
    faces: Optional[np.ndarray] = None,
    cmap: str = _DEFAULT_CMAP,
    clim: Optional[Tuple[float, float]] = None,
    method: str = "tricontourf",
    n_levels: int = 200,
    ax: Optional["matplotlib.axes.Axes"] = None,
    show_border: bool = True,
) -> Tuple["matplotlib.figure.Figure", "matplotlib.axes.Axes"]:
    """Render spectrally-unfolded 2D flatmap.

    Uses mesh triangulation if *faces* are available, else Delaunay on
    the (φ₁, φ₂) coordinates with automatic masking of spurious edges.
    """
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    else:
        fig = ax.figure

    if clim is None:
        lo = float(np.percentile(scalars, 2))
        hi = float(np.percentile(scalars, 98))
        clim = (lo, hi)

    # Build triangulation
    if faces is not None:
        tri = mtri.Triangulation(uv[:, 0], uv[:, 1], triangles=faces)
    else:
        tri = mtri.Triangulation(uv[:, 0], uv[:, 1])
        # Mask overly-long edges (Delaunay artifacts on non-convex domain)
        pts = uv[tri.triangles]  # (T, 3, 2)
        lens = [
            np.linalg.norm(pts[:, a] - pts[:, b], axis=1)
            for a, b in [(0, 1), (1, 2), (2, 0)]
        ]
        max_edge = np.max(lens, axis=0)
        tri.set_mask(max_edge > np.percentile(max_edge, 95) * 1.5)

    # Render
    if method == "tricontourf":
        levels = np.linspace(clim[0], clim[1], n_levels)
        ax.tricontourf(tri, scalars, levels=levels, cmap=cmap, extend="both")
    elif method == "tripcolor":
        tc = ax.tripcolor(tri, scalars, cmap=cmap, shading="gouraud")
        tc.set_clim(*clim)
    else:
        raise ValueError(
            f"flatmap_method must be 'tricontourf' or 'tripcolor', got '{method}'"
        )

    if show_border:
        ax.triplot(tri, linewidth=0.05, color="0.3", alpha=0.15)

    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


# ── Shared helper: ensure mesh + transfer scalars / eigenvectors ─────────

def _prepare_hippo_data(
    pcd_points: np.ndarray,
    eigenvectors: np.ndarray,
    scalars: np.ndarray,
    mesh_vertices: Optional[np.ndarray],
    mesh_faces: Optional[np.ndarray],
    normals: Optional[np.ndarray],
    reconstruct_method: str,
    reconstruct_depth: int,
    ev_indices: Tuple[int, int],
    flip_ap: bool,
    flip_pd: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(mesh_verts, mesh_faces, mesh_scalars, uv)``."""

    if mesh_vertices is None or mesh_faces is None:
        logger.info("Reconstructing mesh from point cloud (%s, depth=%d)...",
                     reconstruct_method, reconstruct_depth)
        mesh_vertices, mesh_faces = reconstruct_mesh_from_pcd(
            pcd_points, normals=normals,
            method=reconstruct_method, depth=reconstruct_depth,
        )
        logger.info("Mesh: %d vertices, %d faces",
                     mesh_vertices.shape[0], mesh_faces.shape[0])

    # Transfer scalars if shapes don't match
    if scalars.shape[0] != mesh_vertices.shape[0]:
        mesh_scalars = _transfer_scalars(pcd_points, scalars, mesh_vertices)
    else:
        mesh_scalars = scalars

    # Transfer eigenvectors + unfold
    if eigenvectors.shape[0] != mesh_vertices.shape[0]:
        mesh_ev = _transfer_scalars(pcd_points, eigenvectors, mesh_vertices)
    else:
        mesh_ev = eigenvectors

    uv = spectral_unfold(
        mesh_ev, ev_indices=ev_indices,
        flip_ap=flip_ap, flip_pd=flip_pd,
    )

    return mesh_vertices, mesh_faces, mesh_scalars, uv


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC: Hippomaps-style single-scalar plot
# ═══════════════════════════════════════════════════════════════════════════


def plot_hippo_folded_unfolded(
    pcd_points: np.ndarray,
    eigenvectors: np.ndarray,
    scalars: np.ndarray,
    *,
    mesh_vertices: Optional[np.ndarray] = None,
    mesh_faces: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None,
    reconstruct_method: str = "poisson",
    reconstruct_depth: int = 9,
    ev_indices: Tuple[int, int] = (1, 2),
    flip_ap: bool = False,
    flip_pd: bool = False,
    cmap: str = _DEFAULT_CMAP,
    clim: Optional[Tuple[float, float]] = None,
    view_angle: str = "lateral",
    flatmap_method: str = "tricontourf",
    title: str = "",
    scalar_label: str = "HKS",
    figsize: Tuple[float, float] = (12, 5),
    dpi: int = _DEFAULT_DPI,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    return_components: bool = False,
) -> Union["matplotlib.figure.Figure",
           Tuple["matplotlib.figure.Figure", np.ndarray, np.ndarray, np.ndarray]]:
    """Hippomaps-style composite: folded 3D surface + spectrally-unfolded 2D.

    Left panel  — 3D folded surface (Poisson-reconstructed mesh from PCD).
    Right panel — 2D flatmap using (φ₁, φ₂) as Cartesian coordinates.
    Bottom      — shared horizontal colorbar.

    Parameters
    ----------
    pcd_points : ndarray, shape (N, 3)
        Original point cloud.
    eigenvectors : ndarray, shape (N, K) or (V, K)
        LBO eigenvectors (from ``compute_pointcloud_eigenpairs``).
    scalars : ndarray, shape (N,) or (V,)
        Per-point scalar overlay (HKS, curvature, thickness, etc.).
    mesh_vertices, mesh_faces : ndarray, optional
        Pre-computed mesh.  If None, reconstructed automatically.
    normals : ndarray, shape (N, 3), optional
        Pre-computed normals for mesh reconstruction.
    reconstruct_method : {"poisson", "ball_pivoting"}
    reconstruct_depth : int
        Poisson octree depth.
    ev_indices : tuple of int
        Which eigenvectors for (x, y) unfolding.  Default ``(1, 2)``.
    flip_ap, flip_pd : bool
        Flip anterior–posterior / proximal–distal axes.
    cmap : str
        Matplotlib colormap.
    clim : tuple of float, optional
        ``(vmin, vmax)``; auto [2nd, 98th] percentile if None.
    view_angle : str or tuple
        Camera preset or ``(azimuth, elevation)`` tuple.
    flatmap_method : {"tricontourf", "tripcolor"}
    title : str
        Suptitle.
    scalar_label : str
        Colorbar label.
    figsize : tuple of float
    dpi : int
    save_path : str or Path, optional
    show : bool
        If True (default), call ``plt.show()`` so the figure renders in
        interactive environments like Spyder.
    return_components : bool
        If True, also return ``(mesh_vertices, mesh_faces, uv)``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Or ``(fig, mesh_vertices, mesh_faces, uv)`` if *return_components*.
        Returns None if *show* is True and *return_components* is False.
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors
    from matplotlib.gridspec import GridSpec

    verts, faces, sc, uv = _prepare_hippo_data(
        pcd_points, eigenvectors, scalars,
        mesh_vertices, mesh_faces, normals,
        reconstruct_method, reconstruct_depth,
        ev_indices, flip_ap, flip_pd,
    )

    if clim is None:
        clim = (float(np.percentile(sc, 2)), float(np.percentile(sc, 98)))

    # 3D folded render
    img_3d = _render_folded_3d(
        verts, faces, sc,
        cmap=cmap, clim=clim, view_angle=view_angle,
    )

    # Compose figure
    fig = plt.figure(figsize=figsize, facecolor="white", dpi=dpi)
    gs = GridSpec(2, 2, height_ratios=[1.0, 0.04], hspace=0.08, wspace=0.05)

    ax_3d = fig.add_subplot(gs[0, 0])
    ax_3d.imshow(img_3d)
    ax_3d.axis("off")
    ax_3d.set_title("Folded", fontsize=11, fontweight="bold", pad=4)

    ax_2d = fig.add_subplot(gs[0, 1])
    _render_unfolded_2d(
        uv, sc, faces=faces,
        cmap=cmap, clim=clim, method=flatmap_method, ax=ax_2d,
    )
    ax_2d.set_title("Unfolded (spectral)", fontsize=11, fontweight="bold", pad=4)

    # Shared colorbar
    cbar_ax = fig.add_subplot(gs[1, :])
    norm = colors.Normalize(vmin=clim[0], vmax=clim[1])
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(scalar_label, fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)

    if save_path:
        fig.savefig(
            str(save_path), dpi=dpi, bbox_inches="tight",
            facecolor="white", edgecolor="none",
        )
        logger.info("Saved hippo figure to %s", save_path)

    if show:
        plt.show()

    if return_components:
        return fig, verts, faces, uv
    if show:
        return None
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC: Multi-row hippomaps-style panel
# ═══════════════════════════════════════════════════════════════════════════


def plot_hippo_multi_scalar(
    pcd_points: np.ndarray,
    eigenvectors: np.ndarray,
    scalar_dict: Dict[str, np.ndarray],
    *,
    mesh_vertices: Optional[np.ndarray] = None,
    mesh_faces: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None,
    reconstruct_method: str = "poisson",
    reconstruct_depth: int = 9,
    ev_indices: Tuple[int, int] = (1, 2),
    flip_ap: bool = False,
    flip_pd: bool = False,
    cmap: str = _DEFAULT_CMAP,
    per_row_clim: bool = False,
    view_angle: str = "lateral",
    flatmap_method: str = "tricontourf",
    figsize_per_row: Tuple[float, float] = (12, 4),
    dpi: int = _DEFAULT_DPI,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> Optional["matplotlib.figure.Figure"]:
    """Multi-row hippomaps-style panel: one row per scalar (folded + unfolded).

    Replicates the hippomaps layout — N rows, each showing the same
    structure with a different scalar overlay.  Left column = 3D folded
    surface, right column = spectrally-unfolded 2D flatmap.

    Parameters
    ----------
    pcd_points : ndarray, shape (N, 3)
    eigenvectors : ndarray, shape (N, K) or (V, K)
    scalar_dict : dict of {str: ndarray}
        Ordered mapping ``{label: per-point-array}``.
        Example: ``{"HKS t=0": hks[:, 0], "HKS t=8": hks[:, 8], ...}``
    per_row_clim : bool
        If True, compute clim independently per row (highlights local
        contrast).  If False (default), use global clim across all rows.
    show : bool
        If True (default), call ``plt.show()``.
    (other params: see ``plot_hippo_folded_unfolded``)

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors
    from matplotlib.gridspec import GridSpec

    # Reconstruct mesh once
    if mesh_vertices is None or mesh_faces is None:
        logger.info("Reconstructing mesh from point cloud...")
        mesh_vertices, mesh_faces = reconstruct_mesh_from_pcd(
            pcd_points, normals=normals,
            method=reconstruct_method, depth=reconstruct_depth,
        )
        logger.info("Mesh: %d verts, %d faces",
                     mesh_vertices.shape[0], mesh_faces.shape[0])

    # Transfer eigenvectors + compute unfolding once
    if eigenvectors.shape[0] != mesh_vertices.shape[0]:
        mesh_ev = _transfer_scalars(pcd_points, eigenvectors, mesh_vertices)
    else:
        mesh_ev = eigenvectors

    uv = spectral_unfold(
        mesh_ev, ev_indices=ev_indices,
        flip_ap=flip_ap, flip_pd=flip_pd,
    )

    n_rows = len(scalar_dict)
    names = list(scalar_dict.keys())

    # Global clim
    if not per_row_clim:
        all_vals = np.concatenate([v.ravel() for v in scalar_dict.values()])
        global_clim = (
            float(np.percentile(all_vals, 2)),
            float(np.percentile(all_vals, 98)),
        )
    else:
        global_clim = None

    # Figure layout
    total_h = figsize_per_row[1] * n_rows + 0.6
    fig = plt.figure(
        figsize=(figsize_per_row[0], total_h),
        facecolor="white", dpi=dpi,
    )
    gs = GridSpec(
        n_rows + 1, 2,
        height_ratios=[1.0] * n_rows + [0.03],
        width_ratios=[1.0, 1.0],
        hspace=0.12, wspace=0.05,
    )

    for row, name in enumerate(names):
        raw = scalar_dict[name]

        # Transfer per-row scalar
        if raw.shape[0] != mesh_vertices.shape[0]:
            sc = _transfer_scalars(pcd_points, raw, mesh_vertices)
        else:
            sc = raw

        row_clim = global_clim if global_clim is not None else (
            float(np.percentile(sc, 2)),
            float(np.percentile(sc, 98)),
        )

        # 3D folded
        img_3d = _render_folded_3d(
            mesh_vertices, mesh_faces, sc,
            cmap=cmap, clim=row_clim, view_angle=view_angle,
            window_size=(800, 600),
        )
        ax_3d = fig.add_subplot(gs[row, 0])
        ax_3d.imshow(img_3d)
        ax_3d.axis("off")
        ax_3d.set_ylabel(
            name, fontsize=10, fontweight="bold",
            rotation=90, labelpad=15,
        )

        # 2D unfolded
        ax_2d = fig.add_subplot(gs[row, 1])
        _render_unfolded_2d(
            uv, sc, faces=mesh_faces,
            cmap=cmap, clim=row_clim,
            method=flatmap_method, ax=ax_2d,
            show_border=(row == 0),
        )

    # Column headers
    fig.text(0.28, 0.98, "Folded", ha="center", fontsize=12, fontweight="bold")
    fig.text(0.74, 0.98, "Unfolded (spectral)",
             ha="center", fontsize=12, fontweight="bold")

    # Shared colorbar
    cbar_ax = fig.add_subplot(gs[n_rows, :])
    use_clim = global_clim if global_clim is not None else (0.0, 1.0)
    norm = colors.Normalize(vmin=use_clim[0], vmax=use_clim[1])
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=8)

    if save_path:
        fig.savefig(
            str(save_path), dpi=dpi, bbox_inches="tight",
            facecolor="white", edgecolor="none",
        )
        logger.info("Saved hippo multi-scalar figure to %s", save_path)

    if show:
        plt.show()
        return None
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC: One-liner HKS panel from PCD pipeline result
# ═══════════════════════════════════════════════════════════════════════════


def plot_hippo_hks_panel(
    pcd_result,
    lb: "LaplaceBeltrami",
    *,
    n_hks_scales: int = 16,
    display_scales: Optional[List[int]] = None,
    cmap: str = _DEFAULT_CMAP,
    view_angle: str = "lateral",
    ev_indices: Tuple[int, int] = (1, 2),
    flip_ap: bool = False,
    flip_pd: bool = False,
    per_row_clim: bool = False,
    reconstruct_method: str = "poisson",
    reconstruct_depth: int = 9,
    dpi: int = _DEFAULT_DPI,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> Optional["matplotlib.figure.Figure"]:
    """One-liner: T1wPCDPipeline result + LaplaceBeltrami → hippomaps panel.

    Computes HKS internally and renders a multi-row folded + unfolded figure.

    Parameters
    ----------
    pcd_result : T1wPCDPipeline.run() result
        Must have a ``.points`` attribute, ndarray (N, 3).
    lb : LaplaceBeltrami
        From ``compute_pointcloud_eigenpairs``.
    n_hks_scales : int
        Number of HKS time scales to compute.
    display_scales : list of int, optional
        Which HKS column indices to display.  Default: 6 evenly-spaced.
    show : bool
        If True (default), call ``plt.show()``.
    (other params: see ``plot_hippo_multi_scalar``)

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    from corticalfields.pointcloud.spectral import hks_pointcloud

    points = pcd_result.points
    hks = hks_pointcloud(lb.eigenvalues, lb.eigenvectors, n_scales=n_hks_scales)

    if display_scales is None:
        n_s = hks.shape[1]
        display_scales = np.linspace(0, n_s - 1, min(6, n_s), dtype=int).tolist()

    scalar_dict = {f"HKS t={s}": hks[:, s] for s in display_scales}

    return plot_hippo_multi_scalar(
        pcd_points=points,
        eigenvectors=lb.eigenvectors,
        scalar_dict=scalar_dict,
        ev_indices=ev_indices,
        flip_ap=flip_ap,
        flip_pd=flip_pd,
        per_row_clim=per_row_clim,
        cmap=cmap,
        view_angle=view_angle,
        reconstruct_method=reconstruct_method,
        reconstruct_depth=reconstruct_depth,
        dpi=dpi,
        save_path=save_path,
        show=show,
    )
