"""
Publication-quality brain point cloud visualization.

Renders cortical and subcortical point clouds as 3D brain figures with
spectral descriptor overlays, eigenpair visualization, curvature maps,
and interhemispheric asymmetry panels. Uses PyVista for 3D rendering
and matplotlib for 2D panels and colorbars.

All functions follow the Jiang et al. 2024 (Nature Communications)
figure style with colorblind-safe palettes, per-view Plotter instances
(to fix PyVista multi-subplot background bugs), and publication DPI.

Functions
---------
plot_pointcloud_brain     : 4-view brain render of a scalar overlay
plot_eigenpairs           : Multi-panel eigenvector visualization
plot_spectral_descriptors : HKS/WKS/GPS at selected scales
plot_curvature_map        : Mean/Gaussian curvature on point cloud
plot_asymmetry_pointcloud : Left vs right hemisphere comparison
plot_registration_result  : Before/after registration overlay
plot_morphometric_summary : Per-ROI morphometric bar/spider charts
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

    Returns
    -------
    pyvista.Plotter or None
        The plotter object if save_path is None (for interactive use).
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
        import matplotlib.cm as cm
        from matplotlib.colors import Normalize

        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        norm = Normalize(
            vmin=clim[0] if clim else scalars.min(),
            vmax=clim[1] if clim else scalars.max(),
        )
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax, label=scalar_name)

    plt.tight_layout()

    if save_path:
        fig.savefig(
            str(save_path), dpi=dpi, bbox_inches="tight",
            facecolor="white", transparent=False,
        )
        logger.info("Saved brain figure to %s", save_path)
        plt.close(fig)
        return None
    else:
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
        plt.close(fig)
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
        plt.close(fig)
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
        plt.close(fig)
        return None
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════


def _render_multi_scalar(
    points: np.ndarray,
    panels: list,
    point_size: float,
    save_path,
    dpi: int,
    suptitle: str = "",
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
        plt.close(fig)
        return None
    return fig
