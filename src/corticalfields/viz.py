"""
Publication-quality visualization of cortical surface data.

Renders surprise maps, z-score maps, spectral features, and graph
metrics on inflated/pial cortical surfaces using PyVista and matplotlib.
Designed for Nature/NeuroImage-quality figures.

Two rendering backends are supported:
    • **PyVista** — interactive 3D rendering with GPU acceleration
    • **matplotlib** — 2D projections (flatmaps) for print figures

For the highest-quality publication figures, use the PyVista backend
with offscreen rendering and export to PNG at 300+ DPI.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

# ── Colour maps ─────────────────────────────────────────────────────────
# Diverging cmap for z-scores: blue (negative/atrophy) → white → red (positive)
ZSCORE_CMAP = "RdBu_r"
# Sequential cmap for surprise: low (white/yellow) → high (red/dark)
SURPRISE_CMAP = "YlOrRd"
# Sequential cmap for features (e.g. thickness)
FEATURE_CMAP = "viridis"


def plot_surface_scalar(
    vertices: np.ndarray,
    faces: np.ndarray,
    scalars: np.ndarray,
    cmap: str = "viridis",
    clim: Optional[Tuple[float, float]] = None,
    title: str = "",
    view: str = "lateral",
    figsize: Tuple[int, int] = (10, 8),
    output_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    **kwargs,
) -> Optional[Figure]:
    """
    Render a scalar map on a cortical surface using PyVista.

    Parameters
    ----------
    vertices : (N, 3) array
    faces : (F, 3) array
    scalars : (N,) array
        Per-vertex scalar values to display.
    cmap : str
        Matplotlib/PyVista colourmap name.
    clim : (vmin, vmax) or None
        Colour limits. If None, uses [5th, 95th] percentiles.
    title : str
        Figure title.
    view : ``'lateral'``, ``'medial'``, ``'dorsal'``, ``'ventral'``
        Camera viewpoint.
    figsize : (width, height) in inches (used for screenshot sizing).
    output_path : path or None
        If provided, saves a screenshot to this path.
    show : bool
        Whether to display the interactive window.

    Returns
    -------
    Figure or None
        If PyVista is unavailable, falls back to matplotlib.
    """
    try:
        import pyvista as pv
    except ImportError:
        logger.warning("PyVista not available; falling back to matplotlib.")
        return _plot_surface_matplotlib(
            vertices, faces, scalars, cmap, clim, title, output_path,
        )

    # Build PyVista mesh
    # Faces need to be prefixed with the number of vertices per face (3)
    pv_faces = np.column_stack([
        np.full(faces.shape[0], 3, dtype=np.int64), faces,
    ]).ravel()
    mesh = pv.PolyData(vertices, pv_faces)
    mesh.point_data["scalars"] = scalars

    # Colour limits
    if clim is None:
        valid = scalars[np.isfinite(scalars)]
        if len(valid) > 0:
            clim = (np.percentile(valid, 5), np.percentile(valid, 95))
        else:
            clim = (0, 1)

    # Camera positions
    camera_positions = {
        "lateral": "xy",
        "medial": "-xy",
        "dorsal": "xz",
        "ventral": "-xz",
    }

    # Render
    pl = pv.Plotter(off_screen=output_path is not None and not show)
    pl.add_mesh(
        mesh,
        scalars="scalars",
        cmap=cmap,
        clim=clim,
        smooth_shading=True,
        show_scalar_bar=True,
        scalar_bar_args={"title": title, "vertical": True},
    )
    pl.view_vector(camera_positions.get(view, "xy"))
    pl.add_text(title, font_size=12, position="upper_left")

    if output_path is not None:
        pl.screenshot(str(output_path), window_size=[
            int(figsize[0] * 100), int(figsize[1] * 100),
        ])
        logger.info("Saved surface plot to %s", output_path)

    if show:
        pl.show()

    return None


def plot_surprise_map(
    vertices: np.ndarray,
    faces: np.ndarray,
    surprise: np.ndarray,
    z_score: np.ndarray,
    title: str = "Surprise Map",
    output_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> None:
    """
    Multi-panel figure showing both surprise and z-score maps.
    """
    try:
        import pyvista as pv
    except ImportError:
        logger.warning("PyVista not available for multi-panel plot.")
        return

    pv_faces = np.column_stack([
        np.full(faces.shape[0], 3, dtype=np.int64), faces,
    ]).ravel()
    mesh = pv.PolyData(vertices, pv_faces)

    pl = pv.Plotter(shape=(1, 2), off_screen=output_path is not None and not show)

    # Left panel: surprise
    pl.subplot(0, 0)
    mesh_s = mesh.copy()
    mesh_s.point_data["surprise"] = surprise
    valid = surprise[np.isfinite(surprise)]
    clim_s = (np.percentile(valid, 5), np.percentile(valid, 95)) if len(valid) else (0, 5)
    pl.add_mesh(
        mesh_s, scalars="surprise", cmap=SURPRISE_CMAP, clim=clim_s,
        smooth_shading=True, show_scalar_bar=True,
        scalar_bar_args={"title": "Surprise", "vertical": True},
    )
    pl.add_text("Surprise (−log p)", font_size=10)

    # Right panel: z-score
    pl.subplot(0, 1)
    mesh_z = mesh.copy()
    mesh_z.point_data["z_score"] = z_score
    z_abs_max = np.percentile(np.abs(z_score[np.isfinite(z_score)]), 95)
    clim_z = (-z_abs_max, z_abs_max)
    pl.add_mesh(
        mesh_z, scalars="z_score", cmap=ZSCORE_CMAP, clim=clim_z,
        smooth_shading=True, show_scalar_bar=True,
        scalar_bar_args={"title": "z-score", "vertical": True},
    )
    pl.add_text("Z-score (deviation from norm)", font_size=10)

    pl.link_views()

    if output_path is not None:
        pl.screenshot(str(output_path), window_size=[2000, 800])
        logger.info("Saved surprise map figure to %s", output_path)

    if show:
        pl.show()


def plot_eigenspectrum(
    eigenvalues: np.ndarray,
    n_show: int = 100,
    title: str = "Laplace–Beltrami Eigenspectrum",
    output_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """
    Plot the LB eigenvalue spectrum (Weyl's law diagnostic).

    The eigenvalue growth rate should follow Weyl's law:
    λ_n ~ 4π n / Area  for a 2D surface, providing a sanity check
    on the mesh quality and eigendecomposition.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    evals = eigenvalues[:n_show]
    n = np.arange(len(evals))

    # Linear plot
    axes[0].plot(n, evals, "o-", markersize=2, linewidth=0.8, color="#2196F3")
    axes[0].set_xlabel("Eigenvalue index")
    axes[0].set_ylabel("λ")
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)

    # Log-log plot (should be approximately linear with slope ~1 by Weyl's law)
    mask = evals > 0
    axes[1].loglog(n[mask] + 1, evals[mask], "o", markersize=2, color="#F44336")
    axes[1].set_xlabel("Eigenvalue index (log)")
    axes[1].set_ylabel("λ (log)")
    axes[1].set_title("Log-log (Weyl's law check)")
    axes[1].grid(True, alpha=0.3, which="both")

    plt.tight_layout()

    if output_path is not None:
        fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
        logger.info("Saved eigenspectrum to %s", output_path)

    return fig


def plot_hks_multiscale(
    hks: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    scales_to_show: Optional[List[int]] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Multi-panel HKS visualization across time scales.
    """
    if scales_to_show is None:
        n_total = hks.shape[1]
        scales_to_show = [0, n_total // 4, n_total // 2, 3 * n_total // 4, n_total - 1]

    try:
        import pyvista as pv
    except ImportError:
        logger.warning("PyVista not available for HKS multi-scale plot.")
        return

    n_panels = len(scales_to_show)
    pv_faces = np.column_stack([
        np.full(faces.shape[0], 3, dtype=np.int64), faces,
    ]).ravel()

    pl = pv.Plotter(shape=(1, n_panels), off_screen=output_path is not None)

    for panel_idx, scale_idx in enumerate(scales_to_show):
        pl.subplot(0, panel_idx)
        mesh = pv.PolyData(vertices, pv_faces)
        vals = hks[:, scale_idx]
        mesh.point_data["hks"] = vals

        valid = vals[np.isfinite(vals)]
        clim = (np.percentile(valid, 2), np.percentile(valid, 98))

        pl.add_mesh(
            mesh, scalars="hks", cmap="magma", clim=clim,
            smooth_shading=True, show_scalar_bar=False,
        )
        pl.add_text(f"t = scale {scale_idx}", font_size=8)

    if output_path is not None:
        pl.screenshot(str(output_path), window_size=[n_panels * 400, 400])
        logger.info("Saved HKS multi-scale plot to %s", output_path)
    else:
        pl.show()


def plot_network_anomaly_profile(
    network_scores: Dict[str, Dict[str, float]],
    title: str = "Network Anomaly Profile",
    output_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """
    Bar chart of mean z-scores and surprise by functional network
    (e.g. Yeo-7), showing which networks are anomalous.
    """
    names = list(network_scores.keys())
    mean_z = [network_scores[n]["mean_z"] for n in names]
    frac_anom = [network_scores[n]["fraction_anomalous"] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Yeo-7 colours (approximate)
    yeo_colors = [
        "#9B59B6", "#E74C3C", "#2ECC71", "#8E44AD",
        "#F39C12", "#3498DB", "#1ABC9C",
    ]
    colors = yeo_colors[:len(names)] if len(names) <= 7 else None

    # Mean z-score
    bars = axes[0].barh(names, mean_z, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].axvline(-2, color="red", linestyle="--", alpha=0.5, label="z = ±2")
    axes[0].axvline(2, color="red", linestyle="--", alpha=0.5)
    axes[0].set_xlabel("Mean z-score")
    axes[0].set_title("Mean deviation from norm")
    axes[0].legend(fontsize=8)

    # Fraction anomalous
    axes[1].barh(names, frac_anom, color=colors, edgecolor="black", linewidth=0.5)
    axes[1].set_xlabel("Fraction of vertices |z| > 2")
    axes[1].set_title("Anomaly burden per network")
    axes[1].set_xlim(0, 1)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_path is not None:
        fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
        logger.info("Saved network anomaly profile to %s", output_path)

    return fig


# ── Matplotlib fallback ─────────────────────────────────────────────────

def _plot_surface_matplotlib(
    vertices, faces, scalars, cmap, clim, title, output_path,
):
    """Simple 2D projection when PyVista is unavailable."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sc = ax.scatter(
        vertices[:, 0], vertices[:, 1],
        c=scalars, cmap=cmap, s=0.1,
        vmin=clim[0] if clim else None,
        vmax=clim[1] if clim else None,
    )
    plt.colorbar(sc, ax=ax, label=title)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.axis("off")

    if output_path is not None:
        fig.savefig(str(output_path), dpi=300, bbox_inches="tight")

    return fig
