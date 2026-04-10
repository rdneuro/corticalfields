"""
Publication-quality visualization for subcortical and hippocampal structures.

This module provides a unified visualization API for all subcortical and
hippocampal analysis outputs from CorticalFields.  It targets figures suitable
for Nature, Brain, NeuroImage, Neurology, and Epilepsia.

Rendering backends
------------------
- **PyVista** (primary) — 3D surface rendering with scalar overlays, multi-view
  layouts, transparency, clipping planes, and glyph visualization.
- **matplotlib** — 2D projections, profile plots, bar charts, multi-panel
  figure composition.
- **surfplot / BrainSpace** — hippocampal surface plots (fold/unfold views).
- **hippunfold_plot** — HippUnfold-native plotting (nilearn/matplotlib engine).

Design patterns
---------------
Every plotting function returns a matplotlib Figure and optionally saves to
disk.  3D PyVista renders are captured as images via pl.screenshot() and
embedded into matplotlib subplots for compositing with 2D elements (colorbars,
bar charts, annotations).

Colour conventions
------------------
- Z-scores / effect sizes : RdBu_r (diverging, centred at 0)
- Morphometric values     : viridis / inferno (perceptually uniform)
- Subfield labels         : Okabe-Ito-based categorical palette
- Gradients               : Spectral / plasma / viridis_r
- P-values                : hot_r / YlOrRd (thresholded)
- Asymmetry               : PuOr (diverging, centred at 0)

References
----------
[1] Jiang, C. et al. Nature Communications (2024) — figure style target.
[2] DeKraker, J. et al. eLife (2022) — hippocampal fold/unfold convention.
[3] Vos de Wael, R. et al. Communications Biology (2020) — BrainSpace.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Publication style configuration
# ═══════════════════════════════════════════════════════════════════════════

# Colorblind-accessible Okabe-Ito palette
OKABE_ITO = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442",
    "#0072B2", "#D55E00", "#CC79A7", "#000000",
]

# Subfield colours (consistent with hippocampus.py)
SUBFIELD_CMAP = {
    "subiculum": "#E69F00",
    "CA1": "#56B4E9",
    "CA2": "#009E73",
    "CA3": "#F0E442",
    "CA4/DG": "#0072B2",
    "SRLM": "#CC79A7",
}

# Default colormaps by data type
CMAP_ZSCORE = "RdBu_r"
CMAP_MORPHO = "viridis"
CMAP_GRADIENT = "Spectral"
CMAP_ASYM = "PuOr"
CMAP_PVAL = "hot_r"
CMAP_SUBFIELDS = "tab10"

# PyVista camera positions for subcortical multi-view layouts
_SUBCORTICAL_VIEWS = {
    "anterior":  (0, 1, 0),
    "posterior": (0, -1, 0),
    "lateral_l": (-1, 0, 0),
    "lateral_r": (1, 0, 0),
    "superior":  (0, 0, 1),
    "inferior":  (0, 0, -1),
}


def _apply_pub_style() -> None:
    """Apply publication-quality matplotlib defaults."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# ═══════════════════════════════════════════════════════════════════════════
# PyVista 3D rendering helpers
# ═══════════════════════════════════════════════════════════════════════════


def _surf_to_pyvista(surf):
    """Convert a SubcorticalSurface or HippocampalSurface to PyVista PolyData."""
    import pyvista as pv
    faces_pv = np.hstack([
        np.full((surf.n_faces, 1), 3, dtype=np.int64), surf.faces,
    ])
    mesh = pv.PolyData(surf.vertices.astype(np.float32), faces_pv)
    for name, vals in surf.overlays.items():
        mesh.point_data[name] = vals.astype(np.float32)
    return mesh


def _render_pyvista_view(
    mesh,
    scalars: Optional[str] = None,
    scalar_array: Optional[np.ndarray] = None,
    cmap: str = "viridis",
    clim: Optional[Tuple[float, float]] = None,
    camera_position: Optional[Any] = None,
    window_size: Tuple[int, int] = (800, 600),
    background: str = "white",
    show_edges: bool = False,
    opacity: float = 1.0,
    nan_color: str = "lightgrey",
    scalar_bar: bool = False,
) -> np.ndarray:
    """Render a single PyVista view and return as RGB image array."""
    import pyvista as pv
    pv.OFF_SCREEN = True

    pl = pv.Plotter(off_screen=True, window_size=window_size)
    pl.set_background(background)

    kwargs = dict(
        cmap=cmap, show_edges=show_edges, opacity=opacity,
        nan_color=nan_color, show_scalar_bar=scalar_bar,
        smooth_shading=True, ambient=0.2, diffuse=0.8, specular=0.3,
    )

    if scalar_array is not None:
        mesh.point_data["_scalar"] = scalar_array.astype(np.float32)
        kwargs["scalars"] = "_scalar"
    elif scalars is not None:
        kwargs["scalars"] = scalars

    if clim is not None:
        kwargs["clim"] = clim

    pl.add_mesh(mesh, **kwargs)

    if camera_position is not None:
        pl.camera_position = camera_position

    img = pl.screenshot(return_img=True)
    pl.close()
    return img


# ═══════════════════════════════════════════════════════════════════════════
# Subcortical multi-view rendering
# ═══════════════════════════════════════════════════════════════════════════


def plot_subcortical_multiview(
    surf,
    overlay: Optional[Union[str, np.ndarray]] = None,
    *,
    views: Sequence[str] = ("anterior", "lateral_l", "superior", "posterior"),
    cmap: str = CMAP_MORPHO,
    clim: Optional[Tuple[float, float]] = None,
    title: str = "",
    cbar_label: str = "",
    figsize: Tuple[float, float] = (12, 3),
    dpi: int = 300,
    output_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """
    Multi-view rendering of a subcortical surface with scalar overlay.

    Generates 4+ viewpoints of the structure using PyVista and composes
    them into a single matplotlib figure with a shared colourbar.

    Parameters
    ----------
    surf : SubcorticalSurface or HippocampalSurface
    overlay : str (overlay name) or (N,) array, or None
    views : sequence of view names
    cmap : str
    clim : (vmin, vmax) or None
    title : str
    cbar_label : str
    figsize : (width, height) inches
    dpi : int
    output_path : path or None

    Returns
    -------
    Figure
    """
    _apply_pub_style()
    mesh = _surf_to_pyvista(surf)

    # Resolve overlay
    scalar_array = None
    if isinstance(overlay, str):
        scalar_array = surf.get_overlay(overlay)
        if not cbar_label:
            cbar_label = overlay
    elif isinstance(overlay, np.ndarray):
        scalar_array = overlay

    if clim is None and scalar_array is not None:
        vmin = float(np.nanpercentile(scalar_array, 2))
        vmax = float(np.nanpercentile(scalar_array, 98))
        clim = (vmin, vmax)

    n_views = len(views)
    fig = plt.figure(figsize=figsize, facecolor="white")
    gs = gridspec.GridSpec(
        2, n_views, height_ratios=[1, 0.05], hspace=0.02, wspace=0.02,
    )

    # Compute camera positions based on mesh centroid
    centre = surf.centroid
    extent = np.linalg.norm(surf.vertices.max(axis=0) - surf.vertices.min(axis=0))
    cam_dist = extent * 2.5

    for i, view_name in enumerate(views):
        direction = np.array(_SUBCORTICAL_VIEWS.get(
            view_name, (0, 1, 0)
        ), dtype=float)
        cam_pos = centre + direction * cam_dist
        camera = [cam_pos.tolist(), centre.tolist(), [0, 0, 1]]

        img = _render_pyvista_view(
            mesh, scalar_array=scalar_array,
            cmap=cmap, clim=clim, camera_position=camera,
            window_size=(600, 500),
        )

        ax = fig.add_subplot(gs[0, i])
        ax.imshow(img)
        ax.set_title(view_name.replace("_", " ").title(), fontsize=7)
        ax.axis("off")

    # Colourbar
    if scalar_array is not None and clim is not None:
        cbar_ax = fig.add_subplot(gs[1, :])
        norm = Normalize(vmin=clim[0], vmax=clim[1])
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
        cbar.set_label(cbar_label, fontsize=8)
        cbar.ax.tick_params(labelsize=6)

    if title:
        fig.suptitle(title, fontsize=10, fontweight="bold", y=1.02)

    if output_path:
        fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight", facecolor="white")
        logger.info("Saved figure → %s", output_path)

    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Multi-structure composite (basal ganglia, limbic, etc.)
# ═══════════════════════════════════════════════════════════════════════════


def plot_subcortical_composite(
    surfaces: Dict[str, "SubcorticalSurface"],
    *,
    overlay_name: Optional[str] = None,
    cmap: str = CMAP_ZSCORE,
    clim: Optional[Tuple[float, float]] = None,
    structure_colors: Optional[Dict[str, str]] = None,
    opacity: float = 0.85,
    figsize: Tuple[float, float] = (10, 6),
    title: str = "Subcortical structures",
    output_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> Figure:
    """
    Composite rendering of multiple subcortical structures.

    Shows multiple structures simultaneously with per-structure colours
    or a shared scalar overlay.

    Parameters
    ----------
    surfaces : dict[name, SubcorticalSurface]
    overlay_name : str or None
        If given, colour each structure by this shared overlay.
        If None, use categorical structure_colors.
    cmap : str
    clim : (vmin, vmax) or None
    structure_colors : dict[name, hex_colour] or None
    opacity : float
    figsize, title, output_path, dpi : as usual

    Returns
    -------
    Figure
    """
    import pyvista as pv
    pv.OFF_SCREEN = True

    _apply_pub_style()

    # Default categorical colours
    if structure_colors is None:
        structure_colors = {}
        for i, name in enumerate(surfaces.keys()):
            structure_colors[name] = OKABE_ITO[i % len(OKABE_ITO)]

    # Render two views: anterior and lateral
    views = [
        ("Anterior", (0, 1, 0)),
        ("Left lateral", (-1, 0, 0)),
    ]

    fig, axes = plt.subplots(1, len(views), figsize=figsize, facecolor="white")
    if len(views) == 1:
        axes = [axes]

    for ax_idx, (view_label, direction) in enumerate(views):
        pl = pv.Plotter(off_screen=True, window_size=(800, 600))
        pl.set_background("white")

        # Compute combined centroid for camera positioning
        all_verts = np.vstack([s.vertices for s in surfaces.values()])
        centre = all_verts.mean(axis=0)
        extent = np.linalg.norm(all_verts.max(axis=0) - all_verts.min(axis=0))

        for name, surf in surfaces.items():
            mesh = _surf_to_pyvista(surf)
            if overlay_name is not None and overlay_name in surf.overlays:
                pl.add_mesh(
                    mesh, scalars=overlay_name, cmap=cmap, clim=clim,
                    opacity=opacity, smooth_shading=True,
                    show_scalar_bar=False,
                )
            else:
                pl.add_mesh(
                    mesh, color=structure_colors.get(name, "grey"),
                    opacity=opacity, smooth_shading=True,
                )

        cam_pos = centre + np.array(direction, dtype=float) * extent * 2.5
        pl.camera_position = [cam_pos.tolist(), centre.tolist(), [0, 0, 1]]

        img = pl.screenshot(return_img=True)
        pl.close()

        axes[ax_idx].imshow(img)
        axes[ax_idx].set_title(view_label, fontsize=8)
        axes[ax_idx].axis("off")

    fig.suptitle(title, fontsize=10, fontweight="bold")
    plt.tight_layout()

    if output_path:
        fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight", facecolor="white")

    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Hippocampal fold/unfold dual-view
# ═══════════════════════════════════════════════════════════════════════════


def plot_hippocampal_foldunfold(
    surf,
    overlay: Optional[Union[str, np.ndarray]] = None,
    *,
    cmap: str = CMAP_MORPHO,
    clim: Optional[Tuple[float, float]] = None,
    cbar_label: str = "",
    title: str = "",
    show_subfield_boundaries: bool = True,
    figsize: Tuple[float, float] = (10, 5),
    output_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> Figure:
    """
    Hippocampal fold + unfold dual-view figure.

    Left panel shows the folded (3D anatomical) surface from a dorsal view.
    Right panel shows the unfolded (flat map) surface with subfield boundaries.
    This is the standard figure format for hippocampal publications.

    Parameters
    ----------
    surf : HippocampalSurface
        Must have unfolded_vertices if showing the unfolded view.
    overlay : str or (N,) array
    cmap : str
    clim : (vmin, vmax) or None
    cbar_label : str
    title : str
    show_subfield_boundaries : bool
    figsize, output_path, dpi : as usual

    Returns
    -------
    Figure
    """
    _apply_pub_style()

    # Resolve overlay
    scalar_array = None
    if isinstance(overlay, str):
        scalar_array = surf.get_overlay(overlay)
        if not cbar_label:
            cbar_label = overlay
    elif isinstance(overlay, np.ndarray):
        scalar_array = overlay

    if clim is None and scalar_array is not None:
        vmin = float(np.nanpercentile(scalar_array[np.isfinite(scalar_array)], 2))
        vmax = float(np.nanpercentile(scalar_array[np.isfinite(scalar_array)], 98))
        clim = (vmin, vmax)

    fig = plt.figure(figsize=figsize, facecolor="white")
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 0.05], hspace=0.15, wspace=0.05)

    # ── Left panel: folded (3D) view ──
    mesh_folded = _surf_to_pyvista(surf)
    img_folded = _render_pyvista_view(
        mesh_folded, scalar_array=scalar_array,
        cmap=cmap, clim=clim,
        camera_position="yz",
        window_size=(700, 600),
    )

    ax_folded = fig.add_subplot(gs[0, 0])
    ax_folded.imshow(img_folded)
    ax_folded.set_title("Folded (3D)", fontsize=8)
    ax_folded.axis("off")

    # ── Right panel: unfolded (flat) view ──
    has_unfolded = hasattr(surf, "unfolded_vertices") and surf.unfolded_vertices is not None

    if has_unfolded:
        # Create a new mesh with unfolded coordinates
        import pyvista as pv
        faces_pv = np.hstack([
            np.full((surf.n_faces, 1), 3, dtype=np.int64), surf.faces,
        ])
        mesh_unfolded = pv.PolyData(
            surf.unfolded_vertices.astype(np.float32), faces_pv,
        )
        if scalar_array is not None:
            mesh_unfolded.point_data["_scalar"] = scalar_array.astype(np.float32)

        img_unfolded = _render_pyvista_view(
            mesh_unfolded,
            scalar_array=scalar_array,
            cmap=cmap, clim=clim,
            camera_position="xy",
            window_size=(700, 600),
        )

        ax_unfolded = fig.add_subplot(gs[0, 1])
        ax_unfolded.imshow(img_unfolded)
        ax_unfolded.set_title("Unfolded (flat map)", fontsize=8)
        ax_unfolded.axis("off")
    else:
        # If no unfolded vertices, show a 2D scatter of AP × PD
        ax_unfolded = fig.add_subplot(gs[0, 1])
        if hasattr(surf, "ap_coord") and surf.ap_coord is not None:
            sc = ax_unfolded.scatter(
                surf.ap_coord, surf.pd_coord if surf.pd_coord is not None else np.zeros_like(surf.ap_coord),
                c=scalar_array if scalar_array is not None else "grey",
                cmap=cmap, s=0.5, rasterized=True,
            )
            if clim is not None and scalar_array is not None:
                sc.set_clim(*clim)
            ax_unfolded.set_xlabel("AP coordinate", fontsize=7)
            ax_unfolded.set_ylabel("PD coordinate", fontsize=7)
            ax_unfolded.set_title("AP × PD space", fontsize=8)
        else:
            ax_unfolded.text(
                0.5, 0.5, "Unfolded surface\nnot available",
                ha="center", va="center", fontsize=9, color="grey",
                transform=ax_unfolded.transAxes,
            )
            ax_unfolded.axis("off")

    # Colourbar
    if scalar_array is not None and clim is not None:
        cbar_ax = fig.add_subplot(gs[1, :])
        norm = Normalize(vmin=clim[0], vmax=clim[1])
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
        cbar.set_label(cbar_label, fontsize=8)
        cbar.ax.tick_params(labelsize=6)

    if title:
        fig.suptitle(title, fontsize=10, fontweight="bold", y=1.02)

    if output_path:
        fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight", facecolor="white")
        logger.info("Saved figure → %s", output_path)

    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Hippocampal ipsi vs contra comparison
# ═══════════════════════════════════════════════════════════════════════════


def plot_hippocampal_comparison(
    ipsi,
    contra,
    overlay_name: str = "thickness",
    *,
    cmap: str = CMAP_MORPHO,
    clim: Optional[Tuple[float, float]] = None,
    title: str = "Hippocampal comparison: ipsi vs contra",
    figsize: Tuple[float, float] = (12, 5),
    output_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> Figure:
    """
    Side-by-side ipsilateral vs contralateral hippocampal comparison.

    Standard MTLE-HS figure layout showing both hippocampi with
    identical colour scales for direct visual comparison.

    Parameters
    ----------
    ipsi, contra : HippocampalSurface
    overlay_name : str
    cmap : str
    clim : (vmin, vmax) or None
    title, figsize, output_path, dpi : as usual

    Returns
    -------
    Figure
    """
    _apply_pub_style()

    scalar_ipsi = ipsi.get_overlay(overlay_name) if overlay_name in ipsi.overlays else None
    scalar_contra = contra.get_overlay(overlay_name) if overlay_name in contra.overlays else None

    if clim is None:
        all_vals = []
        if scalar_ipsi is not None:
            all_vals.append(scalar_ipsi)
        if scalar_contra is not None:
            all_vals.append(scalar_contra)
        if all_vals:
            combined = np.concatenate(all_vals)
            valid = combined[np.isfinite(combined)]
            clim = (float(np.percentile(valid, 2)), float(np.percentile(valid, 98)))

    fig = plt.figure(figsize=figsize, facecolor="white")
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 0.05], hspace=0.15, wspace=0.05)

    # Render both hemispheres
    for idx, (surf, label, scalar) in enumerate([
        (ipsi, f"Ipsilateral ({ipsi.hemi.upper()})", scalar_ipsi),
        (contra, f"Contralateral ({contra.hemi.upper()})", scalar_contra),
    ]):
        mesh = _surf_to_pyvista(surf)
        img = _render_pyvista_view(
            mesh, scalar_array=scalar,
            cmap=cmap, clim=clim,
            camera_position="yz",
            window_size=(700, 600),
        )
        ax = fig.add_subplot(gs[0, idx])
        ax.imshow(img)
        ax.set_title(label, fontsize=8)
        ax.axis("off")

    # Shared colourbar
    if clim is not None:
        cbar_ax = fig.add_subplot(gs[1, :])
        norm = Normalize(vmin=clim[0], vmax=clim[1])
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
        cbar.set_label(overlay_name, fontsize=8)
        cbar.ax.tick_params(labelsize=6)

    fig.suptitle(title, fontsize=10, fontweight="bold", y=1.02)

    if output_path:
        fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight", facecolor="white")

    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Subfield bar chart
# ═══════════════════════════════════════════════════════════════════════════


def plot_subfield_metrics(
    metrics: Dict[str, float],
    *,
    ylabel: str = "Surface area (mm²)",
    title: str = "Hippocampal subfield metrics",
    error_bars: Optional[Dict[str, float]] = None,
    reference: Optional[Dict[str, float]] = None,
    figsize: Tuple[float, float] = (5, 3.5),
    output_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> Figure:
    """
    Bar chart of per-subfield metrics.

    Parameters
    ----------
    metrics : dict[subfield_name, float]
    ylabel : str
    title : str
    error_bars : dict[subfield_name, float] or None
    reference : dict[subfield_name, float] or None
        Normative reference values shown as horizontal lines.
    figsize, output_path, dpi : as usual

    Returns
    -------
    Figure
    """
    _apply_pub_style()

    names = [n for n in SUBFIELD_CMAP if n in metrics]
    values = [metrics[n] for n in names]
    colors = [SUBFIELD_CMAP[n] for n in names]

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(names))

    yerr = [error_bars.get(n, 0) for n in names] if error_bars else None
    ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.5,
           yerr=yerr, capsize=3)

    if reference:
        for i, name in enumerate(names):
            if name in reference:
                ax.hlines(reference[name], i - 0.35, i + 0.35,
                          colors="black", linestyles="--", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold")
    plt.tight_layout()

    if output_path:
        fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight", facecolor="white")

    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Subfield asymmetry chart
# ═══════════════════════════════════════════════════════════════════════════


def plot_subfield_asymmetry(
    asymmetry: Dict[str, float],
    *,
    title: str = "Subfield asymmetry index",
    threshold: float = 0.10,
    figsize: Tuple[float, float] = (5, 3.5),
    output_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> Figure:
    """
    Horizontal bar chart of per-subfield asymmetry indices.

    Parameters
    ----------
    asymmetry : dict[subfield_name, float]
        AI values (positive = ipsilateral larger).
    threshold : float
        AI threshold for clinical significance (shown as vertical lines).
    figsize, title, output_path, dpi : as usual

    Returns
    -------
    Figure
    """
    _apply_pub_style()

    names = [n for n in SUBFIELD_CMAP if n in asymmetry]
    values = [asymmetry[n] for n in names]

    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(names))

    # Colour bars by direction
    colors = ["#D55E00" if v < 0 else "#56B4E9" for v in values]
    ax.barh(y, values, color=colors, edgecolor="black", linewidth=0.5, height=0.6)

    # Threshold lines
    ax.axvline(threshold, color="grey", linestyle="--", linewidth=0.7, alpha=0.7)
    ax.axvline(-threshold, color="grey", linestyle="--", linewidth=0.7, alpha=0.7)
    ax.axvline(0, color="black", linewidth=0.8)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Asymmetry index", fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()

    if output_path:
        fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight", facecolor="white")

    return fig


# ═══════════════════════════════════════════════════════════════════════════
# AP / PD profile plots
# ═══════════════════════════════════════════════════════════════════════════


def plot_ap_profile(
    bin_centres: np.ndarray,
    profiles: Dict[str, np.ndarray],
    *,
    ylabel: str = "Thickness (mm)",
    title: str = "Anterior-posterior thickness profile",
    fill_between: bool = True,
    figsize: Tuple[float, float] = (6, 3),
    output_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> Figure:
    """
    Line plot of morphometric values along the hippocampal AP axis.

    Parameters
    ----------
    bin_centres : (n_bins,) array
    profiles : dict[label, (n_bins,) array]
        E.g. {'ipsi': ..., 'contra': ...} or {'patient': ..., 'normative': ...}.
    ylabel, title : str
    fill_between : bool
        If True, fills the area between paired profiles.
    figsize, output_path, dpi : as usual

    Returns
    -------
    Figure
    """
    _apply_pub_style()

    fig, ax = plt.subplots(figsize=figsize)

    labels = list(profiles.keys())
    for i, (label, vals) in enumerate(profiles.items()):
        color = OKABE_ITO[i % len(OKABE_ITO)]
        ax.plot(bin_centres, vals, "-o", color=color, markersize=3,
                linewidth=1.5, label=label)

    if fill_between and len(labels) == 2:
        v1 = profiles[labels[0]]
        v2 = profiles[labels[1]]
        ax.fill_between(bin_centres, v1, v2, alpha=0.15, color="grey")

    ax.set_xlabel("AP coordinate (0=anterior, 1=posterior)", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.legend(frameon=False, fontsize=7)
    ax.set_xlim(0, 1)
    plt.tight_layout()

    if output_path:
        fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight", facecolor="white")

    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Z-score surface map
# ═══════════════════════════════════════════════════════════════════════════


def plot_zscore_surface(
    surf,
    z_scores: np.ndarray,
    *,
    z_threshold: float = 1.96,
    cmap: str = CMAP_ZSCORE,
    clim: Tuple[float, float] = (-4.0, 4.0),
    title: str = "Z-score map",
    cbar_label: str = "Z-score",
    views: Sequence[str] = ("anterior", "lateral_l", "superior"),
    figsize: Tuple[float, float] = (12, 3.5),
    output_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> Figure:
    """
    Z-score map on a subcortical/hippocampal surface.

    Non-significant vertices (|z| < threshold) are rendered in transparent
    grey.  Significant vertices use a diverging colourmap.

    Parameters
    ----------
    surf : SubcorticalSurface or HippocampalSurface
    z_scores : (N,) array
    z_threshold : float
        Significance threshold.
    cmap, clim : colourmap and limits
    title, cbar_label : str
    views : view names
    figsize, output_path, dpi : as usual

    Returns
    -------
    Figure
    """
    # Mask non-significant vertices as NaN
    z_display = z_scores.copy()
    z_display[np.abs(z_display) < z_threshold] = np.nan

    return plot_subcortical_multiview(
        surf, overlay=z_display,
        views=views, cmap=cmap, clim=clim,
        title=title, cbar_label=cbar_label,
        figsize=figsize, dpi=dpi, output_path=output_path,
    )


# ═══════════════════════════════════════════════════════════════════════════
# ShapeDNA / spectral comparison
# ═══════════════════════════════════════════════════════════════════════════


def plot_shapedna_comparison(
    dna_dict: Dict[str, np.ndarray],
    *,
    title: str = "ShapeDNA spectral fingerprint",
    ylabel: str = "Normalised eigenvalue",
    figsize: Tuple[float, float] = (6, 3),
    output_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> Figure:
    """
    Line plot comparing ShapeDNA fingerprints across subjects/groups.

    Parameters
    ----------
    dna_dict : dict[label, (n_eigenvalues,) array]
    title, ylabel : str
    figsize, output_path, dpi : as usual

    Returns
    -------
    Figure
    """
    _apply_pub_style()
    fig, ax = plt.subplots(figsize=figsize)

    for i, (label, dna) in enumerate(dna_dict.items()):
        color = OKABE_ITO[i % len(OKABE_ITO)]
        ax.plot(np.arange(1, len(dna) + 1), dna, "-", color=color,
                linewidth=1.2, label=label, alpha=0.8)

    ax.set_xlabel("Eigenvalue index", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.legend(frameon=False, fontsize=7)
    plt.tight_layout()

    if output_path:
        fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight", facecolor="white")

    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Point cloud scatter (3D)
# ═══════════════════════════════════════════════════════════════════════════


def plot_point_cloud_3d(
    surf,
    overlay: Optional[Union[str, np.ndarray]] = None,
    *,
    cmap: str = CMAP_MORPHO,
    point_size: float = 1.0,
    title: str = "",
    figsize: Tuple[float, float] = (8, 6),
    output_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> Figure:
    """
    3D scatter plot of surface vertices coloured by an overlay.

    Uses matplotlib 3D projection for static publication figures.
    Useful for point cloud visualization and quick quality checks.

    Parameters
    ----------
    surf : SubcorticalSurface or HippocampalSurface
    overlay : str or (N,) array
    cmap, point_size : as usual
    title, figsize, output_path, dpi : as usual

    Returns
    -------
    Figure
    """
    _apply_pub_style()

    scalar_array = None
    if isinstance(overlay, str):
        scalar_array = surf.get_overlay(overlay)
    elif isinstance(overlay, np.ndarray):
        scalar_array = overlay

    fig = plt.figure(figsize=figsize, facecolor="white")
    ax = fig.add_subplot(111, projection="3d")

    v = surf.vertices
    kwargs = dict(s=point_size, alpha=0.6, rasterized=True)
    if scalar_array is not None:
        sc = ax.scatter(v[:, 0], v[:, 1], v[:, 2], c=scalar_array,
                        cmap=cmap, **kwargs)
        fig.colorbar(sc, ax=ax, shrink=0.6, label=overlay if isinstance(overlay, str) else "")
    else:
        ax.scatter(v[:, 0], v[:, 1], v[:, 2], c="steelblue", **kwargs)

    ax.set_xlabel("X (mm)", fontsize=7)
    ax.set_ylabel("Y (mm)", fontsize=7)
    ax.set_zlabel("Z (mm)", fontsize=7)
    ax.tick_params(labelsize=6)

    if title:
        ax.set_title(title, fontsize=9, fontweight="bold")

    # Equal aspect ratio
    max_range = (v.max(axis=0) - v.min(axis=0)).max() / 2
    mid = v.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    if output_path:
        fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight", facecolor="white")

    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Utility: save figure with multiple formats
# ═══════════════════════════════════════════════════════════════════════════


def save_figure(
    fig: Figure,
    path: Union[str, Path],
    *,
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 300,
) -> None:
    """
    Save a figure in multiple formats for publication.

    Parameters
    ----------
    fig : Figure
    path : path-like
        Base path (without extension).
    formats : sequence of str
        File formats to save ('png', 'pdf', 'svg', 'tiff').
    dpi : int
    """
    path = Path(path)
    for fmt in formats:
        out = path.with_suffix(f".{fmt}")
        fig.savefig(str(out), dpi=dpi, bbox_inches="tight", facecolor="white")
        logger.info("Saved → %s", out)
