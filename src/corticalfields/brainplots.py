"""
Publication-grade brain visualization for CorticalFields.

Renders cortical surfaces, connectivity graphs, functional map matrices,
asymmetry maps, and composite multi-panel figures at journal quality
(Nature, Science, PNAS, Brain, NeuroImage). Three rendering tiers:

  1. **PyVista** (primary) — 3D cortical surfaces with PBR shading,
     SSAO for sulcal contrast, and three-point lighting. Produces
     screenshots composited into matplotlib for panel labelling.
  2. **nilearn** — glass brain connectomes and volume overlays.
  3. **matplotlib** — adjacency matrices, radar plots, eigenspectra,
     chord diagrams, and the final compositing layer for all panels.

Every function returns a ``matplotlib.figure.Figure`` that can be
further customised or saved with :func:`save_figure`.

Design principles
-----------------
- White background, Arial/Helvetica font, ≥7 pt at print size
- Colorblind-safe: Okabe-Ito for categorical, ``cividis``/``RdBu_r`` for sequential/diverging
- Panel labels: bold uppercase (A, B, C) for PNAS/Brain; lowercase (a, b, c) for Nature
- Shared colorbars for group comparisons centred on the right
- Automatic figure sizing for single-column (89 mm) or double-column (183 mm)

Dependencies
------------
- Required: numpy, matplotlib, scipy
- Surface plots: pyvista (≥0.43)
- Connectomes: nilearn
- Graph layouts: netgraph (optional, falls back to networkx)
- Chord diagrams: mpl_chord_diagram (optional)

Notes
-----
This module **adds** to the existing ``viz.py`` without replacing it.
``viz.py`` provides quick-look PyVista windows; ``brainplots.py`` produces
finalised, composited figures for manuscripts.
"""

from __future__ import annotations

import logging
import warnings
from io import BytesIO
from pathlib import Path
from string import ascii_uppercase, ascii_lowercase
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize, TwoSlopeNorm
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, leaves_list

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

# ── Journal specifications (width in inches) ──────────────────────────────
JOURNAL_SPECS = {
    "nature":      {"single": 3.50, "double": 7.20, "panel_label": "lower"},
    "science":     {"single": 2.24, "double": 6.89, "panel_label": "upper"},
    "pnas":        {"single": 3.42, "double": 7.00, "panel_label": "upper"},
    "brain":       {"single": 3.54, "double": 7.28, "panel_label": "upper"},
    "neuroimage":  {"single": 3.54, "double": 7.48, "panel_label": "upper"},
}

# ── Yeo-7 network colours (Okabe-Ito, colorblind-safe) ───────────────────
YEO7_COLORS = {
    "Visual":          "#D55E00",  # Vermillion
    "Somatomotor":     "#56B4E9",  # Sky blue
    "DorsAttn":        "#009E73",  # Bluish green
    "VentAttn":        "#CC79A7",  # Reddish purple
    "Limbic":          "#F0E442",  # Yellow
    "Frontoparietal":  "#E69F00",  # Orange
    "Default":         "#0072B2",  # Blue
}
YEO7_NAMES = list(YEO7_COLORS.keys())
YEO7_HEX = list(YEO7_COLORS.values())

# ── FreeSurfer subcortical colours ────────────────────────────────────────
SUBCORT_COLORS = {
    "Hippocampus": "#DCD814",
    "Amygdala":    "#67FFFF",
    "Thalamus":    "#00760E",
    "Caudate":     "#7ABADC",
    "Putamen":     "#EC0DB0",
    "Pallidum":    "#0C30FF",
    "Accumbens":   "#FFA500",
}

# ── Colormaps ─────────────────────────────────────────────────────────────
CMAP_DIVERGING   = "RdBu_r"      # z-scores, asymmetry, FC
CMAP_SEQUENTIAL  = "YlOrRd"      # surprise, p-values
CMAP_UNIFORM     = "cividis"     # generic sequential, max CVD safety
CMAP_CURVATURE   = "gray"        # sulcal depth underlay
CMAP_SPECTRAL    = "magma"       # HKS, WKS, GPS
CMAP_MATRIX      = "RdBu_r"     # connectivity matrices
CMAP_DISTANCE    = "viridis"     # distance matrices

# ── PyVista camera positions (FreeSurfer RAS) ─────────────────────────────
_D = 350  # camera distance; adjust for individual meshes
CAMERA_POSITIONS = {
    "lateral_left":   [(-_D, 0, 0), (0, 0, 0), (0, 0, 1)],
    "lateral_right":  [( _D, 0, 0), (0, 0, 0), (0, 0, 1)],
    "medial_left":    [( _D, 0, 0), (0, 0, 0), (0, 0, 1)],
    "medial_right":   [(-_D, 0, 0), (0, 0, 0), (0, 0, 1)],
    "dorsal":         [(0, 0,  _D), (0, 0, 0), (0, 1, 0)],
    "ventral":        [(0, 0, -_D), (0, 0, 0), (0, 1, 0)],
    "anterior":       [(0,  _D, 0), (0, 0, 0), (0, 0, 1)],
    "posterior":      [(0, -_D, 0), (0, 0, 0), (0, 0, 1)],
}


# ═══════════════════════════════════════════════════════════════════════════
# Publication style setup
# ═══════════════════════════════════════════════════════════════════════════


def _setup_style(journal: str = "nature") -> None:
    """
    Configure matplotlib rcParams for publication figures.

    Sets Arial/Helvetica font, appropriate sizes for the target journal,
    white background, and clean spines.
    """
    plt.rcdefaults()
    matplotlib.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":          8,
        "axes.titlesize":     9,
        "axes.labelsize":     8,
        "xtick.labelsize":    7,
        "ytick.labelsize":    7,
        "legend.fontsize":    7,
        "figure.titlesize":   10,
        "axes.linewidth":     0.6,
        "xtick.major.width":  0.6,
        "ytick.major.width":  0.6,
        "xtick.major.size":   3,
        "ytick.major.size":   3,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
        "savefig.facecolor":  "white",
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype":       42,   # TrueType (editable in Illustrator)
        "ps.fonttype":        42,
        "mathtext.default":   "regular",
    })


# ═══════════════════════════════════════════════════════════════════════════
# PyVista utilities
# ═══════════════════════════════════════════════════════════════════════════


def _make_pv_mesh(vertices: np.ndarray, faces: np.ndarray) -> Any:
    """
    Convert NumPy vertices + faces to a PyVista PolyData mesh.

    Uses ``pv.make_tri_mesh()`` when available (≥0.43), falling back
    to manual VTK face padding for older versions.
    """
    import pyvista as pv

    if hasattr(pv, "make_tri_mesh"):
        return pv.make_tri_mesh(vertices, faces)
    # Fallback: manual face padding
    n_faces = faces.shape[0]
    pv_faces = np.column_stack([
        np.full(n_faces, 3, dtype=np.int64), faces,
    ]).ravel()
    return pv.PolyData(np.asarray(vertices, dtype=np.float64), pv_faces)


def _render_pv_view(
    mesh: Any,
    scalars: np.ndarray,
    cmap: str,
    clim: Tuple[float, float],
    camera_position: list,
    window_size: Tuple[int, int] = (800, 700),
    curvature: Optional[np.ndarray] = None,
    threshold: Optional[float] = None,
    show_scalar_bar: bool = False,
    scalar_bar_args: Optional[dict] = None,
    use_pbr: bool = True,
    ssao: bool = True,
) -> np.ndarray:
    """
    Render a single PyVista view and return the RGBA screenshot as an array.

    Parameters
    ----------
    mesh : pv.PolyData
        Brain surface mesh.
    scalars : np.ndarray, shape (N,)
        Per-vertex values.
    cmap : str
        Colormap name.
    clim : (vmin, vmax)
        Color range.
    camera_position : list of 3 tuples
        [(camera_xyz), (focal_point), (view_up)].
    window_size : (w, h)
        Render window size in pixels.
    curvature : np.ndarray or None
        If provided, renders sulcal depth underlay (binary sign).
    threshold : float or None
        If set, voxels with |value| < threshold become transparent.
    show_scalar_bar : bool
        Whether to include a color bar in this view.
    scalar_bar_args : dict or None
        Scalar bar customization passed to ``add_mesh``.
    use_pbr : bool
        Use physically-based rendering for realistic shading.
    ssao : bool
        Enable screen-space ambient occlusion for sulcal contrast.

    Returns
    -------
    img : np.ndarray, shape (H, W, 4)
        RGBA screenshot.
    """
    import pyvista as pv

    pv.global_theme.transparent_background = True
    pl = pv.Plotter(off_screen=True, window_size=window_size)
    pl.set_background("white")

    # Three-point lighting for sulcal clarity
    pl.remove_all_lights()
    pl.add_light(pv.Light(position=(200, 200, 300), intensity=1.0,
                          light_type="scene light"))
    pl.add_light(pv.Light(position=(-200, 100, 100), intensity=0.4,
                          light_type="scene light"))
    pl.add_light(pv.Light(position=(0, -200, 200), intensity=0.3,
                          light_type="scene light"))

    # Layer 1: sulcal curvature underlay
    if curvature is not None:
        curv_mesh = mesh.copy()
        curv_mesh.point_data["curv"] = np.sign(curvature)
        pl.add_mesh(curv_mesh, scalars="curv", cmap=CMAP_CURVATURE,
                    clim=[-1, 1], show_scalar_bar=False,
                    smooth_shading=True)

    # Layer 2: statistical overlay
    stat_mesh = mesh.copy()
    stat_mesh.point_data["data"] = scalars

    # Transparency mask for thresholded maps
    opacity = None
    if threshold is not None:
        opacity = np.where(np.abs(scalars) > threshold, 1.0, 0.0)

    sbar = scalar_bar_args or {}
    pl.add_mesh(
        stat_mesh,
        scalars="data",
        cmap=cmap,
        clim=clim,
        smooth_shading=True,
        pbr=use_pbr,
        metallic=0.0,
        roughness=0.35,
        opacity=opacity if opacity is not None else 1.0,
        show_scalar_bar=show_scalar_bar,
        scalar_bar_args=sbar,
    )

    # SSAO for sulcal shadow contrast
    if ssao:
        try:
            pl.enable_ssao(radius=5.0, bias=0.5, kernel_size=128, blur=True)
        except Exception:
            pass  # SSAO not available in all VTK builds

    pl.enable_anti_aliasing("ssaa")
    pl.camera_position = camera_position
    pl.show(auto_close=False)
    img = pl.screenshot(return_img=True)
    pl.close()
    return img


def _auto_clim(
    scalars: np.ndarray,
    symmetric: bool = False,
    percentile: float = 98,
) -> Tuple[float, float]:
    """Compute colour limits from data, optionally symmetric about zero."""
    valid = scalars[np.isfinite(scalars)]
    if len(valid) == 0:
        return (0.0, 1.0)
    if symmetric:
        vmax = np.percentile(np.abs(valid), percentile)
        return (-vmax, vmax)
    return (np.percentile(valid, 100 - percentile),
            np.percentile(valid, percentile))


# ═══════════════════════════════════════════════════════════════════════════
# Panel label helper
# ═══════════════════════════════════════════════════════════════════════════


def _add_panel_labels(
    axes: Sequence,
    style: str = "upper",
    fontsize: int = 12,
    x: float = -0.08,
    y: float = 1.06,
) -> None:
    """
    Add A/B/C… or a/b/c… panel labels to a sequence of axes.

    Parameters
    ----------
    axes : sequence of matplotlib Axes
    style : ``'upper'`` (PNAS, Brain) or ``'lower'`` (Nature)
    fontsize : int
    x, y : float
        Position in axes-fraction coordinates.
    """
    labels = ascii_uppercase if style == "upper" else ascii_lowercase
    for i, ax in enumerate(axes):
        if i >= len(labels):
            break
        ax.text(x, y, labels[i], transform=ax.transAxes,
                fontsize=fontsize, fontweight="bold", va="top", ha="right")


# ═══════════════════════════════════════════════════════════════════════════
# Figure export
# ═══════════════════════════════════════════════════════════════════════════


def save_figure(
    fig: Figure,
    path: Union[str, Path],
    dpi: int = 300,
    formats: Optional[List[str]] = None,
    transparent: bool = False,
) -> None:
    """
    Save a figure in publication-quality format(s).

    Parameters
    ----------
    fig : Figure
    path : str or Path
        Base path (extension added per format).
    dpi : int
        Resolution. 300 for review, 600 for final submission.
    formats : list of str or None
        File formats. Default: ``['png', 'pdf']``.
    transparent : bool
        Transparent background (useful for compositing).
    """
    path = Path(path)
    if formats is None:
        formats = ["png", "pdf"]
    for fmt in formats:
        out = path.with_suffix(f".{fmt}")
        fig.savefig(str(out), dpi=dpi, bbox_inches="tight",
                    facecolor="white" if not transparent else "none",
                    edgecolor="none", transparent=transparent)
        logger.info("Saved figure: %s", out)


# ═══════════════════════════════════════════════════════════════════════════
# ███  SURFACE BRAIN PLOTS  ███
# ═══════════════════════════════════════════════════════════════════════════


def plot_surface_4view(
    vertices: np.ndarray,
    faces: np.ndarray,
    scalars: np.ndarray,
    hemi: str = "lh",
    cmap: str = CMAP_DIVERGING,
    clim: Optional[Tuple[float, float]] = None,
    symmetric: bool = True,
    threshold: Optional[float] = None,
    curvature: Optional[np.ndarray] = None,
    cbar_label: str = "",
    title: str = "",
    views: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    journal: str = "nature",
    dpi: int = 300,
    output_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """
    Four-view brain surface panel with a scalar overlay.

    Renders lateral and medial views of one hemisphere (or lateral views
    of both if ``views`` are customised), composites the screenshots into
    a clean matplotlib figure with a single colour bar, panel labels, and
    an optional title.

    This is the workhorse function for vertex-wise data from:
    - ``spectral.py``: HKS, WKS, GPS at a single scale
    - ``normative.py``: NormativeResult.z_score, .surprise
    - ``surprise.py``: SurpriseMap.z_score, .surprise, .anomaly_probability
    - ``asymmetry.py``: left–right difference maps

    Parameters
    ----------
    vertices : (N, 3) float
        Mesh vertex coordinates.
    faces : (F, 3) int
        Triangle connectivity.
    scalars : (N,) float
        Per-vertex values to display.
    hemi : ``'lh'`` or ``'rh'``
        Which hemisphere — determines camera angles.
    cmap : str
        Matplotlib colourmap.
    clim : (vmin, vmax) or None
        Colour limits. None → auto from data.
    symmetric : bool
        If True and clim is None, centres the colourbar at zero.
    threshold : float or None
        If set, vertices with |value| < threshold are transparent.
    curvature : (N,) float or None
        Sulcal depth overlay (typically FreeSurfer ``curv``).
    cbar_label : str
        Colour bar label (supports LaTeX: ``r'$\\beta$'``).
    title : str
        Figure suptitle.
    views : list of str or None
        Camera view names. Default for lh:
        ``['lateral_left', 'medial_left', 'dorsal', 'ventral']``.
    figsize : (w, h) or None
        Figure size in inches. None → auto from journal.
    journal : str
        Target journal for sizing (``'nature'``, ``'pnas'``, etc.).
    dpi : int
        Render DPI.
    output_path : str or Path or None
        If provided, saves the figure.

    Returns
    -------
    Figure
        Composited matplotlib figure.
    """
    _setup_style(journal)

    if views is None:
        if hemi == "lh":
            views = ["lateral_left", "medial_left", "dorsal", "ventral"]
        else:
            views = ["lateral_right", "medial_right", "dorsal", "ventral"]

    if clim is None:
        clim = _auto_clim(scalars, symmetric=symmetric)

    if figsize is None:
        w = JOURNAL_SPECS.get(journal, JOURNAL_SPECS["nature"])["double"]
        figsize = (w, w * 0.45)

    mesh = _make_pv_mesh(vertices, faces)

    # Render each view
    imgs = []
    for view_name in views:
        cpos = CAMERA_POSITIONS.get(view_name, CAMERA_POSITIONS["lateral_left"])
        img = _render_pv_view(
            mesh, scalars, cmap=cmap, clim=clim,
            camera_position=cpos,
            curvature=curvature,
            threshold=threshold,
        )
        imgs.append(img)

    # Composite into matplotlib
    n_views = len(imgs)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = gridspec.GridSpec(1, n_views + 1, width_ratios=[1] * n_views + [0.05],
                           wspace=0.02)

    panel_axes = []
    for i, img in enumerate(imgs):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(views[i].replace("_", " ").title(), fontsize=7, pad=2)
        panel_axes.append(ax)

    # Shared colour bar on the right
    cbar_ax = fig.add_subplot(gs[0, -1])
    norm = TwoSlopeNorm(vcenter=0, vmin=clim[0], vmax=clim[1]) if symmetric \
        else Normalize(vmin=clim[0], vmax=clim[1])
    cb = ColorbarBase(cbar_ax, cmap=plt.get_cmap(cmap), norm=norm,
                      orientation="vertical")
    cb.set_label(cbar_label, fontsize=8)
    cb.ax.tick_params(labelsize=7)

    if title:
        fig.suptitle(title, fontsize=10, fontweight="bold", y=1.02)

    if output_path:
        save_figure(fig, output_path, dpi=dpi)

    return fig


def plot_surface_comparison(
    vertices_list: List[np.ndarray],
    faces_list: List[np.ndarray],
    scalars_list: List[np.ndarray],
    group_names: List[str],
    hemi: str = "lh",
    cmap: str = CMAP_DIVERGING,
    clim: Optional[Tuple[float, float]] = None,
    symmetric: bool = True,
    threshold: Optional[float] = None,
    curvature_list: Optional[List[np.ndarray]] = None,
    cbar_label: str = "",
    title: str = "",
    views: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    journal: str = "nature",
    dpi: int = 300,
    output_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """
    Multi-group brain surface comparison with shared colour bar.

    Each group occupies one row of brain views (lateral, medial, dorsal,
    ventral). All rows share the same colourbar, positioned on the right
    edge, centred vertically. Group labels appear on the left.

    Use case: comparing cortical thickness, z-scores, or asymmetry between
    MTLE-HS patients vs. controls, or left-focus vs. right-focus groups.

    Parameters
    ----------
    vertices_list : list of (N, 3) arrays
        One mesh per group.
    faces_list : list of (F, 3) arrays
        Faces for each group's mesh.
    scalars_list : list of (N,) arrays
        Per-vertex data per group.
    group_names : list of str
        Row labels (e.g. ``['Controls', 'MTLE-HS']``).
    hemi : ``'lh'`` or ``'rh'``
    cmap, clim, symmetric, threshold, cbar_label, title :
        See :func:`plot_surface_4view`.
    curvature_list : list of (N,) arrays or None
    views : list of str or None
    figsize : (w, h) or None
    journal : str
    dpi : int
    output_path : str/Path or None

    Returns
    -------
    Figure
        Multi-row composited figure.
    """
    _setup_style(journal)

    n_groups = len(scalars_list)
    if views is None:
        if hemi == "lh":
            views = ["lateral_left", "medial_left", "dorsal", "ventral"]
        else:
            views = ["lateral_right", "medial_right", "dorsal", "ventral"]
    n_views = len(views)

    # Compute shared colour limits across all groups
    if clim is None:
        all_scalars = np.concatenate([s.ravel() for s in scalars_list])
        clim = _auto_clim(all_scalars, symmetric=symmetric)

    if figsize is None:
        w = JOURNAL_SPECS.get(journal, JOURNAL_SPECS["nature"])["double"]
        figsize = (w, w * 0.35 * n_groups)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    # Layout: n_groups rows × (n_views + 1) cols (last col = shared cbar)
    gs = gridspec.GridSpec(
        n_groups, n_views + 1,
        width_ratios=[1] * n_views + [0.04],
        wspace=0.02, hspace=0.08,
    )

    for g in range(n_groups):
        mesh = _make_pv_mesh(vertices_list[g], faces_list[g])
        curv = curvature_list[g] if curvature_list else None

        for v, view_name in enumerate(views):
            cpos = CAMERA_POSITIONS.get(view_name, CAMERA_POSITIONS["lateral_left"])
            img = _render_pv_view(
                mesh, scalars_list[g], cmap=cmap, clim=clim,
                camera_position=cpos, curvature=curv, threshold=threshold,
            )
            ax = fig.add_subplot(gs[g, v])
            ax.imshow(img)
            ax.axis("off")
            if g == 0:
                ax.set_title(view_name.replace("_", " ").title(), fontsize=7, pad=2)

        # Group label on the far left via annotation
        ax_first = fig.add_subplot(gs[g, 0])
        ax_first.axis("off")
        ax_first.annotate(
            group_names[g], xy=(-0.05, 0.5),
            xycoords="axes fraction", fontsize=9, fontweight="bold",
            ha="right", va="center", rotation=90,
        )

    # Shared colour bar spanning all rows, on the right
    cbar_ax = fig.add_subplot(gs[:, -1])
    norm = TwoSlopeNorm(vcenter=0, vmin=clim[0], vmax=clim[1]) if symmetric \
        else Normalize(vmin=clim[0], vmax=clim[1])
    cb = ColorbarBase(cbar_ax, cmap=plt.get_cmap(cmap), norm=norm,
                      orientation="vertical")
    cb.set_label(cbar_label, fontsize=8, labelpad=8)
    cb.ax.tick_params(labelsize=7)

    if title:
        fig.suptitle(title, fontsize=10, fontweight="bold", y=1.02)

    if output_path:
        save_figure(fig, output_path, dpi=dpi)

    return fig


# ═══════════════════════════════════════════════════════════════════════════
# ███  MODULE-SPECIFIC SURFACE PLOTS  ███
# ═══════════════════════════════════════════════════════════════════════════


def plot_surprise_brain(
    vertices: np.ndarray,
    faces: np.ndarray,
    surprise_map: "SurpriseMap",
    metric: str = "z_score",
    hemi: str = "lh",
    curvature: Optional[np.ndarray] = None,
    z_threshold: float = 2.0,
    journal: str = "nature",
    output_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """
    Render a SurpriseMap from ``surprise.py`` on the cortical surface.

    Dedicated function for :class:`~corticalfields.surprise.SurpriseMap`
    output — selects appropriate colourmap and thresholding per metric.

    Parameters
    ----------
    vertices, faces : mesh arrays
    surprise_map : SurpriseMap
        From :func:`~corticalfields.surprise.compute_surprise`.
    metric : ``'z_score'``, ``'surprise'``, ``'excess_surprise'``,
        ``'anomaly_probability'``
    hemi : ``'lh'`` or ``'rh'``
    curvature : (N,) float or None
    z_threshold : float
        Threshold for z-score map transparency.
    journal : str
    output_path : path or None

    Returns
    -------
    Figure
    """
    METRIC_CONFIG = {
        "z_score":             (CMAP_DIVERGING, True,  z_threshold, r"$z$-score"),
        "surprise":            (CMAP_SEQUENTIAL, False, None, r"Surprise $(-\log\, p)$"),
        "excess_surprise":     (CMAP_SEQUENTIAL, False, None, "Excess surprise"),
        "anomaly_probability": ("hot", False, 0.5, r"$P(\mathrm{anomalous})$"),
    }

    cmap, sym, thresh, label = METRIC_CONFIG.get(
        metric, (CMAP_DIVERGING, True, None, metric),
    )
    data = getattr(surprise_map, metric)

    return plot_surface_4view(
        vertices, faces, data, hemi=hemi, cmap=cmap,
        symmetric=sym, threshold=thresh, curvature=curvature,
        cbar_label=label,
        title=f"Surprise map — {metric.replace('_', ' ')}",
        journal=journal, output_path=output_path,
    )


def plot_normative_result(
    vertices: np.ndarray,
    faces: np.ndarray,
    result: "NormativeResult",
    hemi: str = "lh",
    curvature: Optional[np.ndarray] = None,
    journal: str = "nature",
    output_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """
    Two-row panel of z-score and surprise maps from a NormativeResult.

    Dedicated for :class:`~corticalfields.normative.NormativeResult` output.
    Row 1: z-score (diverging, thresholded at |z|>2).
    Row 2: surprise (sequential, no threshold).

    Parameters
    ----------
    vertices, faces : mesh arrays
    result : NormativeResult
        From :meth:`CorticalNormativeModel.predict`.
    hemi : ``'lh'`` or ``'rh'``
    curvature : (N,) or None
    journal, output_path : see :func:`plot_surface_4view`

    Returns
    -------
    Figure
    """
    return plot_surface_comparison(
        vertices_list=[vertices, vertices],
        faces_list=[faces, faces],
        scalars_list=[result.z_score, result.surprise],
        group_names=[r"$z$-score", r"Surprise $(-\log\, p)$"],
        hemi=hemi,
        cmap=CMAP_DIVERGING,
        symmetric=True,
        threshold=2.0,
        curvature_list=[curvature, curvature] if curvature is not None else None,
        cbar_label=f"Deviation ({result.feature_name})",
        title="Normative deviation",
        journal=journal,
        output_path=output_path,
    )


def plot_hks_multiscale(
    vertices: np.ndarray,
    faces: np.ndarray,
    hks: np.ndarray,
    scales: Optional[List[int]] = None,
    hemi: str = "lh",
    curvature: Optional[np.ndarray] = None,
    journal: str = "nature",
    output_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """
    Multi-scale HKS panel — one column per diffusion time.

    Dedicated for :func:`~corticalfields.spectral.heat_kernel_signature`
    output (shape N × T). Selects geometrically-spaced time scales and
    renders each on the lateral view, creating a horizontal strip that
    shows how geometry varies from fine curvature (small t) to global
    lobe structure (large t).

    Parameters
    ----------
    vertices, faces : mesh arrays
    hks : (N, T) float
        HKS matrix from ``heat_kernel_signature(lb, n_scales=T)``.
    scales : list of int or None
        Column indices to display. None → 5 geometrically-spaced.
    hemi : ``'lh'`` or ``'rh'``
    curvature : (N,) or None
    journal, output_path : standard

    Returns
    -------
    Figure
    """
    _setup_style(journal)
    T = hks.shape[1]
    if scales is None:
        scales = np.unique(np.geomspace(0, T - 1, 5).astype(int)).tolist()

    n = len(scales)
    w = JOURNAL_SPECS.get(journal, JOURNAL_SPECS["nature"])["double"]
    figsize = (w, w / n * 0.85)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, n + 1, width_ratios=[1] * n + [0.04], wspace=0.02)

    mesh = _make_pv_mesh(vertices, faces)
    view = "lateral_left" if hemi == "lh" else "lateral_right"
    cpos = CAMERA_POSITIONS[view]

    # Global colour range across all selected scales
    all_vals = np.concatenate([hks[:, s] for s in scales])
    clim = _auto_clim(all_vals, symmetric=False, percentile=99)

    for i, s_idx in enumerate(scales):
        img = _render_pv_view(
            mesh, hks[:, s_idx], cmap=CMAP_SPECTRAL, clim=clim,
            camera_position=cpos, curvature=curvature,
        )
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"$t = {s_idx}$", fontsize=7, pad=2)

    cbar_ax = fig.add_subplot(gs[0, -1])
    norm = Normalize(vmin=clim[0], vmax=clim[1])
    cb = ColorbarBase(cbar_ax, cmap=plt.get_cmap(CMAP_SPECTRAL), norm=norm,
                      orientation="vertical")
    cb.set_label("HKS", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    fig.suptitle("Heat Kernel Signature — multi-scale", fontsize=9,
                 fontweight="bold", y=1.01)

    if output_path:
        save_figure(fig, output_path)
    return fig


def plot_asymmetry_brain(
    vertices_lh: np.ndarray,
    faces_lh: np.ndarray,
    scalars_lh: np.ndarray,
    scalars_rh: np.ndarray,
    curvature: Optional[np.ndarray] = None,
    cbar_label: str = "L \u2212 R asymmetry",
    journal: str = "nature",
    output_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """
    Vertex-wise asymmetry map displayed on the left hemisphere surface.

    Computes L − R difference and renders it with a diverging colourmap
    centred at zero. Dedicated for comparing outputs from
    :func:`~corticalfields.asymmetry.classical_asymmetry_index` or
    direct vertex-wise difference maps.

    Parameters
    ----------
    vertices_lh, faces_lh : LH mesh arrays
    scalars_lh, scalars_rh : (N,) vertex-wise values per hemisphere
    curvature : (N,) or None
    cbar_label : str
    journal, output_path : standard

    Returns
    -------
    Figure
    """
    asym = scalars_lh - scalars_rh
    return plot_surface_4view(
        vertices_lh, faces_lh, asym, hemi="lh",
        cmap="PuOr", symmetric=True, curvature=curvature,
        cbar_label=cbar_label,
        title="Hemispheric asymmetry (L \u2212 R)",
        journal=journal, output_path=output_path,
    )


# ═══════════════════════════════════════════════════════════════════════════
# ███  CONNECTIVITY / ADJACENCY MATRIX PLOTS  ███
# ═══════════════════════════════════════════════════════════════════════════


def plot_connectivity_matrix(
    matrix: np.ndarray,
    labels: Optional[List[str]] = None,
    network_labels: Optional[np.ndarray] = None,
    network_names: Optional[List[str]] = None,
    network_colors: Optional[List[str]] = None,
    cmap: str = CMAP_MATRIX,
    vmax: Optional[float] = None,
    center: float = 0.0,
    title: str = "",
    cbar_label: str = "Connectivity",
    order: str = "network",
    show_boundaries: bool = True,
    pvalues: Optional[np.ndarray] = None,
    figsize: Optional[Tuple[float, float]] = None,
    journal: str = "nature",
    output_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """
    Connectivity or similarity matrix with network colour sidebars.

    Dedicated for outputs from ``graphs.py`` (MSN, SSN adjacency matrices)
    and ``transport.py`` (Wasserstein kernel / distance matrices). Supports
    three matrix orderings, network boundary lines, and significance markers.

    Parameters
    ----------
    matrix : (R, R) float
        Symmetric matrix.
    labels : list of str or None
        ROI labels for tick marks.
    network_labels : (R,) int or None
        Network assignment per ROI (for colour sidebar and ordering).
    network_names : list of str or None
        Names for each network ID.
    network_colors : list of str or None
        Colours per network. Default: Yeo-7 Okabe-Ito.
    cmap : str
        Colourmap.
    vmax : float or None
        Colour limit (symmetric about ``center``).
    center : float
        Centre of the diverging colourmap.
    title : str
    cbar_label : str
    order : ``'network'``, ``'hierarchical'``, ``'original'``
        ROI ordering strategy.
    show_boundaries : bool
        Draw white lines at network boundaries.
    pvalues : (R, R) float or None
        If provided, adds significance stars for p < 0.05.
    figsize : (w, h) or None
    journal : str
    output_path : path or None

    Returns
    -------
    Figure
    """
    _setup_style(journal)

    R = matrix.shape[0]

    # Determine ordering
    if order == "network" and network_labels is not None:
        idx = np.argsort(network_labels)
    elif order == "hierarchical":
        dist = 1.0 - np.abs(matrix)
        np.fill_diagonal(dist, 0)
        dist = np.clip(dist, 0, None)
        Z = linkage(squareform(dist + dist.T) / 2, method="average")
        Z = optimal_leaf_ordering(Z, squareform(dist + dist.T) / 2)
        idx = leaves_list(Z)
    else:
        idx = np.arange(R)

    M = matrix[np.ix_(idx, idx)]

    if vmax is None:
        flat = M[np.triu_indices(R, k=1)]
        vmax = np.percentile(np.abs(flat - center), 95)

    if figsize is None:
        s = JOURNAL_SPECS.get(journal, JOURNAL_SPECS["nature"])["double"]
        figsize = (s * 0.85, s * 0.75)

    # Figure layout: sidebar + matrix + colorbar
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        1, 3, width_ratios=[0.02, 1, 0.03], wspace=0.03,
    )

    # Network colour sidebar
    ax_side = fig.add_subplot(gs[0, 0])
    if network_labels is not None:
        net_sorted = network_labels[idx]
        colors = network_colors or YEO7_HEX
        for i, lab in enumerate(net_sorted):
            c = colors[int(lab) % len(colors)]
            ax_side.add_patch(mpatches.Rectangle((0, i), 1, 1, facecolor=c,
                                                  edgecolor="none"))
        ax_side.set_ylim(0, R)
        ax_side.set_xlim(0, 1)
        ax_side.invert_yaxis()
    ax_side.axis("off")

    # Main matrix
    ax_mat = fig.add_subplot(gs[0, 1])
    norm = TwoSlopeNorm(vcenter=center, vmin=center - vmax, vmax=center + vmax)
    im = ax_mat.imshow(M, cmap=cmap, norm=norm, aspect="equal",
                       interpolation="none")

    # Network boundary lines
    if show_boundaries and network_labels is not None:
        net_sorted = network_labels[idx]
        boundaries = np.where(np.diff(net_sorted) != 0)[0] + 1
        for b in boundaries:
            ax_mat.axhline(b - 0.5, color="white", linewidth=0.8)
            ax_mat.axvline(b - 0.5, color="white", linewidth=0.8)

    # Tick labels
    if labels is not None and R <= 50:
        sorted_labels = [labels[i] for i in idx]
        ax_mat.set_xticks(range(R))
        ax_mat.set_xticklabels(sorted_labels, rotation=90, fontsize=5)
        ax_mat.set_yticks(range(R))
        ax_mat.set_yticklabels(sorted_labels, fontsize=5)
    else:
        ax_mat.set_xticks([])
        ax_mat.set_yticks([])

    # Significance markers
    if pvalues is not None:
        P = pvalues[np.ix_(idx, idx)]
        for i in range(R):
            for j in range(i + 1, R):
                if P[i, j] < 0.001:
                    ax_mat.text(j, i, "***", ha="center", va="center",
                                fontsize=4, color="white" if abs(M[i, j]) > vmax * 0.6 else "black")
                elif P[i, j] < 0.01:
                    ax_mat.text(j, i, "**", ha="center", va="center",
                                fontsize=4, color="white" if abs(M[i, j]) > vmax * 0.6 else "black")
                elif P[i, j] < 0.05:
                    ax_mat.text(j, i, "*", ha="center", va="center",
                                fontsize=4, color="white" if abs(M[i, j]) > vmax * 0.6 else "black")

    ax_mat.set_title(title, fontsize=9, fontweight="bold", pad=8)

    # Colour bar
    cbar_ax = fig.add_subplot(gs[0, 2])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label(cbar_label, fontsize=8)
    cb.ax.tick_params(labelsize=7)

    if output_path:
        save_figure(fig, output_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# ███  FUNCTIONAL MAP C MATRIX  ███
# ═══════════════════════════════════════════════════════════════════════════


def plot_functional_map_matrix(
    fm: "FunctionalMap",
    title: str = "Inter-hemispheric functional map",
    cbar_label: str = r"$C_{ij}$",
    show_bands: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    journal: str = "nature",
    output_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """
    Visualise the functional map C matrix with frequency-band annotations.

    Dedicated for :class:`~corticalfields.functional_maps.FunctionalMap`.
    The diagonal represents isotropic correspondence (perfect symmetry);
    off-diagonal elements quantify frequency mixing (asymmetry). Dashed
    lines delineate low / mid / high frequency bands.

    Parameters
    ----------
    fm : FunctionalMap
        From :func:`~corticalfields.functional_maps.compute_functional_map`.
    title : str
    cbar_label : str
    show_bands : bool
        If True, overlays dashed lines at eigenfunction index 20 and 100
        delineating low / mid / high frequency bands.
    figsize, journal, output_path : standard

    Returns
    -------
    Figure
    """
    _setup_style(journal)

    k = min(fm.C.shape)
    C = fm.C[:k, :k]

    if figsize is None:
        s = JOURNAL_SPECS.get(journal, JOURNAL_SPECS["nature"])["single"]
        figsize = (s, s * 0.95)

    fig, ax = plt.subplots(figsize=figsize)
    vmax = np.percentile(np.abs(C), 98)
    im = ax.imshow(C, cmap=CMAP_MATRIX, vmin=-vmax, vmax=vmax,
                   aspect="equal", interpolation="none", origin="upper")

    ax.set_xlabel(r"Source eigenfunction index ($\phi_j^{\mathrm{LH}}$)",
                  fontsize=8)
    ax.set_ylabel(r"Target eigenfunction index ($\phi_i^{\mathrm{RH}}$)",
                  fontsize=8)

    # Frequency band annotations
    if show_bands:
        for b in [20, 100]:
            if b < k:
                ax.axhline(b - 0.5, color="white", linestyle="--",
                           linewidth=0.8, alpha=0.8)
                ax.axvline(b - 0.5, color="white", linestyle="--",
                           linewidth=0.8, alpha=0.8)
        # Band labels
        if k > 25:
            ax.text(10, -3, "Low", fontsize=6, ha="center", color="#333",
                    style="italic")
        if k > 60:
            ax.text(60, -3, "Mid", fontsize=6, ha="center", color="#333",
                    style="italic")
        if k > 150:
            ax.text(150, -3, "High", fontsize=6, ha="center", color="#333",
                    style="italic")

    cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cb.set_label(cbar_label, fontsize=8)
    cb.ax.tick_params(labelsize=7)

    ax.set_title(title, fontsize=9, fontweight="bold", pad=10)

    # Annotation box with asymmetry metrics
    info_text = (
        f"Off-diag energy: {fm.off_diagonal_energy:.3f}\n"
        f"Diag dominance: {fm.diagonal_dominance:.3f}"
    )
    ax.text(0.98, 0.02, info_text, transform=ax.transAxes, fontsize=6,
            va="bottom", ha="right", family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#ccc", alpha=0.9))

    plt.tight_layout()
    if output_path:
        save_figure(fig, output_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# ███  DISTANCE / KERNEL MATRIX  ███
# ═══════════════════════════════════════════════════════════════════════════


def plot_distance_matrix(
    D: np.ndarray,
    subject_ids: Optional[List[str]] = None,
    group_labels: Optional[np.ndarray] = None,
    group_names: Optional[List[str]] = None,
    cmap: str = CMAP_DISTANCE,
    title: str = "Pairwise distance matrix",
    cbar_label: str = "Distance",
    figsize: Optional[Tuple[float, float]] = None,
    journal: str = "nature",
    output_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """
    Heatmap of a pairwise distance or kernel matrix with group sidebars.

    Dedicated for outputs from ``transport.py``
    (:func:`~corticalfields.transport.pairwise_wasserstein_matrix`) and
    ``distance_stats.py`` (kernel matrices).

    Parameters
    ----------
    D : (N, N) float
        Symmetric distance or kernel matrix.
    subject_ids : list of str or None
    group_labels : (N,) int or None
        Group assignment (e.g., 0=control, 1=patient).
    group_names : list of str or None
    cmap : str
    title, cbar_label : str
    figsize, journal, output_path : standard

    Returns
    -------
    Figure
    """
    _setup_style(journal)
    N = D.shape[0]

    # Order by group
    if group_labels is not None:
        idx = np.argsort(group_labels)
        D_sorted = D[np.ix_(idx, idx)]
        gl_sorted = group_labels[idx]
    else:
        D_sorted = D
        idx = np.arange(N)
        gl_sorted = None

    if figsize is None:
        s = JOURNAL_SPECS.get(journal, JOURNAL_SPECS["nature"])["single"]
        figsize = (s, s * 0.9)

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 3, width_ratios=[0.03, 1, 0.03], wspace=0.02)

    # Group sidebar
    ax_side = fig.add_subplot(gs[0, 0])
    if gl_sorted is not None:
        grp_colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
        for i, g in enumerate(gl_sorted):
            ax_side.add_patch(mpatches.Rectangle(
                (0, i), 1, 1, facecolor=grp_colors[int(g) % len(grp_colors)],
                edgecolor="none",
            ))
        ax_side.set_ylim(0, N)
        ax_side.invert_yaxis()
    ax_side.axis("off")

    # Matrix
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(D_sorted, cmap=cmap, aspect="equal", interpolation="none")
    ax.set_title(title, fontsize=9, fontweight="bold", pad=8)

    # Group boundary lines
    if gl_sorted is not None:
        boundaries = np.where(np.diff(gl_sorted) != 0)[0] + 1
        for b in boundaries:
            ax.axhline(b - 0.5, color="white", linewidth=1)
            ax.axvline(b - 0.5, color="white", linewidth=1)

    ax.set_xticks([])
    ax.set_yticks([])

    # Colourbar
    cbar_ax = fig.add_subplot(gs[0, 2])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label(cbar_label, fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # Legend for groups
    if group_names and gl_sorted is not None:
        grp_colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
        patches = [mpatches.Patch(color=grp_colors[i], label=name)
                   for i, name in enumerate(group_names)]
        ax.legend(handles=patches, loc="lower left", fontsize=6,
                  frameon=True, framealpha=0.9)

    plt.tight_layout()
    if output_path:
        save_figure(fig, output_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# ███  STATISTICAL RESULTS PLOTS  ███
# ═══════════════════════════════════════════════════════════════════════════


def plot_permutation_null(
    result: "StatisticalResult",
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    journal: str = "nature",
    output_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """
    Null distribution with observed test statistic.

    Dedicated for :class:`~corticalfields.distance_stats.StatisticalResult`
    from :func:`mdmr`, :func:`hsic`, :func:`distance_correlation`,
    :func:`mantel_test`. Shows the permutation null histogram, the
    observed statistic as a vertical line, and the p-value annotation.

    Parameters
    ----------
    result : StatisticalResult
    title : str or None
        Auto-generated from method name if None.
    figsize, journal, output_path : standard

    Returns
    -------
    Figure
    """
    _setup_style(journal)
    if title is None:
        title = f"{result.method} permutation test"

    if figsize is None:
        s = JOURNAL_SPECS.get(journal, JOURNAL_SPECS["nature"])["single"]
        figsize = (s, s * 0.7)

    fig, ax = plt.subplots(figsize=figsize)

    if result.null_distribution is not None and len(result.null_distribution) > 0:
        ax.hist(result.null_distribution, bins=60, density=True, alpha=0.65,
                color="#999999", edgecolor="white", linewidth=0.3,
                label="Permutation null")

    ax.axvline(result.statistic, color="#D55E00", linewidth=1.8, linestyle="-",
               label=f"Observed = {result.statistic:.4f}", zorder=5)

    # p-value annotation
    sig = "***" if result.p_value < 0.001 else "**" if result.p_value < 0.01 \
          else "*" if result.p_value < 0.05 else "n.s."
    ptext = f"$p$ = {result.p_value:.4f} ({sig})"
    if result.effect_size is not None:
        ptext += f"\nEffect = {result.effect_size:.4f}"

    ax.text(0.97, 0.95, ptext, transform=ax.transAxes, fontsize=7,
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#ccc", alpha=0.95))

    ax.set_xlabel("Test statistic", fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.legend(fontsize=7, frameon=False, loc="upper left")

    plt.tight_layout()
    if output_path:
        save_figure(fig, output_path)
    return fig


def plot_eigenspectrum(
    eigenvalues: np.ndarray,
    n_show: int = 100,
    title: str = r"Laplace\u2013Beltrami eigenspectrum",
    figsize: Optional[Tuple[float, float]] = None,
    journal: str = "nature",
    output_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """
    LB eigenvalue spectrum with Weyl's law diagnostic.

    Dedicated for :class:`~corticalfields.spectral.LaplaceBeltrami`.
    Left panel: linear eigenvalue growth. Right panel: log-log plot
    (should follow slope ~1 for a 2D surface by Weyl's law:
    λ_n ~ 4πn / Area).

    Parameters
    ----------
    eigenvalues : (K,) float
        From ``LaplaceBeltrami.eigenvalues``.
    n_show : int
        Number of eigenvalues to display.
    title : str
    figsize, journal, output_path : standard

    Returns
    -------
    Figure
    """
    _setup_style(journal)
    evals = eigenvalues[:n_show]
    n = np.arange(len(evals))

    if figsize is None:
        w = JOURNAL_SPECS.get(journal, JOURNAL_SPECS["nature"])["double"]
        figsize = (w, w * 0.35)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Linear
    axes[0].plot(n, evals, "-", linewidth=0.9, color="#0072B2", markersize=1.5)
    axes[0].fill_between(n, 0, evals, alpha=0.1, color="#0072B2")
    axes[0].set_xlabel("Eigenvalue index $k$")
    axes[0].set_ylabel(r"$\lambda_k$")
    axes[0].set_title("Eigenvalue spectrum", fontsize=8)

    # Log-log
    mask = evals > 0
    axes[1].loglog(n[mask] + 1, evals[mask], "o", markersize=1.5,
                   color="#D55E00", alpha=0.7)
    # Weyl's law reference line
    if mask.sum() > 10:
        k_ref = n[mask][10:]
        lam_ref = evals[mask][10:]
        slope = np.polyfit(np.log(k_ref + 1), np.log(lam_ref), 1)[0]
        axes[1].text(0.05, 0.92, f"slope = {slope:.2f}\n(Weyl: 1.0)",
                     transform=axes[1].transAxes, fontsize=6,
                     va="top", family="monospace",
                     bbox=dict(boxstyle="round", facecolor="white",
                               edgecolor="#ccc", alpha=0.9))
    axes[1].set_xlabel("$k + 1$ (log)")
    axes[1].set_ylabel(r"$\lambda_k$ (log)")
    axes[1].set_title("Log-log (Weyl's law check)", fontsize=8)
    axes[1].grid(True, alpha=0.2, which="both")

    fig.suptitle(title, fontsize=9, fontweight="bold", y=1.01)
    plt.tight_layout()
    if output_path:
        save_figure(fig, output_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# ███  NETWORK-LEVEL PLOTS  ███
# ═══════════════════════════════════════════════════════════════════════════


def plot_network_radar(
    network_values: Dict[str, float],
    network_colors: Optional[Dict[str, str]] = None,
    title: str = "Network profile",
    value_label: str = "",
    figsize: Optional[Tuple[float, float]] = None,
    journal: str = "nature",
    output_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """
    Radar / spider plot of per-network values.

    Dedicated for :meth:`SurpriseMap.aggregate_by_network` (one metric
    per network) and for per-network asymmetry scores.

    Parameters
    ----------
    network_values : dict
        Network name → scalar value.
    network_colors : dict or None
        Network name → hex colour. Default: Yeo-7 Okabe-Ito.
    title, value_label : str
    figsize, journal, output_path : standard

    Returns
    -------
    Figure
    """
    _setup_style(journal)
    names = list(network_values.keys())
    values = list(network_values.values())
    N = len(names)

    if network_colors is None:
        network_colors = {n: YEO7_COLORS.get(n, "#555555") for n in names}

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values_closed = values + [values[0]]
    angles_closed = angles + [angles[0]]

    if figsize is None:
        s = JOURNAL_SPECS.get(journal, JOURNAL_SPECS["nature"])["single"]
        figsize = (s, s)

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    # Fill and outline
    ax.plot(angles_closed, values_closed, "o-", linewidth=1.2,
            color="#333333", markersize=0, zorder=3)
    ax.fill(angles_closed, values_closed, alpha=0.08, color="#333333")

    # Coloured dots per network
    for i, (angle, val, name) in enumerate(zip(angles, values, names)):
        c = network_colors.get(name, "#555555")
        ax.plot(angle, val, "o", color=c, markersize=10, zorder=5,
                markeredgecolor="white", markeredgewidth=0.8)

    ax.set_xticks(angles)
    ax.set_xticklabels(names, fontsize=7, fontweight="bold")

    # Radial label
    if value_label:
        ax.set_ylabel(value_label, fontsize=7, labelpad=15)

    ax.set_title(title, fontsize=9, fontweight="bold", pad=15)
    ax.tick_params(axis="y", labelsize=6)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        save_figure(fig, output_path)
    return fig


def plot_network_anomaly_bars(
    network_scores: Dict[str, Dict[str, float]],
    metrics: Optional[List[str]] = None,
    network_colors: Optional[Dict[str, str]] = None,
    title: str = "Network anomaly profile",
    figsize: Optional[Tuple[float, float]] = None,
    journal: str = "nature",
    output_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """
    Horizontal bar chart of network anomaly metrics.

    Dedicated for the nested dict output of
    :meth:`SurpriseMap.aggregate_by_network`. Default layout: two panels
    showing mean z-score and fraction anomalous per network.

    Parameters
    ----------
    network_scores : dict
        Network name → dict with keys ``'mean_z'``, ``'fraction_anomalous'``,
        ``'mean_surprise'``, etc.
    metrics : list of str or None
        Which metrics to show. Default: ``['mean_z', 'fraction_anomalous']``.
    network_colors : dict or None
    title : str
    figsize, journal, output_path : standard

    Returns
    -------
    Figure
    """
    _setup_style(journal)
    if metrics is None:
        metrics = ["mean_z", "fraction_anomalous"]

    names = list(network_scores.keys())
    if network_colors is None:
        network_colors = {n: YEO7_COLORS.get(n, "#555555") for n in names}
    colors = [network_colors.get(n, "#555555") for n in names]

    n_metrics = len(metrics)
    if figsize is None:
        w = JOURNAL_SPECS.get(journal, JOURNAL_SPECS["nature"])["double"]
        figsize = (w, w * 0.3 * n_metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]

    metric_labels = {
        "mean_z":             r"Mean $z$-score",
        "fraction_anomalous": r"Fraction $|z| > 2$",
        "mean_surprise":      r"Mean surprise",
        "max_surprise":       r"Max surprise",
        "std_z":              r"SD of $z$",
    }

    for ax, metric in zip(axes, metrics):
        vals = [network_scores[n].get(metric, 0) for n in names]
        y_pos = np.arange(len(names))

        ax.barh(y_pos, vals, color=colors, edgecolor="white", linewidth=0.5,
                height=0.7)

        # Reference lines
        if metric == "mean_z":
            ax.axvline(0, color="black", linewidth=0.5)
            ax.axvline(-2, color="#D55E00", linestyle="--", linewidth=0.6, alpha=0.5)
            ax.axvline(2, color="#D55E00", linestyle="--", linewidth=0.6, alpha=0.5)
        elif metric == "fraction_anomalous":
            ax.set_xlim(0, min(max(vals) * 1.3, 1.0))

        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel(metric_labels.get(metric, metric), fontsize=8)
        ax.invert_yaxis()

    fig.suptitle(title, fontsize=9, fontweight="bold", y=1.02)
    plt.tight_layout()
    if output_path:
        save_figure(fig, output_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# ███  GRAPH NETWORK VISUALIZATION  ███
# ═══════════════════════════════════════════════════════════════════════════


def plot_brain_connectome(
    adjacency: np.ndarray,
    node_coords: np.ndarray,
    node_colors: Optional[np.ndarray] = None,
    node_sizes: Optional[np.ndarray] = None,
    edge_threshold: str = "90%",
    edge_cmap: str = "coolwarm",
    node_cmap: str = "YlOrRd",
    display_mode: str = "lzr",
    title: str = "",
    colorbar: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    journal: str = "nature",
    output_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """
    Glass brain connectome using nilearn.

    Nodes are ROI centroids in MNI space; edges are thresholded by
    percentile. Node size can encode degree/centrality (from
    ``graphs.py`` :func:`graph_metrics`) and node colour can encode
    a scalar (e.g., betweenness centrality, network strength).

    Parameters
    ----------
    adjacency : (R, R) float
        Symmetric connectivity matrix.
    node_coords : (R, 3) float
        MNI coordinates.
    node_colors : (R,) or None
        Scalar per node for colour mapping (e.g. centrality).
    node_sizes : (R,) or None
        Scalar per node for size mapping. Auto-scaled.
    edge_threshold : str or float
        Percentile string (e.g. ``'90%'``) or absolute value.
    edge_cmap : str
    node_cmap : str
    display_mode : str
        nilearn display modes: ``'ortho'``, ``'lzr'``, ``'lyrz'``, etc.
    title : str
    colorbar : bool
    figsize, journal, output_path : standard

    Returns
    -------
    Figure
    """
    try:
        from nilearn import plotting
    except ImportError:
        raise ImportError("nilearn is required for connectome plots.")

    _setup_style(journal)

    if figsize is None:
        w = JOURNAL_SPECS.get(journal, JOURNAL_SPECS["nature"])["double"]
        figsize = (w, w * 0.35)

    fig = plt.figure(figsize=figsize)

    # Node size scaling: sqrt for perceptual proportionality
    if node_sizes is not None:
        ns = node_sizes / node_sizes.max()
        ns = 20 + 180 * np.sqrt(ns)
    else:
        ns = 50

    display = plotting.plot_connectome(
        adjacency, node_coords,
        node_color=node_colors if node_colors is not None else "black",
        node_size=ns,
        edge_cmap=edge_cmap,
        edge_threshold=edge_threshold,
        display_mode=display_mode,
        colorbar=colorbar,
        alpha=0.7,
        figure=fig,
        title=title,
        node_kwargs={"cmap": node_cmap, "edgecolors": "black",
                     "linewidths": 0.3} if node_colors is not None else {},
    )

    if output_path:
        save_figure(fig, output_path)
    return fig


def plot_network_graph(
    adjacency: np.ndarray,
    node_labels: Optional[List[str]] = None,
    network_assignments: Optional[np.ndarray] = None,
    network_names: Optional[List[str]] = None,
    node_metric: Optional[np.ndarray] = None,
    edge_threshold_pct: float = 85.0,
    layout: str = "community",
    title: str = "",
    figsize: Optional[Tuple[float, float]] = None,
    journal: str = "nature",
    output_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """
    Publication-quality graph visualisation with community layout.

    Uses **netgraph** (if available) for community-aware node placement
    and edge bundling, falling back to **networkx** spring layout.
    Node size encodes ``node_metric`` (e.g. betweenness centrality from
    ``graphs.py``), node colour encodes network membership, and edge
    width/colour encode connection weight.

    Parameters
    ----------
    adjacency : (R, R) float
        Symmetric adjacency matrix.
    node_labels : list of str or None
        Node names for labelling hubs.
    network_assignments : (R,) int or None
        Network ID per node (for community layout + colouring).
    network_names : list of str or None
    node_metric : (R,) float or None
        Metric for node sizing (e.g. betweenness centrality).
    edge_threshold_pct : float
        Keep only edges above this percentile.
    layout : ``'community'``, ``'spring'``, ``'circular'``
    title : str
    figsize, journal, output_path : standard

    Returns
    -------
    Figure
    """
    import networkx as nx

    _setup_style(journal)
    R = adjacency.shape[0]

    # Threshold edges
    flat = adjacency[np.triu_indices(R, k=1)]
    flat_nonzero = flat[flat > 0]
    if len(flat_nonzero) > 0:
        thresh = np.percentile(flat_nonzero, edge_threshold_pct)
    else:
        thresh = 0
    A = adjacency.copy()
    A[A < thresh] = 0
    np.fill_diagonal(A, 0)

    G = nx.from_numpy_array(A)

    # Network colours
    net_colors_list = YEO7_HEX
    if network_assignments is not None:
        node_color = [net_colors_list[int(network_assignments[i]) % len(net_colors_list)]
                      for i in range(R)]
    else:
        node_color = ["#0072B2"] * R

    # Node sizes
    if node_metric is not None:
        nm = node_metric / np.max(node_metric)
        node_size = 100 + 500 * np.sqrt(nm)
    else:
        node_size = 200

    if figsize is None:
        s = JOURNAL_SPECS.get(journal, JOURNAL_SPECS["nature"])["double"]
        figsize = (s, s * 0.75)

    fig, ax = plt.subplots(figsize=figsize)

    # Try netgraph first
    try:
        from netgraph import Graph as NGGraph

        node_to_community = {}
        if network_assignments is not None:
            node_to_community = {i: int(network_assignments[i]) for i in range(R)}

        nsize = {i: (node_size[i] if hasattr(node_size, '__len__') else node_size) / 500
                 for i in range(R)}
        ncolor = {i: node_color[i] for i in range(R)}

        # Edge weights for width encoding
        edge_width = {}
        for u, v, d in G.edges(data=True):
            w = d.get("weight", 1.0)
            edge_width[(u, v)] = 0.3 + 2.5 * (w / (A.max() + 1e-12))

        layout_kw = {}
        if layout == "community" and node_to_community:
            layout_kw = dict(node_to_community=node_to_community)

        NGGraph(
            G,
            node_layout=layout if layout != "spring" else "spring",
            node_layout_kwargs=layout_kw,
            node_color=ncolor,
            node_size=nsize,
            node_edge_color=ncolor,
            node_edge_width=1.0,
            edge_layout="curved",
            edge_width=edge_width,
            edge_color="#888888",
            edge_alpha=0.5,
            ax=ax,
        )
        logger.info("Graph rendered with netgraph")

    except ImportError:
        logger.info("netgraph not available; falling back to networkx")
        if layout == "community" or layout == "spring":
            pos = nx.spring_layout(G, seed=42, k=1.5 / np.sqrt(R))
        elif layout == "circular":
            pos = nx.circular_layout(G)
        else:
            pos = nx.spring_layout(G, seed=42)

        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_color,
                               node_size=node_size if hasattr(node_size, '__len__') else [node_size] * R,
                               edgecolors="white", linewidths=0.5)
        edges = list(G.edges())
        weights = [G[u][v]["weight"] for u, v in edges]
        wmax = max(weights) if weights else 1
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=edges,
                               width=[0.3 + 2 * w / wmax for w in weights],
                               edge_color="#888888", alpha=0.5)

    # Hub labels (top-10 by metric)
    if node_labels and node_metric is not None:
        top_k = min(10, R)
        hub_idx = np.argsort(node_metric)[-top_k:]
        for i in hub_idx:
            ax.annotate(node_labels[i], xy=(0, 0), fontsize=5,
                        ha="center", va="center", color="#333")

    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.axis("off")

    # Network legend
    if network_names and network_assignments is not None:
        unique_nets = sorted(set(network_assignments))
        patches = [mpatches.Patch(
            color=net_colors_list[int(n) % len(net_colors_list)],
            label=network_names[int(n)] if int(n) < len(network_names) else f"Net {n}",
        ) for n in unique_nets]
        ax.legend(handles=patches, fontsize=6, loc="lower left",
                  frameon=True, framealpha=0.9, ncol=2)

    plt.tight_layout()
    if output_path:
        save_figure(fig, output_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# ███  ASYMMETRY BAND DECOMPOSITION  ███
# ═══════════════════════════════════════════════════════════════════════════


def plot_asymmetry_bands(
    profiles: List["AsymmetryProfile"],
    group_labels: Optional[List[str]] = None,
    title: str = "Asymmetry by frequency band",
    figsize: Optional[Tuple[float, float]] = None,
    journal: str = "nature",
    output_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """
    Grouped bar chart of frequency-band asymmetry across subjects/groups.

    Dedicated for :class:`~corticalfields.asymmetry.AsymmetryProfile`
    outputs from :func:`asymmetry_from_functional_map`. Shows how
    asymmetry decomposes across low / mid / high frequency bands,
    optionally grouped by clinical group.

    Parameters
    ----------
    profiles : list of AsymmetryProfile
    group_labels : list of str or None
        One label per profile (e.g. subject IDs or group names).
    title : str
    figsize, journal, output_path : standard

    Returns
    -------
    Figure
    """
    _setup_style(journal)

    bands = list(profiles[0].band_asymmetry.keys())
    n_bands = len(bands)
    n_subj = len(profiles)

    if group_labels is None:
        group_labels = [p.subject_id or f"S{i}" for i, p in enumerate(profiles)]

    if figsize is None:
        w = JOURNAL_SPECS.get(journal, JOURNAL_SPECS["nature"])["double"]
        figsize = (w, w * 0.4)

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(n_bands)
    bar_width = 0.8 / n_subj
    band_colors = plt.cm.Set2(np.linspace(0, 1, n_subj))

    for i, (prof, label) in enumerate(zip(profiles, group_labels)):
        vals = [prof.band_asymmetry.get(b, 0) for b in bands]
        offset = (i - n_subj / 2 + 0.5) * bar_width
        ax.bar(x + offset, vals, bar_width, label=label,
               color=band_colors[i], edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([b.replace("_", " ").title() for b in bands], fontsize=7)
    ax.set_ylabel("Off-diagonal energy", fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold")

    if n_subj <= 10:
        ax.legend(fontsize=6, frameon=False, ncol=min(n_subj, 4))

    plt.tight_layout()
    if output_path:
        save_figure(fig, output_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# ███  KRR CROSS-VALIDATION DIAGNOSTIC  ███
# ═══════════════════════════════════════════════════════════════════════════


def plot_krr_diagnostic(
    krr_result: Dict[str, Any],
    y_true: np.ndarray,
    outcome_name: str = "HADS-A",
    figsize: Optional[Tuple[float, float]] = None,
    journal: str = "nature",
    output_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """
    Kernel ridge regression cross-validation diagnostic plot.

    Dedicated for the dict output of
    :func:`~corticalfields.distance_stats.kernel_ridge_regression`.
    Left: predicted vs. observed scatter with identity line.
    Right: per-fold R² bar chart.

    Parameters
    ----------
    krr_result : dict
        From :func:`kernel_ridge_regression`.
    y_true : (N,) float
        True outcome values.
    outcome_name : str
        Label for the y-axis.
    figsize, journal, output_path : standard

    Returns
    -------
    Figure
    """
    _setup_style(journal)

    if figsize is None:
        w = JOURNAL_SPECS.get(journal, JOURNAL_SPECS["nature"])["double"]
        figsize = (w, w * 0.38)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    y_pred = krr_result["y_pred_cv"]
    r2 = krr_result["r2_cv"]
    mae = krr_result["mae_cv"]
    fold_r2 = krr_result["fold_r2"]

    # Scatter: predicted vs observed
    ax = axes[0]
    ax.scatter(y_true, y_pred, s=20, alpha=0.7, color="#0072B2",
               edgecolors="white", linewidths=0.3, zorder=3)
    lims = [min(y_true.min(), y_pred.min()) - 1,
            max(y_true.max(), y_pred.max()) + 1]
    ax.plot(lims, lims, "--", color="#999999", linewidth=0.8, zorder=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(f"Observed {outcome_name}", fontsize=8)
    ax.set_ylabel(f"Predicted {outcome_name}", fontsize=8)
    ax.set_title("Cross-validated prediction", fontsize=8)
    ax.set_aspect("equal")
    ax.text(0.05, 0.95, f"$R^2$ = {r2:.3f}\nMAE = {mae:.2f}",
            transform=ax.transAxes, fontsize=7, va="top",
            bbox=dict(boxstyle="round", facecolor="white",
                      edgecolor="#ccc", alpha=0.9))

    # Per-fold R²
    ax = axes[1]
    folds = np.arange(len(fold_r2)) + 1
    colors = ["#009E73" if r > 0 else "#D55E00" for r in fold_r2]
    ax.bar(folds, fold_r2, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(r2, color="#333333", linestyle="--", linewidth=0.8,
               label=f"Mean $R^2$ = {r2:.3f}")
    ax.set_xlabel("Fold", fontsize=8)
    ax.set_ylabel("$R^2$", fontsize=8)
    ax.set_title("Per-fold performance", fontsize=8)
    ax.legend(fontsize=6, frameon=False)

    _add_panel_labels(axes, style="upper")
    plt.tight_layout()
    if output_path:
        save_figure(fig, output_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# ███  SUBCORTICAL 3D RENDERING  ███
# ═══════════════════════════════════════════════════════════════════════════


def plot_subcortical_3d(
    structures: Dict[str, Any],
    cortex_mesh: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    cortex_opacity: float = 0.12,
    camera_position: Optional[list] = None,
    window_size: Tuple[int, int] = (1200, 1000),
    title: str = "",
    figsize: Optional[Tuple[float, float]] = None,
    journal: str = "nature",
    output_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """
    3D rendering of subcortical structures with transparent cortex shell.

    Each structure is rendered as a smoothed mesh (Taubin smoothing)
    with its canonical FreeSurfer colour. A transparent cortex provides
    anatomical context. Uses depth peeling for correct transparency.

    Parameters
    ----------
    structures : dict
        Name → ``(vertices, faces)`` or PyVista PolyData per structure.
    cortex_mesh : (vertices, faces) or None
        Cortical surface for the transparent shell.
    cortex_opacity : float
        Transparency of the cortex (0.1–0.2 recommended).
    camera_position : list or None
        Custom camera. Default: anterior view.
    window_size : (w, h)
    title : str
    figsize, journal, output_path : standard

    Returns
    -------
    Figure
    """
    import pyvista as pv

    _setup_style(journal)

    pl = pv.Plotter(off_screen=True, window_size=window_size)
    pl.set_background("white")
    pl.enable_depth_peeling(number_of_peels=10, occlusion_ratio=0.0)

    # Three-point lighting
    pl.remove_all_lights()
    pl.add_light(pv.Light(position=(200, 200, 300), intensity=1.0,
                          light_type="scene light"))
    pl.add_light(pv.Light(position=(-200, 100, 100), intensity=0.4,
                          light_type="scene light"))
    pl.add_light(pv.Light(position=(0, -200, 200), intensity=0.3,
                          light_type="scene light"))

    # Subcortical structures (opaque)
    for name, geom in structures.items():
        if isinstance(geom, tuple):
            mesh = _make_pv_mesh(geom[0], geom[1])
        else:
            mesh = geom  # assume already pv.PolyData

        # Taubin smoothing to remove marching-cube staircase
        mesh = mesh.smooth_taubin(n_iter=50, pass_band=0.01)

        color = SUBCORT_COLORS.get(name.split("-")[-1].split("_")[-1],
                                    "#888888")
        pl.add_mesh(mesh, color=color, smooth_shading=True, pbr=True,
                    metallic=0.0, roughness=0.4)

    # Transparent cortex
    if cortex_mesh is not None:
        v, f = cortex_mesh
        ctx = _make_pv_mesh(v, f)
        pl.add_mesh(ctx, color="white", opacity=cortex_opacity,
                    smooth_shading=True, specular=0.3)

    cpos = camera_position or CAMERA_POSITIONS["anterior"]
    pl.camera_position = cpos
    pl.enable_anti_aliasing("ssaa")
    try:
        pl.enable_ssao(radius=5.0, bias=0.5, kernel_size=128, blur=True)
    except Exception:
        pass

    pl.show(auto_close=False)
    img = pl.screenshot(return_img=True)
    pl.close()

    # Composite into matplotlib
    if figsize is None:
        s = JOURNAL_SPECS.get(journal, JOURNAL_SPECS["nature"])["single"]
        figsize = (s, s)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9, fontweight="bold", pad=5)

    # Structure legend
    patches = []
    for name in structures:
        short = name.split("-")[-1].split("_")[-1]
        c = SUBCORT_COLORS.get(short, "#888888")
        patches.append(mpatches.Patch(color=c, label=name))
    if patches:
        ax.legend(handles=patches, fontsize=6, loc="lower right",
                  frameon=True, framealpha=0.9, ncol=2)

    plt.tight_layout()
    if output_path:
        save_figure(fig, output_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# ███  COMPOSITE MULTI-PANEL FIGURE  ███
# ═══════════════════════════════════════════════════════════════════════════


def plot_composite_figure(
    panels: Dict[str, Figure],
    layout: Optional[List[List[str]]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    panel_label_style: str = "upper",
    journal: str = "nature",
    dpi: int = 300,
    output_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """
    Assemble multiple CorticalFields figures into a single composite.

    Takes pre-rendered Figure objects (from any brainplots function),
    converts them to images, and arranges them in a grid with panel
    labels (A, B, C, …).

    Parameters
    ----------
    panels : dict
        Label → matplotlib Figure. E.g.
        ``{'brain': fig1, 'matrix': fig2, 'radar': fig3}``.
    layout : list of list of str or None
        2D grid specifying panel arrangement. E.g.
        ``[['brain', 'brain'], ['matrix', 'radar']]``.
        None → single row.
    figsize : (w, h) or None
    panel_label_style : ``'upper'`` or ``'lower'``
    journal : str
    dpi : int
    output_path : path or None

    Returns
    -------
    Figure
        The composited figure.
    """
    _setup_style(journal)

    names = list(panels.keys())
    if layout is None:
        layout = [names]

    n_rows = len(layout)
    n_cols = max(len(row) for row in layout)

    # Rasterise each panel Figure to image
    images = {}
    for name, panel_fig in panels.items():
        buf = BytesIO()
        panel_fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                          facecolor="white")
        buf.seek(0)
        from PIL import Image
        images[name] = np.array(Image.open(buf))
        buf.close()

    if figsize is None:
        w = JOURNAL_SPECS.get(journal, JOURNAL_SPECS["nature"])["double"]
        figsize = (w, w * 0.5 * n_rows)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = gridspec.GridSpec(n_rows, n_cols, wspace=0.05, hspace=0.1)

    labels_seq = ascii_uppercase if panel_label_style == "upper" else ascii_lowercase
    label_idx = 0

    for r, row in enumerate(layout):
        for c, name in enumerate(row):
            if name not in images:
                continue
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(images[name])
            ax.axis("off")
            if label_idx < len(labels_seq):
                ax.text(-0.02, 1.02, labels_seq[label_idx],
                        transform=ax.transAxes, fontsize=12,
                        fontweight="bold", va="bottom", ha="right")
                label_idx += 1

    if output_path:
        save_figure(fig, output_path, dpi=dpi)
    return fig
