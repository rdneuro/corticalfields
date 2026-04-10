"""
Publication-quality brain graph visualization.

This module provides 12 core visualization functions for brain graphs
constructed from spectral morphometric connectivity, covering glass
brain connectomes, circular connectograms, adjacency matrices with
dendrograms, graph layouts, NBS results, persistent homology, rich-club
curves, and multi-panel comparison figures.

All functions produce matplotlib ``Figure`` / ``Axes`` objects for
journal submission (Nature, NeuroImage, Brain, PNAS).  Every function
accepts an ``ax`` parameter for embedding in composite figures and
returns ``(fig, ax)`` for further customisation.

Dependencies (optional, gracefully degraded):
    - nilearn : glass brain connectomes
    - mne-connectivity : circular connectograms
    - netgraph : publication-quality graph layouts
    - seaborn : adjacency matrix clustermap
    - ripser + persim : persistence diagrams
    - cmcrameri : perceptually uniform colormaps
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Publication-quality rcParams
# ═══════════════════════════════════════════════════════════════════════════

PUBLICATION_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 8,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.linewidth": 0.5,
    "lines.linewidth": 0.75,
    "xtick.major.size": 2,
    "xtick.major.width": 0.5,
    "ytick.major.size": 2,
    "ytick.major.width": 0.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
}

#: Yeo-7 canonical colours (same as graphs module)
YEO7_COLORS = {
    "Visual":          (120/255,  18/255, 134/255),
    "Somatomotor":     ( 70/255, 130/255, 180/255),
    "DorsalAttention":  (  0/255, 118/255,  14/255),
    "VentralAttention": (196/255,  58/255, 250/255),
    "Limbic":          (220/255, 248/255, 164/255),
    "Frontoparietal":  (230/255, 148/255,  34/255),
    "Default":         (205/255,  62/255,  78/255),
}

#: Paul Tol's Bright scheme — 7 colorblind-safe colours for communities
TOL_BRIGHT = [
    "#4477AA", "#EE6677", "#228833", "#CCBB44",
    "#66CCEE", "#AA3377", "#BBBBBB",
]

#: Okabe-Ito palette (Nature Methods recommended)
OKABE_ITO = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442",
    "#0072B2", "#D55E00", "#CC79A7", "#000000",
]


def _apply_pub_style():
    """Apply publication rcParams as context."""
    return mpl.rc_context(PUBLICATION_RC)


def _get_cmap(name: str = "RdBu_r"):
    """Get a colormap, preferring cmcrameri if available."""
    try:
        import cmcrameri.cm as cmc
        crameri_map = {"diverging": cmc.vik, "sequential": cmc.batlow,
                       "RdBu_r": cmc.vik, "hot": cmc.lajolla}
        if name in crameri_map:
            return crameri_map[name]
    except ImportError:
        pass
    return plt.get_cmap(name)


def _community_colors(n: int) -> List[str]:
    """Return n colorblind-safe colours for community assignment."""
    if n <= 7:
        return TOL_BRIGHT[:n]
    # Extend via tab20 for >7 communities
    cmap = plt.get_cmap("tab20")
    return [mcolors.rgb2hex(cmap(i / max(n - 1, 1))) for i in range(n)]


# ═══════════════════════════════════════════════════════════════════════════
# 1. Glass Brain Connectome
# ═══════════════════════════════════════════════════════════════════════════

def plot_glass_brain_connectome(
    adjacency: np.ndarray,
    node_coords: np.ndarray,
    node_colors: Optional[np.ndarray] = None,
    node_sizes: Optional[np.ndarray] = None,
    edge_threshold: str = "95%",
    display_mode: str = "lyrz",
    edge_cmap: str = "RdBu_r",
    node_cmap: str = "hot",
    title: str = "",
    ax: Optional[Any] = None,
    **kwargs,
) -> Any:
    """
    Glass brain connectome overlay at MNI coordinates.

    Parameters
    ----------
    adjacency : (R, R) symmetric weight matrix
    node_coords : (R, 3) MNI coordinates per ROI
    node_colors : (R,) scalar per node (mapped to *node_cmap*), or None
    node_sizes : (R,) node display sizes, or None for uniform
    edge_threshold : str or float — percentile string or absolute value
    display_mode : str — nilearn display mode ('lyrz', 'ortho', etc.)
    """
    try:
        from nilearn.plotting import plot_connectome
    except ImportError:
        raise ImportError("nilearn is required for glass brain plots. "
                          "Install with: pip install nilearn")

    with _apply_pub_style():
        if node_sizes is None:
            node_sizes = np.full(adjacency.shape[0], 25)
        else:
            # Scale to [20, 200] range
            mn, mx = node_sizes.min(), node_sizes.max()
            if mx > mn:
                node_sizes = 20 + 180 * (node_sizes - mn) / (mx - mn)
            else:
                node_sizes = np.full(len(node_sizes), 50)

        display = plot_connectome(
            adjacency, node_coords,
            node_color=node_colors,
            node_size=node_sizes,
            edge_threshold=edge_threshold,
            edge_cmap=edge_cmap,
            display_mode=display_mode,
            alpha=0.7,
            colorbar=True,
            title=title,
            axes=ax,
            **kwargs,
        )
        return display


# ═══════════════════════════════════════════════════════════════════════════
# 2. Circular Connectogram
# ═══════════════════════════════════════════════════════════════════════════

def plot_connectivity_circle(
    adjacency: np.ndarray,
    node_names: List[str],
    node_network: Optional[List[str]] = None,
    n_lines: int = 200,
    title: str = "",
    fig: Optional[plt.Figure] = None,
    colormap: str = "hot",
    **kwargs,
) -> Tuple[plt.Figure, Any]:
    """
    Circular connectogram with Yeo-7 network grouping.

    Parameters
    ----------
    adjacency : (R, R) connectivity matrix
    node_names : list of region names
    node_network : list of network names per node (for Yeo-7 coloring)
    n_lines : int — show only the top-N strongest connections
    """
    try:
        from mne.viz import circular_layout
        from mne_connectivity.viz import plot_connectivity_circle as _plot_cc
    except ImportError:
        raise ImportError("mne-connectivity is required for connectograms. "
                          "Install with: pip install mne-connectivity")

    with _apply_pub_style():
        # Determine node ordering: left hemisphere reversed, right in order
        lh = [n for n in node_names if "lh" in n.lower() or "LH" in n
              or n.startswith("L_")]
        rh = [n for n in node_names if n not in lh]
        node_order = lh[::-1] + rh
        group_boundaries = [0, len(lh)] if lh else None

        node_angles = circular_layout(
            node_names, node_order, start_pos=90,
            group_boundaries=group_boundaries or [0],
        )

        # Node colours from network assignment
        if node_network is not None:
            node_colors = []
            for net in node_network:
                node_colors.append(
                    YEO7_COLORS.get(net, (0.5, 0.5, 0.5)))
        else:
            node_colors = None

        if fig is None:
            fig = plt.figure(figsize=(8, 8), facecolor="white")

        fig_out, axes = _plot_cc(
            adjacency, node_names,
            node_angles=node_angles,
            node_colors=node_colors,
            n_lines=n_lines,
            colormap=colormap,
            title=title,
            fig=fig,
            show=False,
            **kwargs,
        )
        return fig_out, axes


# ═══════════════════════════════════════════════════════════════════════════
# 3. Adjacency Matrix with Dendrograms
# ═══════════════════════════════════════════════════════════════════════════

def plot_adjacency_matrix(
    adjacency: np.ndarray,
    node_names: Optional[List[str]] = None,
    network_labels: Optional[np.ndarray] = None,
    cluster: bool = True,
    cmap: str = "RdBu_r",
    center: float = 0.0,
    figsize: Tuple[float, float] = (10, 10),
    title: str = "",
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, Any]:
    """
    Adjacency / similarity matrix with hierarchical clustering dendrogram
    and optional network-coloured sidebars.

    Parameters
    ----------
    network_labels : (R,) integer community/network labels for sidebar
    """
    import seaborn as sns

    with _apply_pub_style():
        # Build network colour sidebar
        row_colors = None
        if network_labels is not None:
            n_comm = len(np.unique(network_labels))
            palette = _community_colors(n_comm)
            row_colors = [palette[int(l) % len(palette)]
                          for l in network_labels]

        if cluster:
            g = sns.clustermap(
                adjacency,
                method="ward",
                cmap=_get_cmap(cmap),
                center=center,
                row_colors=row_colors,
                col_colors=row_colors,
                figsize=figsize,
                dendrogram_ratio=0.12,
                xticklabels=node_names if node_names else False,
                yticklabels=node_names if node_names else False,
                linewidths=0,
            )
            if title:
                g.fig.suptitle(title, y=1.02, fontsize=9, fontweight="bold")
            return g.fig, g.ax_heatmap
        else:
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                fig = ax.figure
            im = ax.imshow(adjacency, cmap=_get_cmap(cmap),
                           aspect="equal",
                           norm=mcolors.CenteredNorm(vcenter=center)
                           if center == 0 else None)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if node_names:
                ax.set_xticks(range(len(node_names)))
                ax.set_yticks(range(len(node_names)))
                ax.set_xticklabels(node_names, rotation=90, fontsize=4)
                ax.set_yticklabels(node_names, fontsize=4)
            if title:
                ax.set_title(title, fontweight="bold")
            return fig, ax


# ═══════════════════════════════════════════════════════════════════════════
# 4. Edge-weight Distribution
# ═══════════════════════════════════════════════════════════════════════════

def plot_edge_weight_distribution(
    adjacency: np.ndarray,
    threshold: Optional[float] = None,
    density_threshold: Optional[float] = None,
    title: str = "Edge-weight distribution",
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Histogram of edge weights with threshold lines."""
    with _apply_pub_style():
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 3.5))
        else:
            fig = ax.figure

        tri = adjacency[np.triu_indices_from(adjacency, k=1)]
        tri = tri[tri != 0]

        ax.hist(tri, bins=60, color="#4477AA", alpha=0.8, edgecolor="white",
                linewidth=0.3, density=True)
        ax.set_xlabel("Edge weight")
        ax.set_ylabel("Density")
        ax.set_title(title, fontweight="bold")

        if threshold is not None:
            ax.axvline(threshold, color="#EE6677", linestyle="--",
                       linewidth=1.2, label=f"threshold = {threshold:.3f}")
        if density_threshold is not None:
            n_keep = max(1, int(density_threshold * len(tri)))
            t_val = np.sort(tri)[::-1][min(n_keep - 1, len(tri) - 1)]
            ax.axvline(t_val, color="#228833", linestyle="--",
                       linewidth=1.2,
                       label=f"density {density_threshold:.0%} → {t_val:.3f}")
        if threshold or density_threshold:
            ax.legend()

        return fig, ax


# ═══════════════════════════════════════════════════════════════════════════
# 5. Graph Laplacian Eigenspectrum
# ═══════════════════════════════════════════════════════════════════════════

def plot_laplacian_spectrum(
    eigenvalues: np.ndarray,
    n_show: int = 30,
    title: str = "Graph Laplacian spectrum",
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot graph Laplacian eigenvalues with spectral gap annotation."""
    with _apply_pub_style():
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 3.5))
        else:
            fig = ax.figure

        evals = np.sort(eigenvalues)[:n_show]
        ax.plot(range(len(evals)), evals, "o-", color="#4477AA",
                markersize=4, linewidth=1)

        # Annotate Fiedler value (λ₂)
        if len(evals) > 1:
            ax.axhline(evals[1], color="#EE6677", linestyle="--",
                       linewidth=0.8,
                       label=f"λ₂ = {evals[1]:.4f} (algebraic connectivity)")
            # Spectral gaps
            gaps = np.diff(evals[1:])
            if len(gaps) > 0:
                max_gap_idx = np.argmax(gaps) + 2  # +2 for indexing offset
                ax.annotate(
                    f"max gap at k={max_gap_idx}\n({gaps.max():.3f})",
                    xy=(max_gap_idx, evals[max_gap_idx]),
                    xytext=(max_gap_idx + 2, evals[max_gap_idx] * 1.2),
                    arrowprops=dict(arrowstyle="->", color="#228833"),
                    fontsize=6, color="#228833",
                )

        ax.set_xlabel("Eigenvalue index")
        ax.set_ylabel("λ")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=6)

        return fig, ax


# ═══════════════════════════════════════════════════════════════════════════
# 6. Graph Layout (Force-directed / Community-coloured)
# ═══════════════════════════════════════════════════════════════════════════

def plot_graph_layout(
    adjacency: np.ndarray,
    community_labels: Optional[np.ndarray] = None,
    node_sizes: Optional[np.ndarray] = None,
    layout: str = "spring",
    edge_alpha_range: Tuple[float, float] = (0.03, 0.6),
    title: str = "",
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (8, 8),
) -> Tuple[plt.Figure, plt.Axes]:
    """Force-directed graph layout with community colouring.

    Parameters
    ----------
    layout : 'spring', 'kamada_kawai', 'spectral', 'circular'
    """
    import networkx as nx

    with _apply_pub_style():
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        G = nx.from_numpy_array(adjacency)
        rm = [(u, v) for u, v, d in G.edges(data=True)
              if d.get("weight", 0) <= 0]
        G.remove_edges_from(rm)

        # Layout
        layout_fns = {
            "spring": lambda: nx.spring_layout(G, weight="weight", seed=42,
                                                iterations=100),
            "kamada_kawai": lambda: nx.kamada_kawai_layout(G, weight="weight"),
            "spectral": lambda: nx.spectral_layout(G, weight="weight"),
            "circular": lambda: nx.circular_layout(G),
        }
        pos = layout_fns.get(layout, layout_fns["spring"])()

        # Node colours from community labels
        if community_labels is not None:
            n_comm = len(np.unique(community_labels))
            palette = _community_colors(n_comm)
            node_color = [palette[int(community_labels[i]) % len(palette)]
                          for i in range(G.number_of_nodes())]
        else:
            node_color = "#4477AA"

        # Node sizes from degree if not provided
        if node_sizes is None:
            degrees = np.array([G.degree(i) for i in range(G.number_of_nodes())])
            mn, mx = degrees.min(), degrees.max()
            if mx > mn:
                node_sizes = 30 + 200 * (degrees - mn) / (mx - mn)
            else:
                node_sizes = np.full(G.number_of_nodes(), 60)

        # Edge alpha from weight
        edges = G.edges(data=True)
        if edges:
            weights = np.array([d["weight"] for _, _, d in edges])
            wmin, wmax = weights.min(), weights.max()
            if wmax > wmin:
                alphas = (edge_alpha_range[0] +
                          (edge_alpha_range[1] - edge_alpha_range[0]) *
                          (weights - wmin) / (wmax - wmin))
            else:
                alphas = np.full(len(weights), 0.3)

            # Draw edges with individual alpha
            for (u, v, d), a in zip(edges, alphas):
                x = [pos[u][0], pos[v][0]]
                y = [pos[u][1], pos[v][1]]
                ax.plot(x, y, color="#999999", alpha=a, linewidth=0.4,
                        zorder=1)

        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_color,
                               node_size=node_sizes, edgecolors="white",
                               linewidths=0.5, zorder=2)

        if title:
            ax.set_title(title, fontweight="bold")
        ax.set_axis_off()

        return fig, ax


# ═══════════════════════════════════════════════════════════════════════════
# 7. Rich-Club Curve
# ═══════════════════════════════════════════════════════════════════════════

def plot_rich_club_curve(
    rich_club_normalized: Dict[int, float],
    rich_club_raw: Optional[Dict[int, float]] = None,
    title: str = "Rich-club coefficient",
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Normalised rich-club coefficient versus degree."""
    with _apply_pub_style():
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 3.5))
        else:
            fig = ax.figure

        k = sorted(rich_club_normalized.keys())
        phi = [rich_club_normalized[ki] for ki in k]

        ax.plot(k, phi, "o-", color="#4477AA", markersize=3, linewidth=1,
                label="Normalised φ(k)")
        ax.axhline(1.0, color="#EE6677", linestyle="--", linewidth=0.8,
                    label="Random baseline")

        # Fill above 1
        ax.fill_between(k, 1.0, phi,
                         where=[p > 1 for p in phi],
                         alpha=0.2, color="#4477AA",
                         label="Rich-club regime")

        ax.set_xlabel("Degree k")
        ax.set_ylabel("Normalised φ(k)")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=6)

        return fig, ax


# ═══════════════════════════════════════════════════════════════════════════
# 8. NBS Result Overlay
# ═══════════════════════════════════════════════════════════════════════════

def plot_nbs_result(
    significant_mask: np.ndarray,
    node_coords: np.ndarray,
    stat_matrix: Optional[np.ndarray] = None,
    title: str = "NBS significant network",
    display_mode: str = "lyrz",
    **kwargs,
) -> Any:
    """Overlay NBS-significant edges on glass brain."""
    try:
        from nilearn.plotting import plot_connectome
    except ImportError:
        raise ImportError("nilearn required for NBS visualization.")

    with _apply_pub_style():
        # Use stat_matrix values for significant edges, zero elsewhere
        if stat_matrix is not None:
            display_adj = np.where(significant_mask, stat_matrix, 0.0)
        else:
            display_adj = significant_mask.astype(float)

        display = plot_connectome(
            display_adj, node_coords,
            edge_cmap="RdBu_r",
            node_size=20,
            display_mode=display_mode,
            title=title,
            edge_threshold=0.01,  # show all non-zero
            **kwargs,
        )
        return display


# ═══════════════════════════════════════════════════════════════════════════
# 9. Persistence Diagram & Betti Curves
# ═══════════════════════════════════════════════════════════════════════════

def plot_persistence_diagram(
    diagrams: List[np.ndarray],
    betti_numbers: Optional[np.ndarray] = None,
    filtration_values: Optional[np.ndarray] = None,
    title: str = "Persistent homology",
    figsize: Tuple[float, float] = (12, 4.5),
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """Persistence diagram + Betti curves side by side."""
    with _apply_pub_style():
        n_panels = 2 if betti_numbers is not None else 1
        fig, axes = plt.subplots(1, n_panels, figsize=figsize)
        if n_panels == 1:
            axes = [axes]

        # ── Panel A: Persistence diagram ──
        ax = axes[0]
        colors_dim = ["#4477AA", "#EE6677", "#228833"]
        for dim, dgm in enumerate(diagrams):
            finite = dgm[np.isfinite(dgm[:, 1])]
            if len(finite) == 0:
                continue
            ax.scatter(finite[:, 0], finite[:, 1],
                       s=12, alpha=0.6,
                       color=colors_dim[dim % len(colors_dim)],
                       label=f"H{dim}", zorder=3)

        # Diagonal
        lim_max = max(
            np.max(dgm[np.isfinite(dgm[:, 1]), 1])
            for dgm in diagrams if np.any(np.isfinite(dgm[:, 1]))
        )
        ax.plot([0, lim_max], [0, lim_max], "k--", linewidth=0.5, alpha=0.5)
        ax.set_xlabel("Birth")
        ax.set_ylabel("Death")
        ax.set_title("Persistence diagram", fontweight="bold")
        ax.legend(fontsize=6)
        ax.set_aspect("equal")

        # ── Panel B: Betti curves ──
        if betti_numbers is not None and filtration_values is not None:
            ax2 = axes[1]
            for dim in range(betti_numbers.shape[1]):
                ax2.plot(filtration_values, betti_numbers[:, dim],
                         color=colors_dim[dim % len(colors_dim)],
                         linewidth=1.2, label=f"β{dim}")
            ax2.set_xlabel("Filtration threshold")
            ax2.set_ylabel("Betti number")
            ax2.set_title("Betti curves", fontweight="bold")
            ax2.legend(fontsize=6)

        fig.suptitle(title, fontsize=9, fontweight="bold", y=1.02)
        fig.tight_layout()
        return fig, axes


# ═══════════════════════════════════════════════════════════════════════════
# 10. Group Metric Comparison (violin/bar)
# ═══════════════════════════════════════════════════════════════════════════

def plot_metric_comparison(
    comparison_results: Dict[str, Dict[str, float]],
    group_names: Tuple[str, str] = ("Controls", "Patients"),
    title: str = "Graph metric comparison",
    figsize: Optional[Tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Bar plot comparing graph metrics between groups with significance."""
    with _apply_pub_style():
        metrics = list(comparison_results.keys())
        n = len(metrics)
        if figsize is None:
            figsize = (max(4, n * 1.2), 4)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        x = np.arange(n)
        width = 0.35

        g1_vals = [comparison_results[m]["mean_g1"] for m in metrics]
        g2_vals = [comparison_results[m]["mean_g2"] for m in metrics]

        bars1 = ax.bar(x - width / 2, g1_vals, width, label=group_names[0],
                        color="#4477AA", alpha=0.85, edgecolor="white")
        bars2 = ax.bar(x + width / 2, g2_vals, width, label=group_names[1],
                        color="#EE6677", alpha=0.85, edgecolor="white")

        # Significance stars
        for i, m in enumerate(metrics):
            p = comparison_results[m]["p_value"]
            y_max = max(g1_vals[i], g2_vals[i]) * 1.05
            if p < 0.001:
                star = "***"
            elif p < 0.01:
                star = "**"
            elif p < 0.05:
                star = "*"
            else:
                star = "n.s."
            ax.text(i, y_max, star, ha="center", va="bottom", fontsize=7,
                    fontweight="bold" if p < 0.05 else "normal")

        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("_", "\n") for m in metrics],
                           fontsize=6)
        ax.set_ylabel("Metric value")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=6)

        return fig, ax


# ═══════════════════════════════════════════════════════════════════════════
# 11. Small-world Visualization
# ═══════════════════════════════════════════════════════════════════════════

def plot_small_world(
    sigma: float = np.nan,
    omega: float = np.nan,
    title: str = "Small-world topology",
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Visualise small-world indices σ and ω with interpretation."""
    with _apply_pub_style():
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 3))
        else:
            fig = ax.figure

        # ω axis
        ax.axhline(0, color="grey", linewidth=0.5)
        ax.axvspan(-1, -0.5, alpha=0.08, color="#4477AA",
                    label="Lattice-like")
        ax.axvspan(-0.5, 0.5, alpha=0.08, color="#228833",
                    label="Small-world")
        ax.axvspan(0.5, 1, alpha=0.08, color="#EE6677",
                    label="Random-like")

        if np.isfinite(omega):
            ax.axvline(omega, color="#CC3311", linewidth=2,
                       label=f"ω = {omega:.3f}")
            ax.plot(omega, 0, "o", color="#CC3311", markersize=10, zorder=5)

        ax.set_xlim(-1.1, 1.1)
        ax.set_xlabel("ω (small-world omega)")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=6, loc="upper left")

        # Add σ as text annotation
        if np.isfinite(sigma):
            ax.text(0.95, 0.95, f"σ = {sigma:.2f}\n(>1 = small-world)",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=7, bbox=dict(boxstyle="round,pad=0.3",
                                          facecolor="lightyellow", alpha=0.8))

        return fig, ax


# ═══════════════════════════════════════════════════════════════════════════
# 12. Surface-projected Graph Metric
# ═══════════════════════════════════════════════════════════════════════════

def plot_surface_metric(
    node_metric: np.ndarray,
    atlas_img: Any,
    metric_name: str = "Degree centrality",
    cmap: str = "hot",
    threshold: float = 0.1,
    views: List[str] = None,
    figsize: Tuple[float, float] = (12, 5),
) -> Tuple[plt.Figure, Any]:
    """Project a node-level metric onto cortical surface via nilearn.

    Parameters
    ----------
    node_metric : (R,) metric value per ROI
    atlas_img : nibabel Nifti1Image of the parcellation atlas
    """
    try:
        import nibabel as nib
        from nilearn import surface, datasets, plotting
    except ImportError:
        raise ImportError("nilearn required for surface projection.")

    with _apply_pub_style():
        if views is None:
            views = ["lateral", "medial"]

        # Build metric volume from atlas parcellation
        atlas_data = atlas_img.get_fdata()
        metric_vol = np.zeros_like(atlas_data)
        labels = np.unique(atlas_data[atlas_data > 0]).astype(int)
        for i, val in enumerate(node_metric):
            if i < len(labels):
                metric_vol[atlas_data == labels[i]] = val
        metric_img = nib.Nifti1Image(metric_vol, atlas_img.affine)

        fsaverage = datasets.fetch_surf_fsaverage()

        fig, axes = plt.subplots(1, len(views) * 2, figsize=figsize,
                                  subplot_kw={"projection": "3d"})

        idx = 0
        for hemi, hemi_mesh in [("left", fsaverage.pial_left),
                                 ("right", fsaverage.pial_right)]:
            texture = surface.vol_to_surf(metric_img, hemi_mesh)
            for view in views:
                if idx < len(axes):
                    plotting.plot_surf_stat_map(
                        getattr(fsaverage, f"infl_{hemi}"),
                        texture, hemi=hemi, view=view,
                        cmap=_get_cmap(cmap), threshold=threshold,
                        bg_map=getattr(fsaverage, f"sulc_{hemi}"),
                        axes=axes[idx], colorbar=False,
                    )
                    axes[idx].set_title(f"{hemi} {view}", fontsize=6)
                    idx += 1

        fig.suptitle(metric_name, fontsize=9, fontweight="bold")
        fig.tight_layout()
        return fig, axes


# ═══════════════════════════════════════════════════════════════════════════
# Multi-panel composite
# ═══════════════════════════════════════════════════════════════════════════

def plot_graph_composite(
    graph_result,
    metrics=None,
    node_coords: Optional[np.ndarray] = None,
    title: str = "Spectral morphometric connectivity",
    figsize: Tuple[float, float] = (183 / 25.4, 220 / 25.4),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Multi-panel composite figure combining adjacency matrix, graph
    layout, eigenspectrum, and edge-weight distribution.

    This is the recommended figure for a methods/results paper.
    """
    with _apply_pub_style():
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1, 1])

        # A: Adjacency matrix
        ax_adj = fig.add_subplot(gs[0, 0])
        W = (graph_result.thresholded if graph_result.thresholded is not None
             else graph_result.adjacency)
        im = ax_adj.imshow(W, cmap=_get_cmap("RdBu_r"), aspect="equal")
        plt.colorbar(im, ax=ax_adj, fraction=0.046, pad=0.04)
        ax_adj.set_title("a  Similarity matrix", fontweight="bold",
                         fontsize=8, loc="left")

        # B: Graph layout
        ax_layout = fig.add_subplot(gs[0, 1])
        cl = metrics.community_labels if metrics else None
        plot_graph_layout(W, community_labels=cl, ax=ax_layout)
        ax_layout.set_title("b  Graph layout", fontweight="bold",
                            fontsize=8, loc="left")

        # C: Edge distribution
        ax_dist = fig.add_subplot(gs[1, 0])
        plot_edge_weight_distribution(
            graph_result.adjacency, ax=ax_dist,
            title="c  Edge-weight distribution")

        # D: Laplacian spectrum
        ax_spec = fig.add_subplot(gs[1, 1])
        if metrics and metrics.laplacian_eigenvalues is not None:
            plot_laplacian_spectrum(
                metrics.laplacian_eigenvalues, ax=ax_spec,
                title="d  Laplacian eigenspectrum")
        else:
            ax_spec.text(0.5, 0.5, "No spectral data", ha="center",
                         va="center", transform=ax_spec.transAxes)

        # E: Rich-club
        ax_rc = fig.add_subplot(gs[2, 0])
        if metrics and metrics.rich_club_normalized:
            plot_rich_club_curve(metrics.rich_club_normalized, ax=ax_rc,
                                title="e  Rich-club")
        else:
            ax_rc.text(0.5, 0.5, "No rich-club data", ha="center",
                       va="center", transform=ax_rc.transAxes)

        # F: Small-world
        ax_sw = fig.add_subplot(gs[2, 1])
        plot_small_world(
            sigma=metrics.sigma if metrics else np.nan,
            omega=metrics.omega if metrics else np.nan,
            ax=ax_sw, title="f  Small-world topology")

        fig.suptitle(title, fontsize=10, fontweight="bold", y=1.01)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info("Saved composite figure to %s", save_path)

        return fig


# ═══════════════════════════════════════════════════════════════════════════
# Helper: save figure in multiple formats
# ═══════════════════════════════════════════════════════════════════════════

def save_graph_figure(
    fig: plt.Figure,
    path_stem: str,
    formats: Tuple[str, ...] = ("pdf", "png"),
    dpi: int = 300,
) -> List[str]:
    """Save a figure in multiple publication formats.

    Parameters
    ----------
    path_stem : str — path without extension (e.g. 'figures/fig3')
    formats : tuple of extensions

    Returns
    -------
    list of saved file paths
    """
    saved = []
    for fmt in formats:
        fpath = f"{path_stem}.{fmt}"
        fig.savefig(fpath, dpi=dpi, bbox_inches="tight",
                    facecolor="white", transparent=False)
        saved.append(fpath)
        logger.info("Saved %s", fpath)
    return saved
