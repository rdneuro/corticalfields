"""
Bayesian visualization toolkit for CorticalFields.

20 publication-quality plotting functions for Bayesian analysis
results, designed for Nature/NeuroImage/JAMA Neurology standards.

Architecture
------------
Functions are organized by purpose:

  §A POSTERIOR INSPECTION (5 functions)
      plot_posterior_density, plot_forest, plot_ridgeline,
      plot_posterior_hdi_rope, plot_prior_posterior_update

  §B DIAGNOSTICS (5 functions)
      plot_trace_rank, plot_pair_divergences, plot_energy,
      plot_loo_pit, plot_rhat_ess_panel

  §C MODEL-SPECIFIC (5 functions)
      plot_shrinkage, plot_coefficient_path, plot_mediation_diagram,
      plot_group_comparison, plot_hierarchical_caterpillar

  §D ADVANCED (5 functions)
      plot_model_comparison, plot_posterior_predictive_check,
      plot_sensitivity, plot_correlation_matrix_posterior,
      plot_brain_posterior_map

Style
-----
All functions follow the CorticalFields visual identity:
  - Colorblind-safe palettes (Okabe-Ito, viridis)
  - DejaVu Sans / Arial typography
  - 600 DPI PDF output with Type 42 fonts
  - HDI shading in muted blue/orange
  - ROPE region in light gray
  - Consistent axis labeling and annotation

Dependencies
------------
matplotlib, seaborn, arviz, numpy, pandas
Optional: surfplot (for brain surface maps)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import Normalize

logger = logging.getLogger(__name__)

# ── Style configuration ─────────────────────────────────────────────────

# Okabe-Ito colorblind-safe palette
OI = {
    "orange":   "#E69F00",
    "skyblue":  "#56B4E9",
    "green":    "#009E73",
    "yellow":   "#F0E442",
    "blue":     "#0072B2",
    "vermilion":"#D55E00",
    "purple":   "#CC79A7",
    "black":    "#000000",
}

# Yeo-7 network colors
YEO7 = {
    "Vis":        "#781286",
    "SomMot":     "#4682B4",
    "DorsAttn":   "#00760E",
    "SalVentAttn":"#C43AFA",
    "Limbic":     "#9CB86E",
    "Cont":       "#E69422",
    "Default":    "#CD3E4E",
}


def _setup_style():
    """Apply publication-quality matplotlib defaults."""
    try:
        plt.style.use(["science", "no-latex"])
    except Exception:
        pass
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# ═══════════════════════════════════════════════════════════════════════════
#  §A  POSTERIOR INSPECTION
# ═══════════════════════════════════════════════════════════════════════════


def plot_posterior_density(
    samples: np.ndarray,
    var_name: str = "θ",
    hdi_prob: float = 0.94,
    rope: Optional[Tuple[float, float]] = None,
    ref_val: Optional[float] = 0.0,
    ax: Optional[plt.Axes] = None,
    color: str = OI["blue"],
    figsize: Tuple[float, float] = (4, 2.5),
) -> plt.Axes:
    """
    Posterior density with HDI interval and optional ROPE.

    Shows the posterior distribution as a KDE with shaded HDI region,
    vertical reference line, and ROPE if specified. Annotates the
    HDI bounds, posterior mean, and pd (probability of direction).

    Parameters
    ----------
    samples : 1D array of posterior draws
    var_name : display name for the parameter
    hdi_prob : HDI probability (default 94%)
    rope : (low, high) tuple for ROPE shading
    ref_val : reference value (vertical dashed line, typically 0)
    """
    _setup_style()
    import arviz as az

    samples = np.asarray(samples).ravel()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # KDE
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(samples)
    x = np.linspace(samples.min() - 0.2 * np.ptp(samples),
                     samples.max() + 0.2 * np.ptp(samples), 500)
    density = kde(x)
    ax.plot(x, density, color=color, linewidth=1.5)
    ax.fill_between(x, density, alpha=0.15, color=color)

    # HDI
    hdi = az.hdi(samples, hdi_prob=hdi_prob)
    mask = (x >= hdi[0]) & (x <= hdi[1])
    ax.fill_between(x[mask], density[mask], alpha=0.35, color=color,
                    label=f"{int(hdi_prob*100)}% HDI [{hdi[0]:.3f}, {hdi[1]:.3f}]")

    # ROPE
    if rope is not None:
        ax.axvspan(rope[0], rope[1], alpha=0.1, color="gray",
                   label=f"ROPE [{rope[0]}, {rope[1]}]")
        pct_in = np.mean((samples >= rope[0]) & (samples <= rope[1]))
        ax.text(np.mean(rope), ax.get_ylim()[1] * 0.9,
                f"{pct_in:.0%}\nin ROPE", ha="center", fontsize=7, color="gray")

    # Reference
    if ref_val is not None:
        ax.axvline(ref_val, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)

    # Annotations
    pd_val = max(np.mean(samples > 0), np.mean(samples < 0))
    ax.set_title(f"{var_name}  |  mean={samples.mean():.3f}  |  pd={pd_val:.2%}",
                 fontsize=9)
    ax.set_xlabel(var_name)
    ax.set_ylabel("Density")
    ax.legend(fontsize=7, frameon=False)

    return ax


def plot_forest(
    idata,
    var_names: List[str] = None,
    feature_names: Optional[List[str]] = None,
    hdi_prob: float = 0.94,
    rope: Optional[Tuple[float, float]] = None,
    sort_by: str = "mean",
    top_n: Optional[int] = None,
    colors: Optional[Dict[str, str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """
    Forest plot with HDI intervals, colored by Yeo-7 network.

    Shows each coefficient as a point (posterior mean) with thick
    bar (50% HDI) and thin bar (94% HDI). Optional ROPE band and
    Yeo-7 network coloring for brain ROIs.

    Parameters
    ----------
    idata : arviz.InferenceData
    var_names : variables to plot (default: ["beta"])
    feature_names : labels for each coefficient
    sort_by : 'mean', 'abs_mean', or 'hdi_width'
    top_n : show only top N by absolute value
    colors : dict mapping feature name → color
    """
    _setup_style()
    import arviz as az

    if var_names is None:
        var_names = ["beta"]

    # Extract posterior
    post = idata.posterior[var_names[0]].values
    flat = post.reshape(-1, post.shape[-1])
    n_features = flat.shape[1]

    if feature_names is None:
        feature_names = [f"β_{i}" for i in range(n_features)]

    means = flat.mean(axis=0)
    hdi50 = np.array([az.hdi(flat[:, i], hdi_prob=0.50) for i in range(n_features)])
    hdi94 = np.array([az.hdi(flat[:, i], hdi_prob=hdi_prob) for i in range(n_features)])

    # Sorting
    if sort_by == "abs_mean":
        order = np.argsort(np.abs(means))
    elif sort_by == "hdi_width":
        order = np.argsort(hdi94[:, 1] - hdi94[:, 0])
    else:
        order = np.argsort(means)

    if top_n is not None:
        top_idx = np.argsort(np.abs(means))[-top_n:]
        order = np.array([i for i in order if i in top_idx])

    if figsize is None:
        figsize = (5, max(3, len(order) * 0.25))

    fig, ax = plt.subplots(figsize=figsize)

    for j, idx in enumerate(order):
        name = feature_names[idx]
        c = OI["blue"]
        if colors and name in colors:
            c = colors[name]
        elif any(net in name for net in YEO7):
            for net, nc in YEO7.items():
                if net in name:
                    c = nc
                    break

        # Thin bar: 94% HDI
        ax.plot([hdi94[idx, 0], hdi94[idx, 1]], [j, j],
                color=c, linewidth=1, alpha=0.7)
        # Thick bar: 50% HDI
        ax.plot([hdi50[idx, 0], hdi50[idx, 1]], [j, j],
                color=c, linewidth=3, alpha=0.9)
        # Point: mean
        ax.plot(means[idx], j, "o", color=c, markersize=4, zorder=5)

    # ROPE
    if rope is not None:
        ax.axvspan(rope[0], rope[1], alpha=0.08, color="gray", zorder=0)

    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feature_names[i] for i in order], fontsize=7)
    ax.set_xlabel("Coefficient (posterior)")
    ax.set_title(f"Forest Plot — {int(hdi_prob*100)}% HDI")

    fig.tight_layout()
    return fig


def plot_ridgeline(
    idata,
    var_names: List[str] = None,
    feature_names: Optional[List[str]] = None,
    overlap: float = 0.6,
    figsize: Tuple[float, float] = (6, 8),
) -> plt.Figure:
    """
    Ridgeline (joy) plot of posterior densities stacked vertically.

    Each coefficient's posterior is shown as a filled KDE, slightly
    overlapping the one below, creating an elegant layered view
    of all posteriors simultaneously.
    """
    _setup_style()
    from scipy.stats import gaussian_kde

    if var_names is None:
        var_names = ["beta"]

    post = idata.posterior[var_names[0]].values
    flat = post.reshape(-1, post.shape[-1])
    n_feat = flat.shape[1]

    if feature_names is None:
        feature_names = [f"β_{i}" for i in range(n_feat)]

    fig, ax = plt.subplots(figsize=figsize)
    x_all = np.linspace(flat.min(), flat.max(), 300)

    for i in range(n_feat):
        kde = gaussian_kde(flat[:, i])
        density = kde(x_all)
        density = density / density.max() * overlap
        ax.fill_between(x_all, i, i + density, alpha=0.5,
                        color=plt.cm.viridis(i / n_feat))
        ax.plot(x_all, i + density, color="black", linewidth=0.5)

    ax.set_yticks(range(n_feat))
    ax.set_yticklabels(feature_names, fontsize=7)
    ax.set_xlabel("Coefficient value")
    ax.set_title("Ridgeline Plot — Posterior Densities")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")

    fig.tight_layout()
    return fig


def plot_posterior_hdi_rope(
    idata,
    var_name: str = "beta",
    feature_names: Optional[List[str]] = None,
    hdi_prob: float = 0.94,
    rope: Tuple[float, float] = (-0.1, 0.1),
    figsize: Tuple[float, float] = (10, 4),
) -> plt.Figure:
    """
    Combined HDI + ROPE decision plot.

    For each parameter, shows the 94% HDI as a bar and classifies
    the result as: REJECT H₀ (HDI outside ROPE), ACCEPT H₀
    (HDI inside ROPE), or UNDECIDED (HDI overlaps ROPE boundary).
    """
    _setup_style()
    import arviz as az

    post = idata.posterior[var_name].values
    flat = post.reshape(-1, post.shape[-1])
    n = flat.shape[1]

    if feature_names is None:
        feature_names = [f"β_{i}" for i in range(n)]

    fig, ax = plt.subplots(figsize=figsize)

    for i in range(n):
        hdi = az.hdi(flat[:, i], hdi_prob=hdi_prob)

        # Decision
        if hdi[0] > rope[1] or hdi[1] < rope[0]:
            color, decision = OI["vermilion"], "reject"
        elif hdi[0] >= rope[0] and hdi[1] <= rope[1]:
            color, decision = OI["green"], "accept"
        else:
            color, decision = OI["yellow"], "undecided"

        ax.plot([hdi[0], hdi[1]], [i, i], color=color, linewidth=3, alpha=0.8)
        ax.plot(flat[:, i].mean(), i, "o", color=color, markersize=5, zorder=5)

    ax.axvspan(rope[0], rope[1], alpha=0.1, color="gray", label="ROPE")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_yticks(range(n))
    ax.set_yticklabels(feature_names, fontsize=7)
    ax.set_xlabel("Coefficient")
    ax.set_title(f"HDI + ROPE Decision Plot — {int(hdi_prob*100)}% HDI")
    ax.legend(fontsize=7)

    fig.tight_layout()
    return fig


def plot_prior_posterior_update(
    prior_samples: np.ndarray,
    posterior_samples: np.ndarray,
    var_name: str = "θ",
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (4.5, 3),
) -> plt.Axes:
    """
    Prior → Posterior update visualization.

    Shows both distributions overlaid with an arrow indicating
    the direction and magnitude of Bayesian updating. Quantifies
    shrinkage = 1 - Var(posterior)/Var(prior).
    """
    _setup_style()
    from scipy.stats import gaussian_kde

    prior = np.asarray(prior_samples).ravel()
    post = np.asarray(posterior_samples).ravel()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    x = np.linspace(min(prior.min(), post.min()) - np.ptp(prior) * 0.2,
                     max(prior.max(), post.max()) + np.ptp(prior) * 0.2, 300)

    kde_prior = gaussian_kde(prior)(x)
    kde_post = gaussian_kde(post)(x)

    ax.fill_between(x, kde_prior, alpha=0.2, color=OI["orange"], label="Prior")
    ax.plot(x, kde_prior, color=OI["orange"], linewidth=1.2, linestyle="--")
    ax.fill_between(x, kde_post, alpha=0.35, color=OI["blue"], label="Posterior")
    ax.plot(x, kde_post, color=OI["blue"], linewidth=1.5)

    # Shrinkage annotation
    shrinkage = 1 - post.var() / max(prior.var(), 1e-8)
    ax.text(0.02, 0.95, f"Shrinkage: {shrinkage:.0%}",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_xlabel(var_name)
    ax.set_ylabel("Density")
    ax.set_title(f"Prior → Posterior Update: {var_name}")
    ax.legend(fontsize=7, frameon=False)

    return ax


# ═══════════════════════════════════════════════════════════════════════════
#  §B  DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════


def plot_trace_rank(
    idata,
    var_names: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """
    Trace + rank plots (recommended over raw trace plots).

    Rank plots (Vehtari et al. 2021) are more diagnostic than raw
    trace plots — rank-normalized chains should appear as uniform
    histograms when well-mixed.
    """
    _setup_style()
    import arviz as az
    return az.plot_trace(idata, var_names=var_names, kind="rank_bars",
                         figsize=figsize).ravel()[0].figure


def plot_pair_divergences(
    idata,
    var_names: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (8, 8),
) -> plt.Figure:
    """
    Pair plot with divergence markers.

    Divergent transitions (red) indicate geometric pathologies
    in the posterior — typically funnel-shaped geometries from
    centered parameterization of hierarchical models.
    """
    _setup_style()
    import arviz as az
    axes = az.plot_pair(idata, var_names=var_names, divergences=True,
                        figsize=figsize)
    return axes.ravel()[0].figure


def plot_energy(
    idata,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (4, 3),
) -> plt.Axes:
    """
    Energy distribution diagnostic.

    The marginal energy (π_E) and energy transition (π_ΔE)
    distributions should overlap substantially. Large separation
    indicates poor exploration. BFMI < 0.3 is problematic.
    """
    _setup_style()
    import arviz as az

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    az.plot_energy(idata, ax=ax)
    bfmi = az.bfmi(idata)
    ax.set_title(f"Energy — BFMI: {min(bfmi):.3f}")
    return ax


def plot_loo_pit(
    idata,
    y: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (4, 3),
) -> plt.Axes:
    """
    LOO-PIT (Probability Integral Transform) plot.

    Uniform = well-calibrated model.
    U-shaped = overdispersed (model too uncertain).
    Inverted-U = underdispersed (model too confident).
    """
    _setup_style()
    import arviz as az

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    az.plot_loo_pit(idata, y=y, ecdf=True, ax=ax)
    ax.set_title("LOO-PIT Calibration")
    return ax


def plot_rhat_ess_panel(
    idata,
    var_names: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (8, 3.5),
) -> plt.Figure:
    """
    Combined R̂ + ESS diagnostic panel.

    Left: R̂ values per parameter (threshold at 1.01).
    Right: ESS bulk + tail per parameter (threshold at 400).
    """
    _setup_style()
    import arviz as az

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # R-hat
    rhat = az.rhat(idata, var_names=var_names)
    for v in rhat.data_vars:
        vals = rhat[v].values.ravel()
        ax1.barh(range(len(vals)), vals, color=OI["blue"], alpha=0.7)
    ax1.axvline(1.01, color=OI["vermilion"], linewidth=1, linestyle="--",
                label="R̂ = 1.01")
    ax1.set_xlabel("R̂")
    ax1.set_title("R̂ (split, rank-normalized)")
    ax1.legend(fontsize=7)

    # ESS
    ess_b = az.ess(idata, var_names=var_names, method="bulk")
    ess_t = az.ess(idata, var_names=var_names, method="tail")
    for v in ess_b.data_vars:
        vals_b = ess_b[v].values.ravel()
        vals_t = ess_t[v].values.ravel()
        y_pos = np.arange(len(vals_b))
        ax2.barh(y_pos - 0.15, vals_b, 0.3, color=OI["blue"], alpha=0.7, label="Bulk")
        ax2.barh(y_pos + 0.15, vals_t, 0.3, color=OI["orange"], alpha=0.7, label="Tail")
    ax2.axvline(400, color=OI["vermilion"], linewidth=1, linestyle="--",
                label="ESS = 400")
    ax2.set_xlabel("ESS")
    ax2.set_title("Effective Sample Size")
    ax2.legend(fontsize=7)

    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  §C  MODEL-SPECIFIC
# ═══════════════════════════════════════════════════════════════════════════


def plot_shrinkage(
    idata,
    feature_names: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (8, 4),
) -> plt.Figure:
    """
    Horseshoe shrinkage factor plot.

    Shows κ_j = 1/(1 + λ_j² · τ²) per feature. Values near 1
    indicate full shrinkage (noise); near 0 indicate signal.
    Bimodal distribution expected: most features shrunk, few unshrunk.
    """
    _setup_style()

    kappa = idata.posterior["kappa"].values.mean(axis=(0, 1))
    n = len(kappa)
    if feature_names is None:
        feature_names = [f"x_{i}" for i in range(n)]

    order = np.argsort(kappa)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize,
                                    gridspec_kw={"width_ratios": [3, 1]})

    # Bar plot
    colors = [OI["vermilion"] if kappa[i] < 0.5 else OI["blue"] for i in order]
    ax1.barh(range(n), kappa[order], color=colors, alpha=0.8)
    ax1.axvline(0.5, color="gray", linewidth=1, linestyle="--",
                label="Signal/noise threshold")
    ax1.set_yticks(range(n))
    ax1.set_yticklabels([feature_names[i] for i in order], fontsize=6)
    ax1.set_xlabel("Shrinkage κ (1 = fully shrunk)")
    ax1.set_title("Horseshoe Shrinkage Factors")
    ax1.legend(fontsize=7)

    # Histogram
    ax2.hist(kappa, bins=20, orientation="horizontal", color=OI["blue"], alpha=0.6)
    ax2.axhline(0.5, color="gray", linewidth=1, linestyle="--")
    ax2.set_xlabel("Count")
    ax2.set_title("Distribution")

    m_eff = float(idata.posterior["m_eff"].values.mean())
    fig.suptitle(f"Effective non-zero features: m_eff = {m_eff:.1f}", fontsize=10)
    fig.tight_layout()
    return fig


def plot_coefficient_path(
    models: Dict[str, Any],
    feature_names: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (6, 4),
) -> plt.Figure:
    """
    Coefficient path across models (regularization path analog).

    Shows how coefficients change across different model specifications
    (e.g., horseshoe with varying n_relevant, or ridge vs. horseshoe).
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=figsize)

    model_names = list(models.keys())
    for i, (name, idata) in enumerate(models.items()):
        post = idata.posterior["beta"].values
        means = post.reshape(-1, post.shape[-1]).mean(axis=0)
        ax.plot(range(len(means)), means, "o-", label=name,
                alpha=0.7, markersize=4)

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Feature index")
    ax.set_ylabel("Posterior mean coefficient")
    ax.set_title("Coefficient Path Across Models")
    ax.legend(fontsize=7)

    fig.tight_layout()
    return fig


def plot_mediation_diagram(
    idata,
    labels: Tuple[str, str, str] = ("X", "M", "Y"),
    figsize: Tuple[float, float] = (6, 4),
) -> plt.Figure:
    """
    Path diagram for Bayesian mediation analysis.

    Shows X → M (path a), M → Y (path b), X → Y (path c'),
    and the indirect effect a×b with posterior summaries.
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-1, 2)
    ax.axis("off")

    # Boxes
    for x, y, label in [(0, 0, labels[0]), (1.5, 1.5, labels[1]), (3, 0, labels[2])]:
        ax.add_patch(plt.Rectangle((x - 0.4, y - 0.25), 0.8, 0.5,
                                    fill=True, facecolor=OI["skyblue"],
                                    edgecolor="black", linewidth=1.5))
        ax.text(x, y, label, ha="center", va="center", fontsize=12, fontweight="bold")

    # Paths with posterior values
    for var, start, end in [("a", (0.4, 0.25), (1.1, 1.25)),
                            ("b", (1.9, 1.25), (2.6, 0.25)),
                            ("c_prime", (0.4, -0.05), (2.6, -0.05))]:
        post = idata.posterior[var].values.ravel()
        m, lo, hi = post.mean(), np.percentile(post, 3), np.percentile(post, 97)
        ax.annotate("", xy=end, xytext=start,
                    arrowprops=dict(arrowstyle="->", lw=2,
                                    color=OI["vermilion"] if abs(m) > 0.1 else OI["blue"]))
        mid = ((start[0]+end[0])/2, (start[1]+end[1])/2)
        ax.text(mid[0], mid[1] + 0.15, f"{var}={m:.2f}\n[{lo:.2f}, {hi:.2f}]",
                ha="center", fontsize=8, style="italic")

    # Indirect effect
    ab = idata.posterior["indirect_ab"].values.ravel()
    ax.text(1.5, -0.8, f"Indirect (a×b) = {ab.mean():.3f} "
            f"[{np.percentile(ab, 3):.3f}, {np.percentile(ab, 97):.3f}]",
            ha="center", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=OI["yellow"], alpha=0.4))

    ax.set_title("Bayesian Mediation Diagram", fontsize=11, fontweight="bold")
    return fig


def plot_group_comparison(
    idata,
    group_names: Tuple[str, str] = ("Group 1", "Group 2"),
    figsize: Tuple[float, float] = (10, 6),
) -> plt.Figure:
    """
    BEST group comparison multi-panel figure.

    4 panels: (A) posterior distributions of means, (B) posterior
    of difference, (C) posterior of effect size (Cohen's d),
    (D) posterior of ν (normality parameter).
    """
    _setup_style()
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    for var, ax, title, color in [
        (f"mu_{group_names[0]}", axes[0, 0], f"Mean: {group_names[0]}", OI["blue"]),
        (f"mu_{group_names[1]}", axes[0, 0], f"Mean: {group_names[1]}", OI["orange"]),
        ("diff_means", axes[0, 1], "Difference of Means", OI["vermilion"]),
        ("effect_size", axes[1, 0], "Effect Size (Cohen's d)", OI["purple"]),
        ("nu_minus1", axes[1, 1], "Normality (ν − 1)", OI["green"]),
    ]:
        if var in idata.posterior:
            samples = idata.posterior[var].values.ravel()
            plot_posterior_density(samples, var_name=title, ax=ax, color=color)

    fig.suptitle("BEST: Bayesian Group Comparison", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_hierarchical_caterpillar(
    idata,
    group_var: str = "group_effect",
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """
    Caterpillar plot for hierarchical model group effects.

    Shows shrinkage of group-specific estimates toward the grand
    mean (partial pooling). Wider intervals = more shrinkage
    (less data in that group).
    """
    _setup_style()
    import arviz as az

    post = idata.posterior[group_var].values
    flat = post.reshape(-1, post.shape[-1])
    n = flat.shape[1]

    if figsize is None:
        figsize = (5, max(3, n * 0.3))

    fig, ax = plt.subplots(figsize=figsize)

    means = flat.mean(axis=0)
    order = np.argsort(means)

    for j, idx in enumerate(order):
        hdi94 = az.hdi(flat[:, idx], hdi_prob=0.94)
        hdi50 = az.hdi(flat[:, idx], hdi_prob=0.50)
        ax.plot([hdi94[0], hdi94[1]], [j, j], color=OI["blue"],
                linewidth=1, alpha=0.5)
        ax.plot([hdi50[0], hdi50[1]], [j, j], color=OI["blue"],
                linewidth=3, alpha=0.8)
        ax.plot(means[idx], j, "o", color=OI["vermilion"], markersize=4, zorder=5)

    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Group effect")
    ax.set_title("Hierarchical Model — Group Effects (Caterpillar)")
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  §D  ADVANCED
# ═══════════════════════════════════════════════════════════════════════════


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    figsize: Tuple[float, float] = (6, 3),
) -> plt.Figure:
    """
    LOO model comparison plot with ELPD differences and weights.

    Shows ELPD ± SE for each model, with stacking weights annotated.
    Input is the DataFrame from ``az.compare()``.
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=figsize)

    n = len(comparison_df)
    y = range(n)
    elpd = comparison_df["elpd_loo"].values
    se = comparison_df["se"].values

    ax.errorbar(elpd, y, xerr=se, fmt="o", color=OI["blue"],
                capsize=3, markersize=6)

    for i, (name, row) in enumerate(comparison_df.iterrows()):
        weight = row.get("weight", 0)
        ax.text(elpd[i] + se[i] + 0.5, i, f"w={weight:.2f}", fontsize=7, va="center")

    ax.set_yticks(y)
    ax.set_yticklabels(comparison_df.index, fontsize=8)
    ax.set_xlabel("ELPD (LOO)")
    ax.set_title("Model Comparison — LOO-CV")

    fig.tight_layout()
    return fig


def plot_posterior_predictive_check(
    idata,
    y_obs: np.ndarray,
    n_samples: int = 100,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (5, 3.5),
) -> plt.Axes:
    """
    Posterior predictive check — observed vs. replicated data.

    Overlays KDEs of replicated datasets (light) with the observed
    data (dark), revealing systematic mis-specification.
    """
    _setup_style()
    from scipy.stats import gaussian_kde

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    obs_name = list(idata.posterior_predictive.data_vars)[0]
    pp = idata.posterior_predictive[obs_name].values
    pp_flat = pp.reshape(-1, pp.shape[-1])

    x = np.linspace(y_obs.min() - np.ptp(y_obs) * 0.2,
                     y_obs.max() + np.ptp(y_obs) * 0.2, 200)

    idx = np.random.choice(pp_flat.shape[0], min(n_samples, pp_flat.shape[0]),
                           replace=False)
    for i in idx:
        try:
            kde = gaussian_kde(pp_flat[i])
            ax.plot(x, kde(x), color=OI["skyblue"], alpha=0.05, linewidth=0.5)
        except Exception:
            pass

    kde_obs = gaussian_kde(y_obs)
    ax.plot(x, kde_obs(x), color=OI["vermilion"], linewidth=2, label="Observed")

    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title("Posterior Predictive Check")
    ax.legend(fontsize=7)

    return ax


def plot_sensitivity(
    results: Dict[str, pd.DataFrame],
    var_name: str = "beta",
    feature_idx: int = 0,
    figsize: Tuple[float, float] = (5, 3.5),
) -> plt.Figure:
    """
    Prior sensitivity analysis plot.

    Shows how posterior of a key parameter changes across different
    prior specifications. Each entry in ``results`` is a summary
    DataFrame from a different prior configuration.
    """
    _setup_style()
    from scipy.stats import gaussian_kde

    fig, ax = plt.subplots(figsize=figsize)

    colors = list(OI.values())
    for i, (label, idata) in enumerate(results.items()):
        post = idata.posterior[var_name].values
        samples = post.reshape(-1, post.shape[-1])[:, feature_idx]
        kde = gaussian_kde(samples)
        x = np.linspace(samples.min(), samples.max(), 200)
        ax.plot(x, kde(x), label=label, color=colors[i % len(colors)], linewidth=1.5)

    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel(f"{var_name}[{feature_idx}]")
    ax.set_ylabel("Density")
    ax.set_title("Prior Sensitivity Analysis")
    ax.legend(fontsize=7)

    fig.tight_layout()
    return fig


def plot_correlation_matrix_posterior(
    idata,
    var_names: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (6, 5),
) -> plt.Figure:
    """
    Posterior mean correlation matrix heatmap with HDI annotations.

    Shows the posterior mean of the correlation matrix (from
    BayesianCorrelation model with LKJ prior) as a heatmap, with
    cell annotations showing mean ± HDI width.
    """
    _setup_style()
    import seaborn as sns

    corr_post = idata.posterior["corr"].values  # (chains, draws, p, p)
    corr_mean = corr_post.mean(axis=(0, 1))
    corr_std = corr_post.std(axis=(0, 1))

    fig, ax = plt.subplots(figsize=figsize)

    mask = np.triu(np.ones_like(corr_mean, dtype=bool), k=1)
    sns.heatmap(corr_mean, mask=mask, cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, ax=ax, square=True,
                xticklabels=var_names, yticklabels=var_names,
                cbar_kws={"label": "Correlation (posterior mean)"})

    ax.set_title("Posterior Correlation Matrix (LKJ Prior)")
    fig.tight_layout()
    return fig


def plot_brain_posterior_map(
    values: np.ndarray,
    parcellation: str = "schaefer_200",
    hemi: str = "lh",
    cmap: str = "RdBu_r",
    vmax: Optional[float] = None,
    title: str = "Posterior Mean on Cortex",
    figsize: Tuple[float, float] = (6, 4),
) -> plt.Figure:
    """
    Brain surface map of posterior parameter values.

    Projects parcel-level posterior means (e.g., horseshoe β)
    onto the cortical surface using surfplot or nilearn.

    Parameters
    ----------
    values : (R,) array of posterior means per parcel
    parcellation : atlas name for surface mapping
    hemi : hemisphere
    """
    _setup_style()

    try:
        from surfplot import Plot
        from neuromaps.datasets import fetch_fslr

        surfaces = fetch_fslr()
        p = Plot(surfaces["inflated"], views="lateral", size=(600, 400))
        p.add_layer(values, cmap=cmap, color_range=(-vmax, vmax) if vmax else None)
        fig = p.build()
        return fig
    except ImportError:
        # Fallback: bar chart by parcel
        fig, ax = plt.subplots(figsize=figsize)
        colors = [OI["vermilion"] if v > 0 else OI["blue"] for v in values]
        ax.bar(range(len(values)), values, color=colors, alpha=0.7)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_xlabel("Parcel index")
        ax.set_ylabel("Posterior mean")
        ax.set_title(title)
        fig.tight_layout()
        return fig
