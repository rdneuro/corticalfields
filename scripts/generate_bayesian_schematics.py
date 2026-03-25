#!/usr/bin/env python3
"""
generate_bayesian_schematics.py
================================

Generate publication-quality schematic diagrams for every model class
in the CorticalFields bayesian module + the bayes_viz function gallery.

Output: One multi-page PDF with 11 pages + individual PNGs.

Each schematic shows:
  - Graphical model (plate diagram style)
  - When to use / when NOT to use
  - Key hyperparameters
  - Neuroimaging application example

Usage (Spyder): Run all cells sequentially.
Output dir: ./bayesian_schematics/

Author: Velho Mago (rdneuro)
"""

# %% [0] IMPORTS & STYLE
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

try:
    plt.style.use(["science", "no-latex"])
except Exception:
    pass

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

OUT_DIR = Path("bayesian_schematics")
OUT_DIR.mkdir(exist_ok=True)

# ── Color palette (Okabe-Ito) ──────────────────────────────────────────
C = {
    "prior":     "#E69F00",  # orange
    "posterior": "#0072B2",  # blue
    "data":      "#009E73",  # green
    "signal":    "#D55E00",  # vermilion
    "noise":     "#56B4E9",  # sky blue
    "rope":      "#999999",  # gray
    "accent":    "#CC79A7",  # purple
    "bg":        "#F7F7F7",  # light gray
}

pages = []  # collect all figures for multi-page PDF


def _node(ax, x, y, text, color="#0072B2", size=28, shape="circle", alpha=0.15):
    """Draw a graphical model node."""
    if shape == "circle":
        circle = plt.Circle((x, y), 0.12, facecolor=color, alpha=alpha,
                             edgecolor=color, linewidth=1.5, zorder=3)
        ax.add_patch(circle)
    elif shape == "rect":
        rect = FancyBboxPatch((x-0.11, y-0.08), 0.22, 0.16,
                               boxstyle="round,pad=0.02",
                               facecolor=color, alpha=alpha,
                               edgecolor=color, linewidth=1.5, zorder=3)
        ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center", fontsize=size*0.35,
            fontweight="bold", color=color, zorder=4)


def _arrow(ax, x1, y1, x2, y2, color="gray", style="->"):
    """Draw an arrow between nodes."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=1.5),
                zorder=2)


def _plate(ax, x, y, w, h, label="N"):
    """Draw a plate notation rectangle."""
    rect = mpatches.FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.02",
                                    facecolor="none", edgecolor="gray",
                                    linewidth=1, linestyle="--", zorder=1)
    ax.add_patch(rect)
    ax.text(x + w - 0.03, y + 0.03, label, fontsize=7, color="gray",
            ha="right", va="bottom", style="italic")


def _info_box(ax, x, y, title, items, color="#0072B2", width=0.4):
    """Draw an info box with title and bullet items."""
    ax.text(x, y, title, fontsize=8, fontweight="bold", color=color,
            va="top", transform=ax.transAxes)
    for i, item in enumerate(items):
        ax.text(x + 0.01, y - 0.06 - i * 0.045, f"• {item}",
                fontsize=6.5, color="#333333", va="top",
                transform=ax.transAxes, wrap=True)


print("Setup OK ✓")


# %% [1] HORSESHOE REGRESSION
fig, (ax_gm, ax_info) = plt.subplots(1, 2, figsize=(11, 5),
                                       gridspec_kw={"width_ratios": [1.2, 1]})
fig.suptitle("1. HorseshoeRegression — Regularized (Finnish) Horseshoe",
             fontsize=13, fontweight="bold", y=0.98)

# Graphical model
ax = ax_gm
ax.set_xlim(-0.3, 1.5)
ax.set_ylim(-0.3, 1.3)
ax.axis("off")
ax.set_title("Graphical Model", fontsize=10, pad=10)

# Nodes
_node(ax, 0.1, 1.1, "τ", C["prior"])        # global shrinkage
_node(ax, 0.5, 1.1, "c²", C["prior"])       # slab
_node(ax, 0.9, 1.1, "σ", C["noise"])        # noise SD
_node(ax, 0.3, 0.75, "λⱼ", C["prior"])      # local shrinkage
_node(ax, 0.7, 0.75, "α", C["posterior"])    # intercept
_node(ax, 0.5, 0.4, "βⱼ", C["posterior"])   # coefficients
_node(ax, 0.5, 0.05, "yᵢ", C["data"], shape="rect")  # observed

# Arrows
_arrow(ax, 0.1, 0.98, 0.4, 0.52)   # τ → β
_arrow(ax, 0.3, 0.63, 0.45, 0.52)  # λ → β
_arrow(ax, 0.5, 0.98, 0.5, 0.52)   # c² → β
_arrow(ax, 0.5, 0.28, 0.5, 0.17)   # β → y
_arrow(ax, 0.7, 0.63, 0.55, 0.17)  # α → y
_arrow(ax, 0.9, 0.98, 0.6, 0.17)   # σ → y

# Plates
_plate(ax, 0.15, 0.25, 0.7, 0.65, "j = 1…p")
_plate(ax, 0.25, -0.1, 0.5, 0.35, "i = 1…n")

# Key equation
ax.text(0.5, -0.25, r"$\beta_j \sim \mathcal{N}(0, \tau \cdot \tilde{\lambda}_j)$"
        r"  |  $\tau_0 = \frac{p_0}{p - p_0} \cdot \frac{\sigma}{\sqrt{n}}$",
        ha="center", fontsize=9, style="italic", color=C["posterior"])

# Info panel
ax = ax_info
ax.axis("off")

_info_box(ax, 0.02, 0.95, "✅ WHEN TO USE", [
    "High-dimensional: p >> n (e.g., 200 ROIs, n=46)",
    "Most features expected to be irrelevant",
    "Known or estimated sparsity level",
    "Brain-behavior regression (HADS, PSQI)",
], C["data"])

_info_box(ax, 0.02, 0.65, "❌ WHEN NOT TO USE", [
    "Dense effects (all features contribute) → use Ridge",
    "p < 10 (not enough sparsity benefit)",
    "Unknown variance explained → use R2-D2 instead",
], C["signal"])

_info_box(ax, 0.02, 0.42, "⚙️ KEY HYPERPARAMETERS", [
    "n_relevant: expected non-zero coefficients (p₀)",
    "slab_scale: max plausible effect size (default: 2.0)",
    "slab_df: slab heaviness (4 = robust default)",
    "target_accept: ≥ 0.99 (extreme posterior curvature)",
], C["prior"])

_info_box(ax, 0.02, 0.15, "🧠 NEUROIMAGING EXAMPLE", [
    "200 Schaefer ROIs → HADS anxiety (n=46 MTLE-HS)",
    "Expects ~8 relevant ROIs → n_relevant=8",
    "Returns: selected ROIs + posterior β + shrinkage κ",
], C["accent"])

fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT_DIR / "01_horseshoe_regression.png", bbox_inches="tight")
pages.append(fig)
print("  1/11 HorseshoeRegression ✓")


# %% [2] R2-D2 REGRESSION
fig, (ax_gm, ax_info) = plt.subplots(1, 2, figsize=(11, 5),
                                       gridspec_kw={"width_ratios": [1.2, 1]})
fig.suptitle("2. R2D2Regression — R²-D2 Shrinkage Prior",
             fontsize=13, fontweight="bold", y=0.98)

ax = ax_gm
ax.set_xlim(-0.3, 1.5)
ax.set_ylim(-0.3, 1.3)
ax.axis("off")
ax.set_title("Graphical Model", fontsize=10, pad=10)

_node(ax, 0.3, 1.1, "R²", C["prior"])
_node(ax, 0.7, 1.1, "φ", C["prior"])        # Dirichlet allocation
_node(ax, 1.0, 1.1, "σ", C["noise"])
_node(ax, 0.5, 0.6, "βⱼ", C["posterior"])
_node(ax, 0.5, 0.1, "yᵢ", C["data"], shape="rect")

_arrow(ax, 0.3, 0.98, 0.45, 0.72)
_arrow(ax, 0.7, 0.98, 0.55, 0.72)
_arrow(ax, 1.0, 0.98, 0.6, 0.22)
_arrow(ax, 0.5, 0.48, 0.5, 0.22)

_plate(ax, 0.3, 0.45, 0.5, 0.35, "j = 1…p")

ax.text(0.5, -0.2, r"$R^2 \sim \mathrm{Beta}(\mu, \sigma)$"
        r"  |  $\phi \sim \mathrm{Dir}(\mathbf{1}_p)$"
        r"  |  $\beta_j \propto \sqrt{\tau^2 \cdot \phi_j}$",
        ha="center", fontsize=8.5, style="italic", color=C["posterior"])

ax = ax_info
ax.axis("off")
_info_box(ax, 0.02, 0.95, "✅ WHEN TO USE", [
    "Can reason about variance explained (R²) more easily",
    "Moderate-to-high dimensional regression",
    "Brain-behavior with ~30% variance expected",
    "No strong sparsity beliefs (unknown # of non-zeros)",
], C["data"])
_info_box(ax, 0.02, 0.65, "❌ WHEN NOT TO USE", [
    "Strong sparsity known → horseshoe is more targeted",
    "Very low n (<15) — R² posterior becomes diffuse",
], C["signal"])
_info_box(ax, 0.02, 0.45, "⚙️ KEY HYPERPARAMETERS", [
    "r2_mean: prior mean of R² (0.3 typical for brain-behavior)",
    "r2_std: uncertainty (0.15 = moderate)",
    "Uses pymc-extras R2D2M2CP when available",
], C["prior"])
_info_box(ax, 0.02, 0.2, "🧠 NEUROIMAGING EXAMPLE", [
    "Spectral descriptors (42D) → cognitive score",
    "R² prior centered at 0.3 ± 0.15",
], C["accent"])

fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT_DIR / "02_r2d2_regression.png", bbox_inches="tight")
pages.append(fig)
print("  2/11 R2D2Regression ✓")


# %% [3] BAYESIAN RIDGE
fig, (ax_gm, ax_info) = plt.subplots(1, 2, figsize=(11, 5),
                                       gridspec_kw={"width_ratios": [1.2, 1]})
fig.suptitle("3. BayesianRidge — Normal Prior Ridge Regression",
             fontsize=13, fontweight="bold", y=0.98)

ax = ax_gm
ax.set_xlim(-0.3, 1.5)
ax.set_ylim(-0.1, 1.1)
ax.axis("off")
ax.set_title("Graphical Model", fontsize=10, pad=10)

_node(ax, 0.3, 0.9, "σ_β", C["prior"])
_node(ax, 0.8, 0.9, "σ", C["noise"])
_node(ax, 0.5, 0.5, "βⱼ", C["posterior"])
_node(ax, 0.5, 0.1, "yᵢ", C["data"], shape="rect")

_arrow(ax, 0.3, 0.78, 0.45, 0.62)
_arrow(ax, 0.5, 0.38, 0.5, 0.22)
_arrow(ax, 0.8, 0.78, 0.6, 0.22)

_plate(ax, 0.3, 0.35, 0.4, 0.3, "j")

ax.text(0.5, -0.05, r"$\beta_j \sim \mathcal{N}(0, \sigma_\beta)$ — simplest Bayesian regression",
        ha="center", fontsize=9, style="italic", color=C["posterior"])

ax = ax_info
ax.axis("off")
_info_box(ax, 0.02, 0.95, "✅ WHEN TO USE", [
    "Dense effects — all features contribute a little",
    "PCA-reduced features (10-20 components)",
    "Quick baseline before trying horseshoe",
    "Low-dimensional problem (p < 20)",
], C["data"])
_info_box(ax, 0.02, 0.65, "❌ WHEN NOT TO USE", [
    "High-dimensional sparse problems → horseshoe",
    "Need feature selection (ridge keeps all features)",
], C["signal"])
_info_box(ax, 0.02, 0.45, "⚙️ KEY HYPERPARAMETERS", [
    "prior_sigma: scale of Normal prior (1.0 default)",
    "Larger → weaker regularization, smaller → stronger",
], C["prior"])

fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT_DIR / "03_bayesian_ridge.png", bbox_inches="tight")
pages.append(fig)
print("  3/11 BayesianRidge ✓")


# %% [4] BEST — GROUP COMPARISON
fig, (ax_gm, ax_info) = plt.subplots(1, 2, figsize=(11, 5),
                                       gridspec_kw={"width_ratios": [1.2, 1]})
fig.suptitle("4. BayesianGroupComparison — BEST (Kruschke 2013)",
             fontsize=13, fontweight="bold", y=0.98)

ax = ax_gm
ax.set_xlim(-0.3, 1.5)
ax.set_ylim(-0.3, 1.3)
ax.axis("off")
ax.set_title("Graphical Model", fontsize=10, pad=10)

# Two groups share ν
_node(ax, 0.5, 1.15, "ν", C["prior"])
_node(ax, 0.15, 0.75, "μ₁", C["posterior"])
_node(ax, 0.85, 0.75, "μ₂", C["posterior"])
_node(ax, 0.15, 0.4, "σ₁", C["noise"])
_node(ax, 0.85, 0.4, "σ₂", C["noise"])
_node(ax, 0.15, 0.05, "y₁ᵢ", C["data"], shape="rect")
_node(ax, 0.85, 0.05, "y₂ᵢ", C["data"], shape="rect")

_arrow(ax, 0.5, 1.03, 0.2, 0.17)
_arrow(ax, 0.5, 1.03, 0.8, 0.17)
_arrow(ax, 0.15, 0.63, 0.15, 0.17)
_arrow(ax, 0.85, 0.63, 0.85, 0.17)
_arrow(ax, 0.15, 0.28, 0.15, 0.17)
_arrow(ax, 0.85, 0.28, 0.85, 0.17)

# Derived: Cohen's d
ax.text(0.5, 0.45, "Cohen's d =\n(μ₁−μ₂)/σ_pooled",
        ha="center", va="center", fontsize=8, style="italic",
        color=C["accent"],
        bbox=dict(boxstyle="round,pad=0.3", facecolor=C["accent"], alpha=0.1))

# ROPE illustration
ax.text(0.5, -0.2, "ROPE: [-0.1, 0.1]  |  Decision: REJECT / ACCEPT / UNDECIDED",
        ha="center", fontsize=8.5, color=C["rope"])

ax = ax_info
ax.axis("off")
_info_box(ax, 0.02, 0.95, "✅ WHEN TO USE", [
    "Patients vs controls (any metric: CT, FA, volume)",
    "Pre/post treatment comparison",
    "Robust to outliers (Student-t likelihood)",
    "Need effect size + uncertainty (not just p-value)",
], C["data"])
_info_box(ax, 0.02, 0.65, "❌ WHEN NOT TO USE", [
    "More than 2 groups → Bayesian ANOVA (Bambi)",
    "Repeated measures → hierarchical model",
], C["signal"])
_info_box(ax, 0.02, 0.45, "⚙️ ROPE DEFAULTS (neuroimaging)", [
    "Cortical thickness: ±0.1d (standardized)",
    "Brain volume: ±0.1d",
    "FA: ±0.1d",
    "General: ±0.1 (Kruschke default)",
], C["prior"])
_info_box(ax, 0.02, 0.2, "🧠 NEUROIMAGING EXAMPLE", [
    "COVID ICU survivors (n=23) vs healthy controls",
    "Metric: HKS coupling at Schaefer-200 parcels",
    "Returns: P(d>0), P(d in ROPE), full posterior of d",
], C["accent"])

fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT_DIR / "04_group_comparison.png", bbox_inches="tight")
pages.append(fig)
print("  4/11 BayesianGroupComparison ✓")


# %% [5] BAYESIAN CORRELATION
fig, (ax_gm, ax_info) = plt.subplots(1, 2, figsize=(11, 5),
                                       gridspec_kw={"width_ratios": [1.2, 1]})
fig.suptitle("5. BayesianCorrelation — LKJ Prior over Correlation Matrices",
             fontsize=13, fontweight="bold", y=0.98)

ax = ax_gm
ax.set_xlim(-0.3, 1.5)
ax.set_ylim(-0.1, 1.2)
ax.axis("off")
ax.set_title("Graphical Model", fontsize=10, pad=10)

_node(ax, 0.3, 1.0, "η", C["prior"])       # LKJ concentration
_node(ax, 0.7, 1.0, "σₖ", C["noise"])      # per-variable SD
_node(ax, 0.5, 0.6, "C", C["posterior"])    # correlation matrix
_node(ax, 0.5, 0.15, "xᵢ", C["data"], shape="rect")

_arrow(ax, 0.3, 0.88, 0.45, 0.72)
_arrow(ax, 0.7, 0.88, 0.55, 0.72)
_arrow(ax, 0.5, 0.48, 0.5, 0.27)

ax.text(0.5, -0.05, r"$\mathbf{C} \sim \mathrm{LKJ}(\eta)$  |  "
        r"$\eta = 2$ → moderate shrinkage toward $\mathbf{I}$",
        ha="center", fontsize=8.5, style="italic", color=C["posterior"])

ax = ax_info
ax.axis("off")
_info_box(ax, 0.02, 0.95, "✅ WHEN TO USE", [
    "Posterior distribution of pairwise correlations",
    "Multi-ROI analysis (7 Yeo networks, Schaefer parcels)",
    "Need uncertainty on r values, not just point estimates",
    "Small samples where Pearson r is unreliable",
], C["data"])
_info_box(ax, 0.02, 0.65, "❌ WHEN NOT TO USE", [
    "Single correlation → simpler Beta-binomial model",
    "p > 50 variables (LKJ becomes expensive)",
    "Conditional independence → use graphical LASSO",
], C["signal"])
_info_box(ax, 0.02, 0.42, "⚙️ KEY HYPERPARAMETERS", [
    "eta=1.0: uniform over all valid correlation matrices",
    "eta=2.0: moderate shrinkage toward identity (recommended)",
    "eta=5.0: strong shrinkage (very conservative)",
], C["prior"])

fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT_DIR / "05_bayesian_correlation.png", bbox_inches="tight")
pages.append(fig)
print("  5/11 BayesianCorrelation ✓")


# %% [6] BAYESIAN MEDIATION
fig, (ax_gm, ax_info) = plt.subplots(1, 2, figsize=(11, 5),
                                       gridspec_kw={"width_ratios": [1.3, 1]})
fig.suptitle("6. BayesianMediation — Path Analysis: X → M → Y",
             fontsize=13, fontweight="bold", y=0.98)

ax = ax_gm
ax.set_xlim(-0.2, 1.6)
ax.set_ylim(-0.3, 1.0)
ax.axis("off")
ax.set_title("Path Diagram", fontsize=10, pad=10)

# Boxes for X, M, Y
for cx, cy, label in [(0.1, 0.3, "X\n(Structure)"),
                       (0.7, 0.85, "M\n(Function)"),
                       (1.3, 0.3, "Y\n(Behavior)")]:
    rect = FancyBboxPatch((cx-0.15, cy-0.12), 0.3, 0.24,
                           boxstyle="round,pad=0.03",
                           facecolor=C["noise"], alpha=0.2,
                           edgecolor=C["posterior"], linewidth=2)
    ax.add_patch(rect)
    ax.text(cx, cy, label, ha="center", va="center", fontsize=9, fontweight="bold")

# Path arrows
ax.annotate("path a", xy=(0.55, 0.73), xytext=(0.25, 0.42),
            arrowprops=dict(arrowstyle="->", color=C["signal"], lw=2.5),
            fontsize=9, fontweight="bold", color=C["signal"])
ax.annotate("path b", xy=(1.15, 0.42), xytext=(0.85, 0.73),
            arrowprops=dict(arrowstyle="->", color=C["signal"], lw=2.5),
            fontsize=9, fontweight="bold", color=C["signal"])
ax.annotate("path c'", xy=(1.15, 0.3), xytext=(0.25, 0.3),
            arrowprops=dict(arrowstyle="->", color=C["posterior"], lw=1.5,
                            linestyle="dashed"),
            fontsize=9, color=C["posterior"])

# Indirect effect
ax.text(0.7, -0.15, "Indirect effect = a × b\n"
        "Posterior is naturally asymmetric\n(no Sobel test needed)",
        ha="center", fontsize=8, style="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=C["prior"], alpha=0.15))

ax = ax_info
ax.axis("off")
_info_box(ax, 0.02, 0.95, "✅ WHEN TO USE", [
    "Structure → Function → Behavior pathway",
    "CT → HKS coupling → HADS anxiety",
    "Indirect effects with proper uncertainty",
    "Small samples (Bayesian >> bootstrap for n<30)",
], C["data"])
_info_box(ax, 0.02, 0.65, "❌ WHEN NOT TO USE", [
    "Multiple mediators → extend manually",
    "Non-linear relationships → GP mediation",
    "Bidirectional effects → SEM framework",
], C["signal"])
_info_box(ax, 0.02, 0.42, "⚙️ KEY OUTPUTS", [
    "a: X→M path (posterior distribution)",
    "b: M→Y path (posterior distribution)",
    "c': direct effect X→Y",
    "a×b: indirect effect (full posterior)",
    "proportion_mediated: |a×b| / |a×b + c'|",
], C["prior"])

fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT_DIR / "06_bayesian_mediation.png", bbox_inches="tight")
pages.append(fig)
print("  6/11 BayesianMediation ✓")


# %% [7] HIERARCHICAL REGRESSION
fig, (ax_gm, ax_info) = plt.subplots(1, 2, figsize=(11, 5),
                                       gridspec_kw={"width_ratios": [1.2, 1]})
fig.suptitle("7. HierarchicalRegression — Multi-Site Mixed Effects",
             fontsize=13, fontweight="bold", y=0.98)

ax = ax_gm
ax.set_xlim(-0.3, 1.5)
ax.set_ylim(-0.3, 1.4)
ax.axis("off")
ax.set_title("Graphical Model (Non-Centered)", fontsize=10, pad=10)

# Population level
_node(ax, 0.2, 1.2, "α", C["posterior"])
_node(ax, 0.6, 1.2, "β", C["posterior"])
_node(ax, 1.0, 1.2, "σ_site", C["prior"])
_node(ax, 1.0, 0.8, "zₛ", C["noise"])        # raw site offsets
_node(ax, 0.5, 0.5, "μᵢₛ", C["posterior"])
_node(ax, 0.5, 0.1, "yᵢₛ", C["data"], shape="rect")

_arrow(ax, 0.2, 1.08, 0.4, 0.62)
_arrow(ax, 0.6, 1.08, 0.5, 0.62)
_arrow(ax, 1.0, 1.08, 1.0, 0.92)
_arrow(ax, 1.0, 0.68, 0.6, 0.62)
_arrow(ax, 0.5, 0.38, 0.5, 0.22)

_plate(ax, 0.8, 0.65, 0.4, 0.3, "s = 1…S")
_plate(ax, 0.25, -0.05, 0.55, 0.35, "i = 1…nₛ")

ax.text(0.5, -0.25, "site_effect = z_s × σ_site  (non-centered)\n"
        "Partial pooling: small sites shrink toward grand mean",
        ha="center", fontsize=8, style="italic", color=C["posterior"])

ax = ax_info
ax.axis("off")
_info_box(ax, 0.02, 0.95, "✅ WHEN TO USE", [
    "Multi-site data (CEPESC + ds000221 + SARS-CoV-2)",
    "Replace ComBat for harmonization",
    "Unbalanced group sizes across sites",
    "Partial pooling stabilizes small-site estimates",
], C["data"])
_info_box(ax, 0.02, 0.65, "❌ WHEN NOT TO USE", [
    "Single site → standard regression",
    "Need site-specific random slopes → extend model",
    ">30 sites with >100 obs each → try centered param.",
], C["signal"])
_info_box(ax, 0.02, 0.42, "⚙️ KEY DESIGN CHOICES", [
    "Non-centered by default (essential for <30 groups)",
    "Replaces ComBat: preserves biology, proper uncertainty",
    "Bayer et al. 2022: HBR outperformed ComBat on ABIDE",
    "Kia et al. 2022: federated HBR on 33 scanners",
], C["prior"])

fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT_DIR / "07_hierarchical_regression.png", bbox_inches="tight")
pages.append(fig)
print("  7/11 HierarchicalRegression ✓")


# %% [8] BAYESIAN LOGISTIC
fig, (ax_gm, ax_info) = plt.subplots(1, 2, figsize=(11, 5),
                                       gridspec_kw={"width_ratios": [1.2, 1]})
fig.suptitle("8. BayesianLogistic — Classification with Uncertainty",
             fontsize=13, fontweight="bold", y=0.98)

ax = ax_gm
ax.set_xlim(-0.3, 1.5)
ax.set_ylim(-0.2, 1.2)
ax.axis("off")
ax.set_title("Graphical Model", fontsize=10, pad=10)

_node(ax, 0.3, 1.0, "prior", C["prior"])   # ridge or horseshoe
_node(ax, 0.7, 1.0, "α", C["posterior"])
_node(ax, 0.5, 0.55, "βⱼ", C["posterior"])
_node(ax, 0.5, 0.15, "yᵢ∈{0,1}", C["data"], shape="rect")

_arrow(ax, 0.3, 0.88, 0.45, 0.67)
_arrow(ax, 0.5, 0.43, 0.5, 0.27)
_arrow(ax, 0.7, 0.88, 0.55, 0.27)

ax.text(0.5, -0.12, "p(yᵢ=1) = σ(α + Xᵢβ)  |  σ = logistic function\n"
        "predict_proba() → full posterior P(class=1 | X_new)",
        ha="center", fontsize=8.5, style="italic", color=C["posterior"])

ax = ax_info
ax.axis("off")
_info_box(ax, 0.02, 0.95, "✅ WHEN TO USE", [
    "Patient vs control classification from ROI features",
    "Need calibrated probability (not just accuracy)",
    "Clinical decision support with uncertainty",
    "Prior options: 'ridge', 'horseshoe', 'flat'",
], C["data"])
_info_box(ax, 0.02, 0.65, "❌ WHEN NOT TO USE", [
    "Multi-class (>2) → extend to softmax",
    "Non-linear boundaries → Bayesian GP classifier",
    "Very large n (>10K) → variational inference",
], C["signal"])
_info_box(ax, 0.02, 0.42, "🧠 CLINICAL APPLICATION", [
    "COVID ICU survivor (class=1) vs healthy (class=0)",
    "Features: spectral descriptors (42D per ROI)",
    "Output: P(COVID | brain) ± HDI per patient",
    "Abstention: if P ∈ [0.3, 0.7] → refer to specialist",
], C["accent"])

fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT_DIR / "08_bayesian_logistic.png", bbox_inches="tight")
pages.append(fig)
print("  8/11 BayesianLogistic ✓")


# %% [9] BAYESIAN CHANGE-POINT
fig, (ax_gm, ax_info) = plt.subplots(1, 2, figsize=(11, 5),
                                       gridspec_kw={"width_ratios": [1.3, 1]})
fig.suptitle("9. BayesianChangePoint — Regime Detection in Longitudinal Data",
             fontsize=13, fontweight="bold", y=0.98)

ax = ax_gm
ax.set_title("Conceptual Illustration", fontsize=10, pad=10)

np.random.seed(42)
n = 80
cp = 45
y1 = np.random.normal(2.5, 0.3, cp)
y2 = np.random.normal(2.1, 0.4, n - cp)
y_ts = np.concatenate([y1, y2])

ax.plot(range(n), y_ts, "o", color=C["data"], markersize=3, alpha=0.6)
ax.plot(range(cp), y1.mean() * np.ones(cp), "-", color=C["posterior"], lw=2,
        label=f"μ_before = {y1.mean():.2f}")
ax.plot(range(cp, n), y2.mean() * np.ones(n-cp), "-", color=C["signal"], lw=2,
        label=f"μ_after = {y2.mean():.2f}")
ax.axvline(cp, color=C["accent"], lw=2, linestyle="--", label=f"Change-point = {cp}")
ax.fill_betweenx([1.5, 3.2], cp-3, cp+3, alpha=0.1, color=C["accent"],
                  label="Posterior uncertainty")
ax.set_xlabel("Time (sessions)")
ax.set_ylabel("Cortical thickness (mm)")
ax.legend(fontsize=7, loc="lower left")

ax = ax_info
ax.axis("off")
_info_box(ax, 0.02, 0.95, "✅ WHEN TO USE", [
    "Longitudinal cortical thickness trajectories",
    "Detect pre/post COVID onset effects",
    "Treatment onset detection in clinical trials",
    "Seizure frequency regime changes in epilepsy",
], C["data"])
_info_box(ax, 0.02, 0.65, "❌ WHEN NOT TO USE", [
    "Multiple change-points → extend model",
    "Gradual trends → GP regression instead",
    "Need Metropolis sampler (discrete τ; slower)",
], C["signal"])
_info_box(ax, 0.02, 0.42, "⚙️ MODEL STRUCTURE", [
    "y_before ~ Normal(μ₁, σ₁) for t < τ",
    "y_after ~ Normal(μ₂, σ₂) for t ≥ τ",
    "τ ~ DiscreteUniform(2, n-2)",
    "Returns: posterior P(τ = t) for each timepoint",
], C["prior"])

fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT_DIR / "09_bayesian_changepoint.png", bbox_inches="tight")
pages.append(fig)
print("  9/11 BayesianChangePoint ✓")


# %% [10] BAYESIAN DAG
fig, (ax_gm, ax_info) = plt.subplots(1, 2, figsize=(11, 5),
                                       gridspec_kw={"width_ratios": [1.3, 1]})
fig.suptitle("10. BayesianDAG — Causal Structure Learning (pgmpy)",
             fontsize=13, fontweight="bold", y=0.98)

ax = ax_gm
ax.set_xlim(-0.2, 1.6)
ax.set_ylim(-0.1, 1.1)
ax.axis("off")
ax.set_title("Example Learned DAG", fontsize=10, pad=10)

# Nodes
nodes = {
    "Age":    (0.1, 0.9),
    "Sex":    (0.1, 0.5),
    "TIV":    (0.5, 0.9),
    "CT":     (0.9, 0.7),
    "HKS":    (1.3, 0.9),
    "HADS":   (1.3, 0.5),
    "FA":     (0.9, 0.3),
}

for name, (x, y) in nodes.items():
    color = C["data"] if name in ("Age", "Sex") else (
        C["posterior"] if name in ("CT", "FA", "HKS") else C["signal"])
    _node(ax, x, y, name, color, size=24)

# Learned edges (example)
edges = [("Age", "CT"), ("Age", "TIV"), ("Sex", "TIV"), ("TIV", "CT"),
         ("CT", "HKS"), ("CT", "FA"), ("HKS", "HADS"), ("FA", "HADS")]
for parent, child in edges:
    px, py = nodes[parent]
    cx, cy = nodes[child]
    _arrow(ax, px, py, cx, cy, color="#444444")

ax.text(0.7, -0.05, "Edges learned via HillClimb (BDeu) or PC algorithm\n"
        "Confounders (Age, Sex) → mediators (CT, FA) → outcome (HADS)",
        ha="center", fontsize=8, style="italic", color="#555555")

ax = ax_info
ax.axis("off")
_info_box(ax, 0.02, 0.95, "✅ WHEN TO USE", [
    "Discover causal relationships among brain measures",
    "Identify confounders vs mediators vs colliders",
    "Guide covariate selection for regression models",
    "Exploratory causal analysis before confirmatory tests",
], C["data"])
_info_box(ax, 0.02, 0.65, "❌ WHEN NOT TO USE", [
    "Time series data → use temporal DAG methods",
    "Known causal structure → just fit the model",
    "Very small n (<30) → structure unreliable",
], C["signal"])
_info_box(ax, 0.02, 0.42, "⚙️ METHODS AVAILABLE", [
    "'hillclimb': score-based (BDeu scoring)",
    "'pc': constraint-based (conditional independence)",
    "get_parents() / get_children(): query structure",
    "adjacency_matrix(): export for visualization",
], C["prior"])

fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT_DIR / "10_bayesian_dag.png", bbox_inches="tight")
pages.append(fig)
print("  10/11 BayesianDAG ✓")


# %% [11] BAYES_VIZ OVERVIEW — 20 functions
fig = plt.figure(figsize=(14, 10))
fig.suptitle("bayes_viz.py — 20 Publication-Quality Bayesian Visualization Functions",
             fontsize=14, fontweight="bold", y=0.98)

sections = [
    ("§A POSTERIOR INSPECTION", C["posterior"], [
        ("plot_posterior_density", "KDE + HDI + ROPE shading + pd annotation"),
        ("plot_forest", "Forest plot with 50%/94% HDI, Yeo-7 colors"),
        ("plot_ridgeline", "Stacked KDE (joy plot) of all posteriors"),
        ("plot_posterior_hdi_rope", "Decision plot: REJECT / ACCEPT / UNDECIDED"),
        ("plot_prior_posterior_update", "Prior→Posterior overlay + shrinkage %"),
    ]),
    ("§B DIAGNOSTICS", C["signal"], [
        ("plot_trace_rank", "Rank-normalized trace (Vehtari et al. 2021)"),
        ("plot_pair_divergences", "Pair plot with divergent transitions (red)"),
        ("plot_energy", "Energy distribution + BFMI diagnostic"),
        ("plot_loo_pit", "LOO-PIT calibration (uniform = good)"),
        ("plot_rhat_ess_panel", "R̂ + ESS bulk/tail dual panel"),
    ]),
    ("§C MODEL-SPECIFIC", C["prior"], [
        ("plot_shrinkage", "Horseshoe κ bar + m_eff (signal vs noise)"),
        ("plot_coefficient_path", "β across models (regularization path)"),
        ("plot_mediation_diagram", "Path arrows with posterior a, b, c', a×b"),
        ("plot_group_comparison", "4-panel BEST (means, diff, d, ν)"),
        ("plot_hierarchical_caterpillar", "Group effects with partial pooling"),
    ]),
    ("§D ADVANCED", C["accent"], [
        ("plot_model_comparison", "LOO ELPD ± SE with stacking weights"),
        ("plot_posterior_predictive_check", "y_rep KDEs vs observed (PPC)"),
        ("plot_sensitivity", "Prior sensitivity: same β under different priors"),
        ("plot_correlation_matrix_posterior", "LKJ posterior heatmap"),
        ("plot_brain_posterior_map", "surfplot/nilearn projection of β on cortex"),
    ]),
]

y_start = 0.88
for sec_name, sec_color, funcs in sections:
    fig.text(0.05, y_start, sec_name, fontsize=11, fontweight="bold",
             color=sec_color)
    for j, (fname, desc) in enumerate(funcs):
        y = y_start - 0.035 * (j + 1)
        fig.text(0.08, y, f"  {fname}()", fontsize=8, fontweight="bold",
                 fontfamily="monospace", color="#333333")
        fig.text(0.38, y, f"— {desc}", fontsize=7.5, color="#666666")
    y_start -= 0.035 * (len(funcs) + 1) + 0.02

# Style info at bottom
fig.text(0.05, 0.04, "Style: Okabe-Ito colorblind-safe palette  |  "
         "Yeo-7 network colors  |  600 DPI PDF  |  "
         "Type 42 fonts  |  DejaVu Sans",
         fontsize=8, color="#888888", style="italic")

fig.savefig(OUT_DIR / "11_bayes_viz_overview.png", bbox_inches="tight")
pages.append(fig)
print("  11/11 bayes_viz overview ✓")


# %% [12] COMPILE MULTI-PAGE PDF
pdf_path = OUT_DIR / "bayesian_module_schematics.pdf"
with PdfPages(str(pdf_path)) as pdf:
    for fig in pages:
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

print(f"\n{'='*60}")
print(f"  All schematics generated!")
print(f"  PNG files: {OUT_DIR}/01_..._11_*.png (11 files)")
print(f"  Multi-page PDF: {pdf_path}")
print(f"{'='*60}")
