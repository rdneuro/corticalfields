#!/usr/bin/env python3
"""
visualize_hks_surface.py — Multi-scale HKS on the Cortical Surface
=====================================================================

This script creates two publication-quality figures:

  1. MULTI-PANEL 3D RENDER — Real HKS data from a SARS-CoV-2 ICU survivor
     (sub-01) rendered on the pial surface at 4 diffusion time scales,
     showing the progression from local curvature to global shape.

  2. CONCEPTUAL DIAGRAM — SVG schematic explaining the heat diffusion
     principle behind HKS: "release heat at a point, measure how much
     stays" across multiple time scales.

Both figures are suitable for journal submission, conference posters,
and the CorticalFields GitHub README.

Usage (Spyder):
  1. Ensure eigenpairs are cached (run the save block from the tutorial)
  2. Run all cells sequentially

Dependencies:
  - corticalfields (spectral descriptors)
  - pyvista (3D surface rendering)
  - matplotlib (colorbars and layout)
  - nibabel (FreeSurfer surface I/O)

Author: Velho Mago (rdneuro)
"""


# %% [0] IMPORTS
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib import cm

# PyVista for 3D surface rendering
import pyvista as pv
# Enable offscreen rendering (no window popup — essential for Spyder)
pv.OFF_SCREEN = True

# CorticalFields spectral module
from corticalfields.spectral import (
    LaplaceBeltrami,
    heat_kernel_signature,
)

try:
    plt.style.use(["science", "no-latex"])
except Exception:
    pass

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
})

warnings.filterwarnings("ignore")
print("Imports OK ✓")


# %% [1] CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# ── Paths ───────────────────────────────────────────────────────────────
CACHE_DIR = Path(
    "/mnt/nvme1n1p1/sars_cov_2_project/analysis/"
    "structure_function_coupling/eigenpair_cache"
)
FS_DIR = Path("/mnt/nvme1n1p1/sars_cov_2_project/structural/fastsurfer")

FIG_DIR = Path(
    "/mnt/nvme1n1p1/sars_cov_2_project/analysis/"
    "structure_function_coupling/figures"
)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Subject and hemisphere ──────────────────────────────────────────────
SUBJECT = "sub-01"
HEMI = "lh"             # left hemisphere (lateral view is more informative)

# ── HKS time scales to visualize ───────────────────────────────────────
# We show 4 scales spanning the local→global progression.
# These correspond approximately to:
#   t_small  → captures sulcal fundi vs gyral crowns (~1mm scale)
#   t_med1   → captures individual gyri (~5-10mm scale)
#   t_med2   → captures lobar boundaries (~20-30mm scale)
#   t_large  → captures hemispheric shape (~50mm+ scale)
# The exact time values are computed from the eigenvalue spectrum.
HKS_PANEL_INDICES = [1, 5, 9, 15]   # indices into the 16-scale HKS
HKS_PANEL_LABELS = [
    "Local curvature\n(sulci vs gyri)",
    "Meso-scale geometry\n(individual gyri)",
    "Regional structure\n(lobar boundaries)",
    "Global shape\n(hemispheric)",
]

# ── Rendering parameters ───────────────────────────────────────────────
COLORMAP = "inferno"         # perceptually uniform, beautiful on cortex
WINDOW_SIZE = (800, 600)     # per-panel render resolution
CAMERA_POSITION = "lateral"  # lateral view of left hemisphere

print("Configuration ✓")
print(f"  Subject: {SUBJECT}, Hemi: {HEMI}")
print(f"  Cache: {CACHE_DIR}")
print(f"  Panels: {HKS_PANEL_INDICES} → {len(HKS_PANEL_INDICES)} scales")


# %% [2] LOAD EIGENPAIRS AND SURFACE
# ═══════════════════════════════════════════════════════════════════════════
# Load the cached eigenpairs (saved from the tutorial's 197-minute run)
# and the pial surface geometry for rendering.

import scipy.sparse as sp

# ── Load eigenpairs ─────────────────────────────────────────────────────
lb_path = CACHE_DIR / f"{SUBJECT}_{HEMI}_lb.npz"
print(f"Loading eigenpairs: {lb_path.name}")
data = np.load(str(lb_path))

lb = LaplaceBeltrami(
    stiffness=sp.csc_matrix((1, 1)),   # placeholder (not needed for HKS)
    mass=sp.csc_matrix((1, 1)),
    eigenvalues=data["eigenvalues"],
    eigenvectors=data["eigenvectors"],
)
print(f"  Eigenvalues: {lb.n_eigenpairs}, "
      f"range [{lb.eigenvalues[1]:.4f}, {lb.eigenvalues[-1]:.1f}]")
print(f"  Eigenvectors: {lb.eigenvectors.shape}")

# ── Load surface geometry ───────────────────────────────────────────────
# Try cached surface first, fall back to reading from FastSurfer
surf_cache = CACHE_DIR / f"{SUBJECT}_{HEMI}_surface.npz"
if surf_cache.exists():
    print(f"Loading surface: {surf_cache.name}")
    sdata = np.load(str(surf_cache))
    vertices = sdata["vertices"]
    faces = sdata["faces"]
else:
    print(f"Loading surface from FastSurfer: {FS_DIR / SUBJECT}")
    import nibabel.freesurfer as fs_io
    surf_path = FS_DIR / SUBJECT / "surf" / f"{HEMI}.pial"
    vertices, faces = fs_io.read_geometry(str(surf_path))

print(f"  Vertices: {vertices.shape[0]}, Faces: {faces.shape[0]}")


# %% [3] COMPUTE HKS AT MULTIPLE TIME SCALES
# ═══════════════════════════════════════════════════════════════════════════
# Compute HKS at 16 time scales (standard CorticalFields configuration).
# Then extract the 4 scales we want to visualize.
#
# The time scales are log-spaced between t_min and t_max, which are
# determined by the eigenvalue spectrum:
#   t_min = 4·ln(10) / λ_max  (smallest diffusion time → local features)
#   t_max = 4·ln(10) / λ_1    (largest diffusion time → global features)

print("\nComputing HKS (16 time scales)...")
hks_all = heat_kernel_signature(lb, n_scales=16)
print(f"  HKS shape: {hks_all.shape}")
print(f"  Value range: [{hks_all.min():.6f}, {hks_all.max():.6f}]")

# Compute the actual time scales for annotation
evals = lb.eigenvalues[1:]
lam_min = max(evals[0], 1e-8)
lam_max = evals[-1]
t_min = 4.0 * np.log(10) / lam_max
t_max = 4.0 * np.log(10) / lam_min
time_scales = np.logspace(np.log10(t_min), np.log10(t_max), 16)

print(f"  Time range: [{t_min:.6f}, {t_max:.2f}] seconds")
print(f"  Selected scales:")
for idx in HKS_PANEL_INDICES:
    print(f"    hks_{idx}: t = {time_scales[idx]:.4f}, "
          f"range = [{hks_all[:, idx].min():.6f}, {hks_all[:, idx].max():.6f}]")


# %% [4] RENDER MULTI-PANEL HKS ON CORTICAL SURFACE
# ═══════════════════════════════════════════════════════════════════════════
# Create a 4-panel figure showing HKS at increasing time scales.
# Each panel renders the pial surface with the HKS values mapped as
# a scalar colormap.
#
# The visual progression should show:
#   Panel 1 (small t): Fine-grained pattern — every sulcus and gyrus
#            visible as distinct colors. High spatial frequency.
#   Panel 2 (medium t): Smoothed pattern — individual gyri merge into
#            regional features. Intermediate frequency.
#   Panel 3 (large-medium t): Broad regions emerge — lobar structure
#            becomes visible. Low frequency.
#   Panel 4 (large t): Nearly uniform — only the hemispheric shape
#            (anterior/posterior, dorsal/ventral) drives variation.

print("\nRendering multi-panel HKS surface...")

# Build PyVista mesh (faces need to be prepended with vertex count)
faces_pv = np.column_stack([
    np.full(faces.shape[0], 3, dtype=np.int64),
    faces,
]).ravel()
mesh = pv.PolyData(vertices, faces_pv)

# Render each panel
panel_images = []

for panel_idx, hks_idx in enumerate(HKS_PANEL_INDICES):
    hks_values = hks_all[:, hks_idx]

    # Add HKS as scalar data on the mesh
    mesh.point_data["HKS"] = hks_values

    # Create plotter
    plotter = pv.Plotter(off_screen=True, window_size=WINDOW_SIZE)
    plotter.set_background("white")

    # Add the surface mesh with HKS colormap
    plotter.add_mesh(
        mesh,
        scalars="HKS",
        cmap=COLORMAP,
        clim=[np.percentile(hks_values, 2), np.percentile(hks_values, 98)],
        show_scalar_bar=False,
        smooth_shading=True,
        ambient=0.2,
        diffuse=0.8,
        specular=0.3,
    )

    # Set camera for lateral view of left hemisphere
    if HEMI == "lh":
        plotter.camera_position = [
            (-300, 0, 0),    # camera position (looking from left)
            (0, 0, 0),       # focal point
            (0, 0, 1),       # view up
        ]
    else:
        plotter.camera_position = [
            (300, 0, 0),
            (0, 0, 0),
            (0, 0, 1),
        ]

    plotter.camera.zoom(1.5)

    # Capture screenshot
    img = plotter.screenshot(transparent_background=False, return_img=True)
    panel_images.append(img)
    plotter.close()

    print(f"  Panel {panel_idx+1}: hks_{hks_idx} (t={time_scales[hks_idx]:.4f}) ✓")

# ── Assemble multi-panel figure with matplotlib ─────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

for ax, img, idx, label in zip(axes, panel_images, HKS_PANEL_INDICES, HKS_PANEL_LABELS):
    ax.imshow(img)
    ax.set_title(f"HKS scale {idx}\n{label}", fontsize=10, fontweight="bold", pad=8)
    ax.text(
        0.5, 0.02, f"t = {time_scales[idx]:.4f}",
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=8, color="gray", style="italic",
    )
    ax.axis("off")

# Add a shared colorbar
sm = cm.ScalarMappable(
    cmap=COLORMAP,
    norm=Normalize(vmin=0, vmax=1),
)
sm.set_array([])
cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label("HKS value (normalized)", fontsize=9)
cbar.set_ticks([0, 0.5, 1])
cbar.set_ticklabels(["Low\n(sulci)", "Mid", "High\n(gyri)"])

# Suptitle
fig.suptitle(
    f"Heat Kernel Signature — Multi-scale Geometry on the Cortical Surface\n"
    f"{SUBJECT}, {HEMI} hemisphere  |  CorticalFields  |  300 LB eigenpairs",
    fontsize=12, fontweight="bold", y=1.02,
)

# Add annotation arrow showing local → global progression
fig.text(0.15, -0.02, "LOCAL", fontsize=10, fontweight="bold", color="#781286",
         ha="center")
fig.text(0.85, -0.02, "GLOBAL", fontsize=10, fontweight="bold", color="#E69422",
         ha="center")
fig.annotate(
    "", xy=(0.82, -0.01), xytext=(0.20, -0.01),
    xycoords="figure fraction",
    arrowprops=dict(arrowstyle="->", color="gray", lw=1.5),
)
fig.text(0.5, -0.04, "← increasing diffusion time t →", ha="center",
         fontsize=9, color="gray", style="italic")

fig.tight_layout(rect=[0, 0, 0.91, 0.95])

# Save
fig.savefig(FIG_DIR / "hks_multipanel_surface.png", bbox_inches="tight", dpi=300)
fig.savefig(FIG_DIR / "hks_multipanel_surface.pdf", bbox_inches="tight")
print(f"\n✓ Multi-panel figure saved to {FIG_DIR}/hks_multipanel_surface.png")
plt.close(fig)


# %% [5] CONCEPTUAL DIAGRAM — Heat diffusion on a surface
# ═══════════════════════════════════════════════════════════════════════════
# This SVG-like figure explains the HKS concept for a general audience.
# It shows a simplified cortical surface with heat diffusion at 3 time
# points, using a clean schematic style suitable for method figures.

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Simulate a 1D "cortical profile" with sulci and gyri
x = np.linspace(0, 4 * np.pi, 500)
# Cortical profile: mix of frequencies (gyri = broad bumps, sulci = sharp dips)
profile = (
    1.0 * np.sin(x) +            # large-scale folding
    0.4 * np.sin(3 * x) +        # medium-scale gyri
    0.15 * np.sin(8 * x + 0.5)   # fine-scale sulcal detail
)

# Heat source point (vertex of interest)
source_idx = 250  # on a gyral crown

# Simulate heat diffusion at different time scales using Gaussian blur
from scipy.ndimage import gaussian_filter1d

time_params = [
    (2.0,  "t₁ (small)", "Local curvature\nHeat stays near the source",
     "#781286", "Only nearby vertices retain heat →\nreveals sulcal/gyral identity"),
    (15.0, "t₂ (medium)", "Regional geometry\nHeat spreads to neighboring gyri",
     "#4682B4", "Intermediate spread →\nreveals which gyrus cluster"),
    (60.0, "t₃ (large)", "Global shape\nHeat covers the entire region",
     "#E69422", "Broad diffusion →\nreveals which lobe/region"),
]

for ax, (sigma, t_label, title, color, explanation) in zip(axes, time_params):
    # Draw the cortical profile
    ax.fill_between(x, profile - 3, profile, color="#E8E8E8", alpha=0.5)
    ax.plot(x, profile, color="#333333", linewidth=1.5, zorder=3)

    # Simulate heat distribution (Gaussian centered at source)
    heat = np.zeros_like(x)
    heat[source_idx] = 1.0
    heat_diffused = gaussian_filter1d(heat, sigma=sigma)
    heat_diffused = heat_diffused / heat_diffused.max()  # normalize

    # Plot heat as a colored overlay
    for j in range(len(x) - 1):
        alpha = heat_diffused[j] * 0.8
        if alpha > 0.01:
            ax.fill_between(
                x[j:j+2], profile[j:j+2] - 3, profile[j:j+2],
                color=color, alpha=alpha, linewidth=0,
            )

    # Mark the source vertex
    ax.plot(x[source_idx], profile[source_idx], "o",
            color="red", markersize=8, zorder=5, markeredgecolor="white",
            markeredgewidth=1.5)

    # Annotations
    ax.set_title(title, fontsize=10, fontweight="bold", color=color)
    ax.text(0.5, -0.15, explanation, transform=ax.transAxes,
            ha="center", va="top", fontsize=8, color="gray", style="italic")

    # Time label
    ax.text(0.02, 0.95, t_label, transform=ax.transAxes,
            ha="left", va="top", fontsize=9, fontweight="bold",
            color=color, bbox=dict(boxstyle="round,pad=0.3",
                                    facecolor="white", edgecolor=color, alpha=0.9))

    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(-4, 2.5)
    ax.axis("off")

# Add source point legend
fig.text(0.08, -0.08, "●", fontsize=14, color="red", ha="center",
         fontfamily="sans-serif")
fig.text(0.12, -0.07, "= heat source (vertex of interest)", fontsize=8,
         color="gray", ha="left")

# Title
fig.suptitle(
    "Heat Kernel Signature: Intuition\n"
    "\"Release heat at a vertex. Measure how much remains at time t.\"\n"
    "HKS(x, t) = Σᵢ exp(−λᵢ · t) · φᵢ(x)²",
    fontsize=11, fontweight="bold", y=1.08,
)

# Progression arrow
fig.annotate(
    "", xy=(0.88, -0.12), xytext=(0.12, -0.12),
    xycoords="figure fraction",
    arrowprops=dict(arrowstyle="->", color="gray", lw=1.5),
)
fig.text(0.5, -0.15, "← increasing diffusion time →",
         ha="center", fontsize=9, color="gray", style="italic")

fig.tight_layout(rect=[0, 0, 1, 0.92])

fig.savefig(FIG_DIR / "hks_conceptual_diagram.png", bbox_inches="tight", dpi=300)
fig.savefig(FIG_DIR / "hks_conceptual_diagram.pdf", bbox_inches="tight")
print(f"✓ Conceptual diagram saved to {FIG_DIR}/hks_conceptual_diagram.png")
plt.close(fig)


# %% [6] BONUS — Patch for tutorial cell [2] to auto-cache eigenpairs
# ═══════════════════════════════════════════════════════════════════════════
# Add this code block at the END of cell [2] in the tutorial script,
# right after the lb_cache loop. This ensures eigenpairs are always
# saved to disk so you never lose another 197-minute computation.

PATCH_CODE = '''
# ── AUTO-CACHE: Save eigenpairs to disk ─────────────────────────────
# This block runs after all eigenpairs are computed and saves them
# as compressed .npz files. Next time, cell [2] can check for cached
# files and skip recomputation entirely.

_cache_dir = OUTPUT_DIR / "eigenpair_cache"
_cache_dir.mkdir(parents=True, exist_ok=True)

for (sub, hemi), lb in lb_cache.items():
    _cache_path = _cache_dir / f"{sub}_{hemi}_lb.npz"
    if not _cache_path.exists():
        np.savez_compressed(
            str(_cache_path),
            eigenvalues=lb.eigenvalues,
            eigenvectors=lb.eigenvectors,
        )
        
# Also save surface geometry (for visualization)
for (sub, hemi), surf in surfaces.items():
    _surf_path = _cache_dir / f"{sub}_{hemi}_surface.npz"
    if not _surf_path.exists():
        np.savez_compressed(
            str(_surf_path),
            vertices=surf.vertices,
            faces=surf.faces,
        )

print(f"  Eigenpair cache: {_cache_dir} ({len(lb_cache)} hemispheres saved)")
'''

print("\n" + "=" * 60)
print("  PATCH FOR TUTORIAL CELL [2]")
print("=" * 60)
print("  Add this block at the END of cell [2] to auto-cache eigenpairs.")
print("  This prevents ever losing another 197-minute computation.")
print(PATCH_CODE)
