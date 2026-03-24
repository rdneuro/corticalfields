#!/usr/bin/env python3
"""
================================================================================
CorticalFields Tutorial: Structure-Function Coupling in Post-COVID Brain
================================================================================

This tutorial demonstrates the full CorticalFields pipeline integrated with
BrainSpace functional gradients to investigate how cortical GEOMETRY relates
to functional ORGANIZATION in SARS-CoV-2 ICU survivors (n=23).

The central question
--------------------
Does the intrinsic geometry of the cortex — captured by spectral shape
descriptors (HKS, WKS, GPS) from the Laplace-Beltrami decomposition —
predict the organization of resting-state functional connectivity, as
captured by diffusion-map embedding gradients?

And critically: in regions where COVID-19 has damaged the cortical
geometry (high normative surprise), is functional organization also
disrupted (gradient displacement)?

Why this matters
----------------
Pang et al. (Nature 2023) showed that geometric eigenmodes of the cortex
explain more variance in brain activity than structural connectivity modes.
This tutorial extends that insight to a clinical population, asking whether
pathological disruption of cortical geometry (post-COVID atrophy and folding
changes) predicts reorganization of functional gradients — a question that
has never been addressed in the literature.

Prerequisites
-------------
  - CorticalFields (pip install corticalfields)
  - BrainSpace (pip install brainspace)
  - FreeSurfer or FastSurfer outputs for the SARS-CoV-2 cohort
  - rs-fMRI connectivity matrices (Schaefer-200 parcellation)
  - Schaefer-200 annotation files projected to individual subjects

Pipeline overview
-----------------
  Cell  0: Imports and environment setup
  Cell  1: Configuration — paths, parameters, cohort definition
  Cell  2: CorticalFields — Load surfaces and compute LB eigenpairs
  Cell  3: CorticalFields — Extract spectral shape descriptors
  Cell  4: CorticalFields — Aggregate spectral features by Schaefer-200
  Cell  5: BrainSpace — Load functional connectivity matrices
  Cell  6: BrainSpace — Compute functional connectivity gradients
  Cell  7: Structure-function coupling — Parcel-wise analysis
  Cell  8: Network-level analysis — Yeo-7 aggregation
  Cell  9: Visualization — Publication-quality figures
  Cell 10: Export and summary

Each cell is self-contained and can be run independently in Spyder.
Variables are designed to be inspectable in the Variable Explorer
after each cell completes.

Author: Velho Mago (rdneuro) — INNT/UFRJ
License: MIT
Repository: https://github.com/rdneuro/corticalfields
"""


# %% [0] IMPORTS AND ENVIRONMENT SETUP
# ═══════════════════════════════════════════════════════════════════════════
# We import CorticalFields modules explicitly to showcase the API.
# BrainSpace is imported for functional gradient computation.
# Standard scientific Python completes the stack.

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist

# ── CorticalFields: the structural geometry engine ──────────────────────
# Each import maps to a specific module in the library:
#   surface  — I/O for FreeSurfer/FastSurfer/GIfTI surfaces
#   spectral — Laplace-Beltrami operator, HKS, WKS, GPS
#   graphs   — MSN and SSN construction, graph metrics
#   features — Cohort-level morphometric profile extraction
from corticalfields.surface import (
    CorticalSurface,
    load_freesurfer_surface,
    load_annot,
)
from corticalfields.spectral import (
    compute_eigenpairs,
    LaplaceBeltrami,
    heat_kernel_signature,
    wave_kernel_signature,
    global_point_signature,
    spectral_feature_matrix,
)
from corticalfields.graphs import (
    morphometric_similarity_network,
    spectral_similarity_network,
    graph_metrics,
)

# ── BrainSpace: the functional gradient engine ──────────────────────────
# GradientMaps computes diffusion-map embedding of FC matrices.
# This is the core algorithm from Margulies et al. (PNAS 2016) that
# decomposes the functional connectome into continuous axes (gradients)
# capturing the principal dimensions of functional organization.
from brainspace.gradient import GradientMaps
from brainspace.gradient.alignment import ProcrustesAlignment

# ── Standard scientific stack ───────────────────────────────────────────
import nibabel.freesurfer as fs_io

# Plotting (loaded later in visualization cell to avoid import overhead)
# import matplotlib.pyplot as plt
# import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

print("=" * 70)
print("  CorticalFields Tutorial: Structure-Function Coupling")
print("  Post-COVID ICU Survivors (SARS-CoV-2 Cohort)")
print("=" * 70)
print("\nAll imports successful ✓")
print("  CorticalFields modules: surface, spectral, graphs, features")
print("  BrainSpace modules: GradientMaps, ProcrustesAlignment")


# %% [1] CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
# All paths and parameters are defined here. Modify this cell to adapt
# the tutorial to your own dataset.
#
# The SARS-CoV-2 cohort consists of 23 ICU survivors scanned ~6 months
# after discharge, with both structural MRI (T1w) and resting-state fMRI.

# ── Choose which structural pipeline to use ─────────────────────────────
# IMPORTANT: Use ONE pipeline consistently (see concordance report).
# Options: "fastsurfer" or "freesurfer"
PIPELINE = "freesurfer"

# ── Project paths ───────────────────────────────────────────────────────
PROJECT_ROOT = Path("/mnt/nvme1n1p1/sars_cov_2_project")

# Structural data (FreeSurfer/FastSurfer recon-all outputs)
# Each subject has: surf/{lh,rh}.{white,pial,thickness,curv,sulc,sphere.reg}
FS_DIR = {
    "fastsurfer": PROJECT_ROOT / "data" / "output" / "structural" / "fastsurfer",
    "freesurfer": Path("/mnt/nvme1n1p1/sars_cov_2_project/data/output/structural/fs_subjects"),
}[PIPELINE]

# Functional connectivity matrices (Schaefer-200, from rs-fMRI pipeline)
# Expected format: sub-XXX_schaefer200_fc.npy or .csv — 200×200 matrix
FC_DIR = Path("/media/rdx/disk4/analysis/covid/functional/fmri/v4/connectivity/schaefer_200/acompcor")

# Schaefer-200 annotation files (fsaverage space, 163842 vertices)
# These are the same .annot files used in the MTLE-HS analysis.
ANNOT_DIR = Path("/media/rdx/disk4/analysis/cepesc/structural/freesurfer/v741/corticalfields_hads/info/annot/fsaverage")

# Projected Schaefer annotations per subject (individual space)
SCH_DIR = Path("/mnt/nvme1n1p1/sars_cov_2_project/data/proc/schaefer_annot/sarscov2")

# Output directory for this tutorial
OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "cf" / "analysis" / "coupling_covid"

# ── Spectral analysis parameters ────────────────────────────────────────
# Number of Laplace-Beltrami eigenpairs to compute.
# 300 captures geometry from global shape (low modes) down to individual
# sulci (high modes). More eigenpairs = finer geometric resolution but
# longer computation time (~2-3 min per hemisphere at 300).
N_EIGENPAIRS = 300

# Spectral descriptor dimensions:
# HKS: 16 time scales — from local curvature (small t) to global shape (large t)
# WKS: 16 energy levels — frequency decomposition of cortical folding
# GPS: 10 components — intrinsic manifold coordinates
# Total: 42 features per vertex
HKS_SCALES = 16
WKS_ENERGIES = 16
GPS_COMPONENTS = 10

# ── Functional gradient parameters ──────────────────────────────────────
# Number of gradients to compute. The first 3 capture:
#   G1: sensorimotor ↔ transmodal (association) axis
#   G2: visual ↔ somatomotor axis
#   G3: default-mode ↔ task-positive axis
N_GRADIENTS = 10       # compute 10, analyze top 3
GRADIENT_APPROACH = "dm"        # diffusion map embedding (Coifman & Lafon, 2006)
GRADIENT_KERNEL = "cosine"      # affinity kernel for the FC matrix
GRADIENT_SPARSITY = 0.9         # keep top 10% connections (sparsify FC)

# ── Surface and parcellation settings ───────────────────────────────────
SURFACE = "pial"                # which surface for LB decomposition
ANNOT_NAME = "Schaefer2018_200Parcels_7Networks_order"

# ── Yeo-7 network labels and canonical colors ──────────────────────────
YEO7_NETWORKS = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
YEO7_COLORS = {
    "Vis":        (120/255, 18/255, 134/255),    # purple
    "SomMot":     (70/255, 130/255, 180/255),    # steel blue
    "DorsAttn":   (0/255, 118/255, 14/255),      # green
    "SalVentAttn":(196/255, 58/255, 250/255),    # violet
    "Limbic":     (196/255, 223/255, 125/255),   # yellow-green
    "Cont":       (230/255, 148/255, 34/255),    # orange
    "Default":    (205/255, 62/255, 78/255),     # crimson
}

print("\nConfiguration ✓")
print(f"  Pipeline     : {PIPELINE}")
print(f"  FS_DIR       : {FS_DIR}")
print(f"  FC_DIR       : {FC_DIR}")
print(f"  OUTPUT_DIR   : {OUTPUT_DIR}")
print(f"  Eigenpairs   : {N_EIGENPAIRS}")
print(f"  Spectral dim : {HKS_SCALES}+{WKS_ENERGIES}+{GPS_COMPONENTS} = "
      f"{HKS_SCALES+WKS_ENERGIES+GPS_COMPONENTS}D/vertex")
print(f"  Gradients    : {N_GRADIENTS} ({GRADIENT_APPROACH}, {GRADIENT_KERNEL})")


# %% [2] CORTICALFIELDS — Load surfaces and compute LB eigenpairs
# ═══════════════════════════════════════════════════════════════════════════
# This cell performs the computationally expensive step: for each subject
# and hemisphere, we load the cortical surface mesh and compute the leading
# eigenpairs of the Laplace-Beltrami operator.
#
# The LB eigenpairs are the mathematical foundation of CorticalFields.
# They encode the intrinsic geometry of the cortical manifold in a way
# that is:
#   - Independent of the embedding in 3D space (intrinsic, not extrinsic)
#   - Multi-scale (low eigenvalues = global shape, high = local detail)
#   - Geodesic-aware (respects distances along the cortical surface)
#
# Think of it as computing the "spectral DNA" of each hemisphere.

def discover_subjects(fs_dir: Path) -> List[str]:
    """
    Find all subjects with completed surface reconstruction.

    A subject is considered complete if both lh.pial and rh.pial exist
    in the surf/ subdirectory. For FastSurfer, this also implies that
    the full recon-surf pipeline has finished (sphere.reg is generated).
    """
    subjects = []
    for d in sorted(fs_dir.iterdir()):
        if d.is_dir() and not d.name.startswith(".") and d.name != "fsaverage":
            pial_lh = d / "surf" / f"lh.{SURFACE}"
            pial_rh = d / "surf" / f"rh.{SURFACE}"
            if pial_lh.exists() and pial_rh.exists():
                subjects.append(d.name)
    return subjects


# Discover subjects
subjects = discover_subjects(FS_DIR)
print(f"Found {len(subjects)} subjects with completed {PIPELINE} outputs")
for s in subjects:
    print(f"  {s}")

# Compute LB eigenpairs for all subjects
# We store the results in dictionaries keyed by (subject, hemi).
# This allows flexible inspection in the Spyder Variable Explorer.

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

lb_cache: Dict[Tuple[str, str], LaplaceBeltrami] = {}
surfaces: Dict[Tuple[str, str], CorticalSurface] = {}

t0 = time.time()

for i, sub in enumerate(subjects):
    print(f"\n[{i+1}/{len(subjects)}] {sub}")

    for hemi in ["lh", "rh"]:
        t1 = time.time()

        # ── Load surface via CorticalFields API ────────────────────
        # load_freesurfer_surface() reads the binary FreeSurfer surface
        # file and any requested per-vertex overlays (thickness, curvature,
        # sulcal depth). It returns a CorticalSurface dataclass with:
        #   .vertices  — (N, 3) coordinates in RAS space
        #   .faces     — (F, 3) triangle connectivity
        #   .overlays  — dict of per-vertex scalar maps
        #   .n_vertices, .n_faces, .total_area — convenience properties
        print(f"  {hemi}: load...", end=" ", flush=True)
        surf = load_freesurfer_surface(
            subjects_dir=str(FS_DIR),
            subject_id=sub,
            hemi=hemi,
            surface=SURFACE,
            overlays=["thickness", "curv", "sulc", "area"],
        )
        surfaces[(sub, hemi)] = surf

        # ── Compute LB eigenpairs ──────────────────────────────────
        # compute_eigenpairs() builds the discrete LB operator via the
        # cotangent-weight scheme (or robust-laplacian if installed),
        # then solves the generalized eigenvalue problem L φ = λ M φ
        # using ARPACK's shift-invert mode.
        #
        # Returns a LaplaceBeltrami dataclass with:
        #   .eigenvalues  — (K,) array, λ_0 ≤ λ_1 ≤ ... ≤ λ_{K-1}
        #   .eigenvectors — (N, K) array, M-orthonormal eigenfunctions
        #   .stiffness    — sparse cotangent matrix L
        #   .mass         — sparse lumped area matrix M
        print("LB eigenpairs...", end=" ", flush=True)
        lb = compute_eigenpairs(
            surf.vertices,
            surf.faces,
            n_eigenpairs=N_EIGENPAIRS,
        )
        lb_cache[(sub, hemi)] = lb

        print(f"✓ ({surf.n_vertices}v, "
              f"λ₁={lb.eigenvalues[1]:.4f}, "
              f"λ₃₀₀={lb.eigenvalues[-1]:.1f}, "
              f"{time.time()-t1:.0f}s)")

print(f"\nLB computation complete: {len(lb_cache)} hemispheres in "
      f"{(time.time()-t0)/60:.1f} min")


# %% [3] CORTICALFIELDS — Extract spectral shape descriptors
# ═══════════════════════════════════════════════════════════════════════════
# With the LB eigenpairs in hand, we now compute three families of spectral
# shape descriptors that capture different aspects of cortical geometry:
#
# 1. HEAT KERNEL SIGNATURE (HKS)
#    HKS(x, t) = Σ_i exp(-λ_i · t) · φ_i(x)²
#
#    Interpretation: Imagine releasing a unit of heat at vertex x and
#    measuring how much remains at time t. In highly curved regions
#    (sulcal fundi), heat diffuses quickly → low HKS. On flat plateaus
#    (gyral crowns), heat lingers → high HKS. The key insight is that
#    different time scales t capture different spatial scales: small t
#    reveals local curvature details, large t captures global shape
#    (which lobe a vertex belongs to).
#
#    In post-COVID brain: atrophy changes the curvature landscape,
#    altering the heat diffusion profile at multiple scales.
#
# 2. WAVE KERNEL SIGNATURE (WKS)
#    WKS(x, e) = Σ_i exp(-(e - log λ_i)² / 2σ²) · φ_i(x)²
#
#    Interpretation: Instead of heat (exponential decay), WKS uses a
#    quantum-mechanical wave equation with log-normal energy filtering.
#    This provides a frequency decomposition of the surface geometry:
#    low energies = broad folding patterns, high energies = fine sulcal
#    detail. WKS is scale-invariant (unlike HKS), making it robust to
#    global brain size differences.
#
#    In post-COVID brain: changes in gyrification patterns appear as
#    altered WKS energy profiles.
#
# 3. GLOBAL POINT SIGNATURE (GPS)
#    GPS(x) = (φ₁(x)/√λ₁, φ₂(x)/√λ₂, ..., φ_K(x)/√λ_K)
#
#    Interpretation: GPS maps each vertex to a point in K-dimensional
#    space such that the Euclidean distance in this space approximates
#    the Green's function distance on the manifold. This creates an
#    intrinsic coordinate system on the cortex that respects geodesic
#    distances — two vertices that are far apart along the cortical
#    surface will be far apart in GPS space, even if they're close
#    in Euclidean 3D space.
#
#    In post-COVID brain: distortion of the GPS embedding indicates
#    disruption of the cortical manifold's global geometry.

spectral_features: Dict[Tuple[str, str], np.ndarray] = {}

t0 = time.time()

for i, sub in enumerate(subjects):
    print(f"[{i+1}/{len(subjects)}] {sub}", end=" ", flush=True)

    for hemi in ["lh", "rh"]:
        lb = lb_cache[(sub, hemi)]

        # spectral_feature_matrix() is a convenience function that
        # concatenates HKS, WKS, and GPS into a single feature matrix.
        # The result has shape (N_vertices, D) where D = 16 + 16 + 10 = 42.
        #
        # Under the hood, it calls:
        #   heat_kernel_signature(lb, n_scales=16)    → (N, 16)
        #   wave_kernel_signature(lb, n_energies=16)  → (N, 16)
        #   global_point_signature(lb, n_components=10) → (N, 10)
        # and horizontally stacks them.
        spec = spectral_feature_matrix(
            lb,
            hks_scales=HKS_SCALES,
            wks_energies=WKS_ENERGIES,
            gps_components=GPS_COMPONENTS,
        )
        spectral_features[(sub, hemi)] = spec

    print(f"  ✓ ({spec.shape[1]}D)")

print(f"\nSpectral features extracted: {len(spectral_features)} hemispheres "
      f"in {(time.time()-t0)/60:.1f} min")
print(f"Feature matrix shape per hemisphere: {spec.shape}")
print(f"  Columns  0-15 : HKS (16 time scales)")
print(f"  Columns 16-31 : WKS (16 energy levels)")
print(f"  Columns 32-41 : GPS (10 manifold coordinates)")


# %% [4] CORTICALFIELDS — Aggregate spectral features by Schaefer-200
# ═══════════════════════════════════════════════════════════════════════════
# The vertex-level spectral features (~150K × 42) are too high-dimensional
# for direct comparison with functional gradients (which are at the parcel
# level, 200 × N_gradients). We need to aggregate.
#
# For each Schaefer-200 parcel, we compute the mean spectral feature vector
# across all vertices in that parcel. This gives us a 42-dimensional
# "geometric signature" for each brain region.
#
# Additionally, we compute per-parcel morphometric summaries (mean thickness,
# curvature, sulcal depth) for classical structure-function comparison.

def parse_network(parcel_name: str) -> str:
    """Extract Yeo-7 network from Schaefer label name.

    Example: '7Networks_LH_DorsAttn_Post_1' → 'DorsAttn'
    """
    parts = parcel_name.split("_")
    return parts[2] if len(parts) >= 3 else "Unknown"


def aggregate_features_by_parcels(
    vertex_features: np.ndarray,
    labels: np.ndarray,
    names: List[str],
    vertex_areas: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Aggregate vertex-level features to Schaefer-200 parcels.

    For each parcel, computes the area-weighted mean of the feature vector
    across all vertices assigned to that parcel. If vertex_areas is None,
    uses uniform (unweighted) averaging.

    Parameters
    ----------
    vertex_features : (N_vertices, D) array
        Per-vertex feature matrix (e.g., spectral descriptors or overlays).
    labels : (N_vertices,) integer array
        Parcellation labels from the projected Schaefer annotation.
    names : list of str
        Label names corresponding to each integer label.
    vertex_areas : (N_vertices,) array, optional
        Per-vertex surface area for area-weighted averaging.

    Returns
    -------
    parcel_features : (R, D) array
        Mean feature vector per parcel (R = number of valid parcels).
    parcel_names : list of str
        Parcel names in order.
    parcel_networks : list of str
        Yeo-7 network assignment for each parcel.
    """
    unique_labels = np.sort(np.unique(labels[labels > 0]))
    parcel_features = []
    parcel_names = []
    parcel_networks = []

    for lab in unique_labels:
        if lab >= len(names):
            continue
        name = names[lab]
        if name in ("Unknown", "???", "MedialWall", "unknown"):
            continue

        mask = labels == lab
        if mask.sum() == 0:
            continue

        if vertex_areas is not None:
            w = vertex_areas[mask]
            w = w / w.sum()
            mean_feat = np.average(vertex_features[mask], axis=0, weights=w)
        else:
            mean_feat = np.mean(vertex_features[mask], axis=0)

        parcel_features.append(mean_feat)
        parcel_names.append(name)
        parcel_networks.append(parse_network(name))

    return np.array(parcel_features), parcel_names, parcel_networks


# Aggregate for all subjects
parcel_spectral: Dict[Tuple[str, str], np.ndarray] = {}
parcel_morph: Dict[Tuple[str, str], np.ndarray] = {}
parcel_info: Dict[str, Tuple[List[str], List[str]]] = {}  # shared across subjects

for i, sub in enumerate(subjects):
    print(f"[{i+1}/{len(subjects)}] {sub}", end=" ", flush=True)

    for hemi in ["lh", "rh"]:
        # Load projected Schaefer-200 annotation
        annot_path = SCH_DIR / sub / f"{hemi}.{ANNOT_NAME}.annot"
        if not annot_path.exists():
            print(f" ⚠ no annot for {hemi}", end="")
            continue

        labels, ctab, names = fs_io.read_annot(str(annot_path))
        names = [n.decode() if isinstance(n, bytes) else n for n in names]

        # Get vertex areas from the CorticalSurface for area-weighted averaging
        surf = surfaces[(sub, hemi)]
        vertex_areas = np.zeros(surf.n_vertices)
        for face in surf.faces:
            v0, v1, v2 = surf.vertices[face]
            tri_area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
            vertex_areas[face] += tri_area / 3.0

        # Aggregate spectral features (42D per parcel)
        spec_feat = spectral_features[(sub, hemi)]
        p_spec, p_names, p_nets = aggregate_features_by_parcels(
            spec_feat, labels, names, vertex_areas,
        )
        parcel_spectral[(sub, hemi)] = p_spec

        # Aggregate morphometric features (thickness, curvature, sulcal depth)
        morph_list = []
        for overlay_name in ["thickness", "curv", "sulc"]:
            if overlay_name in surf.overlays:
                morph_list.append(surf.get_overlay(overlay_name))
        if morph_list:
            morph_matrix = np.column_stack(morph_list)
            p_morph, _, _ = aggregate_features_by_parcels(
                morph_matrix, labels, names, vertex_areas,
            )
            parcel_morph[(sub, hemi)] = p_morph

        # Store parcel info (same for all subjects if same parcellation)
        if hemi not in parcel_info:
            parcel_info[hemi] = (p_names, p_nets)

    print("  ✓")

# Quick sanity check
if parcel_info:
    hemi_check = list(parcel_info.keys())[0]
    n_parcels = len(parcel_info[hemi_check][0])
    print(f"\nSchaefer-200 parcels per hemisphere: {n_parcels}")
    print(f"  Spectral feature dim per parcel: {p_spec.shape[1]}")
    if parcel_morph:
        print(f"  Morphometric features per parcel: {p_morph.shape[1]}")


# %% [5] BRAINSPACE — Load functional connectivity matrices
# ═══════════════════════════════════════════════════════════════════════════
# FC matrices from the rs-fMRI pipeline are stored per-subject as:
#   {FC_DIR}/{sub}/connectivity_correlation.npy
#
# We use the Pearson correlation matrix (not Fisher-z, not partial
# correlation) because BrainSpace's diffusion map embedding applies
# its own affinity kernel internally.

fc_matrices: Dict[str, np.ndarray] = {}

print("Loading functional connectivity matrices...")
for sub in subjects:
    # Look for the correlation matrix in the subject's subdirectory
    fc_path = FC_DIR / sub / "connectivity_correlation.npy"

    if not fc_path.exists():
        # Fallback: try Fisher-z version
        fc_path = FC_DIR / sub / "connectivity_correlation_fisherz.npy"

    if fc_path.exists():
        fc = np.load(str(fc_path))

        # Ensure symmetric and clean diagonal
        fc = (fc + fc.T) / 2
        np.fill_diagonal(fc, 0)
        fc_matrices[sub] = fc
        print(f"  {sub}: ✓ ({fc.shape[0]}×{fc.shape[1]})")
    else:
        print(f"  {sub}: ⚠ not found in {FC_DIR / sub}")

print(f"\nLoaded FC matrices: {len(fc_matrices)}/{len(subjects)} subjects")

# Subjects with both structural and functional data
subjects_both = [s for s in subjects if s in fc_matrices]
print(f"Subjects with both structural + functional: {len(subjects_both)}")


# %% [6] BRAINSPACE — Compute functional connectivity gradients
# ═══════════════════════════════════════════════════════════════════════════
# Functional connectivity gradients (Margulies et al., PNAS 2016) via
# diffusion map embedding of the FC matrix.
#
# Key fix: the original sparsity=0.9 was too aggressive for this cohort,
# producing disconnected graphs. We use a gentler threshold and ensure
# the graph stays fully connected by keeping at least the top-k neighbors
# for each node.

individual_gradients: Dict[str, np.ndarray] = {}

print("Computing functional connectivity gradients...")
t0 = time.time()

reference_gradients = None
failed_grads = []

for sub in subjects_both:
    fc = fc_matrices[sub]
    N = fc.shape[0]

    # ── Threshold FC to create a sparse affinity matrix ─────────────
    # Strategy: keep each node's top-k strongest connections to guarantee
    # connectivity, then additionally keep any edge above a percentile.
    # This avoids the disconnected-graph problem entirely.
    fc_thresh = fc.copy()

    # Step 1: Zero negative correlations (anti-correlations are ambiguous
    # for diffusion map embedding — they would create negative affinities)
    fc_thresh[fc_thresh < 0] = 0

    # Step 2: For each node, keep at least top-10 connections
    # This guarantees every node has degree >= 10 → graph stays connected
    min_neighbors = 10
    for row in range(N):
        row_vals = fc_thresh[row].copy()
        row_vals[row] = 0  # exclude self
        if np.count_nonzero(row_vals) > min_neighbors:
            threshold_row = np.sort(row_vals)[::-1][min_neighbors]
            mask = (row_vals < threshold_row) & (row_vals > 0)
            fc_thresh[row, mask] = 0

    # Step 3: Symmetrize (keep edge if EITHER node considers it strong)
    fc_thresh = np.maximum(fc_thresh, fc_thresh.T)
    np.fill_diagonal(fc_thresh, 0)

    # Verify connectivity
    degree = (fc_thresh > 0).sum(axis=1)
    if degree.min() == 0:
        print(f"  {sub}: ⚠ still disconnected (min degree=0), skipping")
        failed_grads.append(sub)
        continue

    # ── Compute gradients ───────────────────────────────────────────
    try:
        gm = GradientMaps(
            n_components=N_GRADIENTS,
            approach=GRADIENT_APPROACH,
            kernel=GRADIENT_KERNEL,
            random_state=42,
            alignment="procrustes" if reference_gradients is not None else None,
        )

        if reference_gradients is not None:
            gm.fit(fc_thresh, reference=reference_gradients)
        else:
            gm.fit(fc_thresh)
            reference_gradients = gm.gradients_

        individual_gradients[sub] = gm.gradients_

        g1 = gm.gradients_[:, 0]
        print(f"  {sub}: ✓  G1 range=[{g1.min():.3f}, {g1.max():.3f}], "
              f"span={g1.max()-g1.min():.3f}, "
              f"density={100*(fc_thresh>0).sum()/N**2:.1f}%")

    except Exception as e:
        print(f"  {sub}: ⚠ failed — {e}")
        failed_grads.append(sub)

elapsed = time.time() - t0
print(f"\nGradients computed: {len(individual_gradients)}/{len(subjects_both)} "
      f"subjects in {elapsed/60:.1f} min")

if failed_grads:
    print(f"  Failed: {failed_grads}")

# Compute group-average gradient for reference
if individual_gradients:
    all_grads = np.stack(list(individual_gradients.values()))
    group_mean_gradients = np.mean(all_grads, axis=0)
    print(f"Group mean gradient shape: {group_mean_gradients.shape}")

    # Update subjects_both to only include those with successful gradients
    subjects_both = [s for s in subjects_both if s in individual_gradients]


# %% [7] STRUCTURE-FUNCTION COUPLING — Parcel-wise analysis
# ═══════════════════════════════════════════════════════════════════════════
# This is the core analysis: for each subject, we ask whether the spectral
# shape descriptors (from CorticalFields) predict the functional gradient
# loadings (from BrainSpace) across Schaefer-200 parcels.
#
# The logic is:
#   - Each parcel has a 42D spectral feature vector (geometric signature)
#   - Each parcel has a gradient loading (e.g., G1 value)
#   - We compute the Spearman correlation between each spectral feature
#     and each gradient across the 200 parcels
#   - This gives a "structure-function coupling map" showing which aspects
#     of cortical geometry are linked to functional organization
#
# For the post-COVID analysis, we additionally compute:
#   - The canonical correlation between the full spectral profile and
#     the gradient vector (multivariate coupling)
#   - Per-network coupling (do some Yeo-7 networks show stronger
#     geometry-function relationships than others?)

def compute_parcel_level_coupling(
    spectral_feat: np.ndarray,
    gradient_loadings: np.ndarray,
    n_gradients: int = 3,
) -> pd.DataFrame:
    """
    Compute structure-function coupling at the parcel level.

    For each spectral feature dimension (42) and each gradient (3),
    computes the Spearman rank correlation across parcels.

    Parameters
    ----------
    spectral_feat : (R, 42) array — spectral features per parcel
    gradient_loadings : (R, G) array — gradient loadings per parcel
    n_gradients : int — how many gradients to analyze

    Returns
    -------
    DataFrame with columns: feature_idx, feature_type, gradient, rho, pvalue
    """
    R, D = spectral_feat.shape
    rows = []

    feat_types = (
        [f"hks_{i}" for i in range(HKS_SCALES)] +
        [f"wks_{i}" for i in range(WKS_ENERGIES)] +
        [f"gps_{i}" for i in range(GPS_COMPONENTS)]
    )

    for g in range(min(n_gradients, gradient_loadings.shape[1])):
        grad_vals = gradient_loadings[:, g]
        for d in range(D):
            feat_vals = spectral_feat[:, d]
            rho, pval = stats.spearmanr(feat_vals, grad_vals)
            rows.append({
                "feature_idx": d,
                "feature_name": feat_types[d] if d < len(feat_types) else f"f{d}",
                "feature_type": (
                    "HKS" if d < HKS_SCALES else
                    "WKS" if d < HKS_SCALES + WKS_ENERGIES else "GPS"
                ),
                "gradient": f"G{g+1}",
                "rho": rho,
                "pvalue": pval,
            })

    return pd.DataFrame(rows)


# Compute coupling for each subject
# We combine lh + rh parcels for a whole-brain analysis.
coupling_results: Dict[str, pd.DataFrame] = {}

print("Computing structure-function coupling per subject...")

for sub in subjects_both:
    # Combine hemispheres: spectral features
    spec_parts = []
    for hemi in ["lh", "rh"]:
        if (sub, hemi) in parcel_spectral:
            spec_parts.append(parcel_spectral[(sub, hemi)])
    if not spec_parts:
        continue
    spec_combined = np.vstack(spec_parts)  # (~200, 42)

    # Gradient loadings (already whole-brain, 200 parcels)
    grads = individual_gradients[sub]  # (200, N_GRADIENTS)

    # Ensure dimensions match (parcels may differ slightly)
    n_parcels = min(spec_combined.shape[0], grads.shape[0])
    spec_combined = spec_combined[:n_parcels]
    grads = grads[:n_parcels]

    # Compute coupling
    coupling_df = compute_parcel_level_coupling(spec_combined, grads, n_gradients=3)
    coupling_df["subject"] = sub
    coupling_results[sub] = coupling_df

    # Quick summary: strongest coupling with G1
    g1_df = coupling_df[coupling_df["gradient"] == "G1"]
    best = g1_df.loc[g1_df["rho"].abs().idxmax()]
    print(f"  {sub}: strongest G1 coupling = {best['feature_name']} "
          f"(ρ={best['rho']:+.3f}, p={best['pvalue']:.2e})")

# Combine all subjects into a single DataFrame for group analysis
if coupling_results:
    all_coupling = pd.concat(coupling_results.values(), ignore_index=True)
    print(f"\nTotal coupling results: {all_coupling.shape}")


# %% [8] NETWORK-LEVEL ANALYSIS — Yeo-7 aggregation
# ═══════════════════════════════════════════════════════════════════════════
# To understand structure-function coupling at the network level, we
# aggregate both spectral features and gradient loadings by Yeo-7 network.
#
# For each network, we compute:
#   1. Mean spectral profile (42D → characterizes network geometry)
#   2. Mean gradient loading (where the network sits on the G1 axis)
#   3. Within-network structure-function correlation
#   4. Spectral similarity between networks (cosine similarity of
#      spectral profiles → the Spectral Similarity Network from
#      CorticalFields' graphs module)
#
# This reveals which functional networks are geometrically similar and
# whether geometric similarity predicts functional proximity.

def aggregate_by_network(
    parcel_features: np.ndarray,
    parcel_networks: List[str],
    network_order: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """
    Aggregate parcel-level features by Yeo-7 network.

    Returns (N_networks, D) array and list of network names.
    """
    net_features = []
    net_names = []
    for net in network_order:
        idx = [i for i, n in enumerate(parcel_networks) if n == net]
        if idx:
            net_features.append(np.mean(parcel_features[idx], axis=0))
            net_names.append(net)
    return np.array(net_features), net_names


# Compute network-level profiles for each subject
network_spectral: Dict[str, np.ndarray] = {}  # sub → (7, 42)
network_gradients: Dict[str, np.ndarray] = {}  # sub → (7, N_GRADIENTS)
network_coupling: List[dict] = []

for sub in subjects_both:
    # Combine hemispheres for spectral features
    spec_parts = []
    net_parts = []
    for hemi in ["lh", "rh"]:
        if (sub, hemi) in parcel_spectral:
            spec_parts.append(parcel_spectral[(sub, hemi)])
            net_parts.extend(parcel_info[hemi][1])
    if not spec_parts:
        continue

    spec_combined = np.vstack(spec_parts)
    grads = individual_gradients[sub]
    n_parcels = min(spec_combined.shape[0], grads.shape[0])
    spec_combined = spec_combined[:n_parcels]
    grads = grads[:n_parcels]
    nets = net_parts[:n_parcels]

    # Aggregate by network
    net_spec, net_names = aggregate_by_network(spec_combined, nets, YEO7_NETWORKS)
    net_grad, _ = aggregate_by_network(grads, nets, YEO7_NETWORKS)
    network_spectral[sub] = net_spec
    network_gradients[sub] = net_grad

    # Per-network coupling (correlation within each network)
    for j, net in enumerate(net_names):
        idx = [i for i, n in enumerate(nets) if n == net]
        if len(idx) < 5:
            continue
        for g in range(min(3, grads.shape[1])):
            # Correlate spectral features (mean across 42D) with gradient
            spec_mean = np.mean(spec_combined[idx], axis=1)
            rho, pval = stats.spearmanr(spec_mean, grads[idx, g])
            network_coupling.append({
                "subject": sub,
                "network": net,
                "gradient": f"G{g+1}",
                "rho": rho,
                "pvalue": pval,
                "n_parcels": len(idx),
            })

network_coupling_df = pd.DataFrame(network_coupling)

# Summary: mean coupling per network
if len(network_coupling_df) > 0:
    print("Network-level structure-function coupling (mean ρ with G1):")
    print(f"  {'Network':<15s}  {'ρ (mean)':>10s}  {'ρ (std)':>10s}  {'n_subs':>7s}")
    print(f"  {'─'*15}  {'─'*10}  {'─'*10}  {'─'*7}")
    g1_net = network_coupling_df[network_coupling_df["gradient"] == "G1"]
    for net in YEO7_NETWORKS:
        net_df = g1_net[g1_net["network"] == net]
        if len(net_df) > 0:
            print(f"  {net:<15s}  {net_df['rho'].mean():+10.3f}  "
                  f"{net_df['rho'].std():10.3f}  {len(net_df):7d}")


# %% [9] VISUALIZATION — Publication-quality figures
# ═══════════════════════════════════════════════════════════════════════════
# Create figures suitable for journal submission. We follow the visual
# language established in the CorticalFields paper:
#   - Yeo-7 canonical colors
#   - Science/Nature style via SciencePlots
#   - Okabe-Ito colorblind-safe accents
#   - DejaVu Sans typography

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Spyder
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    plt.style.use(["science", "no-latex"])
except Exception:
    plt.style.use("seaborn-v0_8-whitegrid")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
})

FIG_DIR = OUTPUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Figure 1: Spectral feature × Gradient coupling heatmap ─────────────
# Shows which spectral features (HKS scales, WKS energies, GPS components)
# correlate most strongly with each functional gradient.

if coupling_results:
    # Group-average coupling (mean ρ across subjects)
    g1_all = all_coupling[all_coupling["gradient"] == "G1"]
    mean_rho = g1_all.groupby("feature_idx")["rho"].mean().values

    fig, ax = plt.subplots(figsize=(7, 3))

    # Color by feature type
    colors = []
    for d in range(len(mean_rho)):
        if d < HKS_SCALES:
            colors.append(YEO7_COLORS["Vis"])       # purple for HKS
        elif d < HKS_SCALES + WKS_ENERGIES:
            colors.append(YEO7_COLORS["Cont"])       # orange for WKS
        else:
            colors.append(YEO7_COLORS["DorsAttn"])   # green for GPS

    ax.bar(range(len(mean_rho)), mean_rho, color=colors, width=0.8, alpha=0.85)
    ax.axhline(0, color="gray", linewidth=0.5)

    # Annotations for feature blocks
    ax.axvline(HKS_SCALES - 0.5, color="gray", linewidth=0.3, linestyle="--")
    ax.axvline(HKS_SCALES + WKS_ENERGIES - 0.5, color="gray", linewidth=0.3, linestyle="--")

    ax.text(HKS_SCALES / 2, ax.get_ylim()[1] * 0.9, "HKS",
            ha="center", fontsize=8, color=YEO7_COLORS["Vis"], fontweight="bold")
    ax.text(HKS_SCALES + WKS_ENERGIES / 2, ax.get_ylim()[1] * 0.9, "WKS",
            ha="center", fontsize=8, color=YEO7_COLORS["Cont"], fontweight="bold")
    ax.text(HKS_SCALES + WKS_ENERGIES + GPS_COMPONENTS / 2, ax.get_ylim()[1] * 0.9, "GPS",
            ha="center", fontsize=8, color=YEO7_COLORS["DorsAttn"], fontweight="bold")

    ax.set_xlabel("Spectral feature index")
    ax.set_ylabel("Spearman ρ with G1")
    ax.set_title("Structure-Function Coupling: Spectral Features × Gradient 1")
    ax.set_xlim(-0.5, len(mean_rho) - 0.5)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_spectral_gradient_coupling.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig1_spectral_gradient_coupling.png", bbox_inches="tight")
    print("✓ Figure 1 saved: spectral_gradient_coupling")
    plt.close(fig)

# ── Figure 2: Network-level coupling radar chart ────────────────────────
# Shows the strength of structure-function coupling for each Yeo-7 network.

if len(network_coupling_df) > 0:
    g1_net = network_coupling_df[network_coupling_df["gradient"] == "G1"]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))

    angles = np.linspace(0, 2 * np.pi, len(YEO7_NETWORKS), endpoint=False).tolist()
    angles += angles[:1]

    values = []
    colors_list = []
    for net in YEO7_NETWORKS:
        net_df = g1_net[g1_net["network"] == net]
        values.append(abs(net_df["rho"].mean()) if len(net_df) > 0 else 0)
        colors_list.append(YEO7_COLORS.get(net, (0.5, 0.5, 0.5)))
    values += values[:1]

    ax.plot(angles, values, "o-", color="#D55E00", linewidth=1.5, markersize=5)
    ax.fill(angles, values, alpha=0.15, color="#D55E00")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(YEO7_NETWORKS, fontsize=8)

    # Color each label by Yeo-7 color
    for label, color in zip(ax.get_xticklabels(), colors_list):
        label.set_color(color)
        label.set_fontweight("bold")

    ax.set_title("|ρ| Structure-Function Coupling by Network\n(G1)", pad=20, fontsize=10)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_network_coupling_radar.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig2_network_coupling_radar.png", bbox_inches="tight")
    print("✓ Figure 2 saved: network_coupling_radar")
    plt.close(fig)

print(f"\nAll figures saved to: {FIG_DIR}")


# %% [10] EXPORT AND SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
# Save all results in organized CSV files for downstream analysis,
# reproducibility, and inclusion in supplementary materials.

print("Exporting results...\n")

# ── All coupling results ────────────────────────────────────────────────
if coupling_results:
    all_coupling.to_csv(
        OUTPUT_DIR / "parcel_coupling_all_subjects.csv",
        index=False, float_format="%.6f",
    )
    print(f"  ✓ parcel_coupling_all_subjects.csv ({all_coupling.shape})")

# ── Network-level coupling ──────────────────────────────────────────────
if len(network_coupling_df) > 0:
    network_coupling_df.to_csv(
        OUTPUT_DIR / "network_coupling.csv",
        index=False, float_format="%.6f",
    )
    print(f"  ✓ network_coupling.csv ({network_coupling_df.shape})")

# ── Group-average gradients ─────────────────────────────────────────────
if 'group_mean_gradients' in dir():
    np.save(OUTPUT_DIR / "group_mean_gradients.npy", group_mean_gradients)
    print(f"  ✓ group_mean_gradients.npy ({group_mean_gradients.shape})")

# ── Per-subject spectral features (parcel-level) ───────────────────────
rows = []
for (sub, hemi), spec in parcel_spectral.items():
    names, nets = parcel_info.get(hemi, ([], []))
    for j in range(min(spec.shape[0], len(names))):
        row = {"subject": sub, "hemi": hemi, "parcel": names[j], "network": nets[j]}
        for d in range(spec.shape[1]):
            row[f"spec_{d}"] = spec[j, d]
        rows.append(row)
if rows:
    spec_df = pd.DataFrame(rows)
    spec_df.to_csv(
        OUTPUT_DIR / "parcel_spectral_features.csv",
        index=False, float_format="%.6f",
    )
    print(f"  ✓ parcel_spectral_features.csv ({spec_df.shape})")

# ── Per-subject gradient loadings (parcel-level) ───────────────────────
rows = []
for sub, grads in individual_gradients.items():
    for j in range(grads.shape[0]):
        row = {"subject": sub, "parcel_idx": j}
        for g in range(grads.shape[1]):
            row[f"G{g+1}"] = grads[j, g]
        rows.append(row)
if rows:
    grad_df = pd.DataFrame(rows)
    grad_df.to_csv(
        OUTPUT_DIR / "parcel_gradient_loadings.csv",
        index=False, float_format="%.6f",
    )
    print(f"  ✓ parcel_gradient_loadings.csv ({grad_df.shape})")

# ═══════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  Structure-Function Coupling Tutorial — Summary")
print(f"{'='*70}")
print(f"  Cohort          : SARS-CoV-2 ICU survivors")
print(f"  Pipeline        : {PIPELINE}")
print(f"  Subjects (struct): {len(subjects)}")
print(f"  Subjects (funct) : {len(fc_matrices)}")
print(f"  Subjects (both)  : {len(subjects_both)}")
print(f"  Parcellation     : Schaefer-200, Yeo-7 networks")
print(f"  Spectral features: {HKS_SCALES} HKS + {WKS_ENERGIES} WKS + "
      f"{GPS_COMPONENTS} GPS = 42D per parcel")
print(f"  Gradients        : {N_GRADIENTS} components "
      f"({GRADIENT_APPROACH}, {GRADIENT_KERNEL})")
print(f"  Output           : {OUTPUT_DIR}")
print(f"  Figures          : {FIG_DIR}")
print(f"\n  Key findings to explore:")
print(f"    • Which HKS time scales couple most with G1?")
print(f"      → Small t (local curvature) or large t (global shape)?")
print(f"    • Which Yeo-7 networks show strongest coupling?")
print(f"      → Transmodal (DMN, FPN) or unimodal (Vis, SomMot)?")
print(f"    • Does coupling strength correlate with COVID severity?")
print(f"      → Link to ICU duration, ventilation days, biomarkers")
print(f"{'='*70}")
print(f"\nDone! ✓")
