# CorticalFields

**A unified framework for spectral cortical analysis on meshes and point clouds.**

CorticalFields extracts rich geometric information from structural MRI data by treating cortical and subcortical surfaces as Riemannian manifolds and applying principled spectral, probabilistic, and geometric deep learning methods. It operates natively on both triangle meshes (from FreeSurfer) and point clouds (mesh-free), with GPU acceleration and explicit VRAM management throughout.

Designed for epilepsy research (MTLE-HS) but applicable to any structural neuroimaging study.

## What it does

CorticalFields provides three interlocking analysis pipelines — spectral geometry, atlas-free asymmetry, and Bayesian inference — unified by the Laplace-Beltrami operator as the mathematical backbone.

**Spectral geometry pipeline.** Decomposes cortical surfaces via the Laplace-Beltrami operator, computes multi-scale shape descriptors (HKS, WKS, GPS), builds geodesic-aware Matérn GP kernels, fits normative models, and generates vertex-wise surprise (anomaly) maps — all respecting the manifold's intrinsic geometry rather than Euclidean space.

**Atlas-free asymmetry pipeline.** Quantifies hemispheric asymmetry without parcellations using functional maps (C matrix) and optimal transport (Wasserstein / Sinkhorn distances), decomposed by spatial frequency band and linked to clinical outcomes via kernel-based statistical inference (MDMR, HSIC, KRR).

**Bayesian neuroimaging toolkit.** Ten model classes (regularised horseshoe, R2-D2, BEST, hierarchical, mediation, DAG, and more) with four sampler backends (PyMC/NUTS, nutpie, NumPyro, BlackJAX) and twenty publication-quality diagnostic/visualization functions, purpose-built for high-dimensional, small-sample clinical neuroimaging.

**Point cloud processing (new in v0.2).** A complete mesh-free path from raw T1w MRI to spectral analysis, including GPU-accelerated LBO on point clouds, spectral descriptors, functional maps, optimal transport, morphometrics (area, volume, curvature, thickness, gyrification), registration (ICP, CPD, spectral), and geometric deep learning (DiffusionNet, EGNN) — all with automatic CPU fallback and VRAM-safe batching for large cortical surfaces (~150K points).

## Key innovations

**Discretization-agnostic spectral analysis.** The same mathematical pipeline — LBO eigenpairs → HKS/WKS/GPS → functional maps → optimal transport — operates on both triangle meshes and raw point clouds, following the DiffusionNet paradigm (Sharp, Attaiki & Crane, 2022). The `robust_laplacian` library provides provably convergent LBO on point clouds via intrinsic Delaunay triangulation.

**Geodesic-aware GP kernel.** Uses the spectral Matérn kernel (Borovitskiy et al., NeurIPS 2020) — the only mathematically correct way to define Matérn GPs on curved manifolds. Naïve geodesic-distance substitution does *not* yield a valid kernel.

**Information-theoretic surprise maps.** Vertex-wise negative log-predictive density from a calibrated GP — theoretically optimal anomaly scoring that goes beyond z-scores.

**Spectral shape descriptors for neuroimaging.** Brings HKS/WKS from computer graphics into clinical brain imaging, capturing multi-scale geometry from local curvature to global lobe membership.

**Atlas-free asymmetry via functional maps + optimal transport.** The inter-hemispheric C matrix decomposes asymmetry by spatial frequency — global shape vs. fine gyrification — without any parcellation. Wasserstein distances provide a continuous metric with provably positive-definite kernels for regression.

**GPU-accelerated point cloud morphometrics.** Surface area (Voronoi), volume (divergence theorem), curvature (quadric fitting), cortical thickness (paired surfaces), and gyrification index — all computed directly from point clouds with explicit VRAM management.

**Geometric deep learning for brain analysis.** Self-contained DiffusionNet encoder (learned heat diffusion on LBO eigenbasis) and E(n)-equivariant graph neural networks (EGNN) operating natively on point clouds, with gradient checkpointing and batched inference for 8 GB GPUs.

## Installation

```bash
# Core spectral pipeline
pip install corticalfields

# Point cloud processing (LBO, morphometrics, Open3D)
pip install corticalfields[pointcloud]

# Point cloud + deep learning (DiffusionNet, EGNN, PyG)
pip install corticalfields[pointcloud-dl]

# Point cloud + GPU optimal transport (GeomLoss, POT)
pip install corticalfields[pointcloud-ot]

# All point cloud extras
pip install corticalfields[all-pointcloud]

# Bayesian analysis (PyMC, ArViz, nutpie)
pip install corticalfields[bayesian-fast]

# Everything
pip install corticalfields[all]

# Development installation
git clone https://github.com/rdneuro/corticalfields.git
cd corticalfields
pip install -e ".[all,dev]"
```

### Requirements

Python ≥ 3.9 and PyTorch ≥ 2.0 (with CUDA for GPU acceleration). FreeSurfer is optional — the point cloud pipeline extracts cortical surfaces directly from T1w NIfTI files via deep learning-based brain extraction and marching cubes.

## Quick start

### Normative modeling and surprise maps

```python
import corticalfields as cf
from corticalfields.spectral import compute_eigenpairs

# 1. Load a FreeSurfer surface
surf = cf.load_freesurfer_surface(
    subjects_dir="/data/freesurfer",
    subject_id="fsaverage",
    hemi="lh", surface="pial",
    overlays=["thickness", "curv", "sulc"],
)

# 2. Compute Laplace-Beltrami eigenpairs
lb = compute_eigenpairs(surf.vertices, surf.faces, n_eigenpairs=300)

# 3. Spectral shape descriptors
hks = cf.heat_kernel_signature(lb, n_scales=16)
wks = cf.wave_kernel_signature(lb, n_energies=16)
gps = cf.global_point_signature(lb, n_components=10)

# 4. Train normative model on reference cohort
model = cf.CorticalNormativeModel(lb, nu=2.5, n_inducing=512, device="cuda")
model.fit(reference_thickness_matrix, feature_name="thickness", n_epochs=100)

# 5. Score a patient
result = model.predict(patient_thickness)
print(f"Mean surprise: {result.surprise.mean():.2f}")
print(f"Fraction anomalous (|z| > 2): {(abs(result.z_score) > 2).mean():.1%}")
```

### Point cloud spectral analysis (mesh-free)

```python
from corticalfields.pointcloud import (
    from_freesurfer_surface,
    compute_pointcloud_eigenpairs,
)
from corticalfields.pointcloud.spectral import spectral_features
from corticalfields.pointcloud.morphometrics import (
    compute_surface_area, compute_curvature,
)

# 1. Load hemisphere as point cloud (discards mesh faces)
pc = from_freesurfer_surface("/data/fs", "sub-01", "lh", "pial",
                              overlays=["thickness", "curv"])

# 2. LBO eigenpairs directly on the point cloud (Sharp & Crane 2020)
lb = compute_pointcloud_eigenpairs(pc.points, n_eigenpairs=300)

# 3. GPU-accelerated spectral descriptors (auto CPU fallback)
feats = spectral_features(lb.eigenvalues, lb.eigenvectors,
                          hks_scales=16, wks_scales=100, use_gpu=True)
# feats['hks']: (N, 16), feats['wks']: (N, 100), feats['gps']: (N, 50)

# 4. Morphometrics from point cloud
area, point_areas = compute_surface_area(pc.points, method="voronoi")
mean_H, gauss_K = compute_curvature(pc.points, method="quadric_fit")
```

### Atlas-free cortical asymmetry

```python
from corticalfields.pointcloud import from_freesurfer_surface, compute_pointcloud_eigenpairs
from corticalfields.pointcloud.spectral import hks_pointcloud
from corticalfields.pointcloud.functional_maps import (
    compute_descriptors, compute_functional_map, zoomout_refinement,
)
from corticalfields.pointcloud.transport import interhemispheric_ot

# 1. Load both hemispheres and mirror the right
pc_lh = from_freesurfer_surface(SDIR, "sub-01", "lh", "pial")
pc_rh = from_freesurfer_surface(SDIR, "sub-01", "rh", "pial")
pc_rh_m = pc_rh.mirror_x()  # x → −x for correspondence

# 2. Point cloud LBO on each hemisphere
lb_lh = compute_pointcloud_eigenpairs(pc_lh.points, n_eigenpairs=200)
lb_rh = compute_pointcloud_eigenpairs(pc_rh_m.points, n_eigenpairs=200)

# 3. Inter-hemispheric functional map + ZoomOut refinement
desc_lh = compute_descriptors(lb_lh.eigenvalues, lb_lh.eigenvectors)
desc_rh = compute_descriptors(lb_rh.eigenvalues, lb_rh.eigenvectors)
fmap = compute_functional_map(
    lb_lh.eigenvectors, lb_rh.eigenvectors,
    desc_lh, desc_rh,
    lb_lh.eigenvalues, lb_rh.eigenvalues,
    n_basis=50,
)
fmap = zoomout_refinement(fmap, n_iters=10, use_gpu=True)

# 4. Interhemispheric asymmetry via optimal transport
hks_lh = hks_pointcloud(lb_lh.eigenvalues, lb_lh.eigenvectors, n_scales=16)
hks_rh = hks_pointcloud(lb_rh.eigenvalues, lb_rh.eigenvectors, n_scales=16)
ot_dists = interhemispheric_ot(hks_lh, hks_rh,
                                methods=["sliced_wasserstein", "sinkhorn"])
print(f"Sliced Wasserstein: {ot_dists['sliced_wasserstein']:.4f}")
print(f"Sinkhorn divergence: {ot_dists['sinkhorn']:.4f}")
```

### Bayesian neuroimaging analysis

```python
from corticalfields.bayesian import HorseshoeRegression, SamplerConfig, PUBLICATION
from corticalfields.bayes_viz import plot_forest, plot_posterior_hdi_rope

# Regularised horseshoe for high-dimensional volumetric predictors
model = HorseshoeRegression()
idata = model.fit(
    X=brain_volumes,         # (n_subjects, n_regions)
    y=hads_depression,       # continuous clinical outcome
    feature_names=region_names,
    sampler=SamplerConfig(**PUBLICATION),
)

# Posterior forest plot with ROPE decision regions
fig = plot_forest(idata, rope=(-0.1, 0.1), hdi_prob=0.95)
fig = plot_posterior_hdi_rope(idata, var_name="beta", rope=(-0.1, 0.1))
```

### Publication-quality brain visualization

```python
from corticalfields.pointcloud.viz import (
    plot_pointcloud_brain, plot_eigenpairs, plot_spectral_descriptors,
    plot_curvature_map, plot_asymmetry_pointcloud,
)

# 4-view brain render with HKS overlay
plot_pointcloud_brain(pc.points, scalars=feats['hks'][:, 8],
                      cmap="inferno", scalar_name="HKS (t=8)",
                      save_path="figures/hks_brain.png")

# Eigenvector visualization
plot_eigenpairs(pc.points, lb.eigenvectors, lb.eigenvalues,
                modes=[1, 2, 5, 10, 20, 50],
                save_path="figures/eigenpairs.png")

# Curvature map
plot_curvature_map(pc.points, mean_H, gauss_K,
                   save_path="figures/curvature.png")
```

### GPU backend selection

```python
import corticalfields as cf

# Check what's available
print(cf.available_backends())  # {'scipy': True, 'cupy': True, 'torch': True}

# GPU eigensolver (3-10× faster)
lb = cf.spectral.compute_eigenpairs(
    surf.vertices, surf.faces,
    n_eigenpairs=300,
    backend="cupy",
)

# VRAM monitoring
print(cf.vram_report())
# → {'device': 'NVIDIA RTX 4070', 'total_gb': 8.0, 'used_gb': 2.1, 'free_gb': 5.9}
```

## Architecture

```
corticalfields/
│
│  ── Core spectral pipeline ──────────────────────────────────────
├── surface.py             Surface I/O (FreeSurfer, GIfTI, mesh utilities)
├── spectral.py            Laplace-Beltrami decomposition, HKS, WKS, GPS
├── backends.py            GPU backend abstraction (SciPy / CuPy / PyTorch)
├── kernels.py             Spectral Matérn kernels for GPyTorch
├── features.py            Morphometric feature extraction from FreeSurfer
│
│  ── Point cloud processing (new in v0.2) ────────────────────────
├── pointcloud/
│   ├── __init__.py        Backward-compatible re-exports + lazy loading
│   ├── spectral.py        GPU-accelerated HKS/WKS/GPS + eigenpair validation
│   ├── functional_maps.py Functional map C matrix + ZoomOut on point clouds
│   ├── transport.py       Wasserstein + Sinkhorn GPU (GeomLoss/POT)
│   ├── morphometrics.py   Area, volume, curvature, thickness, gyrification
│   ├── registration.py    ICP, Coherent Point Drift, spectral registration
│   ├── viz.py             PyVista 4-view brain renders with scalar overlays
│   └── deep/
│       ├── diffusion_net.py   DiffusionNet encoder (self-contained)
│       └── egnn.py            E(n)-equivariant graph neural networks
│
│  ── Normative modeling & anomaly detection ──────────────────────
├── normative.py           GP-based normative modeling pipeline
├── surprise.py            Information-theoretic anomaly scoring
├── graphs.py              Cortical similarity networks (MSN, SSN)
│
│  ── Atlas-free asymmetry (functional maps + optimal transport) ──
├── functional_maps.py     C matrix, ZoomOut, inter-hemispheric correspondence
├── transport.py           Wasserstein, Sinkhorn, FUGW, OT kernels
├── asymmetry.py           Multi-scale atlas-free asymmetry quantification
│
│  ── Statistical inference ───────────────────────────────────────
├── bayesian.py            10 Bayesian model classes (horseshoe, R2-D2, …)
├── distance_stats.py      MDMR, HSIC, distance correlation, Mantel, KRR
│
│  ── Visualization ───────────────────────────────────────────────
├── brainplots.py          Publication-grade brain figures (surfaces, composites)
├── bayes_viz.py           20 Bayesian diagnostic/posterior plots
├── viz.py                 Quick brain scatter + trisurf visualization
│
│  ── Subcortical + utilities ─────────────────────────────────────
├── subcortical.py         Subcortical surface extraction + spectral pipeline
├── datasets.py            Toy dataset download from Zenodo
├── eda_qc.py              Exploratory data analysis + quality control
└── utils.py               GPU detection, VRAM management, parcellation loading
```

**32 modules · ~21,400 lines · ~300 public symbols**

## Mathematical foundations

The library rests on five mathematical pillars.

**1. Laplace-Beltrami spectral decomposition.** On a compact Riemannian manifold (Σ, g), the LB operator Δ_g has a discrete spectrum 0 = λ₀ ≤ λ₁ ≤ λ₂ ≤ … with orthonormal eigenfunctions φᵢ. These encode intrinsic geometry at all scales — low eigenvalues capture global shape, high eigenvalues capture fine curvature. For point clouds without mesh connectivity, the intrinsic Delaunay approach (Sharp & Crane, 2020) provably converges to the true LBO.

**2. Spectral Matérn kernel.** The Matérn kernel on the manifold is defined via the spectral density: k(x,y) = σ² Σᵢ S_ν(λᵢ) φᵢ(x) φᵢ(y), where S_ν(λ) = (2ν/κ² + λ)^{−(ν+d/2)}. This yields a valid positive-definite kernel that respects manifold geometry, unlike naïve geodesic-distance substitution.

**3. Information-theoretic surprise.** For a GP with posterior predictive p(y*|x,D) = N(μ(x), σ²(x)), the surprise at vertex x is S(x) = −log p(y_obs|x,D) = ½ log(2πσ²) + (y_obs − μ)²/(2σ²), the theoretically optimal anomaly score under the model.

**4. Functional maps and spectral asymmetry.** A functional map C between two surfaces maps functions in the eigenbasis of one to the eigenbasis of the other (Ovsjanikov et al., 2012). For inter-hemispheric correspondence, the off-diagonal Frobenius energy ‖C − diag(C)‖_F measures asymmetry, decomposable by eigenfunction index (spatial frequency). Low-frequency off-diagonal energy indicates global shape asymmetry; high-frequency indicates gyrification differences.

**5. Optimal transport on cortical geometry.** The sliced Wasserstein distance between hemispheric point clouds provides a continuous, atlas-free asymmetry metric. The Gaussian kernel exp(−γ · SW²) on sliced Wasserstein distances is provably positive definite (Kolouri et al., 2016), enabling kernel-based regression and independence testing that link cortical geometry directly to clinical outcomes.

## References

- Borovitskiy et al. (2020). Matérn Gaussian Processes on Riemannian Manifolds. *NeurIPS*.
- Sun, Ovsjanikov & Guibas (2009). Heat Kernel Signature. *SGP*.
- Aubry, Schlickewei & Cremers (2011). Wave Kernel Signature. *ICCV*.
- Sharp & Crane (2020). A Laplacian for Nonmanifold Triangle Meshes. *SGP*.
- Sharp, Attaiki & Crane (2022). DiffusionNet: Discretization Agnostic Learning on Surfaces. *ACM TOG*.
- Ovsjanikov et al. (2012). Functional Maps: a Flexible Representation of Maps Between Shapes. *ACM TOG*.
- Melzi et al. (2019). ZoomOut: Spectral Upsampling for Efficient Shape Correspondence. *ACM TOG*.
- Satorras, Hoogeboom & Welling (2021). E(n) Equivariant Graph Neural Networks. *ICML*.
- Kolouri et al. (2016). Sliced Wasserstein Kernels for Probability Distributions. *CVPR*.
- Feydy et al. (2019). Interpolating between Optimal Transport and MMD using Sinkhorn Divergences. *AISTATS*.
- Thual et al. (2022). Aligning individual brains with Fused Unbalanced Gromov-Wasserstein. *NeurIPS*.
- Seidlitz et al. (2018). Morphometric Similarity Networks. *Neuron*.
- Marquand et al. (2019). Normative modeling in neuroimaging. *Nature Protocols*.
- Zhu et al. (2025). Geometric Deep Learning with Adaptive Full-Band Spatial Diffusion for Cortical Parcellation. *Medical Image Analysis*.

## Citation

If you use CorticalFields in your research, please cite:

```bibtex
@software{corticalfields2026,
  author = {Debona, R.},
  title = {CorticalFields: Spectral cortical analysis on meshes and point clouds},
  year = {2026},
  url = {https://github.com/rdneuro/corticalfields},
}
```

## License

MIT
