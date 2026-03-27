# CorticalFields

**Geodesic-aware GP normative modeling, atlas-free cortical asymmetry, and Bayesian neuroimaging analysis on cortical surfaces.**

CorticalFields extracts rich geometric information from structural MRI (T1w) data — without needing fMRI or dMRI — by treating the cortical surface as a Riemannian manifold and applying principled probabilistic modeling. It was designed for epilepsy research (MTLE-HS) but is applicable to any structural neuroimaging study.

## What it does

Given FreeSurfer-processed T1w data, CorticalFields:

1. **Decomposes the cortical geometry** via the Laplace–Beltrami operator, extracting the intrinsic spectral structure of the cortical mesh or point cloud.
2. **Computes spectral shape descriptors** (Heat Kernel Signature, Wave Kernel Signature, Global Point Signature) that capture multi-scale geometric information at every vertex — from local curvature to global lobe membership.
3. **Builds a spectral Matérn GP** that respects the cortical manifold's geometry (not Euclidean space), producing a proper probabilistic model of "normal" brain morphology.
4. **Generates vertex-wise surprise maps** — information-theoretic anomaly scores that quantify how unexpected each vertex's morphometry is, given the normative model and the surrounding cortical geometry.
5. **Aggregates anomalies by functional network** (Yeo-7, Schaefer-200, etc.), enabling statements like *"this patient has anomalous DMN structure"* from T1w data alone.
6. **Quantifies hemispheric asymmetry without atlases** — using functional maps (C matrix) and optimal transport (Wasserstein distances) to measure continuous, frequency-decomposed asymmetry between hemispheres.
7. **Tests geometric–clinical associations** — distance-based inference (MDMR, HSIC, kernel ridge regression) links cortical geometry directly to clinical outcomes like HADS anxiety/depression scores.
8. **Performs Bayesian statistical analysis** — 10 model classes (horseshoe, R2-D2, BEST, hierarchical, mediation, DAG, and more) with 4 sampler backends and 20 publication-quality plotting functions, purpose-built for neuroimaging volumetric studies.

## Key innovations

- **Geodesic-aware GP kernel**: Uses the spectral Matérn kernel (Borovitskiy et al., NeurIPS 2020) — the only mathematically correct way to define Matérn GPs on curved manifolds. Naïve geodesic distance substitution does NOT yield a valid kernel.
- **Surprise maps**: Vertex-wise negative log-predictive density from a calibrated probabilistic model — theoretically optimal anomaly scoring that goes beyond z-scores.
- **Spectral shape descriptors for neuroimaging**: Brings HKS/WKS from computer graphics into clinical brain imaging for the first time.
- **Atlas-free asymmetry via functional maps + optimal transport**: The inter-hemispheric C matrix (Ovsjanikov et al., 2012) decomposes asymmetry by spatial frequency — global shape vs. fine gyrification — without any parcellation. Wasserstein distances provide a complementary continuous metric with provably positive-definite kernels for regression.
- **Distance-based statistical inference**: MDMR, HSIC, distance correlation, and kernel ridge regression operate directly on geometric distance/kernel matrices, preserving the full richness of non-Euclidean cortical representations.
- **Bayesian neuroimaging toolkit**: Regularised horseshoe and R2-D2 regression with ROPE+HDI decision rules, ENIGMA-informed priors, and automated diagnostics — designed for the high-dimensional, small-sample regime of clinical neuroimaging.
- **Scalability to 150k vertices**: Spectral truncation (≤1000 LB eigenpairs) + variational sparse GP (≤1000 inducing points) makes the pipeline tractable on a single GPU.

## Installation

```bash
# Core installation
pip install corticalfields

# Full installation (includes optional dependencies)
pip install corticalfields[full]

# Development installation
git clone https://github.com/rdneuro/corticalfields.git
cd corticalfields
pip install -e ".[full,dev]"

# For atlas-free asymmetry pipeline
pip install POT robust-laplacian

# For GPU-accelerated Sinkhorn (optional)
pip install geomloss pykeops

# For FUGW alignment (optional)
pip install fugw
```

### Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0 (with CUDA for GPU acceleration)
- FreeSurfer (for surface reconstruction via `recon-all`)
- Core: numpy, scipy, nibabel, gpytorch, matplotlib, pyvista
- Bayesian: pymc, arviz, preliz (+ optional: nutpie, numpyro, blackjax)
- Asymmetry: POT, robust-laplacian (+ optional: geomloss, pykeops, fugw)

## Quick start

### Normative modeling and surprise maps

```python
import corticalfields as cf
from corticalfields.spectral import compute_eigenpairs
from corticalfields.utils import get_device, setup_logging

setup_logging("INFO")
device = get_device()  # auto-detect CUDA

# 1. Load a FreeSurfer surface
surf = cf.load_freesurfer_surface(
    subjects_dir="/data/freesurfer",
    subject_id="fsaverage",
    hemi="lh",
    surface="pial",
    overlays=["thickness", "curv", "sulc"],
)

# 2. Compute Laplace–Beltrami eigenpairs
lb = compute_eigenpairs(
    surf.vertices, surf.faces,
    n_eigenpairs=300,  # 300 is a good default
)

# 3. Compute spectral shape descriptors
hks = cf.heat_kernel_signature(lb, n_scales=16)
wks = cf.wave_kernel_signature(lb, n_energies=16)
gps = cf.global_point_signature(lb, n_components=10)

# 4. Train normative model on reference cohort thickness
model = cf.CorticalNormativeModel(lb, nu=2.5, n_inducing=512, device=device)
model.fit(reference_thickness_matrix, feature_name="thickness", n_epochs=100)

# 5. Score a patient
result = model.predict(patient_thickness)
print(f"Mean surprise: {result.surprise.mean():.2f}")
print(f"Fraction anomalous (|z| > 2): {(abs(result.z_score) > 2).mean():.1%}")

# 6. Aggregate by network
from corticalfields.surprise import compute_surprise
from corticalfields.utils import load_fsaverage_parcellation

yeo_labels, yeo_names = load_fsaverage_parcellation("yeo_7", hemi="lh")
surprise_map = compute_surprise(
    patient_thickness, result.mean, result.variance,
)
network_scores = surprise_map.aggregate_by_network(yeo_labels, yeo_names)
```

### Atlas-free cortical asymmetry

Quantify hemispheric asymmetry without parcellations using functional maps and optimal transport — decomposed by spatial frequency band and linked to clinical outcomes via kernel regression.

```python
from corticalfields.pointcloud import from_freesurfer_surface, compute_pointcloud_eigenpairs
from corticalfields.functional_maps import compute_interhemispheric_map
from corticalfields.asymmetry import asymmetry_from_functional_map
from corticalfields.transport import pairwise_wasserstein_matrix, wasserstein_kernel
from corticalfields.distance_stats import mdmr, hsic, kernel_ridge_regression, outcome_kernel

SUBJECTS_DIR = "/data/freesurfer"

# 1. Load hemispheres as point clouds (mesh-free path)
pc_lh = from_freesurfer_surface(SUBJECTS_DIR, "sub-01", "lh", "pial")
pc_rh = from_freesurfer_surface(SUBJECTS_DIR, "sub-01", "rh", "pial")

# 2. Compute LBO eigenpairs on point clouds (Sharp & Crane 2020)
lb_lh = compute_pointcloud_eigenpairs(pc_lh.points, n_eigenpairs=300)
lb_rh = compute_pointcloud_eigenpairs(pc_rh.points, n_eigenpairs=300)

# 3. Inter-hemispheric functional map + ZoomOut refinement
fm = compute_interhemispheric_map(lb_lh, lb_rh, k=50, k_final=200)

# 4. Atlas-free asymmetry decomposed by frequency
profile = asymmetry_from_functional_map(fm, "sub-01", hemi_lateralisation="L")
print(f"Total asymmetry:  {profile.total_asymmetry:.4f}")
print(f"Low-freq (shape): {profile.band_asymmetry['low_freq']:.4f}")
print(f"High-freq (gyri): {profile.band_asymmetry['high_freq']:.4f}")

# 5. Cohort-level: Wasserstein distance matrix → PD kernel
D = pairwise_wasserstein_matrix(all_lh_clouds, method="sliced", n_projections=200)
K = wasserstein_kernel(D)

# 6. Link geometry to HADS scores (controlling for age and sex)
result = mdmr(D, hads_anxiety, covariates=np.c_[age, sex], n_permutations=10000)
print(f"MDMR: F={result.statistic:.2f}, p={result.p_value:.4f}, R²={result.effect_size:.4f}")

# 7. Kernel ridge regression: predict HADS from cortical geometry
krr = kernel_ridge_regression(K, hads_anxiety, covariates=np.c_[age, sex])
print(f"KRR cross-validated R²={krr['r2_cv']:.3f}, MAE={krr['mae_cv']:.2f}")
```

### Bayesian neuroimaging analysis

Purpose-built Bayesian regression models for volumetric studies with regularised priors, ENIGMA-informed effect sizes, and automated diagnostics.

```python
from corticalfields.bayesian import HorseshoeRegression, SamplerConfig, PUBLICATION
from corticalfields.bayes_viz import plot_forest, plot_posterior_hdi_rope

# Regularised horseshoe for high-dimensional volumetric predictors
model = HorseshoeRegression()
idata = model.fit(
    X=brain_volumes,         # (n_subjects, n_regions) — e.g., thalamic nuclei
    y=hads_depression,       # continuous clinical outcome
    feature_names=region_names,
    sampler=SamplerConfig(**PUBLICATION),  # 4 chains × 2000 draws, nutpie
)

# Posterior forest plot with ROPE decision regions
fig = plot_forest(idata, rope=(-0.1, 0.1), hdi_prob=0.95)

# ROPE + HDI classification: practically significant, undecided, or null
fig = plot_posterior_hdi_rope(idata, var_name="beta", rope=(-0.1, 0.1))
```

## Practical GPU-Backend Use

The library offers 3 different backends for computationally demanding operations. The "scipy" backend provides reliable CPU-based computation. "CuPy" or "torch" backends accelerate operations through GPU-accelerated linear algebra.

```python
import corticalfields as cf

# Check what's available
print(cf.available_backends())
# → {'scipy': True, 'cupy': True, 'torch': True}

# GPU eigensolver + GPU descriptors (3-10× faster)
lb = cf.spectral.compute_eigenpairs(
    surf.vertices, surf.faces,
    n_eigenpairs=300,
    backend="cupy",        # ← CuPy LOBPCG on GPU
    dtype="float32",       # ← 2× less VRAM, often faster
)

# GPU descriptors (10-30× faster for GEMM)
features = cf.spectral.spectral_feature_matrix(
    lb, backend="cupy",    # ← cuBLAS GEMM on GPU
)
```

## Full pipeline example

See `examples/full_pipeline.py` for a complete walkthrough from raw FreeSurfer outputs to publication-quality surprise map figures.

## Architecture

```
corticalfields/
│
│  ── Core spectral pipeline ──────────────────────────────────────
├── surface.py          # Surface I/O (FreeSurfer, GIfTI, mesh utilities)
├── pointcloud.py       # Mesh-free point cloud geometry (LBO on point clouds)
├── spectral.py         # Laplace–Beltrami decomposition, HKS, WKS, GPS
├── backends.py         # GPU backend abstraction (SciPy, CuPy, PyTorch)
├── kernels.py          # Spectral Matérn kernels for GPyTorch
├── features.py         # Morphometric feature extraction
│
│  ── Normative modeling & anomaly detection ──────────────────────
├── normative.py        # GP-based normative modeling pipeline
├── surprise.py         # Information-theoretic anomaly scoring
├── graphs.py           # Cortical similarity networks (MSN, SSN)
│
│  ── Atlas-free asymmetry (functional maps + optimal transport) ──
├── functional_maps.py  # C matrix, ZoomOut, inter-hemispheric correspondence
├── transport.py        # Wasserstein distances, Sinkhorn, FUGW, OT kernels
├── asymmetry.py        # Multi-scale atlas-free asymmetry quantification
│
│  ── Statistical inference ───────────────────────────────────────
├── bayesian.py         # 10 Bayesian model classes (horseshoe, R2-D2, BEST, …)
├── distance_stats.py   # MDMR, HSIC, distance correlation, Mantel, KRR
│
│  ── Visualization ───────────────────────────────────────────────
├── bayes_viz.py        # 20 publication-quality Bayesian plots
├── viz.py              # Surface visualization (PyVista + matplotlib)
│
│  ── Utilities ───────────────────────────────────────────────────
└── utils.py            # GPU detection, parcellation loading, timing
```

**17 modules · ~10,000 lines · 70 public exports**

## Mathematical foundations

The library rests on five mathematical pillars:

**1. Laplace–Beltrami spectral decomposition.** On a compact Riemannian manifold (Σ, g), the LB operator Δ_g has a discrete spectrum 0 = λ₀ ≤ λ₁ ≤ λ₂ ≤ … with orthonormal eigenfunctions φᵢ. These encode the intrinsic geometry of the surface at all scales — low eigenvalues capture global shape, high eigenvalues capture fine-grained curvature. For point clouds without mesh connectivity, the intrinsic Delaunay approach (Sharp & Crane, 2020) provably converges to the true LBO.

**2. Spectral Matérn kernel.** The Matérn kernel on the manifold is defined via the spectral density: k(x,y) = σ² Σᵢ S_ν(λᵢ) φᵢ(x) φᵢ(y), where S_ν(λ) = (2ν/κ² + λ)^{−(ν+d/2)}. This yields a valid positive-definite kernel that respects the manifold geometry, unlike naïve geodesic-distance substitution.

**3. Information-theoretic surprise.** For a GP with posterior predictive p(y*|x,D) = N(μ(x), σ²(x)), the surprise at vertex x is S(x) = −log p(y_obs|x,D) = ½ log(2πσ²) + (y_obs − μ)²/(2σ²). This is the theoretically optimal anomaly score under the model.

**4. Functional maps and spectral asymmetry.** A functional map C between two surfaces maps functions in the eigenbasis of one to the eigenbasis of the other (Ovsjanikov et al., 2012). For inter-hemispheric correspondence, the off-diagonal Frobenius energy ‖C − diag(C)‖_F measures asymmetry — decomposable by eigenfunction index (spatial frequency). Low-frequency off-diagonal energy indicates global shape asymmetry; high-frequency indicates gyrification differences.

**5. Optimal transport on cortical geometry.** The sliced Wasserstein distance between hemispheric point clouds provides a continuous, atlas-free asymmetry metric. The Gaussian kernel exp(−γ · SW²) on sliced Wasserstein distances is provably positive definite (Kolouri et al., 2016), enabling kernel-based regression (KRR) and independence testing (HSIC) that link cortical geometry directly to clinical outcomes.

## References

- Borovitskiy et al. (2020). Matérn Gaussian Processes on Riemannian Manifolds. *NeurIPS*.
- Sun, Ovsjanikov & Guibas (2009). Heat Kernel Signature. *SGP*.
- Aubry, Schlickewei & Cremers (2011). Wave Kernel Signature. *ICCV*.
- Sharp & Crane (2020). A Laplacian for Nonmanifold Triangle Meshes. *SGP*.
- Ovsjanikov et al. (2012). Functional Maps: a Flexible Representation of Maps Between Shapes. *ACM TOG*.
- Melzi et al. (2019). ZoomOut: Spectral Upsampling for Efficient Shape Correspondence. *ACM TOG*.
- Kolouri et al. (2016). Sliced Wasserstein Kernels for Probability Distributions. *CVPR*.
- Feydy et al. (2019). Interpolating between Optimal Transport and MMD using Sinkhorn Divergences. *AISTATS*.
- Thual et al. (2022). Aligning individual brains with Fused Unbalanced Gromov-Wasserstein. *NeurIPS*.
- Seidlitz et al. (2018). Morphometric Similarity Networks. *Neuron*.
- Marquand et al. (2019). Normative modeling in neuroimaging. *Nature Protocols*.

## Citation

If you use CorticalFields in your research, please cite:

```bibtex
@software{corticalfields2026,
  author = {Debona, R.},
  title = {CorticalFields: Geodesic-aware GP normative modeling and atlas-free cortical asymmetry analysis},
  year = {2026},
  url = {https://github.com/rdneuro/corticalfields},
}
```

## License

MIT
