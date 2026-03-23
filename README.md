# CorticalFields

**Geodesic-aware Gaussian Process normative modeling on cortical surfaces with spectral shape descriptors and information-theoretic surprise maps.**

CorticalFields extracts rich geometric information from structural MRI (T1w) data — without needing fMRI or dMRI — by treating the cortical surface as a Riemannian manifold and applying principled probabilistic modeling. It was designed for epilepsy research (MTLE-HS) but is applicable to any structural neuroimaging study.

## What it does

Given FreeSurfer-processed T1w data, CorticalFields:

1. **Decomposes the cortical geometry** via the Laplace–Beltrami operator, extracting the intrinsic spectral structure of the cortical mesh.
2. **Computes spectral shape descriptors** (Heat Kernel Signature, Wave Kernel Signature, Global Point Signature) that capture multi-scale geometric information at every vertex — from local curvature to global lobe membership.
3. **Builds a spectral Matérn GP** that respects the cortical manifold's geometry (not Euclidean space), producing a proper probabilistic model of "normal" brain morphology.
4. **Generates vertex-wise surprise maps** — information-theoretic anomaly scores that quantify how unexpected each vertex's morphometry is, given the normative model and the surrounding cortical geometry.
5. **Aggregates anomalies by functional network** (Yeo-7, Schaefer-200, etc.), enabling statements like *"this patient has anomalous DMN structure"* from T1w data alone.

## Key innovations

- **Geodesic-aware GP kernel**: Uses the spectral Matérn kernel (Borovitskiy et al., NeurIPS 2020) — the only mathematically correct way to define Matérn GPs on curved manifolds. Naïve geodesic distance substitution does NOT yield a valid kernel.
- **Surprise maps**: Vertex-wise negative log-predictive density from a calibrated probabilistic model — theoretically optimal anomaly scoring that goes beyond z-scores.
- **Spectral shape descriptors for neuroimaging**: Brings HKS/WKS from computer graphics into clinical brain imaging for the first time.
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
```

### Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0 (with CUDA for GPU acceleration)
- FreeSurfer (for surface reconstruction via `recon-all`)
- Core: numpy, scipy, nibabel, gpytorch, matplotlib, pyvista
- Optional: robust-laplacian, geometric-kernels, pymc, giotto-tda

## Quick start

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

## Practical GPU-Backend Use
The library offers 3 different backend for computational demanding operations. "Scipy" equation solving offers practical, however slower, CPU-based work. "Cupy" or "Torch" backends speeds up the operations through GPU mathemathical operations improved performance.

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
├── surface.py      # Surface I/O (FreeSurfer, GIfTI, mesh utilities)
├── spectral.py     # Laplace–Beltrami decomposition, HKS, WKS, GPS
├── kernels.py      # Spectral Matérn kernels for GPyTorch
├── normative.py    # GP-based normative modeling pipeline
├── surprise.py     # Information-theoretic anomaly scoring
├── features.py     # Morphometric feature extraction
├── graphs.py       # Cortical similarity network construction
├── viz.py          # Publication-quality surface visualization
└── utils.py        # Helpers (GPU detection, parcellation, timing)
```

## Mathematical foundations

The core of CorticalFields rests on three mathematical pillars:

**1. Laplace–Beltrami spectral decomposition.** On a compact Riemannian manifold (Σ, g), the LB operator Δ_g has a discrete spectrum 0 = λ₀ ≤ λ₁ ≤ λ₂ ≤ … with orthonormal eigenfunctions φᵢ. These encode the intrinsic geometry of the surface at all scales — low eigenvalues capture global shape, high eigenvalues capture fine-grained curvature.

**2. Spectral Matérn kernel.** The Matérn kernel on the manifold is defined via the spectral density: k(x,y) = σ² Σᵢ S_ν(λᵢ) φᵢ(x) φᵢ(y), where S_ν(λ) = (2ν/κ² + λ)^{−(ν+d/2)}. This yields a valid positive-definite kernel that respects the manifold geometry, unlike naïve geodesic-distance substitution.

**3. Information-theoretic surprise.** For a GP with posterior predictive p(y*|x,D) = N(μ(x), σ²(x)), the surprise at vertex x is S(x) = −log p(y_obs|x,D) = ½ log(2πσ²) + (y_obs − μ)²/(2σ²). This is the theoretically optimal anomaly score under the model.

## References

- Borovitskiy et al. (2020). Matérn Gaussian Processes on Riemannian Manifolds. NeurIPS.
- Sun, Ovsjanikov & Guibas (2009). Heat Kernel Signature. SGP.
- Aubry, Schlickewei & Cremers (2011). Wave Kernel Signature. ICCV.
- Sharp & Crane (2020). A Laplacian for Nonmanifold Triangle Meshes. SGP.
- Seidlitz et al. (2018). Morphometric Similarity Networks. Neuron.
- Marquand et al. (2019). Normative modeling in neuroimaging. Nature Protocols.

## Citation

If you use CorticalFields in your research, please cite:

```bibtex
@software{corticalfields2026,
  author = {Debona, R.},
  title = {CorticalFields: Geodesic-aware GP normative modeling on cortical surfaces},
  year = {2026},
  url = {https://github.com/rdneuro/corticalfields},
}
```

## License

MIT
