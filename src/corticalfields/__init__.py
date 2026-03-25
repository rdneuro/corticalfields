"""
CorticalFields — Geodesic-aware GP normative modeling on cortical surfaces.

A library for computing information-theoretic surprise maps on the cortical
manifold using spectral Matérn Gaussian Processes, Heat Kernel Signatures,
and Laplace–Beltrami spectral analysis. Designed for structural MRI (T1w)
data in clinical neuroimaging, with emphasis on epilepsy (MTLE-HS).

GPU acceleration
----------------
CorticalFields supports three compute backends:
  - ``"scipy"`` (default) — CPU-only, most robust
  - ``"cupy"``  — GPU via NVIDIA CUDA (3–10× eigensolver speedup)
  - ``"torch"`` — GPU via PyTorch CUDA

Use ``corticalfields.backends.available_backends()`` to check what's
available on your system. All compute-intensive functions accept a
``backend=`` parameter.

Bayesian analysis
-----------------
The ``bayesian`` submodule provides 10 reusable model classes for
neuroimaging research (horseshoe, R2-D2, BEST, hierarchical, mediation,
classification, change-point, DAG). All models support 4 sampler
backends: pymc, nutpie, numpyro, blackjax.  The ``bayes_viz`` submodule
provides 20 publication-quality plotting functions.

Core pipeline:
    1. Load FreeSurfer surfaces and morphometric overlays
    2. Compute Laplace–Beltrami eigenpairs on the cortical mesh
    3. Extract spectral shape descriptors (HKS, WKS, GPS)
    4. Build spectral Matérn GP kernels on the manifold
    5. Fit normative models on a reference cohort
    6. Generate vertex-wise surprise / anomaly maps for patients

Modules
-------
surface     : Surface I/O — FreeSurfer, GIfTI, mesh utilities
spectral    : Laplace–Beltrami decomposition, HKS, WKS, GPS
backends    : GPU backend abstraction (CuPy, PyTorch, SciPy)
kernels     : Spectral Matérn kernels for GPyTorch
normative   : GP-based normative modeling pipeline
surprise    : Information-theoretic anomaly scoring
features    : Morphometric feature extraction from FreeSurfer
graphs      : Cortical similarity network construction
bayesian    : Bayesian statistical analysis (PyMC, ArviZ, PreliZ)
bayes_viz   : Publication-quality Bayesian visualization (20 functions)
viz         : Publication-quality surface visualization
"""

__version__ = "0.1.3"
__author__ = "Debpna, R. (rdneuro)"

# ── Lazy imports ────────────────────────────────────────────────────────
# Heavy dependencies (torch, gpytorch, cupy, pymc, arviz) are only loaded
# when the modules that need them are first accessed. This keeps
# `import corticalfields` fast and memory-light for submodule-level usage.


def __getattr__(name: str):
    """Lazy attribute loader — imports submodules on first access."""
    _MAP = {
        # surface.py (lightweight — numpy/nibabel only)
        "CorticalSurface": ("corticalfields.surface", "CorticalSurface"),
        "load_freesurfer_surface": ("corticalfields.surface", "load_freesurfer_surface"),
        # spectral.py (lightweight core — numpy/scipy; GPU via backends)
        "LaplaceBeltrami": ("corticalfields.spectral", "LaplaceBeltrami"),
        "compute_eigenpairs": ("corticalfields.spectral", "compute_eigenpairs"),
        "heat_kernel_signature": ("corticalfields.spectral", "heat_kernel_signature"),
        "wave_kernel_signature": ("corticalfields.spectral", "wave_kernel_signature"),
        "global_point_signature": ("corticalfields.spectral", "global_point_signature"),
        "spectral_feature_matrix": ("corticalfields.spectral", "spectral_feature_matrix"),
        # backends.py (lightweight detection; GPU libs imported lazily)
        "available_backends": ("corticalfields.backends", "available_backends"),
        "available_laplacian_backends": ("corticalfields.backends", "available_laplacian_backends"),
        "resolve_backend": ("corticalfields.backends", "resolve_backend"),
        "compute_graph_metrics": ("corticalfields.backends", "compute_graph_metrics"),
        "vectorized_correlation_matrix": ("corticalfields.backends", "vectorized_correlation_matrix"),
        # kernels.py (heavy — torch + gpytorch)
        "SpectralMaternKernel": ("corticalfields.kernels", "SpectralMaternKernel"),
        # normative.py (heavy — torch + gpytorch)
        "CorticalNormativeModel": ("corticalfields.normative", "CorticalNormativeModel"),
        # surprise.py (lightweight — numpy/scipy only)
        "SurpriseMap": ("corticalfields.surprise", "SurpriseMap"),
        "compute_surprise": ("corticalfields.surprise", "compute_surprise"),
        # features.py (lightweight)
        "MorphometricProfile": ("corticalfields.features", "MorphometricProfile"),
        # bayesian.py (heavy — pymc + arviz; lazy-loaded)
        "SamplerConfig": ("corticalfields.bayesian", "SamplerConfig"),
        "FAST": ("corticalfields.bayesian", "FAST"),
        "PUBLICATION": ("corticalfields.bayesian", "PUBLICATION"),
        "HORSESHOE": ("corticalfields.bayesian", "HORSESHOE"),
        "HorseshoeRegression": ("corticalfields.bayesian", "HorseshoeRegression"),
        "R2D2Regression": ("corticalfields.bayesian", "R2D2Regression"),
        "BayesianRidge": ("corticalfields.bayesian", "BayesianRidge"),
        "BayesianGroupComparison": ("corticalfields.bayesian", "BayesianGroupComparison"),
        "BayesianCorrelation": ("corticalfields.bayesian", "BayesianCorrelation"),
        "BayesianMediation": ("corticalfields.bayesian", "BayesianMediation"),
        "HierarchicalRegression": ("corticalfields.bayesian", "HierarchicalRegression"),
        "BayesianLogistic": ("corticalfields.bayesian", "BayesianLogistic"),
        "BayesianChangePoint": ("corticalfields.bayesian", "BayesianChangePoint"),
        "BayesianDAG": ("corticalfields.bayesian", "BayesianDAG"),
        "compute_diagnostics": ("corticalfields.bayesian", "compute_diagnostics"),
        "model_comparison": ("corticalfields.bayesian", "model_comparison"),
        "bayesian_r2": ("corticalfields.bayesian", "bayesian_r2"),
        "probability_of_direction": ("corticalfields.bayesian", "probability_of_direction"),
        "rope_percentage": ("corticalfields.bayesian", "rope_percentage"),
        "savage_dickey_bf": ("corticalfields.bayesian", "savage_dickey_bf"),
        "shrinkage_metrics": ("corticalfields.bayesian", "shrinkage_metrics"),
        "to_latex_table": ("corticalfields.bayesian", "to_latex_table"),
        "elicit_prior": ("corticalfields.bayesian", "elicit_prior"),
        "enigma_informed_prior": ("corticalfields.bayesian", "enigma_informed_prior"),
        "ENIGMA_EFFECT_SIZES": ("corticalfields.bayesian", "ENIGMA_EFFECT_SIZES"),
    }
    if name in _MAP:
        module_path, attr = _MAP[name]
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, attr)
    raise AttributeError(f"module 'corticalfields' has no attribute {name!r}")


__all__ = [
    # Surface I/O
    "CorticalSurface", "load_freesurfer_surface",
    # Spectral analysis (with GPU support)
    "LaplaceBeltrami", "compute_eigenpairs",
    "heat_kernel_signature", "wave_kernel_signature",
    "global_point_signature", "spectral_feature_matrix",
    # Backend management
    "available_backends", "available_laplacian_backends",
    "resolve_backend", "compute_graph_metrics",
    "vectorized_correlation_matrix",
    # GP kernels & normative modeling
    "SpectralMaternKernel", "CorticalNormativeModel",
    # Surprise maps
    "SurpriseMap", "compute_surprise",
    # Feature extraction
    "MorphometricProfile",
    # Bayesian analysis — sampler config & presets
    "SamplerConfig", "FAST", "PUBLICATION", "HORSESHOE",
    # Bayesian analysis — model classes
    "HorseshoeRegression", "R2D2Regression", "BayesianRidge",
    "BayesianGroupComparison", "BayesianCorrelation",
    "BayesianMediation", "HierarchicalRegression",
    "BayesianLogistic", "BayesianChangePoint", "BayesianDAG",
    # Bayesian analysis — diagnostics & metrics
    "compute_diagnostics", "model_comparison", "bayesian_r2",
    "probability_of_direction", "rope_percentage",
    "savage_dickey_bf", "shrinkage_metrics", "to_latex_table",
    # Bayesian analysis — prior elicitation
    "elicit_prior", "enigma_informed_prior", "ENIGMA_EFFECT_SIZES",
]
