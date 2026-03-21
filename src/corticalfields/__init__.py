"""
CorticalFields — Geodesic-aware GP normative modeling on cortical surfaces.

A library for computing information-theoretic surprise maps on the cortical
manifold using spectral Matérn Gaussian Processes, Heat Kernel Signatures,
and Laplace–Beltrami spectral analysis. Designed for structural MRI (T1w)
data in clinical neuroimaging, with emphasis on epilepsy (MTLE-HS).

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
kernels     : Spectral Matérn kernels for GPyTorch
normative   : GP-based normative modeling pipeline
surprise    : Information-theoretic anomaly scoring
features    : Morphometric feature extraction from FreeSurfer
graphs      : Cortical similarity network construction
viz         : Publication-quality surface visualization
"""

__version__ = "0.1.0"
__author__ = "Velho Mago (rdneuro)"

# ── Lazy imports ────────────────────────────────────────────────────────
# Heavy dependencies (torch, gpytorch) are only loaded when the modules
# that need them are first accessed. This keeps `import corticalfields`
# fast and memory-light for submodule-level usage.


def __getattr__(name: str):
    """Lazy attribute loader — imports submodules on first access."""
    _MAP = {
        # surface.py (lightweight — numpy/nibabel only)
        "CorticalSurface": ("corticalfields.surface", "CorticalSurface"),
        "load_freesurfer_surface": ("corticalfields.surface", "load_freesurfer_surface"),
        # spectral.py (lightweight — numpy/scipy only)
        "LaplaceBeltrami": ("corticalfields.spectral", "LaplaceBeltrami"),
        "heat_kernel_signature": ("corticalfields.spectral", "heat_kernel_signature"),
        "wave_kernel_signature": ("corticalfields.spectral", "wave_kernel_signature"),
        "global_point_signature": ("corticalfields.spectral", "global_point_signature"),
        # kernels.py (heavy — torch + gpytorch)
        "SpectralMaternKernel": ("corticalfields.kernels", "SpectralMaternKernel"),
        # normative.py (heavy — torch + gpytorch)
        "CorticalNormativeModel": ("corticalfields.normative", "CorticalNormativeModel"),
        # surprise.py (lightweight — numpy/scipy only)
        "SurpriseMap": ("corticalfields.surprise", "SurpriseMap"),
        "compute_surprise": ("corticalfields.surprise", "compute_surprise"),
        # features.py (lightweight)
        "MorphometricProfile": ("corticalfields.features", "MorphometricProfile"),
    }
    if name in _MAP:
        module_path, attr = _MAP[name]
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, attr)
    raise AttributeError(f"module 'corticalfields' has no attribute {name!r}")


__all__ = [
    "CorticalSurface", "load_freesurfer_surface",
    "LaplaceBeltrami", "heat_kernel_signature",
    "wave_kernel_signature", "global_point_signature",
    "SpectralMaternKernel",
    "CorticalNormativeModel",
    "SurpriseMap", "compute_surprise",
    "MorphometricProfile",
]
