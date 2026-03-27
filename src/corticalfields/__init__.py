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
brainplots  : Publication-grade brain plots (surfaces, graphs, matrices, composites)
"""

__version__ = "0.1.5"
__author__ = "rdneuro"

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
        # pointcloud.py (FreeSurfer-free cortical extraction)
        "from_t1w": ("corticalfields.pointcloud", "from_t1w"),
        "T1wExtractionResult": ("corticalfields.pointcloud", "T1wExtractionResult"),
        "compute_mesh_laplacian": ("corticalfields.pointcloud", "compute_mesh_laplacian"),
        "compute_mesh_eigenpairs": ("corticalfields.pointcloud", "compute_mesh_eigenpairs"),
        # brainplots.py (publication-grade visualization — pyvista + matplotlib)
        "plot_surface_4view": ("corticalfields.brainplots", "plot_surface_4view"),
        "plot_surface_comparison": ("corticalfields.brainplots", "plot_surface_comparison"),
        "plot_surprise_brain": ("corticalfields.brainplots", "plot_surprise_brain"),
        "plot_normative_result": ("corticalfields.brainplots", "plot_normative_result"),
        "plot_hks_multiscale": ("corticalfields.brainplots", "plot_hks_multiscale"),
        "plot_asymmetry_brain": ("corticalfields.brainplots", "plot_asymmetry_brain"),
        "plot_connectivity_matrix": ("corticalfields.brainplots", "plot_connectivity_matrix"),
        "plot_functional_map_matrix": ("corticalfields.brainplots", "plot_functional_map_matrix"),
        "plot_distance_matrix": ("corticalfields.brainplots", "plot_distance_matrix"),
        "plot_permutation_null": ("corticalfields.brainplots", "plot_permutation_null"),
        "plot_eigenspectrum": ("corticalfields.brainplots", "plot_eigenspectrum"),
        "plot_network_radar": ("corticalfields.brainplots", "plot_network_radar"),
        "plot_network_anomaly_bars": ("corticalfields.brainplots", "plot_network_anomaly_bars"),
        "plot_brain_connectome": ("corticalfields.brainplots", "plot_brain_connectome"),
        "plot_network_graph": ("corticalfields.brainplots", "plot_network_graph"),
        "plot_asymmetry_bands": ("corticalfields.brainplots", "plot_asymmetry_bands"),
        "plot_krr_diagnostic": ("corticalfields.brainplots", "plot_krr_diagnostic"),
        "plot_subcortical_3d": ("corticalfields.brainplots", "plot_subcortical_3d"),
        "plot_composite_figure": ("corticalfields.brainplots", "plot_composite_figure"),
        "save_figure": ("corticalfields.brainplots", "save_figure"),
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
    # FreeSurfer-free cortical extraction
    "from_t1w", "T1wExtractionResult",
    "compute_mesh_laplacian", "compute_mesh_eigenpairs",
    # Brain visualization (publication-grade)
    "plot_surface_4view", "plot_surface_comparison",
    "plot_surprise_brain", "plot_normative_result",
    "plot_hks_multiscale", "plot_asymmetry_brain",
    "plot_connectivity_matrix", "plot_functional_map_matrix",
    "plot_distance_matrix", "plot_permutation_null",
    "plot_eigenspectrum", "plot_network_radar",
    "plot_network_anomaly_bars", "plot_brain_connectome",
    "plot_network_graph", "plot_asymmetry_bands",
    "plot_krr_diagnostic", "plot_subcortical_3d",
    "plot_composite_figure", "save_figure",
]
