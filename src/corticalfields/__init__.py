"""
CorticalFields — Spectral shape analysis on cortical and subcortical surfaces.

A library for computing spectral shape descriptors (HKS, WKS, GPS),
functional maps, optimal-transport distances, and information-theoretic
surprise maps on brain surface meshes.  Designed for structural MRI (T1w)
data in clinical neuroimaging, with emphasis on epilepsy (MTLE-HS).

Core pipeline:
    1. Load FreeSurfer surfaces / subcortical volumes / HippUnfold meshes
    2. Compute Laplace–Beltrami eigenpairs on the surface mesh
    3. Extract spectral shape descriptors (HKS, WKS, GPS)
    4. Build spectral Matérn GP kernels on the manifold
    5. Fit normative models on a reference cohort
    6. Generate vertex-wise surprise / anomaly maps for patients

Modules
-------
surface        : Surface I/O — FreeSurfer, GIfTI, mesh utilities
subcortical    : Subcortical surface extraction + shape analysis + spectral
                 fingerprinting (ShapeDNA, BrainPrint, curvatures, global
                 descriptors, point cloud features, topology, asymmetry,
                 normative z-scoring, Wasserstein distances)
hippocampus    : Hippocampal surface analysis — HippUnfold I/O, subfield
                 analysis, AP/PD axis profiling, gradients, MTLE-HS metrics,
                 texture sampling, vertex-wise GLMs, TFCE
viz_subcortical: Publication-quality visualization for subcortical +
                 hippocampal structures (PyVista 3D, multi-view, fold/unfold,
                 comparison panels, z-score maps, profile plots, bar charts)
spectral       : Laplace–Beltrami decomposition, HKS, WKS, GPS
kernels        : Spectral Matérn kernels for GPyTorch
normative      : GP-based normative modeling pipeline
surprise       : Information-theoretic anomaly scoring
features       : Morphometric feature extraction from FreeSurfer
graphs         : Cortical similarity network construction
viz            : Publication-quality surface visualization (cortical)
brainplots     : Publication-grade brain plots (surfaces, graphs, matrices)
datasets       : Toy dataset download from Zenodo
"""

__version__ = "0.2.0"
__author__ = "rdneuro"


def __getattr__(name: str):
    """Lazy attribute loader — imports submodules on first access."""
    _MAP = {
        # surface.py
        "CorticalSurface": ("corticalfields.surface", "CorticalSurface"),
        "load_freesurfer_surface": ("corticalfields.surface", "load_freesurfer_surface"),
        # spectral.py
        "LaplaceBeltrami": ("corticalfields.spectral", "LaplaceBeltrami"),
        "heat_kernel_signature": ("corticalfields.spectral", "heat_kernel_signature"),
        "wave_kernel_signature": ("corticalfields.spectral", "wave_kernel_signature"),
        "global_point_signature": ("corticalfields.spectral", "global_point_signature"),
        # kernels.py
        "SpectralMaternKernel": ("corticalfields.kernels", "SpectralMaternKernel"),
        # normative.py
        "CorticalNormativeModel": ("corticalfields.normative", "CorticalNormativeModel"),
        # surprise.py
        "SurpriseMap": ("corticalfields.surprise", "SurpriseMap"),
        "compute_surprise": ("corticalfields.surprise", "compute_surprise"),
        # features.py
        "MorphometricProfile": ("corticalfields.features", "MorphometricProfile"),
        # pointcloud.py
        "from_t1w": ("corticalfields.pointcloud", "from_t1w"),
        "T1wExtractionResult": ("corticalfields.pointcloud", "T1wExtractionResult"),
        "compute_mesh_laplacian": ("corticalfields.pointcloud", "compute_mesh_laplacian"),
        "compute_mesh_eigenpairs": ("corticalfields.pointcloud", "compute_mesh_eigenpairs"),
        # eda_qc.py
        "run_clinical_eda": ("corticalfields.eda_qc", "run_clinical_eda"),
        "run_spectral_eda": ("corticalfields.eda_qc", "run_spectral_eda"),
        "detect_clinical_outliers": ("corticalfields.eda_qc", "detect_clinical_outliers"),
        "mcd_mahalanobis_outliers": ("corticalfields.eda_qc", "mcd_mahalanobis_outliers"),
        "distance_matrix_outliers": ("corticalfields.eda_qc", "distance_matrix_outliers"),
        "generate_midthickness": ("corticalfields.eda_qc", "generate_midthickness"),
        "QCReport": ("corticalfields.eda_qc", "QCReport"),
        "EDAResult": ("corticalfields.eda_qc", "EDAResult"),
        # ══════════════════════════════════════════════════════════════
        # subcortical.py (enhanced v0.2.0)
        # ══════════════════════════════════════════════════════════════
        "SubcorticalSurface": ("corticalfields.subcortical", "SubcorticalSurface"),
        "load_subcortical_surface": ("corticalfields.subcortical", "load_subcortical_surface"),
        "load_subcortical_from_nifti": ("corticalfields.subcortical", "load_subcortical_from_nifti"),
        "load_all_subcortical": ("corticalfields.subcortical", "load_all_subcortical"),
        "subcortical_spectral_analysis": ("corticalfields.subcortical", "subcortical_spectral_analysis"),
        "shapedna_distance": ("corticalfields.subcortical", "shapedna_distance"),
        "wasserstein_shape_distance": ("corticalfields.subcortical", "wasserstein_shape_distance"),
        "brainprint_distance": ("corticalfields.subcortical", "brainprint_distance"),
        "batch_shape_descriptors": ("corticalfields.subcortical", "batch_shape_descriptors"),
        "batch_shapedna": ("corticalfields.subcortical", "batch_shapedna"),
        "pairwise_shapedna_distance_matrix": ("corticalfields.subcortical", "pairwise_shapedna_distance_matrix"),
        "FS_ASEG_LABELS": ("corticalfields.subcortical", "FS_ASEG_LABELS"),
        "FS_THALAMIC_NUCLEI": ("corticalfields.subcortical", "FS_THALAMIC_NUCLEI"),
        # ══════════════════════════════════════════════════════════════
        # hippocampus.py (NEW v0.2.0)
        # ══════════════════════════════════════════════════════════════
        "HippocampalSurface": ("corticalfields.hippocampus", "HippocampalSurface"),
        "load_hippocampal_surface": ("corticalfields.hippocampus", "load_hippocampal_surface"),
        "hippocampal_asymmetry_report": ("corticalfields.hippocampus", "hippocampal_asymmetry_report"),
        "hippocampal_spectral_analysis": ("corticalfields.hippocampus", "hippocampal_spectral_analysis"),
        "HIPPUNFOLD_SUBFIELDS": ("corticalfields.hippocampus", "HIPPUNFOLD_SUBFIELDS"),
        "ILAE_HS_TYPES": ("corticalfields.hippocampus", "ILAE_HS_TYPES"),
        # ══════════════════════════════════════════════════════════════
        # viz_subcortical.py (NEW v0.2.0)
        # ══════════════════════════════════════════════════════════════
        "plot_subcortical_multiview": ("corticalfields.viz_subcortical", "plot_subcortical_multiview"),
        "plot_subcortical_composite": ("corticalfields.viz_subcortical", "plot_subcortical_composite"),
        "plot_hippocampal_foldunfold": ("corticalfields.viz_subcortical", "plot_hippocampal_foldunfold"),
        "plot_hippocampal_comparison": ("corticalfields.viz_subcortical", "plot_hippocampal_comparison"),
        "plot_subfield_metrics": ("corticalfields.viz_subcortical", "plot_subfield_metrics"),
        "plot_subfield_asymmetry": ("corticalfields.viz_subcortical", "plot_subfield_asymmetry"),
        "plot_ap_profile": ("corticalfields.viz_subcortical", "plot_ap_profile"),
        "plot_zscore_surface": ("corticalfields.viz_subcortical", "plot_zscore_surface"),
        "plot_shapedna_comparison": ("corticalfields.viz_subcortical", "plot_shapedna_comparison"),
        "plot_point_cloud_3d": ("corticalfields.viz_subcortical", "plot_point_cloud_3d"),
        # ══════════════════════════════════════════════════════════════
        # utils.py
        # ══════════════════════════════════════════════════════════════
        "estimate_n_eigenpairs": ("corticalfields.utils", "estimate_n_eigenpairs"),
        "gc_gpu": ("corticalfields.utils", "gc_gpu"),
        "vram_report": ("corticalfields.utils", "vram_report"),
        "vram_guard": ("corticalfields.utils", "vram_guard"),
        # brainplots.py
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
        "plot_brain_scatter": ("corticalfields.viz", "plot_brain_scatter"),
        "plot_brain_views": ("corticalfields.brainplots", "plot_brain_views"),
        # datasets.py
        "fetch_toy_dataset": ("corticalfields.datasets", "fetch_toy_dataset"),
        "clear_toy_dataset": ("corticalfields.datasets", "clear_toy_dataset"),
        "load_example_surface": ("corticalfields.datasets", "load_example_surface"),
        "ToyDataset": ("corticalfields.datasets", "ToyDataset"),
    }
    if name in _MAP:
        module_path, attr = _MAP[name]
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, attr)
    raise AttributeError(f"module 'corticalfields' has no attribute {name!r}")


__all__ = [
    # Surface
    "CorticalSurface", "load_freesurfer_surface",
    # Spectral
    "LaplaceBeltrami", "heat_kernel_signature",
    "wave_kernel_signature", "global_point_signature",
    # Kernels / Normative / Surprise
    "SpectralMaternKernel", "CorticalNormativeModel",
    "SurpriseMap", "compute_surprise", "MorphometricProfile",
    # Pointcloud
    "from_t1w", "T1wExtractionResult",
    "compute_mesh_laplacian", "compute_mesh_eigenpairs",
    # EDA/QC
    "run_clinical_eda", "run_spectral_eda",
    "detect_clinical_outliers", "mcd_mahalanobis_outliers",
    "distance_matrix_outliers", "generate_midthickness",
    "QCReport", "EDAResult",
    # ─── Subcortical (enhanced v0.2.0) ───
    "SubcorticalSurface", "load_subcortical_surface",
    "load_subcortical_from_nifti", "load_all_subcortical",
    "subcortical_spectral_analysis",
    "shapedna_distance", "wasserstein_shape_distance", "brainprint_distance",
    "batch_shape_descriptors", "batch_shapedna",
    "pairwise_shapedna_distance_matrix",
    "FS_ASEG_LABELS", "FS_THALAMIC_NUCLEI",
    # ─── Hippocampus (NEW v0.2.0) ───
    "HippocampalSurface", "load_hippocampal_surface",
    "hippocampal_asymmetry_report", "hippocampal_spectral_analysis",
    "HIPPUNFOLD_SUBFIELDS", "ILAE_HS_TYPES",
    # ─── Visualization subcortical + hippocampal (NEW v0.2.0) ───
    "plot_subcortical_multiview", "plot_subcortical_composite",
    "plot_hippocampal_foldunfold", "plot_hippocampal_comparison",
    "plot_subfield_metrics", "plot_subfield_asymmetry",
    "plot_ap_profile", "plot_zscore_surface",
    "plot_shapedna_comparison", "plot_point_cloud_3d",
    # Utils
    "estimate_n_eigenpairs", "gc_gpu", "vram_report", "vram_guard",
    # Brain visualization (existing)
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
    "plot_brain_scatter", "plot_brain_views",
    # Datasets
    "fetch_toy_dataset", "clear_toy_dataset",
    "load_example_surface", "ToyDataset",
]
