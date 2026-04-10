"""
CorticalFields — Spectral shape analysis on cortical and subcortical surfaces.

A library for computing spectral shape descriptors (HKS, WKS, GPS),
functional maps, optimal-transport distances, and information-theoretic
surprise maps on brain surface meshes.  Designed for structural MRI (T1w)
data in clinical neuroimaging, with emphasis on epilepsy (MTLE-HS).

Subpackages (v0.2.3)
---------------------
analysis       : Statistical analysis & modeling
    analysis.stats      — MCC, GLM, PERMANOVA, CCA/PLS, RSA, NBS,
                          structural covariance, allometry, harmonization,
                          laterality, conformal prediction, GPU bootstrap
    analysis.bayesian   — Horseshoe / R2D2 / ridge regression, group
                          comparison, mediation, hierarchical, DAG, priors
    analysis.normative  — GP-based normative modeling pipeline
    analysis.eda_qc     — Exploratory data analysis, outlier detection, QC

viz            : Publication-quality visualization
    viz.brainplots      — 4-view, comparison, matrices, radar, composite
    viz.bayes           — Posterior, forest, ridgeline, HDI+ROPE, shrinkage
    viz.viz             — Surface scalar maps, surprise maps, profiles
    viz.subcortical     — Subcortical 3D, fold/unfold, z-score, ShapeDNA

Modules (root level)
--------------------
surface, subcortical, hippocampus, spectral, kernels, surprise, features,
graphs, distance_stats, asymmetry, transport, functional_maps, datasets, utils
"""

__version__ = "0.2.3"
__author__ = "rdneuro"


def __getattr__(name: str):
    """Lazy attribute loader — routes to new subpackage locations."""
    import importlib

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
        # surprise.py
        "SurpriseMap": ("corticalfields.surprise", "SurpriseMap"),
        "compute_surprise": ("corticalfields.surprise", "compute_surprise"),
        # features.py
        "MorphometricProfile": ("corticalfields.features", "MorphometricProfile"),
        # pointcloud
        "from_t1w": ("corticalfields.pointcloud", "from_t1w"),
        "T1wExtractionResult": ("corticalfields.pointcloud", "T1wExtractionResult"),
        "compute_mesh_laplacian": ("corticalfields.pointcloud", "compute_mesh_laplacian"),
        "compute_mesh_eigenpairs": ("corticalfields.pointcloud", "compute_mesh_eigenpairs"),
        # batch processing (in spectral.py)
        "batch_compute_eigenpairs": ("corticalfields.spectral", "batch_compute_eigenpairs"),
        "SubjectMesh": ("corticalfields.spectral", "SubjectMesh"),
        "BatchResult": ("corticalfields.spectral", "BatchResult"),
        "load_cached_eigenpairs": ("corticalfields.spectral", "load_cached_eigenpairs"),
        "estimate_memory_per_subject": ("corticalfields.spectral", "estimate_memory_per_subject"),
        "compute_safe_parallelism": ("corticalfields.spectral", "compute_safe_parallelism"),
        "MemoryEstimate": ("corticalfields.spectral", "MemoryEstimate"),
        # ── analysis.normative (was: normative.py) ──────────────────────
        "CorticalNormativeModel": ("corticalfields.analysis.normative", "CorticalNormativeModel"),
        "NormativeResult": ("corticalfields.analysis.normative", "NormativeResult"),
        # ── analysis.eda_qc (was: eda_qc.py) ───────────────────────────
        "run_clinical_eda": ("corticalfields.analysis.eda_qc", "run_clinical_eda"),
        "run_spectral_eda": ("corticalfields.analysis.eda_qc", "run_spectral_eda"),
        "detect_clinical_outliers": ("corticalfields.analysis.eda_qc", "detect_clinical_outliers"),
        "mcd_mahalanobis_outliers": ("corticalfields.analysis.eda_qc", "mcd_mahalanobis_outliers"),
        "distance_matrix_outliers": ("corticalfields.analysis.eda_qc", "distance_matrix_outliers"),
        "generate_midthickness": ("corticalfields.analysis.eda_qc", "generate_midthickness"),
        "QCReport": ("corticalfields.analysis.eda_qc", "QCReport"),
        "EDAResult": ("corticalfields.analysis.eda_qc", "EDAResult"),
        # ── analysis.bayesian (was: bayesian.py) ────────────────────────
        "SamplerConfig": ("corticalfields.analysis.bayesian", "SamplerConfig"),
        "HorseshoeRegression": ("corticalfields.analysis.bayesian", "HorseshoeRegression"),
        "R2D2Regression": ("corticalfields.analysis.bayesian", "R2D2Regression"),
        "BayesianRidge": ("corticalfields.analysis.bayesian", "BayesianRidge"),
        "BayesianGroupComparison": ("corticalfields.analysis.bayesian", "BayesianGroupComparison"),
        "BayesianCorrelation": ("corticalfields.analysis.bayesian", "BayesianCorrelation"),
        "BayesianMediation": ("corticalfields.analysis.bayesian", "BayesianMediation"),
        "HierarchicalRegression": ("corticalfields.analysis.bayesian", "HierarchicalRegression"),
        "BayesianLogistic": ("corticalfields.analysis.bayesian", "BayesianLogistic"),
        "BayesianChangePoint": ("corticalfields.analysis.bayesian", "BayesianChangePoint"),
        "BayesianDAG": ("corticalfields.analysis.bayesian", "BayesianDAG"),
        "compute_diagnostics": ("corticalfields.analysis.bayesian", "compute_diagnostics"),
        "model_comparison": ("corticalfields.analysis.bayesian", "model_comparison"),
        "bayesian_r2": ("corticalfields.analysis.bayesian", "bayesian_r2"),
        "probability_of_direction": ("corticalfields.analysis.bayesian", "probability_of_direction"),
        "rope_percentage": ("corticalfields.analysis.bayesian", "rope_percentage"),
        "savage_dickey_bf": ("corticalfields.analysis.bayesian", "savage_dickey_bf"),
        "shrinkage_metrics": ("corticalfields.analysis.bayesian", "shrinkage_metrics"),
        "to_latex_table": ("corticalfields.analysis.bayesian", "to_latex_table"),
        "elicit_prior": ("corticalfields.analysis.bayesian", "elicit_prior"),
        "enigma_informed_prior": ("corticalfields.analysis.bayesian", "enigma_informed_prior"),
        # ── analysis.stats  ─────────────────────────────────
        "StatResult": ("corticalfields.analysis.stats", "StatResult"),
        "MultipleComparisonResult": ("corticalfields.analysis.stats", "MultipleComparisonResult"),
        "fdr_correction": ("corticalfields.analysis.stats", "fdr_correction"),
        "bonferroni_correction": ("corticalfields.analysis.stats", "bonferroni_correction"),
        "tfce_surface": ("corticalfields.analysis.stats", "tfce_surface"),
        "cluster_permutation_surface": ("corticalfields.analysis.stats", "cluster_permutation_surface"),
        "max_statistic_correction": ("corticalfields.analysis.stats", "max_statistic_correction"),
        "vertex_wise_glm": ("corticalfields.analysis.stats", "vertex_wise_glm"),
        "vertex_wise_ttest": ("corticalfields.analysis.stats", "vertex_wise_ttest"),
        "permanova": ("corticalfields.analysis.stats", "permanova"),
        "cca": ("corticalfields.analysis.stats", "cca"),
        "pls_regression": ("corticalfields.analysis.stats", "pls_regression"),
        "compute_rdm": ("corticalfields.analysis.stats", "compute_rdm"),
        "compare_rdms": ("corticalfields.analysis.stats", "compare_rdms"),
        "rsa_regression": ("corticalfields.analysis.stats", "rsa_regression"),
        "structural_covariance_network": ("corticalfields.analysis.stats", "structural_covariance_network"),
        "mind_network": ("corticalfields.analysis.stats", "mind_network"),
        "graphical_lasso_network": ("corticalfields.analysis.stats", "graphical_lasso_network"),
        "network_based_statistic": ("corticalfields.analysis.stats", "network_based_statistic"),
        "allometric_scaling": ("corticalfields.analysis.stats", "allometric_scaling"),
        "combat_harmonize": ("corticalfields.analysis.stats", "combat_harmonize"),
        "asymmetry_index": ("corticalfields.analysis.stats", "asymmetry_index"),
        "laterality_features": ("corticalfields.analysis.stats", "laterality_features"),
        "laterality_classifier": ("corticalfields.analysis.stats", "laterality_classifier"),
        "conformal_prediction_intervals": ("corticalfields.analysis.stats", "conformal_prediction_intervals"),
        "bootstrap_gpu": ("corticalfields.analysis.stats", "bootstrap_gpu"),
        "permutation_matrix_gpu": ("corticalfields.analysis.stats", "permutation_matrix_gpu"),
        # ── graphs.py  ──────────────────────────────────
        "GraphResult": ("corticalfields.graphs", "GraphResult"),
        "GraphMetrics": ("corticalfields.graphs", "GraphMetrics"),
        "morphometric_similarity_network": ("corticalfields.graphs", "morphometric_similarity_network"),
        "spectral_similarity_network": ("corticalfields.graphs", "spectral_similarity_network"),
        "mind_divergence_network": ("corticalfields.graphs", "mind_divergence_network"),
        "wasserstein_spectral_network": ("corticalfields.graphs", "wasserstein_spectral_network"),
        "multi_descriptor_network": ("corticalfields.graphs", "multi_descriptor_network"),
        "proportional_threshold": ("corticalfields.graphs", "proportional_threshold"),
        "omst_threshold": ("corticalfields.graphs", "omst_threshold"),
        "backbone_disparity_filter": ("corticalfields.graphs", "backbone_disparity_filter"),
        "apply_threshold": ("corticalfields.graphs", "apply_threshold"),
        "comprehensive_graph_metrics": ("corticalfields.graphs", "comprehensive_graph_metrics"),
        "community_detection": ("corticalfields.graphs", "community_detection"),
        "persistent_homology": ("corticalfields.graphs", "persistent_homology"),
        "nbs_morphometric": ("corticalfields.graphs", "nbs_morphometric"),
        "group_metric_comparison": ("corticalfields.graphs", "group_metric_comparison"),
        "to_pyg_data": ("corticalfields.graphs", "to_pyg_data"),
        "build_population_graph": ("corticalfields.graphs", "build_population_graph"),
        "BrainGraphGCN": ("corticalfields.graphs", "BrainGraphGCN"),
        "spectral_morphometric_pipeline": ("corticalfields.graphs", "spectral_morphometric_pipeline"),
        "YEO7_COLORS": ("corticalfields.graphs", "YEO7_COLORS"),
        # ── viz.graph_viz (NEW v0.2.3) ───────────────────────────────────
        "plot_glass_brain_connectome": ("corticalfields.viz.graph_viz", "plot_glass_brain_connectome"),
        "plot_adjacency_matrix": ("corticalfields.viz.graph_viz", "plot_adjacency_matrix"),
        "plot_edge_weight_distribution": ("corticalfields.viz.graph_viz", "plot_edge_weight_distribution"),
        "plot_laplacian_spectrum": ("corticalfields.viz.graph_viz", "plot_laplacian_spectrum"),
        "plot_graph_layout": ("corticalfields.viz.graph_viz", "plot_graph_layout"),
        "plot_rich_club_curve": ("corticalfields.viz.graph_viz", "plot_rich_club_curve"),
        "plot_nbs_result": ("corticalfields.viz.graph_viz", "plot_nbs_result"),
        "plot_persistence_diagram": ("corticalfields.viz.graph_viz", "plot_persistence_diagram"),
        "plot_metric_comparison": ("corticalfields.viz.graph_viz", "plot_metric_comparison"),
        "plot_small_world": ("corticalfields.viz.graph_viz", "plot_small_world"),
        "plot_surface_metric": ("corticalfields.viz.graph_viz", "plot_surface_metric"),
        "plot_graph_composite": ("corticalfields.viz.graph_viz", "plot_graph_composite"),
        "save_graph_figure": ("corticalfields.viz.graph_viz", "save_graph_figure"),
        # ── utils.py (progress bars) ─────────────────────────────────────
        "cf_progress": ("corticalfields.utils", "cf_progress"),
        # ── subcortical.py ──────────────────────────────────────────────
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
        # ── hippocampus.py ──────────────────────────────────────────────
        "HippocampalSurface": ("corticalfields.hippocampus", "HippocampalSurface"),
        "load_hippocampal_surface": ("corticalfields.hippocampus", "load_hippocampal_surface"),
        "hippocampal_asymmetry_report": ("corticalfields.hippocampus", "hippocampal_asymmetry_report"),
        "hippocampal_spectral_analysis": ("corticalfields.hippocampus", "hippocampal_spectral_analysis"),
        "HIPPUNFOLD_SUBFIELDS": ("corticalfields.hippocampus", "HIPPUNFOLD_SUBFIELDS"),
        "ILAE_HS_TYPES": ("corticalfields.hippocampus", "ILAE_HS_TYPES"),
        # ── viz.subcortical (was: viz_subcortical.py) ───────────────────
        "plot_subcortical_multiview": ("corticalfields.viz.subcortical", "plot_subcortical_multiview"),
        "plot_subcortical_composite": ("corticalfields.viz.subcortical", "plot_subcortical_composite"),
        "plot_hippocampal_foldunfold": ("corticalfields.viz.subcortical", "plot_hippocampal_foldunfold"),
        "plot_hippocampal_comparison": ("corticalfields.viz.subcortical", "plot_hippocampal_comparison"),
        "plot_subfield_metrics": ("corticalfields.viz.subcortical", "plot_subfield_metrics"),
        "plot_subfield_asymmetry": ("corticalfields.viz.subcortical", "plot_subfield_asymmetry"),
        "plot_ap_profile": ("corticalfields.viz.subcortical", "plot_ap_profile"),
        "plot_zscore_surface": ("corticalfields.viz.subcortical", "plot_zscore_surface"),
        "plot_shapedna_comparison": ("corticalfields.viz.subcortical", "plot_shapedna_comparison"),
        "plot_point_cloud_3d": ("corticalfields.viz.subcortical", "plot_point_cloud_3d"),
        # ── viz.brainplots (was: brainplots.py) ─────────────────────────
        "plot_surface_4view": ("corticalfields.viz.brainplots", "plot_surface_4view"),
        "plot_surface_comparison": ("corticalfields.viz.brainplots", "plot_surface_comparison"),
        "plot_surprise_brain": ("corticalfields.viz.brainplots", "plot_surprise_brain"),
        "plot_normative_result": ("corticalfields.viz.brainplots", "plot_normative_result"),
        "plot_hks_multiscale": ("corticalfields.viz.brainplots", "plot_hks_multiscale"),
        "plot_asymmetry_brain": ("corticalfields.viz.brainplots", "plot_asymmetry_brain"),
        "plot_connectivity_matrix": ("corticalfields.viz.brainplots", "plot_connectivity_matrix"),
        "plot_functional_map_matrix": ("corticalfields.viz.brainplots", "plot_functional_map_matrix"),
        "plot_distance_matrix": ("corticalfields.viz.brainplots", "plot_distance_matrix"),
        "plot_permutation_null": ("corticalfields.viz.brainplots", "plot_permutation_null"),
        "plot_eigenspectrum": ("corticalfields.viz.brainplots", "plot_eigenspectrum"),
        "plot_network_radar": ("corticalfields.viz.brainplots", "plot_network_radar"),
        "plot_network_anomaly_bars": ("corticalfields.viz.brainplots", "plot_network_anomaly_bars"),
        "plot_brain_connectome": ("corticalfields.viz.brainplots", "plot_brain_connectome"),
        "plot_network_graph": ("corticalfields.viz.brainplots", "plot_network_graph"),
        "plot_asymmetry_bands": ("corticalfields.viz.brainplots", "plot_asymmetry_bands"),
        "plot_krr_diagnostic": ("corticalfields.viz.brainplots", "plot_krr_diagnostic"),
        "plot_subcortical_3d": ("corticalfields.viz.brainplots", "plot_subcortical_3d"),
        "plot_composite_figure": ("corticalfields.viz.brainplots", "plot_composite_figure"),
        "save_figure": ("corticalfields.viz.brainplots", "save_figure"),
        "plot_brain_views": ("corticalfields.viz.brainplots", "plot_brain_views"),
        # ── viz.viz (was: viz.py) ───────────────────────────────────────
        "plot_brain_scatter": ("corticalfields.viz.viz", "plot_brain_scatter"),
        "plot_surface_scalar": ("corticalfields.viz.viz", "plot_surface_scalar"),
        # ── utils.py ────────────────────────────────────────────────────
        "estimate_n_eigenpairs": ("corticalfields.utils", "estimate_n_eigenpairs"),
        "gc_gpu": ("corticalfields.utils", "gc_gpu"),
        "vram_report": ("corticalfields.utils", "vram_report"),
        "vram_guard": ("corticalfields.utils", "vram_guard"),
        # ── datasets.py ─────────────────────────────────────────────────
        "fetch_toy_dataset": ("corticalfields.datasets", "fetch_toy_dataset"),
        "clear_toy_dataset": ("corticalfields.datasets", "clear_toy_dataset"),
        "load_example_surface": ("corticalfields.datasets", "load_example_surface"),
        "ToyDataset": ("corticalfields.datasets", "ToyDataset"),
    }

    if name in _MAP:
        module_path, attr = _MAP[name]
        mod = importlib.import_module(module_path)
        return getattr(mod, attr)

    # Subpackage access: corticalfields.analysis / corticalfields.viz
    _SUBPACKAGES = {"analysis", "viz"}
    if name in _SUBPACKAGES:
        return importlib.import_module(f"corticalfields.{name}")

    raise AttributeError(f"module 'corticalfields' has no attribute {name!r}")


__all__ = [
    "analysis", "viz",
    "CorticalSurface", "load_freesurfer_surface",
    "LaplaceBeltrami", "heat_kernel_signature",
    "wave_kernel_signature", "global_point_signature",
    "SpectralMaternKernel", "CorticalNormativeModel", "NormativeResult",
    "SurpriseMap", "compute_surprise", "MorphometricProfile",
    "from_t1w", "T1wExtractionResult",
    "compute_mesh_laplacian", "compute_mesh_eigenpairs",
    "run_clinical_eda", "run_spectral_eda",
    "detect_clinical_outliers", "mcd_mahalanobis_outliers",
    "distance_matrix_outliers", "generate_midthickness",
    "QCReport", "EDAResult",
    "SamplerConfig",
    "HorseshoeRegression", "R2D2Regression", "BayesianRidge",
    "BayesianGroupComparison", "BayesianCorrelation",
    "BayesianMediation", "HierarchicalRegression",
    "BayesianLogistic", "BayesianChangePoint", "BayesianDAG",
    "compute_diagnostics", "model_comparison", "bayesian_r2",
    "probability_of_direction", "rope_percentage", "savage_dickey_bf",
    "shrinkage_metrics", "to_latex_table",
    "elicit_prior", "enigma_informed_prior",
    "StatResult", "MultipleComparisonResult",
    "fdr_correction", "bonferroni_correction",
    "tfce_surface", "cluster_permutation_surface", "max_statistic_correction",
    "vertex_wise_glm", "vertex_wise_ttest",
    "permanova", "cca", "pls_regression",
    "compute_rdm", "compare_rdms", "rsa_regression",
    "structural_covariance_network", "mind_network",
    "graphical_lasso_network", "network_based_statistic",
    "allometric_scaling", "combat_harmonize",
    "asymmetry_index", "laterality_features", "laterality_classifier",
    "conformal_prediction_intervals", "bootstrap_gpu", "permutation_matrix_gpu",
    "SubcorticalSurface", "load_subcortical_surface",
    "load_subcortical_from_nifti", "load_all_subcortical",
    "subcortical_spectral_analysis",
    "shapedna_distance", "wasserstein_shape_distance", "brainprint_distance",
    "batch_shape_descriptors", "batch_shapedna",
    "pairwise_shapedna_distance_matrix",
    "FS_ASEG_LABELS", "FS_THALAMIC_NUCLEI",
    "HippocampalSurface", "load_hippocampal_surface",
    "hippocampal_asymmetry_report", "hippocampal_spectral_analysis",
    "HIPPUNFOLD_SUBFIELDS", "ILAE_HS_TYPES",
    "plot_subcortical_multiview", "plot_subcortical_composite",
    "plot_hippocampal_foldunfold", "plot_hippocampal_comparison",
    "plot_subfield_metrics", "plot_subfield_asymmetry",
    "plot_ap_profile", "plot_zscore_surface",
    "plot_shapedna_comparison", "plot_point_cloud_3d",
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
    "plot_brain_scatter", "plot_brain_views", "plot_surface_scalar",
    "estimate_n_eigenpairs", "gc_gpu", "vram_report", "vram_guard",
    "fetch_toy_dataset", "clear_toy_dataset",
    "load_example_surface", "ToyDataset",
    # ── graphs (v0.2.3) ──────────────────────────────────────────────
    "GraphResult", "GraphMetrics",
    "morphometric_similarity_network", "spectral_similarity_network",
    "mind_divergence_network", "wasserstein_spectral_network",
    "multi_descriptor_network",
    "proportional_threshold", "omst_threshold", "backbone_disparity_filter",
    "apply_threshold",
    "comprehensive_graph_metrics", "community_detection",
    "persistent_homology", "nbs_morphometric", "group_metric_comparison",
    "to_pyg_data", "build_population_graph", "BrainGraphGCN",
    "spectral_morphometric_pipeline", "YEO7_COLORS",
    # ── viz.graph_viz (v0.2.3) ────────────────────────────────────────
    "plot_glass_brain_connectome", "plot_adjacency_matrix",
    "plot_edge_weight_distribution", "plot_laplacian_spectrum",
    "plot_graph_layout", "plot_rich_club_curve", "plot_nbs_result",
    "plot_persistence_diagram", "plot_metric_comparison",
    "plot_small_world", "plot_surface_metric",
    "plot_graph_composite", "save_graph_figure",
    # ── utils (progress bars) ─────────────────────────────────────────
    "cf_progress",
]
