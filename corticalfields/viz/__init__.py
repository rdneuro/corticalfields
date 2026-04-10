"""
corticalfields.viz — Publication-quality neuroimaging visualization.

Submodules
----------
brainplots  : Surface 4-view renders, connectivity matrices, eigenspectra,
              network radars, composite figures (Nature/NeuroImage style).
bayes       : Bayesian posterior density, forest, ridgeline, HDI+ROPE,
              shrinkage, mediation diagrams, model comparison, PPC.
viz         : Lightweight cortical surface scalar maps, surprise maps,
              HKS multiscale, network anomaly profiles (matplotlib/PyVista).
subcortical : Subcortical + hippocampal 3D rendering, fold/unfold,
              subfield metrics, AP profiles, z-score surfaces, ShapeDNA.
"""

# ── Lazy import machinery ───────────────────────────────────────────────
# We use __getattr__ so that importing `corticalfields.viz` alone does NOT
# pull in heavy dependencies (PyVista, surfplot, brainspace) until needed.

def __getattr__(name: str):
    """Lazy loader — resolve public names from submodules on first access."""
    import importlib

    _SUBMODULE_MAP = {
        # viz.brainplots
        "plot_surface_4view": ".brainplots",
        "plot_surface_comparison": ".brainplots",
        "plot_surprise_brain": ".brainplots",
        "plot_normative_result": ".brainplots",
        "plot_hks_multiscale": ".brainplots",
        "plot_asymmetry_brain": ".brainplots",
        "plot_connectivity_matrix": ".brainplots",
        "plot_functional_map_matrix": ".brainplots",
        "plot_distance_matrix": ".brainplots",
        "plot_permutation_null": ".brainplots",
        "plot_eigenspectrum": ".brainplots",
        "plot_network_radar": ".brainplots",
        "plot_network_anomaly_bars": ".brainplots",
        "plot_brain_connectome": ".brainplots",
        "plot_network_graph": ".brainplots",
        "plot_asymmetry_bands": ".brainplots",
        "plot_krr_diagnostic": ".brainplots",
        "plot_subcortical_3d": ".brainplots",
        "plot_composite_figure": ".brainplots",
        "save_figure": ".brainplots",
        "plot_brain_views": ".brainplots",
        # viz.bayes
        "plot_posterior_density": ".bayes",
        "plot_forest": ".bayes",
        "plot_ridgeline": ".bayes",
        "plot_posterior_hdi_rope": ".bayes",
        "plot_prior_posterior_update": ".bayes",
        "plot_trace_rank": ".bayes",
        "plot_pair_divergences": ".bayes",
        "plot_energy": ".bayes",
        "plot_loo_pit": ".bayes",
        "plot_rhat_ess_panel": ".bayes",
        "plot_shrinkage": ".bayes",
        "plot_coefficient_path": ".bayes",
        "plot_mediation_diagram": ".bayes",
        "plot_group_comparison": ".bayes",
        "plot_hierarchical_caterpillar": ".bayes",
        "plot_model_comparison": ".bayes",
        "plot_posterior_predictive_check": ".bayes",
        "plot_sensitivity": ".bayes",
        "plot_correlation_matrix_posterior": ".bayes",
        "plot_brain_posterior_map": ".bayes",
        # viz.viz
        "plot_surface_scalar": ".viz",
        "plot_surprise_map": ".viz",
        "plot_network_anomaly_profile": ".viz",
        "plot_brain_scatter": ".viz",
        # viz.subcortical
        "plot_subcortical_multiview": ".subcortical",
        "plot_subcortical_composite": ".subcortical",
        "plot_hippocampal_foldunfold": ".subcortical",
        "plot_hippocampal_comparison": ".subcortical",
        "plot_subfield_metrics": ".subcortical",
        "plot_subfield_asymmetry": ".subcortical",
        "plot_ap_profile": ".subcortical",
        "plot_zscore_surface": ".subcortical",
        "plot_shapedna_comparison": ".subcortical",
        "plot_point_cloud_3d": ".subcortical",
        # viz.graph_viz
        "plot_glass_brain_connectome": ".graph_viz",
        "plot_connectivity_circle": ".graph_viz",
        "plot_adjacency_matrix": ".graph_viz",
        "plot_edge_weight_distribution": ".graph_viz",
        "plot_laplacian_spectrum": ".graph_viz",
        "plot_graph_layout": ".graph_viz",
        "plot_rich_club_curve": ".graph_viz",
        "plot_nbs_result": ".graph_viz",
        "plot_persistence_diagram": ".graph_viz",
        "plot_metric_comparison": ".graph_viz",
        "plot_small_world": ".graph_viz",
        "plot_surface_metric": ".graph_viz",
        "plot_graph_composite": ".graph_viz",
        "save_graph_figure": ".graph_viz",
        "PUBLICATION_RC": ".graph_viz",
        "YEO7_COLORS": ".graph_viz",
        "TOL_BRIGHT": ".graph_viz",
        "OKABE_ITO": ".graph_viz",
    }

    if name in _SUBMODULE_MAP:
        mod = importlib.import_module(_SUBMODULE_MAP[name], package=__name__)
        return getattr(mod, name)

    # Allow `from corticalfields.viz import brainplots` etc.
    _MODULES = {"brainplots", "bayes", "viz", "subcortical", "graph_viz"}
    if name in _MODULES:
        return importlib.import_module(f".{name}", package=__name__)

    raise AttributeError(f"module 'corticalfields.viz' has no attribute {name!r}")


__all__ = [
    # Submodules
    "brainplots", "bayes", "viz", "subcortical",
    # brainplots
    "plot_surface_4view", "plot_surface_comparison",
    "plot_surprise_brain", "plot_normative_result",
    "plot_hks_multiscale", "plot_asymmetry_brain",
    "plot_connectivity_matrix", "plot_functional_map_matrix",
    "plot_distance_matrix", "plot_permutation_null",
    "plot_eigenspectrum", "plot_network_radar",
    "plot_network_anomaly_bars", "plot_brain_connectome",
    "plot_network_graph", "plot_asymmetry_bands",
    "plot_krr_diagnostic", "plot_subcortical_3d",
    "plot_composite_figure", "save_figure", "plot_brain_views",
    # bayes
    "plot_posterior_density", "plot_forest", "plot_ridgeline",
    "plot_posterior_hdi_rope", "plot_prior_posterior_update",
    "plot_trace_rank", "plot_pair_divergences", "plot_energy",
    "plot_loo_pit", "plot_rhat_ess_panel", "plot_shrinkage",
    "plot_coefficient_path", "plot_mediation_diagram",
    "plot_group_comparison", "plot_hierarchical_caterpillar",
    "plot_model_comparison", "plot_posterior_predictive_check",
    "plot_sensitivity", "plot_correlation_matrix_posterior",
    "plot_brain_posterior_map",
    # viz
    "plot_surface_scalar", "plot_surprise_map",
    "plot_network_anomaly_profile", "plot_brain_scatter",
    # subcortical
    "plot_subcortical_multiview", "plot_subcortical_composite",
    "plot_hippocampal_foldunfold", "plot_hippocampal_comparison",
    "plot_subfield_metrics", "plot_subfield_asymmetry",
    "plot_ap_profile", "plot_zscore_surface",
    "plot_shapedna_comparison", "plot_point_cloud_3d",
]
