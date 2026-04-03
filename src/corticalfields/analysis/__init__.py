"""
corticalfields.analysis — Statistical analysis and modeling for neuroimaging.

Submodules
----------
stats      : Conventional, multivariate, and novel statistical methods —
             PERMANOVA, CCA/PLS, RSA, structural covariance networks,
             TFCE, conformal prediction, laterality, GPU bootstrap.
bayesian   : Bayesian regression (horseshoe, R2D2, ridge), group comparison,
             mediation, hierarchical models, DAGs, diagnostics, priors.
normative  : GP-based normative modeling pipeline (spectral Matérn kernels).
eda_qc     : Exploratory data analysis, outlier detection, spectral QC,
             Euler number checks, Weyl's law, cohort EDA reporting.
"""

def __getattr__(name: str):
    """Lazy loader — resolve public names from submodules on first access."""
    import importlib

    # ── stats.py ────────────────────────────────────────────────────────
    _STATS = {
        "StatResult", "MultipleComparisonResult",
        # §1 Multiple comparison correction
        "fdr_correction", "bonferroni_correction", "max_statistic_correction",
        "tfce_surface", "cluster_permutation_surface",
        # §2 Surface-based GLM
        "vertex_wise_glm", "vertex_wise_ttest",
        # §3 PERMANOVA
        "permanova",
        # §4 Multivariate association
        "cca", "pls_regression",
        # §5 RSA
        "compute_rdm", "compare_rdms", "rsa_regression",
        # §6 Structural covariance / networks
        "structural_covariance_network", "mind_network",
        "graphical_lasso_network", "network_based_statistic",
        # §7 Allometric scaling
        "allometric_scaling",
        # §8 Harmonization
        "combat_harmonize",
        # §9 Laterality
        "asymmetry_index", "laterality_features", "laterality_classifier",
        # §10 Conformal prediction
        "conformal_prediction_intervals",
        # §11 GPU utilities
        "bootstrap_gpu", "permutation_matrix_gpu",
    }

    # ── bayesian.py ─────────────────────────────────────────────────────
    _BAYESIAN = {
        "SamplerConfig",
        "HorseshoeRegression", "R2D2Regression", "BayesianRidge",
        "BayesianGroupComparison", "BayesianCorrelation",
        "BayesianMediation", "HierarchicalRegression",
        "BayesianLogistic", "BayesianChangePoint", "BayesianDAG",
        "compute_diagnostics", "model_comparison", "bayesian_r2",
        "probability_of_direction", "rope_percentage", "savage_dickey_bf",
        "shrinkage_metrics", "to_latex_table",
        "elicit_prior", "enigma_informed_prior",
    }

    # ── normative.py ────────────────────────────────────────────────────
    _NORMATIVE = {
        "CorticalNormativeModel", "NormativeResult",
    }

    # ── eda_qc.py ───────────────────────────────────────────────────────
    _EDA_QC = {
        "QCReport", "EDAResult",
        "run_clinical_eda", "run_spectral_eda",
        "detect_clinical_outliers", "mcd_mahalanobis_outliers",
        "distance_matrix_outliers", "generate_midthickness",
        "descriptive_statistics", "correlation_matrix",
        "mad_outliers", "iqr_outliers",
        "extract_euler_numbers", "euler_number_qc",
        "weyls_law_check", "spectral_qc_cohort",
        "pcoa_embedding",
        "plot_eda_clinical", "plot_correlation_heatmap",
        "plot_pcoa_embedding", "plot_weyl_law_cohort",
    }

    if name in _STATS:
        mod = importlib.import_module(".stats", package=__name__)
        return getattr(mod, name)
    if name in _BAYESIAN:
        mod = importlib.import_module(".bayesian", package=__name__)
        return getattr(mod, name)
    if name in _NORMATIVE:
        mod = importlib.import_module(".normative", package=__name__)
        return getattr(mod, name)
    if name in _EDA_QC:
        mod = importlib.import_module(".eda_qc", package=__name__)
        return getattr(mod, name)

    # Allow `from corticalfields.analysis import stats` etc.
    _MODULES = {"stats", "bayesian", "normative", "eda_qc"}
    if name in _MODULES:
        return importlib.import_module(f".{name}", package=__name__)

    raise AttributeError(
        f"module 'corticalfields.analysis' has no attribute {name!r}"
    )


__all__ = [
    # Submodules
    "stats", "bayesian", "normative", "eda_qc",
]
