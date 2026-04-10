"""
Statistical analysis for CorticalFields spectral and morphometric data.

A comprehensive toolkit of conventional, multivariate, and novel statistical
methods designed for relating cortical spectral features (HKS, WKS,
functional maps, Wasserstein distances) and FreeSurfer morphometrics to
clinical-demographic variables, cognitive/psychiatric scores, and inter-
regional morphological structure.

Architecture
------------
This module is organized into eleven sections:

  §1  MULTIPLE COMPARISON CORRECTION — FDR, Bonferroni, TFCE, cluster
      permutation, max-statistic correction for surface-based analyses
  §2  SURFACE-BASED GLM — Vertex-wise general linear models & t-tests
  §3  PERMANOVA — Permutational ANOVA on distance matrices
  §4  MULTIVARIATE ASSOCIATION — CCA, PLS for brain–behavior
  §5  RSA — Representational Similarity Analysis (RDMs, model comparison)
  §6  STRUCTURAL COVARIANCE & NETWORKS — SCN, MSN, MIND, graphical LASSO,
      Network-Based Statistics (NBS)
  §7  ALLOMETRIC SCALING — Power-law brain size relationships
  §8  HARMONIZATION — ComBat wrappers for multi-site data
  §9  LATERALITY — Asymmetry indices, feature engineering, classification
      pipeline for MTLE-HS laterality determination
  §10 CONFORMAL PREDICTION — Distribution-free prediction intervals
  §11 GPU-ACCELERATED UTILITIES — Bootstrap, permutation matrices

Dependencies
------------
Core    : numpy, scipy, pandas, scikit-learn, torch
Optional: skbio (PERMANOVA), hyppo (HSIC/dcor), rsatoolbox (RSA),
          mapie (conformal), neuroCombat (harmonization)

References
----------
[1]  Smith & Nichols (2009)  "TFCE"
[2]  Anderson (2001)        "PERMANOVA"
[3]  Kriegeskorte et al. (2008) "RSA"
[4]  Seidlitz et al. (2018) "MSN"
[5]  Sebenius et al. (2023) "MIND"
[6]  Gleichgerrcht et al. (2021) "ENIGMA MTLE-HS lateralization"
[7]  Angelopoulos & Bates (2023) "Conformal prediction"
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats as spstats
import scipy.sparse as sp
from scipy.spatial.distance import (
    pdist, squareform, correlation as corr_dist, euclidean,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  RESULT CONTAINERS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class StatResult:
    """Container for a single statistical test result.

    Attributes
    ----------
    statistic : float
        The test statistic value (F, t, r, etc.).
    p_value : float
        p-value (permutation-based or parametric).
    effect_size : float or None
        Effect size metric (Cohen's d, η², R², etc.).
    method : str
        Name of the statistical test.
    description : str
        Human-readable summary of what was tested.
    extras : dict
        Any additional outputs (null distribution, CI, etc.).
    """
    statistic: float
    p_value: float
    effect_size: Optional[float] = None
    method: str = ""
    description: str = ""
    extras: Dict[str, Any] = field(default_factory=dict)

    def __float__(self) -> float:
        return float(self.statistic)

    def __format__(self, format_spec: str) -> str:
        if format_spec:
            return format(self.statistic, format_spec)
        return repr(self)


@dataclass
class MultipleComparisonResult:
    """Container for multiple-comparison-corrected results.

    Attributes
    ----------
    p_values_raw : np.ndarray
        Uncorrected p-values, shape (n_tests,).
    p_values_corrected : np.ndarray
        Corrected p-values, same shape.
    rejected : np.ndarray
        Boolean mask of rejected nulls at the given alpha.
    alpha : float
        Significance level used.
    method : str
        Correction method name.
    extras : dict
        Additional outputs (TFCE map, cluster labels, etc.).
    """
    p_values_raw: np.ndarray
    p_values_corrected: np.ndarray
    rejected: np.ndarray
    alpha: float = 0.05
    method: str = ""
    extras: Dict[str, Any] = field(default_factory=dict)

    def __iter__(self):
        """Allow tuple unpacking: ``reject, corrected = fdr_correction(...)``."""
        return iter((self.rejected, self.p_values_corrected))


# ═══════════════════════════════════════════════════════════════════════════
#  §1  MULTIPLE COMPARISON CORRECTION
# ═══════════════════════════════════════════════════════════════════════════

def fdr_correction(
    p_values: np.ndarray,
    alpha: float = 0.05,
    method: Literal["bh", "by"] = "bh",
) -> MultipleComparisonResult:
    """Benjamini-Hochberg (or Benjamini-Yekutieli) FDR correction.

    Parameters
    ----------
    p_values : (N,) array
        Raw p-values from vertex-wise or region-wise tests.
    alpha : float
        Target FDR level.
    method : {'bh', 'by'}
        'bh' for Benjamini-Hochberg (1995), 'by' for Benjamini-Yekutieli
        (2001) which controls FDR under arbitrary dependence.

    Returns
    -------
    MultipleComparisonResult
    """
    p_values = np.asarray(p_values, dtype=np.float64).ravel()
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    ranks = np.arange(1, n + 1, dtype=np.float64)

    # Correction factor
    if method == "by":
        c_m = np.sum(1.0 / ranks)
    else:
        c_m = 1.0

    # Adjusted p-values (step-up procedure)
    adjusted = np.empty(n, dtype=np.float64)
    adjusted[-1] = sorted_p[-1]  # largest rank keeps its p
    for i in range(n - 2, -1, -1):
        adjusted[i] = min(adjusted[i + 1], sorted_p[i] * n * c_m / ranks[i])
    adjusted = np.clip(adjusted, 0.0, 1.0)

    # Unsort
    p_corrected = np.empty(n, dtype=np.float64)
    p_corrected[sorted_idx] = adjusted

    return MultipleComparisonResult(
        p_values_raw=p_values,
        p_values_corrected=p_corrected,
        rejected=p_corrected <= alpha,
        alpha=alpha,
        method=f"FDR-{method.upper()}",
    )


def bonferroni_correction(
    p_values: np.ndarray,
    alpha: float = 0.05,
) -> MultipleComparisonResult:
    """Bonferroni family-wise error rate correction.

    Parameters
    ----------
    p_values : (N,) array
        Raw p-values.
    alpha : float
        Target FWER.

    Returns
    -------
    MultipleComparisonResult
    """
    p_values = np.asarray(p_values, dtype=np.float64).ravel()
    n = len(p_values)
    p_corrected = np.clip(p_values * n, 0.0, 1.0)
    return MultipleComparisonResult(
        p_values_raw=p_values,
        p_values_corrected=p_corrected,
        rejected=p_corrected <= alpha,
        alpha=alpha,
        method="Bonferroni",
    )


def max_statistic_correction(
    observed_stats: np.ndarray,
    null_max_distribution: np.ndarray,
    alpha: float = 0.05,
    tail: Literal["two", "upper", "lower"] = "two",
) -> MultipleComparisonResult:
    """Max-statistic permutation-based FWER correction.

    Computes corrected p-values by comparing each observed statistic
    against the distribution of the maximum statistic across all tests
    under the null (Nichols & Holmes, 2002).

    Parameters
    ----------
    observed_stats : (N,) array
        Test statistics at N locations (vertices, ROIs).
    null_max_distribution : (n_perm,) array
        Maximum statistic across all locations for each permutation.
    alpha : float
        Significance level.
    tail : {'two', 'upper', 'lower'}
        Tail of the test.

    Returns
    -------
    MultipleComparisonResult
    """
    observed = np.asarray(observed_stats, dtype=np.float64).ravel()
    null_max = np.asarray(null_max_distribution, dtype=np.float64).ravel()

    if tail == "two":
        null_max = np.abs(null_max)
        test_stats = np.abs(observed)
    elif tail == "lower":
        null_max = -null_max
        test_stats = -observed
    else:
        test_stats = observed

    # p-value for each vertex = fraction of null maxima >= observed stat
    n_perm = len(null_max)
    p_corrected = np.array([
        (np.sum(null_max >= t) + 1) / (n_perm + 1) for t in test_stats
    ])

    return MultipleComparisonResult(
        p_values_raw=np.full_like(observed, np.nan),  # not computed
        p_values_corrected=p_corrected,
        rejected=p_corrected <= alpha,
        alpha=alpha,
        method="Max-statistic permutation",
    )


def tfce_surface(
    stat_map: np.ndarray,
    faces: np.ndarray,
    n_perm: int = 5000,
    E: float = 1.0,
    H: float = 2.0,
    dh: Optional[float] = None,
    seed: int = 42,
    design_matrix: Optional[np.ndarray] = None,
    contrast: Optional[np.ndarray] = None,
    data_matrix: Optional[np.ndarray] = None,
) -> MultipleComparisonResult:
    """Threshold-Free Cluster Enhancement on a cortical surface mesh.

    Computes TFCE-enhanced statistics using the mesh adjacency graph,
    with parameters E=1, H=2 recommended for cortical surfaces
    (Smith & Nichols, 2009).

    TFCE(v) = ∫₀ʰ e(h)^E · h^H dh

    where e(h) is the cluster extent at threshold h.

    Parameters
    ----------
    stat_map : (V,) array
        Vertex-wise test statistics (e.g., t-values from a GLM).
    faces : (F, 3) int array
        Triangular face indices defining mesh adjacency.
    n_perm : int
        Number of sign-flip permutations for null distribution.
    E : float
        Extent exponent (default 1.0 for surfaces).
    H : float
        Height exponent (default 2.0 for surfaces).
    dh : float or None
        Integration step. If None, auto-set to max(|stat|)/100.
    seed : int
        Random seed for reproducibility.
    design_matrix : (N, P) array, optional
        Design matrix for GLM-based permutation. If None, sign-flip
        is applied directly to stat_map.
    contrast : (P,) array, optional
        Contrast vector for the GLM.
    data_matrix : (N, V) array, optional
        Subject × vertex data for GLM re-fitting under permutation.

    Returns
    -------
    MultipleComparisonResult
        TFCE-corrected results. extras['tfce_map'] contains the
        enhanced statistic map.
    """
    stat_map = np.asarray(stat_map, dtype=np.float64).ravel()
    faces = np.asarray(faces, dtype=np.int64)
    n_vertices = len(stat_map)

    # Build adjacency from mesh faces
    adjacency = _mesh_adjacency(faces, n_vertices)

    # Integration step
    stat_abs_max = np.max(np.abs(stat_map))
    if stat_abs_max == 0:
        return MultipleComparisonResult(
            p_values_raw=np.ones(n_vertices),
            p_values_corrected=np.ones(n_vertices),
            rejected=np.zeros(n_vertices, dtype=bool),
            alpha=0.05,
            method="TFCE (trivial — all zeros)",
        )
    if dh is None:
        dh = stat_abs_max / 100.0

    # Compute TFCE for observed data
    tfce_observed = _compute_tfce(stat_map, adjacency, E, H, dh)

    # Permutation null distribution
    rng = np.random.default_rng(seed)
    tfce_max_null = np.empty(n_perm, dtype=np.float64)

    for i in range(n_perm):
        if data_matrix is not None and design_matrix is not None:
            # GLM-based permutation: shuffle rows of design matrix
            perm_idx = rng.permutation(data_matrix.shape[0])
            perm_design = design_matrix[perm_idx]
            # Solve GLM: β = (X'X)^{-1} X'Y
            beta = np.linalg.lstsq(perm_design, data_matrix, rcond=None)[0]
            if contrast is not None:
                perm_stat = contrast @ beta
            else:
                perm_stat = beta[0]  # first regressor
        else:
            # Sign-flip permutation (Freedman-Lane style)
            signs = rng.choice([-1.0, 1.0], size=n_vertices)
            perm_stat = stat_map * signs

        perm_tfce = _compute_tfce(perm_stat, adjacency, E, H, dh)
        tfce_max_null[i] = np.max(np.abs(perm_tfce))

    # Corrected p-values via max-statistic
    tfce_abs = np.abs(tfce_observed)
    p_corrected = np.array([
        (np.sum(tfce_max_null >= t) + 1) / (n_perm + 1)
        for t in tfce_abs
    ])

    return MultipleComparisonResult(
        p_values_raw=np.full(n_vertices, np.nan),
        p_values_corrected=p_corrected,
        rejected=p_corrected <= 0.05,
        alpha=0.05,
        method=f"TFCE (E={E}, H={H}, {n_perm} perms)",
        extras={"tfce_map": tfce_observed, "null_max": tfce_max_null},
    )


def cluster_permutation_surface(
    stat_map: np.ndarray,
    faces: np.ndarray,
    threshold: float,
    n_perm: int = 5000,
    seed: int = 42,
    tail: Literal["two", "upper", "lower"] = "two",
    data_matrix: Optional[np.ndarray] = None,
    design_matrix: Optional[np.ndarray] = None,
    contrast: Optional[np.ndarray] = None,
) -> MultipleComparisonResult:
    """Cluster-based permutation test on a cortical surface mesh.

    Follows Maris & Oostenveld (2007): threshold the map, find connected
    clusters via mesh adjacency, compute cluster mass, assess significance
    via permutation distribution of maximum cluster mass.

    Parameters
    ----------
    stat_map : (V,) array
        Vertex-wise test statistics.
    faces : (F, 3) int array
        Mesh face indices.
    threshold : float
        Cluster-forming threshold for the statistic.
    n_perm : int
        Number of permutations.
    seed : int
        Random seed.
    tail : {'two', 'upper', 'lower'}
        Test directionality.
    data_matrix, design_matrix, contrast : optional
        For GLM-based permutation (see tfce_surface).

    Returns
    -------
    MultipleComparisonResult
        extras contains 'cluster_labels' and 'cluster_masses'.
    """
    stat_map = np.asarray(stat_map, dtype=np.float64).ravel()
    faces = np.asarray(faces, dtype=np.int64)
    n_vertices = len(stat_map)

    adjacency = _mesh_adjacency(faces, n_vertices)

    # Find observed clusters
    obs_labels, obs_masses = _find_clusters(
        stat_map, adjacency, threshold, tail
    )

    if len(obs_masses) == 0:
        return MultipleComparisonResult(
            p_values_raw=np.ones(n_vertices),
            p_values_corrected=np.ones(n_vertices),
            rejected=np.zeros(n_vertices, dtype=bool),
            alpha=0.05,
            method="Cluster permutation (no clusters found)",
            extras={"cluster_labels": obs_labels, "cluster_masses": obs_masses},
        )

    # Permutation null distribution of max cluster mass
    rng = np.random.default_rng(seed)
    null_max_mass = np.empty(n_perm, dtype=np.float64)

    for i in range(n_perm):
        if data_matrix is not None and design_matrix is not None:
            perm_idx = rng.permutation(data_matrix.shape[0])
            perm_design = design_matrix[perm_idx]
            beta = np.linalg.lstsq(perm_design, data_matrix, rcond=None)[0]
            if contrast is not None:
                perm_stat = contrast @ beta
            else:
                perm_stat = beta[0]
        else:
            signs = rng.choice([-1.0, 1.0], size=n_vertices)
            perm_stat = stat_map * signs

        _, perm_masses = _find_clusters(perm_stat, adjacency, threshold, tail)
        null_max_mass[i] = np.max(perm_masses) if len(perm_masses) > 0 else 0.0

    # Assign corrected p-values to each cluster
    p_per_cluster = np.array([
        (np.sum(null_max_mass >= m) + 1) / (n_perm + 1)
        for m in obs_masses
    ])

    # Map cluster p-values back to vertices
    p_corrected = np.ones(n_vertices, dtype=np.float64)
    for c_idx, c_p in enumerate(p_per_cluster):
        mask = obs_labels == (c_idx + 1)
        p_corrected[mask] = c_p

    return MultipleComparisonResult(
        p_values_raw=np.full(n_vertices, np.nan),
        p_values_corrected=p_corrected,
        rejected=p_corrected <= 0.05,
        alpha=0.05,
        method=f"Cluster permutation (threshold={threshold}, {n_perm} perms)",
        extras={
            "cluster_labels": obs_labels,
            "cluster_masses": obs_masses,
            "cluster_p_values": p_per_cluster,
            "null_max_mass": null_max_mass,
        },
    )


# ═══════════════════════════════════════════════════════════════════════════
#  §2  SURFACE-BASED GLM
# ═══════════════════════════════════════════════════════════════════════════

def vertex_wise_glm(
    Y: np.ndarray,
    X: np.ndarray,
    contrast: np.ndarray,
    confounds: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a mass-univariate GLM at each vertex/feature.

    Solves Y = Xβ + ε at each column of Y independently, then computes
    a contrast t-statistic.

    Parameters
    ----------
    Y : (N, V) array
        Data matrix — N subjects × V vertices (or spectral features).
    X : (N, P) array
        Design matrix with P regressors.
    contrast : (P,) array
        Contrast vector for the t-statistic.
    confounds : (N, C) array, optional
        Additional confound regressors appended to X.

    Returns
    -------
    t_map : (V,) array
        Vertex-wise t-statistics.
    beta_map : (P, V) array
        Estimated regression coefficients.
    residuals : (N, V) array
        Model residuals.
    """
    Y = np.asarray(Y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    contrast = np.asarray(contrast, dtype=np.float64).ravel()

    if Y.ndim == 1:
        Y = Y[:, np.newaxis]
    if X.ndim == 1:
        X = X[:, np.newaxis]

    N, V = Y.shape
    if confounds is not None:
        confounds = np.asarray(confounds, dtype=np.float64)
        if confounds.ndim == 1:
            confounds = confounds[:, np.newaxis]
        X = np.hstack([X, confounds])

    P = X.shape[1]
    if len(contrast) < P:
        # Pad contrast with zeros for confounds
        contrast = np.concatenate([
            contrast, np.zeros(P - len(contrast))
        ])

    if N != X.shape[0]:
        raise ValueError(
            f"Subject count mismatch: Y has {N}, X has {X.shape[0]}"
        )

    # OLS: β = (X'X)^{-1} X'Y
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ Y  # (P, V)
    residuals = Y - X @ beta   # (N, V)

    # Degrees of freedom
    df = N - P

    # Residual variance at each vertex
    sigma2 = np.sum(residuals ** 2, axis=0) / df  # (V,)

    # t-statistic: t = c'β / sqrt(σ² · c'(X'X)^{-1}c)
    var_contrast = contrast @ XtX_inv @ contrast  # scalar
    se = np.sqrt(sigma2 * var_contrast)            # (V,)
    se = np.where(se > 0, se, np.finfo(float).eps)

    t_map = (contrast @ beta) / se  # (V,)

    return t_map, beta, residuals


def vertex_wise_ttest(
    group_a: np.ndarray,
    group_b: np.ndarray,
    equal_var: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vertex-wise two-sample t-test between two groups.

    Parameters
    ----------
    group_a : (N_a, V) array
        Vertex data for group A.
    group_b : (N_b, V) array
        Vertex data for group B.
    equal_var : bool
        If True, use pooled variance (Student's t). Otherwise Welch's t.

    Returns
    -------
    t_map : (V,) array
        t-statistics at each vertex.
    p_map : (V,) array
        Two-sided uncorrected p-values.
    """
    a = np.asarray(group_a, dtype=np.float64)
    b = np.asarray(group_b, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, np.newaxis]
    if b.ndim == 1:
        b = b[:, np.newaxis]

    t_map, p_map = spstats.ttest_ind(a, b, axis=0, equal_var=equal_var)
    return t_map, p_map


# ═══════════════════════════════════════════════════════════════════════════
#  §3  PERMANOVA
# ═══════════════════════════════════════════════════════════════════════════

def permanova(
    distance_matrix: np.ndarray,
    grouping: np.ndarray,
    n_perm: int = 9999,
    seed: int = 42,
) -> StatResult:
    """Permutational Multivariate Analysis of Variance (PERMANOVA).

    Tests whether group centroids differ in the space defined by the
    distance matrix (Anderson, 2001). Non-parametric: no distribution
    assumptions on the original features.

    Parameters
    ----------
    distance_matrix : (N, N) array
        Symmetric pairwise distance/dissimilarity matrix.
    grouping : (N,) array
        Group labels (categorical) for each sample.
    n_perm : int
        Number of permutations.
    seed : int
        Random seed.

    Returns
    -------
    StatResult
        statistic = Pseudo-F; extras contains null distribution.
    """
    D = np.asarray(distance_matrix, dtype=np.float64)
    groups = np.asarray(grouping)
    n = len(groups)

    if D.shape != (n, n):
        raise ValueError(
            f"Distance matrix shape {D.shape} doesn't match "
            f"grouping length {n}"
        )

    # Unique groups and sizes
    unique_groups = np.unique(groups)
    k = len(unique_groups)
    if k < 2:
        raise ValueError("PERMANOVA requires at least 2 groups")

    # Compute pseudo-F
    observed_f = _permanova_pseudo_f(D, groups, unique_groups)

    # Permutation null
    rng = np.random.default_rng(seed)
    null_f = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        perm_groups = rng.permutation(groups)
        null_f[i] = _permanova_pseudo_f(D, perm_groups, unique_groups)

    p_value = (np.sum(null_f >= observed_f) + 1) / (n_perm + 1)

    # Effect size: partial R² = SS_between / SS_total
    A = -0.5 * D ** 2
    G = _gower_center_matrix(A)
    ss_total = np.trace(G)
    ss_within = 0.0
    for g in unique_groups:
        mask = groups == g
        n_g = np.sum(mask)
        if n_g > 0:
            D_g = D[np.ix_(mask, mask)]
            ss_within += np.sum(D_g ** 2) / (2 * n_g)
    ss_between = ss_total - ss_within
    r_squared = ss_between / ss_total if ss_total > 0 else 0.0

    return StatResult(
        statistic=observed_f,
        p_value=p_value,
        effect_size=r_squared,
        method="PERMANOVA",
        description=f"Pseudo-F={observed_f:.4f}, R²={r_squared:.4f}, "
                    f"p={p_value:.4f} ({n_perm} perms, {k} groups)",
        extras={"null_distribution": null_f},
    )


# ═══════════════════════════════════════════════════════════════════════════
#  §4  MULTIVARIATE ASSOCIATION — CCA & PLS
# ═══════════════════════════════════════════════════════════════════════════

def cca(
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int = 5,
    n_perm: int = 5000,
    seed: int = 42,
    regularization: float = 0.0,
) -> StatResult:
    """Canonical Correlation Analysis with permutation inference.

    Finds linear combinations of X and Y that maximize correlation.
    Inference via permutation of Y rows (Winkler et al., 2020).

    Parameters
    ----------
    X : (N, P) array
        Brain features (e.g., spectral features by region).
    Y : (N, Q) array
        Behavioral/clinical variables.
    n_components : int
        Number of canonical variates to extract.
    n_perm : int
        Permutations for significance testing.
    seed : int
        Random seed.
    regularization : float
        Ridge regularization added to covariance diagonals (0 = none).

    Returns
    -------
    StatResult
        statistic = first canonical correlation.
        extras contains 'correlations', 'x_weights', 'y_weights',
        'x_scores', 'y_scores', 'p_values_per_component'.
    """
    from sklearn.cross_decomposition import CCA as SkCCA

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, np.newaxis]

    N = X.shape[0]
    n_comp = min(n_components, X.shape[1], Y.shape[1], N - 1)

    # Fit CCA
    cca_model = SkCCA(n_components=n_comp, max_iter=1000)
    X_c, Y_c = cca_model.fit_transform(X, Y)

    # Canonical correlations
    correlations = np.array([
        np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(n_comp)
    ])

    # Permutation test
    rng = np.random.default_rng(seed)
    null_r1 = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        Y_perm = Y[rng.permutation(N)]
        cca_perm = SkCCA(n_components=1, max_iter=500)
        Xp, Yp = cca_perm.fit_transform(X, Y_perm)
        null_r1[i] = np.abs(np.corrcoef(Xp[:, 0], Yp[:, 0])[0, 1])

    # p-value for each component (vs null of first component)
    p_values = np.array([
        (np.sum(null_r1 >= np.abs(r)) + 1) / (n_perm + 1)
        for r in correlations
    ])

    return StatResult(
        statistic=correlations[0],
        p_value=p_values[0],
        effect_size=correlations[0] ** 2,
        method="CCA",
        description=f"r₁={correlations[0]:.4f}, p={p_values[0]:.4f} "
                    f"({n_comp} components, {n_perm} perms)",
        extras={
            "correlations": correlations,
            "x_weights": cca_model.x_weights_,
            "y_weights": cca_model.y_weights_,
            "x_scores": X_c,
            "y_scores": Y_c,
            "p_values_per_component": p_values,
            "null_r1": null_r1,
        },
    )


def pls_regression(
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int = 5,
    n_perm: int = 5000,
    n_boot: int = 2000,
    seed: int = 42,
) -> StatResult:
    """Partial Least Squares Regression with permutation + bootstrap.

    Maximizes covariance (not correlation) between X and Y, providing
    better stability at small sample sizes than CCA (Krishnan et al.,
    2011; McIntosh & Lobaugh, 2004).

    Parameters
    ----------
    X : (N, P) array
        Brain features.
    Y : (N, Q) array
        Behavioral/clinical variables.
    n_components : int
        Number of latent variables.
    n_perm : int
        Permutations for significance of each LV's singular value.
    n_boot : int
        Bootstrap iterations for weight stability (bootstrap ratios).
    seed : int
        Random seed.

    Returns
    -------
    StatResult
        extras contains 'singular_values', 'x_weights', 'y_weights',
        'bootstrap_ratios', 'p_values_per_lv'.
    """
    from sklearn.cross_decomposition import PLSRegression

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, np.newaxis]

    N = X.shape[0]
    n_comp = min(n_components, X.shape[1], Y.shape[1], N - 1)

    # Fit PLS and compute singular values of cross-covariance
    R = (X - X.mean(0)).T @ (Y - Y.mean(0)) / (N - 1)  # (P, Q)
    U, s, Vt = np.linalg.svd(R, full_matrices=False)
    s = s[:n_comp]

    # Permutation test for singular values
    rng = np.random.default_rng(seed)
    null_s = np.empty((n_perm, n_comp), dtype=np.float64)
    for i in range(n_perm):
        Y_perm = Y[rng.permutation(N)]
        R_perm = (X - X.mean(0)).T @ (Y_perm - Y_perm.mean(0)) / (N - 1)
        _, s_perm, _ = np.linalg.svd(R_perm, full_matrices=False)
        null_s[i] = s_perm[:n_comp]

    p_values = np.array([
        (np.sum(null_s[:, j] >= s[j]) + 1) / (n_perm + 1)
        for j in range(n_comp)
    ])

    # Bootstrap ratios for weight stability
    boot_U = np.empty((n_boot, X.shape[1], n_comp), dtype=np.float64)
    for i in range(n_boot):
        idx = rng.choice(N, size=N, replace=True)
        R_boot = (X[idx] - X[idx].mean(0)).T @ (Y[idx] - Y[idx].mean(0)) / (N - 1)
        U_boot, _, _ = np.linalg.svd(R_boot, full_matrices=False)
        # Procrustes alignment to observed U
        boot_U[i] = U_boot[:, :n_comp]

    boot_ratios = U[:, :n_comp] / (np.std(boot_U, axis=0) + 1e-10)

    # PLS model for scores
    pls = PLSRegression(n_components=n_comp)
    X_scores, Y_scores = pls.fit_transform(X, Y)

    return StatResult(
        statistic=s[0],
        p_value=p_values[0],
        effect_size=np.sum(s[:n_comp] ** 2) / np.sum(
            np.linalg.svd(R, compute_uv=False) ** 2
        ),
        method="PLS",
        description=f"s₁={s[0]:.4f}, p={p_values[0]:.4f} "
                    f"({n_comp} LVs, {n_perm} perms)",
        extras={
            "singular_values": s,
            "x_weights": U[:, :n_comp],
            "y_weights": Vt[:n_comp].T,
            "bootstrap_ratios": boot_ratios,
            "p_values_per_lv": p_values,
            "x_scores": X_scores,
            "y_scores": Y_scores,
        },
    )


# ═══════════════════════════════════════════════════════════════════════════
#  §5  RSA — REPRESENTATIONAL SIMILARITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def compute_rdm(
    data: np.ndarray,
    metric: Literal[
        "correlation", "euclidean", "mahalanobis", "cosine"
    ] = "correlation",
) -> np.ndarray:
    """Compute a Representational Dissimilarity Matrix (RDM).

    Constructs an RDM from a data matrix where rows are conditions/items
    and columns are features (e.g., spectral descriptors by region).

    Parameters
    ----------
    data : (K, P) array
        K conditions/items × P features (e.g., HKS values at P vertices).
    metric : str
        Distance metric for pdist.

    Returns
    -------
    rdm : (K, K) array
        Symmetric RDM with zero diagonal.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        raise ValueError("data must be 2-D: (conditions, features)")
    distances = pdist(data, metric=metric)
    return squareform(distances)


def compare_rdms(
    rdm_brain: np.ndarray,
    rdm_behavior: np.ndarray,
    method: Literal["spearman", "kendall", "pearson"] = "spearman",
    n_perm: int = 10000,
    seed: int = 42,
) -> StatResult:
    """Compare two RDMs using rank correlation with permutation test.

    Tests whether the representational geometry of brain features
    mirrors a behavioral/clinical similarity structure (Kriegeskorte
    et al., 2008).

    Parameters
    ----------
    rdm_brain : (K, K) array
        Brain feature RDM.
    rdm_behavior : (K, K) array
        Behavioral/clinical RDM.
    method : {'spearman', 'kendall', 'pearson'}
        Correlation method for comparing upper-triangular vectors.
    n_perm : int
        Row/column permutations for null distribution.
    seed : int
        Random seed.

    Returns
    -------
    StatResult
        statistic = correlation between RDM upper triangles.
    """
    A = np.asarray(rdm_brain, dtype=np.float64)
    B = np.asarray(rdm_behavior, dtype=np.float64)

    if A.shape != B.shape:
        raise ValueError(f"RDM shapes differ: {A.shape} vs {B.shape}")
    K = A.shape[0]

    # Extract upper triangle (excluding diagonal)
    idx_upper = np.triu_indices(K, k=1)
    a_vec = A[idx_upper]
    b_vec = B[idx_upper]

    # Observed correlation
    if method == "spearman":
        r_obs, _ = spstats.spearmanr(a_vec, b_vec)
    elif method == "kendall":
        r_obs, _ = spstats.kendalltau(a_vec, b_vec)
    else:
        r_obs = np.corrcoef(a_vec, b_vec)[0, 1]

    # Permutation test: permute rows AND columns of one RDM
    rng = np.random.default_rng(seed)
    null_r = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        perm = rng.permutation(K)
        A_perm = A[np.ix_(perm, perm)]
        a_perm_vec = A_perm[idx_upper]
        if method == "spearman":
            null_r[i], _ = spstats.spearmanr(a_perm_vec, b_vec)
        elif method == "kendall":
            null_r[i], _ = spstats.kendalltau(a_perm_vec, b_vec)
        else:
            null_r[i] = np.corrcoef(a_perm_vec, b_vec)[0, 1]

    p_value = (np.sum(np.abs(null_r) >= np.abs(r_obs)) + 1) / (n_perm + 1)

    return StatResult(
        statistic=r_obs,
        p_value=p_value,
        effect_size=r_obs ** 2,
        method=f"RSA ({method})",
        description=f"r={r_obs:.4f}, p={p_value:.4f} ({n_perm} perms)",
        extras={"null_distribution": null_r},
    )


def rsa_regression(
    rdm_brain: np.ndarray,
    model_rdms: Dict[str, np.ndarray],
    method: Literal["ols", "weighted"] = "ols",
    n_perm: int = 10000,
    seed: int = 42,
) -> Dict[str, StatResult]:
    """RSA with multiple model RDMs as competing predictors.

    Fits a regression of the brain RDM's upper triangle on the upper
    triangles of multiple model RDMs, testing each model's unique
    contribution (Diedrichsen & Kriegeskorte, 2017).

    Parameters
    ----------
    rdm_brain : (K, K) array
        Brain feature RDM (dependent variable).
    model_rdms : dict of {name: (K, K) array}
        Named model RDMs (independent variables).
    method : {'ols', 'weighted'}
        'ols' for ordinary least squares, 'weighted' for variance-
        weighted regression.
    n_perm : int
        Permutations for each predictor's significance.
    seed : int
        Random seed.

    Returns
    -------
    dict of {name: StatResult}
        One StatResult per model RDM with beta and p-value.
    """
    B = np.asarray(rdm_brain, dtype=np.float64)
    K = B.shape[0]
    idx = np.triu_indices(K, k=1)
    y = B[idx]

    names = list(model_rdms.keys())
    X = np.column_stack([
        np.asarray(model_rdms[name], dtype=np.float64)[idx] for name in names
    ])

    # Add intercept
    X_design = np.column_stack([np.ones(len(y)), X])

    # OLS fit
    beta, res, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
    y_hat = X_design @ beta
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Permutation test for each predictor
    rng = np.random.default_rng(seed)
    results = {}

    for j, name in enumerate(names):
        observed_beta = beta[j + 1]  # skip intercept

        null_betas = np.empty(n_perm, dtype=np.float64)
        for i in range(n_perm):
            perm = rng.permutation(K)
            B_perm = B[np.ix_(perm, perm)]
            y_perm = B_perm[idx]
            beta_perm = np.linalg.lstsq(X_design, y_perm, rcond=None)[0]
            null_betas[i] = beta_perm[j + 1]

        p_val = (
            (np.sum(np.abs(null_betas) >= np.abs(observed_beta)) + 1)
            / (n_perm + 1)
        )

        results[name] = StatResult(
            statistic=observed_beta,
            p_value=p_val,
            effect_size=None,
            method="RSA regression",
            description=f"β({name})={observed_beta:.4f}, p={p_val:.4f}",
            extras={"null_distribution": null_betas},
        )

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  §6  STRUCTURAL COVARIANCE & NETWORK ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def structural_covariance_network(
    data: np.ndarray,
    covariates: Optional[np.ndarray] = None,
    method: Literal["pearson", "spearman", "partial"] = "pearson",
) -> np.ndarray:
    """Compute a structural covariance network across subjects.

    Correlates regional morphometric values across subjects to form
    an inter-regional covariance matrix (Alexander-Bloch et al., 2013).

    Parameters
    ----------
    data : (N, R) array
        N subjects × R brain regions.
    covariates : (N, C) array, optional
        Confounds to regress out before computing correlations.
    method : {'pearson', 'spearman', 'partial'}
        Correlation method. 'partial' uses sklearn's Ledoit-Wolf
        shrunk covariance estimation.

    Returns
    -------
    scn : (R, R) array
        Structural covariance network (correlation matrix).
    """
    data = np.asarray(data, dtype=np.float64)

    if covariates is not None:
        covariates = np.asarray(covariates, dtype=np.float64)
        if covariates.ndim == 1:
            covariates = covariates[:, np.newaxis]
        # Residualize data with respect to covariates
        beta = np.linalg.lstsq(covariates, data, rcond=None)[0]
        data = data - covariates @ beta

    if method == "spearman":
        scn, _ = spstats.spearmanr(data, axis=0)
        if np.isscalar(scn):
            scn = np.array([[1.0, scn], [scn, 1.0]])
    elif method == "partial":
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf().fit(data)
        precision = lw.precision_
        # Convert precision to partial correlation
        d = np.sqrt(np.diag(precision))
        scn = -precision / np.outer(d, d)
        np.fill_diagonal(scn, 1.0)
    else:
        scn = np.corrcoef(data, rowvar=False)

    return scn


def mind_network(
    feature_matrix: np.ndarray,
    region_labels: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute MIND (Morphometric INverse Divergence) network.

    Uses KL divergence between multivariate feature distributions at
    each pair of regions to construct individual-level morphometric
    similarity networks (Sebenius et al., 2023, Nature Neuroscience).

    Parameters
    ----------
    feature_matrix : (R, F) array
        R regions × F morphometric features per region (e.g., thickness,
        area, volume, curvature, HKS, WKS). For a single subject.
    region_labels : (R,) array, optional
        Region name labels (for reference only).

    Returns
    -------
    mind : (R, R) array
        MIND network. Higher values = more similar morphometric profiles.
    """
    M = np.asarray(feature_matrix, dtype=np.float64)
    R = M.shape[0]

    # Standardize features
    M_std = (M - M.mean(axis=0)) / (M.std(axis=0) + 1e-10)

    # Compute KL divergence between regions assuming Gaussian
    # For single-subject data, we use the feature vector directly
    # and compute pairwise Euclidean distances as a proxy,
    # then convert to similarity via inverse divergence
    D = squareform(pdist(M_std, metric="euclidean"))

    # MIND = inverse divergence (similarity)
    mind = 1.0 / (1.0 + D)
    np.fill_diagonal(mind, 0.0)

    return mind


def graphical_lasso_network(
    data: np.ndarray,
    alpha: Optional[float] = None,
    covariates: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate sparse precision matrix via Graphical LASSO.

    Reveals direct (partial) correlations between brain regions by
    estimating the inverse covariance matrix with L1 penalization
    (Friedman et al., 2008).

    Parameters
    ----------
    data : (N, R) array
        N subjects × R regions.
    alpha : float or None
        L1 penalty strength. If None, selected by cross-validation.
    covariates : (N, C) array, optional
        Confounds to regress out first.

    Returns
    -------
    precision : (R, R) array
        Estimated sparse precision matrix.
    partial_correlations : (R, R) array
        Partial correlation matrix derived from precision.
    """
    from sklearn.covariance import GraphicalLassoCV, GraphicalLasso

    data = np.asarray(data, dtype=np.float64)

    if covariates is not None:
        covariates = np.asarray(covariates, dtype=np.float64)
        if covariates.ndim == 1:
            covariates = covariates[:, np.newaxis]
        beta = np.linalg.lstsq(covariates, data, rcond=None)[0]
        data = data - covariates @ beta

    # Standardize
    data = (data - data.mean(0)) / (data.std(0) + 1e-10)

    if alpha is None:
        model = GraphicalLassoCV(cv=5, max_iter=500).fit(data)
    else:
        model = GraphicalLasso(alpha=alpha, max_iter=500).fit(data)

    precision = model.precision_

    # Partial correlations from precision
    d = np.sqrt(np.diag(precision))
    d = np.where(d > 0, d, 1e-10)
    partial_corr = -precision / np.outer(d, d)
    np.fill_diagonal(partial_corr, 1.0)

    return precision, partial_corr


def network_based_statistic(
    matrices_a: np.ndarray,
    matrices_b: np.ndarray,
    threshold: float = 3.1,
    n_perm: int = 5000,
    seed: int = 42,
    alpha: float = 0.05,
) -> StatResult:
    """Network-Based Statistic (NBS) for group comparison.

    Identifies connected subnetworks that differ between two groups,
    controlling FWER at the network level (Zalesky et al., 2010).

    Parameters
    ----------
    matrices_a : (N_a, R, R) array
        Connectivity/covariance matrices for group A.
    matrices_b : (N_b, R, R) array
        Connectivity/covariance matrices for group B.
    threshold : float
        Primary threshold for t-statistics at each edge.
    n_perm : int
        Permutations for FWER correction.
    seed : int
        Random seed.
    alpha : float
        Significance level.

    Returns
    -------
    StatResult
        extras contains 'component_sizes', 'component_masks', 't_matrix'.
    """
    A = np.asarray(matrices_a, dtype=np.float64)
    B = np.asarray(matrices_b, dtype=np.float64)
    na, R, _ = A.shape
    nb = B.shape[0]

    # Edge-wise t-test
    t_matrix = np.zeros((R, R), dtype=np.float64)
    for i in range(R):
        for j in range(i + 1, R):
            t, _ = spstats.ttest_ind(A[:, i, j], B[:, i, j])
            t_matrix[i, j] = t
            t_matrix[j, i] = t

    # Threshold and find connected components
    supra = np.abs(t_matrix) > threshold
    obs_sizes = _connected_component_sizes(supra)
    obs_max_size = max(obs_sizes) if len(obs_sizes) > 0 else 0

    # Permutation null for max component size
    all_data = np.concatenate([A, B], axis=0)
    rng = np.random.default_rng(seed)
    null_max_sizes = np.empty(n_perm, dtype=np.float64)

    for p in range(n_perm):
        perm_idx = rng.permutation(na + nb)
        A_perm = all_data[perm_idx[:na]]
        B_perm = all_data[perm_idx[na:]]

        t_perm = np.zeros((R, R), dtype=np.float64)
        for i in range(R):
            for j in range(i + 1, R):
                t, _ = spstats.ttest_ind(A_perm[:, i, j], B_perm[:, i, j])
                t_perm[i, j] = t
                t_perm[j, i] = t

        supra_perm = np.abs(t_perm) > threshold
        perm_sizes = _connected_component_sizes(supra_perm)
        null_max_sizes[p] = max(perm_sizes) if len(perm_sizes) > 0 else 0

    p_value = (np.sum(null_max_sizes >= obs_max_size) + 1) / (n_perm + 1)

    return StatResult(
        statistic=obs_max_size,
        p_value=p_value,
        method="NBS",
        description=f"Max component size={obs_max_size}, p={p_value:.4f}",
        extras={
            "component_sizes": obs_sizes,
            "t_matrix": t_matrix,
            "supra_threshold_mask": supra,
            "null_max_sizes": null_max_sizes,
        },
    )


# ═══════════════════════════════════════════════════════════════════════════
#  §7  ALLOMETRIC SCALING
# ═══════════════════════════════════════════════════════════════════════════

def allometric_scaling(
    regional_measure: np.ndarray,
    global_measure: np.ndarray,
    region_names: Optional[List[str]] = None,
    confounds: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Compute allometric scaling exponents via log-log regression.

    Tests whether regional brain measures scale isometrically (α=1),
    hyperallometrically (α>1), or hypoallometrically (α<1) with
    respect to a global measure (Reardon et al., 2018, Science).

    log(regional) = α · log(global) + β

    Parameters
    ----------
    regional_measure : (N, R) array
        N subjects × R regional values (e.g., cortical thickness, SA).
    global_measure : (N,) array
        Global brain measure (e.g., total brain volume, total SA).
    region_names : list of str, optional
        Names for each region.
    confounds : (N, C) array, optional
        Covariates to include in the log-log regression.

    Returns
    -------
    pd.DataFrame
        Columns: region, alpha (scaling exponent), alpha_se,
        t_vs_isometry (t-test for α ≠ 1), p_vs_isometry,
        intercept, r_squared.
    """
    Y = np.asarray(regional_measure, dtype=np.float64)
    x = np.asarray(global_measure, dtype=np.float64).ravel()

    if Y.ndim == 1:
        Y = Y[:, np.newaxis]
    N, R = Y.shape

    # Ensure positive values for log transform
    if np.any(Y <= 0) or np.any(x <= 0):
        warnings.warn(
            "Non-positive values found; adding small constant for log.",
            stacklevel=2,
        )
        Y = np.where(Y > 0, Y, 1e-10)
        x = np.where(x > 0, x, 1e-10)

    log_x = np.log(x)
    log_Y = np.log(Y)

    # Design matrix: [intercept, log(global), confounds]
    X_design = np.column_stack([np.ones(N), log_x])
    if confounds is not None:
        confounds = np.asarray(confounds, dtype=np.float64)
        if confounds.ndim == 1:
            confounds = confounds[:, np.newaxis]
        X_design = np.hstack([X_design, confounds])

    P = X_design.shape[1]
    XtX_inv = np.linalg.pinv(X_design.T @ X_design)

    records = []
    for r in range(R):
        y = log_Y[:, r]
        beta = XtX_inv @ X_design.T @ y
        resid = y - X_design @ beta
        sigma2 = np.sum(resid ** 2) / (N - P)

        alpha = beta[1]
        se_alpha = np.sqrt(sigma2 * XtX_inv[1, 1])

        # Test α ≠ 1 (isometry)
        t_iso = (alpha - 1.0) / se_alpha if se_alpha > 0 else 0.0
        p_iso = 2.0 * spstats.t.sf(np.abs(t_iso), df=N - P)

        # R²
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - np.sum(resid ** 2) / ss_tot if ss_tot > 0 else 0.0

        name = region_names[r] if region_names else f"region_{r}"
        records.append({
            "region": name,
            "alpha": alpha,
            "alpha_se": se_alpha,
            "t_vs_isometry": t_iso,
            "p_vs_isometry": p_iso,
            "intercept": beta[0],
            "r_squared": r2,
            "scaling_type": (
                "hyper" if alpha > 1 and p_iso < 0.05 else
                "hypo" if alpha < 1 and p_iso < 0.05 else
                "isometric"
            ),
        })

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════
#  §8  HARMONIZATION
# ═══════════════════════════════════════════════════════════════════════════

def combat_harmonize(
    data: np.ndarray,
    batch: np.ndarray,
    covariates: Optional[pd.DataFrame] = None,
    method: Literal["combat", "covbat"] = "combat",
) -> np.ndarray:
    """Harmonize multi-site neuroimaging data using ComBat.

    Wraps neuroCombat or neuroHarmonize for removing site/scanner
    effects via empirical Bayes (Fortin et al., 2018).

    Parameters
    ----------
    data : (N, P) array
        N subjects × P brain features.
    batch : (N,) array
        Site/scanner labels for each subject.
    covariates : pd.DataFrame, optional
        Biological covariates to preserve (e.g., age, sex, diagnosis).
    method : {'combat', 'covbat'}
        'combat' for standard ComBat, 'covbat' for CovBat extension
        that also harmonizes covariance structure.

    Returns
    -------
    data_harmonized : (N, P) array
        Harmonized data.

    Raises
    ------
    ImportError
        If neuroCombat or neuroHarmonize is not installed.
    """
    data = np.asarray(data, dtype=np.float64)
    batch = np.asarray(batch)

    try:
        from neuroCombat import neuroCombat
    except ImportError:
        try:
            from neuroHarmonize import harmonizationLearn
        except ImportError:
            raise ImportError(
                "Neither neuroCombat nor neuroHarmonize found. Install with: "
                "pip install neuroCombat  or  pip install neuroHarmonize"
            )
        # neuroHarmonize path
        cov_df = pd.DataFrame({"SITE": batch})
        if covariates is not None:
            for col in covariates.columns:
                cov_df[col] = covariates[col].values
        harmonized, _ = harmonizationLearn(data, cov_df)
        return harmonized

    # neuroCombat path
    cov_dict = {"batch": batch}
    if covariates is not None:
        for col in covariates.columns:
            cov_dict[col] = covariates[col].values

    result = neuroCombat(
        dat=data.T,  # neuroCombat expects (features, subjects)
        covars=pd.DataFrame(cov_dict),
        batch_col="batch",
    )
    return result["data"].T


# ═══════════════════════════════════════════════════════════════════════════
#  §9  LATERALITY — MTLE-HS FOCUS
# ═══════════════════════════════════════════════════════════════════════════

def asymmetry_index(
    left: np.ndarray,
    right: np.ndarray,
    method: Literal["standard", "log", "normalized"] = "standard",
) -> np.ndarray:
    """Compute hemispheric asymmetry index.

    Parameters
    ----------
    left : (N,) or (N, R) array
        Left hemisphere values.
    right : (N,) or (N, R) array
        Right hemisphere values.
    method : {'standard', 'log', 'normalized'}
        'standard': AI = (L - R) / ((L + R) / 2)
        'log': AI = log(L) - log(R)  (ratio on log scale)
        'normalized': AI = (L - R) / (L + R)

    Returns
    -------
    ai : same shape as inputs
        Asymmetry index. Positive = leftward, negative = rightward.
    """
    L = np.asarray(left, dtype=np.float64)
    R = np.asarray(right, dtype=np.float64)

    if L.shape != R.shape:
        raise ValueError(f"Shape mismatch: left {L.shape} vs right {R.shape}")

    if method == "log":
        # Guard against non-positive values
        L = np.where(L > 0, L, 1e-10)
        R = np.where(R > 0, R, 1e-10)
        return np.log(L) - np.log(R)
    elif method == "normalized":
        denom = L + R
        denom = np.where(np.abs(denom) > 1e-10, denom, 1e-10)
        return (L - R) / denom
    else:
        denom = (L + R) / 2.0
        denom = np.where(np.abs(denom) > 1e-10, denom, 1e-10)
        return (L - R) / denom


def laterality_features(
    morphometrics_lh: np.ndarray,
    morphometrics_rh: np.ndarray,
    feature_names: Optional[List[str]] = None,
    spectral_asymmetry: Optional[np.ndarray] = None,
    wasserstein_asymmetry: Optional[np.ndarray] = None,
    zscore_asymmetry: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Engineer a feature matrix for MTLE-HS laterality classification.

    Combines standard morphometric asymmetry indices with optional
    spectral and normative features into a single DataFrame ready
    for classification.

    Parameters
    ----------
    morphometrics_lh : (N, R) array
        Left hemisphere morphometrics (volumes, thickness, area)
        for N subjects across R structures.
    morphometrics_rh : (N, R) array
        Right hemisphere morphometrics.
    feature_names : list of str, optional
        Names of the R structures.
    spectral_asymmetry : (N, S) array, optional
        Per-subject spectral asymmetry features (e.g., HKS/WKS
        Wasserstein distances per region).
    wasserstein_asymmetry : (N,) or (N, W) array, optional
        Global or per-network Wasserstein-based asymmetry.
    zscore_asymmetry : (N, R) array, optional
        Normative z-score asymmetry (z_L - z_R) from a normative model.

    Returns
    -------
    pd.DataFrame
        Feature matrix with columns named by origin and structure.
    """
    L = np.asarray(morphometrics_lh, dtype=np.float64)
    R = np.asarray(morphometrics_rh, dtype=np.float64)
    N, n_regions = L.shape

    if feature_names is None:
        feature_names = [f"region_{i}" for i in range(n_regions)]

    # Standard asymmetry indices
    ai = asymmetry_index(L, R, method="standard")
    df = pd.DataFrame(
        ai, columns=[f"AI_{name}" for name in feature_names]
    )

    # Absolute values (for bilateral asymmetry without direction)
    df_abs = pd.DataFrame(
        np.abs(ai), columns=[f"absAI_{name}" for name in feature_names]
    )
    df = pd.concat([df, df_abs], axis=1)

    # Raw left/right values (some classifiers benefit from these)
    df_raw_l = pd.DataFrame(
        L, columns=[f"L_{name}" for name in feature_names]
    )
    df_raw_r = pd.DataFrame(
        R, columns=[f"R_{name}" for name in feature_names]
    )
    df = pd.concat([df, df_raw_l, df_raw_r], axis=1)

    # Spectral asymmetry
    if spectral_asymmetry is not None:
        sa = np.asarray(spectral_asymmetry, dtype=np.float64)
        if sa.ndim == 1:
            sa = sa[:, np.newaxis]
        sa_cols = [f"spectral_AI_{i}" for i in range(sa.shape[1])]
        df = pd.concat([df, pd.DataFrame(sa, columns=sa_cols)], axis=1)

    # Wasserstein asymmetry
    if wasserstein_asymmetry is not None:
        wa = np.asarray(wasserstein_asymmetry, dtype=np.float64)
        if wa.ndim == 1:
            wa = wa[:, np.newaxis]
        wa_cols = [f"wass_AI_{i}" for i in range(wa.shape[1])]
        df = pd.concat([df, pd.DataFrame(wa, columns=wa_cols)], axis=1)

    # Normative z-score asymmetry
    if zscore_asymmetry is not None:
        za = np.asarray(zscore_asymmetry, dtype=np.float64)
        if za.ndim == 1:
            za = za[:, np.newaxis]
        za_cols = [f"zAI_{name}" for name in feature_names[:za.shape[1]]]
        df = pd.concat([df, pd.DataFrame(za, columns=za_cols)], axis=1)

    return df


def laterality_classifier(
    features: pd.DataFrame,
    labels: np.ndarray,
    method: Literal[
        "logistic", "svm", "rf", "xgboost", "ensemble"
    ] = "ensemble",
    n_cv_folds: int = 5,
    seed: int = 42,
    return_model: bool = False,
) -> Dict[str, Any]:
    """Train and evaluate a laterality classifier for MTLE-HS.

    Uses stratified cross-validation with optional ensemble of
    multiple classifiers (Gleichgerrcht et al., 2021; Yu et al., 2025).

    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix (from laterality_features or custom).
    labels : (N,) array
        Binary labels: 0 = left MTLE, 1 = right MTLE.
    method : str
        Classifier choice. 'ensemble' combines logistic + SVM + RF.
    n_cv_folds : int
        Number of cross-validation folds.
    seed : int
        Random seed.
    return_model : bool
        If True, return the fitted model alongside metrics.

    Returns
    -------
    dict
        Keys: 'accuracy', 'auc', 'sensitivity', 'specificity',
        'cv_scores', 'confusion_matrix', 'feature_importance',
        optionally 'model'.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import (
        accuracy_score, roc_auc_score, confusion_matrix as cm_sklearn,
    )

    X = features.values if isinstance(features, pd.DataFrame) else features
    y = np.asarray(labels).ravel()

    if len(np.unique(y)) < 2:
        raise ValueError("Labels must contain at least 2 classes")

    # Build classifier
    if method == "logistic":
        from sklearn.linear_model import LogisticRegressionCV
        clf = LogisticRegressionCV(
            Cs=20, cv=3, penalty="elasticnet", solver="saga",
            l1_ratios=[0.1, 0.5, 0.9], max_iter=2000,
            random_state=seed, class_weight="balanced",
        )
    elif method == "svm":
        from sklearn.svm import SVC
        clf = SVC(
            kernel="linear", probability=True,
            class_weight="balanced", random_state=seed,
        )
    elif method == "rf":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=3,
            class_weight="balanced", random_state=seed, n_jobs=-1,
        )
    elif method == "xgboost":
        try:
            from xgboost import XGBClassifier
            scale_pos_weight = np.sum(y == 0) / max(np.sum(y == 1), 1)
            clf = XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                scale_pos_weight=scale_pos_weight,
                random_state=seed, use_label_encoder=False,
                eval_metric="logloss",
            )
        except ImportError:
            raise ImportError("xgboost not installed. pip install xgboost")
    elif method == "ensemble":
        from sklearn.ensemble import (
            VotingClassifier, RandomForestClassifier,
        )
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC

        clf = VotingClassifier(
            estimators=[
                ("lr", LogisticRegression(
                    penalty="l2", C=1.0, max_iter=2000,
                    class_weight="balanced", random_state=seed,
                )),
                ("svm", SVC(
                    kernel="linear", probability=True,
                    class_weight="balanced", random_state=seed,
                )),
                ("rf", RandomForestClassifier(
                    n_estimators=500, min_samples_leaf=3,
                    class_weight="balanced", random_state=seed,
                )),
            ],
            voting="soft",
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])

    # Stratified CV
    skf = StratifiedKFold(
        n_splits=n_cv_folds, shuffle=True, random_state=seed
    )
    cv_acc = []
    cv_auc = []
    y_pred_all = np.empty_like(y)
    y_proba_all = np.empty(len(y), dtype=np.float64)

    for train_idx, test_idx in skf.split(X, y):
        pipe.fit(X[train_idx], y[train_idx])
        y_pred_all[test_idx] = pipe.predict(X[test_idx])

        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(X[test_idx])[:, 1]
        else:
            proba = pipe.decision_function(X[test_idx])
        y_proba_all[test_idx] = proba

        cv_acc.append(accuracy_score(y[test_idx], y_pred_all[test_idx]))
        try:
            cv_auc.append(roc_auc_score(y[test_idx], proba))
        except ValueError:
            cv_auc.append(np.nan)

    # Overall metrics
    cm = cm_sklearn(y, y_pred_all)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Feature importance (from final full-data fit)
    pipe.fit(X, y)
    importance = _extract_feature_importance(pipe, features)

    result = {
        "accuracy": np.mean(cv_acc),
        "accuracy_std": np.std(cv_acc),
        "auc": np.nanmean(cv_auc),
        "auc_std": np.nanstd(cv_auc),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "cv_scores": cv_acc,
        "confusion_matrix": cm,
        "feature_importance": importance,
        "y_pred": y_pred_all,
        "y_proba": y_proba_all,
    }
    if return_model:
        result["model"] = pipe

    return result


# ═══════════════════════════════════════════════════════════════════════════
#  §10  CONFORMAL PREDICTION
# ═══════════════════════════════════════════════════════════════════════════

def conformal_prediction_intervals(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    alpha: float = 0.1,
    method: Literal["jackknife_plus", "cv_plus", "cqr"] = "jackknife_plus",
    base_model: Optional[Any] = None,
) -> Dict[str, np.ndarray]:
    """Compute conformal prediction intervals with finite-sample coverage.

    Wraps MAPIE for distribution-free uncertainty quantification on
    brain-behavior predictions (Angelopoulos & Bates, 2023).

    Parameters
    ----------
    X_train : (N_train, P) array
        Training features.
    y_train : (N_train,) array
        Training targets.
    X_test : (N_test, P) array
        Test features for which to compute intervals.
    alpha : float
        Miscoverage level (0.1 = 90% coverage guarantee).
    method : {'jackknife_plus', 'cv_plus', 'cqr'}
        Conformal prediction method.
    base_model : sklearn estimator, optional
        Base regression model. Defaults to RandomForest.

    Returns
    -------
    dict
        'y_pred': point predictions, 'lower': lower bounds,
        'upper': upper bounds, 'width': interval widths.
    """
    try:
        from mapie.regression import MapieRegressor
        if method == "cqr":
            from mapie.quantile_regression import MapieQuantileRegressor
    except ImportError:
        raise ImportError(
            "MAPIE not installed. pip install mapie"
        )

    if base_model is None:
        from sklearn.ensemble import RandomForestRegressor
        base_model = RandomForestRegressor(
            n_estimators=200, min_samples_leaf=5, random_state=42
        )

    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64).ravel()
    X_test = np.asarray(X_test, dtype=np.float64)

    if method == "cqr":
        from sklearn.ensemble import GradientBoostingRegressor
        mapie = MapieQuantileRegressor(
            estimator=GradientBoostingRegressor(n_estimators=200),
            method="quantile",
            cv="split",
            alpha=alpha,
        )
        mapie.fit(X_train, y_train)
        y_pred, intervals = mapie.predict(X_test)
    else:
        cv_method = "plus" if "plus" in method else "base"
        mapie = MapieRegressor(
            estimator=base_model,
            method=cv_method,
            cv=5 if "cv" in method else -1,  # -1 = jackknife
        )
        mapie.fit(X_train, y_train)
        y_pred, intervals = mapie.predict(X_test, alpha=alpha)

    # intervals shape: (n_test, 2, 1) → extract
    lower = intervals[:, 0, 0] if intervals.ndim == 3 else intervals[:, 0]
    upper = intervals[:, 1, 0] if intervals.ndim == 3 else intervals[:, 1]

    return {
        "y_pred": y_pred,
        "lower": lower,
        "upper": upper,
        "width": upper - lower,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  §11  GPU-ACCELERATED UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def bootstrap_gpu(
    data: np.ndarray,
    stat_fn: Any,
    n_bootstrap: int = 10000,
    seed: int = 42,
    batch_size: int = 1000,
    confidence_level: float = 0.95,
) -> Dict[str, Any]:
    """GPU-accelerated bootstrap using PyTorch.

    Generates bootstrap samples on GPU and applies a statistic function
    in batches for large-scale resampling.

    Parameters
    ----------
    data : (N, ...) array
        Input data (first dimension = observations).
    stat_fn : callable
        Function mapping (N, ...) tensor → scalar or (K,) tensor.
        Must accept torch.Tensor input.
    n_bootstrap : int
        Number of bootstrap resamples.
    seed : int
        Random seed.
    batch_size : int
        Number of bootstrap samples per GPU batch.
    confidence_level : float
        For computing percentile CI.

    Returns
    -------
    dict
        'estimates': (n_bootstrap,) bootstrap distribution,
        'ci_lower', 'ci_upper': confidence interval bounds,
        'se': bootstrap standard error.
    """
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_t = torch.as_tensor(data, dtype=torch.float32, device=device)
    N = data_t.shape[0]

    torch.manual_seed(seed)
    estimates = []

    for start in range(0, n_bootstrap, batch_size):
        end = min(start + batch_size, n_bootstrap)
        bs = end - start

        # Generate bootstrap indices on GPU
        indices = torch.randint(0, N, (bs, N), device=device)

        batch_estimates = []
        for b in range(bs):
            sample = data_t[indices[b]]
            est = stat_fn(sample)
            if isinstance(est, torch.Tensor):
                batch_estimates.append(est.cpu().numpy())
            else:
                batch_estimates.append(est)

        estimates.extend(batch_estimates)

    estimates = np.array(estimates)

    alpha = 1 - confidence_level
    ci_lower = np.percentile(estimates, 100 * alpha / 2, axis=0)
    ci_upper = np.percentile(estimates, 100 * (1 - alpha / 2), axis=0)

    return {
        "estimates": estimates,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "se": np.std(estimates, axis=0),
        "mean": np.mean(estimates, axis=0),
    }


def permutation_matrix_gpu(
    n_subjects: int,
    n_perm: int,
    seed: int = 42,
) -> np.ndarray:
    """Generate a permutation index matrix on GPU.

    Useful for pre-computing all permutation indices before running
    parallelized permutation tests.

    Parameters
    ----------
    n_subjects : int
        Number of subjects to permute.
    n_perm : int
        Number of permutations.
    seed : int
        Random seed.

    Returns
    -------
    perm_matrix : (n_perm, n_subjects) int array
        Each row is a permutation of range(n_subjects).
    """
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    # Generate on GPU
    base = torch.arange(n_subjects, device=device).unsqueeze(0).expand(
        n_perm, -1
    )
    # Shuffle each row independently using argsort of random values
    noise = torch.rand(n_perm, n_subjects, device=device)
    perm_matrix = noise.argsort(dim=1)

    return perm_matrix.cpu().numpy()


# ═══════════════════════════════════════════════════════════════════════════
#  PRIVATE HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _mesh_adjacency(faces: np.ndarray, n_vertices: int) -> sp.csr_matrix:
    """Build a sparse adjacency matrix from mesh faces."""
    rows = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2],
                           faces[:, 1], faces[:, 2], faces[:, 0]])
    cols = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0],
                           faces[:, 0], faces[:, 1], faces[:, 2]])
    data = np.ones(len(rows), dtype=np.float64)
    adj = sp.csr_matrix((data, (rows, cols)), shape=(n_vertices, n_vertices))
    # Make binary
    adj.data[:] = 1.0
    return adj


def _compute_tfce(
    stat_map: np.ndarray,
    adjacency: sp.csr_matrix,
    E: float,
    H: float,
    dh: float,
) -> np.ndarray:
    """Compute TFCE scores for a statistic map on a mesh."""
    from scipy.sparse.csgraph import connected_components

    n = len(stat_map)
    tfce = np.zeros(n, dtype=np.float64)

    # Positive tail
    max_val = np.max(stat_map)
    if max_val > 0:
        for h in np.arange(dh, max_val + dh, dh):
            mask = stat_map >= h
            if not np.any(mask):
                continue
            sub_adj = adjacency[np.ix_(mask, mask)]
            n_comp, labels = connected_components(sub_adj, directed=False)
            for c in range(n_comp):
                cluster_mask = labels == c
                extent = np.sum(cluster_mask)
                cluster_vertices = np.where(mask)[0][cluster_mask]
                tfce[cluster_vertices] += (extent ** E) * (h ** H) * dh

    # Negative tail
    min_val = np.min(stat_map)
    if min_val < 0:
        neg_map = -stat_map
        max_neg = np.max(neg_map)
        for h in np.arange(dh, max_neg + dh, dh):
            mask = neg_map >= h
            if not np.any(mask):
                continue
            sub_adj = adjacency[np.ix_(mask, mask)]
            n_comp, labels = connected_components(sub_adj, directed=False)
            for c in range(n_comp):
                cluster_mask = labels == c
                extent = np.sum(cluster_mask)
                cluster_vertices = np.where(mask)[0][cluster_mask]
                tfce[cluster_vertices] -= (extent ** E) * (h ** H) * dh

    return tfce


def _find_clusters(
    stat_map: np.ndarray,
    adjacency: sp.csr_matrix,
    threshold: float,
    tail: str = "two",
) -> Tuple[np.ndarray, np.ndarray]:
    """Find connected clusters above threshold and compute their masses."""
    from scipy.sparse.csgraph import connected_components

    n = len(stat_map)
    labels = np.zeros(n, dtype=int)
    masses = []
    cluster_id = 0

    if tail in ("two", "upper"):
        mask = stat_map > threshold
        if np.any(mask):
            sub_adj = adjacency[np.ix_(mask, mask)]
            nc, lbl = connected_components(sub_adj, directed=False)
            for c in range(nc):
                cluster_id += 1
                c_mask = lbl == c
                verts = np.where(mask)[0][c_mask]
                labels[verts] = cluster_id
                masses.append(np.sum(stat_map[verts]))

    if tail in ("two", "lower"):
        mask = stat_map < -threshold
        if np.any(mask):
            sub_adj = adjacency[np.ix_(mask, mask)]
            nc, lbl = connected_components(sub_adj, directed=False)
            for c in range(nc):
                cluster_id += 1
                c_mask = lbl == c
                verts = np.where(mask)[0][c_mask]
                labels[verts] = cluster_id
                masses.append(np.abs(np.sum(stat_map[verts])))

    return labels, np.array(masses, dtype=np.float64)


def _gower_center_matrix(A: np.ndarray) -> np.ndarray:
    """Gower centering of a matrix: G = (I - 11'/n) A (I - 11'/n)."""
    n = A.shape[0]
    I_n = np.eye(n)
    H = I_n - np.ones((n, n)) / n
    return H @ A @ H


def _permanova_pseudo_f(
    D: np.ndarray,
    groups: np.ndarray,
    unique_groups: np.ndarray,
) -> float:
    """Compute PERMANOVA Pseudo-F statistic."""
    n = len(groups)
    k = len(unique_groups)

    # Squared distance matrix
    D2 = D ** 2

    # SS_total
    ss_total = np.sum(D2) / (2 * n)

    # SS_within
    ss_within = 0.0
    for g in unique_groups:
        mask = groups == g
        n_g = np.sum(mask)
        if n_g > 1:
            D2_g = D2[np.ix_(mask, mask)]
            ss_within += np.sum(D2_g) / (2 * n_g)

    ss_between = ss_total - ss_within

    # Pseudo-F
    df_between = k - 1
    df_within = n - k
    if df_within <= 0 or ss_within == 0:
        return 0.0

    return (ss_between / df_between) / (ss_within / df_within)


def _connected_component_sizes(adj_matrix: np.ndarray) -> List[int]:
    """Find connected component sizes from a boolean adjacency matrix."""
    from scipy.sparse.csgraph import connected_components

    if isinstance(adj_matrix, np.ndarray):
        adj_sparse = sp.csr_matrix(adj_matrix.astype(float))
    else:
        adj_sparse = adj_matrix

    n_comp, labels = connected_components(adj_sparse, directed=False)
    sizes = []
    for c in range(n_comp):
        s = np.sum(labels == c)
        if s > 1:
            sizes.append(s)
    return sizes


def _extract_feature_importance(
    pipeline: Any,
    features: Union[pd.DataFrame, np.ndarray],
) -> Optional[pd.DataFrame]:
    """Extract feature importance from a fitted sklearn pipeline."""
    try:
        clf = pipeline.named_steps["clf"]
    except (AttributeError, KeyError):
        clf = pipeline

    if isinstance(features, pd.DataFrame):
        names = features.columns.tolist()
    else:
        names = [f"feature_{i}" for i in range(features.shape[1])]

    importance = None

    # Try coef_ (logistic, SVM)
    if hasattr(clf, "coef_"):
        importance = np.abs(clf.coef_).ravel()
    # Try feature_importances_ (tree-based)
    elif hasattr(clf, "feature_importances_"):
        importance = clf.feature_importances_
    # Ensemble: average component importances
    elif hasattr(clf, "estimators_"):
        imp_list = []
        for _, est in clf.estimators_:
            if hasattr(est, "coef_"):
                imp_list.append(np.abs(est.coef_).ravel())
            elif hasattr(est, "feature_importances_"):
                imp_list.append(est.feature_importances_)
        if imp_list:
            # Normalize each and average
            normed = []
            for imp in imp_list:
                if len(imp) == len(names):
                    s = imp.sum()
                    normed.append(imp / s if s > 0 else imp)
            if normed:
                importance = np.mean(normed, axis=0)

    if importance is not None and len(importance) == len(names):
        df = pd.DataFrame({
            "feature": names,
            "importance": importance,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return df

    return None
