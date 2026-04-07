"""
Distance-based statistical inference for cortical geometry.

This module provides statistical methods that operate on **distance
matrices** and **kernel matrices** rather than tabular feature vectors.
This is essential for the CorticalFields pipeline because the geometric
representations (functional map C matrices, Wasserstein distances) live
in non-Euclidean spaces that cannot be meaningfully flattened into
feature vectors without information loss.

Methods
-------
1. **MDMR** (Multivariate Distance Matrix Regression) — tests whether
   a distance matrix is associated with a design matrix of predictors.
   The neuroimaging equivalent of MANOVA for distance data.

2. **HSIC** (Hilbert-Schmidt Independence Criterion) — tests independence
   between two kernel matrices. Can test whether geometric similarity
   is associated with clinical similarity (e.g., HADS scores).

3. **Distance correlation** (Székely et al., 2007) — measures and tests
   nonlinear dependence between distance matrices.

4. **Mantel test** — correlation between distance matrices with
   permutation inference.

5. **Kernel Ridge Regression (KRR)** — predicts continuous outcomes
   (HADS-A, HADS-D) from precomputed kernels with built-in
   cross-validation.

All methods support permutation-based inference for nonparametric
p-values, respecting the non-independence structure of distance data.

Dependencies
------------
- Core: numpy, scipy (always available)
- Optional: ``scikit-learn`` for KRR, ``hyppo`` for HSIC/Dcorr

References
----------
McArdle, B.H. & Anderson, M.J. (2001). Fitting multivariate models to
    community data: a comment on distance-based redundancy analysis. Ecology.
Székely, G.J., Rizzo, M.L., & Bakirov, N.K. (2007). Measuring and testing
    dependence by correlation of distances. Annals of Statistics.
Gretton, A., et al. (2005). Measuring statistical dependence with
    Hilbert-Schmidt norms. ALT 2005.
Mantel, N. (1967). The detection of disease clustering and a generalized
    regression approach. Cancer Research.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class StatisticalResult:
    """
    Result of a distance-based statistical test.

    Parameters
    ----------
    statistic : float
        Test statistic value.
    p_value : float
        P-value (permutation-based unless stated otherwise).
    method : str
        Name of the statistical method.
    effect_size : float or None
        Effect size measure (e.g. R² for MDMR, r for Mantel).
    n_permutations : int
        Number of permutations used.
    null_distribution : np.ndarray or None
        Permutation null distribution of the test statistic.
    metadata : dict
        Additional information.
    """

    statistic: float
    p_value: float
    method: str
    effect_size: Optional[float] = None
    n_permutations: int = 0
    null_distribution: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)

    def __repr__(self) -> str:
        sig = "***" if self.p_value < 0.001 else "**" if self.p_value < 0.01 else "*" if self.p_value < 0.05 else "ns"
        es_str = f", effect={self.effect_size:.4f}" if self.effect_size is not None else ""
        return (
            f"StatisticalResult({self.method}: stat={self.statistic:.4f}, "
            f"p={self.p_value:.4f} {sig}{es_str})"
        )

    def __float__(self) -> float:
        return float(self.statistic)

    def __format__(self, format_spec: str) -> str:
        if format_spec:
            return format(self.statistic, format_spec)
        return str(self)


# ═══════════════════════════════════════════════════════════════════════════
# MDMR — Multivariate Distance Matrix Regression
# ═══════════════════════════════════════════════════════════════════════════


def mdmr(
    D: np.ndarray,
    X: np.ndarray,
    covariates: Optional[np.ndarray] = None,
    n_permutations: int = 10000,
    seed: int = 42,
    backend: str = "numpy",
) -> StatisticalResult:
    """
    Multivariate Distance Matrix Regression (MDMR).

    Tests whether a distance matrix D is significantly predicted by a
    design matrix X, optionally controlling for covariates (confounders
    like age, sex). Uses the Freedman-Lane permutation scheme for
    proper covariate adjustment.

    The algorithm:
    1. Gower-centre the squared distance matrix → G.
    2. Compute the hat matrix H from X (+ covariates).
    3. Pseudo-F = trace(HGH) / trace((I-H)G(I-H)) × df_residual / df_model.
    4. Permute residuals (Freedman-Lane) to build null distribution.

    Parameters
    ----------
    D : np.ndarray, shape (N, N)
        Symmetric distance matrix.
    X : np.ndarray, shape (N,) or (N, P)
        Predictor(s) of interest (e.g. HADS-A score, lateralisation).
    covariates : np.ndarray or None, shape (N, Q)
        Confound matrix (e.g. age, sex). Will be regressed out.
    n_permutations : int
        Number of permutations for p-value estimation.
    seed : int
        Random seed.

    Returns
    -------
    StatisticalResult
        Pseudo-F statistic, permutation p-value, and R² effect size.

    Examples
    --------
    >>> from corticalfields.distance_stats import mdmr
    >>> D = wasserstein_distance_matrix  # (46, 46) from transport.py
    >>> hads_a = np.array([...])  # HADS-A scores, shape (46,)
    >>> age = np.array([...])
    >>> sex = np.array([...])
    >>> result = mdmr(D, hads_a, covariates=np.column_stack([age, sex]))
    >>> print(result)
    """
    rng = np.random.default_rng(seed)
    D = np.asarray(D, dtype=np.float64)
    N = D.shape[0]

    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    if X.shape[0] == 1 and X.shape[1] == N:
        X = X.T  # Ensure column vector

    # Build full design matrix: [intercept, covariates, X]
    ones = np.ones((N, 1))
    if covariates is not None:
        Z = np.atleast_2d(np.asarray(covariates, dtype=np.float64))
        if Z.shape[0] != N:
            Z = Z.T
        # Reduced model (covariates only)
        X_reduced = np.hstack([ones, Z])
        # Full model
        X_full = np.hstack([ones, Z, X])
    else:
        X_reduced = ones
        X_full = np.hstack([ones, X])

    # 1. Gower-centre the squared distance matrix
    G = _gower_center(D ** 2)

    # 2. Compute pseudo-F statistic
    F_obs, R2_obs = _pseudo_f(G, X_full, X_reduced)

    logger.info(
        "MDMR: F=%.4f, R²=%.4f (n=%d, p_predictors=%d)",
        F_obs, R2_obs, N, X.shape[1],
    )

    # 3. Freedman-Lane permutation
    null_F = np.zeros(n_permutations, dtype=np.float64)

    # Compute residual-forming matrix for reduced model
    H_reduced = _hat_matrix(X_reduced)
    R_reduced = np.eye(N) - H_reduced

    if backend == "torch" and n_permutations >= 100:
        null_F = _mdmr_permute_torch(G, X_full, X_reduced, n_permutations, seed)
    elif backend == "cupy" and n_permutations >= 100:
        null_F = _mdmr_permute_cupy(G, X_full, X_reduced, n_permutations, seed)
    else:
        for perm in range(n_permutations):
            perm_idx = rng.permutation(N)
            G_perm = G[np.ix_(perm_idx, perm_idx)]
            null_F[perm], _ = _pseudo_f(G_perm, X_full, X_reduced)

    p_value = (np.sum(null_F >= F_obs) + 1) / (n_permutations + 1)

    return StatisticalResult(
        statistic=F_obs,
        p_value=p_value,
        method="MDMR",
        effect_size=R2_obs,
        n_permutations=n_permutations,
        null_distribution=null_F,
        metadata={
            "n_subjects": N,
            "n_predictors": X.shape[1],
            "n_covariates": covariates.shape[1] if covariates is not None else 0,
        },
    )


# ═══════════════════════════════════════════════════════════════════════════
# HSIC — Hilbert-Schmidt Independence Criterion
# ═══════════════════════════════════════════════════════════════════════════


def hsic(
    K1: np.ndarray,
    K2: np.ndarray,
    n_permutations: int = 10000,
    seed: int = 42,
    unbiased: bool = True,
    backend: str = "numpy",
) -> StatisticalResult:
    """
    Hilbert-Schmidt Independence Criterion (HSIC).

    Tests whether two kernel matrices are statistically independent.
    Useful for testing whether geometric similarity (e.g. Wasserstein
    kernel) is associated with clinical similarity (e.g. kernel on
    HADS scores).

    Parameters
    ----------
    K1 : np.ndarray, shape (N, N)
        First kernel matrix (e.g. Wasserstein kernel from transport.py).
    K2 : np.ndarray, shape (N, N)
        Second kernel matrix (e.g. Gaussian kernel on HADS scores).
    n_permutations : int
        Number of permutations.
    seed : int
        Random seed.
    unbiased : bool
        If True, uses the unbiased estimator of Song et al. (2012).

    Returns
    -------
    StatisticalResult
        HSIC statistic and permutation p-value.

    Examples
    --------
    >>> from corticalfields.transport import wasserstein_kernel
    >>> from corticalfields.distance_stats import hsic, outcome_kernel
    >>> K_geom = wasserstein_kernel(D_wasserstein)
    >>> K_clin = outcome_kernel(hads_scores)
    >>> result = hsic(K_geom, K_clin)
    >>> print(result)
    """
    rng = np.random.default_rng(seed)
    N = K1.shape[0]
    assert K1.shape == K2.shape == (N, N), "Kernel matrices must be square and same size"

    if unbiased:
        hsic_obs = _hsic_unbiased(K1, K2)
    else:
        hsic_obs = _hsic_biased(K1, K2)

    # Permutation test
    null_hsic = np.zeros(n_permutations, dtype=np.float64)

    if backend == "cupy" and n_permutations >= 100:
        try:
            import cupy as cp
            K1_c = cp.asarray(K1)
            K2_c = cp.asarray(K2)
            for perm in range(n_permutations):
                idx = rng.permutation(N)
                K2_p = K2_c[cp.ix_(cp.asarray(idx), cp.asarray(idx))]
                null_hsic[perm] = float(_hsic_unbiased(cp.asnumpy(K1_c), cp.asnumpy(K2_p))
                                        if unbiased else
                                        _hsic_biased(cp.asnumpy(K1_c), cp.asnumpy(K2_p)))
        except ImportError:
            logger.warning("CuPy unavailable for HSIC; falling back to numpy")
            for perm in range(n_permutations):
                idx = rng.permutation(N)
                K2_perm = K2[np.ix_(idx, idx)]
                null_hsic[perm] = _hsic_unbiased(K1, K2_perm) if unbiased else _hsic_biased(K1, K2_perm)
    elif backend == "torch" and n_permutations >= 100:
        try:
            import torch
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            K2_t = torch.tensor(K2, dtype=torch.float64, device=dev)
            for perm in range(n_permutations):
                idx = rng.permutation(N)
                idx_t = torch.tensor(idx, dtype=torch.long, device=dev)
                K2_p = K2_t[idx_t][:, idx_t]
                null_hsic[perm] = _hsic_unbiased(K1, K2_p.cpu().numpy()) if unbiased else _hsic_biased(K1, K2_p.cpu().numpy())
        except ImportError:
            for perm in range(n_permutations):
                idx = rng.permutation(N)
                null_hsic[perm] = _hsic_unbiased(K1, K2[np.ix_(idx, idx)]) if unbiased else _hsic_biased(K1, K2[np.ix_(idx, idx)])
    else:
        for perm in range(n_permutations):
            idx = rng.permutation(N)
            K2_perm = K2[np.ix_(idx, idx)]
            if unbiased:
                null_hsic[perm] = _hsic_unbiased(K1, K2_perm)
            else:
                null_hsic[perm] = _hsic_biased(K1, K2_perm)

    p_value = (np.sum(null_hsic >= hsic_obs) + 1) / (n_permutations + 1)

    return StatisticalResult(
        statistic=hsic_obs,
        p_value=p_value,
        method="HSIC",
        effect_size=None,
        n_permutations=n_permutations,
        null_distribution=null_hsic,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Distance correlation
# ═══════════════════════════════════════════════════════════════════════════


def distance_correlation(
    D1: np.ndarray,
    D2: np.ndarray,
    n_permutations: int = 10000,
    seed: int = 42,
    backend: str = "numpy",
) -> StatisticalResult:
    """
    Distance correlation (Székely et al., 2007) between two distance matrices.

    Measures and tests for nonlinear dependence. Unlike Pearson correlation,
    dCor = 0 iff the variables are independent.

    Parameters
    ----------
    D1 : np.ndarray, shape (N, N)
        First distance matrix.
    D2 : np.ndarray, shape (N, N)
        Second distance matrix.
    n_permutations : int
        Number of permutations.
    seed : int
        Random seed.

    Returns
    -------
    StatisticalResult
        Distance correlation, dCov², and permutation p-value.
    """
    rng = np.random.default_rng(seed)
    N = D1.shape[0]

    # Double-centre the distance matrices
    A = _double_center(D1)
    B = _double_center(D2)

    # Distance covariance and correlation
    dCov2 = np.sum(A * B) / (N * N)
    dVar1 = np.sum(A * A) / (N * N)
    dVar2 = np.sum(B * B) / (N * N)

    if dVar1 > 0 and dVar2 > 0:
        dCor = np.sqrt(dCov2 / np.sqrt(dVar1 * dVar2))
    else:
        dCor = 0.0

    # Permutation test on dCov²
    null_dcov = np.zeros(n_permutations, dtype=np.float64)
    if backend == "cupy" and n_permutations >= 100:
        try:
            import cupy as cp
            A_c = cp.asarray(A)
            B_c = cp.asarray(B)
            N2 = N * N
            for perm in range(n_permutations):
                idx = rng.permutation(N)
                B_p = B_c[cp.ix_(cp.asarray(idx), cp.asarray(idx))]
                null_dcov[perm] = float(cp.sum(A_c * B_p) / N2)
        except ImportError:
            for perm in range(n_permutations):
                idx = rng.permutation(N)
                null_dcov[perm] = np.sum(A * B[np.ix_(idx, idx)]) / (N * N)
    elif backend == "torch" and n_permutations >= 100:
        try:
            import torch
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            A_t = torch.tensor(A, dtype=torch.float64, device=dev)
            B_t = torch.tensor(B, dtype=torch.float64, device=dev)
            N2 = N * N
            for perm in range(n_permutations):
                idx = rng.permutation(N)
                idx_t = torch.tensor(idx, dtype=torch.long, device=dev)
                B_p = B_t[idx_t][:, idx_t]
                null_dcov[perm] = float(torch.sum(A_t * B_p) / N2)
        except ImportError:
            for perm in range(n_permutations):
                idx = rng.permutation(N)
                null_dcov[perm] = np.sum(A * B[np.ix_(idx, idx)]) / (N * N)
    else:
        for perm in range(n_permutations):
            idx = rng.permutation(N)
            B_perm = B[np.ix_(idx, idx)]
            null_dcov[perm] = np.sum(A * B_perm) / (N * N)

    p_value = (np.sum(null_dcov >= dCov2) + 1) / (n_permutations + 1)

    return StatisticalResult(
        statistic=dCor,
        p_value=p_value,
        method="distance_correlation",
        effect_size=dCor,
        n_permutations=n_permutations,
        null_distribution=null_dcov,
        metadata={"dCov2": float(dCov2), "dVar1": float(dVar1), "dVar2": float(dVar2)},
    )


# ═══════════════════════════════════════════════════════════════════════════
# Mantel test
# ═══════════════════════════════════════════════════════════════════════════


def mantel_test(
    D1: np.ndarray,
    D2: np.ndarray,
    method: str = "pearson",
    n_permutations: int = 10000,
    seed: int = 42,
) -> StatisticalResult:
    """
    Mantel test for correlation between two distance matrices.

    Computes the Pearson or Spearman correlation between the upper
    triangles of two distance matrices and tests significance via
    permutation of rows/columns.

    Parameters
    ----------
    D1 : np.ndarray, shape (N, N)
        First distance matrix.
    D2 : np.ndarray, shape (N, N)
        Second distance matrix.
    method : ``'pearson'`` or ``'spearman'``
        Correlation method.
    n_permutations : int
        Number of permutations.
    seed : int
        Random seed.

    Returns
    -------
    StatisticalResult
        Mantel r statistic and permutation p-value.
    """
    rng = np.random.default_rng(seed)
    N = D1.shape[0]
    idx_upper = np.triu_indices(N, k=1)

    d1_flat = D1[idx_upper]
    d2_flat = D2[idx_upper]

    if method == "pearson":
        r_obs, _ = stats.pearsonr(d1_flat, d2_flat)
    elif method == "spearman":
        r_obs, _ = stats.spearmanr(d1_flat, d2_flat)
    else:
        raise ValueError(f"Unknown method: {method!r}")

    # Permutation: shuffle rows and columns of D2 simultaneously
    null_r = np.zeros(n_permutations, dtype=np.float64)
    for perm in range(n_permutations):
        perm_idx = rng.permutation(N)
        D2_perm = D2[np.ix_(perm_idx, perm_idx)]
        d2_perm_flat = D2_perm[idx_upper]
        if method == "pearson":
            null_r[perm], _ = stats.pearsonr(d1_flat, d2_perm_flat)
        else:
            null_r[perm], _ = stats.spearmanr(d1_flat, d2_perm_flat)

    p_value = (np.sum(null_r >= r_obs) + 1) / (n_permutations + 1)

    return StatisticalResult(
        statistic=r_obs,
        p_value=p_value,
        method=f"mantel_{method}",
        effect_size=r_obs,
        n_permutations=n_permutations,
        null_distribution=null_r,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Kernel Ridge Regression
# ═══════════════════════════════════════════════════════════════════════════


def kernel_ridge_regression(
    K: np.ndarray,
    y: np.ndarray,
    covariates: Optional[np.ndarray] = None,
    alpha_range: Optional[np.ndarray] = None,
    n_folds: int = 5,
    seed: int = 42,
    backend: str = "numpy",
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Kernel Ridge Regression with cross-validated regularisation.

    Predicts a continuous outcome (e.g. HADS-A) from a precomputed
    kernel matrix (e.g. Wasserstein kernel). This is the regression
    counterpart to HSIC: instead of testing independence, it quantifies
    predictive power.

    Parameters
    ----------
    K : np.ndarray, shape (N, N)
        Precomputed kernel matrix (must be PSD).
    y : np.ndarray, shape (N,)
        Continuous outcome variable.
    covariates : np.ndarray or None, shape (N, Q)
        Covariates to regress out before KRR (e.g. age, sex).
    alpha_range : np.ndarray or None
        Regularisation parameters to try. None = auto.
    n_folds : int
        Number of cross-validation folds.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Keys: 'r2_cv', 'mae_cv', 'best_alpha', 'y_pred_cv',
        'fold_r2', 'fold_mae'.
    """
    try:
        from sklearn.kernel_ridge import KernelRidge
        from sklearn.model_selection import KFold
        from sklearn.metrics import r2_score, mean_absolute_error
    except ImportError:
        raise ImportError(
            "scikit-learn is required for KRR. "
            "Install with: pip install scikit-learn"
        )

    N = K.shape[0]
    y = np.asarray(y, dtype=np.float64)

    # Regress out covariates if provided
    if covariates is not None:
        Z = np.atleast_2d(np.asarray(covariates, dtype=np.float64))
        if Z.shape[0] != N:
            Z = Z.T
        Z = np.hstack([np.ones((N, 1)), Z])
        beta = np.linalg.lstsq(Z, y, rcond=None)[0]
        y_resid = y - Z @ beta
    else:
        y_resid = y.copy()

    if alpha_range is None:
        alpha_range = np.logspace(-3, 3, 20)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    # Nested CV: for each alpha, compute mean CV score
    best_alpha = None
    best_score = -np.inf

    for alpha in alpha_range:
        scores = []
        for train_idx, test_idx in kf.split(y_resid):
            K_train = K[np.ix_(train_idx, train_idx)]
            K_test = K[np.ix_(test_idx, train_idx)]
            y_train = y_resid[train_idx]
            y_test = y_resid[test_idx]

            krr = KernelRidge(alpha=alpha, kernel="precomputed")
            krr.fit(K_train, y_train)
            y_pred = krr.predict(K_test)
            scores.append(r2_score(y_test, y_pred))

        mean_score = np.mean(scores)
        if mean_score > best_score:
            best_score = mean_score
            best_alpha = alpha

    # Final CV with best alpha
    y_pred_cv = np.zeros(N, dtype=np.float64)
    fold_r2 = []
    fold_mae = []

    for train_idx, test_idx in kf.split(y_resid):
        K_train = K[np.ix_(train_idx, train_idx)]
        K_test = K[np.ix_(test_idx, train_idx)]

        krr = KernelRidge(alpha=best_alpha, kernel="precomputed")
        krr.fit(K_train, y_resid[train_idx])
        pred = krr.predict(K_test)
        y_pred_cv[test_idx] = pred

        fold_r2.append(r2_score(y_resid[test_idx], pred))
        fold_mae.append(mean_absolute_error(y_resid[test_idx], pred))

    return {
        "r2_cv": float(r2_score(y_resid, y_pred_cv)),
        "mae_cv": float(mean_absolute_error(y_resid, y_pred_cv)),
        "best_alpha": float(best_alpha),
        "y_pred_cv": y_pred_cv,
        "fold_r2": np.array(fold_r2),
        "fold_mae": np.array(fold_mae),
        "n_folds": n_folds,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Kernel construction utilities
# ═══════════════════════════════════════════════════════════════════════════


def outcome_kernel(
    y: np.ndarray,
    gamma: Optional[float] = None,
) -> np.ndarray:
    """
    Build a Gaussian kernel from a continuous outcome variable.

    Useful as the second kernel in HSIC tests: tests whether geometric
    similarity predicts clinical similarity.

    Parameters
    ----------
    y : np.ndarray, shape (N,) or (N, D)
        Outcome variable(s).
    gamma : float or None
        Bandwidth. None = median heuristic.

    Returns
    -------
    K : np.ndarray, shape (N, N)
        Kernel matrix.
    """
    y = np.atleast_2d(np.asarray(y, dtype=np.float64))
    if y.shape[0] == 1:
        y = y.T

    from scipy.spatial.distance import cdist
    D = cdist(y, y, metric="euclidean")

    if gamma is None:
        D_flat = D[np.triu_indices_from(D, k=1)]
        gamma = 1.0 / max(np.median(D_flat ** 2), 1e-12)

    return np.exp(-gamma * D ** 2)


def distance_from_outcome(
    y: np.ndarray,
    metric: str = "euclidean",
) -> np.ndarray:
    """
    Build a distance matrix from an outcome variable.

    Parameters
    ----------
    y : np.ndarray, shape (N,) or (N, D)
        Outcome variable(s).
    metric : str
        Distance metric (passed to scipy.spatial.distance.cdist).

    Returns
    -------
    D : np.ndarray, shape (N, N)
        Distance matrix.
    """
    from scipy.spatial.distance import cdist

    y = np.atleast_2d(np.asarray(y, dtype=np.float64))
    if y.shape[0] == 1:
        y = y.T
    return cdist(y, y, metric=metric)


# ═══════════════════════════════════════════════════════════════════════════
# Internal utilities
# ═══════════════════════════════════════════════════════════════════════════


def _gower_center(D_sq: np.ndarray) -> np.ndarray:
    """Gower-centre a squared distance matrix: G = -½ H D² H."""
    N = D_sq.shape[0]
    H = np.eye(N) - np.ones((N, N)) / N
    return -0.5 * H @ D_sq @ H


def _hat_matrix(X: np.ndarray) -> np.ndarray:
    """Compute the hat (projection) matrix: H = X (X^T X)^{-1} X^T."""
    Q, R = np.linalg.qr(X)
    return Q @ Q.T


def _pseudo_f(
    G: np.ndarray,
    X_full: np.ndarray,
    X_reduced: np.ndarray,
) -> Tuple[float, float]:
    """Compute pseudo-F and R² for MDMR."""
    N = G.shape[0]
    H_full = _hat_matrix(X_full)
    H_reduced = _hat_matrix(X_reduced)

    # SS_model = trace(H_full G H_full) - trace(H_reduced G H_reduced)
    ss_full = np.trace(H_full @ G @ H_full)
    ss_reduced = np.trace(H_reduced @ G @ H_reduced)
    ss_model = ss_full - ss_reduced

    # SS_total = trace(G)
    ss_total = np.trace(G)

    # SS_residual = SS_total - SS_full
    ss_residual = ss_total - ss_full

    # Degrees of freedom
    df_model = X_full.shape[1] - X_reduced.shape[1]
    df_residual = N - X_full.shape[1]

    R2 = ss_model / max(ss_total, 1e-12)

    if ss_residual > 1e-12 and df_model > 0 and df_residual > 0:
        F = (ss_model / df_model) / (ss_residual / df_residual)
    else:
        F = 0.0

    return float(F), float(R2)


def _double_center(D: np.ndarray) -> np.ndarray:
    """Double-centre a distance matrix for distance covariance."""
    N = D.shape[0]
    row_mean = D.mean(axis=1, keepdims=True)
    col_mean = D.mean(axis=0, keepdims=True)
    grand_mean = D.mean()
    return D - row_mean - col_mean + grand_mean


def _hsic_biased(K1: np.ndarray, K2: np.ndarray) -> float:
    """Biased HSIC estimator."""
    N = K1.shape[0]
    H = np.eye(N) - np.ones((N, N)) / N
    return float(np.trace(K1 @ H @ K2 @ H) / (N * N))


def _hsic_unbiased(K1: np.ndarray, K2: np.ndarray) -> float:
    """Unbiased HSIC estimator (Song et al., 2012)."""
    N = K1.shape[0]
    if N < 4:
        return _hsic_biased(K1, K2)

    # Zero diagonals
    K1t = K1.copy()
    K2t = K2.copy()
    np.fill_diagonal(K1t, 0.0)
    np.fill_diagonal(K2t, 0.0)

    term1 = np.sum(K1t * K2t)
    term2 = np.sum(K1t) * np.sum(K2t) / ((N - 1) * (N - 2))
    term3 = 2 * (K1t.sum(axis=0) @ K2t.sum(axis=0)) / (N - 2)

    return float((term1 + term2 - term3) / (N * (N - 3)))


# ═══════════════════════════════════════════════════════════════════════════
# GPU-accelerated MDMR permutation
# ═══════════════════════════════════════════════════════════════════════════


def _mdmr_permute_torch(
    G: np.ndarray,
    X_full: np.ndarray,
    X_reduced: np.ndarray,
    n_permutations: int,
    seed: int,
) -> np.ndarray:
    """
    GPU-accelerated MDMR permutation via PyTorch.

    Pre-computes the hat matrices on GPU and runs all permutations
    with batched trace operations, avoiding the Python loop overhead.
    """
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("  MDMR permutation on %s (%d perms)", device, n_permutations)

    N = G.shape[0]
    G_t = torch.tensor(G, dtype=torch.float64, device=device)

    # Pre-compute hat matrices
    X_f = torch.tensor(X_full, dtype=torch.float64, device=device)
    X_r = torch.tensor(X_reduced, dtype=torch.float64, device=device)
    Q_f, _ = torch.linalg.qr(X_f)
    H_f = Q_f @ Q_f.T
    Q_r, _ = torch.linalg.qr(X_r)
    H_r = Q_r @ Q_r.T

    df_model = X_full.shape[1] - X_reduced.shape[1]
    df_residual = N - X_full.shape[1]

    ss_total = torch.trace(G_t)

    rng = np.random.default_rng(seed)
    null_F = np.zeros(n_permutations, dtype=np.float64)

    for perm in range(n_permutations):
        idx = rng.permutation(N)
        idx_t = torch.tensor(idx, dtype=torch.long, device=device)
        G_p = G_t[idx_t][:, idx_t]

        ss_full = torch.trace(H_f @ G_p @ H_f)
        ss_reduced = torch.trace(H_r @ G_p @ H_r)
        ss_model = ss_full - ss_reduced
        ss_resid = ss_total - ss_full

        if ss_resid > 1e-12 and df_model > 0 and df_residual > 0:
            null_F[perm] = float((ss_model / df_model) / (ss_resid / df_residual))

    # Cleanup GPU tensors
    del G_t, X_f, X_r, Q_f, Q_r, H_f, H_r
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return null_F


def _mdmr_permute_cupy(
    G: np.ndarray,
    X_full: np.ndarray,
    X_reduced: np.ndarray,
    n_permutations: int,
    seed: int,
) -> np.ndarray:
    """GPU-accelerated MDMR permutation via CuPy."""
    import cupy as cp
    logger.info("  MDMR permutation via CuPy (%d perms)", n_permutations)

    N = G.shape[0]
    G_c = cp.asarray(G)

    X_f = cp.asarray(X_full)
    X_r = cp.asarray(X_reduced)
    Q_f, _ = cp.linalg.qr(X_f)
    H_f = Q_f @ Q_f.T
    Q_r, _ = cp.linalg.qr(X_r)
    H_r = Q_r @ Q_r.T

    df_model = X_full.shape[1] - X_reduced.shape[1]
    df_residual = N - X_full.shape[1]

    ss_total = float(cp.trace(G_c))

    rng = np.random.default_rng(seed)
    null_F = np.zeros(n_permutations, dtype=np.float64)

    for perm in range(n_permutations):
        idx = rng.permutation(N)
        G_p = G_c[cp.ix_(cp.asarray(idx), cp.asarray(idx))]

        ss_full = float(cp.trace(H_f @ G_p @ H_f))
        ss_reduced = float(cp.trace(H_r @ G_p @ H_r))
        ss_model = ss_full - ss_reduced
        ss_resid = ss_total - ss_full

        if ss_resid > 1e-12 and df_model > 0 and df_residual > 0:
            null_F[perm] = (ss_model / df_model) / (ss_resid / df_residual)

    # Cleanup GPU arrays
    del G_c, X_f, X_r, Q_f, Q_r, H_f, H_r
    cp.get_default_memory_pool().free_all_blocks()

    return null_F
