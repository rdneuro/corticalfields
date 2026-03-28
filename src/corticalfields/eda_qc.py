"""
Exploratory data analysis, quality control, and outlier detection.

Provides a complete QC/EDA pipeline for CorticalFields analyses,
covering three domains:

1. **Clinical EDA** — descriptive statistics, normality tests,
   correlations, and multivariate outlier detection (MCD-Mahalanobis)
   for HADS/age/sex/duration variables.

2. **Surface QC** — Euler number extraction, Weyl's law compliance,
   eigenvalue health checks, and spectral anomaly flags.

3. **Distance matrix QC** — row-sum screening, PCoA embedding for
   outlier visualisation, and leverage diagnostics for MDMR.

All functions return both data (DataFrames/dicts) and publication-ready
figures. Outlier decisions are tiered: Tier 1 (auto-exclude), Tier 2
(flag for review), Tier 3 (sensitivity analysis).

References
----------
- Rosen et al. (2018). Euler number QC. NeuroImage 169:407–418.
- Rousseeuw & Van Driessen (1999). MCD estimator.
- Allen et al. (2019). Raincloud plots. Wellcome Open Research.
- Nichols et al. (2017). COBIDAS guidelines. Nature Neuroscience.
- Pang et al. (2023). Geometric eigenmodes. Nature 618:566–574.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class QCReport:
    """
    Quality control report for a single subject or a cohort.

    Attributes
    ----------
    subject_id : str
    tier : int
        1 = auto-exclude, 2 = flag for review, 3 = retain (sensitivity).
    flags : list of str
        Human-readable flag descriptions.
    metrics : dict
        All QC metrics keyed by name.
    """
    subject_id: str = ""
    tier: int = 3  # default: retain
    flags: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def is_excluded(self) -> bool:
        return self.tier == 1

    def is_flagged(self) -> bool:
        return self.tier <= 2

    def summary(self) -> str:
        status = {1: "EXCLUDE", 2: "FLAG", 3: "OK"}[self.tier]
        flags_str = "; ".join(self.flags) if self.flags else "none"
        return f"[{status}] {self.subject_id}: {flags_str}"


@dataclass
class EDAResult:
    """
    Complete EDA result with tables, outlier masks, and figures.

    Attributes
    ----------
    descriptive : pd.DataFrame
        Descriptive statistics table.
    normality : pd.DataFrame
        Normality test results per variable.
    correlations : pd.DataFrame
        Pairwise correlation matrix.
    outlier_mask : np.ndarray
        Boolean array — True = outlier.
    outlier_details : pd.DataFrame
        Per-subject outlier diagnostics.
    qc_reports : list of QCReport
        Per-subject QC reports (for neuroimaging QC).
    """
    descriptive: pd.DataFrame = None
    normality: pd.DataFrame = None
    correlations: pd.DataFrame = None
    outlier_mask: np.ndarray = None
    outlier_details: pd.DataFrame = None
    qc_reports: List[QCReport] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# ███  CLINICAL EDA  ███
# ═══════════════════════════════════════════════════════════════════════════


def descriptive_statistics(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Comprehensive descriptive statistics for clinical variables.

    Computes mean, SD, median, IQR, range, skewness, kurtosis,
    and the Shapiro-Wilk normality test for each numeric column.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list of str or None
        Columns to analyse. None = all numeric.

    Returns
    -------
    pd.DataFrame
        One row per variable with descriptive stats.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    rows = []
    for col in columns:
        vals = df[col].dropna().values.astype(float)
        n = len(vals)
        if n < 3:
            continue

        q1, q3 = np.percentile(vals, [25, 75])
        sw_stat, sw_p = stats.shapiro(vals) if n >= 3 else (np.nan, np.nan)

        rows.append({
            "variable": col,
            "n": n,
            "mean": np.mean(vals),
            "sd": np.std(vals, ddof=1),
            "median": np.median(vals),
            "q1": q1,
            "q3": q3,
            "iqr": q3 - q1,
            "min": np.min(vals),
            "max": np.max(vals),
            "skewness": stats.skew(vals),
            "kurtosis": stats.kurtosis(vals),
            "shapiro_W": sw_stat,
            "shapiro_p": sw_p,
            "normal_5pct": "yes" if sw_p > 0.05 else "no",
        })

    return pd.DataFrame(rows)


def correlation_matrix(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "spearman",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pairwise correlation matrix with p-values.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list of str or None
    method : ``'pearson'`` or ``'spearman'``

    Returns
    -------
    r_matrix : pd.DataFrame
        Correlation coefficients.
    p_matrix : pd.DataFrame
        Two-tailed p-values.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    n = len(columns)
    r_mat = np.ones((n, n))
    p_mat = np.zeros((n, n))

    corr_fn = stats.spearmanr if method == "spearman" else stats.pearsonr

    for i in range(n):
        for j in range(i + 1, n):
            x = df[columns[i]].values.astype(float)
            y = df[columns[j]].values.astype(float)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() >= 3:
                r, p = corr_fn(x[mask], y[mask])
                r_mat[i, j] = r_mat[j, i] = r
                p_mat[i, j] = p_mat[j, i] = p

    r_df = pd.DataFrame(r_mat, index=columns, columns=columns)
    p_df = pd.DataFrame(p_mat, index=columns, columns=columns)
    return r_df, p_df


# ═══════════════════════════════════════════════════════════════════════════
# ███  OUTLIER DETECTION  ███
# ═══════════════════════════════════════════════════════════════════════════


def mad_outliers(
    x: np.ndarray,
    threshold: float = 3.5,
) -> np.ndarray:
    """
    Detect univariate outliers using Median Absolute Deviation (MAD).

    More robust than mean ± 3 SD for skewed data. Uses the Iglewicz &
    Hoaglin (1993) modified Z-score: |0.6745 × (x - median) / MAD|.

    Parameters
    ----------
    x : (N,) array
    threshold : float
        Modified Z-score threshold (3.5 recommended).

    Returns
    -------
    is_outlier : (N,) bool
    """
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad < 1e-12:
        return np.zeros(len(x), dtype=bool)
    modified_z = 0.6745 * (x - med) / mad
    return np.abs(modified_z) > threshold


def iqr_outliers(
    x: np.ndarray,
    factor: float = 1.5,
) -> np.ndarray:
    """
    Detect univariate outliers using the Tukey fence (IQR method).

    Parameters
    ----------
    x : (N,) array
    factor : float
        Fence multiplier (1.5 = standard, 3.0 = extreme).

    Returns
    -------
    is_outlier : (N,) bool
    """
    x = np.asarray(x, dtype=float)
    q1, q3 = np.nanpercentile(x, [25, 75])
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return (x < lower) | (x > upper)


def mcd_mahalanobis_outliers(
    X: np.ndarray,
    support_fraction: float = 0.75,
    chi2_percentile: float = 0.975,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Multivariate outlier detection via robust Mahalanobis distance (MCD).

    Uses the Minimum Covariance Determinant estimator from
    ``sklearn.covariance.MinCovDet`` for breakdown-resistant covariance
    estimation. Robust Mahalanobis distances are compared to the
    chi-squared threshold.

    Parameters
    ----------
    X : (N, P) array
        Data matrix (N subjects, P variables). NaN rows are excluded.
    support_fraction : float
        Fraction of data used for MCD (0.5–0.75 recommended).
    chi2_percentile : float
        Chi-squared threshold percentile (0.975 → ~97.5th percentile).

    Returns
    -------
    is_outlier : (N,) bool
        True for multivariate outliers.
    robust_distances : (N,) float
        Robust Mahalanobis distance per observation.
    """
    try:
        from sklearn.covariance import MinCovDet
    except ImportError:
        raise ImportError(
            "scikit-learn is required for MCD outlier detection. "
            "Install with: pip install scikit-learn"
        )

    X = np.asarray(X, dtype=float)
    valid = np.all(np.isfinite(X), axis=1)
    X_clean = X[valid]

    n, p = X_clean.shape
    mcd = MinCovDet(support_fraction=support_fraction, random_state=42)
    mcd.fit(X_clean)

    distances = np.full(X.shape[0], np.nan)
    distances[valid] = mcd.mahalanobis(X_clean)

    # Chi-squared threshold
    threshold = stats.chi2.ppf(chi2_percentile, df=p)
    is_outlier = distances > threshold

    logger.info(
        "MCD outlier detection: %d/%d flagged (threshold=%.2f, chi2 df=%d)",
        np.nansum(is_outlier), n, threshold, p,
    )
    return is_outlier, distances


def detect_clinical_outliers(
    df: pd.DataFrame,
    numeric_cols: List[str],
    method: str = "mcd",
    **kwargs,
) -> pd.DataFrame:
    """
    Run outlier detection on clinical variables and return annotated table.

    Combines univariate MAD screening per variable with multivariate
    MCD-Mahalanobis distance. Returns a DataFrame with per-subject
    outlier flags, distances, and the combined outlier decision.

    Parameters
    ----------
    df : pd.DataFrame
    numeric_cols : list of str
        Columns to include in outlier analysis.
    method : ``'mcd'``, ``'mad'``, ``'iqr'``, or ``'all'``
        Detection method. ``'all'`` runs all and flags if ANY fires.

    Returns
    -------
    pd.DataFrame
        Annotated DataFrame with outlier columns.
    """
    result = df.copy()

    # Univariate MAD per column
    for col in numeric_cols:
        vals = result[col].values.astype(float)
        result[f"{col}_mad_outlier"] = mad_outliers(vals, **kwargs)
        result[f"{col}_iqr_outlier"] = iqr_outliers(vals)

    # Multivariate MCD
    X = result[numeric_cols].values.astype(float)
    mask_valid = np.all(np.isfinite(X), axis=1)
    mcd_outlier, mcd_dist = mcd_mahalanobis_outliers(X[mask_valid])

    result["mcd_distance"] = np.nan
    result.loc[mask_valid, "mcd_distance"] = mcd_dist[mask_valid]
    result["mcd_outlier"] = False
    result.loc[mask_valid, "mcd_outlier"] = mcd_outlier[mask_valid]

    # Combined: outlier if MCD flags OR ≥2 univariate MAD flags
    mad_cols = [f"{c}_mad_outlier" for c in numeric_cols]
    result["n_univariate_flags"] = result[mad_cols].sum(axis=1)
    result["is_outlier"] = (
        result["mcd_outlier"] | (result["n_univariate_flags"] >= 2)
    )

    n_out = result["is_outlier"].sum()
    logger.info("Clinical outliers: %d/%d subjects flagged", n_out, len(df))
    return result


# ═══════════════════════════════════════════════════════════════════════════
# ███  FREESURFER SURFACE QC  ███
# ═══════════════════════════════════════════════════════════════════════════


def extract_euler_numbers(
    subjects_dir: Union[str, Path],
    subjects: List[str],
    hemis: List[str] = ("lh", "rh"),
) -> pd.DataFrame:
    """
    Extract Euler numbers from FreeSurfer ``orig.nofix`` surfaces.

    The Euler number is the strongest automated predictor of FreeSurfer
    surface quality (AUC 0.98–0.99; Rosen et al. 2018). More negative
    values indicate topological defects requiring more correction.

    Parameters
    ----------
    subjects_dir : path
    subjects : list of str
    hemis : list of str

    Returns
    -------
    pd.DataFrame
        Columns: subject, hemi, euler_number, n_holes.
    """
    subjects_dir = Path(subjects_dir)
    rows = []

    for sub in subjects:
        for hemi in hemis:
            surf_path = subjects_dir / sub / "surf" / f"{hemi}.orig.nofix"
            if not surf_path.exists():
                # Fallback to orig
                surf_path = subjects_dir / sub / "surf" / f"{hemi}.orig"

            if surf_path.exists():
                try:
                    result = subprocess.run(
                        ["mris_euler_number", str(surf_path)],
                        capture_output=True, text=True, timeout=30,
                    )
                    # Parse output: "euler number = X"
                    for line in result.stderr.split("\n") + result.stdout.split("\n"):
                        if "euler" in line.lower():
                            parts = line.split("=")
                            if len(parts) >= 2:
                                euler = int(parts[-1].strip().split()[0])
                                n_holes = (2 - euler) // 2
                                rows.append({
                                    "subject": sub, "hemi": hemi,
                                    "euler_number": euler, "n_holes": n_holes,
                                })
                                break
                except Exception as e:
                    logger.warning("Euler number failed for %s %s: %s", sub, hemi, e)
                    rows.append({
                        "subject": sub, "hemi": hemi,
                        "euler_number": np.nan, "n_holes": np.nan,
                    })
            else:
                rows.append({
                    "subject": sub, "hemi": hemi,
                    "euler_number": np.nan, "n_holes": np.nan,
                })

    return pd.DataFrame(rows)


def euler_number_qc(
    euler_df: pd.DataFrame,
    sd_exclude: float = 4.0,
    sd_flag: float = 2.0,
) -> List[QCReport]:
    """
    QC subjects based on Euler number thresholds (ENIGMA convention).

    Parameters
    ----------
    euler_df : pd.DataFrame
        From :func:`extract_euler_numbers`.
    sd_exclude : float
        Exclude if Euler number > this many SDs below mean.
    sd_flag : float
        Flag for review if between sd_flag and sd_exclude.

    Returns
    -------
    list of QCReport
    """
    reports = {}

    for hemi in euler_df["hemi"].unique():
        subset = euler_df[euler_df["hemi"] == hemi]
        vals = subset["euler_number"].dropna().values
        mean_e = np.mean(vals)
        std_e = np.std(vals, ddof=1)

        for _, row in subset.iterrows():
            sub = row["subject"]
            if sub not in reports:
                reports[sub] = QCReport(subject_id=sub)

            e = row["euler_number"]
            if np.isnan(e):
                reports[sub].tier = min(reports[sub].tier, 2)
                reports[sub].flags.append(f"{hemi}: Euler number unavailable")
                continue

            z = (e - mean_e) / max(std_e, 1e-12)
            reports[sub].metrics[f"euler_{hemi}"] = e
            reports[sub].metrics[f"euler_z_{hemi}"] = z

            if z < -sd_exclude:
                reports[sub].tier = 1
                reports[sub].flags.append(
                    f"{hemi}: Euler={e} ({z:.1f} SD, AUTO-EXCLUDE)"
                )
            elif z < -sd_flag:
                reports[sub].tier = min(reports[sub].tier, 2)
                reports[sub].flags.append(
                    f"{hemi}: Euler={e} ({z:.1f} SD, REVIEW)"
                )

    return list(reports.values())


# ═══════════════════════════════════════════════════════════════════════════
# ███  SPECTRAL QC  ███
# ═══════════════════════════════════════════════════════════════════════════


def weyls_law_check(
    eigenvalues: np.ndarray,
    surface_area: Optional[float] = None,
    tolerance: float = 0.3,
) -> Dict[str, Any]:
    """
    Verify Weyl's law compliance for LB eigenvalues.

    For a compact 2D manifold, λ_k ~ 4πk / Area(M). The slope
    in log-log space should be ~1.0. Deviations indicate mesh
    quality problems (holes, degenerate triangles, disconnected
    components).

    Parameters
    ----------
    eigenvalues : (K,) array
        Eigenvalues from LBO decomposition.
    surface_area : float or None
        If provided, computes expected slope.
    tolerance : float
        Maximum acceptable deviation from slope=1.0.

    Returns
    -------
    dict
        Keys: ``slope``, ``r_squared``, ``expected_slope``, ``passed``,
        ``n_negative``, ``first_nonzero_idx``.
    """
    evals = np.asarray(eigenvalues, dtype=float)

    # Count negative eigenvalues (should be 0 for PSD Laplacian)
    n_negative = int(np.sum(evals < -1e-10))

    # Find first nonzero eigenvalue
    first_nz = int(np.argmax(evals > 1e-10))

    # Fit log-log slope on positive eigenvalues
    mask = evals > 1e-10
    k = np.arange(len(evals))[mask]
    lam = evals[mask]

    if len(k) < 10:
        return {
            "slope": np.nan, "r_squared": np.nan,
            "expected_slope": 1.0, "passed": False,
            "n_negative": n_negative, "first_nonzero_idx": first_nz,
        }

    log_k = np.log(k + 1)
    log_lam = np.log(lam)
    coeffs = np.polyfit(log_k, log_lam, 1)
    slope = coeffs[0]

    # R² of fit
    predicted = np.polyval(coeffs, log_k)
    ss_res = np.sum((log_lam - predicted) ** 2)
    ss_tot = np.sum((log_lam - np.mean(log_lam)) ** 2)
    r2 = 1 - ss_res / max(ss_tot, 1e-12)

    passed = (abs(slope - 1.0) < tolerance) and (n_negative == 0) and (r2 > 0.95)

    result = {
        "slope": float(slope),
        "r_squared": float(r2),
        "expected_slope": 1.0,
        "passed": passed,
        "n_negative": n_negative,
        "first_nonzero_idx": first_nz,
    }

    if surface_area is not None:
        result["expected_slope_4pi_over_area"] = 4 * np.pi / surface_area

    return result


def spectral_qc_cohort(
    eigenvalues_list: List[np.ndarray],
    subjects: List[str],
    weyl_tolerance: float = 0.3,
) -> Tuple[pd.DataFrame, List[QCReport]]:
    """
    Run Weyl's law QC on eigenvalues for all subjects.

    Parameters
    ----------
    eigenvalues_list : list of (K,) arrays
    subjects : list of str
    weyl_tolerance : float

    Returns
    -------
    df : pd.DataFrame
        Weyl's law metrics per subject.
    reports : list of QCReport
    """
    rows = []
    reports = []

    for sub, evals in zip(subjects, eigenvalues_list):
        result = weyls_law_check(evals, tolerance=weyl_tolerance)
        result["subject"] = sub
        rows.append(result)

        qc = QCReport(subject_id=sub)
        qc.metrics.update(result)

        if result["n_negative"] > 0:
            qc.tier = 1
            qc.flags.append(f"Negative eigenvalues: {result['n_negative']}")
        elif not result["passed"]:
            qc.tier = 2
            qc.flags.append(
                f"Weyl's law: slope={result['slope']:.2f} "
                f"(expected 1.0, R²={result['r_squared']:.3f})"
            )
        reports.append(qc)

    return pd.DataFrame(rows), reports


# ═══════════════════════════════════════════════════════════════════════════
# ███  DISTANCE MATRIX QC  ███
# ═══════════════════════════════════════════════════════════════════════════


def distance_matrix_outliers(
    D: np.ndarray,
    subjects: Optional[List[str]] = None,
    sd_threshold: float = 3.0,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Screen distance matrix for subject-level outliers via row sums.

    Subjects with extreme mean distance to all others are candidate
    outliers — they may have poor surface quality or atypical anatomy.

    Parameters
    ----------
    D : (N, N) array
        Symmetric distance matrix.
    subjects : list of str or None
    sd_threshold : float
        Flag if row mean > this many SDs above the cohort mean.

    Returns
    -------
    is_outlier : (N,) bool
    details : pd.DataFrame
        Per-subject row mean, z-score, outlier flag.
    """
    D = np.asarray(D, dtype=float)
    N = D.shape[0]

    row_means = np.nanmean(D + np.eye(N) * np.nan, axis=1)
    mean_rm = np.nanmean(row_means)
    std_rm = np.nanstd(row_means, ddof=1)

    z_scores = (row_means - mean_rm) / max(std_rm, 1e-12)
    is_outlier = np.abs(z_scores) > sd_threshold

    if subjects is None:
        subjects = [f"S{i:03d}" for i in range(N)]

    details = pd.DataFrame({
        "subject": subjects,
        "row_mean_distance": row_means,
        "z_score": z_scores,
        "is_outlier": is_outlier,
    })

    logger.info("Distance matrix QC: %d/%d outliers (threshold=%.1f SD)",
                is_outlier.sum(), N, sd_threshold)
    return is_outlier, details


def pcoa_embedding(
    D: np.ndarray,
    n_components: int = 3,
    correction: str = "lingoes",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Principal Coordinates Analysis (classical MDS) on a distance matrix.

    Embeds the distance matrix into Euclidean space for visualisation
    and outlier inspection. Applies Lingoes or Cailliez correction for
    non-Euclidean distances (negative eigenvalues).

    Parameters
    ----------
    D : (N, N) array
        Symmetric distance matrix.
    n_components : int
        Number of embedding dimensions.
    correction : ``'lingoes'``, ``'cailliez'``, or ``'none'``

    Returns
    -------
    coords : (N, n_components) array
        Embedded coordinates.
    eigenvalues : array
        All eigenvalues of the double-centred matrix.
    """
    D = np.asarray(D, dtype=float)
    N = D.shape[0]

    # Double-centre
    D_sq = D ** 2
    H = np.eye(N) - np.ones((N, N)) / N
    B = -0.5 * H @ D_sq @ H

    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(-eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Correction for negative eigenvalues
    if eigvals[-1] < -1e-10:
        if correction == "lingoes":
            c = np.abs(eigvals[-1])
            D_corrected = np.sqrt(D_sq + 2 * c * (1 - np.eye(N)))
            D_sq2 = D_corrected ** 2
            B = -0.5 * H @ D_sq2 @ H
            eigvals, eigvecs = np.linalg.eigh(B)
            idx = np.argsort(-eigvals)
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]
            logger.info("PCoA: Lingoes correction applied (c=%.4f)", c)
        elif correction == "cailliez":
            logger.warning("Cailliez correction not implemented; using Lingoes")

    # Extract coordinates
    pos_mask = eigvals > 1e-10
    n_pos = min(n_components, np.sum(pos_mask))
    coords = eigvecs[:, :n_pos] * np.sqrt(eigvals[:n_pos])

    # Pad if needed
    if coords.shape[1] < n_components:
        pad = np.zeros((N, n_components - coords.shape[1]))
        coords = np.hstack([coords, pad])

    return coords, eigvals


# ═══════════════════════════════════════════════════════════════════════════
# ███  COMBINED EDA PIPELINE  ███
# ═══════════════════════════════════════════════════════════════════════════


def run_clinical_eda(
    df: pd.DataFrame,
    numeric_cols: List[str],
    subject_col: str = "subject",
    group_col: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> EDAResult:
    """
    Run a comprehensive clinical EDA and save all tables and figures.

    Performs descriptive statistics, normality tests, correlation matrix,
    univariate MAD outlier detection, and multivariate MCD screening.
    Optionally saves all results to disk.

    Parameters
    ----------
    df : pd.DataFrame
        Clinical DataFrame.
    numeric_cols : list of str
        Columns to include in EDA.
    subject_col : str
        Name of the subject ID column.
    group_col : str or None
        Optional grouping variable for stratified EDA.
    output_dir : path or None
        If provided, saves all tables and figures to this directory.

    Returns
    -------
    EDAResult
        Complete EDA results.
    """
    result = EDAResult()

    # 1. Descriptive statistics
    result.descriptive = descriptive_statistics(df, numeric_cols)
    logger.info("Descriptive statistics computed for %d variables", len(numeric_cols))

    # 2. Correlation matrix
    r_mat, p_mat = correlation_matrix(df, numeric_cols)
    result.correlations = r_mat

    # 3. Outlier detection
    outlier_df = detect_clinical_outliers(df, numeric_cols)
    result.outlier_mask = outlier_df["is_outlier"].values
    result.outlier_details = outlier_df

    # 4. Normality tests
    result.normality = result.descriptive[
        ["variable", "n", "skewness", "kurtosis", "shapiro_W", "shapiro_p", "normal_5pct"]
    ].copy()

    # Save to disk
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        result.descriptive.to_csv(out / "eda_descriptive.csv", index=False)
        result.normality.to_csv(out / "eda_normality.csv", index=False)
        r_mat.to_csv(out / "eda_correlations_r.csv")
        p_mat.to_csv(out / "eda_correlations_p.csv")
        outlier_df.to_csv(out / "eda_outlier_details.csv", index=False)

        logger.info("EDA tables saved to %s", out)

    return result


def run_spectral_eda(
    eigenvalues_dict: Dict[str, List[np.ndarray]],
    subjects: List[str],
    output_dir: Optional[Union[str, Path]] = None,
) -> Tuple[pd.DataFrame, List[QCReport]]:
    """
    Run spectral QC on all subjects' eigenvalues.

    Parameters
    ----------
    eigenvalues_dict : dict
        ``{"lh": [evals_sub1, ...], "rh": [evals_sub1, ...]}``
    subjects : list of str
    output_dir : path or None

    Returns
    -------
    df : pd.DataFrame
        Combined Weyl's law QC table.
    reports : list of QCReport
    """
    all_rows = []
    all_reports = {}

    for hemi in ["lh", "rh"]:
        if hemi not in eigenvalues_dict:
            continue
        df_h, reports_h = spectral_qc_cohort(
            eigenvalues_dict[hemi], subjects,
        )
        df_h["hemi"] = hemi
        all_rows.append(df_h)

        for r in reports_h:
            if r.subject_id not in all_reports:
                all_reports[r.subject_id] = r
            else:
                existing = all_reports[r.subject_id]
                existing.tier = min(existing.tier, r.tier)
                existing.flags.extend(r.flags)
                existing.metrics.update(r.metrics)

    df = pd.concat(all_rows, ignore_index=True)

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "eda_spectral_qc.csv", index=False)

    return df, list(all_reports.values())


# ═══════════════════════════════════════════════════════════════════════════
# ███  VISUALIZATION  ███
# ═══════════════════════════════════════════════════════════════════════════


def plot_eda_clinical(
    df: pd.DataFrame,
    numeric_cols: List[str],
    group_col: Optional[str] = None,
    outlier_mask: Optional[np.ndarray] = None,
    figsize: Optional[Tuple[float, float]] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> "Figure":
    """
    Multi-panel clinical EDA figure: distributions, box plots, outliers.

    Creates a publication-grade figure with violin plots + individual
    data points (raincloud-style) for each clinical variable, with
    outliers highlighted.

    Parameters
    ----------
    df : pd.DataFrame
    numeric_cols : list of str
    group_col : str or None
    outlier_mask : (N,) bool or None
    figsize, output_path : standard

    Returns
    -------
    matplotlib Figure
    """
    import matplotlib.pyplot as plt

    n_vars = len(numeric_cols)
    if figsize is None:
        figsize = (min(4 * n_vars, 18), 5)

    fig, axes = plt.subplots(1, n_vars, figsize=figsize)
    if n_vars == 1:
        axes = [axes]

    for ax, col in zip(axes, numeric_cols):
        vals = df[col].dropna().values

        # Violin
        parts = ax.violinplot(vals, positions=[0], showmeans=False,
                               showmedians=True, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor("#0072B2")
            pc.set_alpha(0.3)

        # Jittered points
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
        colors = np.full(len(vals), "#0072B2")
        sizes = np.full(len(vals), 20.0)
        if outlier_mask is not None:
            mask_valid = df[col].notna().values
            out_m = outlier_mask[mask_valid]
            colors = np.where(out_m, "#D55E00", "#0072B2")
            sizes = np.where(out_m, 60.0, 20.0)

        ax.scatter(jitter, vals, c=colors, s=sizes, alpha=0.7,
                   edgecolors="white", linewidths=0.3, zorder=5)

        # Box plot overlay
        bp = ax.boxplot(vals, positions=[0], widths=0.3, patch_artist=True,
                        showfliers=False, zorder=3)
        bp["boxes"][0].set_facecolor("none")
        bp["boxes"][0].set_edgecolor("#333")
        bp["medians"][0].set_color("#D55E00")

        ax.set_title(col.replace("_", " ").title(), fontsize=9)
        ax.set_xticks([])
        ax.tick_params(labelsize=7)

    plt.tight_layout()

    if output_path:
        from corticalfields.brainplots import save_figure
        save_figure(fig, output_path)

    return fig


def plot_correlation_heatmap(
    r_matrix: pd.DataFrame,
    p_matrix: Optional[pd.DataFrame] = None,
    title: str = "Correlation matrix",
    figsize: Optional[Tuple[float, float]] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> "Figure":
    """
    Correlation heatmap with significance annotations.

    Parameters
    ----------
    r_matrix : pd.DataFrame
    p_matrix : pd.DataFrame or None
    title : str
    figsize, output_path : standard

    Returns
    -------
    matplotlib Figure
    """
    import matplotlib.pyplot as plt

    n = len(r_matrix)
    if figsize is None:
        figsize = (max(6, n * 0.6), max(5, n * 0.5))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(r_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")

    # Annotations
    for i in range(n):
        for j in range(n):
            r = r_matrix.values[i, j]
            text = f"{r:.2f}"
            if p_matrix is not None and i != j:
                p = p_matrix.values[i, j]
                if p < 0.001:
                    text += "***"
                elif p < 0.01:
                    text += "**"
                elif p < 0.05:
                    text += "*"
            color = "white" if abs(r) > 0.5 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=6, color=color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    labels = [c.replace("_", "\n") for c in r_matrix.columns]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(title, fontsize=10, fontweight="bold")

    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label(r"Spearman $\rho$", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    plt.tight_layout()

    if output_path:
        from corticalfields.brainplots import save_figure
        save_figure(fig, output_path)

    return fig


def plot_pcoa_embedding(
    D: np.ndarray,
    subjects: Optional[List[str]] = None,
    group_labels: Optional[np.ndarray] = None,
    group_names: Optional[List[str]] = None,
    outcome: Optional[np.ndarray] = None,
    outcome_name: str = "",
    outlier_mask: Optional[np.ndarray] = None,
    title: str = "PCoA embedding of distance matrix",
    figsize: Optional[Tuple[float, float]] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> "Figure":
    """
    PCoA scatter plot for distance matrix visualisation.

    Plots the first 2 principal coordinates, coloured by group or
    outcome, with outliers highlighted by red borders.

    Parameters
    ----------
    D : (N, N) array
    subjects, group_labels, group_names : optional identifiers
    outcome : (N,) float or None
        Continuous variable for colour mapping (e.g. HADS-A).
    outcome_name : str
    outlier_mask : (N,) bool or None
    title, figsize, output_path : standard

    Returns
    -------
    matplotlib Figure
    """
    import matplotlib.pyplot as plt

    coords, eigvals = pcoa_embedding(D, n_components=2)
    var_explained = eigvals[:2] / np.abs(eigvals).sum() * 100

    if figsize is None:
        figsize = (6, 5)

    fig, ax = plt.subplots(figsize=figsize)

    # Colour by outcome or group
    if outcome is not None:
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=outcome, cmap="YlOrRd",
                        s=40, alpha=0.8, edgecolors="white", linewidths=0.5)
        cb = fig.colorbar(sc, ax=ax, shrink=0.8)
        cb.set_label(outcome_name, fontsize=8)
    elif group_labels is not None:
        colors_map = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
        colors = [colors_map[int(g) % len(colors_map)] for g in group_labels]
        ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=40, alpha=0.8,
                   edgecolors="white", linewidths=0.5)
        if group_names:
            import matplotlib.patches as mpatches
            patches = [mpatches.Patch(color=colors_map[i], label=name)
                       for i, name in enumerate(group_names)]
            ax.legend(handles=patches, fontsize=7, loc="best")
    else:
        ax.scatter(coords[:, 0], coords[:, 1], c="#0072B2", s=40, alpha=0.8,
                   edgecolors="white", linewidths=0.5)

    # Highlight outliers
    if outlier_mask is not None:
        out_idx = np.where(outlier_mask)[0]
        ax.scatter(coords[out_idx, 0], coords[out_idx, 1],
                   facecolors="none", edgecolors="#D55E00", linewidths=2,
                   s=120, zorder=10, label="Outlier")

    ax.set_xlabel(f"PCoA 1 ({var_explained[0]:.1f}%)", fontsize=8)
    ax.set_ylabel(f"PCoA 2 ({var_explained[1]:.1f}%)", fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if output_path:
        from corticalfields.brainplots import save_figure
        save_figure(fig, output_path)

    return fig


def plot_weyl_law_cohort(
    eigenvalues_list: List[np.ndarray],
    subjects: List[str],
    n_show: int = 100,
    title: str = "Weyl's law compliance (cohort)",
    figsize: Optional[Tuple[float, float]] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> "Figure":
    """
    Cohort-level Weyl's law diagnostic: overlay all subjects' spectra.

    Parameters
    ----------
    eigenvalues_list : list of (K,) arrays
    subjects : list of str
    n_show : int
    title, figsize, output_path : standard

    Returns
    -------
    matplotlib Figure
    """
    import matplotlib.pyplot as plt

    if figsize is None:
        figsize = (7, 4)

    fig, ax = plt.subplots(figsize=figsize)

    for i, (evals, sub) in enumerate(zip(eigenvalues_list, subjects)):
        ev = evals[:n_show]
        mask = ev > 1e-10
        k = np.arange(len(ev))[mask]
        ax.loglog(k + 1, ev[mask], "-", alpha=0.25, linewidth=0.6,
                  color="#0072B2")

    # Reference line: slope = 1
    k_ref = np.arange(1, n_show + 1)
    median_lam1 = np.median([e[1] for e in eigenvalues_list if len(e) > 1])
    ref_line = median_lam1 * k_ref
    ax.loglog(k_ref, ref_line, "--", color="#D55E00", linewidth=1.5,
              label="Weyl reference (slope=1)")

    ax.set_xlabel("Eigenvalue index $k$", fontsize=8)
    ax.set_ylabel(r"$\lambda_k$", fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2, which="both")
    ax.tick_params(labelsize=7)

    plt.tight_layout()

    if output_path:
        from corticalfields.brainplots import save_figure
        save_figure(fig, output_path)

    return fig


# ═══════════════════════════════════════════════════════════════════════════
# ███  MIDTHICKNESS SURFACE GENERATION  ███
# ═══════════════════════════════════════════════════════════════════════════


def generate_midthickness(
    subjects_dir: Union[str, Path],
    subject_id: str,
    hemi: str = "lh",
    method: str = "average",
) -> Path:
    """
    Generate a midthickness surface if it doesn't already exist.

    Two methods are supported: arithmetic average of white and pial
    vertex coordinates (HCP convention), or FreeSurfer's ``mris_expand``
    which expands white by half the local cortical thickness.

    Parameters
    ----------
    subjects_dir : path
    subject_id : str
    hemi : ``'lh'`` or ``'rh'``
    method : ``'average'`` or ``'expand'``

    Returns
    -------
    Path
        Path to the midthickness surface file.
    """
    import nibabel as nib

    base = Path(subjects_dir) / subject_id / "surf"
    midthick_path = base / f"{hemi}.midthickness"

    if midthick_path.exists():
        return midthick_path

    # Also check alternative name
    graymid_path = base / f"{hemi}.graymid"
    if graymid_path.exists():
        return graymid_path

    white_path = base / f"{hemi}.white"
    pial_path = base / f"{hemi}.pial"

    if not white_path.exists() or not pial_path.exists():
        raise FileNotFoundError(
            f"Need both white and pial surfaces to generate midthickness. "
            f"Missing: {white_path if not white_path.exists() else pial_path}"
        )

    if method == "average":
        # HCP convention: arithmetic average of vertex coordinates
        white_coords, faces = nib.freesurfer.read_geometry(str(white_path))
        pial_coords, _ = nib.freesurfer.read_geometry(str(pial_path))
        mid_coords = (white_coords + pial_coords) / 2.0
        nib.freesurfer.write_geometry(str(midthick_path), mid_coords, faces)
        logger.info("Generated midthickness surface: %s (average method)", midthick_path)
    elif method == "expand":
        # FreeSurfer mris_expand
        try:
            subprocess.run(
                ["mris_expand", "-thickness", str(white_path), "0.5",
                 str(midthick_path)],
                check=True, capture_output=True, timeout=120,
            )
            logger.info("Generated midthickness surface: %s (expand method)", midthick_path)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning("mris_expand failed, falling back to average: %s", e)
            return generate_midthickness(
                subjects_dir, subject_id, hemi, method="average",
            )
    else:
        raise ValueError(f"Unknown method: {method!r}")

    return midthick_path
