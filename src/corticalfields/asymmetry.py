"""
Atlas-free cortical asymmetry quantification.

Provides continuous, parcellation-independent measures of hemispheric
asymmetry by combining two complementary mathematical frameworks:

  1. **Functional map asymmetry** — the off-diagonal energy of the
     inter-hemispheric C matrix, decomposable by spatial frequency band.
     This captures *how* the two hemispheres differ in their intrinsic
     geometry (global shape vs. fine gyrification).

  2. **Optimal transport asymmetry** — the Wasserstein distance between
     left and (mirrored) right hemisphere point clouds. This captures
     *how much* material must be moved to make the hemispheres coincide.

Both metrics are fully atlas-free: they require no parcellation, no
registration to a template, and no a priori definition of "left" and
"right" regions. They can be used as:

  - Scalar asymmetry scores for regression against clinical outcomes
  - Feature vectors (frequency-band decomposed) for classification
  - Distance matrices for kernel-based statistical inference (HSIC, MDMR)

Clinical application
--------------------
In MTLE-HS, focal cortical reorganisation (contralateral to the seizure
focus) creates frequency-specific asymmetry that the classical
AI = (L−R)/(L+R) per ROI cannot capture. The C matrix reveals whether
asymmetry is in low-frequency global shape (atrophy) or high-frequency
gyrification (dysplasia), providing a mechanistic decomposition.

References
----------
Liu, Y., et al. (2025). Systematic bias in FreeSurfer default parcellation
    for surface area asymmetry. Brain Structure and Function.
Ovsjanikov, M., et al. (2012). Functional maps. ACM TOG, 31(4).
Kolouri, S., et al. (2016). Sliced Wasserstein Kernels. CVPR.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class AsymmetryProfile:
    """
    Multi-scale asymmetry profile for a single subject.

    Parameters
    ----------
    subject_id : str
        Subject identifier.
    total_asymmetry : float
        Scalar asymmetry score (e.g. off-diagonal energy or Wasserstein).
    band_asymmetry : dict
        Frequency-band decomposed asymmetry scores.
    method : str
        Which method was used (``'functional_map'``, ``'wasserstein'``,
        ``'combined'``).
    hemi_lateralisation : str or None
        Side of epileptic focus (``'L'``, ``'R'``, or None).
    metadata : dict
        Additional computation details.
    """

    subject_id: str
    total_asymmetry: float
    band_asymmetry: Dict[str, float] = field(default_factory=dict)
    method: str = "functional_map"
    hemi_lateralisation: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def to_feature_vector(self) -> np.ndarray:
        """
        Convert to a flat feature vector for downstream ML/stats.

        Returns
        -------
        np.ndarray, shape (1 + n_bands,)
            [total_asymmetry, band_0, band_1, ...].
        """
        vals = [self.total_asymmetry] + list(self.band_asymmetry.values())
        return np.array(vals, dtype=np.float64)

    @property
    def feature_names(self) -> List[str]:
        """Names corresponding to :meth:`to_feature_vector` entries."""
        return ["total_asymmetry"] + list(self.band_asymmetry.keys())


# ═══════════════════════════════════════════════════════════════════════════
# Functional map-based asymmetry
# ═══════════════════════════════════════════════════════════════════════════


def asymmetry_from_functional_map(
    fm: "FunctionalMap",
    subject_id: str = "",
    bands: Optional[List[Tuple[int, int]]] = None,
    hemi_lateralisation: Optional[str] = None,
) -> AsymmetryProfile:
    """
    Compute asymmetry metrics from an inter-hemispheric functional map.

    The off-diagonal Frobenius energy of the C matrix is used as the
    primary asymmetry score, with optional decomposition by spectral
    frequency band.

    Parameters
    ----------
    fm : FunctionalMap
        Inter-hemispheric functional map (source=LH, target=RH).
    subject_id : str
        Subject identifier.
    bands : list of (start, end) or None
        Custom frequency bands. None uses default (low/mid/high).
    hemi_lateralisation : str or None
        Side of epileptic focus.

    Returns
    -------
    AsymmetryProfile
        Multi-scale asymmetry profile.

    Examples
    --------
    >>> from corticalfields.functional_maps import compute_interhemispheric_map
    >>> from corticalfields.asymmetry import asymmetry_from_functional_map
    >>> fm = compute_interhemispheric_map(lb_lh, lb_rh, k=50, k_final=200)
    >>> profile = asymmetry_from_functional_map(fm, "sub-01", hemi_lateralisation="L")
    >>> print(f"Total asymmetry: {profile.total_asymmetry:.4f}")
    >>> print(f"Low-frequency: {profile.band_asymmetry['low_freq']:.4f}")
    """
    total = fm.off_diagonal_energy
    band_dict = fm.frequency_band_energy(bands)

    return AsymmetryProfile(
        subject_id=subject_id,
        total_asymmetry=total,
        band_asymmetry=band_dict,
        method="functional_map",
        hemi_lateralisation=hemi_lateralisation,
        metadata={
            "diagonal_dominance": fm.diagonal_dominance,
            "k": min(fm.C.shape),
            "source_method": fm.metadata.get("method", "unknown"),
        },
    )


def asymmetry_from_wasserstein(
    lh_points: np.ndarray,
    rh_points: np.ndarray,
    subject_id: str = "",
    mirror_rh: bool = True,
    n_projections: int = 200,
    n_bands: int = 3,
    seed: int = 42,
    lh_features: Optional[np.ndarray] = None,
    rh_features: Optional[np.ndarray] = None,
    feature_weight: float = 1.0,
    hemi_lateralisation: Optional[str] = None,
) -> AsymmetryProfile:
    """
    Compute asymmetry from Wasserstein distance between hemisphere point clouds.

    Optionally decomposes by spatial scale using subsampled versions of the
    point clouds at different resolutions.

    Parameters
    ----------
    lh_points : np.ndarray, shape (N, 3)
        Left hemisphere 3D coordinates.
    rh_points : np.ndarray, shape (M, 3)
        Right hemisphere 3D coordinates.
    subject_id : str
        Subject identifier.
    mirror_rh : bool
        Mirror RH across YZ plane before computing distance.
    n_projections : int
        Number of random projections for sliced Wasserstein.
    n_bands : int
        Number of spatial-scale bands for decomposition.
    seed : int
        Random seed.
    lh_features, rh_features : np.ndarray or None
        Optional morphometric features for joint geometry-feature distance.
    feature_weight : float
        Relative weight of features vs coordinates.
    hemi_lateralisation : str or None
        Side of epileptic focus.

    Returns
    -------
    AsymmetryProfile
        Multi-scale asymmetry profile.
    """
    from corticalfields.transport import (
        sliced_wasserstein_distance,
        sliced_wasserstein_with_features,
    )
    from corticalfields.pointcloud import _farthest_point_sampling

    rh = rh_points.copy()
    if mirror_rh:
        rh[:, 0] *= -1

    # Total asymmetry
    if lh_features is not None and rh_features is not None:
        total = sliced_wasserstein_with_features(
            lh_points, rh,
            features_X=lh_features,
            features_Y=rh_features,
            feature_weight=feature_weight,
            n_projections=n_projections,
            seed=seed,
        )
    else:
        total = sliced_wasserstein_distance(
            lh_points, rh,
            n_projections=n_projections,
            seed=seed,
        )

    # Multi-scale decomposition: subsample at progressively lower resolution
    band_dict = {}
    n_points_bands = _compute_band_sizes(
        min(lh_points.shape[0], rh_points.shape[0]),
        n_bands,
    )

    for i, n_pts in enumerate(n_points_bands):
        lh_idx = _farthest_point_sampling(lh_points, n_pts, seed=seed)
        rh_idx = _farthest_point_sampling(rh, n_pts, seed=seed)

        d = sliced_wasserstein_distance(
            lh_points[lh_idx], rh[rh_idx],
            n_projections=n_projections,
            seed=seed,
        )
        band_name = f"scale_{i}" if n_bands <= 5 else f"scale_{n_pts}"
        band_dict[band_name] = d

    return AsymmetryProfile(
        subject_id=subject_id,
        total_asymmetry=total,
        band_asymmetry=band_dict,
        method="wasserstein",
        hemi_lateralisation=hemi_lateralisation,
        metadata={
            "n_projections": n_projections,
            "mirror_rh": mirror_rh,
            "n_lh": lh_points.shape[0],
            "n_rh": rh_points.shape[0],
        },
    )


# ═══════════════════════════════════════════════════════════════════════════
# Combined asymmetry (functional map + OT)
# ═══════════════════════════════════════════════════════════════════════════


def combined_asymmetry(
    fm_profile: AsymmetryProfile,
    ot_profile: AsymmetryProfile,
    weights: Optional[Dict[str, float]] = None,
) -> AsymmetryProfile:
    """
    Combine functional map and Wasserstein asymmetry into a unified profile.

    The two frameworks capture complementary aspects: functional maps
    measure *spectral* asymmetry (how eigenfunctions differ), while
    Wasserstein measures *spatial* asymmetry (how much mass moves).

    Parameters
    ----------
    fm_profile : AsymmetryProfile
        Functional map-based asymmetry.
    ot_profile : AsymmetryProfile
        Wasserstein-based asymmetry.
    weights : dict or None
        Relative weights for combining. Default: equal.

    Returns
    -------
    AsymmetryProfile
        Combined profile with all band scores from both methods.
    """
    if weights is None:
        weights = {"fm": 0.5, "ot": 0.5}

    # Normalise weights
    w_sum = weights.get("fm", 0.5) + weights.get("ot", 0.5)
    w_fm = weights.get("fm", 0.5) / w_sum
    w_ot = weights.get("ot", 0.5) / w_sum

    combined_total = w_fm * fm_profile.total_asymmetry + w_ot * ot_profile.total_asymmetry

    # Merge band dictionaries with prefixes
    band_dict = {}
    for k, v in fm_profile.band_asymmetry.items():
        band_dict[f"fm_{k}"] = v
    for k, v in ot_profile.band_asymmetry.items():
        band_dict[f"ot_{k}"] = v

    return AsymmetryProfile(
        subject_id=fm_profile.subject_id,
        total_asymmetry=combined_total,
        band_asymmetry=band_dict,
        method="combined",
        hemi_lateralisation=fm_profile.hemi_lateralisation,
        metadata={
            "weights": weights,
            "fm_total": fm_profile.total_asymmetry,
            "ot_total": ot_profile.total_asymmetry,
        },
    )


# ═══════════════════════════════════════════════════════════════════════════
# Cohort-level asymmetry analysis
# ═══════════════════════════════════════════════════════════════════════════


def cohort_asymmetry_matrix(
    profiles: List[AsymmetryProfile],
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Stack asymmetry profiles into a subjects × features matrix.

    Parameters
    ----------
    profiles : list of AsymmetryProfile
        One per subject.

    Returns
    -------
    X : np.ndarray, shape (N_subjects, N_features)
        Feature matrix for downstream analysis.
    subject_ids : list of str
        Subject identifiers (row labels).
    feature_names : list of str
        Feature names (column labels).
    """
    if not profiles:
        raise ValueError("Empty profiles list")

    feature_names = profiles[0].feature_names
    subject_ids = [p.subject_id for p in profiles]

    X = np.zeros((len(profiles), len(feature_names)), dtype=np.float64)
    for i, p in enumerate(profiles):
        X[i] = p.to_feature_vector()

    return X, subject_ids, feature_names


def asymmetry_group_comparison(
    patients: List[AsymmetryProfile],
    controls: List[AsymmetryProfile],
    test: str = "mann_whitney",
) -> Dict[str, Dict[str, float]]:
    """
    Compare asymmetry between patient and control groups.

    Parameters
    ----------
    patients : list of AsymmetryProfile
        Patient group.
    controls : list of AsymmetryProfile
        Control group.
    test : ``'mann_whitney'``, ``'t_test'``, or ``'permutation'``
        Statistical test.

    Returns
    -------
    dict
        Feature name → dict with 'statistic', 'p_value', 'effect_size'
        (Cohen's d for continuous features).
    """
    from scipy import stats

    # Stack feature vectors
    X_pat, _, feat_names = cohort_asymmetry_matrix(patients)
    X_ctrl, _, _ = cohort_asymmetry_matrix(controls)

    results = {}
    for j, name in enumerate(feat_names):
        pat_vals = X_pat[:, j]
        ctrl_vals = X_ctrl[:, j]

        if test == "mann_whitney":
            stat, pval = stats.mannwhitneyu(
                pat_vals, ctrl_vals, alternative="two-sided",
            )
        elif test == "t_test":
            stat, pval = stats.ttest_ind(pat_vals, ctrl_vals)
        elif test == "permutation":
            stat, pval = _permutation_test(pat_vals, ctrl_vals, n_perm=10000)
        else:
            raise ValueError(f"Unknown test: {test!r}")

        # Cohen's d
        pooled_std = np.sqrt(
            ((len(pat_vals) - 1) * pat_vals.std(ddof=1) ** 2 +
             (len(ctrl_vals) - 1) * ctrl_vals.std(ddof=1) ** 2) /
            (len(pat_vals) + len(ctrl_vals) - 2)
        )
        d = (pat_vals.mean() - ctrl_vals.mean()) / max(pooled_std, 1e-12)

        results[name] = {
            "statistic": float(stat),
            "p_value": float(pval),
            "effect_size_d": float(d),
            "mean_patients": float(pat_vals.mean()),
            "mean_controls": float(ctrl_vals.mean()),
        }

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Classical asymmetry index (for comparison)
# ═══════════════════════════════════════════════════════════════════════════


def classical_asymmetry_index(
    lh_values: np.ndarray,
    rh_values: np.ndarray,
    labels: Optional[np.ndarray] = None,
) -> Union[float, np.ndarray]:
    """
    Compute the classical ROI-based asymmetry index AI = (L−R) / (L+R).

    This is provided for **comparison** with the atlas-free methods.
    When labels are provided, computes AI per ROI; otherwise computes
    a single global AI from mean values.

    Parameters
    ----------
    lh_values : np.ndarray, shape (N,)
        Left hemisphere per-vertex values (e.g. thickness).
    rh_values : np.ndarray, shape (M,)
        Right hemisphere per-vertex values.
    labels : np.ndarray or None, shape (N,) or (N,) and (M,)
        Parcellation labels. If provided, returns per-ROI AI.

    Returns
    -------
    float or np.ndarray
        Scalar global AI, or per-ROI AI array.
    """
    if labels is None:
        L = np.nanmean(lh_values)
        R = np.nanmean(rh_values)
        denom = abs(L) + abs(R)
        return float((L - R) / denom) if denom > 1e-12 else 0.0

    roi_labels = np.sort(np.unique(labels[labels > 0]))
    ai = np.zeros(len(roi_labels), dtype=np.float64)
    for i, lab in enumerate(roi_labels):
        mask_l = labels == lab
        mask_r = labels == lab
        L = np.nanmean(lh_values[mask_l]) if mask_l.any() else 0.0
        R = np.nanmean(rh_values[mask_r]) if mask_r.any() else 0.0
        denom = abs(L) + abs(R)
        ai[i] = (L - R) / denom if denom > 1e-12 else 0.0

    return ai


# ═══════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════


def _compute_band_sizes(
    n_total: int,
    n_bands: int,
) -> List[int]:
    """
    Compute geometrically-spaced point counts for multi-scale analysis.

    Returns progressively coarser subsamplings from ~n_total to ~500 points.
    """
    min_pts = min(500, n_total)
    ratios = np.logspace(np.log10(min_pts), np.log10(n_total), n_bands)
    return [int(r) for r in ratios]


def _permutation_test(
    group_a: np.ndarray,
    group_b: np.ndarray,
    n_perm: int = 10000,
    seed: int = 42,
) -> Tuple[float, float]:
    """Two-sample permutation test on mean difference."""
    rng = np.random.default_rng(seed)
    observed = abs(group_a.mean() - group_b.mean())
    combined = np.concatenate([group_a, group_b])
    n_a = len(group_a)

    count = 0
    for _ in range(n_perm):
        rng.shuffle(combined)
        perm_diff = abs(combined[:n_a].mean() - combined[n_a:].mean())
        if perm_diff >= observed:
            count += 1

    return observed, (count + 1) / (n_perm + 1)
