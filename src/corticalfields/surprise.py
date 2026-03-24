"""
Information-theoretic surprise and anomaly maps on cortical surfaces.

This module computes vertex-wise anomaly scores from normative model
predictions. It goes beyond simple z-scores by providing:

    1. **Surprise** — negative log-predictive density:
       S(x) = −log p(y_obs(x) | x, model)
       This is the theoretically optimal anomaly score under the
       assumption that the GP model is well-calibrated.

    2. **Excess surprise** — surprise relative to the expected surprise
       under the normative distribution. Under a calibrated Gaussian
       model, E[surprise] = ½(1 + log 2πσ²), so excess surprise is:
       ΔS(x) = S(x) − ½(1 + log 2πσ²(x))

    3. **Aggregate network scores** — surprise aggregated by cortical
       parcellation (e.g. Yeo-7, Schaefer-200, Desikan-Killiany),
       enabling statements like "this patient has anomalous DMN
       structure" without ever needing fMRI data.

    4. **Bayesian anomaly probability** — the posterior probability
       that a vertex is anomalous under a two-component mixture model
       (normal + anomalous).

Mathematical foundations
------------------------
For a GP normative model with posterior predictive
p(y* | x, D) = N(μ(x), σ²(x)), the surprise at vertex x is:

    S(x) = −log p(y_obs | x, D)
         = ½ log(2πσ²(x)) + (y_obs(x) − μ(x))² / (2σ²(x))

This decomposes into an uncertainty term and a squared deviation term.
Large surprise means either the vertex has high uncertainty (model
knows it's a difficult region) or the observation deviates strongly
from the prediction, or both.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class SurpriseMap:
    """
    Container for vertex-wise anomaly scores with aggregation methods.

    Attributes
    ----------
    surprise : np.ndarray, shape (N,)
        Pointwise negative log-predictive density.
    excess_surprise : np.ndarray, shape (N,)
        Surprise minus expected surprise under the null.
    z_score : np.ndarray, shape (N,)
        Standardised deviation from normative prediction.
    anomaly_probability : np.ndarray, shape (N,)
        Posterior probability of being anomalous (mixture model).
    vertex_mask : np.ndarray, shape (N,), dtype bool
        Valid vertices (non-NaN, non-medial-wall).
    """

    surprise: np.ndarray
    excess_surprise: np.ndarray
    z_score: np.ndarray
    anomaly_probability: np.ndarray
    vertex_mask: np.ndarray

    @property
    def n_vertices(self) -> int:
        return len(self.surprise)

    @property
    def n_valid(self) -> int:
        return int(self.vertex_mask.sum())

    def threshold(
        self,
        z_thresh: float = 2.0,
        direction: str = "both",
    ) -> np.ndarray:
        """
        Binary anomaly mask based on z-score threshold.

        Parameters
        ----------
        z_thresh : float
            Absolute z-score threshold.
        direction : ``'both'``, ``'negative'``, ``'positive'``
            Direction of anomaly.

        Returns
        -------
        mask : np.ndarray, shape (N,), dtype bool
        """
        if direction == "negative":
            return self.vertex_mask & (self.z_score < -z_thresh)
        elif direction == "positive":
            return self.vertex_mask & (self.z_score > z_thresh)
        else:
            return self.vertex_mask & (np.abs(self.z_score) > z_thresh)

    def aggregate_by_parcellation(
        self,
        labels: np.ndarray,
        label_names: Optional[List[str]] = None,
        metric: str = "mean_surprise",
    ) -> Dict[str, float]:
        """
        Aggregate anomaly scores by cortical parcellation.

        Parameters
        ----------
        labels : np.ndarray, shape (N,)
            Integer parcellation labels per vertex.
        label_names : list[str] or None
            Names for each label value.
        metric : str
            Aggregation metric: ``'mean_surprise'``, ``'max_surprise'``,
            ``'mean_z'``, ``'fraction_anomalous'``.

        Returns
        -------
        region_scores : dict[str, float]
            Score per region.
        """
        unique_labels = np.unique(labels[self.vertex_mask])
        result = {}

        for lab in unique_labels:
            if lab < 0:  # skip unknown / medial wall
                continue
            mask = self.vertex_mask & (labels == lab)
            if mask.sum() == 0:
                continue

            name = str(lab) if label_names is None else label_names[int(lab)]

            if metric == "mean_surprise":
                result[name] = float(np.mean(self.surprise[mask]))
            elif metric == "max_surprise":
                result[name] = float(np.max(self.surprise[mask]))
            elif metric == "mean_z":
                result[name] = float(np.mean(self.z_score[mask]))
            elif metric == "fraction_anomalous":
                anom = np.abs(self.z_score[mask]) > 2.0
                result[name] = float(anom.mean())
            else:
                raise ValueError(f"Unknown metric: {metric}")

        return result

    def aggregate_by_network(
        self,
        labels: np.ndarray,
        network_names: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Multi-metric aggregation by network (e.g. Yeo-7).

        Returns a nested dict: network → metric → value.
        """
        result = {}
        unique = np.unique(labels[self.vertex_mask])

        for lab in unique:
            if lab <= 0:
                continue
            mask = self.vertex_mask & (labels == lab)
            if mask.sum() == 0:
                continue

            name = (
                network_names[int(lab)]
                if network_names is not None
                else f"Network_{lab}"
            )
            z = self.z_score[mask]
            s = self.surprise[mask]

            result[name] = {
                "mean_z": float(np.mean(z)),
                "std_z": float(np.std(z)),
                "mean_surprise": float(np.mean(s)),
                "max_surprise": float(np.max(s)),
                "fraction_anomalous": float((np.abs(z) > 2.0).mean()),
                "n_vertices": int(mask.sum()),
            }

        return result


def compute_surprise(
    observed: np.ndarray,
    predicted_mean: np.ndarray,
    predicted_var: np.ndarray,
    prior_anomaly_prob: float = 0.05,
    anomaly_variance_factor: float = 10.0,
    mask: Optional[np.ndarray] = None,
) -> SurpriseMap:
    """
    Compute vertex-wise surprise and anomaly scores.

    This is the main function for scoring a patient against a normative
    model. It computes surprise, excess surprise, z-scores, and
    Bayesian anomaly probabilities.

    Parameters
    ----------
    observed : np.ndarray, shape (N,)
        Observed feature values (e.g. cortical thickness).
    predicted_mean : np.ndarray, shape (N,)
        GP posterior predictive mean.
    predicted_var : np.ndarray, shape (N,)
        GP posterior predictive variance.
    prior_anomaly_prob : float
        Prior probability that any given vertex is anomalous.
        Default: 5% (conservative).
    anomaly_variance_factor : float
        The anomalous component has variance = factor × predicted_var.
    mask : np.ndarray, shape (N,), dtype bool, or None
        Valid vertices. If None, all non-NaN vertices are valid.

    Returns
    -------
    SurpriseMap
        Complete anomaly scoring object.
    """
    N = observed.shape[0]

    # Build validity mask
    if mask is None:
        mask = (
            np.isfinite(observed)
            & np.isfinite(predicted_mean)
            & np.isfinite(predicted_var)
            & (predicted_var > 0)
        )

    # Compute z-scores
    std = np.sqrt(np.maximum(predicted_var, 1e-12))
    z_score = np.zeros(N, dtype=np.float64)
    z_score[mask] = (observed[mask] - predicted_mean[mask]) / std[mask]

    # Surprise: -log p(y | mu, sigma^2) under Gaussian
    surprise = np.zeros(N, dtype=np.float64)
    surprise[mask] = (
        0.5 * np.log(2.0 * np.pi * predicted_var[mask])
        + 0.5 * z_score[mask] ** 2
    )

    # Expected surprise under the null (calibrated Gaussian):
    # E[S] = 0.5 * (1 + log(2π σ²))
    expected_surprise = np.zeros(N, dtype=np.float64)
    expected_surprise[mask] = 0.5 * (
        1.0 + np.log(2.0 * np.pi * predicted_var[mask])
    )

    excess_surprise = np.zeros(N, dtype=np.float64)
    excess_surprise[mask] = surprise[mask] - expected_surprise[mask]

    # Bayesian anomaly probability via two-component mixture:
    #   p(anomalous | y) ∝ p(y | anomalous) · p(anomalous)
    # Normal component:  N(y | μ, σ²)
    # Anomaly component: N(y | μ, α·σ²)  with α >> 1
    anomaly_prob = np.zeros(N, dtype=np.float64)
    if prior_anomaly_prob > 0:
        log_p_normal = stats.norm.logpdf(
            observed[mask], predicted_mean[mask], std[mask],
        )
        anom_std = std[mask] * np.sqrt(anomaly_variance_factor)
        log_p_anomaly = stats.norm.logpdf(
            observed[mask], predicted_mean[mask], anom_std,
        )

        # Log posterior via log-sum-exp
        log_num = log_p_anomaly + np.log(prior_anomaly_prob)
        log_den_normal = log_p_normal + np.log(1.0 - prior_anomaly_prob)

        # log(p_normal + p_anomaly) via log-sum-exp
        log_total = np.logaddexp(log_den_normal, log_num)
        anomaly_prob[mask] = np.exp(log_num - log_total)

    return SurpriseMap(
        surprise=surprise,
        excess_surprise=excess_surprise,
        z_score=z_score,
        anomaly_probability=anomaly_prob,
        vertex_mask=mask,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Multi-feature surprise (combined anomaly across features)
# ═══════════════════════════════════════════════════════════════════════════


def combined_surprise(
    surprise_maps: List[SurpriseMap],
    method: str = "sum",
) -> SurpriseMap:
    """
    Combine surprise maps across multiple features (e.g. thickness,
    curvature, sulcal depth) into a single anomaly map.

    Parameters
    ----------
    surprise_maps : list[SurpriseMap]
        One per feature.
    method : ``'sum'``, ``'max'``, ``'fisher'``
        Combination method.
        - ``'sum'``: additive surprise (assumes independence).
        - ``'max'``: maximum surprise across features.
        - ``'fisher'``: Fisher's method (sum of −2 log p-values).

    Returns
    -------
    SurpriseMap
        Combined anomaly map.
    """
    if not surprise_maps:
        raise ValueError("Need at least one SurpriseMap.")

    # Common valid mask
    mask = surprise_maps[0].vertex_mask.copy()
    for sm in surprise_maps[1:]:
        mask &= sm.vertex_mask

    N = surprise_maps[0].n_vertices

    if method == "sum":
        combined_s = np.sum([sm.surprise for sm in surprise_maps], axis=0)
        combined_z = np.sqrt(
            np.sum([sm.z_score ** 2 for sm in surprise_maps], axis=0)
        ) * np.sign(np.mean([sm.z_score for sm in surprise_maps], axis=0))
    elif method == "max":
        all_s = np.stack([sm.surprise for sm in surprise_maps])
        combined_s = np.max(all_s, axis=0)
        # Z-score of the most extreme feature
        all_z = np.stack([sm.z_score for sm in surprise_maps])
        max_idx = np.argmax(np.abs(all_z), axis=0)
        combined_z = all_z[max_idx, np.arange(N)]
    elif method == "fisher":
        # Fisher's method: -2 Σ log(p_i) ~ χ²(2k) under H0
        # p-value from z-score (two-sided)
        all_p = np.stack([
            2.0 * stats.norm.sf(np.abs(sm.z_score)) for sm in surprise_maps
        ])
        all_p = np.clip(all_p, 1e-300, 1.0)
        fisher_stat = -2.0 * np.sum(np.log(all_p), axis=0)
        dof = 2 * len(surprise_maps)
        combined_p = stats.chi2.sf(fisher_stat, dof)
        combined_s = -np.log(np.maximum(combined_p, 1e-300))
        combined_z = stats.norm.isf(np.maximum(combined_p / 2, 1e-300))
    else:
        raise ValueError(f"Unknown method: {method}")

    # Recompute excess surprise and anomaly prob from combined
    expected_s = np.zeros_like(combined_s)
    expected_s[mask] = 0.5 * (1.0 + np.log(2.0 * np.pi))  # unit variance approx

    # Anomaly prob from combined z-score
    p_anom = np.zeros(N, dtype=np.float64)
    p_anom[mask] = 2.0 * stats.norm.sf(np.abs(combined_z[mask]))
    p_anom[mask] = 1.0 - p_anom[mask]  # probability of being anomalous

    return SurpriseMap(
        surprise=combined_s,
        excess_surprise=combined_s - expected_s,
        z_score=combined_z,
        anomaly_probability=p_anom,
        vertex_mask=mask,
    )
