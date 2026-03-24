"""
Cortical similarity network construction from morphometric features.

This module builds brain graphs from structural MRI data alone (no fMRI
or dMRI required). Two approaches are provided:

    1. **Morphometric Similarity Networks (MSN)** — inter-regional
       Pearson/Spearman correlation of morphometric feature profiles
       (Seidlitz et al., Neuron 2018). Each ROI has a vector of
       features (mean thickness, curvature, sulcal depth, etc.); the
       correlation between ROI profiles becomes the edge weight.

    2. **Spectral Similarity Networks** — inter-regional similarity
       based on spectral shape descriptors (HKS, WKS, GPS). This is
       novel: it captures geometric similarity between regions rather
       than just morphometric covariation.

Both approaches integrate with NetworkX for graph-theoretic analysis
(centrality, modularity, efficiency, small-worldness).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def morphometric_similarity_network(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    method: str = "pearson",
    fisher_z: bool = True,
) -> np.ndarray:
    """
    Construct a Morphometric Similarity Network (MSN).

    For each pair of ROIs (i, j), compute the correlation between
    their average morphometric feature profiles across vertices.

    Parameters
    ----------
    feature_matrix : np.ndarray, shape (N_vertices, N_features)
        Per-vertex feature values (columns: thickness, curv, sulc, …).
    labels : np.ndarray, shape (N_vertices,)
        Integer parcellation labels.
    method : ``'pearson'`` or ``'spearman'``
        Correlation method.
    fisher_z : bool
        Apply Fisher z-transform to correlation values.

    Returns
    -------
    msn : np.ndarray, shape (R, R)
        Symmetric correlation matrix where R is the number of ROIs.
    """
    from scipy.stats import pearsonr, spearmanr

    # Get unique ROI labels (exclude label <= 0 as medial wall/unknown)
    roi_labels = np.sort(np.unique(labels[labels > 0]))
    R = len(roi_labels)

    # Compute mean feature profile per ROI: shape (R, F)
    roi_profiles = np.zeros((R, feature_matrix.shape[1]), dtype=np.float64)
    for i, lab in enumerate(roi_labels):
        mask = labels == lab
        roi_profiles[i] = np.nanmean(feature_matrix[mask], axis=0)

    # Correlation matrix
    msn = np.zeros((R, R), dtype=np.float64)
    for i in range(R):
        for j in range(i + 1, R):
            if method == "pearson":
                r, _ = pearsonr(roi_profiles[i], roi_profiles[j])
            elif method == "spearman":
                r, _ = spearmanr(roi_profiles[i], roi_profiles[j])
            else:
                raise ValueError(f"Unknown method: {method}")

            if fisher_z:
                r = np.arctanh(np.clip(r, -0.9999, 0.9999))

            msn[i, j] = r
            msn[j, i] = r

    return msn


def spectral_similarity_network(
    spectral_features: np.ndarray,
    labels: np.ndarray,
    metric: str = "cosine",
) -> np.ndarray:
    """
    Construct a similarity network from spectral shape descriptors.

    Each ROI's spectral profile is the mean HKS/WKS/GPS vector across
    its vertices. Inter-regional similarity is computed via cosine
    similarity, correlation, or Euclidean distance.

    Parameters
    ----------
    spectral_features : np.ndarray, shape (N_vertices, D)
        Per-vertex spectral descriptors (e.g. from ``spectral_feature_matrix``).
    labels : np.ndarray, shape (N_vertices,)
        Parcellation labels.
    metric : ``'cosine'``, ``'correlation'``, ``'euclidean'``
        Similarity metric.

    Returns
    -------
    ssn : np.ndarray, shape (R, R)
        Symmetric similarity matrix.
    """
    from scipy.spatial.distance import cdist

    roi_labels = np.sort(np.unique(labels[labels > 0]))
    R = len(roi_labels)

    # Mean spectral profile per ROI
    roi_profiles = np.zeros((R, spectral_features.shape[1]), dtype=np.float64)
    for i, lab in enumerate(roi_labels):
        mask = labels == lab
        roi_profiles[i] = np.nanmean(spectral_features[mask], axis=0)

    # Distance matrix → similarity
    if metric == "cosine":
        dist = cdist(roi_profiles, roi_profiles, metric="cosine")
        ssn = 1.0 - dist  # cosine similarity
    elif metric == "correlation":
        dist = cdist(roi_profiles, roi_profiles, metric="correlation")
        ssn = 1.0 - dist
    elif metric == "euclidean":
        dist = cdist(roi_profiles, roi_profiles, metric="euclidean")
        # Convert to similarity: exp(-d / median(d))
        med = np.median(dist[dist > 0])
        ssn = np.exp(-dist / max(med, 1e-8))
    else:
        raise ValueError(f"Unknown metric: {metric}")

    np.fill_diagonal(ssn, 0.0)  # no self-loops
    return ssn


def graph_metrics(
    adjacency: np.ndarray,
    threshold: Optional[float] = None,
    density: Optional[float] = None,
) -> Dict[str, object]:
    """
    Compute standard graph-theoretic metrics from a similarity matrix.

    Parameters
    ----------
    adjacency : np.ndarray, shape (R, R)
        Symmetric similarity/weight matrix.
    threshold : float or None
        Hard threshold on edge weights.
    density : float or None
        Target graph density (proportion of edges to keep).
        Overrides ``threshold`` if both provided.

    Returns
    -------
    metrics : dict
        Keys include: ``degree``, ``clustering``, ``efficiency``,
        ``betweenness``, ``modularity``, ``strength``, ``assortativity``.
    """
    try:
        import networkx as nx
        from networkx.algorithms.community import greedy_modularity_communities
    except ImportError:
        raise ImportError("NetworkX is required for graph metrics.")

    W = adjacency.copy()

    # Apply threshold
    if density is not None:
        flat = np.sort(W[np.triu_indices_from(W, k=1)])[::-1]
        n_edges = int(density * len(flat))
        if n_edges > 0 and n_edges <= len(flat):
            threshold = flat[n_edges - 1]

    if threshold is not None:
        W[W < threshold] = 0.0

    # Build NetworkX graph
    G = nx.from_numpy_array(W)

    # Remove zero-weight edges
    zero_edges = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] <= 0]
    G.remove_edges_from(zero_edges)

    # Metrics
    result = {}
    result["n_nodes"] = G.number_of_nodes()
    result["n_edges"] = G.number_of_edges()
    result["density"] = nx.density(G)
    result["strength"] = dict(G.degree(weight="weight"))
    result["degree"] = dict(G.degree())
    result["clustering"] = nx.clustering(G, weight="weight")

    # Global efficiency
    try:
        result["global_efficiency"] = nx.global_efficiency(G)
    except Exception:
        result["global_efficiency"] = np.nan

    # Betweenness centrality
    result["betweenness"] = nx.betweenness_centrality(G, weight="weight")

    # Modularity via greedy algorithm
    try:
        communities = list(greedy_modularity_communities(G, weight="weight"))
        result["modularity"] = nx.community.modularity(
            G, communities, weight="weight",
        )
        result["communities"] = communities
    except Exception:
        result["modularity"] = np.nan
        result["communities"] = []

    # Assortativity
    try:
        result["assortativity"] = nx.degree_assortativity_coefficient(
            G, weight="weight",
        )
    except Exception:
        result["assortativity"] = np.nan

    return result
