"""
Spectral morphometric connectivity — brain graph construction & analysis.

This module constructs inter-regional brain graphs from spectral shape
descriptors (HKS, WKS, GPS, ShapeDNA eigenvalues) and classical
morphometric features, then analyses these graphs with state-of-the-art
methods spanning classical graph theory, persistent homology, spectral
graph analysis, network controllability, and graph neural networks.

The core novelty is **spectral morphometric connectivity**: using spectral
descriptors computed on the Laplace-Beltrami operator of cortical/sub-
cortical surface meshes as feature vectors for inter-regional similarity
computation.  This bridges Morphometric Similarity Networks (Seidlitz
et al., Neuron 2018) with spectral shape analysis (BrainPrint / ShapeDNA)
in a manner not previously published in the literature.

Supported graph construction methods
-------------------------------------
- Pearson / Spearman correlation (classic MSN)
- KL divergence (MIND, Sebenius et al. Nat Neurosci 2023)
- RBF / polynomial kernel
- Wasserstein distance between spectral distributions
- Cosine / Euclidean / correlation similarity
- Multi-descriptor fusion (concatenation, kernel average, multi-view max)

Thresholding strategies
-----------------------
- Proportional density
- OMST (Orthogonal Minimum Spanning Trees, Dimitriadis et al. 2017)
- Backbone disparity filter (Serrano et al. 2009)

Graph analysis
--------------
- Classical metrics (BCT-equivalent): degree, clustering, efficiency,
  betweenness, modularity, rich-club, small-world
- Spectral graph analysis: graph Laplacian eigendecomposition, Fiedler
  value, spectral gap, connectome harmonics
- Persistent homology: Betti curves, persistence diagrams via filtration
- Network controllability: average & modal controllability
- Community detection: Louvain, Leiden, spectral clustering
- Group-level: NBS, permutation testing on graph metrics
- PyTorch Geometric backend: ``to_pyg_data``, population graph construction
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import sparse as sp
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Constants & canonical colours
# ═══════════════════════════════════════════════════════════════════════════

#: Yeo-7 network canonical RGB colours (FreeSurfer LUT).
YEO7_COLORS: Dict[str, Tuple[float, float, float]] = {
    "Visual":          (120 / 255, 18 / 255, 134 / 255),
    "Somatomotor":     (70 / 255, 130 / 255, 180 / 255),
    "DorsalAttention":  (0 / 255, 118 / 255, 14 / 255),
    "VentralAttention": (196 / 255, 58 / 255, 250 / 255),
    "Limbic":          (220 / 255, 248 / 255, 164 / 255),
    "Frontoparietal":  (230 / 255, 148 / 255, 34 / 255),
    "Default":         (205 / 255, 62 / 255, 78 / 255),
}

YEO7_PREFIX_MAP = {
    "Vis": "Visual", "SomMot": "Somatomotor",
    "DorsAttn": "DorsalAttention", "SalVentAttn": "VentralAttention",
    "Limbic": "Limbic", "Cont": "Frontoparietal",
    "Default": "Default",
}


# ═══════════════════════════════════════════════════════════════════════════
# Data containers
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class GraphResult:
    """Container for a constructed brain graph and its metadata."""
    adjacency: np.ndarray
    roi_labels: np.ndarray
    roi_names: Optional[List[str]] = None
    method: str = ""
    thresholded: Optional[np.ndarray] = None
    threshold_method: Optional[str] = None
    roi_profiles: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_rois(self) -> int:
        return self.adjacency.shape[0]

    def to_networkx(self, weighted: bool = True):
        """Convert to NetworkX ``Graph``."""
        import networkx as nx
        W = self.thresholded if self.thresholded is not None else self.adjacency
        G = nx.from_numpy_array(W)
        zero = [(u, v) for u, v, d in G.edges(data=True)
                if d.get("weight", 0) <= 0]
        G.remove_edges_from(zero)
        if self.roi_names is not None:
            for i, name in enumerate(self.roi_names):
                if i in G:
                    G.nodes[i]["label"] = name
        return G

    def to_igraph(self):
        """Convert to igraph ``Graph``."""
        import igraph as ig
        W = self.thresholded if self.thresholded is not None else self.adjacency
        G = ig.Graph.Weighted_Adjacency(W.tolist(), mode="undirected",
                                        loops=False)
        if self.roi_names:
            G.vs["label"] = self.roi_names[:G.vcount()]
        return G


@dataclass
class GraphMetrics:
    """Container for comprehensive graph-theoretic metrics."""
    n_nodes: int = 0
    n_edges: int = 0
    density: float = 0.0
    degree: Optional[np.ndarray] = None
    strength: Optional[np.ndarray] = None
    clustering: Optional[np.ndarray] = None
    betweenness: Optional[np.ndarray] = None
    closeness: Optional[np.ndarray] = None
    eigenvector_centrality: Optional[np.ndarray] = None
    participation_coefficient: Optional[np.ndarray] = None
    global_efficiency: float = np.nan
    local_efficiency: float = np.nan
    transitivity: float = np.nan
    assortativity: float = np.nan
    modularity: float = np.nan
    communities: Optional[List] = None
    community_labels: Optional[np.ndarray] = None
    sigma: float = np.nan
    omega: float = np.nan
    rich_club_curve: Optional[Dict[int, float]] = None
    rich_club_normalized: Optional[Dict[int, float]] = None
    fiedler_value: float = np.nan
    spectral_gap: float = np.nan
    laplacian_eigenvalues: Optional[np.ndarray] = None
    avg_controllability: Optional[np.ndarray] = None
    modal_controllability: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# § 1  GRAPH CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════

def _aggregate_roi_profiles(
    features: np.ndarray,
    labels: np.ndarray,
    aggregation: str = "mean",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-ROI feature profiles from vertex-level data.

    Parameters
    ----------
    features : (V, D)
    labels : (V,) integer parcellation
    aggregation : 'mean', 'median', 'std', 'robust_mean'

    Returns
    -------
    profiles : (R, D)
    roi_labels : (R,)
    """
    roi_labels = np.sort(np.unique(labels[labels > 0]))
    R = len(roi_labels)
    D = features.shape[1]
    profiles = np.zeros((R, D), dtype=np.float64)

    for i, lab in enumerate(roi_labels):
        mask = labels == lab
        verts = features[mask]
        if verts.shape[0] == 0:
            profiles[i] = np.nan
            continue
        if aggregation == "mean":
            profiles[i] = np.nanmean(verts, axis=0)
        elif aggregation == "median":
            profiles[i] = np.nanmedian(verts, axis=0)
        elif aggregation == "std":
            profiles[i] = np.nanstd(verts, axis=0)
        elif aggregation == "robust_mean":
            lo = np.nanpercentile(verts, 5, axis=0)
            hi = np.nanpercentile(verts, 95, axis=0)
            profiles[i] = np.nanmean(np.clip(verts, lo, hi), axis=0)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

    return profiles, roi_labels


def morphometric_similarity_network(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    method: str = "pearson",
    fisher_z: bool = True,
    aggregation: str = "mean",
) -> GraphResult:
    """Construct a Morphometric Similarity Network (MSN).

    For each pair of ROIs the Pearson or Spearman correlation between
    their mean morphometric feature profiles becomes the edge weight
    (Seidlitz et al., Neuron 2018).
    """
    from scipy.stats import pearsonr, spearmanr
    profiles, roi_labels = _aggregate_roi_profiles(
        feature_matrix, labels, aggregation)
    R = len(roi_labels)
    msn = np.zeros((R, R), dtype=np.float64)
    _fn = pearsonr if method == "pearson" else spearmanr
    for i in range(R):
        for j in range(i + 1, R):
            r, _ = _fn(profiles[i], profiles[j])
            if fisher_z:
                r = np.arctanh(np.clip(r, -0.9999, 0.9999))
            msn[i, j] = r
            msn[j, i] = r
    return GraphResult(adjacency=msn, roi_labels=roi_labels, method="msn",
                       roi_profiles=profiles,
                       metadata={"correlation": method, "fisher_z": fisher_z})


def spectral_similarity_network(
    spectral_features: np.ndarray,
    labels: np.ndarray,
    metric: str = "cosine",
    aggregation: str = "mean",
) -> GraphResult:
    """Construct a similarity network from spectral shape descriptors.

    Each ROI's spectral profile is the mean HKS/WKS/GPS/eigenvalue
    vector across its constituent vertices.
    """
    profiles, roi_labels = _aggregate_roi_profiles(
        spectral_features, labels, aggregation)
    if metric in ("cosine", "correlation", "euclidean"):
        dist = cdist(profiles, profiles, metric=metric)
        if metric == "euclidean":
            med = np.median(dist[dist > 0]) if np.any(dist > 0) else 1.0
            ssn = np.exp(-dist / max(med, 1e-8))
        else:
            ssn = 1.0 - dist
    elif metric == "rbf":
        dist = cdist(profiles, profiles, metric="sqeuclidean")
        sigma2 = np.median(dist[dist > 0]) if np.any(dist > 0) else 1.0
        ssn = np.exp(-dist / (2.0 * sigma2))
    else:
        raise ValueError(f"Unknown metric: {metric}")
    np.fill_diagonal(ssn, 0.0)
    return GraphResult(adjacency=ssn, roi_labels=roi_labels,
                       method="spectral_similarity", roi_profiles=profiles,
                       metadata={"metric": metric, "aggregation": aggregation})


def mind_divergence_network(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 20,
) -> GraphResult:
    """Construct a MIND network via symmetric KL divergence
    (Sebenius et al. Nat Neurosci 2023)."""
    roi_labels = np.sort(np.unique(labels[labels > 0]))
    R = len(roi_labels)
    F = feature_matrix.shape[1]
    hists = np.zeros((R, F, n_bins), dtype=np.float64)
    for i, lab in enumerate(roi_labels):
        mask = labels == lab
        for f in range(F):
            vals = feature_matrix[mask, f]
            vals = vals[np.isfinite(vals)]
            if len(vals) < 5:
                hists[i, f] = 1.0 / n_bins
            else:
                h, _ = np.histogram(vals, bins=n_bins, density=True)
                h = h + 1e-10
                hists[i, f] = h / h.sum()
    mind = np.zeros((R, R), dtype=np.float64)
    for i in range(R):
        for j in range(i + 1, R):
            kl_sum = 0.0
            for f in range(F):
                p, q = hists[i, f], hists[j, f]
                kl_sum += 0.5 * (np.sum(p * np.log(p / q)) +
                                 np.sum(q * np.log(q / p)))
            mind[i, j] = 1.0 / (1.0 + kl_sum / F)
            mind[j, i] = mind[i, j]
    return GraphResult(adjacency=mind, roi_labels=roi_labels, method="mind",
                       metadata={"n_bins": n_bins, "n_features": F})


def wasserstein_spectral_network(
    spectral_features: np.ndarray,
    labels: np.ndarray,
    n_projections: int = 100,
) -> GraphResult:
    """Construct a network using sliced Wasserstein distance between
    spectral descriptor distributions at each ROI."""
    from corticalfields.utils import cf_progress
    roi_labels = np.sort(np.unique(labels[labels > 0]))
    R = len(roi_labels)
    D = spectral_features.shape[1]
    roi_verts = []
    for lab in roi_labels:
        mask = labels == lab
        v = spectral_features[mask]
        roi_verts.append(v[np.all(np.isfinite(v), axis=1)])
    rng = np.random.default_rng(42)
    proj = rng.standard_normal((n_projections, D))
    proj /= np.linalg.norm(proj, axis=1, keepdims=True)
    n_pairs = R * (R - 1) // 2
    dist_matrix = np.zeros((R, R), dtype=np.float64)
    pbar = cf_progress(total=n_pairs, description="Wasserstein distances")
    with pbar as p:
        for i in range(R):
            for j in range(i + 1, R):
                vi, vj = roi_verts[i], roi_verts[j]
                if len(vi) < 2 or len(vj) < 2:
                    dist_matrix[i, j] = dist_matrix[j, i] = np.nan
                    p.update(1)
                    continue
                pi = vi @ proj.T
                pj = vj @ proj.T
                sw = 0.0
                for k in range(n_projections):
                    s1, s2 = np.sort(pi[:, k]), np.sort(pj[:, k])
                    n = max(len(s1), len(s2))
                    g1 = np.interp(np.linspace(0, 1, n),
                                   np.linspace(0, 1, len(s1)), s1)
                    g2 = np.interp(np.linspace(0, 1, n),
                                   np.linspace(0, 1, len(s2)), s2)
                    sw += np.mean(np.abs(g1 - g2))
                dist_matrix[i, j] = dist_matrix[j, i] = sw / n_projections
                p.update(1)
    valid = dist_matrix[dist_matrix > 0]
    sigma = np.nanmedian(valid) if len(valid) > 0 else 1.0
    sim = np.exp(-dist_matrix / max(sigma, 1e-8))
    np.fill_diagonal(sim, 0.0)
    return GraphResult(adjacency=sim, roi_labels=roi_labels,
                       method="wasserstein_spectral",
                       metadata={"n_projections": n_projections,
                                 "sigma": float(sigma)})


def multi_descriptor_network(
    descriptor_dict: Dict[str, np.ndarray],
    labels: np.ndarray,
    fusion: str = "concatenate",
    metric: str = "cosine",
    aggregation: str = "mean",
    weights: Optional[Dict[str, float]] = None,
) -> GraphResult:
    """Construct a network fusing multiple spectral descriptors.

    Parameters
    ----------
    descriptor_dict : dict mapping name -> (V, D_i) arrays
    fusion : 'concatenate', 'kernel_average', 'multi_view_max'
    """
    if fusion == "concatenate":
        arrays = []
        for arr in descriptor_dict.values():
            mu = np.nanmean(arr, axis=0, keepdims=True)
            sd = np.nanstd(arr, axis=0, keepdims=True)
            sd = np.where(sd < 1e-10, 1.0, sd)
            arrays.append((arr - mu) / sd)
        combined = np.hstack(arrays)
        result = spectral_similarity_network(combined, labels, metric, aggregation)
        result.method = "multi_descriptor_concat"
        result.metadata["descriptors"] = list(descriptor_dict.keys())
        return result
    elif fusion == "kernel_average":
        W = weights or {k: 1.0 / len(descriptor_dict) for k in descriptor_dict}
        total_w = sum(W.values())
        W = {k: v / total_w for k, v in W.items()}
        combined_sim = None
        roi_ref = None
        for name, arr in descriptor_dict.items():
            gr = spectral_similarity_network(arr, labels, metric, aggregation)
            if combined_sim is None:
                combined_sim = W.get(name, 1.0) * gr.adjacency
                roi_ref = gr.roi_labels
            else:
                combined_sim += W.get(name, 1.0) * gr.adjacency
        return GraphResult(adjacency=combined_sim, roi_labels=roi_ref,
                           method="multi_descriptor_kernel_avg",
                           metadata={"descriptors": list(descriptor_dict.keys()),
                                     "weights": W, "metric": metric})
    elif fusion == "multi_view_max":
        combined_sim = None
        roi_ref = None
        for arr in descriptor_dict.values():
            gr = spectral_similarity_network(arr, labels, metric, aggregation)
            if combined_sim is None:
                combined_sim = gr.adjacency.copy()
                roi_ref = gr.roi_labels
            else:
                combined_sim = np.maximum(combined_sim, gr.adjacency)
        return GraphResult(adjacency=combined_sim, roi_labels=roi_ref,
                           method="multi_descriptor_max",
                           metadata={"descriptors": list(descriptor_dict.keys())})
    else:
        raise ValueError(f"Unknown fusion: {fusion}")


# ═══════════════════════════════════════════════════════════════════════════
# § 2  THRESHOLDING
# ═══════════════════════════════════════════════════════════════════════════

def proportional_threshold(adjacency: np.ndarray, density: float = 0.15) -> np.ndarray:
    """Keep only the top ``density`` fraction of edges by weight."""
    W = adjacency.copy()
    np.fill_diagonal(W, 0.0)
    tri = W[np.triu_indices_from(W, k=1)]
    n_keep = max(1, int(density * len(tri)))
    if n_keep >= len(tri):
        return W
    cutoff = np.sort(tri)[::-1][n_keep - 1]
    W[W < cutoff] = 0.0
    return W


def omst_threshold(adjacency: np.ndarray, max_trees: int = 10) -> np.ndarray:
    """Orthogonal Minimum Spanning Tree thresholding
    (Dimitriadis et al. 2017)."""
    from scipy.sparse.csgraph import minimum_spanning_tree
    R = adjacency.shape[0]
    sim = np.maximum(adjacency.copy(), 0.0)
    np.fill_diagonal(sim, 0.0)
    dist = sim.max() - sim + 1e-10
    np.fill_diagonal(dist, 0.0)
    best_score = -np.inf
    best_union = np.zeros((R, R), dtype=np.float64)
    union = np.zeros((R, R), dtype=np.float64)
    remaining = dist.copy()
    for t in range(max_trees):
        mst = minimum_spanning_tree(sp.csr_matrix(remaining)).toarray()
        mst = mst + mst.T
        if mst.sum() == 0:
            break
        new_edges = (mst > 0) & (union == 0)
        union[new_edges] = sim[new_edges]
        eff = _global_efficiency_fast(union)
        n_e = np.count_nonzero(union[np.triu_indices(R, k=1)])
        cost = n_e / (R * (R - 1) / 2)
        score = eff - cost
        if score > best_score:
            best_score = score
            best_union = union.copy()
        else:
            break
        remaining[mst > 0] = 0.0
    return best_union


def backbone_disparity_filter(adjacency: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Backbone extraction via disparity filter (Serrano et al. 2009)."""
    R = adjacency.shape[0]
    W = np.maximum(adjacency.copy(), 0.0)
    np.fill_diagonal(W, 0.0)
    backbone = np.zeros_like(W)
    for i in range(R):
        s_i = W[i].sum()
        if s_i == 0:
            continue
        k_i = np.count_nonzero(W[i])
        if k_i <= 1:
            backbone[i] = W[i]
            continue
        for j in range(R):
            if W[i, j] == 0:
                continue
            p_ij = W[i, j] / s_i
            p_value = (1.0 - p_ij) ** (k_i - 1)
            if p_value < alpha:
                backbone[i, j] = W[i, j]
    return np.maximum(backbone, backbone.T)


def apply_threshold(graph_result: GraphResult, method: str = "proportional",
                    **kwargs) -> GraphResult:
    """Apply thresholding to a GraphResult (modifies in-place)."""
    fns = {"proportional": proportional_threshold, "omst": omst_threshold,
           "disparity": backbone_disparity_filter}
    if method not in fns:
        raise ValueError(f"Unknown threshold method: {method}")
    graph_result.thresholded = fns[method](graph_result.adjacency, **kwargs)
    graph_result.threshold_method = method
    return graph_result


# ═══════════════════════════════════════════════════════════════════════════
# § 3  GRAPH ANALYSIS — classical metrics
# ═══════════════════════════════════════════════════════════════════════════

def _global_efficiency_fast(W: np.ndarray) -> float:
    """Fast global efficiency via shortest paths."""
    from scipy.sparse.csgraph import shortest_path
    R = W.shape[0]
    with np.errstate(divide="ignore"):
        D = np.where(W > 0, 1.0 / W, 0.0)
    dist = shortest_path(sp.csr_matrix(D), directed=False)
    with np.errstate(divide="ignore"):
        inv = np.where(np.isinf(dist) | (dist == 0), 0.0, 1.0 / dist)
    np.fill_diagonal(inv, 0.0)
    return inv.sum() / (R * (R - 1)) if R > 1 else 0.0


def comprehensive_graph_metrics(
    graph_result: GraphResult,
    compute_rich_club: bool = True,
    compute_controllability: bool = True,
    compute_spectral: bool = True,
    n_random: int = 100,
) -> GraphMetrics:
    """Compute a full suite of graph-theoretic metrics (BCT-equivalent)."""
    import networkx as nx
    from corticalfields.utils import cf_progress

    W = (graph_result.thresholded if graph_result.thresholded is not None
         else graph_result.adjacency)
    G = nx.from_numpy_array(W)
    rm = [(u, v) for u, v, d in G.edges(data=True) if d.get("weight", 0) <= 0]
    G.remove_edges_from(rm)
    R = G.number_of_nodes()
    gm = GraphMetrics(n_nodes=R, n_edges=G.number_of_edges())
    gm.density = nx.density(G)

    # Node-level
    gm.degree = np.array([G.degree(i) for i in range(R)], dtype=np.int64)
    gm.strength = np.array([G.degree(i, weight="weight") for i in range(R)])
    gm.clustering = np.array([nx.clustering(G, i, weight="weight")
                              for i in range(R)])
    bc = nx.betweenness_centrality(G, weight="weight")
    gm.betweenness = np.array([bc.get(i, 0.0) for i in range(R)])
    gm.closeness = np.array([nx.closeness_centrality(G, i, distance="weight")
                             for i in range(R)])
    try:
        ec = nx.eigenvector_centrality_numpy(G, weight="weight")
        gm.eigenvector_centrality = np.array([ec.get(i, 0) for i in range(R)])
    except Exception:
        gm.eigenvector_centrality = np.full(R, np.nan)

    # Global
    try:
        gm.global_efficiency = nx.global_efficiency(G)
    except Exception:
        gm.global_efficiency = _global_efficiency_fast(W)
    try:
        gm.local_efficiency = nx.local_efficiency(G)
    except Exception:
        pass
    gm.transitivity = nx.transitivity(G)
    try:
        gm.assortativity = nx.degree_assortativity_coefficient(G, weight="weight")
    except Exception:
        pass

    # Community (Louvain)
    try:
        from networkx.algorithms.community import louvain_communities
        comms = louvain_communities(G, weight="weight", seed=42)
    except Exception:
        from networkx.algorithms.community import greedy_modularity_communities
        comms = list(greedy_modularity_communities(G, weight="weight"))
    gm.communities = [sorted(c) for c in comms]
    gm.modularity = nx.community.modularity(G, comms, weight="weight")
    gm.community_labels = np.zeros(R, dtype=np.int64)
    for ci, c in enumerate(comms):
        for n in c:
            gm.community_labels[n] = ci

    # Participation coefficient
    pc = np.zeros(R)
    for i in range(R):
        ki = gm.strength[i]
        if ki == 0:
            continue
        for ci in range(len(gm.communities)):
            ki_s = sum(G[i][j].get("weight", 0)
                       for j in G.neighbors(i)
                       if gm.community_labels[j] == ci)
            pc[i] += (ki_s / ki) ** 2
        pc[i] = 1.0 - pc[i]
    gm.participation_coefficient = pc

    # Rich-club
    if compute_rich_club and G.number_of_edges() > 0:
        try:
            rc = nx.rich_club_coefficient(G, normalized=False)
            gm.rich_club_curve = rc
            if n_random > 0:
                rc_rand = {}
                for _ in cf_progress(range(n_random), description="Rich-club null"):
                    ds = [d for _, d in G.degree()]
                    try:
                        Gr = nx.Graph(nx.configuration_model(ds))
                        Gr.remove_edges_from(nx.selfloop_edges(Gr))
                        rr = nx.rich_club_coefficient(Gr, normalized=False)
                        for k, v in rr.items():
                            rc_rand.setdefault(k, []).append(v)
                    except Exception:
                        continue
                gm.rich_club_normalized = {}
                for k, v in rc.items():
                    nm = np.mean(rc_rand.get(k, [1.0]))
                    gm.rich_club_normalized[k] = v / nm if nm > 0 else np.nan
        except Exception as e:
            logger.warning("Rich-club failed: %s", e)

    # Spectral
    if compute_spectral:
        try:
            L = nx.laplacian_matrix(G, weight="weight").toarray().astype(np.float64)
            evals = np.sort(np.linalg.eigvalsh(L))
            gm.laplacian_eigenvalues = evals
            gm.fiedler_value = float(evals[1]) if len(evals) > 1 else np.nan
            if evals[-1] > 0:
                gm.spectral_gap = float(evals[1] / evals[-1])
        except Exception as e:
            logger.warning("Spectral analysis failed: %s", e)

    # Controllability
    if compute_controllability:
        try:
            gm.avg_controllability, gm.modal_controllability = (
                _network_controllability(W))
        except Exception as e:
            logger.warning("Controllability failed: %s", e)

    return gm


def _network_controllability(adjacency: np.ndarray):
    """Average and modal controllability (Gu et al. 2015)."""
    A = adjacency.copy()
    A = A / (np.linalg.svd(A, compute_uv=False)[0] + 1e-10)
    R = A.shape[0]
    evals, evecs = np.linalg.eigh(A)
    denom = 1.0 - evals ** 2
    denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
    avg_ctrl = np.array([np.sum(evecs[i] ** 2 / denom) for i in range(R)])
    modal_ctrl = np.array([np.sum((1.0 - evals ** 2) * evecs[i] ** 2)
                           for i in range(R)])
    return avg_ctrl, modal_ctrl


def community_detection(
    graph_result: GraphResult,
    method: str = "louvain",
    resolution: float = 1.0,
    n_communities: Optional[int] = None,
) -> Tuple[np.ndarray, float]:
    """Detect communities: 'louvain', 'leiden', or 'spectral'."""
    import networkx as nx
    W = (graph_result.thresholded if graph_result.thresholded is not None
         else graph_result.adjacency)
    G = nx.from_numpy_array(W)
    rm = [(u, v) for u, v, d in G.edges(data=True) if d.get("weight", 0) <= 0]
    G.remove_edges_from(rm)
    R = G.number_of_nodes()

    if method == "louvain":
        from networkx.algorithms.community import louvain_communities
        comms = louvain_communities(G, weight="weight", resolution=resolution,
                                    seed=42)
        labels = np.zeros(R, dtype=np.int64)
        for ci, c in enumerate(comms):
            for n in c:
                labels[n] = ci
        return labels, nx.community.modularity(G, comms, weight="weight")
    elif method == "leiden":
        import igraph as ig
        import leidenalg
        G_ig = graph_result.to_igraph()
        part = leidenalg.find_partition(G_ig, leidenalg.CPMVertexPartition,
                                        weights="weight",
                                        resolution_parameter=resolution)
        return np.array(part.membership, dtype=np.int64), part.quality()
    elif method == "spectral":
        from sklearn.cluster import SpectralClustering
        k = n_communities or max(2, int(np.sqrt(R / 2)))
        sc = SpectralClustering(n_clusters=k, affinity="precomputed",
                                random_state=42)
        labels = sc.fit_predict(np.maximum(W, 0))
        comms = [set(np.where(labels == c)[0]) for c in np.unique(labels)]
        Q = nx.community.modularity(nx.from_numpy_array(np.maximum(W, 0)),
                                     comms, weight="weight")
        return labels, Q
    else:
        raise ValueError(f"Unknown method: {method}")


def persistent_homology(
    graph_result: GraphResult, max_dim: int = 1,
) -> Dict[str, Any]:
    """Compute persistent homology via filtration on the brain graph."""
    W = (graph_result.thresholded if graph_result.thresholded is not None
         else graph_result.adjacency)
    W_pos = np.maximum(W, 0)
    np.fill_diagonal(W_pos, 0.0)
    max_w = W_pos.max() if W_pos.max() > 0 else 1.0
    D = max_w - W_pos
    np.fill_diagonal(D, 0.0)
    try:
        from ripser import ripser
        result = ripser(D, maxdim=max_dim, distance_matrix=True)
        diagrams = result["dgms"]
    except ImportError:
        logger.warning("ripser not installed; H0 only via manual filtration.")
        diagrams = [_manual_h0_filtration(D)]
        max_dim = 0
    n_t = 50
    thresh = np.linspace(0, D.max(), n_t)
    betti = np.zeros((n_t, max_dim + 1), dtype=np.int64)
    for dim, dgm in enumerate(diagrams):
        if dim > max_dim:
            break
        for ti, t in enumerate(thresh):
            betti[ti, dim] = np.sum((dgm[:, 0] <= t) & (dgm[:, 1] > t))
    return {"diagrams": diagrams, "betti_numbers": betti,
            "filtration_values": thresh}


def _manual_h0_filtration(D: np.ndarray) -> np.ndarray:
    """Compute H0 persistence diagram via union-find."""
    R = D.shape[0]
    i_idx, j_idx = np.triu_indices(R, k=1)
    weights = D[i_idx, j_idx]
    order = np.argsort(weights)
    parent = list(range(R))
    rank = [0] * R
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1
        return True
    births = np.zeros(R)
    deaths = np.full(R, np.inf)
    n_m = 0
    for idx in order:
        i, j = int(i_idx[idx]), int(j_idx[idx])
        if find(i) != find(j):
            deaths[n_m] = weights[idx]
            union(i, j)
            n_m += 1
            if n_m >= R - 1:
                break
    return np.column_stack([births[:n_m + 1], deaths[:n_m + 1]])


# ═══════════════════════════════════════════════════════════════════════════
# § 4  GROUP-LEVEL INFERENCE
# ═══════════════════════════════════════════════════════════════════════════

def nbs_morphometric(
    graphs_group1: List[GraphResult],
    graphs_group2: List[GraphResult],
    test_stat: str = "t",
    threshold: float = 3.0,
    n_permutations: int = 5000,
    alpha: float = 0.05,
    use_gpu: bool = False,
) -> Dict[str, Any]:
    """Network-Based Statistic for morphometric connectivity graphs
    (Zalesky et al. 2010)."""
    from corticalfields.utils import cf_progress
    n1, n2 = len(graphs_group1), len(graphs_group2)
    R = graphs_group1[0].n_rois
    def _adj(gr):
        return gr.thresholded if gr.thresholded is not None else gr.adjacency
    A1 = np.stack([_adj(g) for g in graphs_group1])
    A2 = np.stack([_adj(g) for g in graphs_group2])
    A_all = np.concatenate([A1, A2], axis=0)
    N = n1 + n2
    stat_matrix = _edge_tstat(A1, A2, test_stat)
    supra = np.abs(stat_matrix) > threshold
    obs_sizes = _comp_sizes(supra)
    obs_max = max(obs_sizes) if obs_sizes else 0

    if use_gpu:
        null_max = _nbs_perm_gpu(A_all, n1, n2, test_stat, threshold,
                                  n_permutations)
    else:
        null_max = np.zeros(n_permutations)
        rng = np.random.default_rng(42)
        pbar = cf_progress(total=n_permutations, description="NBS permutations")
        with pbar as p:
            for pi in range(n_permutations):
                idx = rng.permutation(N)
                ps = _edge_tstat(A_all[idx[:n1]], A_all[idx[n1:]], test_stat)
                sz = _comp_sizes(np.abs(ps) > threshold)
                null_max[pi] = max(sz) if sz else 0
                p.update(1)

    p_values = [(np.sum(null_max >= sz) + 1) / (n_permutations + 1)
                for sz in obs_sizes]
    sig_mask = np.zeros((R, R), dtype=bool)
    components = _extract_comps(supra)
    for ci, (comp, sz) in enumerate(zip(components, obs_sizes)):
        if ci < len(p_values) and p_values[ci] < alpha:
            for u, v in comp:
                sig_mask[u, v] = sig_mask[v, u] = True
    return {"stat_matrix": stat_matrix, "components": components,
            "component_sizes": obs_sizes, "p_values": p_values,
            "null_distribution": null_max, "significant_mask": sig_mask}


def _edge_tstat(A1, A2, test_stat):
    from scipy.stats import ttest_ind
    R = A1.shape[1]
    stat = np.zeros((R, R))
    for i in range(R):
        for j in range(i + 1, R):
            x1, x2 = A1[:, i, j], A2[:, i, j]
            if np.std(x1) < 1e-10 and np.std(x2) < 1e-10:
                continue
            if test_stat == "t":
                t, _ = ttest_ind(x1, x2, equal_var=False)
                stat[i, j] = t
            stat[j, i] = stat[i, j]
    return stat


def _comp_sizes(supra):
    from scipy.sparse.csgraph import connected_components
    n_c, labels = connected_components(sp.csr_matrix(supra.astype(float)),
                                       directed=False)
    sizes = []
    for c in range(n_c):
        members = np.where(labels == c)[0]
        if len(members) < 2:
            continue
        ne = sum(1 for ii in range(len(members))
                 for jj in range(ii+1, len(members))
                 if supra[members[ii], members[jj]])
        if ne > 0:
            sizes.append(ne)
    return sorted(sizes, reverse=True)


def _extract_comps(supra):
    from scipy.sparse.csgraph import connected_components
    n_c, labels = connected_components(sp.csr_matrix(supra.astype(float)),
                                       directed=False)
    comps = []
    for c in range(n_c):
        members = np.where(labels == c)[0]
        if len(members) < 2:
            continue
        edges = [(members[ii], members[jj])
                 for ii in range(len(members))
                 for jj in range(ii+1, len(members))
                 if supra[members[ii], members[jj]]]
        if edges:
            comps.append(edges)
    return comps


def _nbs_perm_gpu(A_all, n1, n2, test_stat, threshold, n_perm):
    """GPU-accelerated NBS permutation testing."""
    import torch
    from corticalfields.utils import gc_gpu, cf_progress
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N, R, _ = A_all.shape
    triu_i, triu_j = np.triu_indices(R, k=1)
    flat = np.zeros((N, len(triu_i)), dtype=np.float32)
    for s in range(N):
        flat[s] = A_all[s][triu_i, triu_j]
    flat_t = torch.from_numpy(flat).to(device)
    null_max = np.zeros(n_perm)
    chunk = min(500, n_perm)
    pbar = cf_progress(total=n_perm, description="NBS GPU permutations")
    with pbar as p:
        for start in range(0, n_perm, chunk):
            for pi in range(start, min(start + chunk, n_perm)):
                idx = torch.randperm(N, device=device)
                g1, g2 = flat_t[idx[:n1]], flat_t[idx[n1:]]
                m1, m2 = g1.mean(0), g2.mean(0)
                se = torch.sqrt(g1.var(0) / n1 + g2.var(0) / n2 + 1e-10)
                t_vals = (m1 - m2) / se
                supra_f = (torch.abs(t_vals) > threshold).cpu().numpy()
                supra_m = np.zeros((R, R), dtype=bool)
                supra_m[triu_i, triu_j] = supra_f
                supra_m |= supra_m.T
                sz = _comp_sizes(supra_m)
                null_max[pi] = max(sz) if sz else 0
                p.update(1)
            gc_gpu()
    return null_max


def group_metric_comparison(
    graphs_group1: List[GraphResult],
    graphs_group2: List[GraphResult],
    metrics: Optional[List[str]] = None,
    n_permutations: int = 5000,
) -> Dict[str, Dict[str, float]]:
    """Compare graph-level metrics between two groups via permutation."""
    if metrics is None:
        metrics = ["global_efficiency", "modularity", "transitivity",
                    "assortativity", "fiedler_value"]
    def _scalar(gr, mn):
        gm = comprehensive_graph_metrics(gr, compute_rich_club=False,
                                          compute_controllability=False)
        return getattr(gm, mn, np.nan)

    results = {}
    for mn in metrics:
        v1 = np.array([_scalar(g, mn) for g in graphs_group1])
        v2 = np.array([_scalar(g, mn) for g in graphs_group2])
        obs = np.nanmean(v1) - np.nanmean(v2)
        all_v = np.concatenate([v1, v2])
        rng = np.random.default_rng(42)
        null = np.array([np.nanmean(all_v[(idx := rng.permutation(len(all_v)))[:len(v1)]]) -
                         np.nanmean(all_v[idx[len(v1):]]) for _ in range(n_permutations)])
        pv = (np.sum(np.abs(null) >= np.abs(obs)) + 1) / (n_permutations + 1)
        ps = np.sqrt((np.nanvar(v1) + np.nanvar(v2)) / 2)
        results[mn] = {"mean_g1": float(np.nanmean(v1)),
                       "mean_g2": float(np.nanmean(v2)),
                       "diff": float(obs), "p_value": float(pv),
                       "effect_d": float(obs / ps if ps > 0 else 0)}
    return results


# ═══════════════════════════════════════════════════════════════════════════
# § 5  PyTorch Geometric BACKEND
# ═══════════════════════════════════════════════════════════════════════════

def to_pyg_data(graph_result: GraphResult, node_features=None, label=None):
    """Convert GraphResult to PyTorch Geometric Data object."""
    import torch
    from torch_geometric.data import Data
    W = graph_result.thresholded if graph_result.thresholded is not None else graph_result.adjacency
    rows, cols = np.where(W > 0)
    edge_index = torch.tensor(np.stack([rows, cols]), dtype=torch.long)
    edge_attr = torch.tensor(W[rows, cols], dtype=torch.float32).unsqueeze(1)
    if node_features is not None:
        x = torch.tensor(node_features, dtype=torch.float32)
    elif graph_result.roi_profiles is not None:
        x = torch.tensor(graph_result.roi_profiles, dtype=torch.float32)
    else:
        x = torch.eye(graph_result.n_rois, dtype=torch.float32)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if label is not None:
        data.y = torch.tensor([label], dtype=torch.long)
    return data


def build_population_graph(
    subjects: List[GraphResult],
    subject_features=None, labels=None,
    k_neighbours: int = 10,
):
    """Build a population-level graph (Parisot et al. 2018 style)."""
    import torch
    from torch_geometric.data import Data
    from sklearn.neighbors import kneighbors_graph
    N = len(subjects)
    if subject_features is None:
        feats = []
        triu_i, triu_j = np.triu_indices(subjects[0].n_rois, k=1)
        for gr in subjects:
            gm = comprehensive_graph_metrics(gr, compute_rich_club=False,
                                              compute_controllability=False,
                                              compute_spectral=False)
            vec = np.array([gm.global_efficiency, gm.modularity,
                            gm.transitivity, gm.assortativity, gm.density,
                            np.mean(gm.clustering), np.mean(gm.betweenness)])
            W = gr.thresholded if gr.thresholded is not None else gr.adjacency
            flat = W[triu_i, triu_j].astype(np.float32)
            feats.append(np.concatenate([vec, flat]))
        subject_features = np.array(feats, dtype=np.float32)
    knn = kneighbors_graph(subject_features, n_neighbors=k_neighbours,
                           mode="connectivity", include_self=False)
    knn = knn + knn.T
    knn[knn > 0] = 1
    rows, cols = knn.nonzero()
    edge_index = torch.tensor(np.stack([rows, cols]), dtype=torch.long)
    x = torch.tensor(subject_features, dtype=torch.float32)
    data = Data(x=x, edge_index=edge_index)
    if labels is not None:
        data.y = torch.tensor(labels, dtype=torch.long)
    return data


class BrainGraphGCN:
    """Minimal GCN for brain graph classification via PyG."""

    def __init__(self, in_channels=42, hidden_channels=64,
                 n_classes=2, dropout=0.3):
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.n_classes = n_classes
        self.dropout = dropout
        self._model = None
        self._device = None

    def _build_model(self):
        import torch
        import torch.nn.functional as F
        from torch_geometric.nn import GCNConv, global_mean_pool
        ic, hc, nc, dr = (self.in_channels, self.hidden_channels,
                          self.n_classes, self.dropout)
        class _GCN(torch.nn.Module):
            def __init__(s):
                super().__init__()
                s.c1 = GCNConv(ic, hc)
                s.c2 = GCNConv(hc, hc)
                s.c3 = GCNConv(hc, hc)
                s.lin = torch.nn.Linear(hc, nc)
            def forward(s, data):
                x, ei, b = data.x, data.edge_index, data.batch
                ew = data.edge_attr.squeeze() if data.edge_attr is not None else None
                x = F.relu(s.c1(x, ei, ew))
                x = F.dropout(x, p=dr, training=s.training)
                x = F.relu(s.c2(x, ei, ew))
                x = F.dropout(x, p=dr, training=s.training)
                x = s.c3(x, ei, ew)
                return s.lin(global_mean_pool(x, b))
        return _GCN()

    def fit(self, dataset, epochs=100, lr=1e-3, weight_decay=1e-4,
            batch_size=32):
        import torch, torch.nn.functional as F
        from torch_geometric.loader import DataLoader
        from corticalfields.utils import gc_gpu, cf_progress
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._build_model().to(self._device)
        opt = torch.optim.Adam(self._model.parameters(), lr=lr,
                               weight_decay=weight_decay)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self._model.train()
        for ep in cf_progress(range(epochs), description="Training GCN"):
            for batch in loader:
                batch = batch.to(self._device)
                opt.zero_grad()
                loss = F.cross_entropy(self._model(batch), batch.y)
                loss.backward()
                opt.step()
            gc_gpu()

    def predict(self, dataset):
        import torch
        from torch_geometric.loader import DataLoader
        self._model.eval()
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        preds = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self._device)
                preds.append(self._model(batch).argmax(1).cpu().numpy())
        return np.concatenate(preds)


# ═══════════════════════════════════════════════════════════════════════════
# § 6  CONVENIENCE PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def spectral_morphometric_pipeline(
    spectral_features: np.ndarray,
    labels: np.ndarray,
    metric: str = "rbf",
    threshold_method: str = "omst",
    threshold_kwargs: Optional[dict] = None,
    compute_metrics: bool = True,
    roi_names: Optional[List[str]] = None,
) -> Tuple[GraphResult, Optional[GraphMetrics]]:
    """End-to-end: spectral descriptors -> graph -> metrics."""
    gr = spectral_similarity_network(spectral_features, labels, metric=metric)
    gr.roi_names = roi_names
    apply_threshold(gr, method=threshold_method, **(threshold_kwargs or {}))
    gm = comprehensive_graph_metrics(gr) if compute_metrics else None
    return gr, gm
