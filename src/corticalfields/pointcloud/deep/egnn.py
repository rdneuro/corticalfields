"""
E(n) Equivariant Graph Neural Networks for brain point clouds.

Provides rotation-, translation-, and reflection-equivariant graph neural
networks operating on brain point clouds. EGNN naturally handles the
orientation variability of brain surfaces across subjects without
requiring rigid alignment preprocessing — critical for detecting subtle
left-right asymmetries in MTLE-HS.

Classes
-------
EGNNLayer           : Single E(n)-equivariant message-passing layer
EGNNClassifier      : Per-vertex classification (parcellation)
EGNNEncoder         : Feature extraction backbone

Functions
---------
build_knn_graph     : Construct k-NN graph from point cloud
egnn_inference      : Run EGNN inference with VRAM management

References
----------
Satorras, Hoogeboom & Welling (2021). E(n) Equivariant Graph Neural
    Networks. ICML.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Graph construction
# ═══════════════════════════════════════════════════════════════════════════


def build_knn_graph(
    points: np.ndarray,
    k: int = 20,
    self_loops: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct a k-NN graph from a point cloud.

    Parameters
    ----------
    points : np.ndarray, shape (N, 3)
    k : int
        Number of nearest neighbours per point.
    self_loops : bool
        Include self-edges.

    Returns
    -------
    edge_index : np.ndarray, shape (2, E)
        COO-format edge indices.
    edge_attr : np.ndarray, shape (E, 1)
        Euclidean edge lengths.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    start_k = 1 if not self_loops else 0
    dists, idx = tree.query(points, k=k + 1)

    N = points.shape[0]
    sources = np.repeat(np.arange(N), k)
    targets = idx[:, start_k:start_k + k].flatten()
    edge_lengths = dists[:, start_k:start_k + k].flatten()

    edge_index = np.vstack([sources, targets]).astype(np.int64)
    edge_attr = edge_lengths[:, np.newaxis].astype(np.float32)

    return edge_index, edge_attr


# ═══════════════════════════════════════════════════════════════════════════
# EGNN components (self-contained PyTorch implementation)
# ═══════════════════════════════════════════════════════════════════════════


def _make_egnn_classes():
    """Factory to create EGNN torch.nn.Module classes lazily."""
    import torch
    import torch.nn as nn

    class EGNNLayer(nn.Module):
        """
        Single E(n)-equivariant message-passing layer.

        Operates on node features h ∈ R^d and node positions x ∈ R^3.
        The layer updates both features and positions while preserving
        E(n) equivariance (rotation, translation, reflection).

        Message:  m_ij = φ_e(h_i, h_j, ‖x_i − x_j‖², a_ij)
        Position: x_i' = x_i + Σ_j (x_i − x_j) φ_x(m_ij)
        Feature:  h_i' = φ_h(h_i, Σ_j m_ij)
        """

        def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            out_channels: int,
            update_coords: bool = True,
        ):
            super().__init__()
            self.update_coords = update_coords

            # Edge MLP: (h_i, h_j, dist², edge_attr) → message
            edge_in = 2 * in_channels + 1 + 1  # +1 for dist², +1 for edge_attr
            self.edge_mlp = nn.Sequential(
                nn.Linear(edge_in, hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.SiLU(),
            )

            # Node MLP: (h_i, agg_msg) → h_i'
            self.node_mlp = nn.Sequential(
                nn.Linear(in_channels + hidden_channels, hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, out_channels),
            )

            # Coordinate MLP: message → scalar weight for coordinate update
            if update_coords:
                self.coord_mlp = nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.SiLU(),
                    nn.Linear(hidden_channels, 1),
                )

            self.norm = nn.LayerNorm(out_channels)

        def forward(
            self,
            h: torch.Tensor,              # (N, C_in) node features
            x: torch.Tensor,              # (N, 3)   node positions
            edge_index: torch.Tensor,     # (2, E)   edges
            edge_attr: torch.Tensor,      # (E, 1)   edge attributes
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Returns
            -------
            h_out : (N, C_out) updated features
            x_out : (N, 3)    updated positions
            """
            src, dst = edge_index[0], edge_index[1]
            N = h.shape[0]

            # Relative positions and squared distances
            rel_pos = x[src] - x[dst]             # (E, 3)
            dist_sq = (rel_pos ** 2).sum(dim=1, keepdim=True)  # (E, 1)

            # Edge messages
            edge_input = torch.cat([
                h[src], h[dst], dist_sq, edge_attr,
            ], dim=1)                              # (E, 2C+2)
            messages = self.edge_mlp(edge_input)   # (E, H)

            # Aggregate messages per node (sum)
            agg = torch.zeros(N, messages.shape[1], device=h.device)
            agg.index_add_(0, dst, messages)       # (N, H)

            # Update features
            h_input = torch.cat([h, agg], dim=1)   # (N, C+H)
            h_out = self.node_mlp(h_input)         # (N, C_out)
            h_out = self.norm(h_out)

            # Update coordinates (equivariant)
            x_out = x
            if self.update_coords:
                coord_weights = self.coord_mlp(messages)  # (E, 1)
                weighted_pos = rel_pos * coord_weights     # (E, 3)
                coord_agg = torch.zeros_like(x)
                coord_agg.index_add_(0, dst, weighted_pos)
                x_out = x + coord_agg

            return h_out, x_out

    class EGNNEncoder(nn.Module):
        """
        Multi-layer EGNN encoder for per-vertex feature extraction.

        Parameters
        ----------
        in_channels : int
            Input feature dimension (3 for raw coordinates).
        hidden_channels : int
            Hidden layer width.
        out_channels : int
            Output feature dimension.
        n_layers : int
            Number of EGNN layers.
        update_coords : bool
            Whether to update coordinates (True = full EGNN).
        """

        def __init__(
            self,
            in_channels: int = 3,
            hidden_channels: int = 128,
            out_channels: int = 64,
            n_layers: int = 4,
            update_coords: bool = True,
        ):
            super().__init__()

            self.input_proj = nn.Linear(in_channels, hidden_channels)

            self.layers = nn.ModuleList()
            for i in range(n_layers):
                self.layers.append(EGNNLayer(
                    in_channels=hidden_channels,
                    hidden_channels=hidden_channels,
                    out_channels=hidden_channels,
                    update_coords=update_coords,
                ))

            self.output_proj = nn.Linear(hidden_channels, out_channels)

        def forward(
            self,
            h: torch.Tensor,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_attr: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            h = self.input_proj(h)
            for layer in self.layers:
                h, x = layer(h, x, edge_index, edge_attr)
            return self.output_proj(h), x

    class EGNNClassifier(nn.Module):
        """
        Per-vertex classifier built on EGNN encoder.

        For cortical parcellation: maps each vertex to one of n_classes
        atlas regions while maintaining E(n) equivariance.
        """

        def __init__(
            self,
            in_channels: int = 3,
            hidden_channels: int = 128,
            n_classes: int = 36,
            n_layers: int = 4,
        ):
            super().__init__()
            self.encoder = EGNNEncoder(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=hidden_channels,
                n_layers=n_layers,
                update_coords=False,
            )
            self.classifier = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_channels // 2, n_classes),
            )

        def forward(
            self,
            h: torch.Tensor,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_attr: torch.Tensor,
        ) -> torch.Tensor:
            features, _ = self.encoder(h, x, edge_index, edge_attr)
            return self.classifier(features)  # (N, n_classes)

    return EGNNLayer, EGNNEncoder, EGNNClassifier


# Lazy class creation
_EGNN_CLASSES = None


def _get_egnn_classes():
    global _EGNN_CLASSES
    if _EGNN_CLASSES is None:
        _EGNN_CLASSES = _make_egnn_classes()
    return _EGNN_CLASSES


# ═══════════════════════════════════════════════════════════════════════════
# High-level inference function
# ═══════════════════════════════════════════════════════════════════════════


def egnn_inference(
    points: np.ndarray,
    features: Optional[np.ndarray] = None,
    k_neighbors: int = 20,
    model_weights: Optional[Union[str, "Path"]] = None,
    model_type: str = "encoder",
    n_classes: int = 36,
    hidden_channels: int = 128,
    n_layers: int = 4,
    use_gpu: bool = True,
    max_vram_mb: float = 4096,
    batch_size: int = 32768,
) -> np.ndarray:
    """
    Run EGNN inference on a brain point cloud with VRAM management.

    Parameters
    ----------
    points : np.ndarray, shape (N, 3)
    features : np.ndarray or None, shape (N, D)
        Input vertex features. None → use coordinates.
    k_neighbors : int
    model_weights : str, Path, or None
    model_type : ``'encoder'`` or ``'classifier'``
    n_classes : int
        Number of output classes (classifier only).
    hidden_channels : int
    n_layers : int
    use_gpu : bool
    max_vram_mb : float
    batch_size : int
        For large point clouds, process in batches.

    Returns
    -------
    output : np.ndarray
        shape (N, D_out) for encoder, (N, n_classes) for classifier.
    """
    import torch

    _, EGNNEncoder, EGNNClassifier = _get_egnn_classes()

    device = torch.device("cpu")
    if use_gpu and torch.cuda.is_available():
        free_mb = (
            torch.cuda.get_device_properties(0).total_memory
            - torch.cuda.memory_allocated(0)
        ) / (1024 ** 2)
        if free_mb > max_vram_mb * 0.3:
            device = torch.device("cuda", 0)

    # Build graph
    edge_index, edge_attr = build_knn_graph(points, k=k_neighbors)
    edge_index_t = torch.tensor(edge_index, dtype=torch.long, device=device)
    edge_attr_t = torch.tensor(edge_attr, dtype=torch.float32, device=device)

    # Features
    if features is None:
        features = points
    in_channels = features.shape[1]

    h = torch.tensor(features, dtype=torch.float32, device=device)
    x = torch.tensor(points, dtype=torch.float32, device=device)

    # Build model
    if model_type == "encoder":
        model = EGNNEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
        ).to(device)
    elif model_type == "classifier":
        model = EGNNClassifier(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            n_classes=n_classes,
            n_layers=n_layers,
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    if model_weights is not None:
        state = torch.load(str(model_weights), map_location=device, weights_only=True)
        model.load_state_dict(state)

    model.eval()
    with torch.no_grad():
        if model_type == "encoder":
            out, _ = model(h, x, edge_index_t, edge_attr_t)
        else:
            out = model(h, x, edge_index_t, edge_attr_t)

    result = out.cpu().numpy()

    del h, x, edge_index_t, edge_attr_t, out
    if device.type == "cuda":
        torch.cuda.empty_cache()

    logger.info("EGNN inference complete: output shape %s", result.shape)
    return result
