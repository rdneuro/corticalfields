"""
GP-based normative modeling on cortical surfaces.

This module implements the normative modeling pipeline:

    1. **Training** — Fit a Gaussian Process (with spectral Matérn
       kernel) on a reference cohort. The GP learns the joint
       distribution p(feature | vertex position) accounting for spatial
       covariance on the cortical manifold.

    2. **Prediction** — For each new patient, compute the posterior
       predictive distribution at every vertex:
         p(y* | x*, D_train) = N(μ*, σ*²)
       where μ* is the predicted mean and σ*² the predictive variance.

    3. **Anomaly scoring** — Compute vertex-wise z-scores and surprise
       (negative log-predictive density) from the posterior.

Scalability
-----------
Exact GP inference on ~150k vertices is intractable (O(N³)). We use:
  • **Spectral truncation**: L ≤ 1000 eigenpairs reduce the effective
    kernel rank, making kernel evaluations O(L) per pair.
  • **Variational Sparse GP (SVGP)**: m inducing points reduce
    training to O(N·m²) per step, with m ≈ 500–1000.
  • **Minibatch training**: stochastic variational inference allows
    training on subsets of vertices per epoch.

Vertex alignment
----------------
The SpectralMaternKernel operates on vertex **indices** from the LB
eigenvector matrix, which has N_lb rows (one per mesh vertex).
Feature data (thickness, curvature, etc.) may have a **different**
number of vertices N_data — for example, when the medial wall is
masked out, or features were projected to a different template.

To handle this, the model maintains a ``vertex_mask``:
  • If N_data == N_lb: no mask needed, 1-to-1 correspondence.
  • If N_data < N_lb: the user provides a boolean mask of shape
    (N_lb,) with exactly N_data True entries, or passes full-mesh
    data with NaN at excluded vertices (auto-detection).
  • If N_data > N_lb: error.

All internal operations (inducing point selection, kernel evaluation,
GP training/prediction) happen in LB-index space. The mask maps
between "data space" (what the user provides) and "mesh space"
(what the kernel needs).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    VariationalStrategy,
)
from gpytorch.mlls import VariationalELBO
from gpytorch.distributions import MultivariateNormal
from tqdm import trange

from corticalfields.spectral import LaplaceBeltrami
from corticalfields.kernels import SpectralMaternKernel

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Variational GP model (internal)
# ═══════════════════════════════════════════════════════════════════════════


class _SurfaceGP(ApproximateGP):
    """Sparse variational GP on the cortical surface."""

    def __init__(
        self,
        inducing_points: torch.Tensor,
        lb: LaplaceBeltrami,
        nu: float = 2.5,
        learn_inducing_locations: bool = False,
    ):
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0),
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points=inducing_points,
            variational_distribution=variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )
        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = SpectralMaternKernel(lb=lb, nu=nu)

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        mean = self.mean_module(x.float())
        covar = self.covar_module(x, x)
        return MultivariateNormal(mean, covar)


# ═══════════════════════════════════════════════════════════════════════════
# Result container
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class NormativeResult:
    """
    Result of normative prediction for a single subject.

    All arrays have the same shape as the input the user provided:
    (N_data,) if masked data was given, or (N_lb,) if full-mesh data.
    The model handles the mapping internally.
    """

    mean: np.ndarray
    variance: np.ndarray
    z_score: np.ndarray
    surprise: np.ndarray
    observed: np.ndarray
    feature_name: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# Vertex mask helpers
# ═══════════════════════════════════════════════════════════════════════════


def _resolve_vertex_mask(
    n_data: int,
    n_lb: int,
    vertex_mask: Optional[np.ndarray],
    data_for_nan_detection: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """
    Resolve the mapping between data-space and LB-mesh-space.

    Returns None when N_data == N_lb (no mapping needed), or a
    validated boolean mask of shape (N_lb,) otherwise.
    """
    # ── Perfect alignment: no mask needed ─────────────────────────
    if n_data == n_lb:
        if vertex_mask is not None:
            logger.info(
                "Data has %d vertices = LB mesh vertices. "
                "Ignoring vertex_mask (not needed).", n_data,
            )
        return None

    # ── Data has MORE vertices than mesh: impossible ──────────────
    if n_data > n_lb:
        raise ValueError(
            f"Data has {n_data} vertices but the LB eigenpairs were "
            f"computed on a mesh with only {n_lb} vertices. Data cannot "
            f"have more vertices than the mesh. Ensure you compute "
            f"eigenpairs on the same surface template that your features "
            f"are registered to."
        )

    # ── Data has FEWER vertices: we need a mask ───────────────────

    # Option 1: user provided a mask explicitly
    if vertex_mask is not None:
        mask = np.asarray(vertex_mask, dtype=bool)
        if mask.shape[0] != n_lb:
            raise ValueError(
                f"vertex_mask has {mask.shape[0]} entries but LB mesh "
                f"has {n_lb} vertices. Shape must be ({n_lb},)."
            )
        n_true = int(mask.sum())
        if n_true != n_data:
            raise ValueError(
                f"vertex_mask has {n_true} True entries but data has "
                f"{n_data} vertices. These must match exactly."
            )
        logger.info(
            "Using provided vertex_mask: %d/%d vertices valid (%.1f%%).",
            n_true, n_lb, 100 * n_true / n_lb,
        )
        return mask

    # Option 2: auto-detect from full-mesh data with NaN pattern
    if data_for_nan_detection is not None:
        d = data_for_nan_detection
        if d.shape[0] == n_lb:
            if d.ndim == 1:
                mask = np.isfinite(d)
            else:
                mask = np.any(np.isfinite(d), axis=1)

            n_true = int(mask.sum())
            if n_true == n_data:
                logger.info(
                    "Auto-detected vertex_mask from NaN pattern: "
                    "%d/%d vertices valid.", n_true, n_lb,
                )
                return mask

    # Option 3: cannot resolve — give a helpful error
    raise ValueError(
        f"Shape mismatch: data has {n_data} vertices but the LB "
        f"eigenpairs were computed on a mesh with {n_lb} vertices.\n\n"
        f"This typically happens when:\n"
        f"  • The medial wall was excluded from the feature data but\n"
        f"    not from the surface mesh used for eigenpairs.\n"
        f"  • Features were projected to a different resolution template.\n\n"
        f"To fix this, either:\n"
        f"  (a) Provide vertex_mask= a boolean array of shape ({n_lb},)\n"
        f"      with exactly {n_data} True entries. Example:\n"
        f"        model.fit(data, vertex_mask=~medial_wall_mask)\n\n"
        f"  (b) Pass FULL mesh-space data (shape ({n_lb},) or ({n_lb}, S))\n"
        f"      with NaN at excluded vertices — auto-detection will work.\n\n"
        f"  (c) Recompute the LB eigenpairs on the same surface that\n"
        f"      your features are registered to (recommended)."
    )


def _data_to_mesh(
    data: np.ndarray,
    mask: Optional[np.ndarray],
    n_lb: int,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Expand data-space → full LB-mesh-space (fills masked vertices)."""
    if mask is None:
        return data.copy()
    if data.ndim == 1:
        full = np.full(n_lb, fill_value, dtype=np.float64)
        full[mask] = data
    else:
        full = np.full((n_lb, data.shape[1]), fill_value, dtype=np.float64)
        full[mask] = data
    return full


def _mesh_to_data(full: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """Extract data-space from full LB-mesh-space."""
    if mask is None:
        return full
    return full[mask]


# ═══════════════════════════════════════════════════════════════════════════
# Normative model
# ═══════════════════════════════════════════════════════════════════════════


class CorticalNormativeModel:
    """
    GP-based normative model for a single cortical feature.

    Handles vertex-mesh alignment transparently via vertex_mask.

    Parameters
    ----------
    lb : LaplaceBeltrami
        Pre-computed eigenpairs for the reference mesh template.
    nu : float
        Matérn smoothness parameter (½, 3/2, 5/2, or inf).
    n_inducing : int
        Number of inducing points for SVGP.
    device : str or torch.device
        ``'cuda'`` or ``'cpu'``.

    Examples
    --------
    >>> # CASE 1: Data matches LB mesh — no mask needed
    >>> model = CorticalNormativeModel(lb, nu=2.5, n_inducing=512)
    >>> model.fit(train_features)  # shape (N_lb,) or (N_lb, S)
    >>>
    >>> # CASE 2: Data has fewer vertices (medial wall excluded)
    >>> # Option A — provide a vertex_mask explicitly
    >>> model.fit(data, vertex_mask=~medial_wall_mask)
    >>>
    >>> # Option B — pass full-mesh data with NaN at excluded vertices
    >>> full_data = np.full(N_lb, np.nan)
    >>> full_data[~medial_wall] = reduced_data
    >>> model.fit(full_data)  # auto-detects mask
    """

    def __init__(
        self,
        lb: LaplaceBeltrami,
        nu: float = 2.5,
        n_inducing: int = 512,
        device: Union[str, torch.device] = "cpu",
        seed: int = 42,
    ):
        self.lb = lb
        self.nu = nu
        self.n_inducing = n_inducing
        self.device = torch.device(device)
        self.seed = seed

        self._model: Optional[_SurfaceGP] = None
        self._likelihood: Optional[gpytorch.likelihoods.GaussianLikelihood] = None
        self._is_fitted = False
        self._feature_name = ""
        self._train_mean: float = 0.0
        self._train_std: float = 1.0

        # Vertex alignment state
        self._vertex_mask: Optional[np.ndarray] = None  # shape (N_lb,) bool
        self._n_data: int = 0

    @property
    def n_mesh_vertices(self) -> int:
        """Number of vertices in the LB mesh."""
        return self.lb.n_vertices

    @property
    def n_data_vertices(self) -> int:
        """Number of valid data vertices (after masking)."""
        return self._n_data if self._is_fitted else self.n_mesh_vertices

    @property
    def vertex_mask(self) -> Optional[np.ndarray]:
        """Boolean mask (N_lb,) mapping mesh → data, or None."""
        return self._vertex_mask

    def _select_inducing_points(
        self,
        valid_vertex_indices: np.ndarray,
    ) -> torch.Tensor:
        """
        Farthest-point sampling in GPS space, restricted to valid vertices.

        The GPS embedding lives in LB-mesh space (one row per mesh vertex).
        We restrict FPS to only sample from vertices that have valid data.
        The returned indices are in LB-mesh space (ready for the kernel).
        """
        rng = np.random.RandomState(self.seed)

        from corticalfields.spectral import global_point_signature
        gps_full = global_point_signature(self.lb, n_components=20)

        # Restrict GPS to valid vertices only
        gps_valid = gps_full[valid_vertex_indices]  # (N_valid, 20)
        N_valid = len(valid_vertex_indices)

        n = min(self.n_inducing, N_valid)
        local_indices = np.zeros(n, dtype=np.int64)
        local_indices[0] = rng.randint(N_valid)
        dists = np.full(N_valid, np.inf)

        for i in range(1, n):
            new_dists = np.linalg.norm(
                gps_valid - gps_valid[local_indices[i - 1]], axis=1,
            )
            dists = np.minimum(dists, new_dists)
            local_indices[i] = np.argmax(dists)

        # Map local indices → LB mesh indices
        mesh_indices = valid_vertex_indices[local_indices]

        logger.info(
            "Selected %d inducing points via FPS in GPS space "
            "(%d valid / %d mesh vertices).",
            n, N_valid, self.n_mesh_vertices,
        )

        return torch.tensor(mesh_indices, dtype=torch.long).unsqueeze(-1).to(self.device)

    def fit(
        self,
        train_features: np.ndarray,
        feature_name: str = "thickness",
        vertex_mask: Optional[np.ndarray] = None,
        n_epochs: int = 100,
        lr: float = 0.01,
        batch_size: int = 4096,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train the normative GP model on a reference cohort.

        Parameters
        ----------
        train_features : np.ndarray, shape (N,) or (N, S)
            Per-vertex feature values. N can be N_lb (full mesh) or
            < N_lb (masked). If 2D, columns are subjects.
        feature_name : str
            Name of the feature being modelled.
        vertex_mask : np.ndarray, shape (N_lb,), dtype bool, or None
            Which LB mesh vertices have data. Required when N < N_lb
            unless auto-detection from NaN works.
        n_epochs : int
            Training epochs.
        lr : float
            Learning rate.
        batch_size : int
            Minibatch size (vertices per step).
        verbose : bool
            Show progress bar.

        Returns
        -------
        history : dict with 'loss' key
        """
        self._feature_name = feature_name
        N_lb = self.n_mesh_vertices

        # ── Multi-subject → single training vector ───────────────
        if train_features.ndim == 2:
            N_rows, S = train_features.shape
            logger.info(
                "Training on mean of %d subjects (%d vertices each).", S, N_rows,
            )
            y_raw = np.nanmean(train_features, axis=1)
        else:
            N_rows = train_features.shape[0]
            y_raw = train_features.copy()

        # ── Resolve vertex mask ──────────────────────────────────
        self._vertex_mask = _resolve_vertex_mask(
            n_data=N_rows,
            n_lb=N_lb,
            vertex_mask=vertex_mask,
            data_for_nan_detection=(train_features if N_rows == N_lb else None),
        )

        # ── Map to full mesh space ───────────────────────────────
        if self._vertex_mask is not None:
            y_mesh = _data_to_mesh(y_raw, self._vertex_mask, N_lb, fill_value=np.nan)
            valid_mask = self._vertex_mask & np.isfinite(y_mesh)
        else:
            y_mesh = y_raw
            valid_mask = np.isfinite(y_mesh)

        self._n_data = int(valid_mask.sum())
        logger.info(
            "Vertex alignment: %d mesh, %d valid data (%.1f%%).",
            N_lb, self._n_data, 100 * self._n_data / N_lb,
        )

        # ── Valid vertex indices (LB mesh space) ─────────────────
        valid_idx = np.where(valid_mask)[0].astype(np.int64)
        y_valid = y_mesh[valid_mask]

        # ── Normalise ────────────────────────────────────────────
        self._train_mean = float(np.nanmean(y_valid))
        self._train_std = float(np.nanstd(y_valid))
        if self._train_std < 1e-8:
            self._train_std = 1.0
        y_norm = (y_valid - self._train_mean) / self._train_std

        # ── Tensors ──────────────────────────────────────────────
        x_train = torch.tensor(valid_idx, dtype=torch.long).unsqueeze(-1).to(self.device)
        y_train_t = torch.tensor(y_norm, dtype=torch.float64).to(self.device)
        N_valid = len(valid_idx)

        # ── Inducing points (in mesh space, from valid only) ─────
        inducing_pts = self._select_inducing_points(valid_idx)

        # ── Build model ──────────────────────────────────────────
        self._likelihood = gpytorch.likelihoods.GaussianLikelihood().double()
        self._likelihood = self._likelihood.to(self.device)
        self._model = _SurfaceGP(
            inducing_points=inducing_pts, lb=self.lb, nu=self.nu,
        ).double().to(self.device)

        self._model.train()
        self._likelihood.train()

        optimizer = torch.optim.Adam(
            list(self._model.parameters()) + list(self._likelihood.parameters()),
            lr=lr,
        )
        mll = VariationalELBO(self._likelihood, self._model, num_data=N_valid)

        # ── Training loop ────────────────────────────────────────
        history: Dict[str, List[float]] = {"loss": []}
        iterator = trange(n_epochs, desc=f"Training [{feature_name}]", disable=not verbose)

        for epoch in iterator:
            perm = torch.randperm(N_valid, device=self.device)[:batch_size]
            x_batch = x_train[perm]
            y_batch = y_train_t[perm]

            optimizer.zero_grad()
            output = self._model(x_batch)
            loss = -mll(output, y_batch)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            history["loss"].append(loss_val)

            if verbose:
                ls = self._model.covar_module.lengthscale.item()
                noise = self._likelihood.noise.item()
                iterator.set_postfix(
                    loss=f"{loss_val:.3f}", ls=f"{ls:.2f}", noise=f"{noise:.4f}",
                )

        self._is_fitted = True
        logger.info(
            "Training complete. Loss: %.4f, lengthscale: %.2f mm, noise: %.4f",
            history["loss"][-1],
            self._model.covar_module.lengthscale.item(),
            self._likelihood.noise.item(),
        )
        return history

    @torch.no_grad()
    def predict(
        self,
        observed_features: np.ndarray,
        vertex_mask: Optional[np.ndarray] = None,
        batch_size: int = 5000,
    ) -> NormativeResult:
        """
        Compute normative predictions and anomaly scores for a patient.

        Parameters
        ----------
        observed_features : np.ndarray, shape (N,)
            Observed per-vertex feature values. N can be N_lb (full
            mesh) or N_data (masked — must match training).
        vertex_mask : np.ndarray or None
            Overrides the training mask for this patient if provided.
        batch_size : int
            Prediction batch size.

        Returns
        -------
        NormativeResult
            All arrays match the input shape (N,).
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        self._model.eval()
        self._likelihood.eval()

        N_input = observed_features.shape[0]
        N_lb = self.n_mesh_vertices

        # ── Resolve mask for prediction ──────────────────────────
        pred_mask = vertex_mask if vertex_mask is not None else self._vertex_mask

        # ── Map to mesh space ────────────────────────────────────
        if N_input == N_lb:
            obs_mesh = observed_features.copy()
            if pred_mask is not None:
                valid_mask = pred_mask & np.isfinite(obs_mesh)
            else:
                valid_mask = np.isfinite(obs_mesh)
            return_in_mesh_space = True

        elif pred_mask is not None and N_input == int(pred_mask.sum()):
            obs_mesh = _data_to_mesh(observed_features, pred_mask, N_lb, fill_value=np.nan)
            valid_mask = pred_mask & np.isfinite(obs_mesh)
            return_in_mesh_space = False

        elif self._vertex_mask is not None and N_input == self._n_data:
            pred_mask = self._vertex_mask
            obs_mesh = _data_to_mesh(observed_features, pred_mask, N_lb, fill_value=np.nan)
            valid_mask = pred_mask & np.isfinite(obs_mesh)
            return_in_mesh_space = False

        else:
            raise ValueError(
                f"Cannot align patient data ({N_input} vertices) with "
                f"LB mesh ({N_lb} vertices). Expected {N_lb} or "
                f"{self._n_data} vertices, or provide vertex_mask."
            )

        # ── Normalise ────────────────────────────────────────────
        y_norm_mesh = (obs_mesh - self._train_mean) / self._train_std

        # ── Predict at valid vertices ────────────────────────────
        valid_idx = np.where(valid_mask)[0].astype(np.int64)
        N_valid = len(valid_idx)

        means_mesh = np.full(N_lb, np.nan, dtype=np.float64)
        vars_mesh = np.full(N_lb, np.nan, dtype=np.float64)

        for start in range(0, N_valid, batch_size):
            end = min(start + batch_size, N_valid)
            batch_idx = valid_idx[start:end]
            x_batch = torch.tensor(batch_idx, dtype=torch.long).unsqueeze(-1).to(self.device)
            pred = self._likelihood(self._model(x_batch))
            means_mesh[batch_idx] = pred.mean.cpu().numpy()
            vars_mesh[batch_idx] = pred.variance.cpu().numpy()

        # ── Scores in mesh space ─────────────────────────────────
        means_orig = means_mesh * self._train_std + self._train_mean
        vars_orig = vars_mesh * (self._train_std ** 2)

        std_norm = np.sqrt(np.maximum(vars_mesh, 1e-12))
        z_scores = np.full(N_lb, np.nan, dtype=np.float64)
        z_scores[valid_mask] = (
            (y_norm_mesh[valid_mask] - means_mesh[valid_mask]) / std_norm[valid_mask]
        )

        surprise = np.full(N_lb, np.nan, dtype=np.float64)
        surprise[valid_mask] = (
            0.5 * np.log(2.0 * np.pi * np.maximum(vars_mesh[valid_mask], 1e-12))
            + 0.5 * z_scores[valid_mask] ** 2
        )

        # ── Return in input space ────────────────────────────────
        if return_in_mesh_space:
            return NormativeResult(
                mean=means_orig, variance=vars_orig,
                z_score=z_scores, surprise=surprise,
                observed=observed_features, feature_name=self._feature_name,
            )
        else:
            return NormativeResult(
                mean=_mesh_to_data(means_orig, pred_mask),
                variance=_mesh_to_data(vars_orig, pred_mask),
                z_score=_mesh_to_data(z_scores, pred_mask),
                surprise=_mesh_to_data(surprise, pred_mask),
                observed=observed_features, feature_name=self._feature_name,
            )

    # ── Persistence ───────────────────────────────────────────────

    def save(self, path: Union[str, Path]) -> None:
        """Save the trained model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        torch.save(self._model.state_dict(), path / "model_state.pt")
        torch.save(self._likelihood.state_dict(), path / "likelihood_state.pt")

        meta = {
            "nu": self.nu,
            "n_inducing": self.n_inducing,
            "train_mean": self._train_mean,
            "train_std": self._train_std,
            "feature_name": self._feature_name,
            "n_eigenpairs": self.lb.n_eigenpairs,
            "n_lb_vertices": self.lb.n_vertices,
            "n_data_vertices": self._n_data,
        }
        np.savez(path / "meta.npz", **meta)
        if self._vertex_mask is not None:
            np.save(path / "vertex_mask.npy", self._vertex_mask)
        logger.info("Model saved to %s", path)

    def load(self, path: Union[str, Path]) -> None:
        """Load a previously trained model from disk."""
        path = Path(path)

        meta = dict(np.load(path / "meta.npz", allow_pickle=True))
        self._train_mean = float(meta["train_mean"])
        self._train_std = float(meta["train_std"])
        self._feature_name = str(meta["feature_name"])
        self._n_data = int(meta["n_data_vertices"])

        mask_path = path / "vertex_mask.npy"
        self._vertex_mask = np.load(mask_path) if mask_path.exists() else None

        if self._vertex_mask is not None:
            valid_idx = np.where(self._vertex_mask)[0].astype(np.int64)
        else:
            valid_idx = np.arange(self.lb.n_vertices, dtype=np.int64)

        inducing_pts = self._select_inducing_points(valid_idx)
        self._likelihood = gpytorch.likelihoods.GaussianLikelihood().double()
        self._model = _SurfaceGP(
            inducing_points=inducing_pts, lb=self.lb, nu=self.nu,
        ).double()

        self._model.load_state_dict(
            torch.load(path / "model_state.pt", map_location=self.device)
        )
        self._likelihood.load_state_dict(
            torch.load(path / "likelihood_state.pt", map_location=self.device)
        )
        self._model = self._model.to(self.device)
        self._likelihood = self._likelihood.to(self.device)
        self._is_fitted = True
        logger.info("Model loaded from %s", path)

    def release_gpu(self) -> None:
        """
        Move model and likelihood to CPU and free GPU VRAM.

        Call this after prediction when processing multiple features
        sequentially to free VRAM between features. The model remains
        usable on CPU (slower) or can be moved back with
        ``model._model.to('cuda')``.
        """
        if self._model is not None:
            self._model = self._model.cpu()
        if self._likelihood is not None:
            self._likelihood = self._likelihood.cpu()
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("NormativeModel: GPU resources released.")