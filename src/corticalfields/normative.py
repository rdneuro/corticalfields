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

The combination enables training in ~10–30 minutes per feature on
a single GPU (RTX 3090/4070).
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
# Variational GP model
# ═══════════════════════════════════════════════════════════════════════════


class _SurfaceGP(ApproximateGP):
    """
    Sparse variational GP on the cortical surface.

    Uses vertex indices as inputs (not 3D coordinates) and a
    SpectralMaternKernel that evaluates the kernel via the
    pre-computed LB eigenpairs.
    """

    def __init__(
        self,
        inducing_points: torch.Tensor,
        lb: LaplaceBeltrami,
        nu: float = 2.5,
        learn_inducing_locations: bool = False,
    ):
        # Variational distribution: Cholesky parameterisation
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0),
        )
        # Variational strategy
        variational_strategy = VariationalStrategy(
            self,
            inducing_points=inducing_points,
            variational_distribution=variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )
        super().__init__(variational_strategy)

        # Mean function: constant (learned)
        self.mean_module = gpytorch.means.ConstantMean()

        # Covariance: spectral Matérn on the cortical manifold
        self.covar_module = SpectralMaternKernel(lb=lb, nu=nu)

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        """
        Compute the prior GP distribution at vertex indices x.
        """
        mean = self.mean_module(x.float())
        covar = self.covar_module(x, x)
        return MultivariateNormal(mean, covar)


# ═══════════════════════════════════════════════════════════════════════════
# Normative model
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class NormativeResult:
    """
    Result of normative prediction for a single subject.

    Attributes
    ----------
    mean : np.ndarray, shape (N,)
        Posterior predictive mean (expected feature value under the norm).
    variance : np.ndarray, shape (N,)
        Posterior predictive variance.
    z_score : np.ndarray, shape (N,)
        Standardised deviation: (observed − mean) / std.
    surprise : np.ndarray, shape (N,)
        Negative log-predictive density: −log p(y | x, model).
    observed : np.ndarray, shape (N,)
        The observed feature values.
    feature_name : str
        Name of the feature modelled.
    """

    mean: np.ndarray
    variance: np.ndarray
    z_score: np.ndarray
    surprise: np.ndarray
    observed: np.ndarray
    feature_name: str = ""


class CorticalNormativeModel:
    """
    GP-based normative model for a single cortical feature.

    This is the main user-facing class for normative modeling. It
    wraps training, prediction, and anomaly scoring into a clean API.

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
    >>> # 1. Build LB eigenpairs on the template surface
    >>> lb = compute_eigenpairs(template_verts, template_faces, n_eigenpairs=300)
    >>>
    >>> # 2. Create normative model
    >>> model = CorticalNormativeModel(lb, nu=2.5, n_inducing=512)
    >>>
    >>> # 3. Train on reference cohort
    >>> model.fit(train_features, n_epochs=100, lr=0.01)
    >>>
    >>> # 4. Score a patient
    >>> result = model.predict(patient_features)
    >>> surprise_map = result.surprise  # vertex-wise anomaly scores
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

        # Training statistics for normalisation
        self._train_mean: float = 0.0
        self._train_std: float = 1.0

    def _select_inducing_points(
        self, n_vertices: int,
    ) -> torch.Tensor:
        """
        Select inducing point vertex indices.

        Uses farthest-point sampling in the spectral embedding (GPS)
        for geometrically uniform coverage of the cortical surface.
        Falls back to random sampling if GPS is unavailable.
        """
        rng = np.random.RandomState(self.seed)

        # Use GPS embedding for FPS
        from corticalfields.spectral import global_point_signature

        gps_embed = global_point_signature(self.lb, n_components=20)

        # Farthest point sampling in GPS space
        n = min(self.n_inducing, n_vertices)
        indices = np.zeros(n, dtype=np.int64)
        indices[0] = rng.randint(n_vertices)
        dists = np.full(n_vertices, np.inf)

        for i in range(1, n):
            new_dists = np.linalg.norm(
                gps_embed - gps_embed[indices[i - 1]], axis=1,
            )
            dists = np.minimum(dists, new_dists)
            indices[i] = np.argmax(dists)

        logger.info(
            "Selected %d inducing points via farthest-point sampling in GPS space.",
            n,
        )

        return torch.tensor(indices, dtype=torch.long).unsqueeze(-1).to(self.device)

    def fit(
        self,
        train_features: np.ndarray,
        feature_name: str = "thickness",
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
            Per-vertex feature values. If 2D, each column is a subject;
            the model is trained on the mean across subjects (for the
            population-level normative model), and the variance informs
            the likelihood noise.
        feature_name : str
            Name of the feature (for bookkeeping).
        n_epochs : int
            Number of training epochs (passes through inducing pts).
        lr : float
            Learning rate for Adam.
        batch_size : int
            Minibatch size (number of vertices per step).
        verbose : bool
            Show progress bar.

        Returns
        -------
        history : dict
            Training loss history (``'loss'`` key).
        """
        self._feature_name = feature_name

        # Handle multi-subject input: mean across subjects
        if train_features.ndim == 2:
            logger.info(
                "Training on mean of %d subjects (%d vertices each).",
                train_features.shape[1], train_features.shape[0],
            )
            y_train = np.nanmean(train_features, axis=1)
        else:
            y_train = train_features.copy()

        N = y_train.shape[0]

        # Normalise (z-score) for stable GP training
        self._train_mean = float(np.nanmean(y_train))
        self._train_std = float(np.nanstd(y_train))
        if self._train_std < 1e-8:
            self._train_std = 1.0
        y_norm = (y_train - self._train_mean) / self._train_std

        # Replace NaN with 0 (masked vertices)
        nan_mask = np.isnan(y_norm)
        y_norm[nan_mask] = 0.0

        # Vertex indices as input
        x_train = torch.arange(N, dtype=torch.long).unsqueeze(-1).to(self.device)
        y_train_t = torch.tensor(y_norm, dtype=torch.float64).to(self.device)

        # Select inducing points
        inducing_pts = self._select_inducing_points(N)

        # Build model
        self._likelihood = gpytorch.likelihoods.GaussianLikelihood().double()
        self._likelihood = self._likelihood.to(self.device)

        self._model = _SurfaceGP(
            inducing_points=inducing_pts,
            lb=self.lb,
            nu=self.nu,
        ).double()
        self._model = self._model.to(self.device)

        # Training mode
        self._model.train()
        self._likelihood.train()

        # Optimiser
        optimizer = torch.optim.Adam(
            list(self._model.parameters()) + list(self._likelihood.parameters()),
            lr=lr,
        )

        # Variational ELBO objective
        mll = VariationalELBO(self._likelihood, self._model, num_data=N)

        # Training loop
        history: Dict[str, List[float]] = {"loss": []}
        iterator = trange(n_epochs, desc=f"Training [{feature_name}]", disable=not verbose)

        for epoch in iterator:
            # Minibatch sampling
            perm = torch.randperm(N, device=self.device)[:batch_size]
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
            "Training complete. Final loss: %.4f, lengthscale: %.2f mm, noise: %.4f",
            history["loss"][-1],
            self._model.covar_module.lengthscale.item(),
            self._likelihood.noise.item(),
        )

        return history

    @torch.no_grad()
    def predict(
        self,
        observed_features: np.ndarray,
        batch_size: int = 5000,
    ) -> NormativeResult:
        """
        Compute normative predictions and anomaly scores for a patient.

        Parameters
        ----------
        observed_features : np.ndarray, shape (N,)
            Observed per-vertex feature values for the patient.
        batch_size : int
            Prediction batch size.

        Returns
        -------
        NormativeResult
            Contains mean, variance, z-score, and surprise maps.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        self._model.eval()
        self._likelihood.eval()

        N = observed_features.shape[0]

        # Normalise observed values using training statistics
        y_obs_norm = (observed_features - self._train_mean) / self._train_std

        # Predict in batches
        means = np.zeros(N, dtype=np.float64)
        variances = np.zeros(N, dtype=np.float64)

        x_all = torch.arange(N, dtype=torch.long).unsqueeze(-1).to(self.device)

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            x_batch = x_all[start:end]

            pred = self._likelihood(self._model(x_batch))
            means[start:end] = pred.mean.cpu().numpy()
            variances[start:end] = pred.variance.cpu().numpy()

        # De-normalise to original scale
        means_orig = means * self._train_std + self._train_mean
        variances_orig = variances * (self._train_std ** 2)

        # Z-scores in normalised space (more statistically appropriate)
        std_norm = np.sqrt(np.maximum(variances, 1e-12))
        z_scores = (y_obs_norm - means) / std_norm

        # Surprise: negative log-predictive density under Gaussian
        # -log N(y | μ, σ²) = ½ log(2πσ²) + (y - μ)² / (2σ²)
        surprise = (
            0.5 * np.log(2.0 * np.pi * np.maximum(variances, 1e-12))
            + 0.5 * (y_obs_norm - means) ** 2 / np.maximum(variances, 1e-12)
        )

        return NormativeResult(
            mean=means_orig,
            variance=variances_orig,
            z_score=z_scores,
            surprise=surprise,
            observed=observed_features,
            feature_name=self._feature_name,
        )

    def save(self, path: Union[str, Path]) -> None:
        """Save the trained model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Model state
        torch.save(self._model.state_dict(), path / "model_state.pt")
        torch.save(self._likelihood.state_dict(), path / "likelihood_state.pt")

        # Training statistics
        meta = {
            "nu": self.nu,
            "n_inducing": self.n_inducing,
            "train_mean": self._train_mean,
            "train_std": self._train_std,
            "feature_name": self._feature_name,
            "n_eigenpairs": self.lb.n_eigenpairs,
            "n_vertices": self.lb.n_vertices,
        }
        np.savez(path / "meta.npz", **meta)
        logger.info("Model saved to %s", path)

    def load(self, path: Union[str, Path]) -> None:
        """Load a previously trained model from disk."""
        path = Path(path)

        meta = dict(np.load(path / "meta.npz", allow_pickle=True))
        self._train_mean = float(meta["train_mean"])
        self._train_std = float(meta["train_std"])
        self._feature_name = str(meta["feature_name"])

        # Reconstruct model with same inducing points
        inducing_pts = self._select_inducing_points(self.lb.n_vertices)
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
