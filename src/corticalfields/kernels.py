"""
Spectral Matérn kernels on cortical surfaces for GPyTorch.

This module implements the Matérn kernel on a Riemannian manifold via the
spectral approach of Borovitskiy, Terenin, Mostowsky & Deisenroth
(NeurIPS 2020):

    k(x, y) = σ² · Σ_{i=0}^{L-1} S_ν(λ_i) · φ_i(x) · φ_i(y)

where φ_i, λ_i are eigenfunctions/eigenvalues of the Laplace–Beltrami
operator, and the spectral density is:

    S_ν(λ) = ( 2ν/κ² + λ )^{−(ν + d/2)}

with ν (smoothness), κ (inverse lengthscale), d (manifold dimension = 2),
and σ² (output scale).

Why this matters
----------------
Naïvely replacing Euclidean distance with geodesic distance in the
standard Matérn formula does NOT produce a valid positive-definite kernel
on curved manifolds (Feragen et al., CVPR 2015). The spectral construction
is the only correct way to define Matérn-class GPs on the cortex.

The resulting kernel:
  • Respects the intrinsic geometry of the cortical surface.
  • Never requires pairwise geodesic distances (only eigenpairs).
  • Scales to ~150k vertices via truncated spectral approximation.
  • Integrates natively with GPyTorch for SVGP inference.

References
----------
[1] V. Borovitskiy et al., "Matérn Gaussian Processes on Riemannian
    Manifolds", NeurIPS 2020.
[2] GeometricKernels library: https://github.com/geometric-kernels/GeometricKernels
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import gpytorch
from gpytorch.kernels import Kernel
from gpytorch.constraints import Positive
import numpy as np

from corticalfields.spectral import LaplaceBeltrami

logger = logging.getLogger(__name__)


class SpectralMaternKernel(Kernel):
    """
    Matérn kernel on a Riemannian manifold (cortical surface) via
    truncated spectral decomposition of the Laplace–Beltrami operator.

    The kernel is parameterised by:
        • ``nu`` (ν) — smoothness, fixed at construction.
          Common choices: 1/2 (exponential), 3/2, 5/2, ∞ (squared-exp).
        • ``lengthscale`` (κ⁻¹) — spatial extent of correlations in mm.
          Learned during GP training.
        • ``outputscale`` (σ²) — signal variance.
          Learned during GP training.

    Inputs to ``forward()`` are **vertex indices** (integer tensors),
    not 3D coordinates. The kernel looks up the pre-stored eigenvectors
    at those indices and computes the spectral sum.

    Parameters
    ----------
    lb : LaplaceBeltrami
        Pre-computed LB eigenpairs.
    nu : float
        Smoothness parameter (½, 3/2, 5/2, or ``float('inf')`` for
        the squared exponential / heat kernel).
    dim : int
        Manifold dimension (2 for cortical surfaces).
    lengthscale_prior : gpytorch.priors.Prior or None
        Prior on the lengthscale (optional).

    Examples
    --------
    >>> lb = compute_eigenpairs(vertices, faces, n_eigenpairs=300)
    >>> kernel = SpectralMaternKernel(lb, nu=2.5)
    >>> # Kernel matrix for vertex indices [0, 1, 2, 3]
    >>> idx = torch.tensor([[0], [1], [2], [3]])
    >>> K = kernel(idx, idx).evaluate()
    >>> K.shape
    torch.Size([4, 4])
    """

    has_lengthscale = True

    def __init__(
        self,
        lb: LaplaceBeltrami,
        nu: float = 2.5,
        dim: int = 2,
        lengthscale_prior: Optional[gpytorch.priors.Prior] = None,
        outputscale_prior: Optional[gpytorch.priors.Prior] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.nu = nu
        self.dim = dim

        # Store eigenpairs as buffers (not parameters — not optimised)
        eigenvalues = torch.from_numpy(lb.eigenvalues).double()
        eigenvectors = torch.from_numpy(lb.eigenvectors).double()

        self.register_buffer("eigenvalues", eigenvalues)
        self.register_buffer("eigenvectors", eigenvectors)

        # Trainable output scale σ²
        self.register_parameter(
            "raw_outputscale",
            torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64)),
        )
        outputscale_constraint = Positive()
        self.register_constraint("raw_outputscale", outputscale_constraint)

        if outputscale_prior is not None:
            self.register_prior(
                "outputscale_prior",
                outputscale_prior,
                lambda m: m.outputscale,
                lambda m, v: m._set_outputscale(v),
            )

        if lengthscale_prior is not None:
            self.register_prior(
                "lengthscale_prior",
                lengthscale_prior,
                lambda m: m.lengthscale,
                lambda m, v: m._set_lengthscale(v),
            )

        logger.info(
            "SpectralMaternKernel: ν=%.1f, dim=%d, %d eigenpairs, %d vertices",
            nu, dim, lb.n_eigenpairs, lb.n_vertices,
        )

    @property
    def outputscale(self) -> torch.Tensor:
        return self.raw_outputscale_constraint.transform(self.raw_outputscale)

    @outputscale.setter
    def outputscale(self, value: float):
        self._set_outputscale(value)

    def _set_outputscale(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_outputscale)
        self.initialize(
            raw_outputscale=self.raw_outputscale_constraint.inverse_transform(value)
        )

    def _spectral_density(self, eigenvalues: torch.Tensor) -> torch.Tensor:
        """
        Compute the spectral density S_ν(λ) for each eigenvalue.

        S_ν(λ) = (2ν / κ² + λ)^{−(ν + d/2)}

        where κ = 1 / lengthscale. For ν = ∞, this reduces to the
        heat kernel: S(λ) = exp(−λ · lengthscale² / 2).
        """
        # lengthscale shape: (..., 1) from GPyTorch Kernel base
        ls = self.lengthscale.squeeze(-1)  # scalar or batch
        kappa_sq = 1.0 / (ls ** 2 + 1e-12)

        if math.isinf(self.nu):
            # Squared exponential / heat kernel limit
            return torch.exp(-eigenvalues * (ls ** 2) / 2.0)

        # General Matérn spectral density
        exponent = -(self.nu + self.dim / 2.0)
        base = 2.0 * self.nu * kappa_sq + eigenvalues
        return base ** exponent

    def _feature_matrix(
        self, vertex_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build the feature matrix Φ̃ such that k(x, y) ≈ σ² · Φ̃(x) · Φ̃(y)ᵀ.

        Φ̃(x)_i = √S_ν(λ_i) · φ_i(x)

        Parameters
        ----------
        vertex_indices : (B,) or (B, 1) long tensor
            Vertex indices into the mesh.

        Returns
        -------
        features : (B, L) tensor
            Scaled eigenvector rows.
        """
        idx = vertex_indices.long().squeeze(-1)  # (B,)
        phi = self.eigenvectors[idx]  # (B, L)

        S = self._spectral_density(self.eigenvalues)  # (L,)
        sqrt_S = torch.sqrt(torch.clamp(S, min=1e-16))  # (L,)

        return phi * sqrt_S.unsqueeze(0)  # (B, L)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params,
    ) -> torch.Tensor:
        """
        Evaluate the kernel matrix K(x1, x2).

        Parameters
        ----------
        x1 : (N1, 1) tensor of vertex indices
        x2 : (N2, 1) tensor of vertex indices
        diag : bool
            If True, return only the diagonal of K.

        Returns
        -------
        K : (N1, N2) or (N1,) tensor
        """
        feat1 = self._feature_matrix(x1)  # (N1, L)
        feat2 = self._feature_matrix(x2)  # (N2, L)

        sigma_sq = self.outputscale

        if diag:
            return sigma_sq * (feat1 * feat2).sum(dim=-1)
        else:
            return sigma_sq * feat1 @ feat2.t()


# ═══════════════════════════════════════════════════════════════════════════
# Heat kernel (special case: ν → ∞)
# ═══════════════════════════════════════════════════════════════════════════


class SpectralHeatKernel(SpectralMaternKernel):
    """
    Heat (Gaussian / squared-exponential) kernel on a cortical surface.

    This is the ν → ∞ limit of the Matérn kernel, corresponding to
    infinitely smooth sample paths. The spectral density becomes:

        S(λ) = exp(−λ · t)

    where t = lengthscale² / 2 is the diffusion time.
    """

    def __init__(self, lb: LaplaceBeltrami, **kwargs):
        super().__init__(lb, nu=float("inf"), **kwargs)


# ═══════════════════════════════════════════════════════════════════════════
# Utility: kernel matrix for all vertices
# ═══════════════════════════════════════════════════════════════════════════


def full_kernel_matrix(
    kernel: SpectralMaternKernel,
    batch_size: int = 5000,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Compute the full N×N kernel matrix in batches (for visualization
    or debugging; NOT for GP inference at scale).

    Parameters
    ----------
    kernel : SpectralMaternKernel
    batch_size : int
        Number of rows to compute at once.
    device : torch.device or None

    Returns
    -------
    K : np.ndarray, shape (N, N)
    """
    N = kernel.eigenvectors.shape[0]
    K = np.zeros((N, N), dtype=np.float64)

    all_idx = torch.arange(N, dtype=torch.long).unsqueeze(-1)
    if device is not None:
        all_idx = all_idx.to(device)
        kernel = kernel.to(device)

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_idx = all_idx[start:end]
            K_batch = kernel(batch_idx, all_idx).evaluate()
            K[start:end, :] = K_batch.cpu().numpy()

    return K
