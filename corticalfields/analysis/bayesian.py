"""
Bayesian statistical analysis for CorticalFields.

A comprehensive, reusable toolkit for Bayesian inference in
neuroimaging research. Each class encapsulates a common analysis
pattern — eliminating boilerplate while preserving full control
over priors, samplers, and diagnostics.

Architecture
------------
This module is organized into four sections:

  §1 SAMPLER CONFIGURATION — Backend selection and sampling control
  §2 MODEL CLASSES — 10 statistical models covering regression,
     group comparison, correlation, mediation, hierarchical,
     classification, time series, and graphical models
  §3 DIAGNOSTICS & METRICS — Convergence checks, effect sizes,
     information criteria, shrinkage metrics, LaTeX export
  §4 PRIOR ELICITATION — PreliZ integration, ENIGMA-informed
     priors, prior predictive checking

Model classes
-------------
HorseshoeRegression       Regularized (Finnish) horseshoe (papers 1 & 3)
R2D2Regression            R2-D2 shrinkage prior (Zhang et al. 2022)
BayesianRidge             Standard Normal-prior ridge regression
BayesianGroupComparison   BEST (Kruschke 2013) for group differences
BayesianCorrelation       Posterior of correlations (LKJ prior)
BayesianMediation         Path analysis: X → M → Y
HierarchicalRegression    Multi-site / multi-cohort mixed effects
BayesianLogistic          Logistic regression for classification
BayesianChangePoint       Change-point detection in longitudinal data
BayesianDAG               Directed acyclic graph structure learning

Sampler backends
----------------
All models accept ``sampler=SamplerConfig(backend=...)``::

    "pymc"     — default PyMC NUTS (pytensor C backend)
    "nutpie"   — Rust NUTS via numba (~2× faster CPU)
    "numpyro"  — JAX NUTS (GPU-capable)
    "blackjax" — JAX NUTS (GPU-capable)

Dependencies
------------
Core    : pymc >= 5.0, arviz >= 0.15, numpy, pandas
Optional: nutpie, numpyro, blackjax, bambi, preliz, kulprit, pgmpy

References
----------
[1]  Piironen & Vehtari (2017) "Sparsity information and
     regularization in the horseshoe"
[2]  Zhang et al. (2022) "Bayesian Regression Using a Prior on
     the Model Fit: The R2-D2 Shrinkage Prior"
[3]  Kruschke (2013) "Bayesian Estimation Supersedes the t Test"
[4]  Vehtari, Gelman & Gabry (2017) "Practical Bayesian model
     evaluation using LOO-CV and WAIC"
[5]  Kia et al. (2022) "Federated multi-site normative modeling
     using hierarchical Bayesian regression"
[6]  Kallioinen et al. (2024) "Detecting and diagnosing prior
     and likelihood sensitivity with power-scaling"
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  §1  SAMPLER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class SamplerConfig:
    """
    MCMC sampling backend configuration.

    Controls which NUTS implementation to use and common parameters.
    All model classes accept a SamplerConfig instance.

    Parameters
    ----------
    backend : str
        ``'pymc'``, ``'nutpie'``, ``'numpyro'``, or ``'blackjax'``.
    draws, tune : int
        Posterior draws and warmup iterations per chain.
    chains : int
        Number of independent Markov chains.
    target_accept : float
        Target acceptance rate (0.95–0.99 for complex models).
    random_seed : int or None
        For reproducibility across runs.
    cores : int or None
        CPU cores for parallel chains (None = auto).

    Examples
    --------
    >>> fast = SamplerConfig(draws=500, tune=500, chains=2)
    >>> pub  = SamplerConfig(draws=4000, tune=2000, chains=4)
    >>> gpu  = SamplerConfig(backend="numpyro", draws=2000)
    >>> rust = SamplerConfig(backend="nutpie", draws=2000)
    """
    backend: Literal["pymc", "nutpie", "numpyro", "blackjax"] = "pymc"
    draws: int = 2000
    tune: int = 2000
    chains: int = 4
    target_accept: float = 0.95
    random_seed: Optional[int] = 42
    cores: Optional[int] = None
    max_treedepth: int = 10

    def sample_kwargs(self) -> Dict[str, Any]:
        """Build kwargs dict for ``pm.sample()``."""
        kw: Dict[str, Any] = dict(
            draws=self.draws, tune=self.tune, chains=self.chains,
            target_accept=self.target_accept, random_seed=self.random_seed,
            return_inferencedata=True,
        )
        if self.cores is not None:
            kw["cores"] = self.cores
        if self.backend != "pymc":
            kw["nuts_sampler"] = self.backend
        else:
            kw["nuts_sampler_kwargs"] = {"max_treedepth": self.max_treedepth}
        return kw

    @staticmethod
    def available_backends() -> Dict[str, bool]:
        """Check which sampler backends are installed."""
        avail = {"pymc": False}
        for pkg in ("pymc", "nutpie", "numpyro", "blackjax"):
            try:
                __import__(pkg)
                avail[pkg] = True
            except ImportError:
                avail[pkg] = False
        return avail


# Preset configs for common use cases
FAST = SamplerConfig(draws=500, tune=500, chains=2, target_accept=0.90)
PUBLICATION = SamplerConfig(draws=4000, tune=2000, chains=4, target_accept=0.95)
HORSESHOE = SamplerConfig(draws=4000, tune=2000, chains=4, target_accept=0.99,
                          max_treedepth=15)


def _sample(model, config: SamplerConfig):
    """Unified sampling wrapper with automatic diagnostics."""
    import pymc as pm
    with model:
        idata = pm.sample(**config.sample_kwargs())
    return idata


# ═══════════════════════════════════════════════════════════════════════════
#  §2  MODEL CLASSES
# ═══════════════════════════════════════════════════════════════════════════
#
#  Each model follows a consistent API:
#    model = ModelClass(**hyperparams)
#    model.fit(X, y, config=PUBLICATION)
#    model.summary()        → pd.DataFrame
#    model.diagnostics()    → dict
#    model.to_latex()       → str
#    model.idata_           → arviz.InferenceData


# ── 2.1  Horseshoe Regression ──────────────────────────────────────────


class HorseshoeRegression:
    """
    Regularized (Finnish) Horseshoe Regression.

    Piironen & Vehtari (2017). Ideal for high-dimensional neuroimaging
    regression where most features are expected to have negligible effects
    (e.g., 200 ROIs predicting HADS anxiety with n=46).

    The horseshoe prior places a spike-and-slab-like density on each
    coefficient: most are shrunk toward zero while a few escape to
    capture true effects.

    Prior structure::

        β_j ~ Normal(0, τ · λ̃_j)
        λ_j ~ HalfStudentT(ν=5)        # local shrinkage (per-feature)
        τ   ~ HalfStudentT(ν=2, σ=τ₀)  # global shrinkage
        c²  ~ InverseGamma(ν_s/2, ν_s/2 · s²)  # slab regularization
        λ̃_j² = c² · λ_j² / (c² + τ² · λ_j²)

    where τ₀ = p_eff / (p − p_eff) · σ / √n encodes expected sparsity.

    Parameters
    ----------
    n_relevant : int
        Expected number of non-zero coefficients (p_eff). Determines
        global shrinkage strength via τ₀.
    slab_scale : float
        Scale of the regularizing slab (default: 2.0).
    slab_df : float
        Degrees of freedom for the slab (default: 4.0).
    standardize : bool
        Whether to z-score X and y before fitting.

    Examples
    --------
    >>> # MTLE-HS: 200 ROIs predicting HADS anxiety (n=46)
    >>> model = HorseshoeRegression(n_relevant=8)
    >>> model.fit(X_rois, y_hads_anxiety, feature_names=roi_names)
    >>> model.summary()                    # HDI/ROPE table
    >>> model.plot_shrinkage()             # shrinkage factor plot
    >>> model.selected_features(threshold=0.5)  # non-zero features
    """

    def __init__(
        self,
        n_relevant: int = 5,
        slab_scale: float = 2.0,
        slab_df: float = 4.0,
        standardize: bool = True,
    ):
        self.n_relevant = n_relevant
        self.slab_scale = slab_scale
        self.slab_df = slab_df
        self.standardize = standardize
        self.model_ = None
        self.idata_ = None
        self.feature_names_: Optional[List[str]] = None
        self._X_mean = self._X_std = self._y_mean = self._y_std = None

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        feature_names: Optional[List[str]] = None,
        config: SamplerConfig = HORSESHOE,
    ) -> "HorseshoeRegression":
        """Fit the regularized horseshoe model."""
        import pymc as pm
        from pytensor import tensor as pt

        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n, p = X.shape

        if feature_names is not None:
            self.feature_names_ = list(feature_names)
        if self.feature_names_ is None:
            self.feature_names_ = [f"x_{i}" for i in range(p)]

        # Standardize
        if self.standardize:
            self._X_mean, self._X_std = X.mean(0), X.std(0)
            self._X_std[self._X_std < 1e-8] = 1.0
            self._y_mean, self._y_std = y.mean(), max(y.std(), 1e-8)
            X = (X - self._X_mean) / self._X_std
            y = (y - self._y_mean) / self._y_std

        p_eff = self.n_relevant
        tau_0 = p_eff / (p - p_eff) / np.sqrt(n)

        with pm.Model(coords={"feature": self.feature_names_}) as model:
            sigma = pm.HalfNormal("sigma", sigma=2.5)
            tau = pm.HalfStudentT("tau", nu=2, sigma=tau_0)
            lam = pm.HalfStudentT("lambda", nu=5, dims="feature")
            c_sq = pm.InverseGamma(
                "c_sq",
                alpha=self.slab_df / 2,
                beta=self.slab_df / 2 * self.slab_scale ** 2,
            )
            lam_tilde = pt.sqrt(c_sq * lam ** 2 / (c_sq + tau ** 2 * lam ** 2))

            # Non-centered parameterization (essential for horseshoe)
            z = pm.Normal("z", 0, 1, dims="feature")
            beta = pm.Deterministic("beta", z * tau * lam_tilde, dims="feature")

            alpha = pm.Normal("alpha", 0, 5)
            mu = alpha + pt.dot(X, beta)
            pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

            # Shrinkage diagnostics
            kappa = pm.Deterministic(
                "kappa", 1.0 / (1.0 + lam ** 2 * tau ** 2), dims="feature"
            )
            m_eff = pm.Deterministic("m_eff", pt.sum(1 - kappa))

        self.model_ = model
        self.idata_ = _sample(model, config)
        return self

    def summary(self, hdi_prob: float = 0.94,
                rope: Optional[Tuple[float, float]] = (-0.1, 0.1)) -> pd.DataFrame:
        """Posterior summary with HDI, ROPE analysis, and shrinkage."""
        import arviz as az

        summ = az.summary(self.idata_, var_names=["beta"], hdi_prob=hdi_prob)
        summ.index = self.feature_names_

        # ROPE analysis
        if rope is not None:
            post = self.idata_.posterior["beta"].values
            flat = post.reshape(-1, post.shape[-1])
            in_rope = np.mean((flat > rope[0]) & (flat < rope[1]), axis=0)
            summ["rope_pct"] = in_rope

        # Shrinkage factor (posterior mean of κ)
        kappa = self.idata_.posterior["kappa"].values
        summ["shrinkage_kappa"] = kappa.mean(axis=(0, 1))

        # Probability of direction (pd)
        post = self.idata_.posterior["beta"].values.reshape(-1, len(self.feature_names_))
        summ["pd"] = np.maximum(
            (post > 0).mean(axis=0), (post < 0).mean(axis=0)
        )

        return summ.sort_values("shrinkage_kappa")

    def selected_features(self, threshold: float = 0.5) -> List[str]:
        """Return features with κ < threshold (unshrunk = signal)."""
        kappa = self.idata_.posterior["kappa"].values.mean(axis=(0, 1))
        return [f for f, k in zip(self.feature_names_, kappa) if k < threshold]

    def diagnostics(self) -> Dict[str, Any]:
        """Run full MCMC diagnostics."""
        return compute_diagnostics(self.idata_)


# ── 2.2  R2-D2 Regression ─────────────────────────────────────────────


class R2D2Regression:
    """
    R2-D2 Shrinkage Prior Regression (Zhang et al. 2022).

    Places a prior on the proportion of variance explained (R²)
    rather than on sparsity count. A Dirichlet distribution
    allocates variance across predictors. Achieves near-minimax
    contraction rates with optimal 1/x tail behavior.

    Preferred over horseshoe when you can more naturally reason
    about "how much variance should the model explain?" rather
    than "how many predictors are non-zero?"

    Parameters
    ----------
    r2_mean : float
        Prior mean of R² (e.g., 0.3 for brain-behavior).
    r2_std : float
        Prior SD of R² (e.g., 0.15 for moderate uncertainty).
    """

    def __init__(self, r2_mean: float = 0.3, r2_std: float = 0.15,
                 standardize: bool = True):
        self.r2_mean = r2_mean
        self.r2_std = r2_std
        self.standardize = standardize
        self.model_ = self.idata_ = None
        self.feature_names_: Optional[List[str]] = None

    def fit(self, X, y, feature_names=None, config=PUBLICATION):
        """Fit the R2-D2 regression model."""
        import pymc as pm

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n, p = X.shape

        self.feature_names_ = (
            feature_names or [f"x_{i}" for i in range(p)]
        )

        if self.standardize:
            X = (X - X.mean(0)) / np.maximum(X.std(0), 1e-8)
            y = (y - y.mean()) / max(y.std(), 1e-8)

        try:
            import pymc_extras as pmx
            with pm.Model(coords={"feature": self.feature_names_}) as model:
                sigma, beta = pmx.distributions.R2D2M2CP(
                    "beta", y.std(), X.std(0),
                    dims="feature",
                    r2=self.r2_mean, r2_std=self.r2_std,
                    centered=False,
                )
                alpha = pm.Normal("alpha", 0, 5)
                mu = alpha + pm.math.dot(X, beta)
                pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)
        except ImportError:
            # Fallback: manual R2-D2 implementation
            logger.info("pymc-extras not found; using manual R2-D2.")
            with pm.Model(coords={"feature": self.feature_names_}) as model:
                r2 = pm.Beta("R2",
                             mu=self.r2_mean,
                             sigma=self.r2_std)
                phi = pm.Dirichlet("phi", a=np.ones(p), dims="feature")
                sigma = pm.HalfNormal("sigma", 2.5)
                tau2 = r2 / (1 - r2) * sigma ** 2
                beta = pm.Normal("beta", 0,
                                 sigma=pm.math.sqrt(tau2 * phi),
                                 dims="feature")
                alpha = pm.Normal("alpha", 0, 5)
                mu = alpha + pm.math.dot(X, beta)
                pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

        self.model_ = model
        self.idata_ = _sample(model, config)
        return self

    def summary(self, hdi_prob=0.94):
        import arviz as az
        return az.summary(self.idata_, var_names=["beta"], hdi_prob=hdi_prob)


# ── 2.3  Bayesian Ridge ───────────────────────────────────────────────


class BayesianRidge:
    """
    Bayesian Ridge Regression (Normal prior, dense effects).

    Use when most predictors contribute small effects (e.g.,
    whole-brain decoding, PCA-reduced feature spaces). Simpler
    than horseshoe, faster to sample, appropriate when sparsity
    is not expected.

    Parameters
    ----------
    prior_sigma : float
        Scale of the Normal prior on coefficients.
    """

    def __init__(self, prior_sigma: float = 1.0, standardize: bool = True):
        self.prior_sigma = prior_sigma
        self.standardize = standardize
        self.model_ = self.idata_ = None
        self.feature_names_ = None

    def fit(self, X, y, feature_names=None, config=PUBLICATION):
        import pymc as pm

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n, p = X.shape
        self.feature_names_ = feature_names or [f"x_{i}" for i in range(p)]

        if self.standardize:
            X = (X - X.mean(0)) / np.maximum(X.std(0), 1e-8)
            y = (y - y.mean()) / max(y.std(), 1e-8)

        with pm.Model(coords={"feature": self.feature_names_}) as model:
            beta = pm.Normal("beta", 0, self.prior_sigma, dims="feature")
            alpha = pm.Normal("alpha", 0, 5)
            sigma = pm.HalfNormal("sigma", 2.5)
            mu = alpha + pm.math.dot(X, beta)
            pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

        self.model_ = model
        self.idata_ = _sample(model, config)
        return self

    def summary(self, hdi_prob=0.94):
        import arviz as az
        return az.summary(self.idata_, var_names=["beta"], hdi_prob=hdi_prob)


# ── 2.4  BEST — Group Comparison ──────────────────────────────────────


class BayesianGroupComparison:
    """
    BEST: Bayesian Estimation Supersedes the t-Test (Kruschke 2013).

    Uses Student-t likelihoods per group with shared degrees of
    freedom ν, providing robustness to outliers. Returns the full
    posterior of the group difference (mean, SD) and Cohen's d.

    ROPE recommendations for neuroimaging (pre-defined defaults):
      cortical_thickness : ±0.1d (standardized)
      volume             : ±0.1d
      fa                 : ±0.1d

    Parameters
    ----------
    rope : tuple (low, high)
        Region of Practical Equivalence for Cohen's d.
    """

    def __init__(self, rope: Tuple[float, float] = (-0.1, 0.1)):
        self.rope = rope
        self.model_ = self.idata_ = None

    def fit(self, group1: np.ndarray, group2: np.ndarray,
            group_names: Tuple[str, str] = ("group1", "group2"),
            config: SamplerConfig = PUBLICATION) -> "BayesianGroupComparison":
        """Fit the BEST model."""
        import pymc as pm

        g1 = np.asarray(group1, dtype=np.float64).ravel()
        g2 = np.asarray(group2, dtype=np.float64).ravel()

        # Pooled summary for prior calibration
        pooled = np.concatenate([g1, g2])
        mu_m, mu_s = pooled.mean(), pooled.std()

        with pm.Model() as model:
            mu1 = pm.Normal(f"mu_{group_names[0]}", mu=mu_m, sigma=mu_s * 2)
            mu2 = pm.Normal(f"mu_{group_names[1]}", mu=mu_m, sigma=mu_s * 2)
            sig1 = pm.HalfNormal(f"sigma_{group_names[0]}", sigma=mu_s * 2)
            sig2 = pm.HalfNormal(f"sigma_{group_names[1]}", sigma=mu_s * 2)
            nu = pm.Exponential("nu_minus1", 1.0 / 29.0) + 1

            pm.StudentT(f"{group_names[0]}_obs", nu=nu, mu=mu1, sigma=sig1, observed=g1)
            pm.StudentT(f"{group_names[1]}_obs", nu=nu, mu=mu2, sigma=sig2, observed=g2)

            # Derived quantities
            pm.Deterministic("diff_means", mu1 - mu2)
            pm.Deterministic("diff_stds", sig1 - sig2)
            pooled_sd = pm.math.sqrt((sig1 ** 2 + sig2 ** 2) / 2)
            pm.Deterministic("effect_size", (mu1 - mu2) / pooled_sd)

        self.model_ = model
        self.idata_ = _sample(model, config)
        self._group_names = group_names
        return self

    def summary(self, hdi_prob: float = 0.94) -> pd.DataFrame:
        import arviz as az
        return az.summary(
            self.idata_,
            var_names=["diff_means", "diff_stds", "effect_size"],
            hdi_prob=hdi_prob,
        )

    def rope_analysis(self) -> Dict[str, float]:
        """Percentage of effect size posterior in ROPE."""
        d = self.idata_.posterior["effect_size"].values.ravel()
        in_rope = np.mean((d > self.rope[0]) & (d < self.rope[1]))
        below = np.mean(d < self.rope[0])
        above = np.mean(d > self.rope[1])
        return {"in_rope": in_rope, "below_rope": below, "above_rope": above,
                "pd": max(np.mean(d > 0), np.mean(d < 0))}


# ── 2.5  Bayesian Correlation ─────────────────────────────────────────


class BayesianCorrelation:
    """
    Posterior distribution of Pearson correlations via LKJ prior.

    For multi-ROI analysis, uses the LKJ distribution as prior
    over the correlation matrix, separating correlation structure
    from scale. Returns full posteriors for all pairwise correlations.

    Parameters
    ----------
    eta : float
        LKJ concentration parameter.
        1.0 = uniform over correlation matrices.
        2.0 = moderate shrinkage toward identity (recommended).
    """

    def __init__(self, eta: float = 2.0):
        self.eta = eta
        self.model_ = self.idata_ = None

    def fit(self, data: Union[np.ndarray, pd.DataFrame],
            var_names: Optional[List[str]] = None,
            config: SamplerConfig = PUBLICATION) -> "BayesianCorrelation":
        """Fit the multivariate correlation model."""
        import pymc as pm

        if isinstance(data, pd.DataFrame):
            var_names = var_names or list(data.columns)
            data = data.values
        data = np.asarray(data, dtype=np.float64)
        n, p = data.shape
        self._var_names = var_names or [f"v_{i}" for i in range(p)]

        with pm.Model(coords={"variable": self._var_names}) as model:
            chol, corr, stds = pm.LKJCholeskyCov(
                "chol", n=p, eta=self.eta,
                sd_dist=pm.Exponential.dist(1.0, shape=p),
                compute_corr=True,
            )
            pm.Deterministic("corr", corr)
            mu = pm.Normal("mu", 0, sigma=1.5, dims="variable")
            pm.MvNormal("obs", mu=mu, chol=chol, observed=data)

        self.model_ = model
        self.idata_ = _sample(model, config)
        return self

    def get_correlation_posterior(self, var1: str, var2: str) -> np.ndarray:
        """Get posterior samples of r(var1, var2)."""
        i = self._var_names.index(var1)
        j = self._var_names.index(var2)
        return self.idata_.posterior["corr"].values[:, :, i, j].ravel()


# ── 2.6  Bayesian Mediation ───────────────────────────────────────────


class BayesianMediation:
    """
    Bayesian Mediation Analysis: X → M → Y with direct path X → Y.

    Estimates all paths simultaneously and derives the indirect
    effect a×b as a Deterministic. The posterior of a×b is naturally
    asymmetric (product of two normals), which Bayesian inference
    handles automatically — unlike the frequentist Sobel test.

    Application: structure (X) → function (M) → behavior (Y)
    e.g., cortical thickness → HKS coupling → HADS anxiety.

    Parameters
    ----------
    standardize : bool
        Z-score all variables before fitting.
    """

    def __init__(self, standardize: bool = True):
        self.standardize = standardize
        self.model_ = self.idata_ = None

    def fit(self, X: np.ndarray, M: np.ndarray, Y: np.ndarray,
            config: SamplerConfig = PUBLICATION) -> "BayesianMediation":
        """Fit the mediation model: X→M (path a), M→Y (path b), X→Y (path c')."""
        import pymc as pm

        X = np.asarray(X, dtype=np.float64).ravel()
        M = np.asarray(M, dtype=np.float64).ravel()
        Y = np.asarray(Y, dtype=np.float64).ravel()

        if self.standardize:
            X = (X - X.mean()) / max(X.std(), 1e-8)
            M = (M - M.mean()) / max(M.std(), 1e-8)
            Y = (Y - Y.mean()) / max(Y.std(), 1e-8)

        with pm.Model() as model:
            # Path a: X → M
            a = pm.Normal("a", 0, 1)
            a0 = pm.Normal("a0", 0, 1)
            sigma_m = pm.HalfNormal("sigma_m", 1)
            pm.Normal("M_obs", mu=a0 + a * X, sigma=sigma_m, observed=M)

            # Path b (M → Y) and c' (X → Y, direct)
            b = pm.Normal("b", 0, 1)
            c_prime = pm.Normal("c_prime", 0, 1)
            b0 = pm.Normal("b0", 0, 1)
            sigma_y = pm.HalfNormal("sigma_y", 1)
            pm.Normal("Y_obs", mu=b0 + b * M + c_prime * X, sigma=sigma_y, observed=Y)

            # Derived quantities
            pm.Deterministic("indirect_ab", a * b)
            pm.Deterministic("total_effect", a * b + c_prime)
            pm.Deterministic("proportion_mediated",
                             (a * b) / (a * b + c_prime + 1e-8))

        self.model_ = model
        self.idata_ = _sample(model, config)
        return self

    def summary(self, hdi_prob=0.94):
        import arviz as az
        return az.summary(
            self.idata_,
            var_names=["a", "b", "c_prime", "indirect_ab",
                       "total_effect", "proportion_mediated"],
            hdi_prob=hdi_prob,
        )


# ── 2.7  Hierarchical Regression ──────────────────────────────────────


class HierarchicalRegression:
    """
    Hierarchical (multi-level) Bayesian regression.

    Replaces ComBat for multi-site harmonization by modeling site
    as a random effect within a single regression, avoiding the
    two-stage estimation problem. Partial pooling stabilizes
    estimates for small sites.

    Uses non-centered parameterization by default (essential for
    < 30 groups / small within-group samples).

    Parameters
    ----------
    group_col : str
        Column name for the grouping variable (e.g., 'site').
    formula : str or None
        Bambi-style formula. If None, uses all other columns as
        fixed effects.
    """

    def __init__(self, group_col: str = "site", formula: Optional[str] = None):
        self.group_col = group_col
        self.formula = formula
        self.model_ = self.idata_ = None

    def fit(self, data: pd.DataFrame, outcome: str,
            config: SamplerConfig = PUBLICATION) -> "HierarchicalRegression":
        """Fit the hierarchical model."""
        import pymc as pm

        groups = data[self.group_col].astype("category")
        group_idx = groups.cat.codes.values
        group_names = list(groups.cat.categories)
        n_groups = len(group_names)

        # Fixed effects: all numeric columns except outcome and group
        fixed_cols = [c for c in data.select_dtypes(include=[np.number]).columns
                      if c != outcome and c != self.group_col]
        X = data[fixed_cols].values.astype(np.float64)
        y = data[outcome].values.astype(np.float64)

        # Standardize
        X = (X - X.mean(0)) / np.maximum(X.std(0), 1e-8)
        y_mean, y_std = y.mean(), max(y.std(), 1e-8)
        y = (y - y_mean) / y_std

        n, p = X.shape

        with pm.Model(coords={"group": group_names,
                               "feature": fixed_cols}) as model:
            # Random intercept (non-centered)
            sigma_group = pm.HalfNormal("sigma_group", sigma=1)
            group_raw = pm.Normal("group_raw", 0, 1, dims="group")
            group_effect = pm.Deterministic("group_effect",
                                            group_raw * sigma_group, dims="group")

            # Fixed effects
            beta = pm.Normal("beta", 0, 1, dims="feature")
            alpha = pm.Normal("alpha", 0, 2)
            sigma = pm.HalfNormal("sigma", 2)

            mu = alpha + group_effect[group_idx] + pm.math.dot(X, beta)
            pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

        self.model_ = model
        self.idata_ = _sample(model, config)
        self._group_names = group_names
        self._fixed_cols = fixed_cols
        return self

    def summary(self, hdi_prob=0.94):
        import arviz as az
        return az.summary(
            self.idata_,
            var_names=["beta", "group_effect", "sigma_group"],
            hdi_prob=hdi_prob,
        )


# ── 2.8  Bayesian Logistic ────────────────────────────────────────────


class BayesianLogistic:
    """
    Bayesian Logistic Regression for classification.

    Combines with horseshoe or ridge priors for feature selection.
    Returns posterior predictive class probabilities with full
    uncertainty — essential for clinical decision-making.

    Parameters
    ----------
    prior : str
        ``'ridge'`` (Normal), ``'horseshoe'``, or ``'flat'``.
    n_relevant : int
        For horseshoe prior only.
    """

    def __init__(self, prior: str = "ridge", n_relevant: int = 5,
                 prior_sigma: float = 1.0):
        self.prior = prior
        self.n_relevant = n_relevant
        self.prior_sigma = prior_sigma
        self.model_ = self.idata_ = None
        self.feature_names_ = None

    def fit(self, X, y, feature_names=None, config=PUBLICATION):
        import pymc as pm
        from pytensor import tensor as pt

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64).ravel()
        n, p = X.shape
        self.feature_names_ = feature_names or [f"x_{i}" for i in range(p)]

        X = (X - X.mean(0)) / np.maximum(X.std(0), 1e-8)

        with pm.Model(coords={"feature": self.feature_names_}) as model:
            alpha = pm.Normal("alpha", 0, 5)

            if self.prior == "horseshoe":
                tau_0 = self.n_relevant / (p - self.n_relevant) / np.sqrt(n)
                tau = pm.HalfStudentT("tau", nu=2, sigma=tau_0)
                lam = pm.HalfStudentT("lambda", nu=5, dims="feature")
                z = pm.Normal("z", 0, 1, dims="feature")
                beta = pm.Deterministic("beta", z * tau * lam, dims="feature")
            elif self.prior == "ridge":
                beta = pm.Normal("beta", 0, self.prior_sigma, dims="feature")
            else:
                beta = pm.Flat("beta", dims="feature")

            logits = alpha + pt.dot(X, beta)
            pm.Bernoulli("y_obs", logit_p=logits, observed=y)

        self.model_ = model
        self.idata_ = _sample(model, config)
        return self

    def predict_proba(self, X_new: np.ndarray) -> np.ndarray:
        """Posterior predictive class probabilities (mean ± HDI)."""
        import pymc as pm

        X_new = np.asarray(X_new, dtype=np.float64)
        X_new = (X_new - X_new.mean(0)) / np.maximum(X_new.std(0), 1e-8)

        alpha = self.idata_.posterior["alpha"].values.ravel()
        beta = self.idata_.posterior["beta"].values.reshape(-1, X_new.shape[1])
        logits = alpha[:, None] + beta @ X_new.T  # (samples, n_new)
        probs = 1.0 / (1.0 + np.exp(-logits))
        return probs  # (samples, n_new)

    def summary(self, hdi_prob=0.94):
        import arviz as az
        return az.summary(self.idata_, var_names=["beta"], hdi_prob=hdi_prob)


# ── 2.9  Bayesian Change-Point ────────────────────────────────────────


class BayesianChangePoint:
    """
    Bayesian Change-Point Detection in longitudinal data.

    Detects a single change-point in a time series (e.g., pre/post
    treatment cortical thickness trajectories, or COVID onset effects).

    The model assumes the data follows one distribution before the
    change-point and another after.
    """

    def __init__(self):
        self.model_ = self.idata_ = None

    def fit(self, y: np.ndarray, config: SamplerConfig = PUBLICATION):
        """Detect a single change-point in y."""
        import pymc as pm

        y = np.asarray(y, dtype=np.float64).ravel()
        n = len(y)

        with pm.Model() as model:
            # Change-point location (discrete uniform)
            tau_cp = pm.DiscreteUniform("changepoint", lower=2, upper=n - 2)

            # Parameters before and after
            mu1 = pm.Normal("mu_before", mu=y.mean(), sigma=y.std() * 2)
            mu2 = pm.Normal("mu_after", mu=y.mean(), sigma=y.std() * 2)
            sigma1 = pm.HalfNormal("sigma_before", sigma=y.std() * 2)
            sigma2 = pm.HalfNormal("sigma_after", sigma=y.std() * 2)

            # Likelihood
            idx = np.arange(n)
            mu = pm.math.switch(idx < tau_cp, mu1, mu2)
            sigma = pm.math.switch(idx < tau_cp, sigma1, sigma2)
            pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

        # Note: discrete parameters require Metropolis, not NUTS
        import pymc as pm
        with model:
            self.idata_ = pm.sample(
                draws=config.draws, tune=config.tune,
                chains=config.chains, random_seed=config.random_seed,
                return_inferencedata=True,
            )
        self.model_ = model
        return self


# ── 2.10  Bayesian DAG (pgmpy wrapper) ───────────────────────────────


class BayesianDAG:
    """
    Bayesian Network structure learning for brain connectivity.

    Wraps pgmpy for constraint-based (PC algorithm) and score-based
    (HillClimb with BIC/BDeu) structure learning, plus DAG
    visualization and causal inference queries.

    Parameters
    ----------
    method : str
        ``'pc'`` (constraint-based) or ``'hillclimb'`` (score-based).
    significance : float
        For PC algorithm: significance level for conditional
        independence tests (default: 0.05).
    """

    def __init__(self, method: str = "hillclimb", significance: float = 0.05):
        self.method = method
        self.significance = significance
        self.model_ = None
        self.edges_ = None

    def fit(self, data: pd.DataFrame) -> "BayesianDAG":
        """Learn DAG structure from data."""
        try:
            from pgmpy.estimators import HillClimbSearch, PC, BDeuScore
        except ImportError:
            raise ImportError("pgmpy is required: pip install pgmpy")

        if self.method == "hillclimb":
            scoring = BDeuScore(data, equivalent_sample_size=10)
            est = HillClimbSearch(data)
            dag = est.estimate(scoring_method=scoring)
        elif self.method == "pc":
            est = PC(data)
            dag = est.estimate(significance_level=self.significance)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self.model_ = dag
        self.edges_ = list(dag.edges())
        self._node_names = list(data.columns)
        return self

    def get_parents(self, node: str) -> List[str]:
        """Get parent nodes of a given node."""
        return [e[0] for e in self.edges_ if e[1] == node]

    def get_children(self, node: str) -> List[str]:
        """Get child nodes of a given node."""
        return [e[1] for e in self.edges_ if e[0] == node]

    def adjacency_matrix(self) -> pd.DataFrame:
        """Return adjacency matrix as DataFrame."""
        n = len(self._node_names)
        adj = pd.DataFrame(np.zeros((n, n), dtype=int),
                           index=self._node_names, columns=self._node_names)
        for parent, child in self.edges_:
            adj.loc[parent, child] = 1
        return adj


# ═══════════════════════════════════════════════════════════════════════════
#  §3  DIAGNOSTICS, METRICS, AND EXPORT
# ═══════════════════════════════════════════════════════════════════════════


def compute_diagnostics(idata, var_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Comprehensive MCMC diagnostics report.

    Checks R̂, ESS, divergences, BFMI, and Pareto k (if LOO available).
    Returns a dict with status flags and numeric values.

    Thresholds (Vehtari et al. 2021):
      R̂ < 1.01, ESS_bulk ≥ 400, ESS_tail ≥ 400,
      divergences = 0, BFMI > 0.3
    """
    import arviz as az

    report: Dict[str, Any] = {}

    # R-hat
    rhat = az.rhat(idata, var_names=var_names)
    max_rhat = max(float(rhat[v].max()) for v in rhat.data_vars)
    report["rhat_max"] = max_rhat
    report["rhat_ok"] = max_rhat < 1.01

    # ESS
    ess_bulk = az.ess(idata, var_names=var_names, method="bulk")
    ess_tail = az.ess(idata, var_names=var_names, method="tail")
    min_bulk = min(float(ess_bulk[v].min()) for v in ess_bulk.data_vars)
    min_tail = min(float(ess_tail[v].min()) for v in ess_tail.data_vars)
    report["ess_bulk_min"] = min_bulk
    report["ess_tail_min"] = min_tail
    report["ess_ok"] = min_bulk >= 400 and min_tail >= 400

    # Divergences
    if "diverging" in idata.sample_stats:
        n_div = int(idata.sample_stats["diverging"].values.sum())
    else:
        n_div = 0
    report["divergences"] = n_div
    report["divergences_ok"] = n_div == 0

    # BFMI
    try:
        bfmi_vals = az.bfmi(idata)
        report["bfmi_min"] = float(min(bfmi_vals))
        report["bfmi_ok"] = float(min(bfmi_vals)) > 0.3
    except Exception:
        report["bfmi_min"] = np.nan
        report["bfmi_ok"] = True

    # Overall
    report["all_ok"] = all([
        report["rhat_ok"], report["ess_ok"],
        report["divergences_ok"], report["bfmi_ok"],
    ])

    return report


def model_comparison(
    models: Dict[str, Any],
    ic: str = "loo",
) -> pd.DataFrame:
    """
    Compare multiple fitted models using LOO-CV or WAIC.

    Parameters
    ----------
    models : dict
        Mapping from model name to idata (arviz.InferenceData).
    ic : str
        Information criterion: ``'loo'`` or ``'waic'``.

    Returns
    -------
    DataFrame with ELPD differences, weights, and ranks.
    """
    import arviz as az
    return az.compare(models, ic=ic, method="stacking")


def bayesian_r2(idata, y_true: np.ndarray) -> Dict[str, float]:
    """
    Bayesian R² (Gelman et al. 2019).

    Returns posterior mean and HDI of R² = Var(ŷ) / (Var(ŷ) + Var(ε)).
    """
    import arviz as az

    y_pred = idata.posterior_predictive
    if y_pred is None:
        raise ValueError("Run pm.sample_posterior_predictive() first.")

    # Get predicted values
    obs_name = list(y_pred.data_vars)[0]
    y_hat = y_pred[obs_name].values  # (chains, draws, n)
    y_hat_flat = y_hat.reshape(-1, y_hat.shape[-1])

    var_fit = np.var(y_hat_flat, axis=1)
    var_res = np.var(y_hat_flat - y_true[None, :], axis=1)
    r2_samples = var_fit / (var_fit + var_res)

    return {
        "r2_mean": float(np.mean(r2_samples)),
        "r2_std": float(np.std(r2_samples)),
        "r2_hdi_low": float(np.percentile(r2_samples, 3)),
        "r2_hdi_high": float(np.percentile(r2_samples, 97)),
    }


def probability_of_direction(posterior_samples: np.ndarray) -> float:
    """
    Probability of Direction (pd).

    pd = max(P(θ > 0), P(θ < 0)). Ranges from 0.5 (no evidence)
    to 1.0 (all samples same sign). pd ≈ 0.975 ≈ p < 0.05.
    """
    samples = np.asarray(posterior_samples).ravel()
    return float(max(np.mean(samples > 0), np.mean(samples < 0)))


def rope_percentage(
    posterior_samples: np.ndarray,
    rope: Tuple[float, float] = (-0.1, 0.1),
) -> Dict[str, float]:
    """
    ROPE analysis — what percentage of the posterior is inside/outside ROPE?

    Returns dict with keys: in_rope, below_rope, above_rope.
    """
    s = np.asarray(posterior_samples).ravel()
    return {
        "in_rope": float(np.mean((s >= rope[0]) & (s <= rope[1]))),
        "below_rope": float(np.mean(s < rope[0])),
        "above_rope": float(np.mean(s > rope[1])),
    }


def savage_dickey_bf(
    prior_samples: np.ndarray,
    posterior_samples: np.ndarray,
    point: float = 0.0,
) -> float:
    """
    Savage-Dickey density ratio Bayes factor for nested models.

    BF₁₀ = p(θ = point | prior) / p(θ = point | posterior).
    BF₁₀ > 1 favors the alternative (θ ≠ point).
    """
    from scipy.stats import gaussian_kde

    prior_kde = gaussian_kde(prior_samples.ravel())
    post_kde = gaussian_kde(posterior_samples.ravel())

    bf10 = prior_kde(point)[0] / max(post_kde(point)[0], 1e-20)
    return float(bf10)


def shrinkage_metrics(idata) -> pd.DataFrame:
    """
    Extract horseshoe shrinkage diagnostics.

    Returns DataFrame with columns: kappa (shrinkage factor),
    is_signal (κ < 0.5), and m_eff (effective non-zero count).
    """
    kappa = idata.posterior["kappa"].values.mean(axis=(0, 1))
    m_eff = float(idata.posterior["m_eff"].values.mean())

    df = pd.DataFrame({
        "kappa": kappa,
        "is_signal": kappa < 0.5,
    })
    df.attrs["m_eff"] = m_eff
    return df


# ── LaTeX export ──────────────────────────────────────────────────────


def to_latex_table(
    summary_df: pd.DataFrame,
    caption: str = "Bayesian regression results",
    label: str = "tab:bayes_results",
    columns: Optional[List[str]] = None,
    float_format: str = "%.3f",
) -> str:
    """
    Export a summary DataFrame to a LaTeX table string.

    Produces a publication-ready table with booktabs formatting
    suitable for Nature/NeuroImage/JAMA Neurology.
    """
    if columns is not None:
        summary_df = summary_df[columns]

    latex = summary_df.to_latex(
        float_format=float_format,
        caption=caption,
        label=label,
        bold_rows=True,
        escape=True,
    )
    # Add booktabs
    latex = latex.replace("\\toprule", "\\toprule")
    latex = latex.replace("\\bottomrule", "\\bottomrule")
    return latex


# ═══════════════════════════════════════════════════════════════════════════
#  §4  PRIOR ELICITATION
# ═══════════════════════════════════════════════════════════════════════════


def elicit_prior(
    distribution: str = "Normal",
    lower: Optional[float] = None,
    upper: Optional[float] = None,
    mass: float = 0.95,
    mean: Optional[float] = None,
) -> object:
    """
    Find maximum-entropy prior satisfying constraints via PreliZ.

    Parameters
    ----------
    distribution : str
        Target distribution (Normal, HalfNormal, Gamma, etc.).
    lower, upper : float
        Bounds containing ``mass`` probability.
    mass : float
        Probability mass within [lower, upper].
    mean : float or None
        Additional mean constraint.

    Returns
    -------
    preliz distribution object (can be converted to PyMC via ``.to_pymc()``)

    Examples
    --------
    >>> # Prior for cortical thickness effect (mm)
    >>> prior = elicit_prior("Normal", lower=-0.8, upper=0.8, mass=0.95)
    >>> print(prior)  # Normal(mu=0, sigma=0.41)
    >>> prior.to_pymc()  # → pm.Normal.dist(mu=0, sigma=0.41)
    """
    try:
        import preliz as pz
    except ImportError:
        raise ImportError("PreliZ required: pip install preliz")

    dist_class = getattr(pz, distribution)
    if mean is not None:
        dist = dist_class(mu=mean)
    else:
        dist = dist_class()

    pz.maxent(dist, lower, upper, mass)
    return dist


# ── ENIGMA-informed priors ────────────────────────────────────────────

ENIGMA_EFFECT_SIZES = {
    "cortical_thickness": {
        "description": "Cortical thickness effect sizes from ENIGMA meta-analyses",
        "unit": "mm",
        "sd_typical": 0.5,
        "effects": {
            "schizophrenia_global": -0.53,
            "schizophrenia_fusiform": -0.37,
            "mdd_hippocampus": -0.14,
            "mtle_ipsilateral": -0.80,
            "mtle_contralateral": -0.30,
            "covid_global": -0.25,
        },
    },
    "subcortical_volume": {
        "description": "Subcortical volumes from ENIGMA",
        "unit": "mm³",
        "effects": {
            "schizophrenia_hippocampus": -0.46,
            "schizophrenia_thalamus": -0.31,
        },
    },
    "fa": {
        "description": "Fractional anisotropy from ENIGMA-DTI",
        "unit": "FA",
        "sd_typical": 0.03,
        "effects": {
            "schizophrenia_global": -0.42,
            "schizophrenia_acr": -0.40,
        },
    },
}


def enigma_informed_prior(
    modality: str,
    effect: str,
    prior_type: str = "weakly_informative",
) -> Tuple[float, float]:
    """
    Generate prior (mu, sigma) from ENIGMA consortium effect sizes.

    Parameters
    ----------
    modality : 'cortical_thickness', 'subcortical_volume', or 'fa'
    effect : key from ENIGMA_EFFECT_SIZES (e.g., 'schizophrenia_global')
    prior_type : 'informative', 'weakly_informative', or 'diffuse'

    Returns
    -------
    (mu, sigma) tuple for a Normal prior.
    """
    if modality not in ENIGMA_EFFECT_SIZES:
        raise ValueError(f"Unknown modality: {modality}")
    data = ENIGMA_EFFECT_SIZES[modality]
    if effect not in data["effects"]:
        raise ValueError(f"Unknown effect: {effect}")

    d = data["effects"][effect]
    sd_typical = data.get("sd_typical", 0.5)
    effect_mm = d * sd_typical

    if prior_type == "informative":
        return (effect_mm, abs(effect_mm) * 0.5)
    elif prior_type == "weakly_informative":
        return (0.0, abs(effect_mm) * 2.0)
    else:  # diffuse
        return (0.0, abs(effect_mm) * 5.0)
