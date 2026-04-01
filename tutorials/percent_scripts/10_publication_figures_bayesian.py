# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Tutorial 10 — Figuras para Publicação e Modelos Bayesianos
#
# Neste tutorial final, cobriremos dois módulos avançados:
#
# **Parte A — `brainplots`**: Visualizações publication-quality
# com PyVista (4-view, comparison, surprise, connectivity, etc.)
#
# **Parte B — `bayesian` + `bayes_viz`**: 10 modelos Bayesianos
# (Horseshoe, R2D2, BEST, mediation, DAG) com diagnósticos completos
#
# ---
#
# # Parte A — Figuras para Publicação (`brainplots`)
#
# O módulo `brainplots` produz figuras de qualidade Nature/Science:
# - Renderização 3D com PyVista (offscreen)
# - 4 views automáticas (lateral, medial, dorsal, ventral)
# - Colorbars calibradas
# - Exportação em PNG/SVG/PDF a 300+ dpi

# %%
import numpy as np
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

import corticalfields as cf

SUBJECTS_DIR = "./data/freesurfer"
SUBJECT_ID = "sub-010002"

surf = cf.load_freesurfer_surface(SUBJECTS_DIR, SUBJECT_ID, "lh", "pial",
                                   overlays=["thickness", "curv", "sulc"])

# %% [markdown]
# ## A.1 — Catálogo de Funções de Visualização
#
# | Função | O que plota | Uso típico |
# |--------|------------|------------|
# | `plot_surface_4view()` | Superfície com overlay em 4 ângulos | Espessura, curvatura |
# | `plot_surface_comparison()` | Dois sujeitos lado a lado | Controle vs paciente |
# | `plot_surprise_brain()` | z-scores/surprise com threshold | Anomaly maps |
# | `plot_normative_result()` | mean/var/z/surprise composite | Resultado normativo completo |
# | `plot_hks_multiscale()` | HKS em múltiplas escalas | Análise espectral |
# | `plot_asymmetry_brain()` | Mapa de assimetria | LH vs RH |
# | `plot_connectivity_matrix()` | MSN/SSN heatmap anotado | Redes cerebrais |
# | `plot_functional_map_matrix()` | C matrix com bandas | Functional maps |
# | `plot_distance_matrix()` | D matrix com dendrograma | Distâncias OT |
# | `plot_permutation_null()` | Null distribution + observed | Inferência por permutação |
# | `plot_eigenspectrum()` | λ vs k com lei de Weyl | Espectro LB |
# | `plot_network_radar()` | Radar chart de métricas | Graph metrics por rede |
# | `plot_network_anomaly_bars()` | Barplot de anomalia por rede | Surprise por rede |
# | `plot_brain_connectome()` | Conectoma 3D (nodes + edges) | Network visualization |
# | `plot_network_graph()` | Grafo spring-layout 2D | Topologia de rede |
# | `plot_asymmetry_bands()` | Off-diag energy por banda | FM asymmetry decomp |
# | `plot_krr_diagnostic()` | Observed vs predicted + null | KRR results |
# | `plot_subcortical_3d()` | Subcortical 3D rendering | Hippocampus, thalamus |
# | `plot_composite_figure()` | Multi-panel composição | Figure completa |

# %%
# ── plot_surface_4view: 4 ângulos de uma superfície ──────────────────
# Requer PyVista com rendering offscreen (xvfb em servidor)
#
# from corticalfields.brainplots import plot_surface_4view, save_figure
#
# fig = plot_surface_4view(
#     vertices=surf.vertices,
#     faces=surf.faces,
#     scalar=surf.get_overlay("thickness"),
#     cmap="YlOrRd",
#     clim=(1.0, 4.0),
#     title="Cortical Thickness (mm)",
#     hemi="lh",
#     journal="nature",     # estilo: 'nature', 'science', 'cell'
# )
# save_figure(fig, "10_4view_thickness.png", dpi=300)

print("As funções de brainplots requerem PyVista + xvfb.")
print("Veja a documentação para setup em servidor headless.")

# %% [markdown]
# ## A.2 — Figuras compostas

# %%
# from corticalfields.brainplots import plot_composite_figure
#
# fig = plot_composite_figure(
#     panels={
#         "A": ("surface_4view", {"vertices": surf.vertices, "faces": surf.faces,
#                                  "scalar": thickness, "cmap": "YlOrRd"}),
#         "B": ("eigenspectrum", {"eigenvalues": lb.eigenvalues, "area": surf.total_area}),
#         "C": ("connectivity_matrix", {"matrix": msn, "labels": roi_names}),
#         "D": ("permutation_null", {"null_dist": null_dist, "observed": obs_stat}),
#     },
#     layout="2x2",
#     figsize=(16, 14),
#     journal="nature",
# )
print("plot_composite_figure() monta figuras multi-panel automaticamente.")

# %% [markdown]
# # Parte B — Modelos Bayesianos (`bayesian` + `bayes_viz`)
#
# O módulo `bayesian` fornece 10 modelos Bayesianos prontos para uso
# clínico, todos com interface consistente (.fit(), .summary(),
# diagnostics()).
#
# ## B.1 — Catálogo de Modelos
#
# | Modelo | Prior | Uso | Backend |
# |--------|-------|-----|---------|
# | `HorseshoeRegression` | Finnish horseshoe | Feature selection esparsa | PyMC |
# | `R2D2Regression` | R²-D2 shrinkage | Regularização via R² | PyMC |
# | `BayesianRidge` | Normal | Ridge regression padrão | PyMC |
# | `BayesianGroupComparison` | BEST (Kruschke) | Controle vs paciente | PyMC |
# | `BayesianCorrelation` | LKJ prior | Correlação posterior | PyMC |
# | `BayesianMediation` | Normal | X → M → Y | PyMC |
# | `HierarchicalRegression` | Hierárquico | Multi-site | PyMC/Bambi |
# | `BayesianLogistic` | Horseshoe/Ridge | Classificação | PyMC |
# | `BayesianChangePoint` | — | Change-point em longitudinal | PyMC |
# | `BayesianDAG` | — | Descoberta de estrutura causal | pgmpy |

# %%
# ── Exemplo: BayesianGroupComparison (BEST) ─────────────────────────
from corticalfields.bayesian import (
    BayesianGroupComparison,
    BayesianCorrelation,
    HorseshoeRegression,
    SamplerConfig,
    PUBLICATION,
    compute_diagnostics,
    model_comparison,
    bayesian_r2,
    probability_of_direction,
    rope_percentage,
)

# Assimetria de dois grupos: controles vs pacientes
np.random.seed(42)
controls = np.random.randn(30) * 0.5 + 2.5     # espessura média ≈ 2.5mm
patients = np.random.randn(20) * 0.6 + 2.1      # espessura média ≈ 2.1mm (atrofia)

best = BayesianGroupComparison(rope=(-0.1, 0.1))  # ROPE para d de Cohen
trace = best.fit(
    group1=controls,
    group2=patients,
    group_names=["Controles", "Pacientes"],
    config=SamplerConfig(
        n_samples=2000,
        n_chains=2,
        backend="pymc",
    ),
)

summary = best.summary(hdi_prob=0.94)
print(summary)

# ── ROPE analysis ────────────────────────────────────────────────────
rope = best.rope_analysis()
print(f"\nROPE analysis:")
print(f"  P(inside ROPE): {rope['p_rope']:.3f}")
print(f"  P(below ROPE):  {rope['p_below']:.3f}")
print(f"  P(above ROPE):  {rope['p_above']:.3f}")

# %% [markdown]
# ## B.2 — Horseshoe Regression (Feature Selection Esparsa)
#
# Ideal para selecionar quais features espectrais/morfométricas
# predizem um desfecho clínico:

# %%
# Features: frequência-banda asymmetry + OT distance + classical AI
n = 50
X = np.random.randn(n, 20)   # 20 features (muitas irrelevantes)
y = 0.8 * X[:, 2] - 0.5 * X[:, 7] + np.random.randn(n) * 0.3

hs = HorseshoeRegression(tau0=1.0, slab_scale=2.0, slab_df=4)
trace_hs = hs.fit(
    X, y,
    feature_names=[f"feat_{i}" for i in range(20)],
    config=SamplerConfig(n_samples=1000, n_chains=2),
)

# Features selecionadas
selected = hs.selected_features(threshold=0.5)
print(f"\nFeatures selecionadas pelo Horseshoe: {selected}")
print(f"Esperado: feat_2 e feat_7")

# %% [markdown]
# ## B.3 — Visualizações Bayesianas (`bayes_viz`)
#
# O módulo `bayes_viz` complementa ArviZ com plots especializados:

# %%
from corticalfields.bayes_viz import (
    plot_posterior_density,
    plot_forest,
    plot_ridgeline,
    plot_posterior_hdi_rope,
    plot_prior_posterior_update,
    plot_group_comparison,
    plot_model_comparison,
    plot_posterior_predictive_check,
    plot_shrinkage,
    plot_coefficient_path,
    plot_brain_posterior_map,
)

# ── Catálogo de plots ────────────────────────────────────────────────
# plot_posterior_density(trace, var="mu_diff")          # Density + HDI
# plot_forest(trace, var_names=["mu1", "mu2", "sigma"]) # Forest plot
# plot_ridgeline(traces_list, labels=["M1", "M2"])      # Ridgeline comparison
# plot_posterior_hdi_rope(trace, var="d", rope=(-0.1,0.1)) # HDI + ROPE
# plot_prior_posterior_update(prior, posterior)           # Prior → Posterior
# plot_group_comparison(trace)                           # BEST visualization
# plot_shrinkage(trace_hs)                               # Horseshoe shrinkage
# plot_brain_posterior_map(posterior_map, vertices, faces) # 3D posterior

print("Catálogo de 17 funções de visualização Bayesiana.")
print("Todas seguem estilo publication-quality com paleta Bayesian.")

# %% [markdown]
# ## B.4 — Diagnósticos e Métricas

# %%
# ── Diagnósticos de convergência ─────────────────────────────────────
# diag = compute_diagnostics(trace, var_names=["mu_diff", "sigma_diff"])
# print(f"R-hat: {diag['rhat']}")
# print(f"ESS bulk: {diag['ess_bulk']}")
# print(f"Divergences: {diag['divergences']}")

# ── Bayesian R² ──────────────────────────────────────────────────────
# r2 = bayesian_r2(trace, y_true=y)
# print(f"Bayesian R²: {r2['mean']:.3f} [{r2['hdi_low']:.3f}, {r2['hdi_high']:.3f}]")

# ── Probability of Direction ─────────────────────────────────────────
# pd = probability_of_direction(posterior_samples)
# print(f"P(direction): {pd:.3f}")

# ── ROPE ─────────────────────────────────────────────────────────────
# rp = rope_percentage(posterior_samples, rope=(-0.1, 0.1))
# print(f"% in ROPE: {rp:.3f}")

print("compute_diagnostics(), bayesian_r2(), probability_of_direction(), rope_percentage()")

# %% [markdown]
# ## B.5 — Prior Elicitation

# %%
from corticalfields.bayesian import elicit_prior, enigma_informed_prior

# ── Via PreliZ ───────────────────────────────────────────────────────
# prior = elicit_prior(
#     distribution="Normal",
#     mass=0.95,           # 95% da massa entre low e high
#     low=1.5, high=4.0,   # bounds para espessura cortical
# )
# print(f"Prior elicitada: {prior}")

# ── Priors informadas por ENIGMA ─────────────────────────────────────
# prior_enigma = enigma_informed_prior(
#     feature="thickness",
#     structure="hippocampus",
#     population="healthy_adults",
# )
print("elicit_prior() usa PreliZ para elicitação interativa.")
print("enigma_informed_prior() usa dados ENIGMA como priors informadas.")

# %% [markdown]
# # 10.6 — Resumo Final
#
# Parabéns! Ao completar os 10 tutoriais, você aprendeu:
#
# | Tutorial | Módulo(s) | Conceito central |
# |----------|-----------|------------------|
# | 01 | `surface` | Carregar e inspecionar superfícies |
# | 02 | `spectral`, `backends` | Laplace–Beltrami e autovetores |
# | 03 | `spectral` | HKS, WKS, GPS — descritores multi-escala |
# | 04 | `pointcloud` | Pipeline T1w → superfície sem FreeSurfer |
# | 05 | `subcortical` | Superfícies subcorticais + espectral |
# | 06 | `functional_maps`, `asymmetry` | Correspondência inter-hemisférica |
# | 07 | `transport` | Distâncias de Wasserstein |
# | 08 | `kernels`, `normative`, `surprise` | Modelagem normativa GP |
# | 09 | `graphs`, `eda_qc`, `distance_stats` | Redes, QC, inferência |
# | 10 | `brainplots`, `bayesian`, `bayes_viz` | Publicação, Bayes |
#
# ## Pipeline completo típico:
#
# ```
# T1w → from_t1w() ou load_freesurfer_surface()
#     → compute_eigenpairs() — autovetores LB
#     → heat_kernel_signature() / wave_kernel_signature() — descritores
#     → compute_interhemispheric_map() — assimetria
#     → sliced_wasserstein_distance() — OT distances
#     → CorticalNormativeModel.fit() / .predict() — normative
#     → compute_surprise() — anomaly maps
#     → mdmr() / hsic() / kernel_ridge_regression() — inferência
#     → plot_composite_figure() — publicação
# ```
#
# **Bom trabalho!** 🧠
