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
# # Tutorial 08 — Modelagem Normativa e Mapas de Surprise
#
# A **modelagem normativa** é o pipeline clínico central da CorticalFields:
#
# 1. **Treinar** um GP na coorte de referência (controles saudáveis)
# 2. **Predizer** a distribuição esperada para cada paciente
# 3. **Pontuar** anomalias por vértice via z-scores e surprise
#
# O GP usa o **kernel Matérn espectral** (Tutorial 02–03) para respeitar
# a geometria cortical — correlações seguem a superfície, não a distância
# euclidiana.
#
# Neste tutorial:
#
# 1. `SpectralMaternKernel` — kernel on the manifold
# 2. `CorticalNormativeModel` — SVGP normative model
# 3. Training: `.fit()` com minibatch variational inference
# 4. Prediction: `.predict()` → NormativeResult
# 5. `compute_surprise()` → SurpriseMap
# 6. Agregação por parcelação e rede
# 7. Salvar/carregar modelos
#
# ---
#
# ## 8.1 — O Kernel Matérn Espectral
#
# O kernel Matérn na variedade (Borovitskiy et al., NeurIPS 2020):
#
# $$k(x, y) = σ^2 \sum_{i=0}^{L-1} S_ν(λ_i)\, φ_i(x)\, φ_i(y)$$
#
# onde a densidade espectral é:
# $S_ν(λ) = (2ν/κ^2 + λ)^{-(ν + d/2)}$
#
# Parâmetros **aprendidos** durante treinamento:
# - `lengthscale` (κ⁻¹): extensão espacial das correlações em mm
# - `outputscale` (σ²): variância do sinal
# - `noise`: variância do ruído de observação
#
# Parâmetro **fixo**:
# - `nu` (ν): suavidade (½ = exponencial, 3/2, 5/2, ∞ = SE)

# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
import gpytorch
import logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

import corticalfields as cf
from corticalfields.spectral import compute_eigenpairs
from corticalfields.kernels import SpectralMaternKernel, SpectralHeatKernel, full_kernel_matrix
from corticalfields.normative import CorticalNormativeModel, NormativeResult

SUBJECTS_DIR = "./data/freesurfer"
SUBJECT_ID = "sub-010002"

surf = cf.load_freesurfer_surface(SUBJECTS_DIR, SUBJECT_ID, "lh", "pial",
                                   overlays=["thickness", "curv", "sulc"])
lb = compute_eigenpairs(surf.vertices, surf.faces, n_eigenpairs=300)

# %%
# ── Instanciar o kernel ──────────────────────────────────────────────
kernel = SpectralMaternKernel(lb, nu=2.5)
print(f"SpectralMaternKernel:")
print(f"  ν = {kernel.nu}")
print(f"  lengthscale = {kernel.lengthscale.item():.4f} mm")
print(f"  outputscale = {kernel.outputscale.item():.4f}")
print(f"  Eigenvalues: {kernel.eigenvalues.shape}")

# Avaliar kernel entre 4 vértices
idx = torch.tensor([[0], [100], [1000], [10000]])
K = kernel(idx, idx).evaluate()
print(f"\nKernel matrix (4 vértices):\n{K.detach().numpy()}")

# %% [markdown]
# ## 8.2 — Treinando o Modelo Normativo
#
# `CorticalNormativeModel` encapsula:
# - Seleção de inducing points via FPS no GPS embedding
# - SVGP (Sparse Variational GP) para escalabilidade
# - Treinamento via ELBO com minibatch SGD
# - Vertex mask para alinhar dados com a malha LB

# %%
# ── Preparar dados de treinamento ────────────────────────────────────
# Idealmente: carregar espessura de MÚLTIPLOS controles saudáveis
# Aqui usamos um único sujeito como demonstração

thickness = surf.get_overlay("thickness")

# Simular coorte de 6 controles (com ruído)
np.random.seed(42)
n_controls = 6
train_data = np.column_stack([
    thickness + np.random.randn(surf.n_vertices) * 0.1
    for _ in range(n_controls)
])
print(f"Dados de treinamento: {train_data.shape}  (vértices × sujeitos)")

# ── Instanciar e treinar modelo ──────────────────────────────────────
model = CorticalNormativeModel(
    lb=lb,
    nu=2.5,              # suavidade Matérn
    n_inducing=512,      # pontos induzidos (↑ = melhor, mais lento)
    device="cpu",        # 'cuda' se disponível
    seed=42,
)

history = model.fit(
    train_features=train_data,      # (N, S) multi-sujeito
    feature_name="thickness",
    n_epochs=50,                    # 50–200 para convergência
    lr=0.01,                        # learning rate
    batch_size=4096,                # vértices por minibatch
    verbose=True,
)

# %%
# ── Curva de treinamento ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(history["loss"], ".-", color="#3274A1")
ax.set_xlabel("Época")
ax.set_ylabel("−ELBO (loss)")
ax.set_title("Convergência do treinamento SVGP")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("08_training_loss.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 8.3 — Predição Normativa para um Paciente

# %%
# ── Simular dados de um "paciente" com atrofia focal ─────────────────
patient_thickness = thickness.copy()
# Injetar atrofia artificial em um grupo de vértices
atrophy_region = np.abs(surf.vertices[:, 1] - 30) < 15  # região Y ≈ 30mm
patient_thickness[atrophy_region] -= 0.8  # reduzir 0.8mm

# ── Predição ─────────────────────────────────────────────────────────
result = model.predict(patient_thickness, batch_size=5000)

print(f"NormativeResult:")
print(f"  mean:     shape = {result.mean.shape}")
print(f"  variance: shape = {result.variance.shape}")
print(f"  z_score:  min = {np.nanmin(result.z_score):.2f}, max = {np.nanmax(result.z_score):.2f}")
print(f"  surprise: mean = {np.nanmean(result.surprise):.4f}")

# %% [markdown]
# ## 8.4 — Computando o Mapa de Surprise

# %%
from corticalfields.surprise import compute_surprise, SurpriseMap, combined_surprise

surprise_map = compute_surprise(
    observed=patient_thickness,
    predicted_mean=result.mean,
    predicted_var=result.variance,
    prior_anomaly_prob=0.05,         # prior P(anomalous) = 5%
    anomaly_variance_factor=10.0,    # componente anômala tem 10× a variância
)

print(f"SurpriseMap:")
print(f"  {surprise_map.n_valid}/{surprise_map.n_vertices} vértices válidos")
print(f"  z-score: [{np.nanmin(surprise_map.z_score):.2f}, {np.nanmax(surprise_map.z_score):.2f}]")
print(f"  surprise: [{surprise_map.surprise.min():.3f}, {surprise_map.surprise.max():.3f}]")
print(f"  P(anomalous): max = {surprise_map.anomaly_probability.max():.4f}")

# ── Threshold ────────────────────────────────────────────────────────
anomalous = surprise_map.threshold(z_thresh=2.0, direction="negative")
print(f"\nVértices com z < −2: {anomalous.sum():,} ({anomalous.mean()*100:.1f}%)")

# %%
# ── Visualizar z-scores no córtex ────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
idx = np.random.RandomState(0).choice(surf.n_vertices, 25000, replace=False)

for ax, data, title, cmap, vlim in zip(axes,
    [patient_thickness, result.z_score, surprise_map.anomaly_probability],
    ["Espessura paciente (mm)", "z-score normativo", "P(anomalous)"],
    ["YlOrRd", "RdBu_r", "hot"],
    [(1, 4), (-3, 3), (0, 0.5)]):

    sc = ax.scatter(surf.vertices[idx, 1], surf.vertices[idx, 2],
                    c=data[idx], cmap=cmap, s=0.3, vmin=vlim[0], vmax=vlim[1])
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_axis_off()
    plt.colorbar(sc, ax=ax, shrink=0.7)

plt.suptitle("Modelagem normativa — paciente com atrofia simulada", fontsize=13)
plt.tight_layout()
plt.savefig("08_normative_results.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 8.5 — Agregação por Parcelação e Rede

# %%
from corticalfields.surface import load_annot

labels, names = load_annot(SUBJECTS_DIR, SUBJECT_ID, "lh", "aparc")

# ── Surprise médio por região ────────────────────────────────────────
region_scores = surprise_map.aggregate_by_parcellation(
    labels=labels, label_names=names, metric="mean_surprise",
)
print("Top-5 regiões por surprise:")
for name, score in sorted(region_scores.items(), key=lambda x: -x[1])[:5]:
    print(f"  {name}: surprise = {score:.4f}")

# ── Multi-métrica por rede (se tiver Yeo labels) ────────────────────
# network_scores = surprise_map.aggregate_by_network(
#     labels=yeo_labels, network_names=yeo_names,
# )

# %% [markdown]
# ## 8.6 — Combinando Surprise de Múltiplas Features

# %%
# Treinar um 2º modelo para curvatura
# model_curv = CorticalNormativeModel(lb, nu=2.5, n_inducing=512)
# model_curv.fit(curv_train_data, feature_name="curv")
# result_curv = model_curv.predict(patient_curv)
# surprise_curv = compute_surprise(patient_curv, result_curv.mean, result_curv.variance)

# Combinar via soma, max ou Fisher:
# combined = combined_surprise([surprise_thick, surprise_curv], method="fisher")
print("combined_surprise() combina múltiplas features via 'sum', 'max', ou 'fisher'")

# %% [markdown]
# ## 8.7 — Salvar/Carregar Modelo

# %%
# Salvar
model.save("./data/normative_model_thickness")

# Carregar (precisa do mesmo LB)
model2 = CorticalNormativeModel(lb, nu=2.5, n_inducing=512)
model2.load("./data/normative_model_thickness")
print(f"Modelo carregado: fitted={model2._is_fitted}")

# %% [markdown]
# ## 8.8 — Resumo
#
# | Classe/Função | Descrição |
# |---------------|-----------|
# | `SpectralMaternKernel` | Kernel Matérn na variedade cortical |
# | `CorticalNormativeModel` | SVGP normative model completo |
# | `.fit()` | Treinar na coorte de referência |
# | `.predict()` | Predizer + pontuar anomalias |
# | `compute_surprise()` | z-scores + surprise + P(anomalous) |
# | `SurpriseMap.threshold()` | Mask binário de anomalia |
# | `SurpriseMap.aggregate_by_parcellation()` | Scores por ROI |
# | `combined_surprise()` | Combinar múltiplas features |
# | `.save()` / `.load()` | Persistir modelo treinado |
#
# **Próximo**: Tutorial 09 — Redes cerebrais e inferência estatística
