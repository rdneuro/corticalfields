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
# # Tutorial 07 — Optimal Transport para Morfometria Cortical
#
# O **transporte ótimo (OT)** mede quanta "massa" precisa ser movida
# para transformar uma distribuição em outra. Aplicado a superfícies
# corticais, a **distância de Wasserstein** entre dois hemisférios
# quantifica quão diferentes eles são geometricamente — sem parcelas,
# sem registro, sem atlas.
#
# Neste tutorial:
#
# 1. `sliced_wasserstein_distance()` — rápido, escalável, CPU
# 2. `sliced_wasserstein_with_features()` — geometria + morfometria
# 3. `sinkhorn_divergence()` — GPU, resolução espacial ajustável
# 4. `pairwise_wasserstein_matrix()` — distâncias para toda a coorte
# 5. `wasserstein_kernel()` — kernel PD para regressão (KRR)
# 6. `interhemispheric_wasserstein_distances()` — per-subject asymmetry
#
# ---
#
# ## 7.1 — Sliced Wasserstein Distance
#
# A SWD projeta ambas nuvens de pontos em direções aleatórias 1D,
# computa Wasserstein exato em cada projeção, e faz a média.
# Complexidade: $O(N \log N \times n_{\text{projections}})$.
#
# Para ~150k pontos com 200 projeções: **1–5 segundos**.

# %%
import numpy as np
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

import corticalfields as cf
from corticalfields.spectral import compute_eigenpairs
from corticalfields.transport import (
    sliced_wasserstein_distance,
    sliced_wasserstein_with_features,
    sinkhorn_divergence,
    pairwise_wasserstein_matrix,
    interhemispheric_wasserstein_distances,
    wasserstein_kernel,
    functional_map_kernel,
    TransportResult,
)

SUBJECTS_DIR = "./data/freesurfer"
SUBJECT_ID = "sub-010002"

surf_lh = cf.load_freesurfer_surface(SUBJECTS_DIR, SUBJECT_ID, "lh", "pial",
                                      overlays=["thickness", "curv", "sulc"])
surf_rh = cf.load_freesurfer_surface(SUBJECTS_DIR, SUBJECT_ID, "rh", "pial",
                                      overlays=["thickness", "curv", "sulc"])

# Espelhar RH para comparação (x → −x)
rh_mirrored = surf_rh.vertices.copy()
rh_mirrored[:, 0] *= -1

# %%
# ── SWD entre hemisférios (puro 3D) ─────────────────────────────────
d_geom = sliced_wasserstein_distance(
    surf_lh.vertices,
    rh_mirrored,
    n_projections=200,
    p=2,                  # Wasserstein-2
    seed=42,
)
print(f"SW₂(LH, RH_mirrored) = {d_geom:.4f} mm")

# ── SWD com features morfométricas ──────────────────────────────────
d_morph = sliced_wasserstein_with_features(
    coords_X=surf_lh.vertices,
    coords_Y=rh_mirrored,
    features_X=surf_lh.get_overlay("thickness")[:, None],
    features_Y=surf_rh.get_overlay("thickness")[:, None],
    feature_weight=1.0,    # peso relativo features/coordenadas
    n_projections=200,
    seed=42,
)
print(f"SW₂(LH, RH) com thickness = {d_morph:.4f}")

# %% [markdown]
# ## 7.2 — Sinkhorn Divergence (GPU)
#
# O Sinkhorn usa regularização entrópica para aproximar o OT:
# - `blur`: raio de suavização (mm). Menor = mais preciso, mais lento
# - Requer `geomloss` + `pykeops` para GPU
# - Linear em memória, escala para milhões de pontos

# %%
# d_sink = sinkhorn_divergence(
#     X=surf_lh.vertices,
#     Y=rh_mirrored,
#     blur=5.0,          # mm — resolução espacial
#     p=2,
#     device="cuda",     # 'cuda' ou 'cpu'
# )
# print(f"Sinkhorn divergence: {d_sink.distance:.4f} mm")
print("Sinkhorn requer geomloss+pykeops. Descomente para executar com GPU.")

# %% [markdown]
# ## 7.3 — Distâncias pareadas para a coorte

# %%
# Para uma coorte de N sujeitos, computar a matriz N×N de distâncias:
# point_clouds = [pc_sub1, pc_sub2, ..., pc_subN]
# D = pairwise_wasserstein_matrix(
#     point_clouds,
#     n_projections=200,
#     method="sliced",    # 'sliced' ou 'sinkhorn'
#     n_jobs=-1,          # paralelizar
# )
# print(f"Matriz de distâncias: {D.shape}")

# ── Exemplo com 2 sujeitos ───────────────────────────────────────────
# (use múltiplos sujeitos DS000221 para uma matriz real)
from corticalfields.pointcloud import from_cortical_surface

pc_lh = from_cortical_surface(surf_lh)
pc_rh_mirror = from_cortical_surface(surf_rh)
pc_rh_mirror.points[:, 0] *= -1

d = sliced_wasserstein_distance(
    pc_lh.points, pc_rh_mirror.points, n_projections=200, seed=42,
)
print(f"Distância LH↔RH (sujeito 1): {d:.4f} mm")

# %% [markdown]
# ## 7.4 — Wasserstein Kernel para Regressão
#
# O kernel Gaussiano de Wasserstein é **provadamente PD** (Kolouri et al.,
# CVPR 2016):
#
# $$K(X_i, X_j) = \exp\!\left(-\frac{SW(X_i, X_j)^2}{2σ^2}\right)$$
#
# Isso permite usar a geometria cortical diretamente como preditor
# em Kernel Ridge Regression (KRR), SVM, ou HSIC:
#
# **SW distance → kernel → predict HADS score**

# %%
# Exemplo com distâncias fictícias
n_subjects = 12
D_fake = np.random.rand(n_subjects, n_subjects)
D_fake = (D_fake + D_fake.T) / 2
np.fill_diagonal(D_fake, 0)

# Converter distâncias em kernel PD
K = wasserstein_kernel(D_fake, sigma=None)  # sigma=None → heurística automática
print(f"Kernel matrix: {K.shape}")
print(f"  PD? eigenvalues: min = {np.linalg.eigvalsh(K).min():.6f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
im1 = axes[0].imshow(D_fake, cmap="YlOrRd")
axes[0].set_title("Matriz de distâncias SW")
plt.colorbar(im1, ax=axes[0])
im2 = axes[1].imshow(K, cmap="Blues")
axes[1].set_title("Kernel Gaussiano de Wasserstein")
plt.colorbar(im2, ax=axes[1])
plt.tight_layout()
plt.savefig("07_wasserstein_kernel.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 7.5 — Resumo
#
# | Função | Backend | Velocidade | Uso |
# |--------|---------|------------|-----|
# | `sliced_wasserstein_distance()` | POT (CPU) | ~5s/par | Distâncias individuais |
# | `sliced_wasserstein_with_features()` | POT (CPU) | ~5s/par | OT no espaço geom+morph |
# | `sinkhorn_divergence()` | GeomLoss (GPU) | ~1s/par | Alta resolução, GPU |
# | `pairwise_wasserstein_matrix()` | Paralelo | ~min/coorte | Matriz N×N |
# | `wasserstein_kernel()` | NumPy | Instantâneo | D → K Gaussiano PD |
# | `functional_map_kernel()` | NumPy | Instantâneo | FM distances → K |
#
# **Próximo**: Tutorial 08 — Modelagem Normativa e Surprise Maps
