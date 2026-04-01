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
# # Tutorial 02 — Decomposição Espectral: Laplace–Beltrami
#
# O **operador de Laplace–Beltrami (LB)** é o coração matemático da
# CorticalFields. Ele generaliza o Laplaciano ($\nabla^2$) do espaço
# euclidiano para superfícies curvas (variedades Riemannianas).
#
# Neste tutorial, você aprenderá:
#
# 1. O que é o operador LB e por que ele é fundamental
# 2. Como a discretização funciona (cotangent weights)
# 3. Como computar os autovetores e autovalores
# 4. Como escolher o número de autovetores
# 5. Como usar diferentes backends (CPU vs GPU)
# 6. Como interpretar o espectro LB do córtex
#
# ---
#
# ## 2.1 — Intuição Matemática
#
# Em uma variedade compacta $(Σ, g)$, o operador LB $Δ_g$ tem um
# **espectro discreto**:
#
# $$Δ_g φ_i = λ_i φ_i, \quad 0 = λ_0 ≤ λ_1 ≤ λ_2 ≤ …$$
#
# As **autofunções** $φ_i$ são os "harmônicos esféricos" generalizados
# para a superfície cortical. Cada uma oscila em uma frequência
# espacial crescente:
#
# - $φ_0$ é constante (DC) — mesmo valor em todo o córtex
# - $φ_1, φ_2, φ_3$ dividem o córtex em 2 metades, 3 lobos, etc.
# - $φ_{100+}$ capturam girificação fina (sulcos individuais)
#
# Os **autovalores** $λ_i$ codificam a "frequência ao quadrado":
# $λ_i \propto$ (frequência espacial)².
#
# ### Por que isso importa clinicamente?
#
# A atrofia na MTLE-HS altera a geometria cortical em escalas
# específicas. Autovetores de baixa frequência detectam atrofia
# global (encolhimento de lobo), enquanto os de alta frequência
# detectam mudanças locais na girificação.

# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import corticalfields as cf
from corticalfields.spectral import (
    compute_laplacian,
    compute_eigenpairs,
    LaplaceBeltrami,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

# ── Carregar superfície (do Tutorial 01) ──────────────────────────────
SUBJECTS_DIR = "./data/freesurfer"
SUBJECT_ID   = "sub-010002"

surf = cf.load_freesurfer_surface(
    SUBJECTS_DIR, SUBJECT_ID, hemi="lh", surface="pial",
    overlays=["thickness", "curv", "sulc"],
)
print(f"Superfície carregada: {surf.n_vertices:,} vértices, {surf.n_faces:,} faces")

# %% [markdown]
# ## 2.2 — Construindo o Laplaciano Discreto
#
# A discretização do LB em uma malha triangular produz duas matrizes
# esparsas:
#
# - **Stiffness matrix $L$** (N×N) — pesos cotangente (Meyer et al. 2003)
# - **Mass matrix $M$** (N×N) — diagonal, áreas de Voronoi agrupadas
#
# O problema generalizado de autovalores é:
#
# $$L φ = λ M φ$$
#
# A CorticalFields oferece dois backends para o Laplaciano:
#
# | Backend | Método | Vantagem |
# |---------|--------|----------|
# | `robust-laplacian` | Delaunay intrínseco (Sharp & Crane 2020) | Estável em malhas ruins |
# | fallback | Cotangente clássico (Meyer et al. 2003) | Sem dependências extras |

# %%
# ── Computar as matrizes L e M ────────────────────────────────────────
L, M = compute_laplacian(
    surf.vertices, surf.faces,
    use_robust=True,  # usa robust-laplacian se disponível
)

print(f"Stiffness L: shape = {L.shape}, nnz = {L.nnz:,} ({L.nnz / L.shape[0]**2 * 100:.4f}% denso)")
print(f"Mass M:      shape = {M.shape}, nnz = {M.nnz:,}")
print(f"Soma das áreas (tr(M)): {M.diagonal().sum():.1f} mm²")
print(f"L é simétrica? {abs(L - L.T).max():.2e}")
print(f"L é PSD? (soma de cada linha ≈ 0): max |row sum| = {abs(L.sum(axis=1)).max():.2e}")

# %% [markdown]
# ## 2.3 — Computando os Autovetores
#
# `compute_eigenpairs()` é a **função principal** para a análise
# espectral. Ela:
#
# 1. Constrói L e M via `compute_laplacian()`
# 2. Resolve o problema generalizado $L φ = λ M φ$ via ARPACK
#    (shift-invert)
# 3. Ordena e limpa os autovalores (clamp negativos → 0)
# 4. Retorna um objeto `LaplaceBeltrami`
#
# ### Escolhendo o número de autovetores
#
# A função `estimate_n_eigenpairs()` fornece uma heurística calibrada:

# %%
from corticalfields.utils import estimate_n_eigenpairs

# Diferentes modos de estimativa
for mode in ["auto", "conservative", "aggressive"]:
    k = estimate_n_eigenpairs(surf.n_vertices, mode=mode)
    print(f"  modo '{mode}': k = {k} autovetores")

# Estimativa Weyl (usa área da superfície)
k_weyl = estimate_n_eigenpairs(
    surf.n_vertices,
    surface_area_mm2=surf.total_area,
    mode="weyl",
)
print(f"  modo 'weyl' (área = {surf.total_area:.0f} mm²): k = {k_weyl}")

# %% [markdown]
# ### Regra prática
#
# | Tipo de malha | Nº vértices | K recomendado |
# |---------------|-------------|---------------|
# | Córtex (FreeSurfer) | ~150k | 200–300 |
# | Subcortical (marching cubes) | 2–5k | 50–150 |
# | Muito pequena | <1k | 20–50 |

# %%
# ── Computar 300 autovetores ──────────────────────────────────────────
# ATENÇÃO: Isso pode levar 1–3 minutos em CPU.
# Em GPU (backend='cupy'), leva ~30s.

lb = compute_eigenpairs(
    surf.vertices, surf.faces,
    n_eigenpairs=300,
    use_robust=True,
    backend="auto",  # 'auto' seleciona: cupy → torch → scipy
)

print(f"\nResultado: {lb.n_eigenpairs} autovetores, {lb.n_vertices:,} vértices")
print(f"  λ_0 = {lb.eigenvalues[0]:.8f}  (deve ser ≈ 0)")
print(f"  λ_1 = {lb.eigenvalues[1]:.6f}")
print(f"  λ_10 = {lb.eigenvalues[10]:.4f}")
print(f"  λ_100 = {lb.eigenvalues[100]:.2f}")
print(f"  λ_299 = {lb.eigenvalues[-1]:.2f}")

# %% [markdown]
# ## 2.4 — Interpretando o Espectro
#
# O decaimento dos autovalores segue a **lei de Weyl**:
#
# $$λ_k \sim \frac{4πk}{A} \quad (k \to \infty)$$
#
# onde $A$ é a área total da superfície. Isso significa que:
# - Superfícies maiores têm espectros mais densos (mais modos por intervalo de λ)
# - A inclinação do gráfico λ vs k revela a dimensão da superfície (d=2)

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# ── Gráfico 1: Espectro completo ─────────────────────────────────────
k = np.arange(lb.n_eigenpairs)
axes[0].plot(k, lb.eigenvalues, "o-", markersize=1.5, color="#3274A1")
axes[0].set_xlabel("Índice k")
axes[0].set_ylabel("Autovalor $λ_k$")
axes[0].set_title("Espectro do Laplace–Beltrami")
axes[0].grid(True, alpha=0.3)

# Lei de Weyl teórica
weyl = 4 * np.pi * k / surf.total_area
axes[0].plot(k, weyl, "--", color="red", alpha=0.7, label="Lei de Weyl")
axes[0].legend()

# ── Gráfico 2: Espaçamento entre autovalores ─────────────────────────
gaps = np.diff(lb.eigenvalues)
axes[1].semilogy(gaps, ".", markersize=2, color="#E1812C")
axes[1].set_xlabel("Índice k")
axes[1].set_ylabel("Gap espectral $Δλ_k$")
axes[1].set_title("Gaps espectrais (log scale)")
axes[1].grid(True, alpha=0.3)

# ── Gráfico 3: Primeiros 20 autovalores (zoom) ───────────────────────
axes[2].bar(range(20), lb.eigenvalues[:20], color="#3A923A", alpha=0.8)
axes[2].set_xlabel("Índice k")
axes[2].set_ylabel("$λ_k$")
axes[2].set_title("Primeiros 20 autovalores")
axes[2].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("02_eigenspectrum.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 2.5 — Visualizando os Autovetores no Córtex
#
# Cada autovetor $φ_k$ é um mapa escalar por vértice. Os primeiros
# autovetores definem "coordenadas naturais" na superfície cortical:

# %%
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(20, 10))

eigenfunctions_to_show = [1, 2, 3, 5, 10, 50]
idx = np.random.RandomState(0).choice(surf.n_vertices, 25000, replace=False)

for i, k_idx in enumerate(eigenfunctions_to_show):
    ax = fig.add_subplot(2, 3, i + 1, projection="3d")
    phi_k = lb.eigenvectors[:, k_idx]

    sc = ax.scatter(
        surf.vertices[idx, 0], surf.vertices[idx, 1], surf.vertices[idx, 2],
        c=phi_k[idx], cmap="RdBu_r", s=0.3,
        vmin=-np.percentile(np.abs(phi_k), 98),
        vmax=+np.percentile(np.abs(phi_k), 98),
    )
    ax.set_title(f"$φ_{{{k_idx}}}$  (λ = {lb.eigenvalues[k_idx]:.3f})", fontsize=11)
    ax.view_init(elev=0, azim=-90)
    ax.set_axis_off()
    plt.colorbar(sc, ax=ax, shrink=0.5, pad=0.05)

plt.suptitle("Autovetores do Laplace–Beltrami no córtex", fontsize=14, y=0.98)
plt.tight_layout()
plt.savefig("02_eigenvectors_cortex.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### Interpretação dos autovetores
#
# - **φ₁**: divide o córtex em anterior/posterior (ou superior/inferior)
# - **φ₂, φ₃**: subdivisões ortogonais de larga escala
# - **φ₁₀**: começa a capturar giros individuais
# - **φ₅₀+**: resolve sulcos e girificação fina
#
# Esses autovetores formam uma **base ortonormal** no espaço L²
# ponderado pela massa:
#
# $$\langle φ_i, φ_j \rangle_M = φ_i^T M φ_j = δ_{ij}$$

# %%
# Verificar ortonormalidade
M_dense = lb.mass.toarray() if hasattr(lb.mass, "toarray") else lb.mass
inner_products = lb.eigenvectors[:, :10].T @ M_dense @ lb.eigenvectors[:, :10]

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(inner_products, cmap="RdBu_r", vmin=-0.1, vmax=1.1)
ax.set_title("$⟨φ_i, φ_j⟩_M$ — deve ser ≈ identidade")
ax.set_xlabel("j"); ax.set_ylabel("i")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig("02_orthonormality.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 2.6 — Backends de Computação
#
# A CorticalFields suporta três backends para o eigensolver:
#
# | Backend | Pacote | Velocidade | Memória |
# |---------|--------|------------|---------|
# | `scipy` | scipy.sparse | 1× (baseline) | CPU RAM |
# | `cupy` | cupy + cupyx | ~3–5× mais rápido | GPU VRAM |
# | `torch` | torch.sparse | ~2–3× mais rápido | GPU VRAM |
#
# `backend="auto"` tenta cupy → torch → scipy automaticamente.

# %%
from corticalfields.backends import available_backends

print("Backends disponíveis:")
for name, available in available_backends().items():
    status = "✓" if available else "✗"
    print(f"  {status} {name}")

# %% [markdown]
# ## 2.7 — Salvando e carregando o resultado espectral
#
# A decomposição LB é a computação mais cara do pipeline (1–3 min).
# Sempre salve o resultado para não precisar recomputar:

# %%
# ── Salvar ────────────────────────────────────────────────────────────
save_path = Path("./data") / "lb_cache"
save_path.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    save_path / f"{SUBJECT_ID}_lh_lb.npz",
    eigenvalues=lb.eigenvalues,
    eigenvectors=lb.eigenvectors,
    # Não salvamos L e M esparsas aqui (são grandes) — recompute se necessário
)
print(f"Resultado salvo em {save_path / f'{SUBJECT_ID}_lh_lb.npz'}")

# ── Carregar ──────────────────────────────────────────────────────────
data = np.load(save_path / f"{SUBJECT_ID}_lh_lb.npz")
lb_loaded = LaplaceBeltrami(
    stiffness=None,  # pode ser None se não precisar de L diretamente
    mass=None,
    eigenvalues=data["eigenvalues"],
    eigenvectors=data["eigenvectors"],
)
print(f"Carregado: {lb_loaded.n_eigenpairs} autovetores, {lb_loaded.n_vertices} vértices")

# %% [markdown]
# ## 2.8 — Resumo
#
# | Função / Classe | Módulo | O que faz |
# |----------------|--------|-----------|
# | `LaplaceBeltrami` | `spectral` | Container para L, M, λ, φ |
# | `compute_laplacian()` | `spectral` | Constrói L e M (cotangente ou robust) |
# | `compute_eigenpairs()` | `spectral` | Resolve Lφ = λMφ — **ponto de entrada principal** |
# | `estimate_n_eigenpairs()` | `utils` | Heurística para número ideal de autovetores |
# | `available_backends()` | `backends` | Verifica backends disponíveis |
#
# **Próximo**: Tutorial 03 — Descritores espectrais (HKS, WKS, GPS)
# que capturam geometria multi-escala a partir dos autovetores.
