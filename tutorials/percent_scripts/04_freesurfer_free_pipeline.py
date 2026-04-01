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
# # Tutorial 04 — Pipeline sem FreeSurfer: T1w → Superfície Cortical
#
# O FreeSurfer leva **6–24 horas** por sujeito. A CorticalFields oferece
# uma via alternativa que extrai superfícies corticais diretamente de
# imagens T1w em **~2 minutos com GPU**.
#
# Neste tutorial, você aprenderá:
#
# 1. A classe `CorticalPointCloud` e suas diferenças vs `CorticalSurface`
# 2. `from_t1w()` — pipeline completo T1w → superfície
# 3. Backend "morphological" (deepbet + deep_atropos + marching cubes)
# 4. Backend "brainnet" (SimNIBS BrainNet — DL end-to-end)
# 5. `compute_mesh_eigenpairs()` vs `compute_pointcloud_eigenpairs()`
# 6. Como converter entre os formatos
#
# ---
#
# ## 4.1 — Visão Geral dos Caminhos de Entrada
#
# ```
# T1w NIfTI ─┬─ from_t1w(backend="morphological") ─→ mesh (verts+faces) ──→ LB ──→ CF pipeline
#             │                                        deepbet + ANTs +
#             │                                        marching cubes (~2 min GPU)
#             │
#             └─ from_t1w(backend="brainnet") ─────→ mesh (verts+faces) ──→ LB ──→ CF pipeline
#                                                    SimNIBS BrainNet (~1 s GPU)
#
# FreeSurfer ─── load_freesurfer_surface() ──────→ CorticalSurface ─────→ LB ──→ CF pipeline
#                recon-all (6–24 h)
#
# Qualquer  ──── from_nifti_mask() ──────────────→ CorticalPointCloud ──→ LB ──→ CF pipeline
# segmentação    (marching cubes genérico)
# ```

# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

from corticalfields.pointcloud import (
    CorticalPointCloud,
    T1wExtractionResult,
    from_t1w,
    from_freesurfer_surface,
    from_cortical_surface,
    from_nifti_mask,
    compute_mesh_laplacian,
    compute_mesh_eigenpairs,
    compute_pointcloud_laplacian,
    compute_pointcloud_eigenpairs,
    estimate_normals,
    compute_point_areas,
    to_feature_matrix,
)

# %% [markdown]
# ## 4.2 — A classe `CorticalPointCloud`
#
# `CorticalPointCloud` é a contraparte mesh-free de `CorticalSurface`.
# Armazena apenas pontos 3D (sem conectividade de faces):
#
# | Atributo  | Tipo | Descrição |
# |-----------|------|-----------|
# | `points`  | `(N, 3)` | Coordenadas RAS |
# | `normals` | `(N, 3)` ou None | Normais unitárias |
# | `hemi`    | `str` | Hemisfério |
# | `overlays`| `dict` | Mapas escalares por ponto |
# | `metadata`| `dict` | Metadados |
#
# Métodos úteis: `subsample()`, `mirror_x()`, `centroid`, `bounding_box`

# %%
# ── Criar point cloud a partir de superfície do FreeSurfer ───────────
import corticalfields as cf

SUBJECTS_DIR = "./data/freesurfer"
SUBJECT_ID   = "sub-010002"

# Via CorticalSurface → PointCloud
surf = cf.load_freesurfer_surface(SUBJECTS_DIR, SUBJECT_ID, "lh", "pial",
                                   overlays=["thickness", "curv"])
pc = from_cortical_surface(surf)
print(f"PointCloud: {pc.n_points:,} pontos, overlays: {pc.overlay_names}")
print(f"  Centroide: {pc.centroid}")
bbox_min, bbox_max = pc.bounding_box
print(f"  Bounding box: {bbox_min} → {bbox_max}")

# ── Via FreeSurfer diretamente (sem carregar faces) ──────────────────
pc_direct = from_freesurfer_surface(
    SUBJECTS_DIR, SUBJECT_ID, "lh", "pial",
    overlays=["thickness", "curv"],
)
print(f"\nDireto: {pc_direct.n_points:,} pontos, overlays: {pc_direct.overlay_names}")

# %% [markdown]
# ## 4.3 — Pipeline `from_t1w()` (morphological)
#
# Este pipeline faz:
# 1. **Brain extraction** — `deepbet` ou `HD-BET` (skull-stripping via DL)
# 2. **Tissue segmentation** — `deep_atropos` (ANTsPyNet) para segmentar GM cortical
# 3. **Marching cubes** — extrai isosuperfície com smoothing Gaussiano
# 4. **Transformação voxel → RAS** — via affine do NIfTI
#
# **Requisitos**: `pip install deepbet antspynet scikit-image`

# %%
# ── Extrair superfície de um T1w ─────────────────────────────────────
# ATENÇÃO: Requer GPU e ~2 min. Descomente para executar.

T1W_PATH = "./data/t1w/sub-cepesc01_T1w.nii.gz"

# result = from_t1w(
#     t1w_path=T1W_PATH,
#     hemi="lh",                    # rótulo do hemisfério
#     backend="morphological",      # pipeline deepbet + marching cubes
#     brain_extractor="deepbet",    # ou "hdbet"
#     sigma=0.5,                    # smoothing antes de marching cubes
#     use_tissue_seg=True,          # True = deep_atropos (melhor), False = dist.transform
# )
#
# print(f"Resultado:")
# print(f"  Vértices: {result.vertices.shape}")
# print(f"  Faces: {result.faces.shape}")
# print(f"  Brain mask: {result.brain_mask.sum():,} voxels")
# print(f"  Cortical mask: {result.cortical_mask.sum():,} voxels")

# %% [markdown]
# ## 4.4 — Laplaciano na Malha Extraída
#
# Quando `from_t1w()` produz vértices E faces (backend morphological ou
# brainnet), use **`compute_mesh_eigenpairs()`** — que usa o Laplaciano
# robusto de Sharp & Crane (2020) na conectividade do marching cubes.
#
# Quando só tem pontos (sem faces), use
# **`compute_pointcloud_eigenpairs()`** — que constrói uma
# triangulação local via k-NN.
#
# **Regra**: mesh > point cloud (sempre que faces estiverem disponíveis).

# %%
# ── Mesh Laplacian (preferido) ───────────────────────────────────────
# lb_mesh = compute_mesh_eigenpairs(
#     result.vertices,
#     result.faces,
#     n_eigenpairs=300,
#     backend="auto",
# )
# print(f"LB (mesh): {lb_mesh.n_eigenpairs} eigenpairs, {lb_mesh.n_vertices} vertices")

# ── Point cloud Laplacian (fallback) ─────────────────────────────────
# lb_pc = compute_pointcloud_eigenpairs(
#     result.pointcloud.points,
#     n_eigenpairs=300,
#     n_neighbors=30,  # vizinhos para triangulação local
#     backend="auto",
# )
# print(f"LB (pointcloud): {lb_pc.n_eigenpairs} eigenpairs, {lb_pc.n_vertices} vertices")

print("Descomente as células acima para executar com dados reais.")

# %% [markdown]
# ## 4.5 — Subsampling e Espelhamento
#
# Para comparar hemisférios (assimetria), precisamos espelhar o RH:

# %%
# Subsample com farthest-point sampling (cobertura espacial uniforme)
pc_sub = pc.subsample(n_points=10000, method="farthest_point", seed=42)
print(f"Original: {pc.n_points:,} → Subsampled: {pc_sub.n_points:,}")
print(f"  Overlays preservados: {pc_sub.overlay_names}")

# Espelhar para comparação inter-hemisférica
pc_mirrored = pc.mirror_x()
print(f"\nOriginal hemi: {pc.hemi}, Mirrored hemi: {pc_mirrored.hemi}")
print(f"  x original: [{pc.points[:, 0].min():.1f}, {pc.points[:, 0].max():.1f}]")
print(f"  x mirrored: [{pc_mirrored.points[:, 0].min():.1f}, {pc_mirrored.points[:, 0].max():.1f}]")

# %%
# ── Estimar normais (quando não disponíveis) ─────────────────────────
normals = estimate_normals(pc_sub.points, n_neighbors=30)
print(f"Normais estimadas: {normals.shape}, unitárias: {np.allclose(np.linalg.norm(normals, axis=1), 1.0)}")

# ── Áreas por ponto (Voronoi estimado) ──────────────────────────────
areas = compute_point_areas(pc_sub.points, n_neighbors=10)
print(f"Áreas: shape = {areas.shape}, total ≈ {areas.sum():.0f} mm²")

# ── Feature matrix ───────────────────────────────────────────────────
feat = to_feature_matrix(pc_sub, overlay_names=["thickness", "curv"], include_coordinates=True)
print(f"Feature matrix: {feat.shape}  (3D coords + thickness + curv = 5 cols)")

# %% [markdown]
# ## 4.6 — Comparação de Qualidade: FreeSurfer vs Morphological
#
# | Método | Tempo | Erro thickness | Resolução | Requisitos |
# |--------|-------|----------------|-----------|------------|
# | FreeSurfer `recon-all` | 6–24 h | gold standard | ~160k verts | CPU |
# | CorticalFields (morphological) | ~2 min | ~0.8–1.0 mm | ~50–200k verts | GPU |
# | CorticalFields (BrainNet) | ~1 s | ~0.24 mm | ~160k verts | GPU, SimNIBS |
# | `recon-all-clinical` | ~1 min | ~0.50 mm | ~160k verts | GPU, FS7.4+ |
#
# ## 4.7 — Resumo
#
# | Função | Descrição |
# |--------|-----------|
# | `from_t1w()` | Pipeline T1w → superfície (morphological ou brainnet) |
# | `from_freesurfer_surface()` | PointCloud de superfície FS existente |
# | `from_cortical_surface()` | Converter CorticalSurface → PointCloud |
# | `from_nifti_mask()` | PointCloud de qualquer segmentação NIfTI |
# | `compute_mesh_eigenpairs()` | LB em malha (preferido quando faces existem) |
# | `compute_pointcloud_eigenpairs()` | LB em nuvem de pontos (mesh-free) |
# | `.subsample()` | Reduzir resolução com FPS ou random |
# | `.mirror_x()` | Espelhar para comparação inter-hemisférica |
#
# **Próximo**: Tutorial 05 — Superfícies subcorticais
