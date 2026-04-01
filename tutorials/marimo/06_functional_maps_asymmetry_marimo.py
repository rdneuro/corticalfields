import marimo

__generated_with = "0.22.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tutorial 06 — Functional Maps e Assimetria Cortical

    Os **functional maps** (Ovsjanikov et al., 2012) são a ferramenta
    central da CorticalFields para análise de assimetria. Uma functional
    map C é uma matriz pequena (k×k) que mapeia funções de uma superfície
    para outra no domínio espectral.

    Neste tutorial:

    1. O que é uma functional map e como funciona
    2. `compute_functional_map()` — computação inicial via least-squares
    3. `zoomout_refine()` — refinamento ZoomOut para alta resolução
    4. Extrair correspondência ponto-a-ponto e transferir funções
    5. `compute_interhemispheric_map()` — convenience para assimetria
    6. Métricas de assimetria: off-diagonal energy, diagonal dominance
    7. Decomposição por banda de frequência
    8. `asymmetry_from_functional_map()` e `asymmetry_from_wasserstein()`

    ---

    ## 6.1 — Intuição: O que é uma Functional Map?

    Dados dois hemisférios com autovetores $\{φ_i^L\}$ e $\{φ_j^R\}$,
    qualquer função $f$ no hemisf. esquerdo pode ser escrita como:
    $f = \sum_i a_i φ_i^L$ (coeficientes espectrais $a_i$).

    A functional map **C** mapeia esses coeficientes:
    $b = C \cdot a$, onde $b_j$ são os coeficientes no hemisf. direito.

    Em um **cérebro perfeitamente simétrico**, cada autovetor $φ_i^L$
    mapeia exatamente para $φ_i^R$, então C seria a **identidade**.
    A **energia off-diagonal** de C quantifica a assimetria:
    quanto mais off-diagonal, mais os hemisférios diferem.
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    import logging
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

    import corticalfields as cf
    from corticalfields.spectral import compute_eigenpairs
    from corticalfields.functional_maps import (
        FunctionalMap,
        compute_functional_map,
        compute_interhemispheric_map,
        zoomout_refine,
        functional_map_to_pointwise,
        transfer_function,
        compute_descriptor_matrix,
        functional_map_distance,
        compute_cohort_functional_maps,
    )

    SUBJECTS_DIR = "./data/freesurfer"
    SUBJECT_ID = "sub-010002"

    # Carregar ambos hemisférios e computar LB
    surf_lh = cf.load_freesurfer_surface(SUBJECTS_DIR, SUBJECT_ID, "lh", "pial",
                                          overlays=["thickness", "curv", "sulc"])
    surf_rh = cf.load_freesurfer_surface(SUBJECTS_DIR, SUBJECT_ID, "rh", "pial",
                                          overlays=["thickness", "curv", "sulc"])

    lb_lh = compute_eigenpairs(surf_lh.vertices, surf_lh.faces, n_eigenpairs=300)
    lb_rh = compute_eigenpairs(surf_rh.vertices, surf_rh.faces, n_eigenpairs=300)

    print(f"LH: {lb_lh.n_vertices:,} verts, {lb_lh.n_eigenpairs} eigenpairs")
    print(f"RH: {lb_rh.n_vertices:,} verts, {lb_rh.n_eigenpairs} eigenpairs")
    return (
        SUBJECTS_DIR,
        SUBJECT_ID,
        compute_functional_map,
        compute_interhemispheric_map,
        functional_map_to_pointwise,
        lb_lh,
        lb_rh,
        np,
        plt,
        surf_lh,
        surf_rh,
        transfer_function,
        zoomout_refine,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6.2 — Computando a Functional Map
    """)
    return


@app.cell
def _(compute_functional_map, lb_lh, lb_rh):
    # ── Functional map inicial (k=50) ────────────────────────────────────
    fm = compute_functional_map(
        lb_source=lb_lh,
        lb_target=lb_rh,
        k=50,                      # resolução espectral inicial
        descriptor_type="hks",     # 'hks', 'wks', ou 'both'
        n_descriptors=100,         # nº de escalas HKS para correspondência
        alpha_desc=1.0,            # peso do termo de preservação de descritor
        alpha_lap=1e-3,            # peso da comutatividade Laplaciana (→ C diagonal)
    )

    print(f"Functional map C: {fm.shape}")
    print(f"  Off-diagonal energy: {fm.off_diagonal_energy:.4f}")
    print(f"  Diagonal dominance:  {fm.diagonal_dominance:.4f}")
    return (fm,)


@app.cell
def _(fm, lb_lh, lb_rh, zoomout_refine):
    # ── ZoomOut refinement ───────────────────────────────────────────────
    fm_refined = zoomout_refine(
        fm, lb_lh, lb_rh,
        k_final=200,          # resolução espectral final
        n_iterations=10,      # iterações de refinamento
    )

    print(f"\nRefinada: {fm_refined.shape}")
    print(f"  Off-diagonal energy: {fm_refined.off_diagonal_energy:.4f}")
    print(f"  Diagonal dominance:  {fm_refined.diagonal_dominance:.4f}")
    return (fm_refined,)


@app.cell
def _(fm, fm_refined, np, plt):
    # ── Visualizar a matriz C ────────────────────────────────────────────
    _fig, _axes = plt.subplots(1, 2, figsize=(14, 6))
    im1 = _axes[0].imshow(np.abs(fm.C), cmap='hot', vmin=0, vmax=1)
    # Antes do ZoomOut
    _axes[0].set_title(f'C inicial (k={fm.k_source})\nOff-diag = {fm.off_diagonal_energy:.3f}')
    _axes[0].set_xlabel('Fonte (LH)')
    _axes[0].set_ylabel('Alvo (RH)')
    plt.colorbar(im1, ax=_axes[0], shrink=0.8)
    k_show = min(100, fm_refined.C.shape[0])
    # Depois do ZoomOut
    im2 = _axes[1].imshow(np.abs(fm_refined.C[:k_show, :k_show]), cmap='hot', vmin=0, vmax=1)
    _axes[1].set_title(f'C refinada (k={fm_refined.k_source}, mostrando {k_show}×{k_show})\nOff-diag = {fm_refined.off_diagonal_energy:.3f}')
    _axes[1].set_xlabel('Fonte (LH)')
    _axes[1].set_ylabel('Alvo (RH)')
    plt.colorbar(im2, ax=_axes[1], shrink=0.8)
    plt.tight_layout()
    plt.savefig('06_functional_map_C.png', dpi=150, bbox_inches='tight')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6.3 — Decomposição por Frequência

    A assimetria pode ser decomposta por banda de frequência espacial:
    - **Baixa freq** (φ₀–φ₂₀): assimetria de forma global (lobos)
    - **Média freq** (φ₂₀–φ₁₀₀): giros individuais
    - **Alta freq** (φ₁₀₀+): girificação fina (sulcos)
    """)
    return


@app.cell
def _(fm_refined):
    bands = fm_refined.frequency_band_energy()
    print("Energia off-diagonal por banda:")
    for band, energy in bands.items():
        print(f"  {band}: {energy:.4f}")

    # Modos dominantes de assimetria
    U, S, Vt = fm_refined.dominant_asymmetry_modes(n_modes=5)
    print(f"\nSingular values dos modos de assimetria: {S}")
    return S, bands


@app.cell
def _(S, bands, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4.5))
    names = list(bands.keys())
    # Barplot de bandas
    values = list(bands.values())
    _axes[0].bar(names, values, color=['#3274A1', '#E1812C', '#3A923A'])
    _axes[0].set_ylabel('Off-diagonal energy')
    _axes[0].set_title('Assimetria por banda de frequência')
    _axes[0].grid(True, alpha=0.3, axis='y')
    _axes[1].bar(range(len(S)), S, color='#9467BD')
    _axes[1].set_xlabel('Modo de assimetria')
    # Singular values
    _axes[1].set_ylabel('Singular value')
    _axes[1].set_title('Top-5 modos de assimetria')
    _axes[1].grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('06_asymmetry_bands.png', dpi=150, bbox_inches='tight')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6.4 — Correspondência ponto-a-ponto e transferência de funções
    """)
    return


@app.cell
def _(
    fm_refined,
    functional_map_to_pointwise,
    lb_lh,
    lb_rh,
    np,
    surf_lh,
    surf_rh,
    transfer_function,
):
    # ── Mapa ponto-a-ponto: para cada vértice RH, qual vértice LH corresponde?
    p2p = functional_map_to_pointwise(fm_refined, lb_lh, lb_rh)
    print(f"Correspondência: {p2p.shape} (cada vértice RH → vértice LH)")

    # ── Transferir espessura cortical do LH para o RH ───────────────────
    thick_lh = surf_lh.get_overlay("thickness")
    thick_transferred = transfer_function(fm_refined, lb_lh, lb_rh, thick_lh)
    thick_rh = surf_rh.get_overlay("thickness")

    print(f"\nEspessura LH original: mean = {thick_lh.mean():.3f}")
    print(f"Espessura transferida para RH: mean = {thick_transferred.mean():.3f}")
    print(f"Espessura RH real: mean = {thick_rh.mean():.3f}")
    print(f"Correlação (transferida vs real): {np.corrcoef(thick_transferred, thick_rh)[0,1]:.4f}")
    return thick_lh, thick_rh


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6.5 — Módulo de Assimetria
    """)
    return


@app.cell
def _(SUBJECTS_DIR, SUBJECT_ID, fm_refined, thick_lh, thick_rh):
    from corticalfields.asymmetry import (
        AsymmetryProfile,
        asymmetry_from_functional_map,
        classical_asymmetry_index,
        combined_asymmetry,
    )

    # ── Assimetria via functional map ────────────────────────────────────
    asym_fm = asymmetry_from_functional_map(fm_refined)
    print(f"Assimetria (functional map):")
    print(f"  global_asymmetry: {asym_fm.global_asymmetry:.4f}")
    print(f"  Bandas: {asym_fm.band_asymmetry}")

    # ── Assimetria clássica (AI = (L-R)/(L+R) por ROI) ──────────────────
    from corticalfields.surface import load_annot
    labels_lh, names_lh = load_annot(SUBJECTS_DIR, SUBJECT_ID, "lh", "aparc")
    labels_rh, names_rh = load_annot(SUBJECTS_DIR, SUBJECT_ID, "rh", "aparc")

    ai_classical = classical_asymmetry_index(
        feature_lh=thick_lh,
        feature_rh=thick_rh,
        labels_lh=labels_lh,
        labels_rh=labels_rh,
        label_names=names_lh,
    )
    print(f"\nAssimetria clássica (AI) por ROI (primeiras 5):")
    for name, ai in list(ai_classical.items())[:5]:
        print(f"  {name}: AI = {ai:.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6.6 — Convenience: `compute_interhemispheric_map()`
    """)
    return


@app.cell
def _(compute_interhemispheric_map, lb_lh, lb_rh):
    # Faz tudo de uma vez: FM initial + ZoomOut
    fm_auto = compute_interhemispheric_map(
        lb_lh, lb_rh,
        k=50, k_final=200,
        descriptor_type="hks",
        n_descriptors=100,
        alpha_lap=1e-3,
        zoomout=True,
        n_zoomout_iters=10,
    )
    print(f"Inter-hemispheric map: {fm_auto.shape}")
    print(f"  Off-diagonal: {fm_auto.off_diagonal_energy:.4f}")
    print(f"  Diagonal dominance: {fm_auto.diagonal_dominance:.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6.7 — Processamento em batch (coorte inteira)
    """)
    return


@app.cell
def _():
    # Se tivéssemos LB para múltiplos sujeitos:
    # lb_pairs = [(lb_lh_s1, lb_rh_s1), (lb_lh_s2, lb_rh_s2), ...]
    # subject_ids = ["sub-001", "sub-002", ...]
    # fms = compute_cohort_functional_maps(lb_pairs, subject_ids, k=50, k_final=200)
    #
    # # Distância entre mapas de dois sujeitos:
    # d = functional_map_distance(fms[0], fms[1], metric="frobenius")
    print("Veja o Tutorial 09 para usar essas distâncias em análises estatísticas.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6.8 — Resumo

    | Função | Descrição |
    |--------|-----------|
    | `compute_functional_map()` | C matrix via least-squares |
    | `zoomout_refine()` | Refinamento iterativo para alta resolução |
    | `compute_interhemispheric_map()` | Convenience (FM + ZoomOut) |
    | `functional_map_to_pointwise()` | C → correspondência vértice-a-vértice |
    | `transfer_function()` | Transferir overlay entre hemisférios |
    | `FunctionalMap.off_diagonal_energy` | Assimetria total |
    | `FunctionalMap.diagonal_dominance` | Fração de energia na diagonal |
    | `FunctionalMap.frequency_band_energy()` | Assimetria por banda |
    | `asymmetry_from_functional_map()` | Profile de assimetria multi-escala |
    | `classical_asymmetry_index()` | AI clássico por ROI |

    **Próximo**: Tutorial 07 — Optimal Transport
    """)
    return


if __name__ == "__main__":
    app.run()
