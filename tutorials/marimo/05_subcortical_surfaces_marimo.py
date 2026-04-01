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
    # Tutorial 05 — Superfícies Subcorticais

    As estruturas subcorticais (hipocampo, tálamo, amígdala, caudado, etc.)
    são tão informativas quanto o córtex para epilepsia. A CorticalFields
    estende a análise espectral para **superfícies subcorticais fechadas**.

    Neste tutorial:

    1. `SubcorticalSurface` — o container para superfícies subcorticais
    2. `load_subcortical_surface()` — carrega de aseg, subcortical seg, ou HippUnfold
    3. Propriedades geométricas: esfericidade, volume, curvatura Gaussiana, energia de Willmore
    4. `subcortical_spectral_analysis()` — pipeline completo para subcortical
    5. Comparação entre estruturas e hemisférios

    ---

    ## 5.1 — A classe `SubcorticalSurface`

    Diferenças chave vs `CorticalSurface`:
    - Inclui `structure_name` e `structure_id` (e.g., "Left-Hippocampus", 17)
    - Computa `enclosed_volume` (volume 3D encerrado pela superfície fechada)
    - Computa `sphericity` — quão esférica é a estrutura:
      $\psi = \frac{36\pi V^2}{A^3}$ (1 = esfera perfeita, <1 = irregular)
    - Computa **curvatura Gaussiana** e **energia de Willmore**
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    import logging
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
    from corticalfields.subcortical import SubcorticalSurface, load_subcortical_surface, load_subcortical_from_nifti, subcortical_spectral_analysis, FS_ASEG_LABELS
    print('Estruturas subcorticais reconhecidas pelo FreeSurfer:')
    for label_id, _name in sorted(FS_ASEG_LABELS.items()):
    # Mostrar labels conhecidos
        print(f'  {label_id:3d}: {_name}')
    return load_subcortical_surface, np, plt, subcortical_spectral_analysis


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5.2 — Carregando superfícies subcorticais

    `load_subcortical_surface()` aceita três fontes:

    | Fonte | Como | Qualidade |
    |-------|------|-----------|
    | FreeSurfer `aseg.mgz` | Marching cubes na segmentação volumétrica | Boa (resolução voxel) |
    | FreeSurfer subcortical seg | HippUnfold ou amygdala_subfields | Excelente (subfields) |
    | NIfTI segmentação externa | `load_subcortical_from_nifti()` | Variável |
    """)
    return


@app.cell
def _(load_subcortical_surface):
    SUBJECTS_DIR = "./data/freesurfer"
    SUBJECT_ID = "sub-010002"

    # ── Carregar hipocampo esquerdo do aseg ──────────────────────────────
    hipp_lh = load_subcortical_surface(
        subjects_dir=SUBJECTS_DIR,
        subject_id=SUBJECT_ID,
        structure="Left-Hippocampus",  # ou structure_id=17
        source="aseg",                 # 'aseg', 'hippocampal_subfields', 'hippunfold'
        smooth_iterations=10,          # smoothing pós-marching-cubes
        target_vertices=3000,          # decimação para ~3k vértices
    )

    print(f"Hipocampo LH:")
    print(f"  Vértices: {hipp_lh.n_vertices:,}")
    print(f"  Faces:    {hipp_lh.n_faces:,}")
    print(f"  Área:     {hipp_lh.total_area:.1f} mm²")
    print(f"  Volume:   {hipp_lh.enclosed_volume:.1f} mm³")
    print(f"  Esfericidade: {hipp_lh.sphericity:.4f}")
    print(f"  Estrutura: {hipp_lh.structure_name} (ID={hipp_lh.structure_id})")
    return SUBJECTS_DIR, SUBJECT_ID, hipp_lh


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5.3 — Propriedades geométricas avançadas
    """)
    return


@app.cell
def _(hipp_lh, np):
    # ── Curvatura principal ──────────────────────────────────────────────
    k1, k2 = hipp_lh.compute_curvatures()
    gauss_curv = k1 * k2       # curvatura Gaussiana
    mean_curv = (k1 + k2) / 2  # curvatura média

    print(f"Curvatura Gaussiana: min={gauss_curv.min():.4f}, max={gauss_curv.max():.4f}")
    print(f"Curvatura média:     min={mean_curv.min():.4f}, max={mean_curv.max():.4f}")

    # ── Energia de Willmore ──────────────────────────────────────────────
    # W = ∫ H² dA — mede quão "lisa" é a superfície
    # Para uma esfera perfeita, W = 4π ≈ 12.57
    willmore = hipp_lh.willmore_energy
    print(f"Energia de Willmore: {willmore:.2f} (esfera = {4*np.pi:.2f})")
    return gauss_curv, mean_curv


@app.cell
def _(SUBJECT_ID, gauss_curv, hipp_lh, mean_curv, np, plt):
    # ── Visualizar curvatura na superfície subcortical ───────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    idx = np.arange(hipp_lh.n_vertices)

    for ax, data, title, cmap in zip(axes,
        [gauss_curv, mean_curv, np.linalg.norm(hipp_lh.vertices - hipp_lh.vertices.mean(0), axis=1)],
        ["Curvatura Gaussiana", "Curvatura média", "Distância ao centroide"],
        ["RdBu_r", "RdBu_r", "viridis"]):

        vmin, vmax = np.percentile(data, [2, 98])
        sc = ax.scatter(hipp_lh.vertices[:, 1], hipp_lh.vertices[:, 2],
                        c=data, cmap=cmap, s=2, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_axis_off()
        plt.colorbar(sc, ax=ax, shrink=0.8)

    plt.suptitle(f"Hipocampo esquerdo — {SUBJECT_ID}", fontsize=12)
    plt.tight_layout()
    plt.savefig("05_hippocampus_geometry.png", dpi=150, bbox_inches="tight")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5.4 — Pipeline espectral subcortical completo

    `subcortical_spectral_analysis()` faz tudo de uma vez:
    marching cubes → smoothing → decimação → LBO → HKS/WKS/GPS
    """)
    return


@app.cell
def _(SUBJECTS_DIR, SUBJECT_ID, subcortical_spectral_analysis):
    result = subcortical_spectral_analysis(
        subjects_dir=SUBJECTS_DIR,
        subject_id=SUBJECT_ID,
        structure="Left-Hippocampus",
        source="aseg",
        n_eigenpairs=100,          # menos que córtex (malha menor)
        hks_scales=16,
        wks_energies=16,
        gps_components=10,
        smooth_iterations=10,
        target_vertices=3000,
        backend="auto",
    )

    print(f"Pipeline completo:")
    print(f"  Superfície: {result['surface'].n_vertices} verts, {result['surface'].total_area:.0f} mm²")
    print(f"  LB: {result['lb'].n_eigenpairs} eigenpairs")
    print(f"  HKS: {result['hks'].shape}")
    print(f"  WKS: {result['wks'].shape}")
    print(f"  GPS: {result['gps'].shape}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5.5 — Comparando hemisférios subcorticais
    """)
    return


@app.cell
def _(SUBJECTS_DIR, SUBJECT_ID, hipp_lh, load_subcortical_surface):
    # ── Hipocampo direito ────────────────────────────────────────────────
    hipp_rh = load_subcortical_surface(
        SUBJECTS_DIR, SUBJECT_ID,
        structure="Right-Hippocampus", source="aseg",
        smooth_iterations=10, target_vertices=3000,
    )

    # Comparação volumétrica
    vol_ai = (hipp_lh.enclosed_volume - hipp_rh.enclosed_volume) / (
        (hipp_lh.enclosed_volume + hipp_rh.enclosed_volume) / 2
    ) * 100

    print(f"Hipocampo LH: vol = {hipp_lh.enclosed_volume:.1f} mm³, área = {hipp_lh.total_area:.1f} mm²")
    print(f"Hipocampo RH: vol = {hipp_rh.enclosed_volume:.1f} mm³, área = {hipp_rh.total_area:.1f} mm²")
    print(f"Índice de assimetria volumétrica: {vol_ai:.2f}%")

    # Comparação de esfericidade
    print(f"\nEsfericidade LH: {hipp_lh.sphericity:.4f}")
    print(f"Esfericidade RH: {hipp_rh.sphericity:.4f}")
    print(f"Δ esfericidade: {hipp_lh.sphericity - hipp_rh.sphericity:.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5.6 — Múltiplas estruturas subcorticais
    """)
    return


@app.cell
def _(SUBJECTS_DIR, SUBJECT_ID, load_subcortical_surface):
    structures = ['Left-Hippocampus', 'Right-Hippocampus', 'Left-Thalamus', 'Right-Thalamus', 'Left-Caudate', 'Right-Caudate', 'Left-Amygdala', 'Right-Amygdala']
    results = {}
    for struct in structures:
        try:
            ss = load_subcortical_surface(SUBJECTS_DIR, SUBJECT_ID, structure=struct, source='aseg', smooth_iterations=10, target_vertices=2000)
            results[struct] = {'volume': ss.enclosed_volume, 'area': ss.total_area, 'sphericity': ss.sphericity, 'n_vertices': ss.n_vertices}
        except Exception as e:
            print(f'  ⚠ {struct}: {e}')
    print(f'\n{'Estrutura':<25} {'Volume (mm³)':>12} {'Área (mm²)':>10} {'ψ':>6} {'Verts':>6}')
    print('-' * 65)
    for _name, r in results.items():
    # Tabela de resultados
        print(f'{_name:<25} {r['volume']:>12.1f} {r['area']:>10.1f} {r['sphericity']:>6.4f} {r['n_vertices']:>6}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5.7 — Resumo

    | Função | Descrição |
    |--------|-----------|
    | `SubcorticalSurface` | Container com volume, esfericidade, curvatura, Willmore |
    | `load_subcortical_surface()` | Carrega de aseg/subfields/HippUnfold |
    | `load_subcortical_from_nifti()` | Carrega de qualquer segmentação NIfTI |
    | `subcortical_spectral_analysis()` | Pipeline completo: mesh → LB → HKS/WKS/GPS |
    | `.compute_curvatures()` | Curvaturas principais (Gaussiana e média) |
    | `.willmore_energy` | Energia de Willmore (regularidade da superfície) |
    | `.sphericity` | Quão esférica é a estrutura (1 = esfera perfeita) |

    **Próximo**: Tutorial 06 — Functional Maps e assimetria
    """)
    return


if __name__ == "__main__":
    app.run()
