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
    # Tutorial 03 — Descritores Espectrais de Forma: HKS, WKS, GPS

    Os autovetores do LB sozinhos não são diretamente úteis como features
    (eles dependem da escolha de sinal ±). Os **descritores espectrais**
    combinam autovalores e autovetores em features que são:

    - **Invariantes a isometrias** (dobrar o córtex não muda o HKS)
    - **Multi-escala** (capturam curvatura local E forma global)
    - **Estáveis** numericamente (pequenas perturbações → pequenas mudanças)

    Neste tutorial, cobriremos três descritores e sua combinação:

    1. **HKS** — Heat Kernel Signature (Sun, Ovsjanikov & Guibas, 2009)
    2. **WKS** — Wave Kernel Signature (Aubry et al., 2011)
    3. **GPS** — Global Point Signature (Rustamov, 2007)
    4. **spectral_feature_matrix** — concatenação das três

    ---

    ## 3.1 — Setup
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    import corticalfields as cf
    from corticalfields.spectral import (
        compute_eigenpairs,
        heat_kernel_signature,
        wave_kernel_signature,
        global_point_signature,
        spectral_feature_matrix,
    )
    import logging
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

    # Carregar superfície e autovetores (do Tutorial 02)
    SUBJECTS_DIR = "./data/freesurfer"
    SUBJECT_ID = "sub-010002"

    surf = cf.load_freesurfer_surface(SUBJECTS_DIR, SUBJECT_ID, "lh", "pial",
                                       overlays=["thickness", "curv", "sulc"])
    lb = compute_eigenpairs(surf.vertices, surf.faces, n_eigenpairs=300, backend="auto")
    print(f"LB: {lb.n_eigenpairs} autovetores, {lb.n_vertices:,} vértices")
    return (
        global_point_signature,
        heat_kernel_signature,
        lb,
        np,
        plt,
        spectral_feature_matrix,
        surf,
        wave_kernel_signature,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.2 — Heat Kernel Signature (HKS)

    A HKS é baseada na **equação do calor** na superfície:

    $$\text{HKS}(x, t) = \sum_{i=1}^{K} e^{-λ_i t}\, φ_i(x)^2$$

    - Em **tempos curtos** ($t$ pequeno): a HKS captura **curvatura local**
      (calor se difunde pouco → sensível a vizinhança imediata)
    - Em **tempos longos** ($t$ grande): a HKS captura **forma global**
      (calor se espalha → sensível a qual lobo/giro o vértice pertence)

    Os tempos são escolhidos em escala log entre
    $t_{\min} = \frac{4\ln 10}{λ_{\max}}$ e
    $t_{\max} = \frac{4\ln 10}{λ_1}$.

    ### Por que a HKS detecta atrofia?

    A atrofia altera a curvatura Gaussiana do córtex, mudando como o
    calor se difunde. Em regiões com atrofia hipocampal, os sulcos se
    "abrem" e a HKS em tempos curtos diminui. Em tempos longos, a HKS
    muda porque o lobo encolheu globalmente.
    """)
    return


@app.cell
def _(heat_kernel_signature, lb, np):
    # ── Computar HKS com 16 escalas temporais ────────────────────────────
    hks = heat_kernel_signature(lb, n_scales=16)
    print(f"HKS: shape = {hks.shape} (vértices × escalas)")
    print(f"  min = {hks.min():.6f}, max = {hks.max():.6f}")

    # ── HKS com escalas customizadas ─────────────────────────────────────
    custom_times = np.logspace(-2, 3, 32)
    hks_custom = heat_kernel_signature(lb, time_scales=custom_times)
    print(f"HKS (custom): shape = {hks_custom.shape}")
    return custom_times, hks


@app.cell
def _(custom_times, hks, np, plt, surf):
    # ── Visualizar HKS em diferentes escalas temporais ───────────────────
    _fig, _axes = plt.subplots(2, 4, figsize=(18, 8))
    idx = np.random.RandomState(0).choice(surf.n_vertices, 25000, replace=False)
    for _i, (ax_row, scale_idx) in enumerate(zip(_axes.flat, [0, 2, 5, 8, 10, 12, 14, 15])):
        hks_scale = hks[:, scale_idx]
        _vmin, _vmax = np.percentile(hks_scale[idx], [2, 98])
        _sc = ax_row.scatter(surf.vertices[idx, 1], surf.vertices[idx, 2], c=hks_scale[idx], cmap='inferno', s=0.2, vmin=_vmin, vmax=_vmax)
        ax_row.set_title(f'Escala {scale_idx} (t ≈ {(custom_times[scale_idx] if scale_idx < len(custom_times) else '?')})')
        ax_row.set_aspect('equal')
        ax_row.set_axis_off()
    plt.suptitle('HKS — da curvatura local (esquerda) à forma global (direita)', fontsize=13)
    plt.tight_layout()
    plt.savefig('03_hks_multiscale.png', dpi=150, bbox_inches='tight')
    plt.show()
    return (idx,)


@app.cell
def _(hks, np, plt, surf):
    # ── Correlação entre HKS e overlays do FreeSurfer ─────────────────────
    thickness = surf.get_overlay('thickness')
    curvature = surf.get_overlay('curv')
    correlations = []
    for t_idx in range(hks.shape[1]):
        r_thick = np.corrcoef(hks[:, t_idx], thickness)[0, 1]
        r_curv = np.corrcoef(hks[:, t_idx], curvature)[0, 1]
        correlations.append((r_thick, r_curv))
    corr_arr = np.array(correlations)
    _fig, _ax = plt.subplots(figsize=(8, 4))
    _ax.plot(corr_arr[:, 0], 'o-', label='HKS × espessura', color='#3274A1')
    _ax.plot(corr_arr[:, 1], 's-', label='HKS × curvatura', color='#E1812C')
    _ax.set_xlabel('Índice da escala temporal')
    _ax.set_ylabel('Correlação de Pearson')
    _ax.set_title('HKS captura informação complementar à morfometria clássica')
    _ax.legend()
    _ax.grid(True, alpha=0.3)
    _ax.axhline(0, color='gray', ls=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig('03_hks_correlation.png', dpi=150, bbox_inches='tight')
    plt.show()
    return curvature, thickness


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.3 — Wave Kernel Signature (WKS)

    A WKS substitui o decaimento exponencial do calor por um filtro
    **log-normal em energia**:

    $$\text{WKS}(x, e) = \frac{\sum_{i=1}^{K} \exp\!\left(-\frac{(e - \log λ_i)^2}{2σ^2}\right) φ_i(x)^2}{C(e)}$$

    Vantagens sobre a HKS:
    - **Invariante a escala** (logaritmo nos autovalores)
    - **Melhor resolução em frequência** (filtro mais seletivo)
    - **Sensível a mudanças locais de girificação** (sulcos, displasia)

    Na prática, WKS e HKS são **complementares**: HKS captura melhor
    formas globais, WKS captura melhor padrões locais de dobramento.
    """)
    return


@app.cell
def _(lb, wave_kernel_signature):
    # ── Computar WKS ─────────────────────────────────────────────────────
    wks = wave_kernel_signature(lb, n_energies=16)
    print(f"WKS: shape = {wks.shape}")

    # ── Com parâmetros customizados ──────────────────────────────────────
    wks_fine = wave_kernel_signature(lb, n_energies=32, sigma=None)
    print(f"WKS (fino): shape = {wks_fine.shape}")
    return (wks,)


@app.cell
def _(idx, np, plt, surf, wks):
    _fig, _axes = plt.subplots(2, 4, figsize=(18, 8))
    for _i, _ax in enumerate(_axes.flat):
        energy_idx = _i * 2
        if energy_idx >= wks.shape[1]:
            break
        wks_e = wks[:, energy_idx]
        _vmin, _vmax = np.percentile(wks_e[idx], [2, 98])
        _sc = _ax.scatter(surf.vertices[idx, 1], surf.vertices[idx, 2], c=wks_e[idx], cmap='magma', s=0.2, vmin=_vmin, vmax=_vmax)
        _ax.set_title(f'Energia {energy_idx}')
        _ax.set_aspect('equal')
        _ax.set_axis_off()
    plt.suptitle('WKS — decomposição por frequência de girificação', fontsize=13)
    plt.tight_layout()
    plt.savefig('03_wks_multienergy.png', dpi=150, bbox_inches='tight')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.4 — Global Point Signature (GPS)

    O GPS é o mais simples dos descritores:

    $$\text{GPS}(x) = \left(\frac{φ_1(x)}{\sqrt{λ_1}}, \frac{φ_2(x)}{\sqrt{λ_2}}, \ldots, \frac{φ_K(x)}{\sqrt{λ_K}}\right)$$

    O GPS embute cada vértice em $\mathbb{R}^K$ de modo que a
    **distância euclidiana no embedding ≈ distância de Green** na
    superfície. Isso fornece um "sistema de coordenadas natural" que
    respeita a geometria intrínseca do córtex.

    Uso principal: seleção de inducing points para o SVGP (Tutorial 08).
    """)
    return


@app.cell
def _(curvature, global_point_signature, idx, lb, plt, thickness):
    # ── Computar GPS ─────────────────────────────────────────────────────
    gps = global_point_signature(lb, n_components=10)
    print(f'GPS: shape = {gps.shape}')
    _fig = plt.figure(figsize=(12, 5))
    # Visualizar as 3 primeiras componentes GPS como coordenadas
    ax1 = _fig.add_subplot(121, projection='3d')
    _sc = ax1.scatter(gps[idx, 0], gps[idx, 1], gps[idx, 2], c=thickness[idx], cmap='YlOrRd', s=0.5, vmin=1, vmax=4)
    ax1.set_title('GPS embedding (primeiras 3 componentes)\ncolorido por espessura')
    ax1.set_xlabel('GPS₁')
    ax1.set_ylabel('GPS₂')
    ax1.set_zlabel('GPS₃')
    plt.colorbar(_sc, ax=ax1, shrink=0.5)
    ax2 = _fig.add_subplot(122)
    ax2.scatter(gps[idx, 0], gps[idx, 1], c=curvature[idx], cmap='RdBu_r', s=0.3, vmin=-0.3, vmax=0.3)
    ax2.set_title('GPS₁ × GPS₂ — colorido por curvatura')
    ax2.set_xlabel('GPS₁')
    ax2.set_ylabel('GPS₂')
    ax2.set_aspect('equal')
    plt.tight_layout()
    plt.savefig('03_gps_embedding.png', dpi=150, bbox_inches='tight')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.5 — Matriz de Features Espectrais Combinada

    `spectral_feature_matrix()` concatena HKS, WKS e GPS em uma
    única matriz de features por vértice, ideal para clustering,
    classificação e construção de redes de similaridade.
    """)
    return


@app.cell
def _(lb, np, plt, spectral_feature_matrix):
    features = spectral_feature_matrix(lb, hks_scales=16, wks_energies=16, gps_components=10, include_hks=True, include_wks=True, include_gps=True)
    print(f'Feature matrix: {features.shape}  (vértices × dimensão)')
    print(f'  HKS: 16 colunas | WKS: 16 colunas | GPS: 10 colunas = 42 total')
    corr = np.corrcoef(features[:, :16].T)
    _fig, _ax = plt.subplots(figsize=(6, 5))
    im = _ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    _ax.set_title('Correlação entre escalas HKS')
    _ax.set_xlabel('Escala')
    _ax.set_ylabel('Escala')
    plt.colorbar(im, ax=_ax)
    plt.tight_layout()
    plt.savefig('03_hks_intercorrelation.png', dpi=150, bbox_inches='tight')
    # ── Correlação entre as colunas ──────────────────────────────────────
    plt.show()  # HKS inter-scale correlation
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.6 — Resumo dos Descritores

    | Descritor | Fórmula | Escalas | Captura | Uso principal |
    |-----------|---------|---------|---------|---------------|
    | **HKS** | $\sum_i e^{-λ_i t} φ_i^2$ | Tempo de difusão | Local → global | Anomaly detection, normative models |
    | **WKS** | $\sum_i \exp(-(e-\log λ_i)^2/2σ^2) φ_i^2$ | Energia (log-freq) | Girificação fina | Scale-invariant asymmetry |
    | **GPS** | $φ_i / \sqrt{λ_i}$ | Componentes | Coordenadas globais | FPS inducing points, embedding |
    | **Combined** | HKS ‖ WKS ‖ GPS | Todas | Multi-escala | Redes de similaridade, clustering |

    **Próximo**: Tutorial 04 — Pipeline sem FreeSurfer (T1w → superfície)
    """)
    return


if __name__ == "__main__":
    app.run()
