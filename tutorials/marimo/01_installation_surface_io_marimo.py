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
    # Tutorial 01 — Instalação, Dados e Superfícies Corticais

    **CorticalFields** é uma biblioteca Python para análise espectral de
    forma (*spectral shape analysis*) em superfícies corticais e
    subcorticais. Foi projetada para dados de RM estrutural (T1w) em
    neuroimagem clínica, com ênfase em epilepsia (MTLE-HS).

    Neste primeiro tutorial você aprenderá:

    1. Como instalar a CorticalFields e suas dependências
    2. Como baixar os dados de exemplo do Zenodo
    3. Como carregar superfícies do FreeSurfer (`CorticalSurface`)
    4. Como inspecionar a malha triangular e seus overlays
    5. Como salvar e exportar superfícies em formato GIfTI

    ---

    ## 1.1 — Instalação

    A CorticalFields pode ser instalada diretamente do repositório.
    Recomendamos criar um ambiente conda dedicado:

    ```bash
    conda create -n cf python=3.11
    conda activate cf
    pip install corticalfields
    ```

    As dependências são divididas em **leves** (numpy, scipy, nibabel) e
    **pesadas** (torch, gpytorch — necessárias apenas para kernels/GP).
    O `import corticalfields` é sempre rápido graças aos *lazy imports*:
    os módulos pesados só são importados quando usados pela primeira vez.
    """)
    return


@app.cell
def _():
    # Importações básicas
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    import logging

    # Configurar logging para ver o que a CF faz internamente
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)s — %(levelname)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    import corticalfields as cf

    print(f"CorticalFields v{cf.__version__}")
    print(f"Autor: {cf.__author__}")
    return Path, cf, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.2 — Baixando os dados de exemplo

    Preparamos um repositório no Zenodo com dados reais para os
    tutoriais:

    - **6 T1w do CEPESC** — imagens defaced (anonimizadas) de pacientes e
      controles do Centro de Epilepsia de Santa Catarina
    - **6 pastas recon-all do DS000221** — saídas completas do FreeSurfer
      a partir do dataset público OpenNeuro DS000221

    Cada pasta de recon-all foi comprimida em `.tar.gz` individual.

    **Link do repositório Zenodo:**
    https://zenodo.org/records/19365607

    Para baixar automaticamente:
    """)
    return


@app.cell
def _(Path):
    import os, tarfile, urllib.request

    # ── Configuração de diretórios ────────────────────────────────────────
    DATA_DIR   = Path("./data")
    FS_DIR     = DATA_DIR / "freesurfer"   # $SUBJECTS_DIR
    T1W_DIR    = DATA_DIR / "t1w"          # T1w NIfTI files
    DATA_DIR.mkdir(exist_ok=True)
    FS_DIR.mkdir(exist_ok=True)
    T1W_DIR.mkdir(exist_ok=True)

    # ── URL base do Zenodo (substitua pelo link permanente após publicação)
    ZENODO_RECORD = "https://zenodo.org/records/19365607"
    ZENODO_TOKEN  = (
        "eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjcwZDYyZDFjLTQyZDUtNGU1ZC1iMGI3LWU4"
        "YWY5M2QxNWMwNiIsImRhdGEiOnt9LCJyYW5kb20iOiIxZjdjNjUwMDA0ODQ1NWFiNW"
        "JiNWNjNDk0ZjUzM2ZlNiJ9.IrYKhP-cRgVLENLW5iPoUG_bBfGmFQ_W9HH-AeO31d"
        "Dct-3fG973MJyM9GHfKvfbcrWPGtN6Xk6CeTt9VJua1g"
    )

    # IDs dos sujeitos do DS000221 (recon-all)
    DS000221_SUBJECTS = [
        "sub-010002", "sub-010004", "sub-010006",
        "sub-010008", "sub-010010", "sub-010012",
    ]

    # IDs dos sujeitos do CEPESC (T1w .nii.gz)
    CEPESC_SUBJECTS = [
        "sub-cepesc01", "sub-cepesc02", "sub-cepesc03",
        "sub-cepesc04", "sub-cepesc05", "sub-cepesc06",
    ]

    def download_zenodo_file(filename: str, dest_dir: Path) -> Path:
        """Baixa um arquivo do Zenodo com token de preview."""
        dest = dest_dir / filename
        if dest.exists():
            print(f"  ✓ {filename} já existe, pulando download.")
            return dest
        url = f"{ZENODO_RECORD}/files/{filename}?token={ZENODO_TOKEN}"
        print(f"  ⬇ Baixando {filename}...")
        urllib.request.urlretrieve(url, str(dest))
        print(f"  ✓ {filename} salvo em {dest}")
        return dest

    def download_and_extract_subject(subject_id: str):
        """Baixa e extrai um .tar.gz de recon-all do Zenodo."""
        tarname = f"{subject_id}.tar.gz"
        dest = FS_DIR / subject_id
        if dest.exists():
            print(f"  ✓ {subject_id} já extraído.")
            return
        tarpath = download_zenodo_file(tarname, DATA_DIR)
        print(f"  📦 Extraindo {tarname}...")
        with tarfile.open(str(tarpath), "r:gz") as tar:
            tar.extractall(path=str(FS_DIR))
        print(f"  ✓ {subject_id} extraído em {dest}")

    # ── Baixar um sujeito para este tutorial ──────────────────────────────
    # (Descomente as linhas abaixo para baixar de fato)
    #
    # print("Baixando dados do FreeSurfer (DS000221)...")
    # download_and_extract_subject(DS000221_SUBJECTS[0])
    #
    # print("\nBaixando T1w do CEPESC...")
    # for sid in CEPESC_SUBJECTS[:2]:
    #     download_zenodo_file(f"{sid}_T1w.nii.gz", T1W_DIR)

    print("Configure as variáveis acima e descomente para baixar os dados.")
    return DATA_DIR, DS000221_SUBJECTS, FS_DIR


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.3 — A classe `CorticalSurface`

    `CorticalSurface` é a **estrutura de dados central** da
    CorticalFields. Ela armazena:

    | Atributo    | Tipo                   | Descrição |
    |-------------|------------------------|-----------|
    | `vertices`  | `ndarray (N, 3)`       | Coordenadas 3D dos vértices em espaço RAS (mm) |
    | `faces`     | `ndarray (F, 3)`       | Conectividade triangular (0-indexada) |
    | `hemi`      | `str`                  | Hemisfério: `'lh'` ou `'rh'` |
    | `overlays`  | `dict[str, ndarray]`   | Mapas escalares por vértice (espessura, curvatura, etc.) |
    | `metadata`  | `dict`                 | Metadados arbitrários |

    Todos os módulos downstream (espectral, GP, surprise) operam sobre
    instâncias de `CorticalSurface`.
    """)
    return


@app.cell
def _(DS000221_SUBJECTS, FS_DIR, cf):
    # ── Carregar uma superfície do FreeSurfer ─────────────────────────────
    # Ajuste SUBJECTS_DIR e SUBJECT_ID para os seus dados:
    SUBJECTS_DIR = str(FS_DIR)
    SUBJECT_ID   = DS000221_SUBJECTS[0]  # "sub-010002"

    # load_freesurfer_surface carrega a malha + overlays automaticamente
    surf_lh = cf.load_freesurfer_surface(
        subjects_dir=SUBJECTS_DIR,
        subject_id=SUBJECT_ID,
        hemi="lh",
        surface="pial",                          # 'pial', 'white', 'inflated'
        overlays=["thickness", "curv", "sulc"],  # None = carrega todos
    )

    print(f"Superfície: {surf_lh.hemi}.pial")
    print(f"  Vértices:   {surf_lh.n_vertices:,}")
    print(f"  Faces:      {surf_lh.n_faces:,}")
    print(f"  Overlays:   {surf_lh.overlay_names}")
    print(f"  Área total: {surf_lh.total_area:.1f} mm²")
    print(f"  Metadata:   {surf_lh.metadata}")
    return SUBJECTS_DIR, SUBJECT_ID, surf_lh


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### O que cada overlay representa?

    | Overlay       | Descrição |
    |---------------|-----------|
    | `thickness`   | Espessura cortical (mm) — distância entre superfície pial e white |
    | `curv`        | Curvatura média — positiva em sulcos, negativa em giros |
    | `sulc`        | Profundidade sulcal (mm) — quão profundo está o vértice |
    | `area`        | Área de Voronoi por vértice (mm²) |
    | `volume`      | Volume cortical local (mm³) |
    | `pial_lgi`    | Índice de girificação local (lGI) |
    """)
    return


@app.cell
def _(surf_lh):
    # ── Inspecionar overlays ──────────────────────────────────────────────
    thickness = surf_lh.get_overlay("thickness")
    curvature = surf_lh.get_overlay("curv")
    sulc      = surf_lh.get_overlay("sulc")

    print("Espessura cortical:")
    print(f"  min  = {thickness.min():.3f} mm")
    print(f"  max  = {thickness.max():.3f} mm")
    print(f"  mean = {thickness.mean():.3f} mm")
    print(f"  std  = {thickness.std():.3f} mm")

    print("\nCurvatura:")
    print(f"  min  = {curvature.min():.4f}")
    print(f"  max  = {curvature.max():.4f}")
    print(f"  mean = {curvature.mean():.4f}")
    return curvature, sulc, thickness


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.4 — Explorando a malha triangular

    A malha cortical é um **triangulated 2-manifold**: uma superfície
    fechada composta de triângulos onde cada aresta é compartilhada por
    exatamente dois triângulos.

    Propriedades topológicas importantes:
    - **Euler-Poincaré**: χ = V − E + F = 2 para uma esfera (genus 0)
    - Cada hemisfério do FreeSurfer é topologicamente equivalente a uma esfera
    """)
    return


@app.cell
def _(surf_lh):
    # ── Propriedades geométricas da malha ─────────────────────────────────
    from corticalfields.utils import validate_mesh

    report = validate_mesh(surf_lh.vertices, surf_lh.faces)
    for k, v in report.items():
        print(f"  {k}: {v}")
    return


@app.cell
def _(np, surf_lh):
    # ── Normais por vértice ───────────────────────────────────────────────
    normals = surf_lh.vertex_normals
    print(f"Normais: shape = {normals.shape}")
    print(f"  São unitárias? max |n| = {np.linalg.norm(normals, axis=1).max():.6f}")

    # ── Áreas das faces ──────────────────────────────────────────────────
    face_areas = surf_lh.face_areas
    print(f"\nÁreas das faces:")
    print(f"  min  = {face_areas.min():.4f} mm²")
    print(f"  mean = {face_areas.mean():.4f} mm²")
    print(f"  max  = {face_areas.max():.4f} mm²")

    # ── Adjacência ───────────────────────────────────────────────────────
    adj = surf_lh.vertex_adjacency()
    n_neighbors = [len(a) for a in adj]
    print(f"\nGrau dos vértices (nº de vizinhos):")
    print(f"  min  = {min(n_neighbors)}")
    print(f"  mean = {np.mean(n_neighbors):.1f}")
    print(f"  max  = {max(n_neighbors)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.5 — Visualização básica da superfície

    Vamos plotar as distribuições dos overlays e uma visualização 3D
    simples com matplotlib.
    """)
    return


@app.cell
def _(curvature, plt, sulc, thickness):
    _fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(thickness, bins=80, color='#3274A1', edgecolor='white', linewidth=0.3)
    # Histograma de espessura cortical
    axes[0].set_xlabel('Espessura cortical (mm)')
    axes[0].set_ylabel('Nº de vértices')
    axes[0].set_title('Distribuição da espessura cortical')
    axes[0].axvline(thickness.mean(), color='red', ls='--', label=f'μ = {thickness.mean():.2f}')
    axes[0].legend()
    axes[1].hist(curvature, bins=80, color='#E1812C', edgecolor='white', linewidth=0.3)
    axes[1].set_xlabel('Curvatura média')
    # Histograma de curvatura
    axes[1].set_title('Distribuição da curvatura')
    axes[1].axvline(0, color='black', ls=':', alpha=0.5)
    axes[2].hist(sulc, bins=80, color='#3A923A', edgecolor='white', linewidth=0.3)
    axes[2].set_xlabel('Profundidade sulcal (mm)')
    axes[2].set_title('Distribuição da profundidade sulcal')
    # Histograma de profundidade sulcal
    plt.tight_layout()
    plt.savefig('01_overlay_distributions.png', dpi=150, bbox_inches='tight')
    plt.show()
    return


@app.cell
def _(curvature, np, plt, surf_lh, thickness):
    # ── Scatter 3D colorido por espessura ─────────────────────────────────
    from mpl_toolkits.mplot3d import Axes3D
    _fig = plt.figure(figsize=(12, 5))
    ax1 = _fig.add_subplot(121, projection='3d')
    idx = np.random.RandomState(42).choice(surf_lh.n_vertices, 20000, replace=False)
    # Vista lateral
    sc = ax1.scatter(surf_lh.vertices[idx, 0], surf_lh.vertices[idx, 1], surf_lh.vertices[idx, 2], c=thickness[idx], cmap='YlOrRd', s=0.3, vmin=1, vmax=4)
    # Subsample para performance
    ax1.set_title('Vista lateral — Espessura cortical')
    ax1.view_init(elev=0, azim=-90)
    ax1.set_axis_off()
    plt.colorbar(sc, ax=ax1, label='mm', shrink=0.6)
    ax2 = _fig.add_subplot(122, projection='3d')
    sc2 = ax2.scatter(surf_lh.vertices[idx, 0], surf_lh.vertices[idx, 1], surf_lh.vertices[idx, 2], c=curvature[idx], cmap='RdBu_r', s=0.3, vmin=-0.5, vmax=0.5)
    ax2.set_title('Vista medial — Curvatura')
    ax2.view_init(elev=0, azim=90)
    ax2.set_axis_off()
    plt.colorbar(sc2, ax=ax2, label='curv', shrink=0.6)
    plt.tight_layout()
    plt.savefig('01_surface_scatter.png', dpi=150, bbox_inches='tight')
    # Vista medial
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.6 — Adicionando overlays customizados

    Você pode criar e anexar qualquer mapa escalar por vértice à
    superfície. Isso é útil para resultados de análises que precisam
    ser mapeados de volta ao cérebro.
    """)
    return


@app.cell
def _(np, surf_lh):
    # Exemplo: criar um overlay de "assimetria local de espessura"
    # (apenas placeholder — assimetria real requer ambos hemisférios)
    random_overlay = np.random.randn(surf_lh.n_vertices) * 0.5
    surf_lh.add_overlay("random_noise", random_overlay)

    print(f"Overlays atuais: {surf_lh.overlay_names}")

    # Recuperar o overlay
    noise = surf_lh.get_overlay("random_noise")
    print(f"Shape: {noise.shape}, dtype: {noise.dtype}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.7 — Carregando parcelas (annotations)

    O FreeSurfer produz arquivos de parcelação (`.annot`) que dividem
    o córtex em regiões anatômicas. As mais comuns são:

    - **aparc** (Desikan-Killiany) — 34 regiões por hemisfério
    - **aparc.a2009s** (Destrieux) — 74 regiões por hemisfério
    """)
    return


@app.cell
def _(SUBJECTS_DIR, SUBJECT_ID, np, thickness):
    from corticalfields.surface import load_annot

    labels, names = load_annot(
        subjects_dir=SUBJECTS_DIR,
        subject_id=SUBJECT_ID,
        hemi="lh",
        annot="aparc",  # Desikan-Killiany atlas
    )

    print(f"Labels: shape = {labels.shape}, unique = {len(np.unique(labels))}")
    print(f"Regiões (primeiras 10): {names[:10]}")

    # Espessura média por região
    for i, name in enumerate(names[:5]):
        mask = labels == i
        if mask.sum() > 0:
            mean_thick = thickness[mask].mean()
            print(f"  {name}: {mean_thick:.3f} mm ({mask.sum()} vértices)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.8 — Exportando superfícies para GIfTI

    O formato GIfTI (`.gii`) é o padrão do NIfTI Working Group para
    dados de superfície. É o formato preferido pelo connectome-workbench
    e outras ferramentas modernas.
    """)
    return


@app.cell
def _(DATA_DIR, surf_lh):
    # Salvar como GIfTI (com todos os overlays)
    output_path = DATA_DIR / "exported_surface.surf.gii"
    surf_lh.to_gifti(output_path)
    print(f"Superfície exportada para: {output_path}")

    # Re-carregar para verificar
    from corticalfields.surface import load_gifti_surface
    surf_reload = load_gifti_surface(output_path, hemi="lh")
    print(f"Re-carregada: {surf_reload.n_vertices} vértices, overlays: {surf_reload.overlay_names}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.9 — Carregando ambos os hemisférios

    Para análises de assimetria (Tutoriais 06–07), precisamos carregar
    ambos os hemisférios:
    """)
    return


@app.cell
def _(SUBJECTS_DIR, SUBJECT_ID, cf, surf_lh):
    # Hemisfério direito
    surf_rh = cf.load_freesurfer_surface(
        subjects_dir=SUBJECTS_DIR,
        subject_id=SUBJECT_ID,
        hemi="rh",
        surface="pial",
        overlays=["thickness", "curv", "sulc"],
    )

    print(f"LH: {surf_lh.n_vertices:,} vértices, {surf_lh.total_area:.0f} mm²")
    print(f"RH: {surf_rh.n_vertices:,} vértices, {surf_rh.total_area:.0f} mm²")
    print(f"Assimetria de área: {(surf_lh.total_area - surf_rh.total_area) / surf_rh.total_area * 100:.2f}%")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.10 — Resumo

    Neste tutorial, aprendemos:

    | Função / Classe | Módulo | O que faz |
    |----------------|--------|-----------|
    | `CorticalSurface` | `surface` | Estrutura de dados central — malha + overlays |
    | `load_freesurfer_surface()` | `surface` | Carrega superfícies do FreeSurfer |
    | `load_gifti_surface()` | `surface` | Carrega superfícies GIfTI |
    | `load_annot()` | `surface` | Carrega parcelações (aparc, Destrieux) |
    | `validate_mesh()` | `utils` | Valida topologia da malha |
    | `.to_gifti()` | `CorticalSurface` | Exporta para formato GIfTI |
    | `.vertex_normals` | `CorticalSurface` | Normais unitárias por vértice |
    | `.face_areas` | `CorticalSurface` | Área de cada triângulo |
    | `.vertex_adjacency()` | `CorticalSurface` | Lista de vizinhos por vértice |

    No próximo tutorial, usaremos essa superfície para computar o
    **operador de Laplace–Beltrami** — o coração matemático da CF.
    """)
    return


if __name__ == "__main__":
    app.run()
