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
    # Tutorial 09 — Redes Cerebrais, EDA/QC e Inferência Estatística

    Neste tutorial, cobriremos três módulos interconectados:

    **Parte A — Redes de similaridade cerebral** (`graphs`)
    1. MSN (Morphometric Similarity Networks)
    2. SSN (Spectral Similarity Networks)
    3. Métricas grafos-teóricas (clustering, eficiência, modularidade)

    **Parte B — EDA e Quality Control** (`eda_qc`)
    4. `run_clinical_eda()` — estatísticas descritivas + outliers
    5. `run_spectral_eda()` — QC espectral (Weyl, Euler, PCoA)
    6. Detecção de outliers (MAD, IQR, MCD-Mahalanobis)

    **Parte C — Inferência estatística** (`distance_stats`)
    7. MDMR — Regressão em matrizes de distância
    8. HSIC — Independência entre kernels
    9. Distance correlation
    10. KRR — Kernel Ridge Regression com CV

    ---

    # Parte A — Redes de Similaridade Cerebral
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import logging
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

    import corticalfields as cf
    from corticalfields.spectral import compute_eigenpairs, spectral_feature_matrix
    from corticalfields.features import extract_cohort_profiles, MorphometricProfile
    from corticalfields.graphs import (
        morphometric_similarity_network,
        spectral_similarity_network,
        graph_metrics,
    )

    SUBJECTS_DIR = "./data/freesurfer"
    SUBJECT_ID = "sub-010002"

    surf = cf.load_freesurfer_surface(SUBJECTS_DIR, SUBJECT_ID, "lh", "pial",
                                       overlays=["thickness", "curv", "sulc", "area", "volume"])
    return (
        SUBJECTS_DIR,
        SUBJECT_ID,
        compute_eigenpairs,
        graph_metrics,
        morphometric_similarity_network,
        np,
        plt,
        spectral_feature_matrix,
        spectral_similarity_network,
        surf,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A.1 — Morphometric Similarity Networks (Seidlitz et al., 2018)

    Cada ROI tem um perfil morfométrico (média de espessura, curvatura,
    sulc, área, volume). A correlação entre perfis de ROIs dá a "similaridade
    morfométrica" — ROIs com perfis parecidos estão conectadas.
    """)
    return


@app.cell
def _(
    SUBJECTS_DIR,
    SUBJECT_ID,
    graph_metrics,
    morphometric_similarity_network,
    np,
    surf,
):
    from corticalfields.surface import load_annot

    labels, names = load_annot(SUBJECTS_DIR, SUBJECT_ID, "lh", "aparc")

    # Construir matriz de features por vértice
    feature_matrix = np.column_stack([
        surf.get_overlay("thickness"),
        surf.get_overlay("curv"),
        surf.get_overlay("sulc"),
    ])
    print(f"Feature matrix: {feature_matrix.shape}")

    # ── MSN ──────────────────────────────────────────────────────────────
    msn = morphometric_similarity_network(
        feature_matrix, labels,
        method="pearson",
        fisher_z=True,
    )
    print(f"MSN: {msn.shape} ({len(np.unique(labels[labels > 0]))} ROIs)")

    # ── Métricas de grafo ────────────────────────────────────────────────
    metrics = graph_metrics(msn, density=0.15)
    print(f"\nGraph metrics (density=15%):")
    print(f"  Nodes: {metrics['n_nodes']}, Edges: {metrics['n_edges']}")
    print(f"  Global efficiency: {metrics['global_efficiency']:.4f}")
    print(f"  Modularity: {metrics['modularity']:.4f}")
    print(f"  Assortativity: {metrics['assortativity']:.4f}")
    return labels, msn


@app.cell
def _(
    compute_eigenpairs,
    labels,
    msn,
    plt,
    spectral_feature_matrix,
    spectral_similarity_network,
    surf,
):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    im1 = axes[0].imshow(msn, cmap="RdBu_r", vmin=-2, vmax=2)
    axes[0].set_title("MSN (Fisher-z)")
    axes[0].set_xlabel("ROI"); axes[0].set_ylabel("ROI")
    plt.colorbar(im1, ax=axes[0])

    # Spectral similarity
    lb = compute_eigenpairs(surf.vertices, surf.faces, n_eigenpairs=300)
    spec_features = spectral_feature_matrix(lb, hks_scales=16, wks_energies=16, gps_components=10)
    ssn = spectral_similarity_network(spec_features, labels, metric="cosine")

    im2 = axes[1].imshow(ssn, cmap="Blues", vmin=0, vmax=1)
    axes[1].set_title("SSN (cosine similarity)")
    axes[1].set_xlabel("ROI"); axes[1].set_ylabel("ROI")
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.savefig("09_brain_networks.png", dpi=150, bbox_inches="tight")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Parte B — EDA e Quality Control
    """)
    return


@app.cell
def _():
    from corticalfields.eda_qc import (
        run_clinical_eda,
        run_spectral_eda,
        detect_clinical_outliers,
        mcd_mahalanobis_outliers,
        distance_matrix_outliers,
        QCReport,
        EDAResult,
    )

    return (run_clinical_eda,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## B.1 — EDA Clínica

    `run_clinical_eda()` computa estatísticas descritivas e detecta outliers
    em dados morfométricos de toda a coorte.
    """)
    return


@app.cell
def _(np, run_clinical_eda, surf):
    # Simular dados de coorte (6 sujeitos)
    np.random.seed(42)
    n_subs = 6
    cohort_thickness = np.column_stack([
        surf.get_overlay("thickness") + np.random.randn(surf.n_vertices) * 0.1
        for _ in range(n_subs)
    ])

    eda = run_clinical_eda(
        data=cohort_thickness,
        feature_name="thickness",
        subject_ids=[f"sub-{i:03d}" for i in range(n_subs)],
        method="mad",          # 'mad', 'iqr', ou 'mcd'
        threshold=3.0,         # nº de MADs para outlier
    )

    print(f"EDA Result:")
    print(f"  Sujeitos: {eda.n_subjects}")
    print(f"  Outliers detectados: {eda.n_outliers}")
    print(f"  Mean ± std: {eda.global_mean:.3f} ± {eda.global_std:.3f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## B.2 — EDA Espectral (Weyl, Euler)
    """)
    return


@app.cell
def _():
    # eda_spectral = run_spectral_eda(
    #     eigenvalues_list=[lb.eigenvalues],
    #     surface_areas=[surf.total_area],
    #     subject_ids=[SUBJECT_ID],
    # )
    # print(f"Weyl law deviation: {eda_spectral.weyl_deviation:.4f}")
    print("run_spectral_eda() verifica conformidade com lei de Weyl e Euler characteristic")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Parte C — Inferência Estatística em Matrizes de Distância
    """)
    return


@app.cell
def _():
    from corticalfields.distance_stats import (
        StatisticalResult,
        mdmr,
        hsic,
        distance_correlation,
        mantel_test,
        kernel_ridge_regression,
        outcome_kernel,
    )

    return (
        distance_correlation,
        hsic,
        kernel_ridge_regression,
        mdmr,
        outcome_kernel,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## C.1 — MDMR (Multivariate Distance Matrix Regression)

    Testa: "A matriz de distâncias geométricas está associada a
    preditores clínicos (idade, HADS, lateralidade)?"

    É o equivalente de MANOVA para dados de distância.
    """)
    return


@app.cell
def _(mdmr, np):
    # Criar dados sintéticos
    n = 12
    D = np.random.rand(n, n)
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0)

    # Preditores: idade e HADS-A
    age = np.random.randn(n)
    hads = np.random.randn(n)
    X = np.column_stack([age, hads])

    result_mdmr = mdmr(
        D=D,
        X=X,
        n_permutations=999,
        variable_names=["age", "HADS-A"],
    )

    print(f"MDMR Result:")
    print(f"  Omnibus: F={result_mdmr.statistic:.4f}, p={result_mdmr.p_value:.4f}")
    print(f"  Per-variable:")
    for name, pval in result_mdmr.per_variable_p.items():
        print(f"    {name}: p = {pval:.4f}")
    return D, hads


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## C.2 — HSIC (Hilbert-Schmidt Independence Criterion)
    """)
    return


@app.cell
def _(D, hads, hsic, np, outcome_kernel):
    K1 = np.exp(-D**2 / (2 * np.median(D)**2))  # kernel geométrico
    K2 = outcome_kernel(hads)                     # kernel de desfecho

    result_hsic = hsic(
        K1=K1, K2=K2,
        n_permutations=999,
        method="biased",
    )
    print(f"\nHSIC: stat={result_hsic.statistic:.6f}, p={result_hsic.p_value:.4f}")
    return (K1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## C.3 — Distance Correlation (Székely et al.)
    """)
    return


@app.cell
def _(D, distance_correlation, hads, np):
    result_dcor = distance_correlation(
        X=D,
        Y=np.abs(np.subtract.outer(hads, hads)),
        n_permutations=999,
    )
    print(f"Distance correlation: r={result_dcor.statistic:.4f}, p={result_dcor.p_value:.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## C.4 — Kernel Ridge Regression (KRR)

    Prediz desfechos clínicos contínuos (HADS, idade) a partir de
    kernels geométricos com cross-validation:
    """)
    return


@app.cell
def _(K1, hads, kernel_ridge_regression, np):
    result_krr = kernel_ridge_regression(
        K=K1,
        y=hads,
        alphas=np.logspace(-3, 3, 20),  # grid de regularização
        cv=3,                            # 3-fold CV (n pequeno)
    )
    print(f"\nKRR Result:")
    print(f"  Best α = {result_krr.best_alpha:.4f}")
    print(f"  CV R² = {result_krr.cv_r2:.4f}")
    print(f"  CV MAE = {result_krr.cv_mae:.4f}")
    print(f"  Permutation p = {result_krr.p_value:.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9.5 — Resumo

    | Módulo | Função | O que faz |
    |--------|--------|-----------|
    | `graphs` | `morphometric_similarity_network()` | MSN de features clássicas |
    | `graphs` | `spectral_similarity_network()` | SSN de features espectrais |
    | `graphs` | `graph_metrics()` | Métricas de grafo (NetworkX/igraph) |
    | `eda_qc` | `run_clinical_eda()` | Estatísticas + outlier detection |
    | `eda_qc` | `run_spectral_eda()` | QC espectral (Weyl, Euler) |
    | `eda_qc` | `mcd_mahalanobis_outliers()` | Outliers multivariados (MCD) |
    | `distance_stats` | `mdmr()` | MANOVA para distâncias |
    | `distance_stats` | `hsic()` | Independência entre kernels |
    | `distance_stats` | `distance_correlation()` | Correlação de distâncias |
    | `distance_stats` | `kernel_ridge_regression()` | KRR com CV e permutation |

    **Próximo**: Tutorial 10 — Figuras para publicação e modelos Bayesianos
    """)
    return


if __name__ == "__main__":
    app.run()
