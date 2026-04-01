# CorticalFields — 10 Tutoriais Completos

Série de 10 tutoriais progressivos que documentam todas as funcionalidades
da biblioteca **CorticalFields** (v0.1.5), desde carregamento de superfícies
até modelagem normativa Bayesiana e figuras para publicação.

## Dados de Exemplo

Os tutoriais usam dados reais disponíveis no Zenodo:
- **6 T1w do CEPESC** (defaced/anonimizados)
- **6 pastas recon-all do DS000221** (OpenNeuro)

📦 **Zenodo**: https://zenodo.org/records/19365607

## Estrutura de Arquivos

```
corticalfields_tutorials/
├── jupyter/                     # Notebooks Jupyter (.ipynb)
│   ├── 01_installation_surface_io.ipynb
│   ├── 02_spectral_decomposition.ipynb
│   ├── 03_spectral_descriptors.ipynb
│   ├── 04_freesurfer_free_pipeline.ipynb
│   ├── 05_subcortical_surfaces.ipynb
│   ├── 06_functional_maps_asymmetry.ipynb
│   ├── 07_optimal_transport.ipynb
│   ├── 08_normative_modeling_surprise.ipynb
│   ├── 09_networks_statistics.ipynb
│   └── 10_publication_figures_bayesian.ipynb
├── marimo/                      # Notebooks Marimo (.py)
│   ├── 01_installation_surface_io_marimo.py
│   ├── ...
│   └── 10_publication_figures_bayesian_marimo.py
├── percent_scripts/             # Scripts percent-format (jupytext)
│   ├── 01_installation_surface_io.py
│   ├── ...
│   └── 10_publication_figures_bayesian.py
└── README.md
```

## Grade dos Tutoriais

| # | Tutorial | Módulos | Conceitos |
|---|----------|---------|-----------|
| 01 | **Instalação e Surface I/O** | `surface`, `utils` | CorticalSurface, overlays, GIfTI, parcelações |
| 02 | **Decomposição Espectral** | `spectral`, `backends` | Laplace–Beltrami, autovetores, lei de Weyl |
| 03 | **Descritores Espectrais** | `spectral` | HKS, WKS, GPS, feature matrix |
| 04 | **Pipeline sem FreeSurfer** | `pointcloud` | from_t1w, CorticalPointCloud, mesh LBO |
| 05 | **Superfícies Subcorticais** | `subcortical` | SubcorticalSurface, volume, esfericidade, Willmore |
| 06 | **Functional Maps** | `functional_maps`, `asymmetry` | C matrix, ZoomOut, assimetria multi-escala |
| 07 | **Optimal Transport** | `transport` | Sliced Wasserstein, Sinkhorn, kernel PD |
| 08 | **Modelagem Normativa** | `kernels`, `normative`, `surprise` | SVGP, Matérn espectral, surprise maps |
| 09 | **Redes e Estatística** | `graphs`, `eda_qc`, `distance_stats` | MSN/SSN, MDMR, HSIC, KRR |
| 10 | **Publicação e Bayesian** | `brainplots`, `bayesian`, `bayes_viz` | 4-view, Horseshoe, BEST, ROPE |

## Módulos Documentados (21 módulos, ~15.500 linhas)

| Módulo | Linhas | Descrição |
|--------|--------|-----------|
| `surface` | 369 | Surface I/O (FreeSurfer, GIfTI) |
| `spectral` | 516 | Laplace–Beltrami, HKS, WKS, GPS |
| `kernels` | 304 | Spectral Matérn kernel (GPyTorch) |
| `normative` | 686 | GP normative modeling (SVGP) |
| `surprise` | 381 | Information-theoretic anomaly scoring |
| `features` | 258 | Morphometric feature extraction |
| `pointcloud` | 1123 | FreeSurfer-free cortical extraction |
| `subcortical` | 1180 | Subcortical surface analysis |
| `functional_maps` | 1008 | Functional maps & ZoomOut |
| `transport` | 775 | Optimal transport distances |
| `asymmetry` | 523 | Atlas-free asymmetry metrics |
| `graphs` | 234 | Brain network construction |
| `distance_stats` | 894 | MDMR, HSIC, Dcor, KRR |
| `eda_qc` | 1265 | EDA, QC, outlier detection |
| `bayesian` | 1294 | 10 Bayesian statistical models |
| `bayes_viz` | 959 | Bayesian visualization |
| `brainplots` | 2170 | Publication-quality brain figures |
| `viz` | 357 | Basic visualization |
| `backends` | 753 | GPU/CPU backend selection |
| `utils` | 333 | Utilities (timing, validation) |
| `__init__` | 144 | Lazy imports |

## Como Usar

### Jupyter
```bash
cd jupyter/
jupyter notebook 01_installation_surface_io.ipynb
```

### Marimo
```bash
cd marimo/
marimo edit 01_installation_surface_io_marimo.py
```

### Jupytext (sync)
```bash
cd percent_scripts/
jupytext --sync 01_installation_surface_io.py
```

## Dependências

```bash
# Core (leve)
pip install corticalfields numpy scipy nibabel matplotlib

# GPU (pesado — opcional)
pip install torch gpytorch

# FreeSurfer-free pipeline
pip install deepbet antspynet scikit-image robust-laplacian

# Optimal transport
pip install POT geomloss pykeops

# Bayesian
pip install pymc arviz preliz

# Visualization
pip install pyvista

# Notebook
pip install jupytext marimo
```

## Autor

rdneuro — CorticalFields v0.1.5
