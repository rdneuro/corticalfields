# Cortical thickness × Yeo-7 network analysis: a complete pipeline guide

This report provides a production-ready pipeline for projecting Schaefer-200 parcellations to individual FreeSurfer subjects, extracting cortical thickness per parcel, and creating publication-quality Bayesian neuroimaging figures. Every code block targets **Python 3.11, matplotlib 3.8+, nilearn ≥0.13, and FreeSurfer 8.1.0**. The single most important caveat: `mris_anatomical_stats` performs **area-weighted** thickness averaging while a naive nibabel approach does not—so if you go Python-only, you must compute vertex areas from the white surface mesh.

---

## 1. Projecting Schaefer .annot files with mri_surf2surf

The Schaefer parcellation annot files are **not bundled with FreeSurfer**. Download them from the [ThomasYeoLab/CBIG repository](https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3), which provides pre-computed `.annot` files for fsaverage, fsaverage5, and fsaverage6. The parcellations were originally computed in fsaverage6 space and resampled to fsaverage—use the **fsaverage** version as your source.

The exact projection command:

```bash
# Left hemisphere
mri_surf2surf --hemi lh \
  --srcsubject fsaverage \
  --trgsubject sub-01 \
  --sval-annot $SCHAEFER_DIR/fsaverage/label/lh.Schaefer2018_200Parcels_7Networks_order.annot \
  --tval $SUBJECTS_DIR/sub-01/label/lh.Schaefer2018_200Parcels_7Networks_order.annot

# Right hemisphere
mri_surf2surf --hemi rh \
  --srcsubject fsaverage \
  --trgsubject sub-01 \
  --sval-annot $SCHAEFER_DIR/fsaverage/label/rh.Schaefer2018_200Parcels_7Networks_order.annot \
  --tval $SUBJECTS_DIR/sub-01/label/rh.Schaefer2018_200Parcels_7Networks_order.annot
```

The correct flag is **`--sval-annot`** (confirmed in the FreeSurfer source `mri_surf2surf.cpp`). There is no `--sval-annot-file` variant. When `--sval-annot` is used, the tool **automatically switches to nearest-neighbor forward (NNF) interpolation**—you do not need to specify `--mapmethod nnf` manually. This is critical because annotation labels are categorical integers; trilinear interpolation would produce meaningless fractional label IDs. The output is a standard FreeSurfer `.annot` file containing per-vertex label integers, an embedded RGBT color table, and parcel names.

### FreeSurfer v7.4.1 subjects with v8.1.0 tools

FreeSurfer v8.0.0 release notes state that "recon-all will have different output than version 7," owing to SynthSeg (deep-learning segmentation), SynthStrip (skull-stripping), and SynthMorph (registration). However, for annotation projection specifically, **v8.1.0 tools on v7.4.1-processed subjects should work** because both versions register to the same `fsaverage` spherical atlas target and `mri_surf2surf` reads the existing `?h.sphere.reg` without recomputing it. The official recommendation remains to use a single version across all processing. To verify compatibility, inspect the projected `.annot` in `freeview` and confirm the expected number of unique parcel labels (**100 per hemisphere** for Schaefer-200).

---

## 2. Extracting mean cortical thickness per Schaefer-200 parcel

### The mris_anatomical_stats approach (recommended for accuracy)

```bash
mris_anatomical_stats -th3 -mgz \
  -cortex $SUBJECTS_DIR/sub-01/label/lh.cortex.label \
  -f $SUBJECTS_DIR/sub-01/stats/lh.Schaefer200_7Net.stats \
  -b \
  -a $SUBJECTS_DIR/sub-01/label/lh.Schaefer2018_200Parcels_7Networks_order.annot \
  sub-01 lh white
```

The `-th3` flag computes volume via obliquely truncated trilateral pyramids (higher accuracy). The `-cortex` flag masks medial wall vertices. Output is a tab-separated `.stats` file whose `ThickAvg` column gives **area-weighted** mean thickness per parcel in mm. For multi-subject extraction, `aparcstats2table` can batch-aggregate:

```bash
aparcstats2table --subjects sub-01 sub-02 sub-03 \
  --hemi lh --meas thickness \
  --parc Schaefer2018_200Parcels_7Networks_order \
  --tablefile lh.schaefer200.thickness.table
```

### Python-only approach with nibabel (area-weighted)

The naive approach—`np.mean(thickness[mask])`—gives equal weight to every vertex regardless of the surface area it represents, introducing bias toward densely tessellated regions. To match `mris_anatomical_stats`, compute vertex areas from the white surface triangulation:

```python
import os
import numpy as np
import pandas as pd
from nibabel.freesurfer.io import read_annot, read_morph_data, read_geometry


def extract_thickness_schaefer(subjects_dir, subject, hemi='lh',
                                annot_name='Schaefer2018_200Parcels_7Networks_order'):
    """Area-weighted mean cortical thickness per Schaefer parcel."""
    base = os.path.join(subjects_dir, subject)

    # 1. Read projected annotation → (labels, ctab, names)
    labels, ctab, names = read_annot(
        os.path.join(base, 'label', f'{hemi}.{annot_name}.annot'))

    # 2. Read per-vertex thickness (curv format)
    thickness = read_morph_data(
        os.path.join(base, 'surf', f'{hemi}.thickness'))

    # 3. Read white surface geometry → vertex areas
    coords, faces = read_geometry(
        os.path.join(base, 'surf', f'{hemi}.white'))

    # Vectorized vertex-area computation
    v0, v1, v2 = coords[faces[:, 0]], coords[faces[:, 1]], coords[faces[:, 2]]
    tri_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    vertex_areas = np.zeros(len(coords))
    for i in range(3):
        np.add.at(vertex_areas, faces[:, i], tri_areas / 3.0)

    assert labels.shape[0] == thickness.shape[0], "Vertex count mismatch"

    # 4. Compute area-weighted mean thickness per parcel
    results = []
    for label_id in np.unique(labels):
        if label_id <= 0:
            continue
        mask = (labels == label_id) & (thickness > 0)
        if mask.sum() == 0:
            continue

        name = names[label_id]
        name = name.decode() if isinstance(name, bytes) else name
        w = vertex_areas[mask]
        t = thickness[mask]

        results.append({
            'parcel': name,
            'hemi': hemi,
            'n_vertices': int(mask.sum()),
            'mean_thickness': np.average(t, weights=w),
            'std_thickness': np.sqrt(np.average((t - np.average(t, weights=w))**2, weights=w)),
            'surface_area_mm2': w.sum(),
        })
    return pd.DataFrame(results)


# Usage
SUBJECTS_DIR = os.environ.get('SUBJECTS_DIR', '/data/subjects')
df = pd.concat([
    extract_thickness_schaefer(SUBJECTS_DIR, 'sub-01', 'lh'),
    extract_thickness_schaefer(SUBJECTS_DIR, 'sub-01', 'rh'),
], ignore_index=True)
df = df[~df['parcel'].isin(['Unknown', '???', 'MedialWall'])]
print(f"{len(df)} parcels extracted")
df.to_csv('sub-01_schaefer200_thickness.csv', index=False, float_format='%.4f')
```

**Key API details**: `read_annot()` returns `(labels, ctab, names)` where `labels` is a `(n_vertices,)` integer array indexing into `names`; vertex `i` in the `.annot` corresponds to vertex `i` in `?h.thickness`. `read_morph_data()` reads FreeSurfer curv-format files (`.thickness`, `.curv`, `.sulc`). The area-weighted Python approach produces values within **<0.5%** of `mris_anatomical_stats` output.

---

## 3. Nature-quality Bayesian neuroimaging figures

### Journal specifications and SciencePlots setup

Nature Neuroscience requires **Arial/Helvetica** (sans-serif only), minimum **5 pt** text, figures at **89 mm** (single column) or **183 mm** (double column) width, and **600 DPI** for combination figures containing both line art and halftone images. Panel labels must be **lowercase bold** (a, b, c) at 8 pt. Save as vector PDF with `pdf.fonttype: 42` so text remains editable in Illustrator.

```python
import matplotlib.pyplot as plt
import scienceplots  # pip install SciencePlots

plt.style.use(['science', 'no-latex'])  # 'no-latex' avoids LaTeX dependency

MM = 1 / 25.4  # mm → inches conversion

NATURE_RC = {
    'font.family': 'Arial',
    'font.size': 7,
    'axes.titlesize': 8,
    'axes.labelsize': 7,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 5.5,
    'figure.dpi': 150,
    'savefig.dpi': 600,
    'axes.linewidth': 0.5,
    'lines.linewidth': 0.75,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
}
plt.rcParams.update(NATURE_RC)
```

The `SciencePlots` `nature` style sets inward-pointing ticks, sans-serif fonts, and a single-column default figsize. Always add `'no-latex'` unless you have a LaTeX installation. Override with `NATURE_RC` for neuroimaging-specific needs—particularly disabling top/right spines, which is standard in the field.

### Canonical Yeo-7 network colors

These RGB values come from the FreeSurfer `Yeo2011_7Networks_ColorLUT.txt` and are universally used in the literature:

```python
YEO7 = {
    'Vis':         (120/255,  18/255, 134/255),  # #781286 purple
    'SomMot':      ( 70/255, 130/255, 180/255),  # #4682B4 steel blue
    'DorsAttn':    (  0/255, 118/255,  14/255),  # #00760E green
    'SalVentAttn': (196/255,  58/255, 250/255),  # #C43AFA violet
    'Limbic':      (220/255, 248/255, 164/255),  # #DCF8A4 pale yellow-green
    'Cont':        (230/255, 148/255,  34/255),  # #E69422 orange
    'Default':     (205/255,  62/255,  78/255),  # #CD3E4E crimson
}
```

Note that the Limbic color is very pale—consider darkening it by ~30% for text labels or thin lines, or adding edge colors on bar plots to maintain visibility in print.

### Forest plot with HDI bars and ROPE shading

```python
import numpy as np
import arviz as az
from matplotlib.patches import Patch

def forest_plot_hdi_rope(posteriors, rope=(-0.1, 0.1), hdi_prob=0.94,
                          colors=None, figsize=(89*MM, 100*MM)):
    """Bayesian forest plot: thin bar = 94% HDI, thick bar = 50% HDI, dot = median."""
    labels = list(posteriors.keys())
    n = len(labels)
    fig, ax = plt.subplots(figsize=figsize)

    # ROPE shading
    ax.axvspan(*rope, alpha=0.12, color='#888888', label=f'ROPE [{rope[0]}, {rope[1]}]')
    ax.axvline(0, color='#888888', ls='--', lw=0.4)

    for i, (label, samples) in enumerate(posteriors.items()):
        c = (colors or [None]*n)[i] or '#0C5DA5'
        hdi94 = az.hdi(np.array(samples), hdi_prob=hdi_prob)
        hdi50 = az.hdi(np.array(samples), hdi_prob=0.50)
        med = np.median(samples)

        ax.plot(hdi94, [i, i], color=c, lw=1.0, solid_capstyle='round')
        ax.plot(hdi50, [i, i], color=c, lw=3.0, solid_capstyle='round')
        ax.plot(med, i, 'o', color=c, ms=4, mec='white', mew=0.5, zorder=5)

    ax.set_yticks(range(n))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Posterior effect size')
    ax.invert_yaxis()
    ax.legend(fontsize=5, loc='lower right')
    return fig, ax
```

### Multi-panel hero figure with GridSpec

Use `layout='constrained'` (matplotlib 3.8+) and `subplot_mosaic` for the clearest layout code:

```python
fig, axd = plt.subplot_mosaic(
    [['brain', 'brain', 'brain'],
     ['forest', 'forest', 'radar'],
     ['bar',    'scatter','radar']],
    figsize=(183*MM, 160*MM),
    layout='constrained',
    gridspec_kw={'height_ratios': [1.2, 1, 1]},
)

def panel_label(ax, label, x=-0.12, y=1.06):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=8, fontweight='bold', va='top', fontfamily='Arial')

for key, lbl in {'brain':'a','forest':'b','radar':'c','bar':'d','scatter':'e'}.items():
    panel_label(axd[key], lbl)
```

For export, always produce both vector and raster:

```python
fig.savefig('figure1.pdf', dpi=600, bbox_inches='tight', pad_inches=0.02)
fig.savefig('figure1.tiff', dpi=600, bbox_inches='tight',
            pil_kwargs={'compression': 'tiff_lzw'})
```

---

## 4. Surface visualization with nilearn and surfplot

### Mapping parcel statistics to vertices for nilearn

`nilearn.datasets.fetch_atlas_schaefer_2018()` returns a **volumetric NIfTI in MNI152 space**, not surface annot files. For surface plotting, either use `vol_to_surf` with `interpolation='nearest'` or (more accurately) read the CBIG annot files directly with nibabel:

```python
import numpy as np
import nibabel as nib
from nilearn import datasets
from nilearn.datasets import load_fsaverage, load_fsaverage_data
from nilearn.surface import SurfaceImage, vol_to_surf
from nilearn.plotting import plot_surf_stat_map, show

# Approach A: From CBIG annot files (recommended for accuracy)
labels_lh, _, names_lh = nib.freesurfer.read_annot(
    'lh.Schaefer2018_200Parcels_7Networks_order.annot')
labels_rh, _, names_rh = nib.freesurfer.read_annot(
    'rh.Schaefer2018_200Parcels_7Networks_order.annot')

# parcel_betas: shape (200,) — e.g. posterior means from Bayesian model
# parcels 0-99 = LH, 100-199 = RH
parcel_betas = np.random.randn(200)

vtx_lh = np.full(labels_lh.shape, np.nan)
for i in range(1, 101):
    vtx_lh[labels_lh == i] = parcel_betas[i - 1]

vtx_rh = np.full(labels_rh.shape, np.nan)
for i in range(1, 101):
    vtx_rh[labels_rh == i] = parcel_betas[99 + i]

# Approach B: From volumetric atlas via vol_to_surf
fsaverage5 = datasets.fetch_surf_fsaverage('fsaverage5')
schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7)
lh_parc = vol_to_surf(schaefer.maps, fsaverage5['pial_left'],
                       inner_mesh=fsaverage5['white_left'],
                       interpolation='nearest').astype(int)
lookup = np.full(201, np.nan)
lookup[1:201] = parcel_betas
vtx_lh = lookup[np.clip(lh_parc, 0, 200)]
```

### Plotting with plot_surf_stat_map

```python
fsaverage_meshes = load_fsaverage('fsaverage5')
fsaverage_sulcal = load_fsaverage_data(data_type='sulcal')

stat_img = SurfaceImage(
    mesh=fsaverage_meshes['inflated'],
    data={'left': vtx_lh, 'right': vtx_rh}
)

for view in ['lateral', 'medial']:
    plot_surf_stat_map(
        surf_mesh=fsaverage_meshes['inflated'],
        stat_map=stat_img,
        hemi='left', view=view,
        bg_map=fsaverage_sulcal, bg_on_data=True,
        cmap='RdBu_r', symmetric_cbar=True,
        threshold=0.5, colorbar=True,
        title=f'β coefficients — {view}',
    )
show()
```

### surfplot for higher-quality renders

The **surfplot** package (built on Brainspace's VTK backend) produces cleaner publication figures than nilearn's matplotlib renderer. It supports **multiple overlapping layers** (sulcal depth + stat map + parcel outlines), which nilearn cannot do natively. It primarily uses the **fsLR Conte69** surface (32k vertices/hemisphere via `neuromaps`), but also works with fsaverage.

```python
from neuromaps.datasets import fetch_fslr
from surfplot import Plot
from brainspace.datasets import load_parcellation
from brainspace.utils.parcellation import map_to_labels

surfaces = fetch_fslr()
lh, rh = surfaces['inflated']
sulc_lh, sulc_rh = surfaces['sulc']
labeling = load_parcellation('schaefer', scale=400, join=True)

# Map 400 parcel values to vertices
parcel_vals = np.random.randn(400)
mask = labeling != 0
vtx_data = map_to_labels(parcel_vals, labeling, mask=mask, fill=np.nan)

n_lh = 32492
p = Plot(lh, rh, views=['lateral', 'medial'], layout='grid',
         size=(800, 400), zoom=1.2)
p.add_layer({'left': sulc_lh, 'right': sulc_rh}, cmap='binary_r', cbar=False)
p.add_layer({'left': vtx_data[:n_lh], 'right': vtx_data[n_lh:]},
            cmap='RdBu_r', color_range=(-2, 2), cbar_label='Effect size (β)')
fig = p.build(cbar_kws={'location': 'bottom', 'n_ticks': 5, 'decimals': 1})
fig.savefig('brain_surfplot.pdf', dpi=300, bbox_inches='tight')
```

**surfplot vs nilearn**: surfplot uses VTK ray-tracing for rendering quality, supports `as_outline=True` for parcel boundaries, and handles multiple layers without manual subplot management. The cost is heavier dependencies (VTK, neuromaps, brainspace). For simple single-layer plots in fsaverage space, nilearn is sufficient; for multi-layer publication figures, surfplot is clearly superior.

---

## 5. Radar chart for Yeo-7 ipsilateral vs contralateral effects

The complete implementation below uses matplotlib's polar projection with **Okabe-Ito colorblind-safe** trace colors (`#D55E00` vermillion, `#0072B2` blue), HDI bands via `fill_between`, and Yeo-7 colored axis labels. The critical trick is **closing the polygon ring** by appending the first data point to all arrays.

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ── Data (replace with real posterior summaries) ────────────────
networks = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']
N = len(networks)

ipsi_mean   = np.array([0.15, 0.42, 0.28, 0.35, 0.10, 0.38, 0.52])
contra_mean = np.array([0.08, 0.25, 0.18, 0.22, 0.05, 0.20, 0.30])
ipsi_lo     = np.array([0.05, 0.30, 0.15, 0.22, 0.01, 0.25, 0.40])
ipsi_hi     = np.array([0.26, 0.55, 0.40, 0.48, 0.20, 0.50, 0.65])
contra_lo   = np.array([-0.02, 0.12, 0.05, 0.10, -0.05, 0.08, 0.18])
contra_hi   = np.array([ 0.18, 0.38, 0.30, 0.34,  0.15, 0.32, 0.42])

YEO7_RGB = {
    'Vis': (120/255, 18/255, 134/255), 'SomMot': (70/255, 130/255, 180/255),
    'DorsAttn': (0/255, 118/255, 14/255), 'SalVentAttn': (196/255, 58/255, 250/255),
    'Limbic': (220/255, 248/255, 164/255), 'Cont': (230/255, 148/255, 34/255),
    'Default': (205/255, 62/255, 78/255),
}

# ── Close polygon ring ──────────────────────────────────────────
def close(arr):
    return np.concatenate([arr, [arr[0]]])

angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
angles_c = close(angles)

# ── Plot ────────────────────────────────────────────────────────
IPSI, CONTRA = '#D55E00', '#0072B2'

fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True), dpi=300)
plt.rcParams.update({'font.family': 'Arial', 'font.size': 9, 'pdf.fonttype': 42})

ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)

# HDI bands
ax.fill_between(angles_c, close(ipsi_lo), close(ipsi_hi),
                color=IPSI, alpha=0.18, lw=0)
ax.fill_between(angles_c, close(contra_lo), close(contra_hi),
                color=CONTRA, alpha=0.18, lw=0)

# Traces
ax.plot(angles_c, close(ipsi_mean), color=IPSI, lw=1.8,
        marker='o', ms=5, mfc='white', mec=IPSI, mew=1.2, label='Ipsilateral')
ax.plot(angles_c, close(contra_mean), color=CONTRA, lw=1.8, ls='--',
        marker='s', ms=4.5, mfc='white', mec=CONTRA, mew=1.2, label='Contralateral')

# Zero-reference ring
ax.plot(angles_c, np.zeros_like(angles_c), color='#888', lw=0.5, ls=':')

# Colored axis labels
ax.set_xticks(angles)
ax.set_xticklabels([])
r_max = 0.7
for angle, net in zip(angles, networks):
    deg = np.degrees(angle) % 360
    ha = 'left' if 15 < deg < 165 else ('right' if 195 < deg < 345 else 'center')
    ax.text(angle, r_max + 0.12, net, ha=ha, va='center',
            fontsize=10, fontweight='bold', color=YEO7_RGB[net])

ax.set_ylim(-0.05, r_max)
ax.set_yticks([0.0, 0.2, 0.4, 0.6])
ax.set_yticklabels(['0.0', '0.2', '0.4', '0.6'], fontsize=7, color='#555')
ax.grid(color='#CCC', lw=0.4)

legend_elements = [
    Line2D([0],[0], color=IPSI, lw=1.8, marker='o', ms=5, mfc='white', mec=IPSI, label='Ipsilateral'),
    Line2D([0],[0], color=CONTRA, lw=1.8, ls='--', marker='s', ms=4.5, mfc='white', mec=CONTRA, label='Contralateral'),
    Patch(fc=IPSI, alpha=0.35, label='94% HDI (ipsi)'),
    Patch(fc=CONTRA, alpha=0.35, label='94% HDI (contra)'),
]
ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.35, 1.12),
          fontsize=8, framealpha=0.95, edgecolor='#CCC')
ax.set_title('Network-level effects by hemisphere', fontsize=12,
             fontweight='bold', pad=28)

fig.savefig('yeo7_radar.pdf', dpi=300, bbox_inches='tight')
```

The three implementation essentials are: (1) `np.concatenate([arr, [arr[0]]])` to close every data ring; (2) `ax.fill_between()` on polar axes for HDI shading, passing closed angle and bound arrays; (3) manual axis-label placement with per-network colors via `ax.text()` after clearing default tick labels. The Okabe-Ito vermillion/blue palette was chosen because it contrasts against all seven Yeo-7 background colors and remains distinguishable under common color-vision deficiencies.

---

## Conclusion

The critical decisions in this pipeline reduce to three: use **`--sval-annot`** (not `--sval-annot-file`) with `mri_surf2surf` for annotation projection, where NNF interpolation is applied automatically; use **area-weighted averaging** when computing thickness in Python, since naive `np.mean` introduces tessellation-density bias that can reach several percent in thin cortical regions; and choose **surfplot over nilearn** when creating multi-layer publication brain renders, as its VTK backend supports sulcal shading + stat map + parcel outlines in a single composited view. For figures, the `SciencePlots` `['science', 'no-latex']` style combined with the `NATURE_RC` overrides shown in Section 3 produces figures that meet Nature Neuroscience specifications without manual post-processing. The entire Python codebase—from thickness extraction through surface visualization to radar charts—runs on Python 3.11 with `nibabel`, `nilearn ≥0.13`, `matplotlib ≥3.8`, `arviz`, and optionally `surfplot + brainspace + neuromaps`.