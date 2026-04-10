"""
Hippocampal surface analysis for MTLE-HS and beyond.

GPU-accelerated backends for vertex-wise operations:
  - vertex_glm(): batched OLS across all vertices simultaneously via
    torch.linalg.lstsq (single GPU call replaces per-vertex loop)
  - tfce(): GPU-accelerated threshold scoring (the per-threshold
    contribution computation parallelises over vertices)
  - vertex_asymmetry_map(): element-wise on GPU for large cohorts

All other hippocampus-specific methods (subfield metrics, AP/PD profiles,
atrophy classification) are inherently lightweight and stay on CPU.

See v0.1 hippocampus.py docstring for full references [1]-[5].
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp

from corticalfields.subcortical import (
    SubcorticalSurface, _get_device, _to_torch, _vram_available_gb, _estimate_vram_gb,
)

logger = logging.getLogger(__name__)


# ===================================================================
# Constants
# ===================================================================

HIPPUNFOLD_SUBFIELDS = {
    "subiculum": 1, "CA1": 2, "CA2": 3, "CA3": 4, "CA4/DG": 5, "SRLM": 6,
}
_SUBFIELD_ID_TO_NAME = {v: k for k, v in HIPPUNFOLD_SUBFIELDS.items()}

SUBFIELD_COLORS = {
    "subiculum": "#E69F00", "CA1": "#56B4E9", "CA2": "#009E73",
    "CA3": "#F0E442", "CA4/DG": "#0072B2", "SRLM": "#CC79A7",
}

HIPPUNFOLD_DENSITIES = {"0p5mm": 7262, "1mm": 2004, "2mm": 419}

ILAE_HS_TYPES = {
    1: "Type 1 (classical): severe loss throughout CA1-CA4 + DG",
    2: "Type 2 (CA1-predominant): loss limited to CA1",
    3: "Type 3 (hilar-predominant): loss limited to CA4/hilus",
}


# ===================================================================
# HippocampalSurface (extends SubcorticalSurface)
# ===================================================================

@dataclass
class HippocampalSurface(SubcorticalSurface):
    """Hippocampal surface with subfield labels, axis coordinates, and GPU analysis.

    Inherits all SubcorticalSurface methods (including GPU curvatures,
    ShapeDNA, FPFH, ICP, etc.) and adds hippocampus-specific capabilities.

    Additional Parameters
    ---------------------
    subfield_labels : (N,) int — per-vertex subfield (1-6, HippUnfold convention)
    ap_coord : (N,) float or None — anterior-posterior (0=ant, 1=post)
    pd_coord : (N,) float or None — proximal-distal (0=prox/sub, 1=dist/DG)
    unfolded_vertices : (N, 3) or None — flat-map coordinates
    density : str — HippUnfold density ('0p5mm', '1mm', '2mm')
    """
    subfield_labels: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    ap_coord: Optional[np.ndarray] = None
    pd_coord: Optional[np.ndarray] = None
    unfolded_vertices: Optional[np.ndarray] = None
    density: str = "0p5mm"

    def __post_init__(self):
        super().__post_init__()
        if self.subfield_labels.size > 0:
            self.subfield_labels = np.asarray(self.subfield_labels, dtype=np.int32)

    # -- subfield properties --
    @property
    def subfield_names(self):
        return [_SUBFIELD_ID_TO_NAME.get(int(s), f"unk_{s}") for s in np.unique(self.subfield_labels) if s > 0]

    def subfield_mask(self, subfield):
        lid = HIPPUNFOLD_SUBFIELDS[subfield] if isinstance(subfield, str) else int(subfield)
        return self.subfield_labels == lid

    def subfield_vertices(self, subfield):
        return self.vertices[self.subfield_mask(subfield)]

    def subfield_areas(self):
        if self.subfield_labels.size == 0 or self.subfield_labels.max() == 0:
            return {n: 0.0 for n in HIPPUNFOLD_SUBFIELDS}
        fa = self.face_areas
        fl = np.zeros(self.n_faces, dtype=np.int32)
        for i in range(self.n_faces):
            tl = self.subfield_labels[self.faces[i]]
            vals, counts = np.unique(tl[tl > 0], return_counts=True)
            if len(vals) > 0: fl[i] = vals[np.argmax(counts)]
        return {n: float(fa[fl == lid].sum()) for n, lid in HIPPUNFOLD_SUBFIELDS.items()}

    def subfield_mean_thickness(self):
        if "thickness" not in self.overlays:
            raise KeyError("No 'thickness' overlay.")
        th = self.overlays["thickness"]
        return {n: float(np.nanmean(th[self.subfield_mask(n)])) if self.subfield_mask(n).sum() > 0
                else float("nan") for n in HIPPUNFOLD_SUBFIELDS}

    def subfield_overlay_stats(self, overlay_name):
        data = self.get_overlay(overlay_name); result = {}
        for name in HIPPUNFOLD_SUBFIELDS:
            vals = data[self.subfield_mask(name)]; valid = vals[np.isfinite(vals)]
            if len(valid) > 0:
                result[name] = {"mean": float(np.mean(valid)), "std": float(np.std(valid)),
                                "median": float(np.median(valid)), "min": float(np.min(valid)),
                                "max": float(np.max(valid)), "n_vertices": int(self.subfield_mask(name).sum())}
            else:
                result[name] = {k: float("nan") for k in ["mean","std","median","min","max"]}
                result[name]["n_vertices"] = 0
        return result

    # -- axis profiling --
    def ap_profile(self, overlay_name, n_bins=20, aggregation="mean"):
        if self.ap_coord is None: raise ValueError("AP coordinates not loaded.")
        data = self.get_overlay(overlay_name)
        edges = np.linspace(0, 1, n_bins+1); centres = 0.5*(edges[:-1]+edges[1:])
        agg = {"mean": np.nanmean, "median": np.nanmedian, "std": np.nanstd}[aggregation]
        profile = np.array([agg(data[(self.ap_coord >= edges[i]) & (self.ap_coord < edges[i+1])])
                            if ((self.ap_coord >= edges[i]) & (self.ap_coord < edges[i+1])).sum() > 0
                            else np.nan for i in range(n_bins)])
        return centres, profile

    def pd_profile(self, overlay_name, n_bins=10, aggregation="mean"):
        if self.pd_coord is None: raise ValueError("PD coordinates not loaded.")
        data = self.get_overlay(overlay_name)
        edges = np.linspace(0, 1, n_bins+1); centres = 0.5*(edges[:-1]+edges[1:])
        agg = {"mean": np.nanmean, "median": np.nanmedian, "std": np.nanstd}[aggregation]
        profile = np.array([agg(data[(self.pd_coord >= edges[i]) & (self.pd_coord < edges[i+1])])
                            if ((self.pd_coord >= edges[i]) & (self.pd_coord < edges[i+1])).sum() > 0
                            else np.nan for i in range(n_bins)])
        return centres, profile

    # -- MTLE-HS metrics --
    def subfield_asymmetry_indices(self, contralateral, metric="area"):
        if metric == "area": iv, cv = self.subfield_areas(), contralateral.subfield_areas()
        elif metric == "thickness": iv, cv = self.subfield_mean_thickness(), contralateral.subfield_mean_thickness()
        else: raise ValueError(f"Unknown metric '{metric}'")
        return {n: float(2*(iv.get(n,0)-cv.get(n,0))/(iv.get(n,0)+cv.get(n,0))) if abs(iv.get(n,0)+cv.get(n,0))>1e-12
                else 0.0 for n in HIPPUNFOLD_SUBFIELDS}

    def vertex_asymmetry_map(self, contralateral, overlay_name="thickness", *, device="auto"):
        """Vertex-wise asymmetry. GPU path for large vertex counts."""
        ipsi = self.get_overlay(overlay_name); contra = contralateral.get_overlay(overlay_name)
        if ipsi.shape[0] != contra.shape[0]:
            raise ValueError(f"Vertex mismatch: {ipsi.shape[0]} vs {contra.shape[0]}")
        dev = _get_device(device)
        if dev.type == "cuda":
            import torch
            with torch.no_grad():
                it, ct = _to_torch(ipsi, dev), _to_torch(contra, dev)
                mean_v = (it + ct) / 2; mean_v[mean_v.abs() < 1e-12] = 1e-12
                asym = ((it - ct) / mean_v).cpu().numpy().astype(np.float64)
                del it, ct, mean_v; torch.cuda.empty_cache()
        else:
            mean_v = (ipsi + contra) / 2; mean_v[np.abs(mean_v) < 1e-12] = 1e-12
            asym = (ipsi - contra) / mean_v
        self.add_overlay(f"asym_{overlay_name}", asym)
        return asym

    def atrophy_classification(self, normative_means, normative_stds, z_threshold=-2.0):
        """Classify HS type from subfield z-scores (ILAE typology)."""
        if "thickness" not in self.overlays: raise KeyError("Need 'thickness' overlay.")
        st = self.subfield_mean_thickness()
        sz = {n: float((st.get(n,0)-normative_means.get(n,0))/max(normative_stds.get(n,1),1e-6))
              for n in HIPPUNFOLD_SUBFIELDS if n != "SRLM"}
        atroph = [n for n, z in sz.items() if z < z_threshold]
        if not atroph: hs_type, desc = None, "No significant subfield atrophy."
        elif set(atroph) >= {"CA1","CA3","CA4/DG"}: hs_type, desc = 1, ILAE_HS_TYPES[1]
        elif set(atroph) == {"CA1"}: hs_type, desc = 2, ILAE_HS_TYPES[2]
        elif "CA4/DG" in atroph and "CA1" not in atroph: hs_type, desc = 3, ILAE_HS_TYPES[3]
        else: hs_type, desc = None, f"Atypical: atrophy in {', '.join(atroph)}"
        return {"subfield_z": sz, "atrophic_subfields": atroph, "hs_type": hs_type, "hs_description": desc}

    # -- gradients --
    def compute_hippocampal_gradients(self, connectivity_matrix, n_components=5,
                                      kernel="cosine", approach="dm"):
        """Diffusion map gradients via BrainSpace GradientMaps."""
        from brainspace.gradient import GradientMaps
        gm = GradientMaps(kernel=kernel, approach=approach, n_components=n_components, random_state=42)
        gm.fit(connectivity_matrix)
        grads = gm.gradients_
        for i in range(min(n_components, 5)):
            self.add_overlay(f"gradient_G{i+1}", grads[:, i])
        self.metadata["gradient_lambdas"] = gm.lambdas_
        return grads

    # -- texture / qMRI sampling --
    def sample_volume_on_surface(self, nifti_path, method="nearest"):
        """Sample a NIfTI volume at surface vertex positions."""
        import nibabel as nib
        img = nib.load(str(nifti_path)); data = np.asarray(img.dataobj, dtype=np.float64)
        inv_aff = np.linalg.inv(img.affine)
        vox = (inv_aff @ np.hstack([self.vertices, np.ones((self.n_vertices,1))]).T).T[:,:3]
        if method == "nearest":
            ijk = np.round(vox).astype(np.int64)
            for d in range(3): ijk[:,d] = np.clip(ijk[:,d], 0, data.shape[d]-1)
            return data[ijk[:,0], ijk[:,1], ijk[:,2]].astype(np.float64)
        from scipy.ndimage import map_coordinates
        return map_coordinates(data, vox.T, order=1, mode="nearest").astype(np.float64)

    # ===============================================================
    # VERTEX GLM — GPU-accelerated batched OLS
    # ===============================================================
    def vertex_glm(self, group_data, design_matrix, contrast, *, device="auto"):
        """Vertex-wise GLM.  GPU backend computes OLS for ALL vertices
        in a single batched torch.linalg.lstsq call.

        Parameters
        ----------
        group_data : (n_subjects, n_vertices) — morphometric data
        design_matrix : (n_subjects, n_regressors) — GLM design
        contrast : (n_regressors,) — contrast vector

        Returns
        -------
        t_stats : (n_vertices,)
        p_values : (n_vertices,)
        """
        from scipy import stats

        n_subj, n_vert = group_data.shape
        n_reg = design_matrix.shape[1]
        df = n_subj - n_reg

        dev = _get_device(device)
        if dev.type == "cuda":
            import torch
            # VRAM: X (s,r)*8 + Y (s,v)*4 + beta (r,v)*4 + resid (s,v)*4
            needed = _estimate_vram_gb([
                ((n_subj, n_reg), 4), ((n_subj, n_vert), 4),
                ((n_reg, n_vert), 4), ((n_subj, n_vert), 4),
            ])
            avail = _vram_available_gb()

            if needed < avail * 0.8:
                logger.info("vertex_glm GPU: %.2f/%.2f GB (%d subj, %d vert)",
                            needed, avail, n_subj, n_vert)
                with torch.no_grad():
                    X = _to_torch(design_matrix, dev)       # (s, r) f32
                    Y = _to_torch(group_data, dev)           # (s, v) f32
                    c = _to_torch(contrast, dev)              # (r,)  f32

                    # Batched OLS: beta = (X'X)^-1 X'Y
                    # torch.linalg.lstsq solves min||Xb - Y||^2
                    result = torch.linalg.lstsq(X, Y)
                    beta = result.solution                    # (r, v) f32
                    residuals = Y - X @ beta                  # (s, v) f32

                    # MSE per vertex
                    mse = (residuals**2).sum(dim=0) / max(df, 1)  # (v,) f32

                    # Contrast: t = c'beta / sqrt(var_c * mse)
                    c_beta = c @ beta                          # (v,) f32
                    XtX_inv = torch.linalg.inv(X.T @ X)       # (r, r) f32
                    var_c = (c @ XtX_inv @ c).item()           # scalar
                    se = torch.sqrt(var_c * mse)               # (v,) f32
                    se[se < 1e-12] = 1e-12
                    t_map = (c_beta / se).cpu().numpy().astype(np.float64)

                    del X, Y, c, beta, residuals, mse, c_beta, se
                    torch.cuda.empty_cache()

                p_values = 2.0 * stats.t.sf(np.abs(t_map), df=df)
                return t_map, p_values
            else:
                logger.info("vertex_glm: VRAM low (%.2f<%.2f) — CPU", needed, avail)

        # CPU fallback
        XtX_inv = np.linalg.pinv(design_matrix.T @ design_matrix)
        beta = XtX_inv @ design_matrix.T @ group_data          # (r, v)
        residuals = group_data - design_matrix @ beta            # (s, v)
        mse = np.sum(residuals**2, axis=0) / max(df, 1)         # (v,)
        c_beta = contrast @ beta                                  # (v,)
        var_c = contrast @ XtX_inv @ contrast                     # scalar
        se = np.sqrt(var_c * mse); se[se < 1e-12] = 1e-12
        t_stats = c_beta / se
        p_values = 2.0 * stats.t.sf(np.abs(t_stats), df=df)
        return t_stats, p_values

    # ===============================================================
    # TFCE — GPU-accelerated scoring
    # ===============================================================
    def tfce(self, stat_map, H=2.0, E=1.0, dh=0.1, *, device="auto"):
        """Threshold-Free Cluster Enhancement on the hippocampal mesh.

        TFCE(v) = sum_h [extent(h)^E * h^H * dh]

        The connected-components step is inherently CPU (graph traversal),
        but the per-vertex contribution accumulation is GPU-accelerated
        for large meshes.

        Parameters
        ----------
        stat_map : (N,) array
        H, E : float — height and extent exponents
        dh : float — height increment
        device : str

        Returns
        -------
        tfce_map : (N,) array
        """
        from scipy.sparse.csgraph import connected_components

        N = self.n_vertices
        # Build adjacency
        rows, cols = [], []
        for tri in self.faces:
            for i in range(3):
                a, b = int(tri[i]), int(tri[(i+1)%3])
                rows.extend([a, b]); cols.extend([b, a])
        adj = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N))

        tfce_map = np.zeros(N, dtype=np.float64)
        max_val = stat_map.max()
        if max_val <= 0:
            return tfce_map

        dev = _get_device(device)
        use_gpu = dev.type == "cuda"

        thresholds = np.arange(dh, max_val + dh, dh)

        if use_gpu:
            import torch
            # Pre-allocate GPU accumulator
            tfce_gpu = torch.zeros(N, device=dev, dtype=torch.float32)

            for h in thresholds:
                supra = stat_map >= h
                if not np.any(supra):
                    continue
                # Connected components on CPU (graph traversal)
                sub_adj = adj[np.ix_(supra, supra)]
                n_comp, comp_labels = connected_components(sub_adj, directed=False)

                supra_idx = np.where(supra)[0]
                # Compute extent per component
                for c in range(n_comp):
                    cluster_local = comp_labels == c
                    extent = float(cluster_local.sum())
                    contribution = (extent ** E) * (h ** H) * dh
                    cluster_global = supra_idx[cluster_local]
                    # Accumulate on GPU
                    tfce_gpu[cluster_global] += contribution

            tfce_map = tfce_gpu.cpu().numpy().astype(np.float64)
            del tfce_gpu; torch.cuda.empty_cache()
        else:
            for h in thresholds:
                supra = stat_map >= h
                if not np.any(supra): continue
                sub_adj = adj[np.ix_(supra, supra)]
                n_comp, comp_labels = connected_components(sub_adj, directed=False)
                supra_idx = np.where(supra)[0]
                for c in range(n_comp):
                    cl = comp_labels == c
                    tfce_map[supra_idx[cl]] += (float(cl.sum())**E) * (h**H) * dh

        return tfce_map

    def summary(self):
        return (f"HippocampalSurface: {self.structure} ({self.hemi})\n"
                f"  Density: {self.density}  V={self.n_vertices:,}  F={self.n_faces:,}\n"
                f"  Vol={self.enclosed_volume:.1f}mm3  A={self.total_area:.1f}mm2\n"
                f"  Subfields: {', '.join(self.subfield_names) or '(none)'}\n"
                f"  AP coord: {self.ap_coord is not None}  PD coord: {self.pd_coord is not None}\n"
                f"  Unfolded: {self.unfolded_vertices is not None}\n"
                f"  Overlays: {', '.join(self.overlay_names) or '(none)'}")


# ===================================================================
# IO: Load HippUnfold outputs
# ===================================================================

def load_hippocampal_surface(hippunfold_dir, subject_id, hemi="lh", *, density="0p5mm",
                             label="hipp", space="T1w", surf_type="midthickness",
                             load_overlays=True, load_subfields=True, load_unfolded=True,
                             load_coords=True):
    """Load HippUnfold hippocampal surface with all available data."""
    import nibabel as nib
    hdir = Path(hippunfold_dir); hb = "L" if hemi == "lh" else "R"; prefix = f"sub-{subject_id}"

    def _find(parts):
        fname = "_".join(parts)
        for sd in ["surf", "anat"]:
            for base in [hdir/prefix/sd, hdir/"hippunfold"/prefix/sd]:
                if (base/fname).exists(): return base/fname
        matches = list(hdir.rglob(fname))
        return matches[0] if matches else None

    # Surface mesh
    sp_ = _find([prefix, f"hemi-{hb}", f"space-{space}", f"den-{density}", f"label-{label}", f"{surf_type}.surf.gii"])
    if sp_ is None: raise FileNotFoundError(f"Surface not found for {prefix} {hb} {label} in {hdir}")
    gii = nib.load(str(sp_))
    coords = gii.darrays[0].data.astype(np.float64); faces = gii.darrays[1].data.astype(np.int64)

    # Overlays
    overlays = {}
    if load_overlays:
        for mn in ["thickness", "curvature", "gyrification", "surfarea"]:
            op = _find([prefix, f"hemi-{hb}", f"space-{space}", f"den-{density}", f"label-{label}", f"{mn}.shape.gii"])
            if op: overlays[mn] = nib.load(str(op)).darrays[0].data.astype(np.float64)

    # Subfield labels
    sf_labels = np.zeros(coords.shape[0], dtype=np.int32)
    if load_subfields:
        sfp = _find([prefix, f"hemi-{hb}", f"space-{space}", f"den-{density}", f"label-{label}", "subfields.label.gii"])
        if sfp: sf_labels = nib.load(str(sfp)).darrays[0].data.astype(np.int32)

    # Unfolded surface
    unf = None
    if load_unfolded:
        ufp = _find([prefix, f"hemi-{hb}", "space-unfold", f"den-{density}", f"label-{label}", f"{surf_type}.surf.gii"])
        if ufp: unf = nib.load(str(ufp)).darrays[0].data.astype(np.float64)

    # AP/PD coordinates
    ap, pd = None, None
    if load_coords:
        for cn, attr in [("AP", "ap"), ("PD", "pd")]:
            cp = _find([prefix, f"hemi-{hb}", f"space-{space}", f"den-{density}", f"label-{label}", f"desc-{cn}", "coords.shape.gii"])
            if cp:
                val = nib.load(str(cp)).darrays[0].data.astype(np.float64)
                if attr == "ap": ap = val
                else: pd = val

    name = f"{'Left' if hemi=='lh' else 'Right'}-Hippocampus"
    return HippocampalSurface(
        vertices=coords, faces=faces, structure=name, hemi=hemi, overlays=overlays,
        metadata={"origin":"hippunfold", "hippunfold_dir":str(hdir), "subject_id":subject_id,
                  "surf_path":str(sp_), "label":label, "space":space},
        subfield_labels=sf_labels, ap_coord=ap, pd_coord=pd, unfolded_vertices=unf, density=density)


# ===================================================================
# Pairwise comparison functions
# ===================================================================

def hippocampal_asymmetry_report(ipsi, contra):
    """Comprehensive asymmetry report for MTLE-HS lateralisation.

    Accepts both HippocampalSurface and SubcorticalSurface.  Subfield-level
    asymmetry indices are only included when both surfaces are HippocampalSurface
    instances with populated subfield_labels.
    """
    report = {"volume_ai": ipsi.volume_asymmetry_index(contra.enclosed_volume)}
    d = ipsi.total_area + contra.total_area
    report["area_ai"] = 2*(ipsi.total_area-contra.total_area)/d if abs(d) > 1e-12 else 0.0

    # Shape metrics
    report["sphericity_ipsi"] = ipsi.sphericity
    report["sphericity_contra"] = contra.sphericity
    report["elongation_ipsi"] = ipsi.elongation
    report["elongation_contra"] = contra.elongation

    # Subfield metrics — only for HippocampalSurface with labels
    _has_subfields = (
        isinstance(ipsi, HippocampalSurface) and isinstance(contra, HippocampalSurface)
        and ipsi.subfield_labels.size > 0 and ipsi.subfield_labels.max() > 0
        and contra.subfield_labels.size > 0 and contra.subfield_labels.max() > 0
    )
    if _has_subfields:
        report["subfield_area_ai"] = ipsi.subfield_asymmetry_indices(contra, "area")
        if "thickness" in ipsi.overlays and "thickness" in contra.overlays:
            report["subfield_thickness_ai"] = ipsi.subfield_asymmetry_indices(contra, "thickness")
            try: report["vertex_asym_map"] = ipsi.vertex_asymmetry_map(contra, "thickness")
            except ValueError as e: logger.warning("Vertex asym failed: %s", e)
    return report

def hippocampal_spectral_analysis(surf, *, n_eigenpairs=None, hks_scales=16,
                                  wks_energies=16, compute_curvatures=True,
                                  backend="auto", device="auto"):
    """Full spectral + shape + subfield pipeline for a hippocampal surface.

    Accepts both HippocampalSurface and SubcorticalSurface.  Subfield-specific
    metrics (subfield_areas, subfield_thickness, ap_thickness_profile) are only
    included when *surf* is a HippocampalSurface with populated subfield_labels.
    """
    from corticalfields.subcortical import subcortical_spectral_analysis
    results = subcortical_spectral_analysis(
        surf, n_eigenpairs=n_eigenpairs, hks_scales=hks_scales, wks_energies=wks_energies,
        compute_curvatures=compute_curvatures, compute_principal=True, compute_shapedna=True,
        backend=backend, device=device)
    # Subfield metrics — only for HippocampalSurface with labels
    if isinstance(surf, HippocampalSurface) and surf.subfield_labels.size > 0 and surf.subfield_labels.max() > 0:
        results["subfield_areas"] = surf.subfield_areas()
        if "thickness" in surf.overlays:
            results["subfield_thickness"] = surf.subfield_mean_thickness()
            if surf.ap_coord is not None:
                results["ap_thickness_profile"] = surf.ap_profile("thickness", n_bins=20)
    return results
