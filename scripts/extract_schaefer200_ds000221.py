#!/usr/bin/env python3
"""
annot_schaefer200_ds000221.py
================================

Normative cohort (ds000221) extraction pipeline:
  FastSurfer outputs → Schaefer-200 parcellation → Yeo-7 network aggregation

Unlike the CEPESC script, there is NO lateralization (healthy controls).
This script extracts MULTIPLE morphometric features per parcel:
  - Cortical thickness (area-weighted mean, mm)
  - Mean curvature (1/mm)
  - Sulcal depth (mm)
  - Surface area (mm²)

These multi-feature outputs serve as the normative reference for:
  1. CorticalFields normative modeling (z-scores, surprise maps)
  2. Direct comparison with CEPESC MTLE-HS cohort
  3. MSN (Morphometric Similarity Network) construction

Steps:
  1. Discover subjects with completed FastSurfer outputs
  2. Project Schaefer-200 .annot from fsaverage → individual subjects
  3. Extract area-weighted morphometrics per parcel (200 parcels × 4 features)
  4. Aggregate by Yeo-7 network (7 networks × 2 hemis × 4 features)
  5. Build group-level matrices

Requirements:
  - FreeSurfer 8.x (for mri_surf2surf)
  - Completed FastSurfer outputs in FS_DIR
  - Python: nibabel, numpy, pandas

Usage:
  source /usr/local/freesurfer/8.1.0/SetUpFreeSurfer.sh
  python extract_schaefer200_ds000221.py

Author: Velho Mago (rdneuro)
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# FastSurfer output directory (FreeSurfer-compatible structure)
FS_DIR = Path("/media/rdx/disk4/analysis/ds000221/output/fastsurfer")

# Schaefer-200 annotation files (fsaverage space, 163842 vertices)
ANNOT_DIR = Path(
    "/media/rdx/disk4/analysis/cepesc/structural/freesurfer/v741/"
    "corticalfields_hads/info"
)
ANNOT_LH = ANNOT_DIR / "lh.Schaefer2018_200Parcels_7Networks_order.annot"
ANNOT_RH = ANNOT_DIR / "rh.Schaefer2018_200Parcels_7Networks_order.annot"

# Output directory
OUTPUT_DIR = Path("/media/rdx/disk4/analysis/ds000221/output/sch200yeo7")

# FreeSurfer home
FREESURFER_HOME = Path(
    os.environ.get("FREESURFER_HOME", "/usr/local/freesurfer/8.1.0")
)

ANNOT_NAME = "Schaefer2018_200Parcels_7Networks_order"

YEO7_NETWORKS = [
    "Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"
]

# Morphometric overlays to extract
# key = FreeSurfer filename (without hemi prefix), value = output column name
MORPH_OVERLAYS = {
    "thickness": "thickness",
    "curv":      "curvature",
    "sulc":      "sulcal_depth",
    "area":      "surface_area",
}


# ═══════════════════════════════════════════════════════════════════════════
#  FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def ensure_fsaverage(fs_dir: Path) -> None:
    """Ensure fsaverage is accessible in FS_DIR for mri_surf2surf."""
    fsavg = fs_dir / "fsaverage"
    if fsavg.exists():
        print(f"  ✓ fsaverage found")
        return
    sys_fsavg = FREESURFER_HOME / "subjects" / "fsaverage"
    if not sys_fsavg.exists():
        raise FileNotFoundError(f"fsaverage not found at {sys_fsavg}")
    fsavg.symlink_to(sys_fsavg)
    print(f"  ✓ Symlink: {fsavg} → {sys_fsavg}")


def discover_subjects(fs_dir: Path) -> list:
    """Find all subjects with completed FastSurfer outputs."""
    subjects = []
    for d in sorted(fs_dir.iterdir()):
        if d.is_dir() and d.name.startswith("sub-"):
            if (d / "surf" / "lh.white").exists() and \
               (d / "surf" / "lh.thickness").exists() and \
               (d / "surf" / "rh.sphere.reg").exists():
                subjects.append(d.name)
    return subjects


def project_annot(
    subject: str, hemi: str, src_annot: Path,
    fs_dir: Path, out_dir: Path,
) -> Path:
    """Project fsaverage Schaefer .annot to individual subject."""
    out_annot = out_dir / f"{hemi}.{ANNOT_NAME}.annot"
    if out_annot.exists():
        return out_annot

    env = os.environ.copy()
    env["SUBJECTS_DIR"] = str(fs_dir)

    result = subprocess.run([
        "mri_surf2surf",
        "--hemi", hemi,
        "--srcsubject", "fsaverage",
        "--trgsubject", subject,
        "--sval-annot", str(src_annot),
        "--tval", str(out_annot),
    ], capture_output=True, text=True, env=env)

    if result.returncode != 0:
        raise RuntimeError(
            f"mri_surf2surf failed for {subject} {hemi}: "
            f"{result.stderr.strip()}"
        )
    return out_annot


def compute_vertex_areas(coords: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area per vertex via Voronoi-like triangle distribution."""
    v0, v1, v2 = coords[faces[:, 0]], coords[faces[:, 1]], coords[faces[:, 2]]
    tri_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    vertex_areas = np.zeros(len(coords))
    for i in range(3):
        np.add.at(vertex_areas, faces[:, i], tri_areas / 3.0)
    return vertex_areas


def extract_parcel_morphometrics(
    subject: str, hemi: str,
    fs_dir: Path, annot_path: Path,
) -> pd.DataFrame:
    """
    Extract area-weighted morphometric features per Schaefer-200 parcel.

    For each parcel, extracts:
      - thickness: area-weighted mean cortical thickness (mm)
      - curvature: area-weighted mean curvature (1/mm)
      - sulcal_depth: area-weighted mean sulcal depth (mm)
      - surface_area: total surface area of the parcel (mm²)

    Returns DataFrame with columns:
      parcel, hemi, network, n_vertices, thickness, curvature,
      sulcal_depth, surface_area
    """
    import nibabel.freesurfer as fs

    base = fs_dir / subject

    # Read projected annotation
    labels, ctab, names = fs.read_annot(str(annot_path))

    # Read white surface geometry for vertex areas
    coords, faces = fs.read_geometry(str(base / "surf" / f"{hemi}.white"))
    vertex_areas = compute_vertex_areas(coords, faces)

    # Read all morphometric overlays
    morph_data = {}
    for fs_name, col_name in MORPH_OVERLAYS.items():
        morph_path = base / "surf" / f"{hemi}.{fs_name}"
        if morph_path.exists():
            morph_data[col_name] = fs.read_morph_data(str(morph_path))
        else:
            print(f"      ⚠ {hemi}.{fs_name} not found, skipping")

    n_vertices = labels.shape[0]
    assert all(v.shape[0] == n_vertices for v in morph_data.values()), \
        "Vertex count mismatch between overlays"

    # Extract per-parcel statistics
    results = []
    for label_id in np.unique(labels):
        if label_id <= 0:
            continue

        if label_id < len(names):
            name = names[label_id]
            name = name.decode("utf-8") if isinstance(name, bytes) else name
        else:
            continue

        if name in ("Unknown", "???", "MedialWall", "unknown"):
            continue

        # Parse network from label
        parts = name.split("_")
        network = parts[2] if len(parts) >= 3 else "Unknown"

        # Mask: vertices in this parcel with positive thickness
        mask = (labels == label_id)
        if "thickness" in morph_data:
            mask = mask & (morph_data["thickness"] > 0)
        if mask.sum() == 0:
            continue

        w = vertex_areas[mask]
        row = {
            "parcel": name,
            "hemi": hemi,
            "network": network,
            "n_vertices": int(mask.sum()),
        }

        for col_name, data in morph_data.items():
            vals = data[mask]
            if col_name == "surface_area":
                # Surface area: sum, not weighted mean
                row[col_name] = float(w.sum())
            else:
                # Area-weighted mean for all other metrics
                row[col_name] = float(np.average(vals, weights=w))
                row[f"{col_name}_std"] = float(
                    np.sqrt(np.average((vals - np.average(vals, weights=w))**2,
                                       weights=w))
                )

        results.append(row)

    return pd.DataFrame(results)


def aggregate_by_network(parcel_df: pd.DataFrame) -> pd.DataFrame:
    """Area-weighted aggregation of parcel metrics → Yeo-7 networks."""
    rows = []
    for hemi in ["lh", "rh"]:
        hemi_df = parcel_df[parcel_df["hemi"] == hemi]
        for net in YEO7_NETWORKS:
            net_df = hemi_df[hemi_df["network"] == net]
            if len(net_df) == 0:
                continue

            areas = net_df["surface_area"].values if "surface_area" in net_df else \
                    np.ones(len(net_df))
            row = {
                "hemi": hemi,
                "network": net,
                "n_parcels": len(net_df),
                "n_vertices_total": int(net_df["n_vertices"].sum()),
                "surface_area": float(areas.sum()),
            }

            # Area-weighted mean for thickness, curvature, sulcal_depth
            for col in ["thickness", "curvature", "sulcal_depth"]:
                if col in net_df.columns:
                    row[col] = float(np.average(net_df[col].values, weights=areas))

            rows.append(row)

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    print("═" * 65)
    print("  ds000221 — Schaefer-200 → Yeo-7 Normative Extraction")
    print("═" * 65)
    print(f"  FS_DIR    : {FS_DIR}")
    print(f"  OUTPUT_DIR: {OUTPUT_DIR}")
    print()

    # ── Discover subjects ───────────────────────────────────────────────
    print("[1/5] Discovering subjects with completed FastSurfer outputs...")
    subjects = discover_subjects(FS_DIR)
    print(f"  Found {len(subjects)} completed subjects")

    if len(subjects) == 0:
        print("  ⚠ No completed subjects found! Run FastSurfer first.")
        print(f"    Expected structure: {FS_DIR}/sub-XXXXXX/surf/lh.white")
        return

    for s in subjects[:5]:
        print(f"    {s}")
    if len(subjects) > 5:
        print(f"    ... and {len(subjects)-5} more")

    # ── Validate annotation files ───────────────────────────────────────
    for annot in [ANNOT_LH, ANNOT_RH]:
        if not annot.exists():
            raise FileNotFoundError(f"Annotation not found: {annot}")
    print(f"  ✓ Schaefer-200 annotations found")

    # ── Ensure fsaverage ────────────────────────────────────────────────
    print("\n[2/5] Checking fsaverage...")
    ensure_fsaverage(FS_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Process each subject ────────────────────────────────────────────
    print(f"\n[3/5] Projecting annotations + extracting morphometrics...")

    group_thickness_rows = []    # subject × 14 (Yeo-7 lh+rh thickness)
    group_schaefer_rows = []     # subject × 200 (parcel-level thickness)
    group_full_rows = []         # subject × 70 (Yeo-7 × all features)
    failed = []

    for i, sub in enumerate(subjects):
        sub_dir = OUTPUT_DIR / sub
        sub_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  [{i+1}/{len(subjects)}] {sub}")

        try:
            all_parcels = []

            for hemi, src_annot in [("lh", ANNOT_LH), ("rh", ANNOT_RH)]:
                print(f"    {hemi}: projecting...", end=" ", flush=True)
                proj_annot = project_annot(sub, hemi, src_annot, FS_DIR, sub_dir)

                print("extracting...", end=" ", flush=True)
                parcel_df = extract_parcel_morphometrics(
                    sub, hemi, FS_DIR, proj_annot,
                )
                all_parcels.append(parcel_df)
                print(f"✓ ({len(parcel_df)} parcels)")

            # Combine hemispheres
            parcels_df = pd.concat(all_parcels, ignore_index=True)
            parcels_df.to_csv(
                sub_dir / "schaefer200_morphometrics.csv",
                index=False, float_format="%.4f",
            )

            # Aggregate by Yeo-7
            network_df = aggregate_by_network(parcels_df)
            network_df.to_csv(
                sub_dir / "yeo7_network_morphometrics.csv",
                index=False, float_format="%.4f",
            )

            # ── Group-level: parcel thickness (200 cols) ────────────────
            schaefer_row = {"subject": sub}
            for _, row in parcels_df.iterrows():
                if "thickness" in row:
                    schaefer_row[row["parcel"]] = row["thickness"]
            group_schaefer_rows.append(schaefer_row)

            # ── Group-level: Yeo-7 thickness (14 cols) ──────────────────
            thick_row = {"subject": sub}
            for _, row in network_df.iterrows():
                if "thickness" in row:
                    thick_row[f"{row['hemi']}_{row['network']}"] = row["thickness"]
            group_thickness_rows.append(thick_row)

            # ── Group-level: Yeo-7 all features (70 cols) ───────────────
            full_row = {"subject": sub}
            for _, row in network_df.iterrows():
                prefix = f"{row['hemi']}_{row['network']}"
                for col in ["thickness", "curvature", "sulcal_depth", "surface_area"]:
                    if col in row:
                        full_row[f"{prefix}_{col}"] = row[col]
            group_full_rows.append(full_row)

            print(f"    ✓ Saved: {sub_dir}/")

        except Exception as e:
            print(f"    ✗ FAILED: {e}")
            failed.append((sub, str(e)))

    # ── Build group-level matrices ──────────────────────────────────────
    print(f"\n[4/5] Building group-level matrices...")

    # Yeo-7 thickness (14 cols: lh_Vis, lh_SomMot, ..., rh_Default)
    if group_thickness_rows:
        df_thick = pd.DataFrame(group_thickness_rows)
        df_thick.to_csv(
            OUTPUT_DIR / "group_yeo7_thickness.csv",
            index=False, float_format="%.4f",
        )
        print(f"  ✓ group_yeo7_thickness.csv: {df_thick.shape}")

    # Schaefer-200 thickness (200 cols)
    if group_schaefer_rows:
        df_sch = pd.DataFrame(group_schaefer_rows)
        df_sch.to_csv(
            OUTPUT_DIR / "group_schaefer200_thickness.csv",
            index=False, float_format="%.4f",
        )
        print(f"  ✓ group_schaefer200_thickness.csv: {df_sch.shape}")

    # Yeo-7 all features (70 cols: 7 nets × 2 hemis × 5 features)
    if group_full_rows:
        df_full = pd.DataFrame(group_full_rows)
        df_full.to_csv(
            OUTPUT_DIR / "group_yeo7_all_features.csv",
            index=False, float_format="%.4f",
        )
        print(f"  ✓ group_yeo7_all_features.csv: {df_full.shape}")

    # ── QC summary ──────────────────────────────────────────────────────
    if group_thickness_rows:
        print(f"\n[5/5] QC — Mean thickness per Yeo-7 network (mm)")
        print(f"  {'Network':<15s}  {'LH':>12s}  {'RH':>12s}")
        print(f"  {'─'*15}  {'─'*12}  {'─'*12}")
        for net in YEO7_NETWORKS:
            lh_col = f"lh_{net}"
            rh_col = f"rh_{net}"
            if lh_col in df_thick.columns and rh_col in df_thick.columns:
                lh_m = df_thick[lh_col].mean()
                lh_s = df_thick[lh_col].std()
                rh_m = df_thick[rh_col].mean()
                rh_s = df_thick[rh_col].std()
                print(f"  {net:<15s}  {lh_m:.3f}±{lh_s:.3f}  {rh_m:.3f}±{rh_s:.3f}")

    # ── Report ──────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n  Summary:")
    print(f"    Processed: {len(group_thickness_rows)}/{len(subjects)} subjects")
    print(f"    Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    if failed:
        print(f"    ⚠ Failed ({len(failed)}):")
        for sub, err in failed:
            print(f"      • {sub}: {err}")
    else:
        print(f"    ✓ All subjects processed successfully!")

    print(f"\n  Output: {OUTPUT_DIR}")
    print("  Done! ✓")


if __name__ == "__main__":
    main()
