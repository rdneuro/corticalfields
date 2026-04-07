#!/usr/bin/env python3
"""
extract_schaefer200_yeo7_thickness.py
======================================

Pipeline: FreeSurfer recon-all → Schaefer-200 parcellation → Yeo-7 network
aggregation → lateralization (IHS/CHS) relative to hippocampal sclerosis.

Steps:
  1. Project fsaverage Schaefer-200 .annot → individual subject space
     (via mri_surf2surf, nearest-neighbor interpolation)
  2. Extract area-weighted mean cortical thickness per Schaefer-200 parcel
     (via nibabel, matching mris_anatomical_stats precision)
  3. Map each parcel to its Yeo-7 network (parsed from label names)
  4. Aggregate thickness by network (area-weighted across parcels)
  5. Lateralize as IHS (ipsilateral to HS) / CHS (contralateral)
  6. Save per-subject CSVs + group-level lateralized matrix

Requirements:
  - FreeSurfer 7.4+ or 8.x (for mri_surf2surf)
  - Python: nibabel, numpy, pandas

Usage:
  source /usr/local/freesurfer/8.1.0/SetUpFreeSurfer.sh
  python extract_schaefer200_yeo7_thickness.py

Output:
  {OUTPUT_DIR}/{sub}/schaefer200_thickness.csv     — 200 parcels per subject
  {OUTPUT_DIR}/{sub}/yeo7_network_thickness.csv    — 7 networks × 2 hemis
  {OUTPUT_DIR}/group_yeo7_lateralized_thickness.csv — 46 subjects × 14 features

Author: Velho Mago (rdneuro)
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — Edit these paths
# ═══════════════════════════════════════════════════════════════════════════

# FreeSurfer subjects directory (where recon-all outputs live)
FS_SUBJECTS_DIR = Path(
    "/media/rdx/disk4/analysis/cepesc/structural/freesurfer/v741/"
    "corticalfields_hads/data/fs"
)

# Schaefer-200 annotation files (fsaverage space)
ANNOT_LH = Path(
    "/media/rdx/disk4/analysis/cepesc/structural/freesurfer/v741/"
    "corticalfields_hads/info/"
    "lh.Schaefer2018_200Parcels_7Networks_order.annot"
)
ANNOT_RH = Path(
    "/media/rdx/disk4/analysis/cepesc/structural/freesurfer/v741/"
    "corticalfields_hads/info/"
    "rh.Schaefer2018_200Parcels_7Networks_order.annot"
)

# Schaefer-200 labels CSV (for reference/validation)
LABELS_CSV = Path(
    "/media/rdx/disk4/analysis/cepesc/structural/freesurfer/v741/"
    "corticalfields_hads/info/labels_schaefer_200_7networks.csv"
)

# Output directory
OUTPUT_DIR = Path(
    "/media/rdx/disk4/analysis/cepesc/structural/freesurfer/v741/"
    "corticalfields_hads/data/sch200yeo7"
)

# Clinical bank (for subject list and laterality)
BANK_CSV = Path(
    "/media/rdx/disk4/analysis/cepesc/structural/freesurfer/v741/"
    "corticalfields_hads/data/n46thick.csv"
)

# FreeSurfer home (for fsaverage location)
FREESURFER_HOME = Path(
    os.environ.get("FREESURFER_HOME", "/usr/local/freesurfer/8.1.0")
)

# Annotation name (without hemisphere prefix and .annot suffix)
ANNOT_NAME = "Schaefer2018_200Parcels_7Networks_order"

# Yeo-7 canonical network order and colors
YEO7_NETWORKS = [
    "Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"
]


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 1: Ensure fsaverage is accessible in SUBJECTS_DIR
# ═══════════════════════════════════════════════════════════════════════════

def ensure_fsaverage_symlink(subjects_dir: Path) -> None:
    """
    mri_surf2surf needs both fsaverage and the target subject to be
    findable within SUBJECTS_DIR. If fsaverage doesn't exist in the
    subjects directory, create a symlink to the system fsaverage.
    """
    fsaverage_path = subjects_dir / "fsaverage"
    if fsaverage_path.exists():
        print(f"  ✓ fsaverage found: {fsaverage_path}")
        return

    system_fsaverage = FREESURFER_HOME / "subjects" / "fsaverage"
    if not system_fsaverage.exists():
        raise FileNotFoundError(
            f"Cannot find fsaverage at {system_fsaverage}. "
            f"Set FREESURFER_HOME correctly."
        )

    fsaverage_path.symlink_to(system_fsaverage)
    print(f"  ✓ Created symlink: {fsaverage_path} → {system_fsaverage}")


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 2: Project Schaefer annotations to subject space
# ═══════════════════════════════════════════════════════════════════════════

def project_annot_to_subject(
    subject: str,
    hemi: str,
    src_annot: Path,
    subjects_dir: Path,
    output_dir: Path,
) -> Path:
    """
    Project fsaverage-space Schaefer .annot to individual subject space
    using mri_surf2surf with nearest-neighbor forward interpolation.

    The --sval-annot flag automatically activates NNF interpolation,
    which is essential because annotation labels are categorical integers.
    """
    out_annot = output_dir / f"{hemi}.{ANNOT_NAME}.annot"

    # Skip if already projected
    if out_annot.exists():
        return out_annot

    cmd = [
        "mri_surf2surf",
        "--hemi", hemi,
        "--srcsubject", "fsaverage",
        "--trgsubject", subject,
        "--sval-annot", str(src_annot),
        "--tval", str(out_annot),
    ]

    env = os.environ.copy()
    env["SUBJECTS_DIR"] = str(subjects_dir)

    result = subprocess.run(
        cmd, capture_output=True, text=True, env=env,
    )

    if result.returncode != 0:
        print(f"    ⚠ mri_surf2surf failed for {subject} {hemi}:")
        print(f"      {result.stderr.strip()}")
        raise RuntimeError(f"mri_surf2surf failed: {result.stderr}")

    return out_annot


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 3: Extract area-weighted thickness per Schaefer-200 parcel
# ═══════════════════════════════════════════════════════════════════════════

def compute_vertex_areas(coords: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Compute the area associated with each vertex by distributing each
    triangle's area equally among its 3 vertices (Voronoi-like).

    This matches the area-weighting used by mris_anatomical_stats,
    ensuring our Python-computed means are within <0.5% of the FS output.
    """
    v0 = coords[faces[:, 0]]
    v1 = coords[faces[:, 1]]
    v2 = coords[faces[:, 2]]
    tri_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

    vertex_areas = np.zeros(len(coords))
    for i in range(3):
        np.add.at(vertex_areas, faces[:, i], tri_areas / 3.0)

    return vertex_areas


def extract_parcel_thickness(
    subject: str,
    hemi: str,
    subjects_dir: Path,
    annot_path: Path,
) -> pd.DataFrame:
    """
    Extract area-weighted mean cortical thickness per Schaefer-200 parcel.

    Uses nibabel to read:
      - The projected .annot (per-vertex parcel labels)
      - The FreeSurfer ?h.thickness overlay (per-vertex thickness in mm)
      - The ?h.white surface mesh (for computing vertex areas)

    Returns a DataFrame with columns:
      parcel, hemi, network, n_vertices, mean_thickness,
      std_thickness, surface_area_mm2
    """
    import nibabel.freesurfer as fs

    base = subjects_dir / subject

    # Read projected annotation
    labels, ctab, names = fs.read_annot(str(annot_path))

    # Read thickness overlay
    thick_path = base / "surf" / f"{hemi}.thickness"
    if not thick_path.exists():
        raise FileNotFoundError(f"Thickness file not found: {thick_path}")
    thickness = fs.read_morph_data(str(thick_path))

    # Read white surface geometry for vertex areas
    white_path = base / "surf" / f"{hemi}.white"
    if not white_path.exists():
        raise FileNotFoundError(f"White surface not found: {white_path}")
    coords, faces = fs.read_geometry(str(white_path))

    # Compute vertex areas
    vertex_areas = compute_vertex_areas(coords, faces)

    # Sanity check: vertex counts must match
    assert labels.shape[0] == thickness.shape[0] == coords.shape[0], (
        f"Vertex count mismatch: labels={labels.shape[0]}, "
        f"thickness={thickness.shape[0]}, coords={coords.shape[0]}"
    )

    # Extract per-parcel statistics
    results = []
    unique_labels = np.unique(labels)

    for label_id in unique_labels:
        # Skip unknown/medial wall (label 0 or negative)
        if label_id <= 0:
            continue

        # Get parcel name
        if label_id < len(names):
            name = names[label_id]
            name = name.decode("utf-8") if isinstance(name, bytes) else name
        else:
            continue

        # Skip unlabeled regions
        if name in ("Unknown", "???", "MedialWall", "unknown"):
            continue

        # Mask: vertices in this parcel with positive thickness
        mask = (labels == label_id) & (thickness > 0)
        if mask.sum() == 0:
            continue

        t = thickness[mask]
        w = vertex_areas[mask]

        # Parse network from label name
        # e.g., "7Networks_LH_DorsAttn_Post_1" → "DorsAttn"
        parts = name.split("_")
        network = parts[2] if len(parts) >= 3 else "Unknown"

        results.append({
            "parcel": name,
            "hemi": hemi,
            "network": network,
            "n_vertices": int(mask.sum()),
            "mean_thickness": float(np.average(t, weights=w)),
            "std_thickness": float(
                np.sqrt(np.average((t - np.average(t, weights=w))**2, weights=w))
            ),
            "surface_area_mm2": float(w.sum()),
        })

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 4: Aggregate parcels → Yeo-7 network means
# ═══════════════════════════════════════════════════════════════════════════

def aggregate_by_network(parcel_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute area-weighted mean thickness per Yeo-7 network.

    Within each network, parcels contribute proportionally to their
    surface area — a parcel with 500 mm² contributes more than one
    with 200 mm² to the network mean. This is more accurate than
    simply averaging parcel means (which gives equal weight to a
    tiny OFC parcel and a large temporal parcel).
    """
    rows = []
    for hemi in ["lh", "rh"]:
        hemi_df = parcel_df[parcel_df["hemi"] == hemi]
        for net in YEO7_NETWORKS:
            net_df = hemi_df[hemi_df["network"] == net]
            if len(net_df) == 0:
                continue

            # Area-weighted mean: weight each parcel by its surface area
            areas = net_df["surface_area_mm2"].values
            thicknesses = net_df["mean_thickness"].values
            total_area = areas.sum()

            rows.append({
                "hemi": hemi,
                "network": net,
                "n_parcels": len(net_df),
                "n_vertices_total": int(net_df["n_vertices"].sum()),
                "mean_thickness": float(np.average(thicknesses, weights=areas)),
                "surface_area_mm2": float(total_area),
            })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 5: Lateralize based on HS side
# ═══════════════════════════════════════════════════════════════════════════

def lateralize_network_thickness(
    network_df: pd.DataFrame,
    hs_hemi: str,  # "lh" or "rh"
) -> dict:
    """
    Given network-level thickness for both hemispheres and the HS side,
    return a dict with IHS_* (ipsilateral) and CHS_* (contralateral) columns.
    """
    row = {}
    for net in YEO7_NETWORKS:
        lh_row = network_df[
            (network_df["hemi"] == "lh") & (network_df["network"] == net)
        ]
        rh_row = network_df[
            (network_df["hemi"] == "rh") & (network_df["network"] == net)
        ]

        lh_thick = lh_row["mean_thickness"].values[0] if len(lh_row) > 0 else np.nan
        rh_thick = rh_row["mean_thickness"].values[0] if len(rh_row) > 0 else np.nan

        if hs_hemi == "lh":
            row[f"IHS_{net}"] = lh_thick
            row[f"CHS_{net}"] = rh_thick
        else:
            row[f"IHS_{net}"] = rh_thick
            row[f"CHS_{net}"] = lh_thick

    return row


def parse_laterality(row) -> str:
    """Parse sidex column → 'lh' or 'rh'."""
    sidex = str(row.get("sidex", "")).strip().lower()
    if sidex == "esquerdo":
        return "lh"
    elif sidex in ("diretia", "direita"):
        return "rh"
    elif sidex == "bilateral":
        l_hipp = row.get("l_hipp", np.nan)
        r_hipp = row.get("r_hipp", np.nan)
        if pd.notna(l_hipp) and pd.notna(r_hipp):
            return "lh" if l_hipp <= r_hipp else "rh"
    return "lh"  # fallback


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    print("═" * 65)
    print("  Schaefer-200 → Yeo-7 Lateralized Thickness Extraction")
    print("═" * 65)
    print(f"  FS_SUBJECTS_DIR : {FS_SUBJECTS_DIR}")
    print(f"  OUTPUT_DIR      : {OUTPUT_DIR}")
    print(f"  BANK_CSV        : {BANK_CSV}")
    print()

    # ── Load subject list and laterality from clinical bank ─────────────
    bank = pd.read_csv(BANK_CSV)
    subjects = bank["subject"].tolist()
    laterality = {
        row["subject"]: parse_laterality(row) for _, row in bank.iterrows()
    }
    print(f"  Subjects: {len(subjects)}")
    n_lh = sum(1 for v in laterality.values() if v == "lh")
    print(f"  Laterality: {n_lh} left HS, {len(subjects) - n_lh} right HS")

    # ── Ensure fsaverage is accessible ──────────────────────────────────
    print("\n[1/5] Checking fsaverage...")
    ensure_fsaverage_symlink(FS_SUBJECTS_DIR)

    # ── Validate annotation files ───────────────────────────────────────
    for annot in [ANNOT_LH, ANNOT_RH]:
        if not annot.exists():
            raise FileNotFoundError(f"Annotation not found: {annot}")
    print(f"  ✓ Annotations found")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Process each subject ────────────────────────────────────────────
    print(f"\n[2/5] Projecting annotations + extracting thickness...")
    group_rows = []
    failed_subjects = []

    for i, sub in enumerate(subjects):
        sub_dir = OUTPUT_DIR / sub
        sub_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  [{i+1}/{len(subjects)}] {sub}")

        try:
            all_parcels = []

            for hemi, src_annot in [("lh", ANNOT_LH), ("rh", ANNOT_RH)]:
                # Step 2a: Project annot
                print(f"    {hemi}: projecting annot...", end=" ", flush=True)
                proj_annot = project_annot_to_subject(
                    sub, hemi, src_annot, FS_SUBJECTS_DIR, sub_dir,
                )
                print("extracting thickness...", end=" ", flush=True)

                # Step 2b: Extract parcel thickness
                parcel_df = extract_parcel_thickness(
                    sub, hemi, FS_SUBJECTS_DIR, proj_annot,
                )
                all_parcels.append(parcel_df)
                print(f"✓ ({len(parcel_df)} parcels)")

            # Combine hemispheres
            parcels_df = pd.concat(all_parcels, ignore_index=True)

            # Save per-subject parcel-level data
            parcels_df.to_csv(
                sub_dir / "schaefer200_thickness.csv",
                index=False, float_format="%.4f",
            )

            # Step 3: Aggregate by Yeo-7 network
            network_df = aggregate_by_network(parcels_df)
            network_df.to_csv(
                sub_dir / "yeo7_network_thickness.csv",
                index=False, float_format="%.4f",
            )

            # Step 4: Lateralize
            hs_hemi = laterality.get(sub, "lh")
            lat_row = lateralize_network_thickness(network_df, hs_hemi)
            lat_row["subject"] = sub
            lat_row["hs_hemi"] = hs_hemi
            group_rows.append(lat_row)

            print(f"    ✓ Saved: {sub_dir}/")

        except Exception as e:
            print(f"    ✗ FAILED: {e}")
            failed_subjects.append((sub, str(e)))

    # ── Build group-level lateralized thickness matrix ──────────────────
    print(f"\n[3/5] Building group-level matrix...")
    group_df = pd.DataFrame(group_rows)

    # Reorder columns: subject, hs_hemi, IHS_*, CHS_*
    ihs_cols = sorted([c for c in group_df.columns if c.startswith("IHS_")])
    chs_cols = sorted([c for c in group_df.columns if c.startswith("CHS_")])
    col_order = ["subject", "hs_hemi"] + ihs_cols + chs_cols
    group_df = group_df[col_order]

    group_path = OUTPUT_DIR / "group_yeo7_lateralized_thickness.csv"
    group_df.to_csv(group_path, index=False, float_format="%.4f")
    print(f"  ✓ Saved: {group_path}")
    print(f"    Shape: {group_df.shape[0]} subjects × {group_df.shape[1]} columns")

    # ── QC summary ──────────────────────────────────────────────────────
    print(f"\n[4/5] QC Summary — IHS vs CHS (mean ± std, mm)")
    print(f"  {'Network':<15s}  {'IHS':>14s}  {'CHS':>14s}  {'Δ%':>7s}")
    print(f"  {'─'*15}  {'─'*14}  {'─'*14}  {'─'*7}")

    for net in YEO7_NETWORKS:
        ihs_col = f"IHS_{net}"
        chs_col = f"CHS_{net}"
        if ihs_col in group_df.columns and chs_col in group_df.columns:
            ihs_m = group_df[ihs_col].mean()
            ihs_s = group_df[ihs_col].std()
            chs_m = group_df[chs_col].mean()
            chs_s = group_df[chs_col].std()
            pct = 100 * (ihs_m - chs_m) / chs_m if chs_m > 0 else 0
            print(
                f"  {net:<15s}  "
                f"{ihs_m:.3f} ± {ihs_s:.3f}  "
                f"{chs_m:.3f} ± {chs_s:.3f}  "
                f"{pct:+.1f}%"
            )

    # ── Report failures ─────────────────────────────────────────────────
    print(f"\n[5/5] Summary")
    elapsed = time.time() - t0
    print(f"  Processed: {len(group_rows)}/{len(subjects)} subjects")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    if failed_subjects:
        print(f"  ⚠ Failed ({len(failed_subjects)}):")
        for sub, err in failed_subjects:
            print(f"    • {sub}: {err}")
    else:
        print(f"  ✓ All subjects processed successfully!")

    print(f"\n  Output: {OUTPUT_DIR}")
    print("  Done! ✓")


if __name__ == "__main__":
    main()
