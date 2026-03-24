#!/usr/bin/env python3
"""
project_schaefer200_annot.py
==============================

Project Schaefer-200 (7-Networks) annotation files from fsaverage space
to individual subject space using FreeSurfer's mri_surf2surf.

This is a prerequisite for any parcel-level analysis in CorticalFields
(spectral descriptor aggregation, MSN/SSN, surprise map aggregation).

The .annot files in fsaverage space contain 163,842 vertex labels
(one per fsaverage vertex). mri_surf2surf uses the spherical registration
(sphere.reg) to map these labels to each subject's native mesh via
nearest-neighbor interpolation — appropriate for discrete labels
(unlike trilinear, which is for continuous overlays like thickness).

Requirements:
  - FreeSurfer 8.x (mri_surf2surf)
  - Completed surface reconstruction (FreeSurfer or FastSurfer)
    with sphere.reg for each subject
  - Schaefer-200 .annot files in fsaverage resolution (163,842 vertices)

Usage (Spyder):
  1. Set COHORT and paths in cell [1]
  2. Run all cells

Author: Velho Mago (rdneuro)
"""


# %% [0] IMPORTS
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

print("Imports OK ✓")


# %% [1] CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
# ┌─────────────────────────────────────────────────────────────────────┐
# │  Set the cohort and paths below. The script auto-discovers         │
# │  subjects and skips those already projected.                       │
# └─────────────────────────────────────────────────────────────────────┘

# Which cohort to process
COHORT = "ds000221"  # ← "sars", "cepesc", or "ds000221"

# FreeSurfer/FastSurfer output directories (subjects_dir)
# Each subject must have: surf/{lh,rh}.sphere.reg
FS_DIRS = {
    "sars": Path("/mnt/nvme1n1p1/sars_cov_2_project/data/output/structural/fs_subjects"),
    "cepesc": Path("/media/rdx/disk4/analysis/cepesc/structural/freesurfer/v741/corticalfields_hads/data/fs"),
    "ds000221": Path("/media/rdx/disk4/analysis/ds000221/output/fastsurfer"),
}

# Schaefer-200 .annot files in fsaverage space (163,842 vertices)
# These are the same files for all cohorts — they live in fsaverage space
ANNOT_DIR = Path("/media/rdx/disk4/analysis/cepesc/structural/freesurfer/v741/corticalfields_hads/info/annot/fsaverage")

# Output directories for projected annotations (per-subject .annot)
SCH_DIRS = {
    "sars": Path("/mnt/nvme1n1p1/sars_cov_2_project/data/proc/schaefer_annot/sarscov2"),
    "cepesc": Path("/mnt/nvme1n1p1/sars_cov_2_project/data/proc/schaefer_annot/cepesc"),
    "ds000221": Path("/mnt/nvme1n1p1/sars_cov_2_project/data/proc/schaefer_annot/ds000221"),
}

# FreeSurfer home (for fsaverage and mri_surf2surf)
FREESURFER_HOME = Path(
    os.environ.get("FREESURFER_HOME", "/usr/local/freesurfer/8.1.0")
)

# Annotation name
ANNOT_NAME = "Schaefer2018_200Parcels_7Networks_order"

# ── Resolve paths for selected cohort ──────────────────────────────────
FS_DIR = FS_DIRS[COHORT]
OUTPUT_DIR = SCH_DIRS[COHORT]

ANNOT_LH = ANNOT_DIR / f"lh.{ANNOT_NAME}.annot"
ANNOT_RH = ANNOT_DIR / f"rh.{ANNOT_NAME}.annot"

print(f"Configuration ✓")
print(f"  Cohort       : {COHORT}")
print(f"  FS_DIR       : {FS_DIR}")
print(f"  OUTPUT_DIR   : {OUTPUT_DIR}")
print(f"  ANNOT_LH     : {ANNOT_LH}")
print(f"  ANNOT_RH     : {ANNOT_RH}")

# Validate that annotation files exist
for annot in [ANNOT_LH, ANNOT_RH]:
    if not annot.exists():
        raise FileNotFoundError(f"Annotation not found: {annot}")
print(f"  ✓ Both .annot files found (fsaverage space)")


# %% [2] DISCOVER SUBJECTS AND ENSURE FSAVERAGE
# ═══════════════════════════════════════════════════════════════════════════

# Ensure fsaverage symlink exists in FS_DIR (required by mri_surf2surf)
fsavg_link = FS_DIR / "fsaverage"
if not fsavg_link.exists():
    fsavg_target = FREESURFER_HOME / "subjects" / "fsaverage"
    if not fsavg_target.exists():
        raise FileNotFoundError(f"fsaverage not found at {fsavg_target}")
    fsavg_link.symlink_to(fsavg_target)
    print(f"  Created symlink: {fsavg_link} → {fsavg_target}")
else:
    print(f"  ✓ fsaverage found in FS_DIR")

# Discover subjects with sphere.reg (required for annotation projection)
subjects = []
for d in sorted(FS_DIR.iterdir()):
    if d.is_dir() and not d.name.startswith(".") and d.name != "fsaverage":
        if (d / "surf" / "lh.sphere.reg").exists():
            subjects.append(d.name)

print(f"\n  Found {len(subjects)} subjects with sphere.reg")
for s in subjects[:5]:
    print(f"    {s}")
if len(subjects) > 5:
    print(f"    ... and {len(subjects) - 5} more")


# %% [3] PROJECT ANNOTATIONS
# ═══════════════════════════════════════════════════════════════════════════
# For each subject × hemisphere, run mri_surf2surf to project the
# Schaefer-200 annotation from fsaverage → individual subject space.
#
# mri_surf2surf uses the spherical registration (sphere.reg) to find
# the correspondence between fsaverage vertices and subject vertices,
# then assigns the nearest fsaverage label to each subject vertex.

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

env = os.environ.copy()
env["SUBJECTS_DIR"] = str(FS_DIR)

t0 = time.time()
projected = 0
skipped = 0
failed = []

for i, sub in enumerate(subjects):
    sub_out = OUTPUT_DIR / sub
    sub_out.mkdir(parents=True, exist_ok=True)

    print(f"[{i+1}/{len(subjects)}] {sub}", end=" ", flush=True)

    for hemi, src_annot in [("lh", ANNOT_LH), ("rh", ANNOT_RH)]:
        out_annot = sub_out / f"{hemi}.{ANNOT_NAME}.annot"

        # Skip if already projected
        if out_annot.exists():
            skipped += 1
            continue

        result = subprocess.run([
            "mri_surf2surf",
            "--hemi", hemi,
            "--srcsubject", "fsaverage",
            "--trgsubject", sub,
            "--sval-annot", str(src_annot),
            "--tval", str(out_annot),
        ], capture_output=True, text=True, env=env)

        if result.returncode != 0:
            print(f"⚠{hemi}", end=" ")
            failed.append((sub, hemi, result.stderr.strip()[:100]))
        else:
            projected += 1

    print("✓")

elapsed = time.time() - t0


# %% [4] SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"  Schaefer-200 Annotation Projection — {COHORT}")
print(f"{'='*60}")
print(f"  Subjects    : {len(subjects)}")
print(f"  Projected   : {projected} hemispheres")
print(f"  Skipped     : {skipped} (already existed)")
print(f"  Failed      : {len(failed)}")
print(f"  Time        : {elapsed:.0f}s")
print(f"  Output      : {OUTPUT_DIR}")

if failed:
    print(f"\n  ⚠ Failures:")
    for sub, hemi, err in failed:
        print(f"    {sub} {hemi}: {err}")

# Quick validation: check a random subject
if subjects:
    check_sub = subjects[0]
    for hemi in ["lh", "rh"]:
        p = OUTPUT_DIR / check_sub / f"{hemi}.{ANNOT_NAME}.annot"
        if p.exists():
            import nibabel.freesurfer as fs_io
            labels, ctab, names = fs_io.read_annot(str(p))
            names = [n.decode() if isinstance(n, bytes) else n for n in names]
            n_valid = (labels > 0).sum()
            print(f"\n  QC ({check_sub} {hemi}):")
            print(f"    Total vertices : {len(labels)}")
            print(f"    Labeled        : {n_valid} ({100*n_valid/len(labels):.1f}%)")
            print(f"    Parcels        : {len(set(labels[labels > 0]))}")
            print(f"    Networks       : {sorted(set(n.split('_')[2] for n in names if len(n.split('_')) >= 3 and n not in ('Unknown','???','MedialWall')))}")

print("\nDone! ✓")
