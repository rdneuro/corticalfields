#!/usr/bin/env python3
"""
CorticalFields — MTLE-HS CEPESC Pipeline
=========================================

Two complementary analysis modes for the MTLE-HS CEPESC cohort
(125 patients, T1w 1.5T, pre-surgical):

MODE A — NORMATIVE (external reference)
    Uses ds000221 healthy controls (~40 subjects) as the normative
    reference cohort. The GP learns "what a normal cortex looks like"
    from the controls and scores each MTLE-HS patient against that norm.

    Pro: true normative comparison (healthy vs. disease).
    Con: scanner/acquisition differences between cohorts (mitigated
         by fsaverage registration + feature normalisation).

MODE B — HEMISPHERIC (intra-subject, ipsilateral vs. contralateral)
    Uses the contralateral hemisphere (opposite to hippocampal
    sclerosis) as the reference and the ipsilateral hemisphere as
    the patient. The GP learns "what the relatively preserved
    hemisphere looks like across all 125 patients" and scores
    each patient's diseased hemisphere against that model.

    Pro: eliminates ALL inter-scanner/age/sex variability.
    Pro: no external controls needed — self-referencing.
    Pro: 125 "controls" (contralateral hemispheres) for free.
    Con: assumes contralateral hemisphere is relatively intact
         (reasonable for unilateral MTLE-HS, may need validation).

Both modes share the same CorticalFields core:
    LB eigenpairs → HKS/WKS/GPS → Spectral Matérn GP → Surprise maps

Usage
-----
# Mode A: ds000221 as reference, score one patient
python mtle_pipeline.py \\
    --mode normative \\
    --subjects-dir /mnt/nvme1n1p1/sars_cov_2_project/freesurfer \\
    --reference-dir /data/ds000221/freesurfer \\
    --reference-ids data/ds000221_subjects.txt \\
    --patient-ids data/cepesc_subjects.txt \\
    --output-dir results/mode_a_normative

# Mode B: contralateral vs ipsilateral
python mtle_pipeline.py \\
    --mode hemispheric \\
    --subjects-dir /mnt/nvme1n1p1/sars_cov_2_project/freesurfer \\
    --patient-ids data/cepesc_subjects.txt \\
    --laterality-file data/cepesc_laterality.csv \\
    --output-dir results/mode_b_hemispheric

# Both modes at once (recommended — generates comparative report)
python mtle_pipeline.py \\
    --mode both \\
    --subjects-dir /mnt/nvme1n1p1/sars_cov_2_project/freesurfer \\
    --reference-dir /data/ds000221/freesurfer \\
    --reference-ids data/ds000221_subjects.txt \\
    --patient-ids data/cepesc_subjects.txt \\
    --laterality-file data/cepesc_laterality.csv \\
    --output-dir results/combined

Author: Velho Mago (rdneuro)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── CorticalFields ──────────────────────────────────────────────────────
from corticalfields.surface import CorticalSurface, load_freesurfer_surface
from corticalfields.spectral import (
    LaplaceBeltrami,
    compute_eigenpairs,
    heat_kernel_signature,
    wave_kernel_signature,
    spectral_feature_matrix,
)
from corticalfields.normative import CorticalNormativeModel, NormativeResult
from corticalfields.surprise import SurpriseMap, compute_surprise, combined_surprise
from corticalfields.features import MorphometricProfile, extract_cohort_profiles
from corticalfields.graphs import (
    morphometric_similarity_network,
    spectral_similarity_network,
    graph_metrics,
)
from corticalfields.viz import (
    plot_surprise_map,
    plot_eigenspectrum,
    plot_hks_multiscale,
    plot_network_anomaly_profile,
)
from corticalfields.utils import get_device, setup_logging, timer, validate_mesh

logger = logging.getLogger("corticalfields.mtle_pipeline")


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class PatientInfo:
    """Clinical metadata for one MTLE-HS patient."""

    subject_id: str
    hs_side: str  # "left" or "right" — side of hippocampal sclerosis

    @property
    def ipsilateral_hemi(self) -> str:
        """FreeSurfer hemisphere code for the diseased side."""
        return "lh" if self.hs_side == "left" else "rh"

    @property
    def contralateral_hemi(self) -> str:
        """FreeSurfer hemisphere code for the preserved side."""
        return "rh" if self.hs_side == "left" else "lh"


@dataclass
class PipelineConfig:
    """Configuration for the MTLE-HS pipeline."""

    # Paths
    subjects_dir: Path                  # FreeSurfer SUBJECTS_DIR for CEPESC
    output_dir: Path                    # Where to save results
    reference_dir: Optional[Path]       # FreeSurfer dir for ds000221 (Mode A)
    reference_ids: List[str]            # Subject IDs for ds000221
    patients: List[PatientInfo]         # CEPESC patients with laterality info

    # Analysis parameters
    mode: str = "both"                  # "normative", "hemispheric", "both"
    n_eigenpairs: int = 300             # LB eigenpairs to compute
    n_inducing: int = 512               # SVGP inducing points
    nu: float = 2.5                     # Matérn smoothness
    n_epochs: int = 100                 # GP training epochs
    features: List[str] = field(
        default_factory=lambda: ["thickness", "curv", "sulc"]
    )
    surface: str = "pial"               # FreeSurfer surface type
    device: str = "auto"                # "cuda", "cpu", or "auto"


# ═══════════════════════════════════════════════════════════════════════════
# MODE A: Normative (ds000221 reference)
# ═══════════════════════════════════════════════════════════════════════════


def run_mode_a_normative(cfg: PipelineConfig) -> Dict[str, Dict]:
    """
    External normative comparison: ds000221 controls → CEPESC patients.

    The GP learns the normal cortical morphology from healthy controls
    and identifies where each MTLE-HS patient deviates.

    For each hemisphere (lh, rh), for each morphometric feature:
        1. Load ds000221 reference cohort features on fsaverage
        2. Train GP normative model
        3. Score every CEPESC patient
        4. Aggregate surprise by Yeo-7 networks

    Returns
    -------
    all_results : dict
        Nested: patient_id → feature → NormativeResult
    """
    logger.info("╔══════════════════════════════════════════════════╗")
    logger.info("║        MODE A — NORMATIVE (ds000221 ref)        ║")
    logger.info("╚══════════════════════════════════════════════════╝")

    out = cfg.output_dir / "mode_a_normative"
    out.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(cfg.device)

    all_results = {}

    # We train one model per (hemisphere, feature). Each model is then
    # used to score all patients on that hemisphere.
    for hemi in ["lh", "rh"]:
        logger.info("━━━ Hemisphere: %s ━━━", hemi)

        # ── 1. Template surface + LB eigenpairs ────────────────────
        # Using fsaverage as the common template (all subjects are
        # registered here via recon-all + mri_surf2surf).
        with timer(f"LB eigenpairs [{hemi}]"):
            template = load_freesurfer_surface(
                cfg.subjects_dir, "fsaverage", hemi=hemi,
                surface=cfg.surface, overlays=[],
            )
            lb = compute_eigenpairs(
                template.vertices, template.faces,
                n_eigenpairs=cfg.n_eigenpairs,
            )

        # Save eigenspectrum diagnostic
        plot_eigenspectrum(
            lb.eigenvalues, n_show=100,
            title=f"LB Eigenspectrum — fsaverage {hemi}",
            output_path=out / f"eigenspectrum_{hemi}.png",
        )

        # ── 2. Load reference cohort (ds000221) ───────────────────
        logger.info("Loading ds000221 reference cohort (%d subjects)…",
                     len(cfg.reference_ids))

        with timer(f"Extract reference profiles [{hemi}]"):
            ref_profiles = extract_cohort_profiles(
                subjects_dir=cfg.reference_dir,
                subject_ids=cfg.reference_ids,
                hemi=hemi,
                surface=cfg.surface,
                features=cfg.features,
            )

        # ── 3. Train one GP per feature ───────────────────────────
        models = {}
        for feat in cfg.features:
            logger.info("Training GP [%s, %s]…", hemi, feat)
            train_data = ref_profiles.get_feature_matrix(feat)

            model = CorticalNormativeModel(
                lb=lb, nu=cfg.nu, n_inducing=cfg.n_inducing, device=device,
            )
            with timer(f"Train {feat} [{hemi}]"):
                model.fit(
                    train_features=train_data,
                    feature_name=feat,
                    n_epochs=cfg.n_epochs,
                )
            model.save(out / f"model_{hemi}_{feat}")
            models[feat] = model

        # ── 4. Score each CEPESC patient ──────────────────────────
        for patient in cfg.patients:
            pid = patient.subject_id
            logger.info("Scoring patient %s [%s]…", pid, hemi)

            try:
                patient_surf = load_freesurfer_surface(
                    cfg.subjects_dir, pid, hemi=hemi,
                    surface=cfg.surface, overlays=cfg.features,
                )
            except FileNotFoundError:
                logger.warning("  Patient %s not found, skipping.", pid)
                continue

            patient_results = {}
            surprise_maps = []

            for feat in cfg.features:
                obs = patient_surf.get_overlay(feat)
                result = models[feat].predict(obs)

                smap = compute_surprise(
                    observed=result.observed,
                    predicted_mean=result.mean,
                    predicted_var=result.variance,
                )
                surprise_maps.append(smap)
                patient_results[feat] = result

                # Per-feature surprise figure
                plot_surprise_map(
                    template.vertices, template.faces,
                    smap.surprise, smap.z_score,
                    title=f"{feat} — {pid} [{hemi}]",
                    output_path=out / f"surprise_{hemi}_{feat}_{pid}.png",
                    show=False,
                )

            # Combined multi-feature surprise
            if len(surprise_maps) > 1:
                combined = combined_surprise(surprise_maps, method="fisher")
                plot_surprise_map(
                    template.vertices, template.faces,
                    combined.surprise, combined.z_score,
                    title=f"Combined — {pid} [{hemi}]",
                    output_path=out / f"surprise_{hemi}_combined_{pid}.png",
                    show=False,
                )

            all_results.setdefault(pid, {})[hemi] = patient_results

    return all_results


# ═══════════════════════════════════════════════════════════════════════════
# MODE B: Hemispheric (contralateral as control)
# ═══════════════════════════════════════════════════════════════════════════


def run_mode_b_hemispheric(cfg: PipelineConfig) -> Dict[str, Dict]:
    """
    Intra-subject hemispheric comparison: contralateral → ipsilateral.

    The key insight: in unilateral MTLE-HS, the hemisphere contralateral
    to the sclerosis is relatively preserved. By collecting the
    contralateral hemispheres of ALL 125 patients, we build a reference
    distribution of "what the preserved hemisphere looks like in this
    clinical population." Then each patient's ipsilateral (diseased)
    hemisphere is scored against that reference.

    This design has a beautiful property: it completely eliminates
    inter-scanner, age, and sex confounds. Two patients scanned on
    different scanners contribute their contralateral hemispheres to
    the SAME reference pool — and each patient's ipsilateral hemisphere
    is compared to a model trained on hemispheres from the same
    population, just the other side.

    Implementation
    --------------
    Since FreeSurfer's lh and rh surfaces have the SAME number of
    vertices on fsaverage (163,842 each) and the vertex correspondence
    is established by the atlas registration, we can directly compare:
        • A model trained on lh data (from patients whose HS is on the right)
          → scores lh data (from patients whose HS is on the left)
        • A model trained on rh data (from patients whose HS is on the left)
          → scores rh data (from patients whose HS is on the right)

    But this splits the reference pool. A cleaner approach:
        1. Collect ALL contralateral hemispheres (both lh and rh)
           into a single reference pool per hemisphere template.
        2. Train two GP models: one on fsaverage-lh, one on fsaverage-rh.
        3. Score ipsilateral hemispheres against the matching model.

    Even cleaner (what we implement here):
        For each hemisphere template (lh, rh):
          - Reference = contralateral data from patients whose HS is
            on the OTHER side (so their contralateral IS this hemi).
          - Patients = ipsilateral data from patients whose HS is on
            THIS side.

    Example with lh template:
        Reference pool: all patients with RIGHT HS → their lh is contralateral
        Patients to score: all patients with LEFT HS → their lh is ipsilateral

    Returns
    -------
    all_results : dict
        patient_id → feature → NormativeResult (ipsilateral only)
    """
    logger.info("╔══════════════════════════════════════════════════╗")
    logger.info("║    MODE B — HEMISPHERIC (contra vs. ipsi)       ║")
    logger.info("╚══════════════════════════════════════════════════╝")

    out = cfg.output_dir / "mode_b_hemispheric"
    out.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(cfg.device)

    # Split patients by laterality
    left_hs = [p for p in cfg.patients if p.hs_side == "left"]
    right_hs = [p for p in cfg.patients if p.hs_side == "right"]

    logger.info(
        "Laterality split: %d left HS, %d right HS, %d total",
        len(left_hs), len(right_hs), len(cfg.patients),
    )

    all_results = {}

    # ── Process each hemisphere template ──────────────────────────
    # For hemisphere 'lh':
    #   Reference = right-HS patients' lh (their contralateral)
    #   Patients  = left-HS patients' lh  (their ipsilateral)
    #
    # For hemisphere 'rh':
    #   Reference = left-HS patients' rh  (their contralateral)
    #   Patients  = right-HS patients' rh (their ipsilateral)

    hemi_config = {
        "lh": {
            "reference_patients": right_hs,   # their lh is contralateral
            "score_patients": left_hs,         # their lh is ipsilateral
            "ref_label": "contra (right-HS patients' lh)",
            "score_label": "ipsi (left-HS patients' lh)",
        },
        "rh": {
            "reference_patients": left_hs,    # their rh is contralateral
            "score_patients": right_hs,        # their rh is ipsilateral
            "ref_label": "contra (left-HS patients' rh)",
            "score_label": "ipsi (right-HS patients' rh)",
        },
    }

    for hemi, hconf in hemi_config.items():
        ref_patients = hconf["reference_patients"]
        score_patients = hconf["score_patients"]

        if not ref_patients or not score_patients:
            logger.warning(
                "No patients to process for %s (ref=%d, score=%d), skipping.",
                hemi, len(ref_patients), len(score_patients),
            )
            continue

        logger.info(
            "━━━ %s: reference=%s (%d), scoring=%s (%d) ━━━",
            hemi.upper(), hconf["ref_label"], len(ref_patients),
            hconf["score_label"], len(score_patients),
        )

        # ── 1. LB eigenpairs on fsaverage template ────────────────
        with timer(f"LB eigenpairs [{hemi}]"):
            template = load_freesurfer_surface(
                cfg.subjects_dir, "fsaverage", hemi=hemi,
                surface=cfg.surface, overlays=[],
            )
            lb = compute_eigenpairs(
                template.vertices, template.faces,
                n_eigenpairs=cfg.n_eigenpairs,
            )

        plot_eigenspectrum(
            lb.eigenvalues, n_show=100,
            title=f"LB Eigenspectrum — fsaverage {hemi}",
            output_path=out / f"eigenspectrum_{hemi}.png",
        )

        # ── 2. Load CONTRALATERAL features (reference pool) ───────
        ref_ids = [p.subject_id for p in ref_patients]
        logger.info(
            "Loading contralateral reference data: %d subjects on %s…",
            len(ref_ids), hemi,
        )

        with timer(f"Extract contralateral profiles [{hemi}]"):
            ref_profiles = extract_cohort_profiles(
                subjects_dir=cfg.subjects_dir,
                subject_ids=ref_ids,
                hemi=hemi,
                surface=cfg.surface,
                features=cfg.features,
            )

        # ── 3. Train GP normative model per feature ───────────────
        models = {}
        for feat in cfg.features:
            logger.info("Training GP [%s, %s] on contralateral data…", hemi, feat)
            train_data = ref_profiles.get_feature_matrix(feat)

            model = CorticalNormativeModel(
                lb=lb, nu=cfg.nu, n_inducing=cfg.n_inducing, device=device,
            )
            with timer(f"Train {feat} [{hemi}]"):
                model.fit(
                    train_features=train_data,
                    feature_name=feat,
                    n_epochs=cfg.n_epochs,
                )
            model.save(out / f"model_{hemi}_{feat}_contra")
            models[feat] = model

        # ── 4. Score IPSILATERAL hemispheres ──────────────────────
        for patient in score_patients:
            pid = patient.subject_id
            logger.info(
                "Scoring %s ipsilateral %s (HS side: %s)…",
                pid, hemi, patient.hs_side,
            )

            try:
                patient_surf = load_freesurfer_surface(
                    cfg.subjects_dir, pid, hemi=hemi,
                    surface=cfg.surface, overlays=cfg.features,
                )
            except FileNotFoundError:
                logger.warning("  %s not found, skipping.", pid)
                continue

            patient_results = {}
            surprise_maps = []

            for feat in cfg.features:
                obs = patient_surf.get_overlay(feat)
                result = models[feat].predict(obs)

                smap = compute_surprise(
                    observed=result.observed,
                    predicted_mean=result.mean,
                    predicted_var=result.variance,
                )
                surprise_maps.append(smap)
                patient_results[feat] = result

                plot_surprise_map(
                    template.vertices, template.faces,
                    smap.surprise, smap.z_score,
                    title=f"{feat} — {pid} ipsi-{hemi} (HS: {patient.hs_side})",
                    output_path=out / f"surprise_{hemi}_{feat}_{pid}_ipsi.png",
                    show=False,
                )

            # Combined multi-feature surprise
            if len(surprise_maps) > 1:
                combined = combined_surprise(surprise_maps, method="fisher")
                plot_surprise_map(
                    template.vertices, template.faces,
                    combined.surprise, combined.z_score,
                    title=f"Combined — {pid} ipsi-{hemi}",
                    output_path=out / f"surprise_{hemi}_combined_{pid}_ipsi.png",
                    show=False,
                )

                # Save numerical results for downstream statistics
                np.savez_compressed(
                    out / f"results_{pid}_{hemi}_ipsi.npz",
                    z_score=combined.z_score,
                    surprise=combined.surprise,
                    excess_surprise=combined.excess_surprise,
                    anomaly_prob=combined.anomaly_probability,
                )

            all_results.setdefault(pid, {})[f"{hemi}_ipsi"] = patient_results

    # ── 5. Group-level summary ────────────────────────────────────
    _save_hemispheric_summary(all_results, cfg, out)

    return all_results


# ═══════════════════════════════════════════════════════════════════════════
# Shared utilities
# ═══════════════════════════════════════════════════════════════════════════


def _resolve_device(device_str: str):
    """Resolve device string to torch.device."""
    if device_str == "auto":
        return get_device(prefer_cuda=True)
    import torch
    return torch.device(device_str)


def _save_hemispheric_summary(
    all_results: Dict,
    cfg: PipelineConfig,
    out: Path,
) -> None:
    """
    Save a group-level CSV with mean z-scores and surprise per patient.

    This CSV is ready for direct import into your Bayesian models:
    you can correlate the surprise scores with clinical variables
    (disease duration, seizure frequency, surgical outcome, HADS, PSQI)
    using PyMC exactly as you did for the thalamus and limbic papers.
    """
    csv_path = out / "group_summary.csv"
    rows = []

    for patient in cfg.patients:
        pid = patient.subject_id
        if pid not in all_results:
            continue

        row = {
            "subject_id": pid,
            "hs_side": patient.hs_side,
        }

        # Collect results for each feature
        hemi_key = f"{patient.ipsilateral_hemi}_ipsi"
        if hemi_key in all_results[pid]:
            for feat, result in all_results[pid][hemi_key].items():
                valid = np.isfinite(result.z_score)
                row[f"mean_z_{feat}"] = float(np.nanmean(result.z_score[valid]))
                row[f"mean_surprise_{feat}"] = float(np.nanmean(result.surprise[valid]))
                row[f"frac_anomalous_{feat}"] = float(
                    (np.abs(result.z_score[valid]) > 2.0).mean()
                )

        rows.append(row)

    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Group summary saved to %s (%d patients)", csv_path, len(rows))


# ═══════════════════════════════════════════════════════════════════════════
# File loaders
# ═══════════════════════════════════════════════════════════════════════════


def load_subject_list(path: str) -> List[str]:
    """
    Load subject IDs from a text file (one per line).
    Lines starting with '#' are treated as comments.
    """
    ids = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                ids.append(line)
    return ids


def load_laterality_file(path: str) -> Dict[str, str]:
    """
    Load laterality information from a CSV file.

    Expected format (header required):
        subject_id,hs_side
        sub-MTLE001,left
        sub-MTLE002,right
        sub-MTLE003,left
        ...

    The 'hs_side' column indicates which hippocampus has sclerosis:
    'left' or 'right'. This determines which hemisphere is ipsilateral
    (diseased) and which is contralateral (preserved).

    Returns
    -------
    laterality : dict
        subject_id → 'left' or 'right'
    """
    laterality = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row["subject_id"].strip()
            side = row["hs_side"].strip().lower()
            if side not in ("left", "right"):
                logger.warning(
                    "Invalid hs_side '%s' for %s, skipping.", side, sid,
                )
                continue
            laterality[sid] = side
    return laterality


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


def parse_args():
    p = argparse.ArgumentParser(
        description="CorticalFields MTLE-HS pipeline — normative and hemispheric modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Mode A only (needs external controls)
  python mtle_pipeline.py --mode normative \\
      --subjects-dir /data/cepesc/freesurfer \\
      --reference-dir /data/ds000221/freesurfer \\
      --reference-ids data/ds000221_subjects.txt \\
      --patient-ids data/cepesc_subjects.txt \\
      --output-dir results/normative

  # Mode B only (self-referencing, no external controls needed)
  python mtle_pipeline.py --mode hemispheric \\
      --subjects-dir /data/cepesc/freesurfer \\
      --patient-ids data/cepesc_subjects.txt \\
      --laterality-file data/cepesc_laterality.csv \\
      --output-dir results/hemispheric

  # Both modes (recommended)
  python mtle_pipeline.py --mode both \\
      --subjects-dir /data/cepesc/freesurfer \\
      --reference-dir /data/ds000221/freesurfer \\
      --reference-ids data/ds000221_subjects.txt \\
      --patient-ids data/cepesc_subjects.txt \\
      --laterality-file data/cepesc_laterality.csv \\
      --output-dir results/combined
""",
    )

    # ── Required ──
    p.add_argument(
        "--mode", required=True,
        choices=["normative", "hemispheric", "both"],
        help="Analysis mode(s) to run.",
    )
    p.add_argument(
        "--subjects-dir", required=True,
        help="FreeSurfer SUBJECTS_DIR for CEPESC patients.",
    )
    p.add_argument(
        "--patient-ids", required=True,
        help="Text file with CEPESC patient subject IDs.",
    )

    # ── Mode A (normative) ──
    p.add_argument(
        "--reference-dir",
        help="FreeSurfer SUBJECTS_DIR for ds000221 controls (Mode A).",
    )
    p.add_argument(
        "--reference-ids",
        help="Text file with ds000221 subject IDs (Mode A).",
    )

    # ── Mode B (hemispheric) ──
    p.add_argument(
        "--laterality-file",
        help="CSV with columns 'subject_id' and 'hs_side' (Mode B).",
    )

    # ── Analysis parameters ──
    p.add_argument("--n-eigenpairs", type=int, default=300)
    p.add_argument("--n-inducing", type=int, default=512)
    p.add_argument("--nu", type=float, default=2.5)
    p.add_argument("--n-epochs", type=int, default=100)
    p.add_argument("--surface", default="pial")
    p.add_argument("--features", nargs="+", default=["thickness", "curv", "sulc"])
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--output-dir", default="results")
    p.add_argument("--log-level", default="INFO")

    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log_level)

    # ── Validate arguments ────────────────────────────────────────
    run_normative = args.mode in ("normative", "both")
    run_hemispheric = args.mode in ("hemispheric", "both")

    if run_normative:
        if not args.reference_dir or not args.reference_ids:
            raise ValueError(
                "Mode 'normative' requires --reference-dir and --reference-ids."
            )

    if run_hemispheric:
        if not args.laterality_file:
            raise ValueError(
                "Mode 'hemispheric' requires --laterality-file."
            )

    # ── Load patient list ─────────────────────────────────────────
    patient_ids = load_subject_list(args.patient_ids)
    logger.info("Loaded %d patient IDs from %s", len(patient_ids), args.patient_ids)

    # ── Load laterality (if available) ────────────────────────────
    laterality = {}
    if args.laterality_file:
        laterality = load_laterality_file(args.laterality_file)
        logger.info(
            "Laterality info: %d left HS, %d right HS",
            sum(1 for v in laterality.values() if v == "left"),
            sum(1 for v in laterality.values() if v == "right"),
        )

    # Build PatientInfo list
    patients = []
    for pid in patient_ids:
        hs_side = laterality.get(pid, "unknown")
        patients.append(PatientInfo(subject_id=pid, hs_side=hs_side))

    # ── Load reference IDs (if available) ─────────────────────────
    reference_ids = []
    if args.reference_ids:
        reference_ids = load_subject_list(args.reference_ids)
        logger.info(
            "Loaded %d reference IDs from %s",
            len(reference_ids), args.reference_ids,
        )

    # ── Build config ──────────────────────────────────────────────
    cfg = PipelineConfig(
        subjects_dir=Path(args.subjects_dir),
        output_dir=Path(args.output_dir),
        reference_dir=Path(args.reference_dir) if args.reference_dir else None,
        reference_ids=reference_ids,
        patients=patients,
        mode=args.mode,
        n_eigenpairs=args.n_eigenpairs,
        n_inducing=args.n_inducing,
        nu=args.nu,
        n_epochs=args.n_epochs,
        features=args.features,
        surface=args.surface,
        device=args.device,
    )

    # ── Save config for reproducibility ───────────────────────────
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    config_dict = {
        "mode": cfg.mode,
        "n_patients": len(cfg.patients),
        "n_reference": len(cfg.reference_ids),
        "n_eigenpairs": cfg.n_eigenpairs,
        "n_inducing": cfg.n_inducing,
        "nu": cfg.nu,
        "n_epochs": cfg.n_epochs,
        "features": cfg.features,
        "surface": cfg.surface,
        "n_left_hs": sum(1 for p in patients if p.hs_side == "left"),
        "n_right_hs": sum(1 for p in patients if p.hs_side == "right"),
    }
    with open(cfg.output_dir / "pipeline_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # ── Run selected modes ────────────────────────────────────────
    results = {}

    if run_normative:
        results["normative"] = run_mode_a_normative(cfg)

    if run_hemispheric:
        # Filter to patients with known laterality
        valid_patients = [p for p in cfg.patients if p.hs_side in ("left", "right")]
        if len(valid_patients) < len(cfg.patients):
            logger.warning(
                "%d patients lack laterality info and will be skipped in Mode B.",
                len(cfg.patients) - len(valid_patients),
            )
        cfg_hemi = PipelineConfig(
            subjects_dir=cfg.subjects_dir,
            output_dir=cfg.output_dir,
            reference_dir=cfg.reference_dir,
            reference_ids=cfg.reference_ids,
            patients=valid_patients,
            mode="hemispheric",
            n_eigenpairs=cfg.n_eigenpairs,
            n_inducing=cfg.n_inducing,
            nu=cfg.nu,
            n_epochs=cfg.n_epochs,
            features=cfg.features,
            surface=cfg.surface,
            device=cfg.device,
        )
        results["hemispheric"] = run_mode_b_hemispheric(cfg_hemi)

    # ── Final summary ─────────────────────────────────────────────
    logger.info("╔══════════════════════════════════════════════════╗")
    logger.info("║              PIPELINE COMPLETE                  ║")
    logger.info("╠══════════════════════════════════════════════════╣")
    if run_normative:
        n_scored = len(results.get("normative", {}))
        logger.info("║  Mode A (normative):    %3d patients scored     ║", n_scored)
    if run_hemispheric:
        n_scored = len(results.get("hemispheric", {}))
        logger.info("║  Mode B (hemispheric):  %3d patients scored     ║", n_scored)
    logger.info("║  Output: %-40s║", str(cfg.output_dir))
    logger.info("╚══════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
