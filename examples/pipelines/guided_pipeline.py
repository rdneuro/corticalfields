#!/usr/bin/env python3
"""
CorticalFields — Full Pipeline Example
=======================================

This script demonstrates the complete CorticalFields pipeline:
    T1w FreeSurfer output → LB spectral decomposition → HKS/WKS/GPS →
    GP normative model → surprise/anomaly maps → network aggregation.

Designed for the MTLE-HS CEPESC cohort (125 patients, T1w 1.5T)
but adaptable to any FreeSurfer-processed dataset.

Usage:
    python full_pipeline.py \
        --subjects-dir /data/freesurfer \
        --reference-ids ref_subjects.txt \
        --patient-id sub-MTLE001 \
        --hemi lh \
        --output-dir results/

Author: Velho Mago (rdneuro)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

# ── CorticalFields imports ──────────────────────────────────────────────
import corticalfields as cf
from corticalfields.spectral import (
    compute_eigenpairs,
    heat_kernel_signature,
    wave_kernel_signature,
    global_point_signature,
    spectral_feature_matrix,
)
from corticalfields.features import extract_cohort_profiles
from corticalfields.surprise import compute_surprise, combined_surprise
from corticalfields.graphs import (
    morphometric_similarity_network,
    spectral_similarity_network,
    graph_metrics,
)
from corticalfields.viz import (
    plot_surface_scalar,
    plot_surprise_map,
    plot_eigenspectrum,
    plot_hks_multiscale,
    plot_network_anomaly_profile,
)
from corticalfields.utils import (
    get_device,
    setup_logging,
    timer,
    validate_mesh,
)


def main():
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger("corticalfields.pipeline")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(prefer_cuda=True)

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 1: Load template surface and compute spectral decomposition
    # ══════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("PHASE 1: Spectral decomposition of template surface")
    logger.info("=" * 60)

    # Load the fsaverage template (all subjects are registered to this)
    with timer("Load template surface"):
        template = cf.load_freesurfer_surface(
            subjects_dir=args.subjects_dir,
            subject_id="fsaverage",
            hemi=args.hemi,
            surface="pial",
            overlays=["thickness", "curv", "sulc", "area"],
        )

    # Validate mesh quality
    mesh_report = validate_mesh(template.vertices, template.faces)
    logger.info(
        "Template mesh: %d vertices, %d faces, Euler χ = %d, "
        "area = %.0f mm²",
        mesh_report["n_vertices"],
        mesh_report["n_faces"],
        mesh_report["euler_characteristic"],
        mesh_report["total_area"],
    )

    # Compute Laplace–Beltrami eigenpairs
    # 300 eigenpairs captures structure from global lobes (~λ₁) down to
    # small gyral folds (~λ₃₀₀). This takes 1-3 min on a modern CPU.
    with timer("LB eigendecomposition"):
        lb = compute_eigenpairs(
            template.vertices,
            template.faces,
            n_eigenpairs=args.n_eigenpairs,
        )

    # Sanity check: plot the eigenspectrum (Weyl's law diagnostic)
    plot_eigenspectrum(
        lb.eigenvalues,
        n_show=min(100, lb.n_eigenpairs),
        output_path=output_dir / "eigenspectrum.png",
    )

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 2: Compute spectral shape descriptors
    # ══════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("PHASE 2: Spectral shape descriptors (HKS, WKS, GPS)")
    logger.info("=" * 60)

    with timer("HKS computation"):
        hks = heat_kernel_signature(lb, n_scales=16)
        logger.info("  HKS shape: %s", hks.shape)

    with timer("WKS computation"):
        wks = wave_kernel_signature(lb, n_energies=16)
        logger.info("  WKS shape: %s", wks.shape)

    with timer("GPS computation"):
        gps = global_point_signature(lb, n_components=10)
        logger.info("  GPS shape: %s", gps.shape)

    # Visualise HKS across scales
    plot_hks_multiscale(
        hks, template.vertices, template.faces,
        output_path=output_dir / "hks_multiscale.png",
    )

    # Combined spectral feature matrix (for downstream analyses)
    spec_features = spectral_feature_matrix(lb, hks_scales=16, wks_energies=16, gps_components=10)
    logger.info("Combined spectral feature matrix: %s", spec_features.shape)

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 3: Extract morphometric profiles for the cohort
    # ══════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("PHASE 3: Extract morphometric profiles")
    logger.info("=" * 60)

    # Load reference subject IDs (the "normative" cohort)
    ref_ids = _load_subject_list(args.reference_ids)
    logger.info("Reference cohort: %d subjects", len(ref_ids))

    with timer("Extract reference profiles"):
        ref_profiles = extract_cohort_profiles(
            subjects_dir=args.subjects_dir,
            subject_ids=ref_ids,
            hemi=args.hemi,
            surface="pial",
            features=["thickness", "curv", "sulc"],
        )

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 4: Train GP normative models
    # ══════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("PHASE 4: Train GP normative models")
    logger.info("=" * 60)

    # One model per morphometric feature
    features_to_model = ["thickness", "curv", "sulc"]
    models = {}

    for feat_name in features_to_model:
        logger.info("Training normative model for '%s'…", feat_name)

        # Get the (N_vertices, N_subjects) matrix for this feature
        train_data = ref_profiles.get_feature_matrix(feat_name)

        model = cf.CorticalNormativeModel(
            lb=lb,
            nu=2.5,           # Matérn smoothness (2.5 is a good default)
            n_inducing=512,    # Number of inducing points for SVGP
            device=device,
        )

        with timer(f"Train {feat_name}"):
            history = model.fit(
                train_features=train_data,
                feature_name=feat_name,
                n_epochs=args.n_epochs,
                lr=0.01,
                batch_size=4096,
            )

        models[feat_name] = model

        # Save model
        model.save(output_dir / f"model_{feat_name}")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 5: Score a patient — generate surprise maps
    # ══════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("PHASE 5: Score patient '%s'", args.patient_id)
    logger.info("=" * 60)

    # Load patient's surface data (registered to fsaverage)
    patient_surf = cf.load_freesurfer_surface(
        subjects_dir=args.subjects_dir,
        subject_id=args.patient_id,
        hemi=args.hemi,
        surface="pial",
        overlays=features_to_model,
    )

    surprise_maps = []
    for feat_name in features_to_model:
        patient_data = patient_surf.get_overlay(feat_name)
        model = models[feat_name]

        with timer(f"Predict {feat_name}"):
            result = model.predict(patient_data)

        logger.info(
            "  %s — mean z: %.2f, mean surprise: %.2f, "
            "fraction |z|>2: %.1f%%",
            feat_name,
            np.nanmean(result.z_score),
            np.nanmean(result.surprise),
            100 * (np.abs(result.z_score) > 2).mean(),
        )

        # Build SurpriseMap object
        smap = compute_surprise(
            observed=result.observed,
            predicted_mean=result.mean,
            predicted_var=result.variance,
        )
        surprise_maps.append(smap)

        # Visualise per-feature surprise
        plot_surprise_map(
            template.vertices, template.faces,
            smap.surprise, smap.z_score,
            title=f"{feat_name} — {args.patient_id}",
            output_path=output_dir / f"surprise_{feat_name}_{args.patient_id}.png",
            show=False,
        )

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 6: Combined multi-feature surprise & network aggregation
    # ══════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("PHASE 6: Combined surprise & network aggregation")
    logger.info("=" * 60)

    # Combine surprise across features (Fisher's method)
    combined = combined_surprise(surprise_maps, method="fisher")

    logger.info(
        "Combined surprise — mean: %.2f, fraction anomalous: %.1f%%",
        np.nanmean(combined.surprise[combined.vertex_mask]),
        100 * (np.abs(combined.z_score[combined.vertex_mask]) > 2).mean(),
    )

    # Visualise combined surprise map
    plot_surprise_map(
        template.vertices, template.faces,
        combined.surprise, combined.z_score,
        title=f"Combined Surprise — {args.patient_id}",
        output_path=output_dir / f"surprise_combined_{args.patient_id}.png",
        show=False,
    )

    # Network-level aggregation (requires Yeo-7 or similar parcellation)
    try:
        from corticalfields.utils import load_fsaverage_parcellation
        yeo_labels, yeo_names = load_fsaverage_parcellation("yeo_7", hemi=args.hemi)

        network_scores = combined.aggregate_by_network(yeo_labels, yeo_names)

        logger.info("Network-level anomaly scores:")
        for net_name, scores in network_scores.items():
            logger.info(
                "  %s: mean_z=%.2f, fraction_anom=%.1f%%, surprise=%.2f",
                net_name,
                scores["mean_z"],
                100 * scores["fraction_anomalous"],
                scores["mean_surprise"],
            )

        plot_network_anomaly_profile(
            network_scores,
            title=f"Network Anomaly Profile — {args.patient_id}",
            output_path=output_dir / f"network_profile_{args.patient_id}.png",
        )
    except Exception as e:
        logger.warning("Could not load Yeo-7 parcellation: %s", e)

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 7 (BONUS): Spectral Similarity Network
    # ══════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("PHASE 7: Spectral Similarity Network")
    logger.info("=" * 60)

    try:
        from corticalfields.surface import load_annot
        dk_labels, dk_names = load_annot(
            args.subjects_dir, "fsaverage", hemi=args.hemi, annot="aparc",
        )

        # Build SSN from spectral features
        ssn = spectral_similarity_network(spec_features, dk_labels, metric="cosine")
        logger.info("SSN shape: %s", ssn.shape)

        # Graph metrics
        gm = graph_metrics(ssn, density=0.15)
        logger.info(
            "SSN graph: %d edges, density=%.3f, modularity=%.3f, "
            "global_efficiency=%.3f",
            gm["n_edges"], gm["density"],
            gm["modularity"], gm["global_efficiency"],
        )

        # Save SSN matrix
        np.save(output_dir / f"ssn_{args.patient_id}.npy", ssn)

    except Exception as e:
        logger.warning("Could not compute SSN: %s", e)

    logger.info("=" * 60)
    logger.info("Pipeline complete! Results in: %s", output_dir)
    logger.info("=" * 60)


# ── Helpers ─────────────────────────────────────────────────────────────


def _load_subject_list(path: str) -> list:
    """Load subject IDs from a text file (one per line)."""
    with open(path) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def parse_args():
    parser = argparse.ArgumentParser(
        description="CorticalFields full pipeline: T1w → surprise maps",
    )
    parser.add_argument(
        "--subjects-dir", required=True,
        help="FreeSurfer SUBJECTS_DIR path",
    )
    parser.add_argument(
        "--reference-ids", required=True,
        help="Text file with reference cohort subject IDs (one per line)",
    )
    parser.add_argument(
        "--patient-id", required=True,
        help="Subject ID to score against the normative model",
    )
    parser.add_argument(
        "--hemi", default="lh", choices=["lh", "rh"],
        help="Hemisphere (default: lh)",
    )
    parser.add_argument(
        "--output-dir", default="results",
        help="Output directory for figures and models",
    )
    parser.add_argument(
        "--n-eigenpairs", type=int, default=300,
        help="Number of LB eigenpairs to compute (default: 300)",
    )
    parser.add_argument(
        "--n-epochs", type=int, default=100,
        help="GP training epochs (default: 100)",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        help="Logging level (default: INFO)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
