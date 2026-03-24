"""
Morphometric feature extraction for CorticalFields.

Extracts and manages multi-feature morphometric profiles from FreeSurfer
outputs. These profiles serve as the observation vectors for GP normative
modeling: for each vertex, we have a vector of structural features
(cortical thickness, curvature, sulcal depth, surface area, gyrification,
etc.) that characterise local brain morphology.

The module also handles feature normalisation, missing data, and the
construction of population-level feature matrices for training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from corticalfields.surface import CorticalSurface, load_freesurfer_surface

logger = logging.getLogger(__name__)

# Default features to extract (standard FreeSurfer outputs)
DEFAULT_FEATURES = ["thickness", "curv", "sulc", "area", "volume"]


@dataclass
class MorphometricProfile:
    """
    Multi-feature morphometric profile for one hemisphere.

    Stores per-vertex feature values for a set of subjects, enabling
    population-level analyses.

    Attributes
    ----------
    features : dict[str, np.ndarray]
        Feature name → array of shape (N_vertices, N_subjects).
    feature_names : list[str]
        Ordered list of feature names.
    subject_ids : list[str]
        Subject identifiers corresponding to columns.
    n_vertices : int
        Number of mesh vertices.
    hemi : str
        Hemisphere (``'lh'`` or ``'rh'``).
    """

    features: Dict[str, np.ndarray] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    subject_ids: List[str] = field(default_factory=list)
    n_vertices: int = 0
    hemi: str = "lh"

    @property
    def n_subjects(self) -> int:
        return len(self.subject_ids)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def get_feature_matrix(self, feature_name: str) -> np.ndarray:
        """
        Get the (N_vertices, N_subjects) matrix for one feature.
        """
        return self.features[feature_name]

    def get_subject_profile(
        self,
        subject_id: str,
    ) -> Dict[str, np.ndarray]:
        """
        Get all features for one subject.

        Returns
        -------
        profile : dict[str, np.ndarray]
            Feature name → (N_vertices,) array.
        """
        idx = self.subject_ids.index(subject_id)
        return {
            name: self.features[name][:, idx]
            for name in self.feature_names
        }

    def get_vertex_feature_vector(
        self,
        vertex_idx: int,
        subject_idx: int = 0,
    ) -> np.ndarray:
        """
        Get the multi-feature vector at one vertex for one subject.

        Returns
        -------
        vec : np.ndarray, shape (n_features,)
        """
        return np.array([
            self.features[name][vertex_idx, subject_idx]
            for name in self.feature_names
        ])

    def population_mean(self, feature_name: str) -> np.ndarray:
        """Mean across subjects for one feature. Shape: (N_vertices,)."""
        return np.nanmean(self.features[feature_name], axis=1)

    def population_std(self, feature_name: str) -> np.ndarray:
        """Std across subjects for one feature. Shape: (N_vertices,)."""
        return np.nanstd(self.features[feature_name], axis=1)

    def normalise(
        self,
        method: str = "z_score",
        reference_mean: Optional[Dict[str, np.ndarray]] = None,
        reference_std: Optional[Dict[str, np.ndarray]] = None,
    ) -> "MorphometricProfile":
        """
        Normalise features vertex-wise.

        Parameters
        ----------
        method : ``'z_score'`` or ``'robust'``
            Normalisation method. ``'z_score'`` uses mean/std;
            ``'robust'`` uses median/IQR.
        reference_mean, reference_std : dict or None
            External reference statistics (e.g. from a normative cohort).
            If None, uses internal population statistics.

        Returns
        -------
        MorphometricProfile
            New profile with normalised features.
        """
        norm_features = {}

        for name in self.feature_names:
            data = self.features[name].copy()

            if method == "z_score":
                mu = (
                    reference_mean[name]
                    if reference_mean
                    else np.nanmean(data, axis=1)
                )
                sigma = (
                    reference_std[name]
                    if reference_std
                    else np.nanstd(data, axis=1)
                )
                sigma[sigma < 1e-8] = 1.0
                norm_features[name] = (data - mu[:, None]) / sigma[:, None]

            elif method == "robust":
                med = np.nanmedian(data, axis=1)
                q25 = np.nanpercentile(data, 25, axis=1)
                q75 = np.nanpercentile(data, 75, axis=1)
                iqr = q75 - q25
                iqr[iqr < 1e-8] = 1.0
                norm_features[name] = (data - med[:, None]) / iqr[:, None]

        return MorphometricProfile(
            features=norm_features,
            feature_names=self.feature_names.copy(),
            subject_ids=self.subject_ids.copy(),
            n_vertices=self.n_vertices,
            hemi=self.hemi,
        )


def extract_cohort_profiles(
    subjects_dir: Union[str, Path],
    subject_ids: List[str],
    hemi: str = "lh",
    surface: str = "pial",
    features: Optional[List[str]] = None,
) -> MorphometricProfile:
    """
    Extract morphometric profiles for an entire cohort.

    Loads FreeSurfer surface and overlays for each subject and assembles
    them into a population-level MorphometricProfile.

    Parameters
    ----------
    subjects_dir : path-like
        FreeSurfer SUBJECTS_DIR.
    subject_ids : list[str]
        Subject folder names.
    hemi : ``'lh'`` or ``'rh'``
    surface : str
        Surface type (``'pial'``, ``'white'``, etc.).
    features : list[str] or None
        Feature names to extract (default: ``DEFAULT_FEATURES``).

    Returns
    -------
    MorphometricProfile
        Population-level feature matrices.

    Notes
    -----
    All subjects must be in the same template space (e.g. fsaverage)
    or registered to a common surface template, so that vertex indices
    correspond across subjects.
    """
    if features is None:
        features = DEFAULT_FEATURES.copy()

    profile = MorphometricProfile(
        feature_names=features,
        subject_ids=list(subject_ids),
        hemi=hemi,
    )

    # Load first subject to get dimensions
    first_surf = load_freesurfer_surface(
        subjects_dir, subject_ids[0], hemi=hemi, surface=surface,
        overlays=features,
    )
    N = first_surf.n_vertices
    S = len(subject_ids)
    profile.n_vertices = N

    # Initialise feature matrices
    for name in features:
        profile.features[name] = np.full((N, S), np.nan, dtype=np.float64)

    # Fill in data
    for s_idx, sid in enumerate(subject_ids):
        try:
            surf = load_freesurfer_surface(
                subjects_dir, sid, hemi=hemi, surface=surface,
                overlays=features,
            )
            for name in features:
                if name in surf.overlays:
                    data = surf.get_overlay(name)
                    if data.shape[0] == N:
                        profile.features[name][:, s_idx] = data
                    else:
                        logger.warning(
                            "Subject %s: %s has %d vertices (expected %d), skipping.",
                            sid, name, data.shape[0], N,
                        )
        except Exception as e:
            logger.error("Failed to load subject %s: %s", sid, e)

    logger.info(
        "Extracted profiles: %d subjects, %d vertices, %d features.",
        S, N, len(features),
    )

    return profile
