# Cortical surface selection and EDA strategies for spectral neuroimaging pipelines

**The midthickness surface is the consensus choice for Laplace-Beltrami spectral analysis on FreeSurfer outputs**, supported by the HCP pipeline convention, the landmark Pang et al. (2023) geometric eigenmodes paper, and mesh-quality arguments favoring its reduced self-intersections over the pial surface. For EDA and outlier detection in a small MTLE-HS cohort (n=46), a tiered strategy combining Euler number–based surface QC, Weyl's law eigenvalue verification, robust Mahalanobis distance on clinical variables, and sensitivity analysis with Manhattan-distance MDMR provides the most defensible workflow. Notably, **no published study directly compares LBO eigendecompositions across FreeSurfer surface types**—this is a clear methodological gap worth addressing. This report covers both topics in detail with specific recommendations, thresholds, tools, and reporting standards.

---

## Topic 1: Midthickness wins for spectral analysis, but the evidence is indirect

### The case for midthickness

FreeSurfer produces seven surface types per hemisphere, but three are candidates for LBO computation: the **white** surface (WM/GM boundary), the **pial** surface (GM/CSF boundary), and the **midthickness** surface (their geometric average). The strongest endorsement of midthickness comes from Pang et al. (2023, *Nature*, 618:566–574), who used a triangular mesh representation of the midthickness cortical surface comprising **32,492 vertices per hemisphere** from a downsampled fs_LR symmetric template to compute geometric eigenmodes via the LaPy library. Their eigenmodes explained a substantial proportion of cortical activity variance, and the paper has become the de facto reference for cortical eigenmode analysis.

The geometric rationale is straightforward. Van Essen (2005, *PNAS*) and Winkler et al. (2012, *Cerebral Cortex*) demonstrated that the midthickness surface **does not over- or under-represent gyri or sulci**, unlike the white surface (which overrepresents sulcal fundi in area calculations) or the pial surface (which overrepresents gyral crowns). For spectral analysis, this areal neutrality translates into a more balanced eigendecomposition that captures cortical geometry without systematic bias toward either boundary.

The HCP minimal preprocessing pipeline (Glasser et al., 2013, *NeuroImage*, 80:105–124) adopted midthickness as the reference surface for all CIFTI grayordinate-based analyses. The pipeline generates it by averaging white and pial vertex coordinates via `wb_command -surface-average`, and all downstream analyses—fMRI mapping, myelin maps (T1w/T2w ratio), surface registration—use this surface. The rationale is that midthickness sits at the geometric center of cortical gray matter, **minimizing partial volume contamination** from CSF (near pial) and white matter (near white) while reducing cross-sulcal signal leakage. The fMRIPrep/smriprep pipeline follows the same convention using `mris_expand -thickness lh.white 0.5 lh.graymid`, which expands the white surface outward by half the local cortical thickness.

### Midthickness eigendecomposition stability versus pial

While no paper directly benchmarks eigenvalue stability across surface types, converging evidence strongly favors midthickness. The pial surface has well-documented mesh quality problems: vertices crowd together in deep sulci where opposing banks approach **<1 mm** separation, dura/blood vessel misclassification inflates the pial boundary, and self-intersections occur despite FreeSurfer's Möller (1997) triangle-intersection detection. Reports from cortical tetrahedralization work cite approximately **10,000 self-intersecting faces** on typical pial meshes. These degenerate triangles cause numerical instability in the cotangent-weight FEM discretization of the LBO, potentially producing spurious eigenvalues or near-zero eigenvalues from disconnected mesh patches.

The white surface is more reliably reconstructed—adjacent gyral banks are separated by at least twice cortical thickness at the WM/GM boundary (Fischl, 2012, *NeuroImage*)—but it overrepresents sulcal geometry. The midthickness, being less convoluted in deep sulci than pial, has **fewer self-intersections and better triangle aspect ratios**, making it the most numerically stable choice for spectral decomposition.

### BrainPrint takes a different approach

The BrainPrint framework (Wachinger et al., 2015, *NeuroImage*, 109:232–248) computes ShapeDNA eigenvalues on **both white and pial surfaces independently** (plus 3D tetrahedral volumes), concatenating them into a composite fingerprint for subject identification (>99.8% accuracy). BrainPrint does not use midthickness because its goal is shape discrimination at the boundary surfaces, not functional analysis at the cortical midpoint. For LBO-based spectral descriptors intended for functional or clinical analysis rather than pure morphometric fingerprinting, the HCP/Pang convention of midthickness is more appropriate.

### Functional maps and inter-hemispheric asymmetry

For computing functional map C-matrices between left and right hemispheres, **no published study directly evaluates how surface choice affects correspondence quality**. However, theoretical considerations from Lombaert et al. (2013, 2015, *IPMI*) are instructive. The functional maps framework (Ovsjanikov et al., 2012, *ACM TOG*) relies on eigenvalues being well-separated—near-multiplicities cause eigenvector sign/orientation ambiguities that propagate into the C-matrix. The midthickness surface, with its smoother deep-sulcal geometry, is more likely to yield well-separated eigenvalues than the pial surface. Additionally, for left-right correspondence, using midthickness on both hemispheres averages out asymmetric reconstruction errors that affect the pial and white surfaces differently.

### Clinical cohorts with cortical pathology

For **MTLE-HS with hippocampal sclerosis**, surface choice has direct practical implications. The ENIGMA-Epilepsy working group (Whelan et al., 2018, *Brain*, 141:391–408) documented widespread cortical thinning in MTLE-HS (ipsilateral hippocampal volume reduction d = −1.73 to −1.91), meaning the pial surface may be systematically displaced inward in affected regions, altering both local geometry and spectral properties. The white surface is more robust in atrophic conditions because WM/GM contrast remains sharper than GM/CSF contrast even with cortical thinning. **Midthickness is the safest choice** for clinical cohorts because it averages errors from both boundaries, but researchers should apply rigorous surface QC (Euler number screening, visual inspection of temporal lobe surfaces) before spectral analysis. In severe cases, consider using FreeSurfer's longitudinal pipeline (Reuter & Fischl, 2011) for more stable surface reconstruction.

---

## Topic 2: A tiered EDA pipeline from raw scores to distance matrices

### Stage 1: Clinical and demographic EDA for n=46

With **46 MTLE-HS patients**, the statistical landscape is constrained: formal normality tests have reduced power, effect size estimates carry wide confidence intervals, and multivariate methods approach the minimum viable sample size. The overarching principle is to **prioritize visual assessment and robust statistics over formal parametric tests**.

For HADS scores (anxiety and depression subscales, each 0–21), standard outlier thresholds require careful interpretation. HADS is ordinal, bounded, and typically right-skewed with floor effects—psychometric studies show **52–78% of respondents endorse the lowest response** on individual items. A HADS-A score of 19 may appear as a statistical outlier but represents clinically severe anxiety (≥16 cutoff). The recommendation is to **never auto-remove HADS outliers based solely on statistical criteria**; instead, flag values for clinical review and check for data entry errors (scores outside 0–21 or fractional values). In epilepsy populations specifically, ~55% score above the anxiety cutoff (≥8) and ~27% above the depression cutoff, so high scores are expected.

For **univariate outlier detection** on age and epilepsy duration, the Tukey fence method (Q1 − 1.5×IQR to Q3 + 1.5×IQR) works well for approximately symmetric variables, but epilepsy duration is typically right-skewed. Apply the **modified Z-score using Median Absolute Deviation** (MAD): flag observations where |0.6745 × (x − median) / MAD| > 3.5 (Iglewicz & Hoaglin, 1993), optionally on log-transformed duration.

For **multivariate outlier detection** across all clinical variables simultaneously, robust Mahalanobis distance via the **Minimum Covariance Determinant (MCD)** estimator is strongly recommended. Classical Mahalanobis distance uses the sample covariance matrix, which is itself distorted by the very outliers it aims to detect. The MCD finds the subset of ~75% of observations with the smallest covariance determinant, yielding breakdown-resistant estimates. At n=46 with 4 variables, compare robust distances to χ²(4, 0.975) ≈ **11.14** (Cerioli, 2010, confirmed MCD reliability for n ≥ ~50; the ONDRI study demonstrated MCD superiority over univariate approaches in clinical neuroimaging). Implementation is straightforward via `sklearn.covariance.MinCovDet` with `support_fraction=0.75`.

For **normality assessment**, the Shapiro-Wilk test is the most powerful option for n=46 (Razali & Wah, 2011), but a non-significant result merely fails to reject normality—it does not confirm it. Supplement with Q-Q plots and report skewness (SE ≈ √(6/46) ≈ 0.36) and excess kurtosis (SE ≈ √(24/46) ≈ 0.72). Use non-parametric methods (Spearman correlations, Mann-Whitney U) as the default for HADS analyses, and consider log-transforming epilepsy duration if it appears log-normal.

For visualization, **raincloud plots** (Allen et al., 2019, *Wellcome Open Research*) are the gold standard for small samples—they combine half-violin density, jittered raw data points, and box plot summaries, making n=46 individual observations visible. Use Spearman correlation matrices via `pingouin.pairwise_corr()` for mixed ordinal-continuous data, which also returns confidence intervals, power, and Bayes factors per pair.

### Stage 2: Post-processing quality control of spectral metrics

**Weyl's law verification** is the primary eigendecomposition quality check. For a compact 2D Riemannian manifold, eigenvalue growth follows λ_k ~ 4πk / Area(M) asymptotically. Plot λ_k versus k; the slope should approximate **4π/Area(M)**. Deviations indicate mesh quality problems. Specific red flags include: negative eigenvalues (impossible for the positive semi-definite LBO; indicates cotangent weight failures from obtuse/degenerate triangles), anomalous spectral gaps (suggesting near-disconnected surface components), eigenvalue degeneracies beyond symmetry expectations, and an abnormally small first nonzero eigenvalue λ₁ (indicating near-disconnected regions or mesh holes).

The **Euler number** of the pre-topology-correction surface (`orig.nofix`) is the single best automated predictor of FreeSurfer surface quality, achieving AUC 0.98–0.99 for detecting unusable images (Rosen et al., 2018, *NeuroImage*, 169:407–418). ENIGMA recommends excluding subjects with Euler number **>4 SD below the sample mean**; subjects between 2–4 SD should be visually inspected. Extract via `mris_euler_number lh.orig.nofix`. The **BrainPrint** toolkit and the FSQC toolbox (both from Deep-MI) provide additional spectral QC by computing ShapeDNA eigenvalues per structure and flagging subjects with outlier spectral profiles or abnormal left-right asymmetry.

For **distance matrix outlier screening** before MDMR, compute row sums of the Wasserstein distance matrix—subjects with extreme mean distances to all others are candidate outliers. Embed the distance matrix via PCoA (classical MDS) and plot the first 2–3 principal coordinates; outliers appear as points separated from the main cluster. For non-Euclidean distances producing negative eigenvalues in PCoA, apply the Lingoes or Cailliez correction. UMAP with `metric='precomputed'` and `n_neighbors=10–15` (appropriate for n=46) provides a complementary nonlinear visualization; run multiple times with different seeds to verify embedding stability.

### Stage 3: The outlier removal decision framework

**Remove outliers BEFORE computing distance matrices, not after.** Distance matrices are N×N; removing a subject post-hoc requires deleting a row and column, which changes the properties of the Gower-centered matrix G = −½HD²H used internally by MDMR. The centering matrix H depends on N, so post-removal introduces subtle biases. The recommended workflow is: identify problematic subjects via QC metrics → remove → compute distance matrices on the clean set. If outlier detection depends on the distances themselves (e.g., Wasserstein distance outliers), use a two-stage approach: compute distances → screen → recompute on retained subjects.

For the **outlier removal philosophy**, the field has converged on a "flag and review" approach rather than blind automatic exclusion. The Maastricht Study QC paper (2021, *NeuroImage*) found that **excluding outliers based on global morphological measures actually increased unexplained variance**—roughly 40% of segmentation errors are invisible to morphological outlier screens, and removal discards genuine biological variation. The recommended hierarchy is:

- **Tier 1 (automatic exclusion):** Euler number >4 SD below mean, negative LBO eigenvalues, catastrophic segmentation failures
- **Tier 2 (flag for manual review):** 2–4 SD on Euler number, Weyl's law deviation ratio >1.3, HKS network outliers >3 SD
- **Tier 3 (sensitivity analysis):** Subjects that are outliers on only one metric but normal on others should be retained in the primary analysis and removed in a secondary sensitivity analysis

When a subject is an outlier in one metric but normal in others, **subject-level exclusion is preferred for MDMR** (which treats each subject as a unit), but metric-specific flagging should be documented to distinguish localized mesh defects from global quality failures.

### MDMR-specific robustness considerations

MDMR's permutation approach provides robustness to **distributional assumptions** but not to **outliers** per se. The pseudo-F statistic is a ratio of sums of squares, inherently sensitive to extreme values. McArtor et al. (2017, *Psychometrika*) showed that **Manhattan-distance MDMR is substantially more robust** to heavy-tailed distributions and outliers than Euclidean-distance MDMR. The Freedman-Lane permutation method is relatively robust to extreme values (Winkler et al., 2014, *NeuroImage*), while the ter Braak method becomes more conservative.

The practical recommendation is a **three-way sensitivity analysis**: run MDMR on (1) the full sample with no exclusions, (2) the QC-filtered sample after Euler number and spectral checks, and (3) the additionally filtered sample after statistical outlier removal. Consistency across all three strengthens conclusions; divergence indicates the results are driven by a subset of subjects and warrants careful investigation.

### Reporting to COBIDAS and STROBE standards

COBIDAS guidelines (Nichols et al., 2017, *Nature Neuroscience*) require transparent reporting of all preprocessing steps with exact software versions, statistical modeling choices, and quality control decisions across >100 checklist items. STROBE guidelines mandate a participant flow diagram showing initial N → each exclusion step with counts and reasons → final analytic N.

For outlier exclusion reporting, document:

- Pre-registered exclusion criteria (registered on OSF before analysis)
- Specific thresholds with justification ("subjects with Euler number >4 SD below sample mean were excluded, following ENIGMA convention")
- Number excluded at each tier with reasons
- Distributional comparison of excluded versus retained subjects on clinical variables
- Results both with and without exclusions as formal sensitivity analysis
- QC code and metrics shared as supplementary material

---

## Practical pipeline summary

The complete workflow integrates both topics into a single pipeline. After FreeSurfer `recon-all`, extract Euler numbers from `orig.nofix` surfaces and apply automatic exclusion at >4 SD. Generate midthickness surfaces via `wb_command -surface-average` (HCP method) or `mris_expand -thickness lh.white 0.5 lh.graymid` (fMRIPrep method). Compute LBO eigendecompositions using the LaPy library and verify Weyl's law compliance on retained subjects. Extract HKS features aggregated by Yeo-7 networks and compute functional map C-matrices for asymmetry analysis. Run clinical EDA with MCD-based multivariate outlier detection on HADS/age/duration, using raincloud plots and Spearman correlation matrices for visualization. Screen Wasserstein distance matrices via row-sum outlier detection and PCoA/UMAP visualization. Remove flagged subjects, recompute distance matrices, and run Manhattan-distance MDMR with three-way sensitivity analysis. Report per COBIDAS/STROBE with full exclusion documentation.

## Conclusion

The midthickness surface occupies a unique position in the field: adopted as standard by HCP and endorsed by the highest-profile eigenmode analysis to date, yet never formally validated against pial or white surfaces for spectral stability. This represents both a practical consensus and a research opportunity. For the MTLE-HS pipeline specifically, midthickness offers the best balance of geometric neutrality and mesh quality, but the temporal lobe pathology in this cohort demands vigilant surface QC that goes beyond Euler number screening. On the EDA side, the critical insight is that **outlier handling is not a single decision but a documented, tiered process** with pre-registered criteria and sensitivity analyses—an approach that transforms a potential methodological weakness into transparent, reproducible science. The combination of MCD-based robust multivariate detection for clinical variables with spectral QC metrics (Weyl's law, BrainPrint asymmetry) for imaging data provides coverage across both data domains, while the three-way MDMR sensitivity analysis ensures conclusions are not artifacts of outlier inclusion or exclusion.