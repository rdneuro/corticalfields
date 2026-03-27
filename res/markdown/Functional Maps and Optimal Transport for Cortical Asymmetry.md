# A genuinely novel combination, but not yet disruptive

**The proposed method — functional maps + optimal transport + distance-based regression for cortical asymmetry — is verifiably novel in its specific combination.** No published paper joins these three elements. However, after exhaustive prior-art search across PubMed, arXiv, bioRxiv, and Google Scholar through early 2026, the honest verdict is that this sits at a **strong "4x" contribution** — clearly beyond incremental, but short of paradigm-shifting unless paired with a killer clinical application. The gap between "novel combination of existing tools" and "disruption that changes how the field thinks" will be closed or not by execution choices, not by the mathematics alone.

The timing is excellent. A March 2025 paper (Liu et al., *Brain Structure and Function*) demonstrated **systematic bias in FreeSurfer's default parcellation** for surface area asymmetry — atlas-defined left and right hemisphere labels are not symmetric, producing spurious lateralization in key speech/language regions. This creates urgent demand for exactly the kind of atlas-free, intrinsic geometric approach proposed here.

---

## The novelty claim survives scrutiny, but barely in one critical spot

**Functional maps (C matrix) between left and right brain hemispheres: genuinely novel.** Lombaert's BrainTransfer (IPMI 2015) applied the functional maps framework to brain cortex — but exclusively for *inter-subject* correspondence of the same hemisphere. It never touched left-right asymmetry. No follow-up from Lombaert's group or anyone else has computed a C matrix between contralateral hemispheres. The Rustamov et al. (2013) shape difference operators, which would enable spatial localization of asymmetry via functional maps, have never been applied to brain surfaces at all.

**Wasserstein distance for hemispheric asymmetry: genuinely novel.** Optimal transport in neuroimaging has been applied to inter-subject cortical shape classification (Su et al., IPMI 2015; Shi & Wang, TPAMI 2020), connectome remapping across atlases (CAROT, Dadashkarimi et al. 2023), volumetric transport-based morphometry (Kolouri/Rohde group, NeuroImage 2018), and brain network topology (Chung et al., NeuroImage 2023). None of these compute OT between a subject's own left and right hemispheres.

**OT + functional maps in any neuroimaging context: genuinely novel.** Le et al. (CVPR 2024) combined sliced Wasserstein distance with functional map regularizers for generic 3D shape matching, and SRE-FMaps (2025) integrated Sinkhorn-regularized transport with functional maps — but both are computer graphics papers tested on FAUST/SCAPE benchmarks, never on brain surfaces. Abulnaga et al. (arXiv, January 2025) explicitly noted that functional maps "scale poorly to high-curvature, high-resolution cortical surfaces," signaling this is a recognized open problem.

**The critical near-miss is Chen et al.'s Matched Asymmetry Signature (MAS), published in *Brain Communications* 2024.** This paper computes a **500×500 correlation matrix between left and right hemisphere eigenfunctions** and extracts maximum cross-correlations at each scale. This is conceptually adjacent to computing a functional map C matrix — it's essentially computing one column of information (the diagonal of the best-matching) from what would be the full C matrix. The MAS paper acknowledges that eigenfunctions at the same index do not align between hemispheres (their 11th left eigenfunction correlates at r=0.96 with the 12th right eigenfunction, but only r=0.17 with the 11th right). This is precisely the phenomenon the functional map C matrix captures systematically. **The proposed approach needs to clearly articulate what the full C matrix reveals that MAS's greedy maximum-correlation matching does not** — likely the off-diagonal structure encoding coupled multi-scale deformations.

---

## What paradigm this actually challenges

The current standard pipeline for cortical asymmetry is rigid and brittle: FreeSurfer → fsaverage registration → Desikan-Killiany/Schaefer parcellation → per-region scalar features → asymmetry index AI = (L−R)/((L+R)/2) → GLM. This pipeline has three fundamental weaknesses the proposed method addresses directly.

First, **information destruction through parcellation**. Averaging cortical thickness across 34 DK regions annihilates sub-regional geometric patterns. The ENIGMA Laterality Working Group's landmark study of **17,141 individuals** (Kong et al., PNAS 2018) used exactly this approach and acknowledged the limitation, recommending "vertex-wise approaches combined with cross-hemispheric registration methods" for future work. Eight years later, that recommendation remains unimplemented at scale.

Second, **registration-induced artifacts**. The 2025 bias paper showed that FreeSurfer's left and right atlas parcellations were independently drawn and are not symmetric — Heschl's gyrus, inferior frontal gyrus pars opercularis, and banks of the superior temporal sulcus show persistent spurious leftward bias regardless of actual hemisphere identity. Every study using the default pipeline inherits this artifact.

Third, **scalar reductionism**. Current measures treat asymmetry as a number per region. The proposed approach reframes asymmetry as a *geometric correspondence structure* — the full C matrix encodes not just "how different" but "how the difference is organized across spatial scales." This is conceptually analogous to how BrainSpace's gradient framework replaced discrete network assignments with continuous organizational dimensions.

**Compared to previous paradigm shifts, this ranks as follows:** FreeSurfer's introduction of surface-based morphometry (~1999) was a true paradigm shift — it changed the fundamental data representation from voxels to manifolds. BrainSpace's cortical gradients (2020) was a strong paradigm shift for connectivity analysis — it changed the question from "which network?" to "where on the gradient?" The proposed method would be a paradigm shift specifically for asymmetry analysis — changing the question from "how much different is this region?" to "what is the geometric correspondence structure between hemispheres?" This is narrower in scope than FreeSurfer or BrainSpace but potentially deeper within its domain.

---

## The competitive landscape has clear gaps — and one dangerous overlap

**BrainPrint / Shape-DNA (Wachinger/Reuter group)** applied Laplace-Beltrami spectral asymmetry to Alzheimer's disease in a landmark *Brain* 2016 paper, showing hippocampal shape asymmetry predicts MCI-to-dementia conversion when volume asymmetry fails entirely. However, BrainPrint operates on **subcortical structures only** for asymmetry, produces **a single scalar distance** per structure pair (Mahalanobis distance between eigenvalue spectra), and cannot spatially localize where asymmetry occurs. The proposed method addresses all three limitations.

**SAS (Chen et al., eLife 2022)** is the strongest direct competitor. It uses LBO eigenvalue spectra to quantify cortical shape asymmetry as a continuous, atlas-free, multi-scale measure — and demonstrated that shape asymmetry is more individualized than cortical thickness, surface area, or even functional connectivity. But SAS produces **only global/scale-specific scalar scores**, establishes **no correspondence between hemispheres**, and generates **no spatial maps** of where asymmetry is localized. The authors explicitly described their approach as studying "seismic waves at the global tectonic scale, instead of focusing on a particular city." The proposed functional maps approach is precisely about focusing on the city.

**FUGW (Thual et al., NeurIPS 2022)** provides OT-based brain surface correspondence but was designed for inter-subject alignment, not intra-subject asymmetry measurement. It could theoretically be adapted for left-right correspondence, but this hasn't been done.

**The MELD project (Brain 2022; JAMA Neurology 2025)** represents state-of-the-art focal cortical dysplasia detection using graph neural networks on **33 scalar surface features** extracted via FreeSurfer. Even MELD Graph, which operates on the cortical mesh, uses extracted feature vectors — not raw surface geometry. Sensitivity for MRI-negative patients is **63.7%**, leaving substantial room for improvement. This is the killer application opportunity.

**Distance-based regression (MDMR, HSIC) with geometric brain features has never been done.** MDMR exists in neuroimaging via the CWAS framework (Shehzad et al., NeuroImage 2014), but exclusively for functional connectivity distance matrices. HSIC appears entirely absent from brain morphometry. Applying these to spectral/geometric features is a clear methodological contribution.

---

## What would make reviewers excited versus hostile

**Excitement triggers for Nature Methods / NeuroImage reviewers:**
The off-diagonal energy interpretation is mathematically elegant — a perfectly symmetric brain produces a diagonal C matrix; pathology manifests as specific off-diagonal structure. This is the kind of clean conceptual framework that methods reviewers appreciate. The atlas-free property directly addresses the newly-documented FreeSurfer bias. Individualized, spatially-resolved asymmetry maps that go beyond both SAS's global scores and parcellation's coarse regions would represent genuine capability expansion.

**The strongest objections a skeptical reviewer would raise:**

The "why not just" challenge will be severe. A reviewer will compute standard FreeSurfer asymmetry indices on the same n=46 HADS dataset, run a correlation, and ask why the proposed method's marginal improvement justifies orders-of-magnitude greater computational complexity. The paper must include an **ablation study** demonstrating what each component (functional maps → OT → distance regression) contributes independently, with quantitative evidence that simpler alternatives fail.

**Sample size is a serious vulnerability.** The BWAS paper (Marek et al., *Nature* 2022) demonstrated that brain-behavior associations require thousands of subjects for reproducibility. Any brain-HADS association from **n=46** will be received with justified skepticism. This subset must be framed as "illustrative proof-of-concept," never as validated clinical finding. The **n=439** dataset is adequate for method validation (reproducibility, sensitivity analysis, comparison with established measures) — BrainPrint validated on similar scales before its larger ADNI analyses.

**Test-retest reliability is non-negotiable.** Standard cortical thickness achieves ICC 0.74–0.93; surface area and volume exceed 0.95. Novel asymmetry metrics must achieve at minimum ICC > 0.75 (good) and ideally > 0.90 (excellent) on a standard test-retest dataset like HCP's 45-subject retest cohort. Without this, no clinical association is credible.

**Scalability concerns are real.** Abulnaga et al. (2025) explicitly flagged that functional maps "scale poorly to high-curvature, high-resolution cortical surfaces." The paper must report computation times per subject and demonstrate feasibility at cohort scale. If processing one subject takes longer than FreeSurfer's ~6 hours, adoption will be limited regardless of accuracy.

**Mathematical accessibility matters.** Functional maps, optimal transport, and RKHS-based independence tests are individually unfamiliar to most neuroimagers. Combining all three creates a steep comprehension barrier. Tutorial notebooks and clear geometric intuitions are not optional — they are survival requirements for peer review.

---

## The difference between a 10x paper and a 2x paper

A **2x paper** validates the method on a single dataset, shows it correlates with clinical scores marginally better than FreeSurfer AI, publishes in NeuroImage, and gets cited by the computational geometry community but ignored by clinical neuroimagers.

A **10x paper** does four specific things:

**First, it detects focal cortical dysplasia without atlas registration.** An open dataset of 85 FCD patients + 85 controls exists (Scientific Data, 2023). If atlas-free geometric asymmetry maps identify FCD lesions that MELD's scalar-feature approach misses — particularly in the ~36% of MRI-negative cases where MELD Graph fails — this alone justifies a high-impact publication. FCD detection is the single most valuable clinical application because **30–50% of drug-resistant epilepsy patients** have MRI-negative lesions, and surgical outcomes depend critically on lesion localization.

**Second, it produces individualized asymmetry maps that are clinically interpretable.** A January 2025 *Molecular Psychiatry* paper demonstrated individualized cortical thickness asymmetry in ASD and schizophrenia using normative modeling. The proposed method could generate similar individualized maps but at vertex-level resolution and for geometric (not scalar) features, enabling clinicians to see *where* a patient's cortex deviates from expected symmetry.

**Third, it demonstrates that atlas-free measurement solves the documented FreeSurfer bias.** Running the method on the same data where Liu et al. (2025) identified systematic parcellation bias, and showing the bias disappears, would be a powerful validation with immediate practical relevance.

**Fourth, it ships as a pip-installable toolbox with tutorial notebooks.** BrainPrint, BrainSpace, and neuromaps all succeeded partly because they lowered adoption barriers. CorticalFields does not currently exist as a public package. A polished release with documentation transforms a methods paper into a community resource.

---

## Honest venue and framing assessment

**Nature Methods** is achievable but requires the full package: validation on ≥3 datasets (yours + HCP test-retest + one public clinical dataset), head-to-head comparison with FreeSurfer AI / BrainPrint / SAS showing clear superiority, at least one killer application (FCD detection is ideal), and a polished open-source toolbox. The framing must emphasize basic neuroscience capability ("a new geometric framework for understanding cortical organization") rather than clinical diagnostics, which is out of Nature Methods' scope. BrainSpace was published in *Communications Biology*, not Nature Methods — so calibrate expectations accordingly.

**NeuroImage** is the natural primary target. BrainPrint was published here. The validation requirements are substantial but achievable with n=439 + HCP test-retest + comparison with standard approaches. The framing should be "computational methods with clinical relevance" — exactly NeuroImage's wheelhouse.

**Human Brain Mapping's Toolbox category** is the pragmatic fallback if validation is limited to the existing datasets. HBM explicitly evaluates accessibility and usability over methodological novelty.

**The optimal framing is a methods paper with clinical application, not a pure methods paper or a pure clinical paper.** Lead with the representational insight (asymmetry as geometric correspondence structure), validate the method rigorously, and close with the clinical demonstration on epilepsy data. This mirrors BrainPrint's successful structure: introduce the mathematics, validate extensively, then show clinical relevance.

---

## The bottom line

This is not hype. The combination is genuinely novel after thorough prior-art search. But it is also not yet disruptive — disruption requires changing behavior, not just publishing mathematics. The Chen et al. MAS work (2024) is closer to the functional maps idea than the proposers may realize, narrowing the conceptual gap. The path from "novel" to "disruptive" runs through exactly one gate: **demonstrating a clinical capability that was previously impossible.** Atlas-free FCD detection in MRI-negative epilepsy patients is the most promising candidate. Without that killer application, this is a well-executed methods paper for NeuroImage. With it, the ceiling rises considerably.