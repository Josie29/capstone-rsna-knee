# Experiment log

Curated registry — one row per meaningful experiment. Raw runs, curves, and full
hyperparameters live in wandb (`rsna-knee` project); this file records what we tried,
why, and what we decided. Aggregate metrics only — no report text, labels, or
StudyInstanceUIDs (Competition Data, rule 2.4.b).

Eval protocol note: scores are only comparable within the same protocol. A protocol
change starts a new comparison regime — mark it clearly. `blended-cv` (current) =
pooled out-of-fold stratified 5-fold x 5-repeat CV of the full ensemble over all
blended-labeled studies, seed 0, AUC vs labels thresholded at 0.5 (`src/knee/cv.py`,
DECISIONS.md #5); Val AUC cells report its macro mean, with the per-repeat spread in
the log entry when it matters. Because the labels are miner-derived, blended-cv
measures agreement with the report miner, not ground truth — the public LB stays the
truth check. `gold58-cv` (retired with DECISIONS.md #5) was the same procedure on the
58 gold studies only (`cv_gold.py`, since generalized into `cv.py`).

| ID | Date | Data (labels / n / series) | Model | Eval protocol | Val AUC | Public LB | Inference runtime | Pointers |
|---|---|---|---|---|---|---|---|---|
| E003-blended-labels | 2026-09-01 | blended_v1 soft labels / 4,407 / 3 fluid planes | same arch as E001 (frozen resnet34 + linear heads), soft-target BCE | **new regime**: pooled-OOF 5x5 stratified CV over all blended studies (vs labels thresholded at 0.5) + public LB A/B vs E002 | pending | pending | pending | this file (log below), PR pending |
| E002-plane-prior-combiner | 2026-09-01 | gold-58 checkpoints (unchanged) | E001 models + clinical per-label plane weights | public LB A/B vs E001 | — | pending | — | docs/clinical understanding/plane-abnormality-relevance.md |
| E001-pipe-check-gold58 | 2026-08-31 | gold-58 / 56-58 per plane / 3 fluid planes | 3x frozen resnet34 + linear head | none (in-sample only) | 1.0 in-sample (memorized, expected) | 0.691 | train ~13 min CPU; scoring completed ~4h wall | issue #6, commit 4ef6afc, kernels train v5 / inference v2 |

## Log

### E001-pipe-check-gold58
- **Hypothesis:** the full pipeline (DICOM decode → series selection → train → checkpoint → offline inference) runs end to end and produces non-degenerate probabilities. Near-random LB expected; trains on the future gold-58 eval set, so this checkpoint is never compared against anything evaluated on those studies.
- **Outcome (training half, 2026-08-31):** three per-plane fluid specialists trained on Kaggle (kernel v5, commit 4ef6afc). Zero DICOM decode failures across ~170 real series — transfer-syntax risk retired. Per-plane skips matched measured coverage exactly (sag 56/58, cor 56/58, ax 58/58). In-sample AUC 1.0 everywhere = memorization at n≈56, the expected fit-sanity signal. Environment lessons now baked into the notebooks: pip needs --no-deps on Kaggle (numpy upgrade breaks the image), competition data mounts at /kaggle/input/competitions/<slug>, and the default GPU is too old for the image's torch (cu128) — CPU sufficed here; pick T4/L4 in the UI when training gets heavy. LB score pending the inference half.
- **Outcome (submission, 2026-09-01): public LB macro AUC 0.691.** Far above the ~0.5 expectation for a gold-58 prototype — frozen ImageNet features + linear heads generalize despite n≈56 training rows. Issue #6 complete: full train→checkpoint→datasets→offline ensemble→submission path proven. 0.691 is now the baseline every lever pull gets measured against.

### E002-plane-prior-combiner
- **Hypothesis:** weighting each label's ensemble average by clinical plane-of-choice (e.g. MCL trusts coronal, PF OA trusts axial) beats the uniform mean. Controlled A/B vs E001: identical checkpoints, combiner-only change, so any LB delta is attributable to the weighting.
- **Outcome:** _pending_

### E003-blended-labels
- **Hypothesis:** 76x sample size is the biggest available lever. Training the same frozen-backbone + linear-head specialists on all 4,407 studies with blended soft labels (report-mined probabilities, `knee-labels` dataset) beats the 0.691 gold-58 baseline (and E002's score once known). Controlled vs E002: identical architecture and inference (plane-prior combiner unchanged), labels-only change.
- **Design notes:** heads train on the soft probabilities directly (BCE accepts soft targets; the `__weight`/`__tier` columns are validated but deliberately unused — tier-weighted loss is the next experiment). Labels are thresholded at 0.5 only where a binary quantity is required (fold stratification, AUC). The blended labels scored macro AUC 0.887 against the 58 gold studies (measured 2026-09-01, wide CIs at n=58) — a rough ceiling for what training on them can reach. Gold-58 is retired as a special set (DECISIONS.md #5): the labeled pool is now the full 4,407 and CV runs over all of it. One threaded decode pass fills a persisted feature bank (`feature_bank_blended_v1.pt` in the kernel output), so head retrains and CV repeats cost seconds; checkpoints now embed `label_source`/`n_studies` provenance.
- **Outcome:** _pending_
