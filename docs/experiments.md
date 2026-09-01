# Experiment log

Curated registry — one row per meaningful experiment. Raw runs, curves, and full
hyperparameters live in wandb (`rsna-knee` project); this file records what we tried,
why, and what we decided. Aggregate metrics only — no report text, labels, or
StudyInstanceUIDs (Competition Data, rule 2.4.b).

Eval protocol note: scores are only comparable within the same protocol. A protocol
change starts a new comparison regime — mark it clearly. `gold58-cv` = pooled
out-of-fold stratified 5-fold x 5-repeat CV of the full ensemble on the gold studies,
seed 0 (`src/knee/cv_gold.py`, DECISIONS.md #4); Val AUC cells report its macro mean,
with the per-repeat spread in the log entry when it matters.

| ID | Date | Data (labels / n / series) | Model | Eval protocol | Val AUC | Public LB | Inference runtime | Pointers |
|---|---|---|---|---|---|---|---|---|
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
