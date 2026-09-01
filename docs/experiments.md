# Experiment log

Curated registry — one row per meaningful experiment. Raw runs, curves, and full
hyperparameters live in wandb (`rsna-knee` project); this file records what we tried,
why, and what we decided. Aggregate metrics only — no report text, labels, or
StudyInstanceUIDs (Competition Data, rule 2.4.b).

Eval protocol note: scores are only comparable within the same protocol. A protocol
change starts a new comparison regime — mark it clearly.

| ID | Date | Data (labels / n / series) | Model | Eval protocol | Val AUC | Public LB | Inference runtime | Pointers |
|---|---|---|---|---|---|---|---|---|
| E001-pipe-check-gold58 | 2026-08-31 | gold-58 / 56-58 per plane / 3 fluid planes | 3x frozen resnet34 + linear head | none (in-sample only) | 1.0 in-sample (memorized, expected) | pending | train ~13 min CPU | issue #6, commit 4ef6afc, kernel rsna-knee-train v5 |

## Log

### E001-pipe-check-gold58
- **Hypothesis:** the full pipeline (DICOM decode → series selection → train → checkpoint → offline inference) runs end to end and produces non-degenerate probabilities. Near-random LB expected; trains on the future gold-58 eval set, so this checkpoint is never compared against anything evaluated on those studies.
- **Outcome (training half, 2026-08-31):** three per-plane fluid specialists trained on Kaggle (kernel v5, commit 4ef6afc). Zero DICOM decode failures across ~170 real series — transfer-syntax risk retired. Per-plane skips matched measured coverage exactly (sag 56/58, cor 56/58, ax 58/58). In-sample AUC 1.0 everywhere = memorization at n≈56, the expected fit-sanity signal. Environment lessons now baked into the notebooks: pip needs --no-deps on Kaggle (numpy upgrade breaks the image), competition data mounts at /kaggle/input/competitions/<slug>, and the default GPU is too old for the image's torch (cu128) — CPU sufficed here; pick T4/L4 in the UI when training gets heavy. LB score pending the inference half.
