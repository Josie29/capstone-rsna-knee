# Experiment log

Curated registry — one row per meaningful experiment. Raw runs, curves, and full
hyperparameters live in wandb (`rsna-knee` project); this file records what we tried,
why, and what we decided. Aggregate metrics only — no report text, labels, or
StudyInstanceUIDs (Competition Data, rule 2.4.b).

Eval protocol note: scores are only comparable within the same protocol. A protocol
change starts a new comparison regime — mark it clearly.

| ID | Date | Data (labels / n / series) | Model | Eval protocol | Val AUC | Public LB | Inference runtime | Pointers |
|---|---|---|---|---|---|---|---|---|
| E000-constant-priors | — | none | all 0.5 (sample_submission equivalent) | LB only | — | — | — | issue #6 |
| E001-pipe-check-gold58 | — | gold-58 / 58 / fluid-sag | frozen timm ResNet + linear head | none (LB smoke signal) | — | — | — | issue #6 |

## Log

### E000-constant-priors
- **Hypothesis:** none — validates submission mechanics (column names, offline run, scoring completes) before any model is in the loop.
- **Outcome:** _pending_

### E001-pipe-check-gold58
- **Hypothesis:** the full pipeline (DICOM decode → series selection → train → checkpoint → offline inference) runs end to end and produces non-degenerate probabilities. Near-random LB expected; trains on the future gold-58 eval set, so this checkpoint is never compared against anything evaluated on those studies.
- **Outcome:** _pending_
