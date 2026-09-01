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
| E004-dinov2-frozen | 2026-09-01 | blended_v1 soft labels / 4,407 / 3 fluid planes | 3x frozen DINOv2 ViT-S/14 @518 + linear head, plane-prior combiner | blended-cv A/B vs resnet34 @224 (baseline = E003's recorded run) | pending | pending | — | this file (log below), PR #14 |
| E003-blended-labels | 2026-09-01 | blended_v1 soft labels / 4,407 / 3 fluid planes | same arch as E001 (frozen resnet34 + linear heads), soft-target BCE | **new regime**: pooled-OOF 5x5 stratified CV over all blended studies (vs labels thresholded at 0.5) + public LB A/B vs E002 | 0.771 | pending | pending | this file (log below), kernels train v6 |
| E002-plane-prior-combiner | 2026-09-01 | gold-58 checkpoints (unchanged) | E001 models + clinical per-label plane weights | public LB A/B vs E001 | — | 0.692 | — | docs/clinical understanding/plane-abnormality-relevance.md |
| E001-pipe-check-gold58 | 2026-08-31 | gold-58 / 56-58 per plane / 3 fluid planes | 3x frozen resnet34 + linear head | none (in-sample only) | 1.0 in-sample (memorized, expected) | 0.691 | train ~13 min CPU; scoring completed ~4h wall | issue #6, commit 4ef6afc, kernels train v5 / inference v2 |

## Log

### E001-pipe-check-gold58
- **Hypothesis:** the full pipeline (DICOM decode → series selection → train → checkpoint → offline inference) runs end to end and produces non-degenerate probabilities. Near-random LB expected; trains on the future gold-58 eval set, so this checkpoint is never compared against anything evaluated on those studies.
- **Outcome (training half, 2026-08-31):** three per-plane fluid specialists trained on Kaggle (kernel v5, commit 4ef6afc). Zero DICOM decode failures across ~170 real series — transfer-syntax risk retired. Per-plane skips matched measured coverage exactly (sag 56/58, cor 56/58, ax 58/58). In-sample AUC 1.0 everywhere = memorization at n≈56, the expected fit-sanity signal. Environment lessons now baked into the notebooks: pip needs --no-deps on Kaggle (numpy upgrade breaks the image), competition data mounts at /kaggle/input/competitions/<slug>, and the default GPU is too old for the image's torch (cu128) — CPU sufficed here; pick T4/L4 in the UI when training gets heavy. LB score pending the inference half.
- **Outcome (submission, 2026-09-01): public LB macro AUC 0.691.** Far above the ~0.5 expectation for a gold-58 prototype — frozen ImageNet features + linear heads generalize despite n≈56 training rows. Issue #6 complete: full train→checkpoint→datasets→offline ensemble→submission path proven. 0.691 is now the baseline every lever pull gets measured against.

### E002-plane-prior-combiner
- **Hypothesis:** weighting each label's ensemble average by clinical plane-of-choice (e.g. MCL trusts coronal, PF OA trusts axial) beats the uniform mean. Controlled A/B vs E001: identical checkpoints, combiner-only change, so any LB delta is attributable to the weighting.
- **Outcome (submission, 2026-09-01): public LB macro AUC 0.692 vs E001's 0.691 — +0.001, a null.** Well inside the noise floor (paired σ on comparisons this size is ~0.01+; see docs/rsna_brain.md §2.7), so the clinical prior neither helped nor hurt measurably on gold-58-quality checkpoints. Keeping the combiner: it costs nothing at inference, the renormalization-over-present-planes behavior is load-bearing for missing series, and the E003/E004 blended-labels regime re-tests it with far better-trained heads where per-plane differences may actually surface. The learned combiner (DECISIONS.md #3) remains the eventual replacement.

### E003-blended-labels
- **Hypothesis:** 76x sample size is the biggest available lever. Training the same frozen-backbone + linear-head specialists on all 4,407 studies with blended soft labels (report-mined probabilities, `knee-labels` dataset) beats the 0.691 gold-58 baseline (and E002's score once known). Controlled vs E002: identical architecture and inference (plane-prior combiner unchanged), labels-only change.
- **Design notes:** heads train on the soft probabilities directly (BCE accepts soft targets; the `__weight`/`__tier` columns are validated but deliberately unused — tier-weighted loss is the next experiment). Labels are thresholded at 0.5 only where a binary quantity is required (fold stratification, AUC). The blended labels scored macro AUC 0.887 against the 58 gold studies (measured 2026-09-01, wide CIs at n=58) — a rough ceiling for what training on them can reach. Gold-58 is retired as a special set (DECISIONS.md #5): the labeled pool is now the full 4,407 and CV runs over all of it. One threaded decode pass fills a persisted feature bank (`feature_bank_blended_v1.pt` in the kernel output), so head retrains and CV repeats cost seconds; checkpoints now embed `label_source`/`n_studies` provenance.
- **Outcome (training half, 2026-09-01): blended-cv macro OOF AUC 0.771** (kernel v6, ~75 min wall on T4, decode-dominated). Per-repeat spread 0.770–0.772 — the ±0.001 error bar that n=4,407 buys vs gold-58's ~±0.01. Plane coverage sag 4,150 / cor 4,247 / ax 4,406; exactly two series failed decode, and they are the two forum-confirmed corrupt-pixel studies (docs/rsna_brain.md §2.30) — skipped gracefully, no blacklist needed. Per-label: large/diffuse findings lead (Medial OA 0.82, Effusion 0.81, Synovitis 0.80, Fracture 0.80), thin structures trail (Lateral Meniscus 0.71, MCL 0.73, Medial Meniscus 0.75) — the same large-vs-thin pattern independently reported on the forum (§2.34/§2.35), pointing at input geometry/ROI as the next lever after backbones. Reminder: 0.771 measures agreement with the miner (whose own gold agreement is 0.887, the soft ceiling); LB A/B vs E002's 0.692 pending checkpoint publish + inference run.

### E004-dinov2-frozen
- **Hypothesis:** frozen self-supervised DINOv2 ViT-S/14 features at native 518px beat
  frozen supervised ImageNet ResNet-34 features at 224px in our linear-probe regime,
  now trained on the blended 4,407-study pool. Controlled A/B under `blended-cv`:
  same labels, folds/seed, and combiner as E003 — the delta is attributable to
  backbone + resolution jointly. The dinov2 checkpoints ship and get submitted
  regardless of the CV comparison; the resnet arm's CV is the diagnostic baseline —
  if the LB or CV disappoints, it tells us whether to debug the backbone swap or
  look elsewhere. Forum context (docs/rsna_brain.md §2.31/§2.32/§2.36): backbone
  *size* scaling is a measured null and one controlled test saw DINOv2-Small lose
  to ResNet-34 — so the A/B numbers also decide whether backbone investment
  continues (unfreezing next) or stops in favor of input-geometry work.
- **Outcome:** _pending_
