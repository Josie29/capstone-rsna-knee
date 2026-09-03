# Experiment log

Curated registry — one row per meaningful experiment. Raw runs, curves, and full
hyperparameters live in wandb (`rsna-knee` project); this file records what we tried,
why, and what we decided. Aggregate metrics only — no report text, labels, or
StudyInstanceUIDs (Competition Data, rule 2.4.b).

Eval protocol note: scores are only comparable within the same protocol. A protocol
change starts a new comparison regime — mark it clearly. `blended-cv` (current) =
pooled out-of-fold stratified 5-fold x 3-repeat CV of the full ensemble over all
blended-labeled studies, seed 0, AUC vs labels thresholded at 0.5 (`src/knee/cv.py`,
DECISIONS.md #5). (E003/E004 ran 5 repeats; reduced to 3 on 2026-09-02 — repeats
only set the error-bar precision, ~±0.001 at n=4.4k, so means stay comparable.) Val AUC cells report its macro mean, with the per-repeat spread in
the log entry when it matters. Because the labels are miner-derived, blended-cv
measures agreement with the report miner, not ground truth — the public LB stays the
truth check. `gold58-cv` (retired with DECISIONS.md #5) was the same procedure on the
58 gold studies only (`cv_gold.py`, since generalized into `cv.py`).

| ID | Date | Data (labels / n / series) | Model | Eval protocol | Val AUC | Public LB | Inference runtime | Pointers |
|---|---|---|---|---|---|---|---|---|
| E007-unified-multiplane | 2026-09-02 | blended_v1 soft labels + tier weights / 4,407 / all planes in one bag | ONE MultiPlaneModel: DINOv2 @224 fine-tuned, per-plane embeddings, per-label attention over the cross-plane bag — no combiner | fixed 90/10 holdout (seed 0, paired with E006) | 0.720 (best epoch = frozen warm-up; unfreeze regressed) | — (gate failed, not submitted) | train ~2h T4 | this file (log below), PR #22, kernels train v11 |
| E006a-tier-weighted-loss | 2026-09-02 | blended_v1 soft labels + per-cell `__weight` companions / 4,407 | best frozen E005 config, per-cell weighted BCE | blended-cv A/B vs unweighted, same bank/folds (zero decode — bank refit) | pending | pending | — | this file (log below), PR #20 |
| E006-dinov2-stack | 2026-09-02 | blended_v1 soft labels + tier weights / 4,407 / 3 fluid planes | DINOv2 ViT-S/14 @224 fine-tuned end-to-end on 2.5D triplets, crop140, per-label attention, tier-weighted loss | **new regime**: fixed stratified 90/10 holdout (full CV infeasible at one fine-tune per fold); internal control = frozen warm-up epochs on the same split | pending | pending | — | this file (log below), PR #20/#21 |
| E005a-mm-crop | 2026-09-02 | blended_v1 soft labels / 4,407 / 3 fluid planes | resnet34 @224 with fixed 140mm center crop (PixelSpacing-derived) | blended-cv, crop margin of the 2x2 run shared with E005b | pending | pending | — | this file (log below), PR #18 |
| E005b-perlabel-attention | 2026-09-02 | blended_v1 soft labels / 4,407 / 3 fluid planes | resnet34 @224 + per-label gated attention MIL heads, plane-prior combiner | blended-cv, pooling margin of the 2x2 run shared with E005a | pending | pending | — | this file (log below), PR #17/#18 |
| E004-dinov2-frozen | 2026-09-01 | blended_v1 soft labels / 4,407 / 3 fluid planes | 3x frozen DINOv2 ViT-S/14 @518 + linear head, plane-prior combiner | blended-cv A/B vs resnet34 @224 (baseline = E003's recorded run) | **0.783** | 0.773 (tie w/ E003) | train ~4h11m T4; scoring rerun ~2h (vs E003's ~25 min) | this file (log below), PR #14, kernels train v8 / inference v5 |
| E003-blended-labels | 2026-09-01 | blended_v1 soft labels / 4,407 / 3 fluid planes | same arch as E001 (frozen resnet34 + linear heads), soft-target BCE | **new regime**: pooled-OOF 5x5 stratified CV over all blended studies (vs labels thresholded at 0.5) + public LB A/B vs E002 | 0.771 | **0.773** | scoring rerun ~25 min wall | this file (log below), kernels train v6 / inference v4 |
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
- **Outcome (training half, 2026-09-01): blended-cv macro OOF AUC 0.771** (kernel v6, ~75 min wall on T4, decode-dominated). Per-repeat spread 0.770–0.772 — the ±0.001 error bar that n=4,407 buys vs gold-58's ~±0.01. Plane coverage sag 4,150 / cor 4,247 / ax 4,406; exactly two series failed decode, and they are the two forum-confirmed corrupt-pixel studies (docs/rsna_brain.md §2.30) — skipped gracefully, no blacklist needed. Per-label: large/diffuse findings lead (Medial OA 0.82, Effusion 0.81, Synovitis 0.80, Fracture 0.80), thin structures trail (Lateral Meniscus 0.71, MCL 0.73, Medial Meniscus 0.75) — the same large-vs-thin pattern independently reported on the forum (§2.34/§2.35), pointing at input geometry/ROI as the next lever after backbones. Reminder: 0.771 measures agreement with the miner (whose own gold agreement is 0.887, the soft ceiling).
- **Outcome (submission, 2026-09-01): public LB macro AUC 0.773 vs E002's 0.692 — +0.081, the labels lever confirmed as the dominant one.** Sample size 58→4,407 with identical architecture and inference bought more than every prior change combined. Two calibration notes: (1) blended-cv (0.771) landed within 0.002 of the LB — the local protocol is tracking the real metric well at this operating point, though that alignment is one data point, not a law; (2) the miner's own 0.887 gold agreement and the public-notebook plateau (~0.87, docs/rsna_brain.md §2.34) mark the headroom that better labels/geometry could still buy at this architecture tier. Scoring rerun completed in ~25 min — comfortable inside the 9h cap and a good efficiency-track baseline. Next levers in flight: E004 (DINOv2 backbone, same labels) trains now.

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
- **Outcome (training half, 2026-09-01): blended-cv macro OOF AUC 0.783 vs E003's 0.771 — +0.012, DINOv2 wins the A/B.** Repeat spreads ±0.001 on both arms, so the delta is ~12x the error bar; and it's broad-based (11 of 12 labels moved in DINOv2's favor — the "what a real effect looks like" signature from docs/rsna_brain.md §2.36, not a one-rare-label artifact). Biggest per-label gains: Medial OA 0.820→0.842, Effusion 0.814→0.835, Baker's 0.758→0.778, Medial Meniscus 0.748→0.763; Lateral Meniscus flat at 0.711 — thin-structure labels still trail, consistent with the geometry/ROI hypothesis. Same plane coverage and the same two corrupt-study skips as E003. Cost: ~4h11m T4 vs E003's 75 min (the 518px ViT extraction pass). Verdict for the lever ladder: backbone investment continues — unfreezing (staged fine-tune warm-started from these heads) is the next backbone rung, competing for priority with input geometry. LB pending inference run.
- **Outcome (submission, 2026-09-02): public LB macro AUC 0.773 — an exact tie with E003, despite the +0.012 CV win.** The training-half verdict above is revised by this: DINOv2's CV gain did not transfer to the clean image-derived LB labels, which is the miner-agreement caveat made concrete — part of that +0.012 was likely "agreeing with the report miner better," not "reading the knee better" (a stronger backbone can fit label noise a linear probe on weaker features cannot). Second read: the public LB has its own noise floor, so a small true gain could hide in a tie — but the burden of proof flipped. Consequences: (1) the CV→LB calibration from E003 (±0.002) does not generalize across architecture changes — blended-cv remains a ranking signal, not a predictor, exactly as DECISIONS.md #5 warned; (2) backbone investment (incl. the unfreeze) is DE-prioritized behind input geometry (E005) and label quality, matching the forum consensus (docs/rsna_brain.md §3.5) that we half-hoped to beat; (3) at equal LB, E003's resnet is strictly better on the efficiency axis (~25 min vs ~2h scoring rerun) — the E003 checkpoints are the production set unless a later experiment separates them; per-label complementarity (dinov2 wins big/diffuse findings on CV) leaves a cheap two-arm ensemble as an open option.

### E005a-mm-crop
- **Hypothesis:** a fixed 140mm center crop (converted per study via `PixelSpacing`)
  beats full-frame input at the same 224px: one physical mm-per-pixel scale across
  all 71 field-of-view variants, every pixel spent on joint instead of thigh and
  background. Forum grounding (docs/rsna_brain.md §2.35-2.36): crop-geometry fixes
  measured as real effects (+0.0059, 10/12 labels same direction) where backbone
  scaling was a null; our own E004 tie points the same way.
- **Design notes:** crop happens in `load_volume` before percentile normalization
  (background excluded from the intensity range too); series without a usable
  `PixelSpacing` fall back to the full frame, frames smaller than the window are
  used whole, and `crop_mm` is stamped into checkpoints so inference reproduces the
  training geometry automatically. Known caveat: the rare both-knees-in-one-frame
  studies (§2.28) may center-crop between the joints — bounded by the full-frame
  fallback being per-series and the plane sit-out machinery. Runs as the crop
  margin of a 2x2 kernel run (bank x head type) shared with E005b: four CVs off
  identical folds, anchored by the full_frame/mean_max cell reproducing E003's
  0.771; margins attribute each lever, the fourth cell shows interaction.
- **Outcome:** _pending_

### E005b-perlabel-attention
- **Hypothesis:** per-label gated attention MIL pooling beats mean+max on the same
  features — each of the 12 findings learns its own weighting over slices, so
  few-slice findings (Lateral Meniscus 0.711, MCL 0.733, our measured floor) stop
  being diluted by a whole-stack mean. Controlled A/B: both arms fit from the SAME
  per-slice bank (resnet34 @224) on identical folds/seed — the delta is pooling
  alone. The bank refactor is validated in-run: the mean_max control must reproduce
  E003's 0.771 before the attention number counts.
- **Design notes:** shared gate trunk + per-label scorer/classifier vectors (~100k
  params vs the backbone's 21M), dropout + weight decay as the fitting-miner-noise
  guard. Attention weights double as per-finding "where it looked" overlays for the
  demo. Decision rule (E004's lesson): submit only if attention's CV macro beats the
  control by more than the per-repeat spread — CV gains may not transfer, and a CV
  null isn't worth a submission. Combiner untouched; learning it from OOF
  predictions is the follow-up (E005c). Runs as the pooling margin of the 2x2
  kernel run shared with E005a (see its design notes).
- **Outcome:** _pending_

### E006-dinov2-stack
- **Hypothesis:** stacking every individually-evidenced lever — DINOv2 features
  (E004: best CV recorded), end-to-end fine-tuning (student-teacher gap: model
  0.773 vs teacher labels' 0.887 gold agreement), 2.5D adjacent-slice triplets +
  140mm crop (forum §2.35-2.36 measured wins), per-label attention
  (thin-structure floor), and tier-weighted loss (Ryan's methodology) — beats
  every frozen number. **Attribution knowingly traded for speed** (team decision
  2026-09-02): each lever is separately evidenced, untangling a disappointment
  falls to the E005 single-lever rows plus this run's internal control — the
  frozen warm-up epochs' val score, a frozen-DINO-triplet baseline on the same
  split. Not gated on the E005 2x2 readout.
- **Design notes:** ViT runs at 224 via position-embedding interpolation
  (`image_size` override, stored in the checkpoint; fine-tuning at native 518
  would need a ~100GB pixel cache — E004's null was about frozen features at 518,
  a different lever than training). Heads cold-start: E005's resnet heads are
  512-d and don't fit 384-d ViT features. One decode pass; fixed 90/10 holdout
  seed 0, shared with all fine-tune-era rows. Staged unfreeze (2 frozen epochs,
  backbone at 1/10 head LR, cosine), augmentation without horizontal flips,
  best-epoch selection on val macro. Submission gate: holdout ensemble macro must
  clear 0.783 decisively.
- **Outcome:** _pending_
### E006a-tier-weighted-loss
- **Hypothesis:** weighting each label cell by Ryan's per-cell confidence weight
  (docs/modeling understanding/blended-labels-methodology.md: tier 1 explicit
  statement = full weight, tier 2 proxy-filled = partial, tier 3 ungrounded guess =
  reduced) beats treating every cell equally — the model learns hardest from cells
  grounded in explicit report language and is punished less for disagreeing with
  guesses. This is the "validated but deliberately unused" lever E003 staged;
  down-weighting, not downsampling — no rows are dropped (weight 0 is the limiting
  case). Controlled A/B: identical bank, folds, and head; only the loss weighting
  changes. Costs minutes, not hours — labels/loss changes refit from cached banks
  with zero decode.
- **Design notes:** `weighted_bce` multiplies per-cell BCE by the `__weight`
  companion and normalizes by total weight; `pos_weight` stays computed from
  unweighted targets (class imbalance and evidence confidence are separate
  concerns). Stratification and AUC stay unweighted — only training listens to
  confidence. Plumbed through all three trainers (frozen linear, attention MIL,
  fine-tune), so if it wins here E006 adopts it as its training loss.
- **Outcome:** _pending_

### E007-unified-multiplane
- **Outcome (2026-09-02): best val macro 0.720 — at epoch 2, i.e. while the backbone was still FROZEN. Unfreezing regressed the model and never recovered.** The trajectory is the whole diagnosis: frozen epochs climbed 0.683 -> 0.720 (cold head + plane embeddings still converging), then the instant the backbone unfroze the macro crashed to 0.644 and crawled back to only 0.684 by epoch 15. Classic catastrophic forgetting at the unfreeze boundary — backbone_lr 1e-4 is too hot for a DINOv2 ViT (the forum warned its fine-tunes are finicky; our first data point agrees). Per-label at best epoch: Effusion 0.776, Medial OA 0.763, Synovitis 0.764 / MCL 0.652, ACL 0.675 — thin-structure ordering roughly as before at a lower level. Submission gate (0.783) failed decisively — no inference run. What this does NOT yet refute: the unified architecture itself (its frozen trajectory was still climbing when the warm-up ended — frozen_epochs=2 starves a cold head) or fine-tuning in general (only this recipe at this LR). Next-recipe hypotheses, in order: backbone_lr 1e-5 with post-unfreeze warmup, frozen_epochs 5+, gradual unfreeze (last ViT blocks first), and/or warm-start now that E006 checkpoints exist.
- **Hypothesis (as run):** one model beats the E006 stack of three per-plane specialists at
  ~1/3 the parameters. Per study, a bag of 2.5D triplets from every available
  plane flows through one shared fine-tuned DINOv2 backbone, gets a learned
  per-plane embedding (the "which camera" signal, ~1k params replacing the
  clinical plane-prior matrix), and per-label attention weighs the whole bag —
  the learned per-label plane weighting DECISIONS.md #3 promised. Missing planes
  shrink the bag; the masked softmax renormalizes (what `combiner_weights` did by
  hand). The shared backbone sees ~13k series instead of ~4.4k per specialist.
  Matches the strong forum pipelines (docs/rsna_brain.md §2.35: attention over
  multi-plane sequence slots).
- **Design notes:** same geometry/labels/loss/split as E006 — the holdout
  comparison is paired, so the delta is the unification. Warm start from E006's
  best per-plane checkpoint (backbone + attention head transfer; embeddings start
  fresh) or cold — set from E006's result. Inference auto-detects the multiplane
  checkpoint kind; if E007 ships, the plane-prior combiner retires for unified
  checkpoints (log the per-label plane-attention weights vs the clinical prior
  table — rediscovered or refuted, either is a finding and a deck beat).
  Submission gate: clear 0.783 decisively AND beat E006's paired number.
- **Outcome:** _pending_
