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
| E009-laterality-sagt1 | 2026-09-03 | blended_v1 soft labels + tier weights / 4,407 / 4 sequence slots in one bag | E008 recipe + laterality normalization (canonical right-knee frame from patient-space geometry) + Sagittal T1 fourth slot | fixed 90/10 holdout (seed 0, paired with E008) | **0.803** (first run over the 0.80 bar; +0.018 vs E008) | **0.789** (new best, +0.016 over E003/E004's 0.773) | train ~3h47m T4 | this file (log below), PR #25, kernels train v13 |
| E008-resnet-unified-finetune | 2026-09-03 | blended_v1 soft labels + tier weights / 4,407 / all planes in one bag | unified resnet34 fine-tuned on 8-anchor 2.5D bags, per-label attention, AMP + no-decay-norms trainer fixes | fixed 90/10 holdout (seed 0); frozen warm-up epochs = internal baseline | **0.785** (fine-tune WORKED: +0.081 over its frozen stage) | — (below the 0.80 bar, not submitted) | train ~2h20m T4 | this file (log below), PR #23, kernels train v12 |
| E007-unified-multiplane | 2026-09-02 | blended_v1 soft labels + tier weights / 4,407 / all planes in one bag | ONE MultiPlaneModel: DINOv2 @224 fine-tuned, per-plane embeddings, per-label attention over the cross-plane bag — no combiner | fixed 90/10 holdout (seed 0, paired with E006) | 0.720 (best epoch = frozen warm-up; unfreeze regressed) | — (gate failed, not submitted) | train ~2h T4 | this file (log below), PR #22, kernels train v11 |
| E006a-tier-weighted-loss | 2026-09-02 | blended_v1 soft labels + per-cell `__weight` companions / 4,407 | best frozen E005 config, per-cell weighted BCE | blended-cv A/B vs unweighted, same bank/folds (zero decode — bank refit) | pending | pending | — | this file (log below), PR #20 |
| E006-dinov2-stack | 2026-09-02 | blended_v1 soft labels + tier weights / 4,407 / 3 fluid planes | DINOv2 ViT-S/14 @224 fine-tuned end-to-end on 2.5D triplets, crop140, per-label attention, tier-weighted loss | **new regime**: fixed stratified 90/10 holdout (full CV infeasible at one fine-tune per fold); internal control = frozen warm-up epochs on the same split | 0.748 ensemble (all three planes best at frozen epoch 2; unfreeze regressed) | — (gate failed, not submitted) | train ~1h22m T4 | this file (log below), PR #20/#21, kernels train v10 |
| E005a-mm-crop | 2026-09-02 | blended_v1 soft labels / 4,407 / 3 fluid planes | resnet34 @224 with fixed 140mm center crop (PixelSpacing-derived) | blended-cv, crop margin of the 2x2 run shared with E005b | +0.003 on mean_max (0.774 vs 0.771), +0.001 on attention — weak/null | — (not separately submitted) | 2x2 run ~5h17m T4 | this file (log below), PR #18, kernels train v9 |
| E005b-perlabel-attention | 2026-09-02 | blended_v1 soft labels / 4,407 / 3 fluid planes | resnet34 @224 + per-label gated attention MIL heads, plane-prior combiner | blended-cv, pooling margin of the 2x2 run shared with E005a | **0.785-0.786** (+0.014 over mean_max) | pending decision | (shared 2x2 run) | this file (log below), PR #17/#18, kernels train v9 |
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
- **Outcome (2026-09-03, kernel v9, ~5h17m): weak-to-null.** crop140/mean_max 0.774 (0.773-0.775) vs full_frame/mean_max 0.771 — +0.003, ~2x the repeat spread, marginal; and on top of attention the crop adds only +0.001 (0.786 vs 0.785), pure noise. The forum's crop win did not replicate at our operating point — possibly because attention pooling already ignores background slices, doing part of the crop's job. No interaction. Keep crop140 as harmless default (it never hurt and shrinks decode I/O slightly), but it is not a lever. Note: the 2x2's anchor validated perfectly — full_frame/mean_max reproduced E003's 0.771 exactly, proving the per-slice bank refactor lossless.
- **Hypothesis (as run):** a fixed 140mm center crop (converted per study via `PixelSpacing`)
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
- **Outcome (2026-09-03, kernel v9): the real winner of the 2x2 — attention 0.785/0.786 vs mean_max 0.771/0.774, +0.012-0.014, ~14x the ±0.001 repeat spread, and broad-based per label** (ACL 0.769->0.792, Medial OA 0.820->0.836, Baker's 0.757->0.789, Effusion 0.815->0.829, MCL 0.728->0.737 — the "real effect" signature). crop140/attention 0.786 is the best CV ever recorded, edging E004's 0.783 — and unlike E004 this is a within-family change (same resnet backbone as E003, whose CV tracked the LB within 0.002), so transfer odds are meaningfully better than the backbone-swap case. Submission decision pending (checkpoints live in v9's output; retrievable via UI or a ~1.5h refit run). Also recontextualizes E006/E007: the triplet input (9 slices) likely cost information relative to full-slice attention — the frozen full-slice attention models are the strongest frozen models we have.
- **Hypothesis (as run):** per-label gated attention MIL pooling beats mean+max on the same
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
- **Outcome (2026-09-02): ensemble val macro 0.748 — and every per-plane specialist's best epoch was a FROZEN warm-up epoch** (sag 0.700, cor 0.690, ax 0.710, all epoch 2). All three showed the identical signature E007 showed: climbing through the frozen epochs, an immediate crash the moment the backbone unfroze (e.g. axial 0.710 -> 0.625), then a slow crawl that never recovered the frozen peak. **Combined with E007's identical trajectory, the diagnosis is definitive: the fine-tune recipe (backbone_lr 1e-4 on a DINOv2 ViT, hard unfreeze, no warmup) is the culprit — four independent crashes, zero exceptions — not the stack, not the unified architecture.** The shipped "finetuned" ensemble is therefore really a frozen-backbone triplet ensemble; its 0.748 beat E007's single unified model (0.720, whose cold head got only 2 frozen epochs). Per-label: Fracture 0.801, Effusion 0.796, Medial/Lateral OA 0.785/0.786 / PF OA 0.706, Lateral Meniscus 0.709, MCL 0.700. Gate (0.783) failed — no submission. Note the frozen curves were still rising when the 2-epoch warm-up ended everywhere: longer frozen training is free upside in any rerun. Runtime surprise: ~1h22m total (decode ~50 min faster than feared at 224px).
- **Hypothesis (as run):** stacking every individually-evidenced lever — DINOv2 features
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
- **Outcome (2026-09-02): best val macro 0.720 — at epoch 2, i.e. while the backbone was still FROZEN. Unfreezing regressed the model and never recovered.** The trajectory is the whole diagnosis: frozen epochs climbed 0.683 -> 0.720 (cold head + plane embeddings still converging), then the instant the backbone unfroze the macro crashed to 0.644 and crawled back to only 0.684 by epoch 15. Classic catastrophic forgetting at the unfreeze boundary — backbone_lr 1e-4 is too hot for a DINOv2 ViT (the forum warned its fine-tunes are finicky; our first data point agrees). Per-label at best epoch: Effusion 0.776, Medial OA 0.763, Synovitis 0.764 / MCL 0.652, ACL 0.675 — thin-structure ordering roughly as before at a lower level. Submission gate (0.783) failed decisively — no inference run. What this does NOT yet refute: the unified architecture itself (its frozen trajectory was still climbing when the warm-up ended — frozen_epochs=2 starves a cold head) or fine-tuning in general (only this recipe at this LR). E006's log (read after this entry was first written) shows the identical crash in all three per-plane specialists — recipe confirmed as the sole culprit. Next-recipe hypotheses, in order: backbone_lr 1e-5 with post-unfreeze warmup, frozen_epochs 5+ (curves were still climbing at the cutoff), gradual unfreeze (last ViT blocks first), and/or warm-start from the E006 checkpoints.
- **Gold audit (error-report kernel v3, 2026-09-03), run on this checkpoint:** over the 696 gold cells — both right 366, **model error while labels were fine 195 (28%)**, model caught a label error 66 (9.5%), both wrong 69. The 3:1 model-error-to-label-error ratio says the model lever, not the labels lever, is still where the points are (caveat: measured on our weakest model; the ratio should improve with stronger checkpoints). Independently confirmed the miner's gold agreement at **0.887 macro** — Ryan's claimed number reproduced exactly. The 66 caught-label-error cells are direct input for miner v2 and a deck beat ("the image model catches labeling errors").
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

### E009-laterality-sagt1
- **Hypothesis:** the two input-level deltas vs the documented 0.92-tier pipelines
  are worth real AUC on top of the working E008 recipe. (a) Laterality: left/right
  knees are mirror images and we feed both raw, so medial/lateral anatomy swaps
  sides per knee — canonicalizing (side from image-center patient-x, 97% resolvable
  per forum-validated method; coronal/axial flip horizontally, sagittal reverses
  slice order; bilateral/ambiguous volumes left unmirrored) should move the four
  side-specific thin-structure labels most (E008: MCL 0.722, MedMen 0.746, ACL
  0.750, LatMen 0.766). (b) Sagittal T1 fourth slot: marrow/bone contrast — read =
  OA/Fracture/Contusion move. Provenance: checkpoint stamps laterality_normalized
  so inference reproduces the frame. Single arm, paired vs E008's 0.785; >=0.80
  clears the submission bar.
- **Outcome (2026-09-03, kernel v13, ~3h47m): 0.803 (epoch 15) — the first model over the 0.80 submission bar, +0.018 over E008 on the identical split.** Both hypotheses paid, and the frozen stage alone improved (0.642 -> 0.728 by epoch 4, vs E008's 0.613 -> 0.704), so the input fixes help independent of fine-tuning. Laterality tally: 8,775 right / 7,736 left_mirrored / 558 ambiguous / 0 no_geometry across 17,069 volumes — 96.7% resolved, matching the forum-validated ~97%. Per-label deltas vs E008: the laterality read mostly landed (MCL 0.722->0.758, Medial Meniscus 0.746->0.770, ACL 0.750->0.768; Lateral Meniscus dipped 0.766->0.750), and the T1 read landed (Fracture 0.834->0.860, Contusion 0.810->0.823, Lateral OA 0.796->0.816, Medial OA 0.834->0.844). Biggest single mover: Baker's 0.788->0.845. Curve plateaus at ~0.801-0.803 over epochs 12-18 (fully annealed). Clears the bar -> publish/infer/submit path opens. **LB (2026-09-04, submission 56003207): 0.789** — new team best, +0.016 over the 0.773 that stood since E003/E004. Holdout->LB gap -0.014 (E004's frozen-era gap was -0.010), so the local ruler maps honestly and the fine-tune-era gains transfer.

### E008-resnet-unified-finetune
- **Outcome (2026-09-03, kernel v12, ~2h20m): the crash is CURED and fine-tuning finally works — best val macro 0.785 (epoch 17), the strongest model on the holdout by a wide margin.** The trajectory tells the whole story: frozen epochs climbed 0.613 -> 0.704, and at the unfreeze (epoch 5) the model IMPROVED to 0.724 and kept climbing monotonically to 0.785 — the exact opposite of the four crashes, validating the GradScaler + recipe + input fixes. Paired comparison: E006's frozen triplet ensemble scored 0.748 on this same split; E008 is a SINGLE model at +0.037 over it. Per-label: Fracture/Medial OA 0.834, Effusion 0.813, Contusion 0.810, Baker's 0.788 / MCL 0.722, Medial Meniscus 0.746, ACL 0.750 — thin structures still trail. Against the bar: 0.785 < 0.80, so no submission; the curve is nearly flat over the last 6 epochs under the fully-annealed cosine, so more epochs alone won't close the gap. Candidate next increments: more anchors (the input is still subsampled vs full slices), the fourth sequence slot (Sagittal T1), a longer schedule with restarts, and whatever the E008 gold audit says. **Gold audit (report kernel v4):** 696 cells — both right 426 (was 366 on E007), model error/labels fine 135 (was 195), model caught label error 51 (was 66), both wrong 84 (was 69). Model-error:label-error ratio ~2.6:1 (was ~3:1) — model levers still dominate. Model vs gold 0.788 (consistent with the 0.785 holdout); miner teacher ceiling re-confirmed at 0.887, and the model is closing on it (model vs miner labels 0.841).
- **Hypothesis:** with the two trainer defects fixed and the input gap closed, the
  fine-tune finally adds instead of crashing, and >=0.80 holdout macro is in
  range. The three changes from E007, each independently motivated: (1) fp16
  autocast ran WITHOUT a GradScaler in every fine-tune so far — small gradients
  (exactly the backbone's) silently underflow on a T4; likely a contributor to
  all four unfreeze crashes. Fixed, plus weight decay no longer applies to
  norms/biases (standard recipe; ViT-relevant if we retry DINOv2). (2) Input: 8
  anchors over a (0.1, 0.9) window (~24 slices) — the E005 2x2 showed full-slice
  attention beats 9-slice triplets by +0.014-class margins; E006/E007 fine-tuned
  the weaker format. (3) Backbone: resnet34 — fine-tuned CNNs are the community's
  0.87 plateau, CNN fine-tuning is robust at these learning rates, and frozen
  DINOv2 bought nothing at the LB (E004) while crashing twice in training.
- **Design notes:** unified MultiPlaneModel, cold start, tier weights, crop140
  (measured harmless), frozen_epochs=4 (E007's cold head was still climbing at
  2), epochs=18. Success reads, in order: frozen stage approaching ~0.77-0.78
  validates the input fix; the unfreeze ADDING validates the AMP fix; >=0.80
  clears the submission bar (team policy 2026-09-03). A recurring crash despite
  the fixes points at something deeper and hands the baton to the error-analysis
  tool.
- **Outcome:** _pending_
