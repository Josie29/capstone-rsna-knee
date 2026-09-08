# Capstone Demo Deck — Content

Source of truth for the slide deck (`deck.html` renders this content). Iterate here first,
then sync visuals. Numbers current as of CNN v3 (2026-09-08, public score 0.887); update
when a new experiment lands. All model builds are framed as team work — individual names
appear only as speaker labels.

**Format: 5 minutes total, three speakers, ~100 seconds each.**
Order: Josie (model) → Kelly (labeling) → Ryan (harness).
Kelly's section is still a drafted skeleton — marked `KELLY-TODO` — she owns the final
content. Ryan's slides (6–8) are final.

## Editing guide (Kelly & Ryan)

- Edit only your own slide sections; committing straight to main is fine.
- This file is the source of truth — change it first, then sync your slides in
  `deck.html` (your slides are the `<section>` blocks with your name in `data-speaker`;
  remove the amber draft/placeholder badge when done).
- House rules: no leaderboard/prize-money references (say "hidden test set"); model
  builds are team work — individual names appear only as speaker labels; ~100 seconds
  per speaker; verify rendering by opening `deck.html` in a browser before pushing.
- Q&A depth: don't cram technical detail into your timed slides — add an appendix slide
  instead and reference it if a question comes up. In `deck.html`, copy an appendix
  `<section>` block (they carry `data-appendix`) after A1; appendix slides sit outside
  the timed 5 minutes and number themselves A2, A3, ...
- Read "Iteration notes" at the bottom before restructuring — it records the standing
  design decisions.

---

## Slide 1 — Title (Josie, ~10s)

**Reading the Scans, Mining the Reports**

Weakly-supervised knee-MRI abnormality detection

- **0.887 macro AUC on the hidden test set** — trained with 58 ground-truth labels and 4,349 we mined ourselves
- RSNA Knee Abnormality Detection (Kaggle, 2026)
- Gauntlet AI Capstone — Josie Machalek · Kelly · Ryan

Stat tiles: 0.887 test-set AUC (hot) · 58 ground-truth studies · 1.3% of data labeled

Talk track: one sentence — "Three of us, one Kaggle competition: read a knee MRI, predict
12 findings, with almost no labels. I'll cover the model, Kelly the labels, Ryan the product."

---

## Slide 2 — The Result (Josie, ~50s)

**From 58 ground-truth labels to 0.887 macro AUC.**

Visual: experiment-progression chart (left) + compact experiment table (right).

Chart — macro AUC 0.65–0.90 over E001…E010 then the pipeline rebuild (v2, v3), two series:

- *Local validation* (muted): E003 0.771 → E004 0.783 → E005 0.786 → E006 0.748 →
  E007 0.720 → E008 0.785 → E009 0.803 → E010 0.799 → v2 0.840 → v3 0.848
  (E001/E002 had no local eval)
- *Submitted, hidden test* (teal diamonds): E001 0.691 · E002 0.692 · E003 0.773 ·
  E004 0.773 · E009 0.789 · v2 0.868 · **v3 0.887 — highlighted as the best, ring + hot label**
- Reference lines: 0.80 submission bar; 0.887 label ceiling (miner vs gold — Kelly setup;
  the v3 test point lands exactly on it, a coincidence worth a spoken beat)
- Annotated story points: the E006/E007 crash dip; the E004 CV-win-that-tied-on-test

Table — one row per experiment, verdict glyph (✓ paid / ∅ null / ✗ crash), v3 row highlighted:

| ID | lever | outcome |
|---|---|---|
| E001 | pipeline check, 58 gold labels | 0.691 test |
| E002 | clinical plane prior | +0.001 ∅ |
| E003 | mined labels 58 → 4,407 | **+0.081 test** ✓ |
| E004 | DINOv2 backbone | +0.012 CV → 0.000 test ∅ |
| E005 | per-label attention | +0.014 ✓ |
| E006 | fine-tune the stack | crash ✗ |
| E007 | unified multi-plane | crash ✗ |
| E008 | fixed fine-tune recipe | +0.037 ✓ |
| E009 | laterality + Sagittal T1 | +0.018 → 0.789 test ✓ |
| E010 | denser slice sampling | −0.004 ∅ |
| v2 | rebuild: EfficientNet, 2-level attention | 0.868 test ✓ |
| v3 | + non-fluid series, 24-slice depth | **0.887 test** ✓ |

Talk track: "Ten controlled experiments, then a
rebuild. Two crashes — fp16 gradients silently underflowing at backbone unfreeze, one
missing gradient scaler. One lesson — DINOv2 won our local eval and exactly tied on the
hidden test: local CV against mined labels measures agreement with the miner, not skill
at reading knees. Then we rebuilt the pipeline end to end — EfficientNet encoder, two
levels of attention, slices then series — 0.868; adding the non-fluid series and deeper
slice stacks took it to 0.887."

---

## Slide 3 — The Model (Josie, ~40s)

One knee MRI study in, 12 probabilities out.

- Input: a multi-planar knee MRI study; output: per-study probability for 12 binary
  findings — ligaments, menisci, osteoarthritis, effusion, fracture...
- Scored by macro ROC AUC; delivered as an offline Kaggle notebook, 9-hour cap

Visual: full-width architecture flow diagram (SVG) of the current best model (v3), left → right:

1. **Study** — ~5.5 series, 6 series types
2. **Select** — 5 series buckets (Sag / Cor / Ax fluid + Sag / Cor non-fluid)
3. **Resample** — cached volumes, 24 slices per series @224
4. **EfficientNet-B0** — one shared 2.5D slice encoder, ~5M params
5. **Slice attention** — pool each series' 24 slices → series embedding
6. **Series attention** — + bucket embedding; masked pool over available series
7. **12 probabilities** — mini output bars

Talk track: "Findings live in different planes and slices, and studies are ragged —
series can be missing. Attention twice: each series pools its own slices, then the study
pools its available series."

Handoff (footer): "The biggest lever isn't in this diagram — **+0.081** came from the
labels. Kelly."

---

## Slide 4 — KELLY-TODO: The Label Problem (~50s)

Placeholder — Kelly to write. Josie's handoff into this slide is "+0.081 came from the
labels", so it should open on the label problem (only 58 of 4,407 studies labeled).
Technical depth (mining prompts, validation method, ...) can go to an appendix slide
for Q&A — see the editing guide.

- The label problem: [KELLY]
- How the labels were mined: [KELLY]
- How mining quality was validated: [KELLY]
- (If useful: positives-per-finding chart data is in `docs/competition-notes.md`)

---

## Slide 5 — KELLY-TODO: Labeling, part 2 (~50s)

Placeholder — Kelly to write; ends her segment, hands off to Ryan.

- [KELLY]
- [KELLY]
- (If useful: the gold-audit counts live in `docs/experiments.md`)

---

## Slide 6 — Early wins (Ryan, ~33s)

**Early wins for AI-augmented systems**

Visual: the Nature Cancer paper on AI for breast cancer screening
(`assets/early-wins-breast-screening.webp`) mounted left, statement right.

- Single diagnosis
- Rapidly identifiable symptoms

---

## Slide 7 — The frontier (Ryan, ~33s)

**Now the frontier looks more…**

Visual: the review harness — three linked knee-MRI viewports with the attention overlay
lit, beside the findings panel (`assets/frontier-review-ui.webp`) — mounted left, the
three points right.

1. Broad: multiple, simultaneous diagnoses
2. Complex: disentangling injuries that often concur
3. Ambiguous: final diagnoses where often <u>panels of doctors disagree</u> on full scope of injury

---

## Slide 8 — The harness (Ryan, ~33s)

**Radiologists are fundamentally accountable.
Let’s give them the best thought partner to resolve the trickiest cases.**

- DICOM processing pipeline
  `:: Slice ordering, standard cropping, instance manifest`
- MSK reasoning skills
  `:: area tracing, diagnosis thresholds, response schema`
- Judgement panels: MMLMs + ViT
  `:: See scans through multiple architectures`
- Ambiguity resolution focused platform
  `:: Takes radiologists to the most difficult to resolve questions`

→ **Faster, more accurate diagnoses with less cognitive load.**

---

## Appendix A1 — ROC AUC, in one picture (backup — not in the 5 minutes)

For Q&A on the metric. Visual: ROC curve plot — true-positive rate vs false-positive
rate, chance diagonal, shaded area labeled AUC ≈ 0.89, one marked threshold point.

- Each finding gets a probability; ROC (Receiver Operating Characteristic) sweeps every
  possible decision threshold and plots true-positive rate vs false-positive rate
- AUC (Area Under the Curve) = the probability a random positive study is ranked above a
  random negative one — threshold-free, robust to class imbalance
- 0.5 = no signal (a coin flip ranks pairs correctly half the time); 1.0 = perfect ranking
- **Macro** = compute AUC per finding, average all 12 equally — a rare fracture counts as
  much as a common effusion
- Our 0.887: the model ranks a random abnormal study above a random normal one ~89% of
  the time, averaged across the 12 findings

---

## Iteration notes

- Timing: Josie slides 1–3 (~10s/50s/40s), Kelly 4–5 (~100s), Ryan 6–8 (~33s each, ~100s) — leaves ~60s buffer for transitions/demo latency in a 5-minute slot.
- The lever story (what paid / what didn't) lives entirely in slide 2's table + talk track — no separate ledger slide; Josie's segment stays at 3 slides to hold ~100s.
- Chart protocol caveat: the local-validation series mixes eval protocols (blended-cv for E003–E005, fixed 90/10 holdout E006–E010, 5-fold CV vs mined labels for v2/v3) — the caption says "local eval" generically; if a judge asks, the honest answer is the protocol followed the training regime.
- v2/v3 are the pipeline rebuild (branch `feat/knee-cnn-v3`): EfficientNet-B0 2.5D encoder, slice-then-series attention with bucket embeddings, 5-fold, cached volumes. v2 = 3 fluid buckets/depth 16; v3 = 5 buckets/depth 24. Local numbers: v2 mean val 0.840 (gold58 0.848), v3 mean val 0.848 (gold58 0.855). The v2→v3 rebuild jump vs E009 is uncontrolled (many changes at once) — no single-lever attribution claimed on the slide.
- The v3 test point (0.887) landing exactly on the old miner-vs-gold ceiling line (0.887) is coincidence — different quantities. Worth one spoken beat, not a claim.
- No "random guessing scores 0.5" framing on the main slides (too basic for a headline) — the 0.5 baseline belongs in appendix A1, where the metric is explained properly.
- Appendix slides sit after slide 8 in `deck.html`; they are backup material for Q&A, not part of the timed 5 minutes.
- Slide 2 frontloads the result by design: the audience gets the payoff before the two handoffs. v3 is highlighted as the best (ring + hot label + table-row accent).
- Josie's segment ends on the +0.081 handoff (slide 3 footer) into Kelly's section; Ryan closes with why-it-matters for the team.
- The war stories (gradient scaler, mirror-image knees) stay compressed to one line each in slide 2's talk track — no dedicated slides; full detail is Q&A material from `docs/experiments.md`.
- All E-series deltas and audit counts trace to `docs/experiments.md` (E001–E010). v2/v3 numbers come from the `feat/knee-cnn-v3` branch logs + the Kaggle submissions list — they have NO registry rows yet; add them when the branch merges. Chart data for the 58-gold positives (if Kelly wants it) is in `docs/competition-notes.md`.
- `deck.html` is synced to this 8-slide structure; the bottom-left corner shows the active speaker per slide (via `data-speaker`). Kelly's draft slides carry amber "draft"/"placeholder" badges — remove when she finalizes.
- Slides 6–7 mount a screenshot beside a statement. The two captures arrive on opposite grounds (the quoted paper is white to its own edges, the reviewer UI near-black), and neither is recoloured, inverted or filtered — that would alter Ryan's slides. Instead both get the same frame, radius and lift, over a surface matched to the capture (`.shot.paper` white, `.shot.viewport` `--panel-2`), so they read as one kind of object. Images live in `docs/presentation/assets/` as WebP rather than inline base64, to keep `deck.html` diffable.
- Ryan's slides carry no written talk track — the wording on the slides is his, taken verbatim from his own deck, and he speaks to it.
