# Capstone Demo Deck — Content

Source of truth for the slide deck (`deck.html` renders this content). Iterate here first,
then sync visuals. Numbers current as of CNN v3 (2026-09-08, public score 0.887); update
when a new experiment lands. All model builds are framed as team work — individual names
appear only as speaker labels.

**Format: 5 minutes total, three speakers, ~100 seconds each.**
Order: Josie (model) → Kelly (labeling) → Ryan (harness).
All three sections are final content — no `TODO` placeholders remain. Each speaker
owns their own slides.

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

**Knee MRI Diagnostics: From Mined Reports to a Harnessed Model**

Weakly-supervised knee-MRI abnormality detection

- **0.887 macro AUC on the hidden test set** — trained with 58 ground-truth labels and 4,349 we mined from the radiologists' own reports
- RSNA Knee Abnormality Detection (Kaggle, 2026)
- Gauntlet AI Capstone — Josie Machalek · Kelly He · Ryan Highfill

Stat tiles: 0.887 test-set AUC (hot) · 570 GB of image data · 1.3% of studies labeled

Talk track: one sentence — "Three of us, one Kaggle competition: read a knee MRI, predict
12 findings, with almost no labels. I'll cover the model, Kelly the labels, Ryan the product."

---

## Slide 2 — The Result (Josie, ~50s)

**Label mining and controlled model experiments delivered 0.887 macro AUC.**

Visual: experiment-progression chart (left) + compact experiment table (right).

Chart — macro AUC 0.65–0.90, one condensed series (each experiment's headline number,
test score where submitted, local eval otherwise — deliberately not separated; this is
storytelling, not peer review):

- E001 0.691 → E002 0.692 → E003 0.773 → E004 0.773 → E005 0.786 → E008 0.785 →
  E009 0.789 → E011 0.868 → **E012 0.887 — highlighted as the best, ring + hot label**
- Reference lines: 0.80 submission bar; 0.887 label ceiling (miner vs gold — Kelly setup;
  the E012 point lands exactly on it, a coincidence worth a spoken beat)
- Crashed experiments (E006/E007) and the E010 null are omitted from the slide — the ID
  gaps stay visible on the x-axis; full history in `docs/experiments.md`

Table — one row per experiment, Data and Model columns to show which lever each pulled
("—" = unchanged from the row above), E012 highlighted:

| ID | data | model | AUC |
|---|---|---|---|
| E001 | 58 gold labels | frozen ResNet-34 | 0.691 |
| E002 | — | + clinical plane weighting | 0.692 |
| E003 | 4,407 mined labels | frozen ResNet-34 | 0.773 |
| E004 | — | frozen DINOv2 ViT | 0.773 |
| E005 | — | + per-label attention | 0.786 |
| E008 | — | fine-tuned unified ResNet-34 | 0.785 |
| E009 | + laterality fix, + Sagittal T1 | — | 0.789 |
| E011 | improved mined labels | EfficientNet-B0, 2-level attention | 0.868 |
| E012 | + non-fluid series, 24 slices | — | **0.887** |

Talk track: "Twelve experiments, every one a controlled A/B. Two aren't on this chart —
spectacular fine-tune crashes, ask me in Q&A. Two lessons in the flat spots: a fancier
DINOv2 backbone moved nothing, E004; and the fine-tune only paid once the recipe was
fixed. The steps that mattered: mined labels, +0.081. Attention pooling. The laterality
fix — knees are mirror images, half our data was anatomically backwards. Then the
rebuild: EfficientNet with attention over slices then series, 0.868 — and non-fluid
series with deeper sampling took it to 0.887."

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

## Slide 4 — The Label Problem (Kelly, ~50s)

**Only 58 of 4,407 studies are labeled.** The other 4,349 labels were mined from the
radiologists' reports that ship with every exam.

- Read, then score: readers extract what each report *says* per finding, with a verbatim
  quote that is mechanically verified; one deterministic function turns readings into
  probabilities — same report, same label
- Visual: three-block strip (measured on the 58 gold exams) — **denied ≈ 0.00** ·
  **silent 0.07–0.50** · **asserted 0.41–0.79**
- The trap: **silence is not absence.** A denial is an observation; silence is a missing
  observation, and findings a report never mentions are present up to half the time.
  ~⅓ of training cells are silent — scoring them 0 would poison the training set

Talk track: "That +0.081 came from solving our label problem: fifty-eight ground-truth
studies out of four and a half thousand. But every exam ships with the radiologist's
report — so we mined those. Readers extract what each report says, with verbatim quotes
we verify mechanically; then a single deterministic function turns readings into
probabilities. The trap is what reports *don't* say. Measured on our gold exams: when a
report denies a finding, it's essentially never there. When it's silent, the finding is
present up to half the time — radiologists don't dictate every incidental. A third of
our cells are silent, so scoring silence as zero would poison the set. Denials go near
zero, assertions go high, and silence gets the measured rate given silence, adjusted per
clinic — because a clinic that never mentions a finding tells you nothing by omitting it."

---

## Slide 5 — One Real Exam (Kelly, ~50s)

**Watch one report become a training row.** Interactive: a real English report + this
study's actual label row, driven by one button (3 clicks; clicks inside the demo don't
advance the deck).

- Click 1 — reader highlights: tear **asserted**, ligaments/effusion/cartilage **denied**,
  four findings **silent**
- Click 2 — observed cells score (denied cells keep 0.04–0.10, never exactly 0);
  silent cells stay open
- Click 3 — the exam's real axial MRI strip appears: the report never mentions a Baker's
  cyst, the image reader found one ("well-defined ovoid posteromedial cyst", slices 9–12,
  amber box). That cell fills at 0.30 with training weight 0.6; image-cleared cells drop;
  unread silence keeps the prior at weight 0.3
- Validation footline: mined labels score **0.887 macro AUC vs the 58 gold exams** — the
  ceiling line on Josie's chart; gold never trains, it only grades

Talk track: "Here's a real exam. [click] The reader marks the meniscus tear as asserted
and the ligaments as denied — every highlight is a verbatim quote. [click] Observed cells
score directly; even denials keep a little probability, because radiologists are
occasionally wrong. But four findings are silent — the report can't answer them. [click]
So we looked. Our first model flagged silent cells where the pixels disagreed with the
prior, and an image reader checked them. This report never mentions a Baker's cyst — and
there it is on the actual MRI. That cell moves up and earns more training weight; cells
the reader cleared move down. How do we know the mining works? The mined labels score
0.887 against the fifty-eight gold exams — that's the ceiling line on Josie's chart —
and those gold exams never enter training. They're the ruler, not the teacher. Ryan."

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

No footer on this slide: the old deadline/score `footer-note` was deck chrome inherited
from the placeholder closing slide, and was dropped so slide 8 carries Ryan's content only.

---

## Appendix A1 — Divider

**Appendix**

Backup material for Q&A — not part of the timed 5 minutes.

- A2 — ROC AUC, the evaluation metric explained
- (Kelly / Ryan: add technical-depth slides after, per the editing guide)

---

## Appendix A2 — ROC AUC, in one picture (backup — not in the 5 minutes)

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
- Slide 2's chart is one condensed series by deliberate choice: test score where submitted (E001–E004, E009, E011, E012), local eval otherwise (E005 0.786, E008 0.785). Crashes (E006/E007) and the E010 null are omitted; the x-axis keeps the ID gaps so nothing is hidden, just decluttered. If a judge asks about the mix or the gaps, the full per-protocol numbers are in `docs/experiments.md`.
- E011/E012 on the slide = the pipeline rebuild (branch `feat/knee-cnn-v3`, internally "cnn v2/v3"): EfficientNet-B0 2.5D encoder, slice-then-series attention with bucket embeddings, 5-fold, cached volumes. E011 = 3 fluid buckets/depth 16 (0.868 test); E012 = 5 buckets/depth 24 (0.887 test). The IDs are presentational until their registry rows land. The rebuild jump vs E009 is uncontrolled (many changes at once) — no single-lever attribution claimed on the slide.
- The E012 test point (0.887) landing exactly on the old miner-vs-gold ceiling line (0.887) is coincidence — different quantities. Worth one spoken beat, not a claim.
- No "random guessing scores 0.5" framing on the main slides (too basic for a headline) — the 0.5 baseline belongs in appendix A1, where the metric is explained properly.
- Appendix slides sit after slide 8 in `deck.html`; they are backup material for Q&A, not part of the timed 5 minutes.
- Slide 2 frontloads the result by design: the audience gets the payoff before the two handoffs. v3 is highlighted as the best (ring + hot label + table-row accent).
- Josie's segment ends on the +0.081 handoff (slide 3 footer) into Kelly's section; Ryan closes with why-it-matters for the team.
- The war stories (gradient scaler, mirror-image knees) stay compressed to one line each in slide 2's talk track — no dedicated slides; full detail is Q&A material from `docs/experiments.md`.
- All E-series deltas and audit counts trace to `docs/experiments.md` (E001–E010). v2/v3 numbers come from the `feat/knee-cnn-v3` branch logs + the Kaggle submissions list — they have NO registry rows yet; add them when the branch merges. Chart data for the 58-gold positives (if Kelly wants it) is in `docs/competition-notes.md`.
- `deck.html` is synced to this 8-slide structure; the bottom-left corner shows the active speaker per slide (via `data-speaker`). No slide carries an amber "draft"/"placeholder" badge any more — the `.tbd` style stays in the stylesheet for the next draft slide.
- Ryan's screenshots arrive on opposite grounds — the quoted paper is white to its own edges, the review harness is near-black to its own edges. Neither is recoloured, inverted or filtered: both get the same mount (hairline border, 10px radius, same lift) over a surface matched to the capture (white under the paper, `--panel-2` under the UI), so they read as one kind of object while staying unaltered. Assets live in `docs/presentation/assets/` as WebP rather than inline base64, so `deck.html` stays diffable (110 KB + 253 KB; the UI capture is 2.5 MB as PNG).
- Ryan's slides (6–8) carry no written talk track — his wording is the slide text itself, taken verbatim from his source deck; don't paraphrase it when editing.
