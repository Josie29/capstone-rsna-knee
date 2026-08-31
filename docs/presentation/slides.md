# Capstone Demo Deck — Content

Source of truth for the slide deck (`deck.html` renders this content). Iterate here first,
then sync visuals. `[TBD]` marks numbers that don't exist yet — fill as results land.

---

## Slide 1 — Title

**Reading the Scans, Mining the Reports**

Weakly-supervised knee-MRI abnormality detection

- RSNA Knee Abnormality Detection (Kaggle, 2026, $77k)
- Gauntlet AI Capstone — Josie Machalek

---

## Slide 2 — The Task

One knee MRI study in, 12 probabilities out.

- Input: a multi-planar knee MRI study (~5.5 series: sagittal / coronal / axial, fluid-sensitive and non-fluid)
- Output: per-study probability for 12 binary findings — ACL, MCL, medial + lateral meniscus, 3 osteoarthritis compartments, effusion, synovitis, Baker's cyst, contusion, fracture
- Scored by macro-averaged ROC AUC — every finding counts equally, rare or common
- Delivered as a Kaggle notebook: no internet, 9-hour runtime cap

Scale stat row: 4,407 train studies · 24,371 series · 247 GiB · ~1,300 test studies

---

## Slide 3 — The Catch

**Only 58 of 4,407 training studies are labeled. 1.3%.**

- The other 98.7% carry only a free-text radiology report — written in ~a dozen languages across ~20 countries
- The report field does NOT exist at test time — text is a training tool, not an input
- So the real task: mine labels out of multilingual reports, then train an imaging model on what you mined
- The 58 gold studies are the only ground truth to check the mining against — n=58 validation AUC is too noisy to steer by alone

Visual: bar chart — positives per finding among the 58 gold studies (Effusion 35 ... MCL 9).

---

## Slide 4 — The Plan (four stages)

Pipeline overview:

1. **Mine** — multilingual report → 12 pseudo-labels per study (validated against the 58 gold studies)
2. **Select** — pick the right series per study: 6 series types (3 planes × fluid/non-fluid), decode mixed DICOM formats, normalize
3. **Train** — pretrained 2D backbones per plane + fusion to study-level predictions; trained on pseudo-labels on Kaggle GPUs
4. **Submit** — offline inference notebook, weights attached as Kaggle datasets, budgeted under 9 hours

Key sequencing insight: mining quality sets the ceiling on everything downstream — and it can be built and iterated without touching a single DICOM.

---

## Slide 5 — Stage 1: Mine the reports

The ceiling-setter.

- Multilingual encoder (Hugging Face transformers) reads each report, emits 12 binary pseudo-labels
- Regex/keyword rules rejected: ~12 languages × 12 findings × negation handling = brittle
- Translate-then-English-model rejected: adds a failure stage; multilingual encoders skip it
- Validation: agreement with the 58 gold studies, per finding
- Every 1% of label noise here is noise the imaging model trains on

`[TBD]` miner agreement vs gold set (per-column table or single headline %)

---

## Slide 6 — Stages 2+3: From DICOM to model

- Series metadata collapses to 6 types: 3 planes × fluid-sensitive/non-fluid — every study has all 3 planes; selection (which duplicate to use), not availability, is the decision
- Mixed compression formats (JPEG Lossless, JPEG 2000, uncompressed) — half the data won't decode without the right handlers
- Per-plane 2D backbones (timm, ImageNet-pretrained) over slices → aggregate to series → fuse planes → 12 study-level probabilities
- Trained on Kaggle GPUs (data pre-mounted — skips the 247 GiB download), tracked in Weights & Biases

`[TBD]` architecture final choice + cross-validation AUC

---

## Slide 7 — Stage 4: The 9-hour box

- No internet: code + weights ship as Kaggle datasets attached to the notebook
- ~1,300 studies × ~30 slices × several series must fit in 9 hours — inference cost budgeted before choosing architecture
- Efficiency track: a second prize pool scores AUC per unit runtime — being fast is itself a leaderboard
- Repo is source of truth: notebooks are thin wrappers over `src/knee/`, pushed via API, never edited in the web UI

---

## Slide 8 — Results

Scoreboard (all `[TBD]` until runs land):

| Metric | Value |
| --- | --- |
| Miner agreement vs 58 gold studies | `[TBD]` |
| Cross-validation macro AUC | `[TBD]` |
| Public leaderboard macro AUC | `[TBD]` |
| Inference runtime | `[TBD]` / 9h |

Benchmark context: sample submission (all 0.5) scores 0.5 AUC; competition benchmark on LB.

---

## Slide 9 — Why this matters

- Expert annotation is the bottleneck of medical AI — radiologist time is the most expensive input in the pipeline
- But hospitals already have millions of studies with reports attached — weak supervision from reports is how those archives get unlocked
- Multilingual mining matters: real-world archives span languages and sites (this dataset: 22 sites, ~20 countries)
- The pattern generalizes: any paired image+report modality (chest X-ray, CT, pathology) can use the same recipe

---

## Slide 10 — What's next

- **Better miner**: LLM-based labeling vs local encoder (pending competition-rules check on sending report text to external APIs)
- **Self-training loop**: retrain the miner/imaging model on each other's confident predictions
- **3D architectures**: MONAI volumetric models vs per-slice 2D — does true 3D context pay for its runtime?
- **Per-column calibration**: macro AUC rewards ranking each finding well independently
- **Efficiency track**: distillation / pruning for the runtime-scored prize pool
- Final deadline: 2026-10-22

---

## Iteration notes

- Results slide is a placeholder scoreboard by design — the deck stays honest about status.
- Order of slides 5–7 can compress to one slide if time is tight (demo slots are usually 5 min).
- Chart data on slide 3 comes from `docs/competition-notes.md` (positives per column among the 58).
