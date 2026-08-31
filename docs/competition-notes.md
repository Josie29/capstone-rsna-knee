# RSNA Knee Abnormality Detection — competition notes

Read off the Kaggle Overview / Data / Rules tabs 2026-08-31. Archive size confirmed via
the API the same day (see Data). Train row counts still need the data in hand.

## Task

Per-**study** probability for each of 12 binary findings, from multi-planar knee MRI plus
(training only) the free-text radiology report.

## Labels — submission column order

`StudyInstanceUID, ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA,
PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture`

ACL/MCL = ligament injury. OA columns are the three compartments (medial tibiofemoral,
lateral tibiofemoral, patellofemoral). `Baker's` = Baker's cyst; `Contusion` = bone bruise.
Note the apostrophe and the spaces — column names must match exactly.

## The central problem

> "Only a small subset of training studies carry per-condition labels. We also provide the
> original text of the radiology report from which you may wish to derive the labels for
> the remaining studies."

So this is **weak supervision**: the bulk of the training signal has to be mined out of
multilingual free-text reports first. **The report field is NOT provided at test time.**
Text is a label-generation and auxiliary-supervision tool, not a test-time input.

Measured from `train.csv` 2026-08-31 — "small subset" is smaller than it sounds:

- **58 of 4,407 studies are labeled (1.3%).** All 12 columns are populated together, so
  it is 58 fully-labeled studies, not 4,407 partially-labeled ones. No column has extra
  coverage to exploit.
- All 4,407 studies have report text. Every unlabeled study is reachable only through text.
- Positives among the 58, per column: Effusion 35, Synovitis 27, Medial Meniscus 26,
  ACL 24, Lateral Meniscus 23, PF OA 21, Contusion 19, Fracture 18, Medial OA 15,
  Baker's 12, Lateral OA 11, MCL 9. Reasonably balanced, but n=58 means a
  validation AUC computed on this set alone has enormous variance — it cannot be the
  primary model-selection signal.

Consequence for sequencing: report-mining quality sets the ceiling on everything
downstream, and it can be built and iterated **without a single DICOM**. The 58 labeled
studies are the only ground truth available to check the miner against.

## Data

| File | Contents |
| --- | --- |
| `train.csv` | 4,407 rows, one per study: `StudyInstanceUID`, `Report` (free text, many languages), 12 binary label columns (only 58 rows populated). 5.4 MiB. |
| `train_series.csv` | 24,371 rows, one per series: `StudyInstanceUID`, `SeriesInstanceUID`, `Fluid_Sensitive` (0/1), `Fat_Suppression` (0/1), `Anatomical_Plane`. 3.3 MiB. |
| `train_series/` | `<StudyUID>/<SeriesUID>/<SOPUID>.dcm` — one slice per file. |
| `test.csv` | `StudyInstanceUID` only. ~1300 studies at scoring time; the 3 shipped rows are placeholders. |
| `test_series.csv`, `test_series/` | Same schema/layout as train, swapped in at scoring. |
| `sample_submission.csv` | All 12 columns at 0.5. |

- **Archive size: 265,018,885,676 bytes = 247 GiB compressed.** Confirmed 2026-08-31 from
  the `Content-Length` of the Kaggle `competitions/data/download-all` endpoint. The naive
  download-then-unzip path needs ~500 GiB at peak; streaming the unzip avoids the double.
- 4,407 train studies / 24,371 train series = **5.5 series per study** on average.
  Planes: Sagittal 9,864, Coronal 8,609, Axial 5,898 — so most studies have more than one
  series per plane, and series selection is a real preprocessing decision, not a given.
- Series metadata is far more uniform than the free-text `SeriesDescription`s suggest.
  Measured on train_series.csv 2026-08-31: `Fat_Suppression == Fluid_Sensitive` on every
  row, so only **6 series types** exist (3 planes x fluid/non-fluid — see
  `src/knee/series.py:SeriesType`), and only 12 per-study compositions. The Data tab
  warns the two flags are "not necessarily equivalent for every case", i.e. they can
  diverge on hidden test data — so key logic on `Fluid_Sensitive`, and use
  `Fat_Suppression` only to prefer the train-like variant when a study offers both. Every study has
  all 3 planes, ≥1 fluid-sensitive, and ≥1 non-fluid series; 94% have a fluid-sensitive
  sagittal. Studies with >6 series carry duplicate types, so series *selection* is the
  preprocessing decision, not series availability.
- 19 primary + 3 additional contributing sites across ~20 countries.
- Series: 20–45 slices typical, median 30, long tail to a few hundred.
- Mixed transfer syntaxes: Explicit VR LE (uncompressed), JPEG Lossless, JPEG 2000,
  Implicit VR LE. **pydicom needs `pylibjpeg`/`gdcm` handlers or ~half the data won't decode.**
- DICOMs stripped to an allowlist of 86 tags. Intensities, orientation, resolution all vary.
- Prevalence is **not** guaranteed to match across train / public LB / private LB.

## Evaluation

**Macro-averaged ROC AUC** over the 12 targets — unweighted mean of per-column AUC. Rare
classes count as much as common ones, so per-column calibration matters more than a single
global loss.

### Efficiency track (3 extra prizes)

Minimize `Efficiency = AUC / (Benchmark - maxAUC) + RuntimeSeconds / 32400`, where
Benchmark is the `sample_submission.csv` score and maxAUC is the best private-LB AUC.
Eligible only if selected as a final submission and ranked above the benchmark. A
submission can win both tracks. Daily public efficiency leaderboard notebook exists.

## Submission mechanics — notebook-only code competition

- CPU **and** GPU notebooks: ≤ 9 hours runtime (32,400s — the efficiency denominator).
- **Internet access disabled.** Pretrained weights must be attached as Kaggle datasets/models.
- Freely & publicly available external data and pretrained models are allowed.
- Output must be named `submission.csv`.
- 5 submissions/day; 2 final submissions selected; max team size 5.

~1300 test studies × ~median 30 slices × several series in 9 hours is the real constraint.
Budget inference cost before choosing an architecture.

## Timeline (all 11:59 PM UTC)

| Date | Milestone |
| --- | --- |
| 2026-07-30 | Start |
| 2026-10-15 | Entry deadline + team merger deadline |
| 2026-10-22 | Final submission deadline |
| 2026-11-05 | Winners' requirements (training code, video, method description) |
| 2026-11-29 – 12-03 | RSNA 2026, Chicago — recognition event, fee waived |

## Rules that constrain this repo

- **Data security (2.4.b): do not redistribute Competition Data to non-participants.**
  Nothing derived from `train.csv` — report text, label CSVs, extracted DICOM pixel data —
  goes in this public repo. `.gitignore` excludes all of `data/`.
- Winner license is **CC-BY-NC 4.0**; winners must also publish a video, an open-source
  code link, and publicly distributable weights.
- Data use governed by the RSNA MIRA license: <http://rsna.org/mira-license>
- One Kaggle account per person; no submitting from multiple accounts.

## Sources

- <https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview>
- <https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data>
- <https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules>
