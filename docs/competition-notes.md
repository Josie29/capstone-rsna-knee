# RSNA Knee Abnormality Detection — competition notes

Facts gathered 2026-08-31 from the RSNA challenge page and press coverage. **Everything
here needs confirming against the Kaggle Overview, Data, and Evaluation tabs before it
drives a design decision** — the Kaggle pages render client-side and could not be read
without a logged-in browser.

## Task

Detect and classify clinically important abnormalities on multi-planar knee MRI. Reported
as 12 findings spanning ligament injury, meniscal damage, cartilage loss, bone marrow
lesions, effusion, synovitis, and cysts. **Exact label list: unconfirmed.**

The distinguishing feature: this is the first RSNA challenge to pair images *with*
radiology report text, and the reports are multilingual — sources say ~9 to ~12 languages.
Labels were derived from those reports by experts.

## Data

- \>5,000 knee MRI exams, 16–19 institutions worldwide (sources disagree on the site count).
- Multi-planar (expect sagittal / coronal / axial series per exam), DICOM.
- Per-exam radiology report text in mixed languages.
- **Unconfirmed:** total archive size, series naming, whether report text is provided for
  the test split or training only.

## Evaluation

Hidden test set. Sources mention ROC-AUC and log loss; a community repo cites both.
**Unconfirmed — read the Evaluation tab.** There is a separate **efficiency track** that
scores accuracy against compute cost, which usually implies a notebook-submission format
with a wall-clock limit.

## Timeline

| Date | Milestone |
| --- | --- |
| 2026-07-30 | Competition opened |
| 2026-10-15 | Entry / team-merger deadline |
| 2026-10-22 | Final submission deadline |
| 2026-11-05 | Winner requirements due |
| 2026-11-29 – 12-03 | RSNA 2026, Chicago — winners recognized |

$77,000 prize pool across the main leaderboard and the efficiency track.

## Open questions to resolve first

1. Exact label set and whether the target is per-exam multi-label or per-series.
2. Metric — if it is mean column-wise AUC, class imbalance strategy matters more than loss choice.
3. Submission mechanics: notebook-only? GPU/CPU quota? runtime cap? This decides whether a
   heavyweight multilingual text encoder is even affordable at inference.
4. Is report text available at test time, or is it train-only supervision (i.e. a
   distillation / auxiliary-loss setup rather than a true multimodal model)?
5. Archive size, to size the data volume before renting a GPU box.

## Sources

- <https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview>
- <https://www.rsna.org/artificial-intelligence/ai-image-challenge/knee-mri-ai-challenge>
- <https://www.rsna.org/news/2026/august/ai-challenge-knee-mri>
- <https://runtimewire.com/article/rsna-knee-mri-ai-challenge-2026>
