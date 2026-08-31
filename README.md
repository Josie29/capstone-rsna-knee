# Capstone — RSNA Knee Abnormality Detection

Capstone for the Gauntlet AI program. Entry in the Kaggle competition
[RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview)
(2026, $77k, hosted by RSNA).

**Task:** per-study probability for 12 binary knee-MRI findings — ACL, MCL, medial and
lateral meniscus, three OA compartments, effusion, synovitis, Baker's cyst, contusion,
fracture. **Metric:** macro-averaged ROC AUC. **Submission:** Kaggle notebook, ≤9h runtime,
no internet.

**The interesting part:** only a small subset of training studies carry per-condition
labels. The rest have to be mined from free-text radiology reports written in a dozen
languages — and the report field is *not* available at test time.

Full specifics: [docs/competition-notes.md](docs/competition-notes.md).

## Status

Scaffold only. No data downloaded, no baseline, no submission.

**Entry deadline 2026-10-15 · final submission 2026-10-22.**

## Setup

```bash
uv sync --extra data
uv run scripts/download_data.py
```

Needs a Kaggle API token at `~/.kaggle/kaggle.json` (or `KAGGLE_USERNAME`/`KAGGLE_KEY`) and
the competition rules accepted on the website — the script says which is missing.

Heavier extras once modeling starts:

```bash
uv sync --extra data --extra train --extra text
```

## Competition data never enters this repo

Rule 2.4.b forbids redistributing Competition Data to anyone who has not accepted the
rules, and this repo is public. `.gitignore` excludes all of `data/`, `checkpoints/`, and
`submissions/`. Report text, label CSVs, and fold assignments keyed to `StudyInstanceUID`
all count as Competition Data.

## Layout

| Path | What lives here |
| --- | --- |
| `src/knee/labels.py` | The 12 target names in submission-column order. Single source of truth. |
| `src/knee/paths.py` | Where data lives. `KNEE_DATA_ROOT` overrides; auto-detects Kaggle notebooks. |
| `scripts/` | Entry points, run from the repo root via `uv run`. |
| `data/`, `checkpoints/`, `submissions/` | Gitignored. Redownloadable or regenerable. |
| `notebooks/` | Exploration, plus the Kaggle submission notebook. |
| `docs/` | Competition notes, tech-stack decisions, brainlift. |
| `tests/` | pytest. |

## Checks

```bash
uv run pytest && uv run pyright && uv run ruff check .
```
