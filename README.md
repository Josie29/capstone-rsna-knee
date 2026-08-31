# Capstone — RSNA Knee Abnormality Detection

Capstone project for the Gauntlet AI program. Entry in the Kaggle competition
[RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview)
(2026): detect 12 clinically important abnormalities from multi-planar knee MRI, with
per-exam radiology reports supplied in ~9–12 languages.

Verified competition facts, open questions, and sources:
[docs/competition-notes.md](docs/competition-notes.md).

## Status

Scaffold only. No data downloaded, no baseline, no submission.

**Deadlines:** entry closes **2026-10-15**, final submission **2026-10-22**.

## Setup

```bash
uv sync --extra data
uv run scripts/download_data.py
```

`download_data.py` needs a Kaggle API token at `~/.kaggle/kaggle.json` (or
`KAGGLE_USERNAME` / `KAGGLE_KEY`) and the competition rules accepted on the website —
it will tell you which is missing.

Add the heavier extras when you get to modeling:

```bash
uv sync --extra data --extra train --extra text
```

## Layout

| Path | What lives here |
| --- | --- |
| `src/knee/` | The package. `paths.py` is the single source of truth for where data lives. |
| `scripts/` | Entry points run from the repo root via `uv run`. |
| `data/raw/`, `data/interim/` | Gitignored. Redownloadable / regenerable. |
| `data/processed/` | Committed. Small manifests and folds a grader needs to read. |
| `notebooks/` | Exploration, and the Kaggle submission notebook once there is one. |
| `docs/` | Competition notes, tech-stack decisions, brainlift. |
| `tests/` | pytest. |

Set `KNEE_DATA_ROOT` to move the data and checkpoint directories onto a mounted volume;
inside a Kaggle notebook the paths resolve to `/kaggle/input` and `/kaggle/working`
automatically.

## Checks

```bash
uv run pytest && uv run pyright && uv run ruff check .
```
