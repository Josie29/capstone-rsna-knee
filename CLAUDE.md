# CLAUDE.md

Gauntlet AI capstone: entry in the Kaggle competition
[RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview).
Predict per-study probabilities for 12 knee-MRI findings; metric is macro ROC AUC;
submission is an offline Kaggle notebook (≤9h, no internet). We intend to submit.

## Team

Three developers collaborate on this repo: Josie, Kelly, and Ryan.
Issues are tracked via GitHub Issues (`gh issue ...`), not a separate tracker.

Contribution workflow is TODO — see "Contributing" in `README.md` for the options
under discussion. Until decided, assume feature branches + PR + squash merge.

## Key docs

- `docs/competition-notes.md` — full competition spec and constraints
- `docs/tech-stack.md` — stack choices and rejected alternatives
- `docs/DECISIONS.md` — running decision log
- `docs/experiments.md` — curated experiment registry; every meaningful experiment gets a
  row + hypothesis/outcome log entry (conventions in the file header)
- `docs/presentation/slides.md` — final demo deck content (`deck.html` is the visual)

## Presentation

The capstone demo deck lives in `docs/presentation/` and evolves with the project.
When a milestone lands (results, a pivot, a new insight), consider whether
`slides.md` should be updated — both to keep the deck demo-ready and to force the
"why does this matter / what's the value-add" framing while working. Update
`slides.md` first, sync `deck.html` after content is agreed.

## Layout

- `src/knee/` — all real logic (notebooks are thin wrappers)
- `notebooks/` — exploration, train, inference scaffolds
- `scripts/` — data download etc.
- `tests/` — pytest

## Conventions

- Python 3.12, managed with `uv` (`uv sync`, `uv run ...`)
- pyright strict + ruff; pydantic v2 for structured data
- Never commit data or log report text/UIDs (competition rule 2.4.b)
- Experiments: ID `E###-short-slug`; add the `docs/experiments.md` row + hypothesis when
  starting one, fill the outcome when it resolves. Raw runs/curves go to wandb
  (`rsna-knee` project) — the markdown file is the curated story, not a run dump.
- Keep as much train/inference logic in `src/knee/` modules as you can, not notebooks — .py files diff and
  review cleanly; .ipynb files don't.
- Notebooks: thin shells over `src/knee/`; strip outputs before commit
  (`jupyter nbconvert --clear-output --inplace <nb>`); Kaggle kernels are deployed
  push-only via `kaggle kernels push` — never edited in the Kaggle web UI. Details in
  `notebooks/README.md`.
