# Notebooks

Three notebooks, three jobs. All real logic lives in `src/knee/` — notebooks are thin shells
that wire config, paths, and secrets together.

| Path | Runs where | Purpose |
| --- | --- | --- |
| `exploration/eda.ipynb` | Local | EDA on the CSVs and sample DICOMs. Never leaves this machine. |
| `kaggle/train/` | Kaggle (GPU, internet ON) | Installs `knee` from GitHub at a pinned commit, trains on the pre-mounted competition data, exports a checkpoint. |
| `kaggle/inference/` | Kaggle (GPU, internet OFF) | The submission notebook. Attached datasets supply code + weights; writes `submission.csv`. |

## Deploying to Kaggle

The repo is the source of truth — never edit kernels in the Kaggle web UI. Deploy with:

```bash
uv run python scripts/normalize_notebooks.py  # canonical nbformat, else Kaggle renders raw JSON
kaggle kernels push -p notebooks/kaggle/train
kaggle kernels push -p notebooks/kaggle/inference
```

Normalization matters because programmatic cell edits leave `source` as one string;
Jupyter executes that fine, but Kaggle's viewer shows the whole file as raw JSON
(train kernel v6 is an example of the failure mode).

Each folder's `kernel-metadata.json` pins the environment: kernel slug, GPU, internet,
and which datasets are attached. Replace `KAGGLE_USERNAME` with the real username before
the first push.

## Hygiene

- Strip outputs before committing (`jupyter nbconvert --clear-output --inplace <nb>`)
  so diffs stay reviewable.
- Nothing derived from Competition Data (report text, label values, StudyInstanceUIDs)
  belongs in committed notebook cells or outputs — this repo is public.
