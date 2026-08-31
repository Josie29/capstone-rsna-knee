# Tech Stack — RSNA Knee Abnormality Detection

Context: Kaggle research code competition ([overview](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview)).
Macro-averaged ROC AUC over 12 findings; submission is an offline Kaggle notebook, ≤9h runtime; efficiency
track adds `AUC/(Benchmark−maxAUC) + RuntimeSeconds/32400`. Training happens off-Kaggle; only inference
runs there. Labels for ~98.7% of train studies must be mined from multilingual radiology reports
(text not available at test time). Full constraints: [competition-notes.md](competition-notes.md).

| Layer | Component | Choice | Reason |
|---|---|---|---|
| Language | Runtime | Python 3.12 | ML ecosystem default; matches Kaggle notebook images |
| Language | Package/env manager | uv | Fast lockfile-based installs; one tool for venv + deps + extras |
| Language | Structured data models | Pydantic v2 | Validated typed models for series/label schemas over bare dicts |
| Data | Tabular | pandas + numpy | CSVs are small (≤6 MiB); no need for anything heavier |
| Data | DICOM decode | pydicom + pylibjpeg[all] | Mixed transfer syntaxes (JPEG Lossless, JPEG 2000) — pydicom alone decodes only uncompressed |
| Modeling | Deep learning framework | PyTorch | timm/MONAI/transformers all sit on it; Kaggle GPU support first-class |
| Modeling | 2D backbones | timm | Largest zoo of pretrained ImageNet backbones for per-slice models |
| Modeling | Volumetric transforms/losses | MONAI | 3D/medical-imaging transforms torchvision lacks |
| Modeling | Metrics + CV splits | scikit-learn | Standard AUC + stratified splitting; nothing exotic needed |
| Text (train-time only) | Report-label miner | Hugging Face transformers + sentencepiece | Reports span ~9–12 languages; needs multilingual pretrained models |
| Training | Hardware | Kaggle GPU notebooks (30h/week free) | Competition data pre-mounted — skips the 247 GiB download entirely; local Mac (MPS) only for sample-scale dev |
| Training | Experiment tracking | Weights & Biases (metrics/hyperparams only) | Survives ephemeral Kaggle training sessions; free solo tier; never log report text or UIDs (rule 2.4.b) |
| Submission | Runtime | Kaggle notebook (thin wrapper over `src/knee/`) | Required by competition; all logic stays in versioned .py files |
| Submission | Code/weights delivery | Kaggle datasets attached to notebook | Internet disabled at submission time — only way to ship artifacts |
| Quality | Tests | pytest | Project standard; `tests/` already scaffolded |
| Quality | Type checking | pyright (strict) | Catches schema drift in label/series plumbing; configured in pyproject |
| Quality | Lint/format | ruff | Single fast tool replaces flake8+isort+black |

## Rejected alternatives

| Component | Option | Why not |
|---|---|---|
| Package/env manager | conda | Slow resolver; lockfiles awkward; uv covers the same ground |
| Package/env manager | poetry | Slower than uv, no interpreter management |
| DICOM decode | GDCM | Heavier native dependency; pylibjpeg covers the same syntaxes pip-only |
| DICOM decode | SimpleITK | Volume-level API fights per-slice file layout; weaker tag access |
| Deep learning framework | TensorFlow/Keras | Pretrained medical/vision ecosystem is PyTorch-centric |
| Deep learning framework | JAX | Too little pretrained-model leverage for a deadline competition |
| 2D backbones | torchvision models | Far smaller pretrained zoo than timm |
| Report-label miner | Regex/keyword rules | ~12 languages × 12 findings × negations — brittle beyond a first sanity pass |
| Report-label miner | Translate-then-English-model | Adds a failure stage; multilingual encoders skip it |
| Training hardware | Local Apple Silicon (MPS) only | No CUDA — timm/MONAI mixed-precision paths are CUDA-first; nowhere to put 247 GiB |
| Training hardware | Cloud GPU rental (RunPod / Lambda / Vast) | Real money + must stage 247 GiB yourself; revisit only if Kaggle's 30h/week or 12h sessions become the bottleneck |
| Training hardware | Colab Pro | Same download problem as rented cloud, weaker GPUs than rented, no data mount |
| Experiment tracking | MLflow (local file store) | Local `mlruns/` dies with ephemeral Kaggle sessions unless manually exported; self-hosting a server is overkill solo |
| Experiment tracking | TensorBoard | Curves only — no run/hyperparameter comparison table across experiments |
| Experiment tracking | Plain CSV logs | Zero-dep but reinvents the comparison UI right when experiments multiply |
| Submission runtime | Dev in .ipynb | Poor diffs/review; notebook is generated at the end instead |
| Submission runtime | Kaggle script kernel | Preferred if accepted — unverified for this competition, notebook is the safe default |
| Type checking | mypy | pyright is stricter and faster; house standard |

## Open sub-decisions

- **Miner approach** (issue #2): local multilingual encoder vs LLM — check rule 2.4.b first: sending
  report text to an external LLM API may count as redistributing Competition Data.
- **Script kernel vs notebook** (issue #5): try `kaggle kernels push` with a script kernel once; fall back to
  the thin-wrapper notebook if the competition rejects it.
