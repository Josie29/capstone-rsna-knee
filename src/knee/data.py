from pathlib import Path

import numpy as np
import pandas as pd

from knee.labels import LABEL_COLUMNS, STUDY_ID_COLUMN

# Per-label companion columns in the blended-labels CSV (see
# docs/modeling understanding/blended-labels-methodology.md): __weight is the
# per-cell training weight (how much the estimate should influence the model),
# __tier records which kind of evidence produced it (explicit / proxy / guess).
_BLENDED_COMPANION_SUFFIXES = ("__weight", "__tier")

WEIGHT_SUFFIX = "__weight"


def weight_matrix(blended: pd.DataFrame) -> np.ndarray:
    """Per-cell training weights aligned with `LABEL_COLUMNS`.

    Args:
        blended: Frame from `load_blended_labels(..., include_weights=True)`.

    Returns:
        (n_studies, 12) float32 array of the `__weight` companions.

    Raises:
        KeyError: If the frame was loaded without weights.
    """
    return blended[[f"{label}{WEIGHT_SUFFIX}" for label in LABEL_COLUMNS]].to_numpy(dtype=np.float32)


def gold_studies(train_df: pd.DataFrame) -> pd.DataFrame:
    """Select the fully-labeled ("gold") studies from train.csv.

    Only a small subset of training studies (58 as measured 2026-08-31) carry
    per-condition labels, and on those rows all 12 columns are populated together.
    Everything else must come from report mining, so these rows are the only ground
    truth available.

    Args:
        train_df: The train.csv frame; must contain `StudyInstanceUID` and all 12
            label columns.

    Returns:
        The rows where every label column is populated, with label columns cast to
        int, indexed as in `train_df`.

    Raises:
        ValueError: If a label column is missing, no fully-labeled rows exist, or a
            populated label value is not 0/1 — any of which means the train.csv
            contract changed and downstream training cannot be trusted.
    """
    label_columns = list(LABEL_COLUMNS)
    missing = [c for c in (STUDY_ID_COLUMN, *label_columns) if c not in train_df.columns]
    if missing:
        raise ValueError(f"train.csv is missing expected columns: {missing}")

    gold = train_df.dropna(subset=label_columns).copy()
    if gold.empty:
        raise ValueError("No fully-labeled studies found in train.csv")

    values = gold[label_columns]
    if not values.isin((0, 1)).all().all():
        bad = values[~values.isin((0, 1)).all(axis=1)]
        raise ValueError(f"Non-binary label values in fully-labeled rows: indexes {list(bad.index)}")

    gold[label_columns] = values.astype(int)
    return gold


def load_blended_labels(csv_path: Path, *, include_weights: bool = False) -> pd.DataFrame:
    """Load the blended (report-mined) soft labels for every train study.

    The CSV carries, per label, a probability estimate plus `__weight`/`__tier`
    companion columns describing how the estimate was produced. The companions are
    always validated present (so a truncated export fails loudly); with
    `include_weights` the `__weight` columns are kept for tier-weighted training
    (E006a) — extract them with `weight_matrix`.

    Args:
        csv_path: Path to `blended_labels_v1.csv` (locally under `data/processed/`,
            on Kaggle under the mounted `knee-labels` dataset).
        include_weights: Keep the per-cell `__weight` companions (validated finite
            and non-negative) alongside the probabilities.

    Returns:
        One row per study: `StudyInstanceUID` plus the 12 label columns as float32
        probabilities in [0, 1] (plus the 12 `__weight` columns when requested).

    Raises:
        ValueError: If a label or companion column is missing, a study UID repeats,
            a probability is NaN or outside [0, 1], or (with `include_weights`) a
            weight is NaN or negative — any of which means the blended-labels
            contract changed and training on the file is unsafe.
    """
    frame = pd.read_csv(csv_path)
    label_columns = list(LABEL_COLUMNS)

    expected = [STUDY_ID_COLUMN] + [
        f"{label}{suffix}" for label in label_columns for suffix in ("", *_BLENDED_COMPANION_SUFFIXES)
    ]
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise ValueError(f"{csv_path.name} is missing expected columns: {missing}")

    if frame[STUDY_ID_COLUMN].duplicated().any():
        duplicated = frame[STUDY_ID_COLUMN][frame[STUDY_ID_COLUMN].duplicated()]
        raise ValueError(f"{csv_path.name} has duplicate study UIDs: {len(duplicated)} rows")

    probabilities = frame[label_columns]
    if probabilities.isna().any().any():
        raise ValueError(f"{csv_path.name} has NaN probabilities")
    if ((probabilities < 0) | (probabilities > 1)).any().any():
        raise ValueError(f"{csv_path.name} has probabilities outside [0, 1]")

    kept_columns = [STUDY_ID_COLUMN, *label_columns]
    if include_weights:
        weight_columns = [f"{label}{WEIGHT_SUFFIX}" for label in label_columns]
        weights = frame[weight_columns]
        if weights.isna().any().any():
            raise ValueError(f"{csv_path.name} has NaN weights")
        if (weights < 0).any().any():
            raise ValueError(f"{csv_path.name} has negative weights")
        kept_columns += weight_columns
        frame[weight_columns] = weights.astype("float32")

    blended = frame[kept_columns].copy()
    blended[label_columns] = probabilities.astype("float32")
    return blended
