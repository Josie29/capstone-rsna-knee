import pandas as pd

from knee.labels import LABEL_COLUMNS, STUDY_ID_COLUMN


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
