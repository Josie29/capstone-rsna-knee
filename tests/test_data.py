from collections.abc import Sequence

import pandas as pd
import pytest

from knee.data import gold_studies
from knee.labels import LABEL_COLUMNS, STUDY_ID_COLUMN


def _frame(rows: Sequence[Sequence[float | None]]) -> pd.DataFrame:
    frame = pd.DataFrame([list(row) for row in rows], columns=list(LABEL_COLUMNS))
    frame.insert(0, STUDY_ID_COLUMN, [f"uid{i}" for i in range(len(rows))])
    frame.insert(1, "Report", "text")
    return frame


def test_only_fully_labeled_rows_are_gold() -> None:
    """Catches the bug where NaN-labeled rows leak into the gold set — the trainer
    would then fit on unlabeled studies coerced to fake labels."""
    labeled = [1.0] + [0.0] * 11
    unlabeled: list[float | None] = [None] * 12
    gold = gold_studies(_frame([labeled, unlabeled, labeled]))
    assert list(gold[STUDY_ID_COLUMN]) == ["uid0", "uid2"]
    assert all(str(dtype) == "int64" for dtype in gold[list(LABEL_COLUMNS)].dtypes)


def test_non_binary_labels_are_rejected() -> None:
    """Catches a train.csv refresh switching to e.g. severity grades — silently
    thresholding those would corrupt every downstream label."""
    bad = [2.0] + [0.0] * 11
    with pytest.raises(ValueError, match="Non-binary"):
        gold_studies(_frame([bad]))


def test_missing_label_column_is_rejected() -> None:
    """Catches a schema change (renamed label column) that would otherwise surface
    as a confusing KeyError deep inside training."""
    frame = _frame([[1.0] + [0.0] * 11]).drop(columns=[LABEL_COLUMNS[0]])
    with pytest.raises(ValueError, match="missing expected columns"):
        gold_studies(frame)
