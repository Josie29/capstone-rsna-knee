from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from knee.data import gold_studies, load_blended_labels, weight_matrix
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


def _write_blended_csv(path: Path, probabilities: Sequence[Sequence[float]]) -> pd.DataFrame:
    """Blended-labels CSV with the prob/__weight/__tier triplet per label."""
    frame = pd.DataFrame({STUDY_ID_COLUMN: [f"uid{i}" for i in range(len(probabilities))]})
    for column, label in enumerate(LABEL_COLUMNS):
        frame[label] = [row[column] for row in probabilities]
        frame[f"{label}__weight"] = 1.0
        frame[f"{label}__tier"] = 1
    frame.to_csv(path, index=False)
    return frame


def test_blended_labels_keep_soft_values(tmp_path: Path) -> None:
    """Catches the loader rounding or coercing probabilities to 0/1 — the whole point
    of the blended labels is that the trainer sees the miner's soft estimates."""
    csv_path = tmp_path / "blended.csv"
    _write_blended_csv(csv_path, [[0.9123] + [0.0372] * 11, [0.5] * 12])

    blended = load_blended_labels(csv_path)

    assert list(blended.columns) == [STUDY_ID_COLUMN, *LABEL_COLUMNS]
    assert all(str(dtype) == "float32" for dtype in blended[list(LABEL_COLUMNS)].dtypes)
    assert blended.loc[0, LABEL_COLUMNS[0]] == pytest.approx(0.9123)


def test_blended_labels_missing_companion_column_is_rejected(tmp_path: Path) -> None:
    """Catches a truncated export (e.g. the values-only variant) sneaking in — training
    would proceed but the tier/weight columns the methodology promises would be gone,
    and the later weighted-training experiment would silently have nothing to use."""
    csv_path = tmp_path / "blended.csv"
    frame = _write_blended_csv(csv_path, [[0.5] * 12])
    frame.drop(columns=[f"{LABEL_COLUMNS[0]}__tier"]).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="missing expected columns"):
        load_blended_labels(csv_path)


def test_blended_labels_out_of_range_probability_is_rejected(tmp_path: Path) -> None:
    """Catches a labels re-export switching to logits or percentages — training on
    those as probabilities would corrupt every head silently."""
    csv_path = tmp_path / "blended.csv"
    _write_blended_csv(csv_path, [[1.5] + [0.5] * 11])

    with pytest.raises(ValueError, match="outside"):
        load_blended_labels(csv_path)


def test_blended_labels_duplicate_uid_is_rejected(tmp_path: Path) -> None:
    """Catches a bad join upstream duplicating studies — duplicated rows would
    double-weight those studies in training and skew CV fold assignment."""
    csv_path = tmp_path / "blended.csv"
    frame = _write_blended_csv(csv_path, [[0.5] * 12, [0.5] * 12])
    frame[STUDY_ID_COLUMN] = "uid0"
    frame.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="duplicate"):
        load_blended_labels(csv_path)


def test_blended_weights_are_exposed_on_request(tmp_path: Path) -> None:
    """Catches the E006a plumbing breaking — tier-weighted training only works if
    the __weight companions survive loading and extract aligned with LABEL_COLUMNS;
    silently dropped or reordered weights would weight the wrong findings."""
    csv_path = tmp_path / "blended.csv"
    _write_blended_csv(csv_path, [[0.9] * 12, [0.1] * 12])

    blended = load_blended_labels(csv_path, include_weights=True)
    weights = weight_matrix(blended)

    assert weights.shape == (2, len(LABEL_COLUMNS))
    assert weights.dtype == np.float32
    assert (weights == 1.0).all()
    # The default path stays weight-free, so existing callers are untouched.
    assert f"{LABEL_COLUMNS[0]}__weight" not in load_blended_labels(csv_path).columns


def test_negative_blended_weights_are_rejected(tmp_path: Path) -> None:
    """Catches a corrupted weights export — a negative weight would *reward* the
    model for being wrong on that cell, a silent training inversion."""
    csv_path = tmp_path / "blended.csv"
    frame = _write_blended_csv(csv_path, [[0.5] * 12])
    frame[f"{LABEL_COLUMNS[0]}__weight"] = -0.5
    frame.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="negative weights"):
        load_blended_labels(csv_path, include_weights=True)
