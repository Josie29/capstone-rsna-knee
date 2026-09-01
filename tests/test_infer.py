from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import write_dicom_slice

from knee.infer import merge_predictions, predict_studies
from knee.labels import LABEL_COLUMNS, STUDY_ID_COLUMN, SUBMISSION_COLUMNS
from knee.model import KneeModel, save_model
from knee.series import SeriesType


def _write_series(root: Path, study: str, series: str, n_slices: int = 3) -> None:
    rng = np.random.default_rng(hash(series) % 2**32)
    for index in range(n_slices):
        write_dicom_slice(
            root / "test_series" / study / series / f"s{index}.dcm",
            rng.integers(0, 1000, (32, 32)),
            instance_number=index,
        )


def _synthetic_test_root(root: Path) -> Path:
    """Three test studies exercising the merge paths:

    - study_full: sagittal + axial fluid series (both models run)
    - study_sag_only: sagittal fluid only (axial model sits out, mean renormalizes)
    - study_unreadable: only a corrupt series (no model runs -> 0.5 row)
    """
    studies = ["study_full", "study_sag_only", "study_unreadable"]
    pd.DataFrame({STUDY_ID_COLUMN: studies}).to_csv(root / "test.csv", index=False)

    series_rows = [
        ("study_full", "ser_full_sag", "Sagittal", 1, 1),
        ("study_full", "ser_full_ax", "Axial", 1, 1),
        ("study_sag_only", "ser_sagonly", "Sagittal", 1, 1),
        ("study_unreadable", "ser_broken", "Sagittal", 1, 1),
    ]
    pd.DataFrame(
        series_rows,
        columns=[STUDY_ID_COLUMN, "SeriesInstanceUID", "Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression"],
    ).to_csv(root / "test_series.csv", index=False)

    _write_series(root, "study_full", "ser_full_sag")
    _write_series(root, "study_full", "ser_full_ax")
    _write_series(root, "study_sag_only", "ser_sagonly")
    broken = root / "test_series" / "study_unreadable" / "ser_broken"
    broken.mkdir(parents=True)
    (broken / "bad.dcm").write_bytes(b"not a dicom file")
    return root


def _checkpoint(path: Path, series_type: SeriesType) -> Path:
    save_model(KneeModel(pretrained=False), path, input_size=64, series_type=series_type)
    return path


def test_ensemble_fans_out_merges_and_never_emits_nan(tmp_path: Path) -> None:
    """Catches the ensemble breaking on the real coverage gaps (~10% of studies lack
    a fluid plane; some series won't decode): a NaN or missing row in submission.csv
    scores the whole submission invalid, silently costing a daily slot."""
    root = _synthetic_test_root(tmp_path)
    checkpoints = [
        _checkpoint(tmp_path / "sag.pt", SeriesType.SAGITTAL_FLUID),
        _checkpoint(tmp_path / "ax.pt", SeriesType.AXIAL_FLUID),
    ]

    frame = predict_studies(root, checkpoints, log=lambda _: None)

    assert list(frame.columns) == list(SUBMISSION_COLUMNS)
    assert list(frame[STUDY_ID_COLUMN]) == ["study_full", "study_sag_only", "study_unreadable"]
    assert not frame.isna().any().any()
    probs = frame[list(LABEL_COLUMNS)].to_numpy()
    assert ((probs >= 0.0) & (probs <= 1.0)).all()
    # The unreadable study gets the sample-submission constant, not a model output.
    assert (probs[2] == 0.5).all()


def test_duplicate_series_type_checkpoints_are_rejected(tmp_path: Path) -> None:
    """Catches accidentally attaching two sagittal checkpoints — a plain mean would
    silently double-weight that view for every study."""
    root = _synthetic_test_root(tmp_path)
    checkpoints = [
        _checkpoint(tmp_path / "a.pt", SeriesType.SAGITTAL_FLUID),
        _checkpoint(tmp_path / "b.pt", SeriesType.SAGITTAL_FLUID),
    ]
    with pytest.raises(ValueError, match="Duplicate series types"):
        predict_studies(root, checkpoints, log=lambda _: None)


def test_no_checkpoints_is_rejected(tmp_path: Path) -> None:
    """Catches an empty weights-dataset glob in the notebook producing an all-0.5
    'submission' that looks valid but contains no model at all."""
    with pytest.raises(ValueError, match="at least one checkpoint"):
        predict_studies(tmp_path, [], log=lambda _: None)


def test_merge_weights_toward_the_plane_of_choice() -> None:
    """Catches the prior being applied on the wrong axis or ignored — for MCL the
    coronal model (grade 3) must dominate the sagittal one (grade 1), so the merged
    value must land closer to coronal's opinion than a plain mean would."""
    types = [SeriesType.SAGITTAL_FLUID, SeriesType.CORONAL_FLUID]
    per_model = np.stack([np.full(12, 0.0), np.full(12, 1.0)])  # sag says no, cor says yes
    merged = merge_predictions(per_model, types)

    mcl = list(LABEL_COLUMNS).index("MCL")
    acl = list(LABEL_COLUMNS).index("ACL")
    assert merged[mcl] == pytest.approx(3 / 4)  # cor 3 vs sag 1
    assert merged[acl] == pytest.approx(2 / 5)  # sag 3 vs cor 2


def test_merge_handles_missing_and_absent_models() -> None:
    """Catches NaN rows leaking into the weighted sum — one unreadable plane must
    renormalize, and all-unreadable must yield the 0.5 fallback, never NaN."""
    types = [SeriesType.SAGITTAL_FLUID, SeriesType.CORONAL_FLUID]
    one_out = np.stack([np.full(12, 0.8), np.full(12, np.nan)])
    assert merge_predictions(one_out, types) == pytest.approx(np.full(12, 0.8))

    all_out = np.full((2, 12), np.nan)
    assert (merge_predictions(all_out, types) == 0.5).all()
