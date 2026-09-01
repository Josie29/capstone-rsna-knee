from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import write_dicom_slice

from knee.labels import LABEL_COLUMNS, STUDY_ID_COLUMN
from knee.model import KneeModel, load_model
from knee.series import SeriesType
from knee.train_blended import train_blended


def _blended_comp_root(root: Path) -> tuple[Path, pd.DataFrame]:
    """Fake competition layout: 3 loadable studies + 1 corrupt, with soft labels."""
    rng = np.random.default_rng(0)
    study_uids = ["study0", "study1", "study2", "study_corrupt"]
    # Soft probabilities that threshold to both classes on every label at 0.5.
    label_rows = [[0.9123] * 12, [0.0372] * 12, [0.9 if i % 2 else 0.1 for i in range(12)], [0.5941] * 12]
    labels = pd.DataFrame(label_rows, columns=list(LABEL_COLUMNS)).astype("float32")
    labels.insert(0, STUDY_ID_COLUMN, study_uids)

    series = pd.DataFrame(
        {
            STUDY_ID_COLUMN: study_uids,
            "SeriesInstanceUID": [f"series{i}" for i in range(4)],
            "Anatomical_Plane": "Sagittal",
            "Fluid_Sensitive": 1,
            "Fat_Suppression": 1,
        }
    )
    series.to_csv(root / "train_series.csv", index=False)

    for study_uid, series_uid in zip(study_uids, series["SeriesInstanceUID"]):
        series_dir = root / "train_series" / study_uid / series_uid
        series_dir.mkdir(parents=True)
        if study_uid == "study_corrupt":
            (series_dir / "bad.dcm").write_bytes(b"not a dicom file")
            continue
        for slice_index in range(3):
            write_dicom_slice(
                series_dir / f"slice{slice_index}.dcm",
                rng.integers(0, 1000, (32, 32)),
                instance_number=slice_index,
            )
    return root, labels


def test_train_blended_end_to_end(tmp_path: Path) -> None:
    """Catches integration bugs between the threaded feature collection, soft-target
    head fitting, and checkpoint export — e.g. soft labels tripping a binary-only code
    path, or prefetch reordering features against labels. This composition otherwise
    only runs on Kaggle, the costliest place to debug."""
    comp_root, labels = _blended_comp_root(tmp_path)
    out_dir = tmp_path / "checkpoints"

    bank, results = train_blended(
        comp_root,
        labels,
        out_dir,
        series_types=[SeriesType.SAGITTAL_FLUID],
        model=KneeModel(pretrained=False),
        input_size=64,
        log=lambda _: None,
    )

    assert bank.study_uids == list(labels[STUDY_ID_COLUMN])
    (result,) = results
    assert result.series_type is SeriesType.SAGITTAL_FLUID
    assert result.n_studies == 3  # the corrupt study's plane sat out
    assert set(result.in_sample_auc) == set(LABEL_COLUMNS)

    # The checkpoint must reproduce the trained function offline, and carry the
    # provenance that distinguishes blended-trained weights from gold-trained ones.
    loaded = load_model(result.checkpoint_path)
    assert loaded.series_type is SeriesType.SAGITTAL_FLUID
    payload = torch.load(result.checkpoint_path, weights_only=True)
    assert payload["label_source"] == "blended_v1"
    assert payload["n_studies"] == 3
    probs = loaded.model.predict_study(torch.rand(3, 64, 64))
    assert probs.shape == (len(LABEL_COLUMNS),)


def test_plane_with_no_usable_studies_fails_loudly(tmp_path: Path) -> None:
    """Catches a missing plane silently shipping a smaller ensemble — the strict-typing
    rule means a coronal head must never quietly not exist because no coronal series
    decoded."""
    comp_root, labels = _blended_comp_root(tmp_path)  # layout only has sagittal series

    with pytest.raises(ValueError, match="No usable studies"):
        train_blended(
            comp_root,
            labels,
            tmp_path / "checkpoints",
            series_types=[SeriesType.CORONAL_FLUID],
            model=KneeModel(pretrained=False),
            input_size=64,
            log=lambda _: None,
        )
