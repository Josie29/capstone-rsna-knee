from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import write_dicom_slice

from knee.labels import LABEL_COLUMNS, STUDY_ID_COLUMN
from knee.model import KneeModel, load_model, save_model
from knee.series import SeriesType
from knee.train_gold import train_gold

# Random-init (pretrained=False) so tests run offline; small volumes keep them fast.
_VOLUME = torch.rand(4, 64, 64, generator=torch.Generator().manual_seed(0))


def test_model_round_trips_through_checkpoint(tmp_path: Path) -> None:
    """Catches the bug where save/load drops or misaligns weights — predictions after
    reload on Kaggle would silently differ from what training validated, and the
    offline submission notebook has no way to notice."""
    model = KneeModel(pretrained=False)
    model.eval()
    before = model.predict_study(_VOLUME)

    save_model(model, tmp_path / "ck.pt", input_size=64, series_type=SeriesType.CORONAL_FLUID)
    loaded = load_model(tmp_path / "ck.pt")
    after = loaded.model.predict_study(_VOLUME)

    assert loaded.input_size == 64
    assert loaded.series_type is SeriesType.CORONAL_FLUID
    torch.testing.assert_close(before, after)


def test_predictions_are_probabilities_per_label(tmp_path: Path) -> None:
    """Catches the bug where the model emits logits instead of probabilities — AUC is
    rank-invariant so the leaderboard would not catch it, but any later calibration
    or ensembling step would be silently wrong."""
    probs = KneeModel(pretrained=False).predict_study(_VOLUME)
    assert probs.shape == (len(LABEL_COLUMNS),)
    assert ((probs > 0.0) & (probs < 1.0)).all()


def test_tampered_label_order_is_rejected(tmp_path: Path) -> None:
    """Catches a checkpoint from a stale code version whose label order differs —
    loading it would silently assign every probability to the wrong finding."""
    model = KneeModel(pretrained=False)
    save_model(model, tmp_path / "ck.pt", input_size=64, series_type=SeriesType.SAGITTAL_FLUID)
    payload = torch.load(tmp_path / "ck.pt", weights_only=True)
    payload["label_columns"] = list(reversed(payload["label_columns"]))
    torch.save(payload, tmp_path / "bad.pt")

    with pytest.raises(ValueError, match="label order"):
        load_model(tmp_path / "bad.pt")


def test_empty_volume_is_rejected() -> None:
    """Catches a series-loading bug (zero slices) surfacing as a cryptic backbone
    shape error instead of a clear message naming the contract."""
    with pytest.raises(ValueError, match="non-empty"):
        KneeModel(pretrained=False).pool_features(torch.zeros(0, 64, 64))


def _synthetic_comp_root(root: Path) -> Path:
    """Fake competition layout: 3 loadable gold studies + 1 with a corrupt series."""
    rng = np.random.default_rng(0)
    study_uids = ["study0", "study1", "study2", "study_corrupt"]
    # Every label column sees both classes across the loadable studies.
    label_rows = [[1] * 12, [0] * 12, [i % 2 for i in range(12)], [1] * 12]

    train = pd.DataFrame(label_rows, columns=list(LABEL_COLUMNS))
    train.insert(0, STUDY_ID_COLUMN, study_uids)
    train.insert(1, "Report", "synthetic")
    train.to_csv(root / "train.csv", index=False)

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
    return root


def test_train_gold_end_to_end(tmp_path: Path) -> None:
    """Catches integration bugs between selection, decoding, feature caching, and head
    training that the unit tests miss — e.g. features cached under inference_mode
    crashing backward, or feature/label misalignment when a study is skipped mid-loop.
    This composition otherwise only runs on Kaggle, the costliest place to debug."""
    comp_root = _synthetic_comp_root(tmp_path)
    checkpoint = tmp_path / "ck.pt"

    result = train_gold(
        comp_root,
        checkpoint,
        model=KneeModel(pretrained=False),
        input_size=64,
        log=lambda _: None,
    )

    assert result.series_type is SeriesType.SAGITTAL_FLUID
    assert result.n_studies == 3
    assert [s.study_uid for s in result.skipped] == ["study_corrupt"]
    # All 12 labels have both classes across the 3 loaded studies, so no NaNs.
    assert set(result.in_sample_auc) == set(LABEL_COLUMNS)
    assert all(0.0 <= v <= 1.0 for v in result.in_sample_auc.values())

    # The checkpoint must reproduce the trained function offline.
    loaded = load_model(checkpoint)
    assert loaded.input_size == 64
    assert loaded.series_type is SeriesType.SAGITTAL_FLUID
    probs = loaded.model.predict_study(torch.rand(3, 64, 64))
    assert probs.shape == (len(LABEL_COLUMNS),)


def test_training_a_type_the_studies_lack_fails_loudly(tmp_path: Path) -> None:
    """Catches the strict-typing rule regressing into a fallback cascade — a coronal
    model silently training on sagittal series would specialize on the wrong view
    while reporting success."""
    comp_root = _synthetic_comp_root(tmp_path)  # layout only has sagittal fluid series

    with pytest.raises(ValueError, match="Every gold study failed"):
        train_gold(
            comp_root,
            tmp_path / "ck.pt",
            series_type=SeriesType.CORONAL_FLUID,
            model=KneeModel(pretrained=False),
            input_size=64,
            log=lambda _: None,
        )
