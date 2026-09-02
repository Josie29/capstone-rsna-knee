from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import write_dicom_slice

from knee.cv import stratified_holdout
from knee.finetune import FinetuneConfig, build_pixel_cache, cache_path, finetune_plane
from knee.labels import LABEL_COLUMNS, STUDY_ID_COLUMN
from knee.model import HeadType, InputMode, KneeModel, load_model, sample_triplets
from knee.series import SeriesType


def _indexed_stack(n_slices: int) -> torch.Tensor:
    """Stack where slice i is constant i/(n-1): slice identity readable from values."""
    return torch.stack([torch.full((8, 8), i / max(n_slices - 1, 1)) for i in range(n_slices)])


def test_triplet_channels_are_adjacent_slices() -> None:
    """Catches the 2.5D contract breaking — if channels are not [i-1, i, i+1] of the
    physically sorted stack, the backbone loses the persists-across-neighbors cue
    that separates a real tear from a one-slice artifact."""
    stack = _indexed_stack(30)
    triplets = sample_triplets(stack, n_anchors=3, window=(0.2, 0.8))

    assert triplets.shape == (3, 3, 8, 8)
    for image in triplets:
        anchor = float(image[1, 0, 0]) * 29  # recover the anchor index from the value
        assert float(image[0, 0, 0]) * 29 == pytest.approx(anchor - 1)
        assert float(image[2, 0, 0]) * 29 == pytest.approx(anchor + 1)
        # Anchors stay inside the central window: edge slices are mostly muscle.
        assert 0.2 * 29 - 1 <= anchor <= 0.8 * 29 + 1


def test_triplets_are_deterministic_without_rng_and_clamp_at_edges() -> None:
    """Catches nondeterministic inference (anchors must be fixed when no rng is
    passed — otherwise resubmitting the same notebook scores different predictions)
    and index crashes on degenerate 1-2 slice stacks."""
    stack = _indexed_stack(30)
    torch.testing.assert_close(sample_triplets(stack), sample_triplets(stack))

    tiny = sample_triplets(_indexed_stack(1), n_anchors=2)
    assert tiny.shape == (2, 3, 8, 8)  # all channels clamp to the only slice
    torch.testing.assert_close(tiny[0, 0], tiny[0, 1])


def test_stratified_holdout_keeps_rare_labels_on_both_sides() -> None:
    """Catches a split that starves one side of a rare label — the fine-tune's val
    macro would silently drop that label and every early-stopping decision would be
    made on a different metric than the one reported."""
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, (100, len(LABEL_COLUMNS))).astype(np.float32)
    labels[:, 0] = 0.0
    labels[rng.choice(100, 10, replace=False), 0] = 1.0  # rare label: 10/100

    val = stratified_holdout(labels, val_fraction=0.2, seed=0)

    assert 10 <= val.sum() <= 30  # roughly the requested fraction
    assert labels[val, 0].sum() >= 1 and labels[~val, 0].sum() >= 1


def _synthetic_cache(cache_dir: Path, n_studies: int, series_type: SeriesType) -> np.ndarray:
    """Cache of 6-slice stacks where positives carry a bright band; returns targets."""
    rng = np.random.default_rng(3)
    (cache_dir / series_type.value).mkdir(parents=True)
    targets = np.zeros((n_studies, len(LABEL_COLUMNS)), dtype=np.float32)
    for row in range(n_studies):
        stack = rng.integers(40, 80, (6, 32, 32)).astype(np.uint8)
        if row % 2 == 0:
            stack[2:5, 8:24, 8:24] = 220  # signal on central slices only
            targets[row] = 0.9
        else:
            targets[row] = 0.1
        np.save(cache_path(cache_dir, series_type, row), stack)
    return targets


def test_finetune_plane_end_to_end(tmp_path: Path) -> None:
    """Catches integration bugs across sampling, augmentation, staged unfreezing,
    best-epoch selection, and checkpoint export — the composition otherwise only
    runs on Kaggle, the costliest place to debug. Also pins the provenance chain:
    a triplet-trained checkpoint must reload as a triplet model and predict from a
    raw volume without the caller knowing how it was trained."""
    targets = _synthetic_cache(tmp_path, 24, SeriesType.SAGITTAL_FLUID)
    val_mask = np.zeros(24, dtype=bool)
    val_mask[[0, 1, 4, 5, 8, 9]] = True  # 6 validation studies, both classes present

    model = KneeModel(pretrained=False, input_mode=InputMode.TRIPLETS, head_type=HeadType.MEAN_MAX)
    result = finetune_plane(
        tmp_path,
        targets,
        val_mask,
        series_type=SeriesType.SAGITTAL_FLUID,
        model=model,
        out_path=tmp_path / "ck.pt",
        config=FinetuneConfig(epochs=2, frozen_epochs=1, batch_studies=8, seed=0),
        input_size=32,
        crop_mm=140.0,
        log=lambda _: None,
    )

    assert result.n_train == 18 and result.n_val == 6
    assert 0.0 <= result.best_val_macro_auc <= 1.0
    assert set(result.val_auc_per_label) == set(LABEL_COLUMNS)

    loaded = load_model(tmp_path / "ck.pt")
    assert loaded.model.input_mode is InputMode.TRIPLETS
    assert loaded.crop_mm == 140.0
    probs = loaded.model.predict_study(torch.rand(6, 32, 32))
    assert probs.shape == (len(LABEL_COLUMNS),)


def test_ensemble_holdout_covers_missing_planes(tmp_path: Path) -> None:
    """Catches the fine-tuned ensemble eval diverging from the production combiner —
    a study whose plane has no cached stack must fall back through the plane
    sit-out path, not crash or skew the paired frozen-vs-finetuned comparison."""
    from knee.finetune import evaluate_ensemble_holdout

    targets = _synthetic_cache(tmp_path, 8, SeriesType.SAGITTAL_FLUID)
    cache_path(tmp_path, SeriesType.SAGITTAL_FLUID, 0).unlink()  # study 0 loses its plane
    val_mask = np.zeros(8, dtype=bool)
    val_mask[[0, 1, 2]] = True

    model = KneeModel(pretrained=False, input_mode=InputMode.TRIPLETS)
    model.eval()
    scores = evaluate_ensemble_holdout(tmp_path, {SeriesType.SAGITTAL_FLUID: model}, targets, val_mask)

    assert set(scores) == set(LABEL_COLUMNS)
    assert all(np.isnan(v) or 0.0 <= v <= 1.0 for v in scores.values())


def test_finetune_rejects_slice_mode_models(tmp_path: Path) -> None:
    """Catches silently fine-tuning a slices-mode model on triplet batches — the
    training input and the checkpoint's inference path would disagree, producing a
    model that scores garbage at submission time."""
    targets = _synthetic_cache(tmp_path, 8, SeriesType.AXIAL_FLUID)
    with pytest.raises(ValueError, match="TRIPLETS"):
        finetune_plane(
            tmp_path,
            targets,
            np.zeros(8, dtype=bool),
            series_type=SeriesType.AXIAL_FLUID,
            model=KneeModel(pretrained=False),
            out_path=tmp_path / "ck.pt",
            log=lambda _: None,
        )


def test_build_pixel_cache_writes_uint8_and_skips_corrupt(tmp_path: Path) -> None:
    """Catches the cache pass diverging from collect_features' skip semantics — a
    corrupt series must sit out (no file) rather than crash the 75-minute pass, and
    stacks must round-trip as uint8 so epochs read what load_volume produced."""
    rng = np.random.default_rng(0)
    study_uids = ["study0", "study_corrupt"]
    labels = pd.DataFrame([[0.9] * 12, [0.1] * 12], columns=list(LABEL_COLUMNS)).astype("float32")
    labels.insert(0, STUDY_ID_COLUMN, study_uids)
    series = pd.DataFrame(
        {
            STUDY_ID_COLUMN: study_uids,
            "SeriesInstanceUID": ["series0", "series1"],
            "Anatomical_Plane": "Sagittal",
            "Fluid_Sensitive": 1,
            "Fat_Suppression": 1,
        }
    )
    series.to_csv(tmp_path / "train_series.csv", index=False)
    good = tmp_path / "train_series" / "study0" / "series0"
    for index in range(3):
        write_dicom_slice(good / f"s{index}.dcm", rng.integers(0, 1000, (16, 16)), instance_number=index)
    bad = tmp_path / "train_series" / "study_corrupt" / "series1"
    bad.mkdir(parents=True)
    (bad / "bad.dcm").write_bytes(b"not a dicom file")

    cache_dir = tmp_path / "cache"
    result = build_pixel_cache(
        tmp_path, labels, cache_dir,
        series_types=[SeriesType.SAGITTAL_FLUID], input_size=16, log=lambda _: None,
    )

    assert result.coverage == {SeriesType.SAGITTAL_FLUID: 1}
    cached = np.load(cache_path(cache_dir, SeriesType.SAGITTAL_FLUID, 0))
    assert cached.dtype == np.uint8 and cached.shape == (3, 16, 16)
    assert not cache_path(cache_dir, SeriesType.SAGITTAL_FLUID, 1).exists()
