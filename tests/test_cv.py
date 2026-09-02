from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import write_dicom_slice

from knee.cv import (
    FeatureBank,
    _stratified_fold_assignments,  # pyright: ignore[reportPrivateUsage]
    collect_features,
    cross_validate,
    evaluate_holdout,
    load_feature_bank,
    save_feature_bank,
)
from knee.labels import LABEL_COLUMNS, STUDY_ID_COLUMN
from knee.model import HeadType, KneeModel
from knee.series import SeriesType

_FEATURE_DIM = 16


def _synthetic_bank(
    missing: dict[SeriesType, set[int]] | None = None,
    *,
    soft: bool = False,
) -> FeatureBank:
    """9 studies x 2 planes with deterministic features; `missing` knocks planes out."""
    missing = missing or {}
    rng = np.random.default_rng(7)
    labels = rng.integers(0, 2, (9, len(LABEL_COLUMNS))).astype(np.float32)
    labels[0] = 1.0  # force both classes into every label column
    labels[1] = 0.0
    if soft:
        # Nudge hard labels into open-interval probabilities that still threshold back
        # to the same classes at 0.5.
        labels = np.clip(labels, 0.05, 0.95) + rng.uniform(-0.04, 0.04, labels.shape).astype(np.float32)
    generator = torch.Generator().manual_seed(7)
    series_types = [SeriesType.SAGITTAL_FLUID, SeriesType.AXIAL_FLUID]
    # Per-slice matrices with deliberately varying slice counts (3-5 per study).
    features = {
        series_type: [
            None
            if study in missing.get(series_type, set())
            else torch.rand(3 + study % 3, _FEATURE_DIM, generator=generator)
            for study in range(9)
        ]
        for series_type in series_types
    }
    return FeatureBank(
        series_types=series_types,
        study_uids=[f"study{i}" for i in range(9)],
        labels=labels,
        features=features,
    )


def _multiplane_comp_root(root: Path) -> tuple[Path, pd.DataFrame]:
    """3 studies: full 2-plane, sagittal-only, and corrupt-sagittal + good axial."""
    rng = np.random.default_rng(0)
    study_uids = ["study0", "study1", "study2"]
    label_rows = [[1.0] * 12, [0.0] * 12, [i % 2 for i in range(12)]]

    labels = pd.DataFrame(label_rows, columns=list(LABEL_COLUMNS)).astype("float32")
    labels.insert(0, STUDY_ID_COLUMN, study_uids)

    series_layout = [
        ("study0", "series0-sag", "Sagittal", False),
        ("study0", "series0-ax", "Axial", False),
        ("study1", "series1-sag", "Sagittal", False),
        ("study2", "series2-sag", "Sagittal", True),
        ("study2", "series2-ax", "Axial", False),
    ]
    series = pd.DataFrame(
        {
            STUDY_ID_COLUMN: [s for s, _, _, _ in series_layout],
            "SeriesInstanceUID": [u for _, u, _, _ in series_layout],
            "Anatomical_Plane": [p for _, _, p, _ in series_layout],
            "Fluid_Sensitive": 1,
            "Fat_Suppression": 1,
        }
    )
    series.to_csv(root / "train_series.csv", index=False)

    for study_uid, series_uid, _, corrupt in series_layout:
        series_dir = root / "train_series" / study_uid / series_uid
        series_dir.mkdir(parents=True)
        if corrupt:
            (series_dir / "bad.dcm").write_bytes(b"not a dicom file")
            continue
        for slice_index in range(2):
            write_dicom_slice(
                series_dir / f"slice{slice_index}.dcm",
                rng.integers(0, 1000, (32, 32)),
                instance_number=slice_index,
            )
    return root, labels


def test_collect_features_aligns_planes_and_survives_bad_series(tmp_path: Path) -> None:
    """Catches misalignment between labels and per-plane features when a study lacks a
    plane or a series fails to decode mid-loop — a shifted row would train every fold's
    heads on the wrong studies and make the reported val AUC meaningless."""
    comp_root, labels = _multiplane_comp_root(tmp_path)

    bank = collect_features(
        comp_root,
        labels,
        series_types=[SeriesType.SAGITTAL_FLUID, SeriesType.AXIAL_FLUID],
        model=KneeModel(pretrained=False),
        input_size=64,
        log=lambda _: None,
    )

    assert bank.study_uids == ["study0", "study1", "study2"]
    assert bank.labels.shape == (3, len(LABEL_COLUMNS))
    # study2's sagittal is corrupt (None) but its axial still loads; study1 has no axial.
    sagittal = bank.features[SeriesType.SAGITTAL_FLUID]
    axial = bank.features[SeriesType.AXIAL_FLUID]
    assert [f is not None for f in sagittal] == [True, True, False]
    assert [f is not None for f in axial] == [True, False, True]
    assert bank.plane_coverage() == {SeriesType.SAGITTAL_FLUID: 2, SeriesType.AXIAL_FLUID: 2}


def test_feature_bank_round_trips_through_disk(tmp_path: Path) -> None:
    """Catches a persistence bug that would silently invalidate every downstream head
    fit and CV run against a reloaded bank — the whole point of persisting is that a
    reloaded bank is interchangeable with a freshly extracted one."""
    bank = _synthetic_bank(missing={SeriesType.AXIAL_FLUID: {3}}, soft=True)

    save_feature_bank(bank, tmp_path / "bank.pt")
    loaded = load_feature_bank(tmp_path / "bank.pt")

    assert loaded.series_types == bank.series_types
    assert loaded.study_uids == bank.study_uids
    np.testing.assert_array_equal(loaded.labels, bank.labels)
    for series_type in bank.series_types:
        for original, restored in zip(bank.features[series_type], loaded.features[series_type], strict=True):
            if original is None or restored is None:
                assert original is None and restored is None
            else:
                # Storage is fp16 by contract; a reloaded bank equals the fp16 cast.
                assert restored.dtype == torch.float32
                torch.testing.assert_close(original.half().float(), restored)


def test_cross_validate_is_complete_and_reproducible() -> None:
    """Catches nondeterminism (unseeded folds or head init) and gaps in the OOF matrix —
    either would make experiments.md val-AUC rows incomparable across runs, which is
    the whole point of the protocol."""
    bank = _synthetic_bank(missing={SeriesType.AXIAL_FLUID: {3}})

    first = cross_validate(bank, n_splits=3, n_repeats=2, seed=0, log=lambda _: None)
    second = cross_validate(bank, n_splits=3, n_repeats=2, seed=0, log=lambda _: None)

    assert first.oof_probabilities.shape == (2, 9, len(LABEL_COLUMNS))
    assert not np.isnan(first.oof_probabilities).any()  # every study gets an OOF row
    assert set(first.per_label_auc) == set(LABEL_COLUMNS)
    assert all(0.0 <= v <= 1.0 for v in first.per_label_auc.values())
    assert 0.0 <= first.macro_auc <= 1.0
    assert len(first.macro_auc_per_repeat) == 2
    assert first.macro_auc == second.macro_auc
    np.testing.assert_array_equal(first.oof_probabilities, second.oof_probabilities)


def test_cross_validate_attention_head_is_complete_and_reproducible() -> None:
    """Catches the attention CV path diverging from the protocol contract — gaps in
    the OOF matrix or unseeded fits would make the E005b A/B against mean_max
    meaningless, since the whole comparison rests on identical folds and repeatable
    numbers."""
    bank = _synthetic_bank(missing={SeriesType.AXIAL_FLUID: {3}})

    first = cross_validate(bank, head_type=HeadType.ATTENTION, n_splits=3, n_repeats=1, seed=0, log=lambda _: None)
    second = cross_validate(bank, head_type=HeadType.ATTENTION, n_splits=3, n_repeats=1, seed=0, log=lambda _: None)

    assert not np.isnan(first.oof_probabilities).any()
    assert 0.0 <= first.macro_auc <= 1.0
    assert first.macro_auc == second.macro_auc


def test_evaluate_holdout_scores_only_the_validation_side() -> None:
    """Catches the E006 paired baseline leaking training studies into its score —
    the frozen-vs-finetuned comparison is only meaningful if both numbers come from
    models that never saw the validation split."""
    bank = _synthetic_bank(missing={SeriesType.AXIAL_FLUID: {3}})
    val_mask = np.zeros(9, dtype=bool)
    val_mask[[0, 1, 2]] = True

    first = evaluate_holdout(bank, val_mask, seed=0)
    second = evaluate_holdout(bank, val_mask, seed=0)
    weighted = evaluate_holdout(bank, val_mask, cell_weights=np.ones((9, 12), dtype=np.float32), seed=0)

    assert set(first) == set(LABEL_COLUMNS)
    assert first == second  # seeded: reproducible for the paired comparison
    # All-ones weights are a no-op, so the weighted path must agree exactly.
    assert first == weighted


def test_cross_validate_accepts_soft_labels() -> None:
    """Catches the blended-labels regression where continuous targets reach
    stratification or roc_auc_score directly — either raises (sklearn) or silently
    treats every study as positive (stratifier), and the local eval protocol for the
    4.4k-study regime stops working."""
    bank = _synthetic_bank(soft=True)

    result = cross_validate(bank, n_splits=3, n_repeats=1, seed=0, log=lambda _: None)

    assert not np.isnan(result.oof_probabilities).any()
    assert 0.0 <= result.macro_auc <= 1.0


def test_study_no_plane_can_read_gets_submission_fallback() -> None:
    """Catches the local protocol diverging from inference for unreadable studies —
    inference emits 0.5s for them, so CV must too or the local AUC measures a
    different pipeline than the one the leaderboard scores."""
    bank = _synthetic_bank(
        missing={SeriesType.SAGITTAL_FLUID: {4}, SeriesType.AXIAL_FLUID: {4}}
    )

    result = cross_validate(bank, n_splits=3, n_repeats=1, seed=0, log=lambda _: None)

    assert (result.oof_probabilities[0, 4] == 0.5).all()


def test_rare_label_positives_spread_across_folds() -> None:
    """Catches unstratified splitting: with a rare label, a random fold can hold every
    positive, leaving its training complement unable to learn the label at all and
    adding pure noise to the val AUC the team compares levers with."""
    labels = np.zeros((9, len(LABEL_COLUMNS)), dtype=np.float32)
    labels[:, 1:] = np.random.default_rng(1).integers(0, 2, (9, len(LABEL_COLUMNS) - 1))
    labels[[0, 4, 8], 0] = 1.0  # rare label: exactly 3 positives across 9 studies

    assignment = _stratified_fold_assignments(labels, n_splits=3, rng=np.random.default_rng(0))

    assert sorted(np.bincount(assignment).tolist()) == [3, 3, 3]
    positives_per_fold = [int(labels[assignment == fold, 0].sum()) for fold in range(3)]
    assert positives_per_fold == [1, 1, 1]


def test_invalid_fold_counts_are_rejected() -> None:
    """Catches a silent degenerate protocol — k=1 would evaluate models on their own
    training data, reporting memorization (AUC 1.0) as validation skill."""
    bank = _synthetic_bank()
    with pytest.raises(ValueError, match="n_splits"):
        cross_validate(bank, n_splits=1, log=lambda _: None)
    with pytest.raises(ValueError, match="n_repeats"):
        cross_validate(bank, n_splits=3, n_repeats=0, log=lambda _: None)
