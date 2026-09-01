from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from knee.dicom import DicomDecodeError, load_volume
from knee.labels import LABEL_COLUMNS, STUDY_ID_COLUMN, SUBMISSION_COLUMNS, Label
from knee.model import LoadedModel, load_model, resolve_device
from knee.plane_prior import combiner_weights
from knee.series import SeriesType, best_series_of_type

TEST_SERIES_DIR = "test_series"

# Emitted when no model produced a prediction for a study; matches sample_submission.
_FALLBACK_PROBABILITY = 0.5


def _predict_one(
    model: LoadedModel,
    study_series: pd.DataFrame,
    study_dir: Path,
    device: str,
    log: Callable[[str], None],
) -> np.ndarray:
    """One model's 12 probabilities for one study, or a NaN row when it can't run.

    NaN (not a fallback constant) so the ensemble mean renormalizes over the models
    that did run. Decode failures are logged and skipped, never raised — a crash at
    scoring time costs a submission.
    """
    series_uid = best_series_of_type(study_series, model.series_type)
    if series_uid is None:
        return np.full(len(LABEL_COLUMNS), np.nan)
    try:
        volume = load_volume(study_dir / series_uid, size=model.input_size)
        return model.model.predict_study(volume.to(device)).cpu().numpy()
    except (ValueError, DicomDecodeError) as exc:
        log(f"model {model.series_type}: skipping series {series_uid}: {exc}")
        return np.full(len(LABEL_COLUMNS), np.nan)


def merge_predictions(per_model: np.ndarray, series_types: list[SeriesType]) -> np.ndarray:
    """Merge per-model probability rows into one study row via the clinical plane prior.

    Each label is a weighted average over the models that ran (non-NaN rows), with
    `combiner_weights` giving the plane of choice for that finding the larger say —
    interim fixed weights until learned ones exist (DECISIONS.md #3). A study no
    model could read gets the sample-submission constant, never NaN.

    Args:
        per_model: (n_models, 12) probabilities; a model that sat out is a NaN row.
        series_types: Series type per row of `per_model`.

    Returns:
        (12,) merged probabilities in `LABEL_COLUMNS` order.
    """
    present = [index for index, row in enumerate(per_model) if not np.isnan(row).all()]
    if not present:
        return np.full(len(LABEL_COLUMNS), _FALLBACK_PROBABILITY)
    present_types = [series_types[index] for index in present]
    merged = np.empty(len(LABEL_COLUMNS))
    for column, label in enumerate(Label):
        weights = combiner_weights(present_types, label)
        merged[column] = sum(
            weight * per_model[index][column]
            for weight, index in zip(weights, present, strict=True)
        )
    return merged


def predict_studies(
    comp_root: Path,
    checkpoint_paths: list[Path],
    *,
    log: Callable[[str], None] = print,
) -> pd.DataFrame:
    """Ensemble inference over the test set: fan each study out to every model, merge.

    Each checkpoint declares its native series type; per study, each model receives
    the best series of exactly that type (or sits out via a NaN row). Rows merge by
    NaN-aware mean — the stage-1 combiner; a learned per-label combiner replaces the
    mean here later without changing callers.

    Args:
        comp_root: Competition data root containing `test.csv`, `test_series.csv`,
            and `test_series/`.
        checkpoint_paths: One .pt per model (see `knee.model.save_model`).
        log: Progress sink (`print` in notebooks).

    Returns:
        A frame with `SUBMISSION_COLUMNS`, one row per test.csv study, no NaNs —
        studies no model could read get 0.5s (never NaN, which scores as invalid).

    Raises:
        ValueError: If `checkpoint_paths` is empty or two checkpoints share a series
            type (an averaging bug: the duplicated type would be double-weighted).
    """
    if not checkpoint_paths:
        raise ValueError("predict_studies needs at least one checkpoint")
    models = [load_model(path) for path in checkpoint_paths]
    types = [m.series_type for m in models]
    if len(set(types)) != len(types):
        raise ValueError(f"Duplicate series types across checkpoints: {types}")

    device = resolve_device()
    for loaded in models:
        loaded.model.to(device)
    log(f"{len(models)} models ({[t.value for t in types]}) on {device}")

    test_df = pd.read_csv(comp_root / "test.csv")
    series_df = pd.read_csv(comp_root / "test_series.csv")

    rows: list[list[object]] = []
    for position, study_uid in enumerate(test_df[STUDY_ID_COLUMN].astype(str), start=1):
        study_series = series_df[series_df[STUDY_ID_COLUMN] == study_uid]
        study_dir = comp_root / TEST_SERIES_DIR / study_uid
        per_model = np.stack(
            [_predict_one(m, study_series, study_dir, device, log) for m in models]
        )
        if np.isnan(per_model).all():
            log(f"no model could read study {study_uid}; emitting {_FALLBACK_PROBABILITY}")
        rows.append([study_uid, *merge_predictions(per_model, types).tolist()])
        if position % 100 == 0:
            log(f"predicted {position}/{len(test_df)}")

    frame = pd.DataFrame(rows, columns=list(SUBMISSION_COLUMNS))
    if frame.isna().any().any():  # belt and braces: NaN scores as an invalid submission
        raise ValueError("NaN survived into the submission frame")
    return frame
