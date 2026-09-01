from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel

from knee.data import gold_studies
from knee.dicom import DicomDecodeError, load_volume
from knee.fitting import fit_head, per_label_auc
from knee.labels import LABEL_COLUMNS, STUDY_ID_COLUMN
from knee.model import KneeModel, resolve_device, save_model
from knee.series import TRAIN_SERIES_DIR, SeriesType, best_series_of_type


class SkippedStudy(BaseModel):
    """A gold study excluded from training, with the reason."""

    study_uid: str
    reason: str


class GoldTrainResult(BaseModel):
    """Outcome of a gold-58 training run."""

    series_type: SeriesType
    n_studies: int
    skipped: list[SkippedStudy]
    # In-sample only: the head trains on all rows, so these say "the features carry
    # signal", not "the model generalizes". NaN when a label had a single class.
    in_sample_auc: dict[str, float]
    checkpoint_path: Path


def train_gold(
    comp_root: Path,
    out_path: Path,
    *,
    series_type: SeriesType = SeriesType.SAGITTAL_FLUID,
    model: KneeModel | None = None,
    input_size: int = 224,
    log: Callable[[str], None] = print,
) -> GoldTrainResult:
    """Train one per-type prototype on the fully-labeled gold studies.

    One `KneeModel` specialized to `series_type`, backbone frozen, only the linear
    head trained — all gold studies used for fitting (no validation; n=58 cannot
    support one). Studies without a series of the type are skipped, never substituted
    (the strict-typing rule from DECISIONS.md). Checkpoints are `pipe_check_gold58`
    prototypes and must never be evaluated against the gold studies.

    Args:
        comp_root: Competition data root containing `train.csv`, `train_series.csv`,
            and `train_series/` (on Kaggle:
            `/kaggle/input/rsna-knee-abnormality-detection`).
        out_path: Where to write the .pt checkpoint.
        series_type: The one series type this model trains on and will consume.
        model: Model to train; a fresh pretrained `KneeModel` when omitted.
        input_size: Slice resize target fed to `load_volume`.
        log: Progress sink (`print` in notebooks).

    Returns:
        The training result; studies whose series failed to decode are skipped and
        recorded rather than aborting the run.

    Raises:
        ValueError: If no gold study could be loaded at all.
    """
    train_df = pd.read_csv(comp_root / "train.csv")
    series_df = pd.read_csv(comp_root / "train_series.csv")
    gold = gold_studies(train_df)
    log(f"{len(gold)} gold studies")

    model = model or KneeModel()
    model.freeze_backbone()
    device = resolve_device()
    model.to(device)
    model.eval()

    # Features and labels are appended together so they cannot fall out of alignment
    # when a study is skipped.
    samples: list[tuple[torch.Tensor, np.ndarray]] = []
    skipped: list[SkippedStudy] = []
    for position, (_, row) in enumerate(gold.iterrows(), start=1):
        study_uid = str(row[STUDY_ID_COLUMN])
        study_series = series_df[series_df[STUDY_ID_COLUMN] == study_uid]
        try:
            series_uid = best_series_of_type(study_series, series_type)
            if series_uid is None:
                raise ValueError(f"no {series_type} series")
            volume = load_volume(comp_root / TRAIN_SERIES_DIR / study_uid / series_uid, size=input_size)
            with torch.no_grad():  # backbone is frozen; cache features once
                study_features = model.pool_features(volume.to(device)).cpu()
            samples.append((study_features, row[list(LABEL_COLUMNS)].to_numpy(dtype=np.float32)))
        except (ValueError, DicomDecodeError) as exc:
            skipped.append(SkippedStudy(study_uid=study_uid, reason=str(exc)))
            log(f"skipping {study_uid}: {exc}")
        if position % 10 == 0:
            log(f"processed {position}/{len(gold)}")

    if not samples:
        raise ValueError("Every gold study failed to load; cannot train")

    features = torch.stack([study_features for study_features, _ in samples])
    labels = np.stack([study_labels for _, study_labels in samples])
    targets = torch.from_numpy(labels)  # pyright: ignore[reportUnknownMemberType]

    model.to("cpu")  # head training on cached features is trivial; keep it simple
    fit_head(model.head, features, targets)

    with torch.no_grad():
        probabilities = torch.sigmoid(model.head(features)).numpy()

    save_model(
        model,
        out_path,
        input_size=input_size,
        series_type=series_type,
        label_source="gold58",
        n_studies=len(samples),
    )
    log(f"checkpoint -> {out_path}")

    return GoldTrainResult(
        series_type=series_type,
        n_studies=len(samples),
        skipped=skipped,
        in_sample_auc=per_label_auc(labels, probabilities),
        checkpoint_path=out_path,
    )
