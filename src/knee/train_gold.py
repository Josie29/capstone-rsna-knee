import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel

# sklearn ships no py.typed marker, so pyright sees partially-unknown types here.
from sklearn.metrics import roc_auc_score  # pyright: ignore[reportUnknownVariableType]

from knee.data import gold_studies
from knee.dicom import DicomDecodeError, load_volume
from knee.labels import LABEL_COLUMNS, STUDY_ID_COLUMN
from knee.model import KneeModel, save_model
from knee.series import select_series

TRAIN_SERIES_DIR = "train_series"

_HEAD_EPOCHS = 300
_HEAD_LR = 1e-3


class SkippedStudy(BaseModel):
    """A gold study excluded from training, with the reason."""

    study_uid: str
    reason: str


class GoldTrainResult(BaseModel):
    """Outcome of a gold-58 training run."""

    n_studies: int
    skipped: list[SkippedStudy]
    # In-sample only: the head trains on all rows, so these say "the features carry
    # signal", not "the model generalizes". NaN when a label had a single class.
    in_sample_auc: dict[str, float]
    checkpoint_path: Path


def _fit_head(model: KneeModel, features: torch.Tensor, targets: torch.Tensor) -> None:
    """Train the linear head on cached study features, backbone untouched.

    Args:
        model: The model whose `head` is trained in place.
        features: (n_studies, feature_dim) pooled study features.
        targets: (n_studies, 12) float 0/1 labels.
    """
    # Up-weight positives per label so rare findings (MCL: 9/58) aren't drowned out;
    # a label with no positives contributes no positive terms, so weight 1 is inert.
    positives = targets.sum(dim=0)
    negatives = targets.shape[0] - positives
    pos_weight = torch.where(positives > 0, negatives / positives.clamp(min=1), torch.ones_like(positives))
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.head.parameters(), lr=_HEAD_LR)
    model.head.train()
    for _ in range(_HEAD_EPOCHS):  # full-batch: 58 rows, converges in seconds
        optimizer.zero_grad()
        loss = loss_fn(model.head(features), targets)
        loss.backward()  # pyright: ignore[reportUnknownMemberType]
        optimizer.step()  # pyright: ignore[reportUnknownMemberType]
    model.head.eval()


def train_gold(
    comp_root: Path,
    out_path: Path,
    *,
    model: KneeModel | None = None,
    input_size: int = 224,
    log: Callable[[str], None] = print,
) -> GoldTrainResult:
    """Train the prototype on the fully-labeled gold studies.

    The vertical-slice trainer for issue #6: one `KneeModel` with the backbone frozen
    and only the linear head trained — all gold studies used for fitting (no
    validation; n=58 cannot support one). The checkpoint is named for
    `pipe_check_gold58` and must never be evaluated against the gold studies.

    Args:
        comp_root: Competition data root containing `train.csv`, `train_series.csv`,
            and `train_series/` (on Kaggle:
            `/kaggle/input/rsna-knee-abnormality-detection`).
        out_path: Where to write the .pt checkpoint.
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    feature_rows: list[torch.Tensor] = []
    used_rows: list[int] = []
    skipped: list[SkippedStudy] = []
    for position, (row_index, row) in enumerate(gold.iterrows(), start=1):
        study_uid = str(row[STUDY_ID_COLUMN])
        study_series = series_df[series_df[STUDY_ID_COLUMN] == study_uid]
        try:
            series_uid = select_series(study_series)
            volume = load_volume(comp_root / TRAIN_SERIES_DIR / study_uid / series_uid, size=input_size)
            with torch.inference_mode():  # backbone is frozen; cache features once
                feature_rows.append(model.pool_features(volume.to(device)).cpu())
            used_rows.append(row_index)  # pyright: ignore[reportArgumentType]
        except (ValueError, DicomDecodeError) as exc:
            skipped.append(SkippedStudy(study_uid=study_uid, reason=str(exc)))
            log(f"skipping {study_uid}: {exc}")
        if position % 10 == 0:
            log(f"processed {position}/{len(gold)}")

    if not feature_rows:
        raise ValueError("Every gold study failed to load; cannot train")

    features = torch.stack(feature_rows)
    labels = gold.loc[used_rows, list(LABEL_COLUMNS)].to_numpy(dtype=np.float32)
    targets = torch.from_numpy(labels)  # pyright: ignore[reportUnknownMemberType]

    model.to("cpu")  # head training on cached features is trivial; keep it simple
    _fit_head(model, features, targets)

    with torch.inference_mode():
        probabilities = torch.sigmoid(model.head(features)).numpy()
    in_sample_auc: dict[str, float] = {}
    for column_index, label in enumerate(LABEL_COLUMNS):
        column_targets = labels[:, column_index]
        if len(np.unique(column_targets)) < 2:  # AUC undefined for a single class
            in_sample_auc[label] = math.nan
            continue
        score = roc_auc_score(column_targets, probabilities[:, column_index])  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]
        in_sample_auc[label] = float(score)  # pyright: ignore[reportUnknownArgumentType]

    save_model(model, out_path, input_size=input_size)
    log(f"checkpoint -> {out_path}")

    return GoldTrainResult(
        n_studies=len(used_rows),
        skipped=skipped,
        in_sample_auc=in_sample_auc,
        checkpoint_path=out_path,
    )
