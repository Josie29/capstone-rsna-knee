from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd
import torch
from pydantic import BaseModel
from torch import nn

from knee.cv import (
    DEFAULT_CV_SERIES_TYPES,
    EVAL_THRESHOLD,
    FeatureBank,
    collect_features,
)
from knee.fitting import fit_head, per_label_auc
from knee.labels import LABEL_COLUMNS
from knee.model import KneeModel, save_model
from knee.series import SeriesType

# Stamped into checkpoints and filenames; bump when the labels dataset revs.
BLENDED_LABEL_SOURCE = "blended_v1"


class BlendedTrainResult(BaseModel):
    """Outcome of training one per-plane head on the blended labels."""

    series_type: SeriesType
    n_studies: int
    # In-sample, against labels thresholded at EVAL_THRESHOLD: says "the features
    # carry signal against the miner's labels", not "the model generalizes" — the
    # generalization number is the pooled-OOF CV on the same bank.
    in_sample_auc: dict[str, float]
    checkpoint_path: Path


def train_heads_from_bank(
    bank: FeatureBank,
    out_dir: Path,
    model: KneeModel,
    *,
    input_size: int = 224,
    label_source: str = BLENDED_LABEL_SOURCE,
    seed: int = 0,
    log: Callable[[str], None] = print,
) -> list[BlendedTrainResult]:
    """Fit one linear head per plane from cached features and export checkpoints.

    Heads train on the bank's labels as-is — soft probabilities stay soft targets in
    the BCE loss. Each plane's head is written into `model` and saved as a full
    checkpoint (backbone + head), so `model` MUST be the same instance (or share the
    exact backbone weights) that extracted the bank's features — otherwise the saved
    backbone would not compose with the head that was trained on those features.

    Args:
        bank: Cached features from `collect_features` (or `load_feature_bank`).
        out_dir: Directory for the `.pt` checkpoints, one per plane, named
            `{label_source}_{series_type}.pt`.
        model: The feature-extractor model; its head is overwritten per plane.
        input_size: Slice resize target the features were extracted with.
        label_source: Provenance string stamped into filenames and checkpoints.
        seed: Head-init seed, so retraining from the same bank is reproducible.
        log: Progress sink (`print` in notebooks).

    Returns:
        One result per plane, in the bank's series-type order.

    Raises:
        ValueError: If a plane has no usable studies at all — a checkpoint for it
            cannot be trained, and shipping a silently missing plane would change
            the ensemble without anyone deciding that.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[BlendedTrainResult] = []
    for plane_index, series_type in enumerate(bank.series_types):
        plane_features = bank.features[series_type]
        rows = [index for index, feature in enumerate(plane_features) if feature is not None]
        if not rows:
            raise ValueError(f"No usable studies for {series_type}; cannot train its head")
        stacked = torch.stack([f for i in rows if (f := plane_features[i]) is not None])
        targets = torch.from_numpy(bank.labels[rows])  # pyright: ignore[reportUnknownMemberType]

        torch.manual_seed(seed * 100 + plane_index)  # pyright: ignore[reportUnknownMemberType] # reproducible head init
        head = nn.Linear(stacked.shape[1], len(LABEL_COLUMNS))
        fit_head(head, stacked, targets)
        model.head.load_state_dict(head.state_dict())

        with torch.no_grad():
            probabilities = torch.sigmoid(head(stacked)).numpy()
        binary = (bank.labels[rows] >= EVAL_THRESHOLD).astype("float32")

        out_path = out_dir / f"{label_source}_{series_type.value}.pt"
        save_model(
            model,
            out_path,
            input_size=input_size,
            series_type=series_type,
            label_source=label_source,
            n_studies=len(rows),
        )
        log(f"{series_type.value}: head trained on {len(rows)} studies -> {out_path}")
        results.append(
            BlendedTrainResult(
                series_type=series_type,
                n_studies=len(rows),
                in_sample_auc=per_label_auc(binary, probabilities),
                checkpoint_path=out_path,
            )
        )
    return results


def train_blended(
    comp_root: Path,
    labels: pd.DataFrame,
    out_dir: Path,
    *,
    series_types: Sequence[SeriesType] = DEFAULT_CV_SERIES_TYPES,
    model: KneeModel | None = None,
    input_size: int = 224,
    label_source: str = BLENDED_LABEL_SOURCE,
    log: Callable[[str], None] = print,
) -> tuple[FeatureBank, list[BlendedTrainResult]]:
    """Train the per-plane fluid specialists on soft (report-mined) labels.

    One decode pass over all studies fills a feature bank (the expensive part);
    the per-plane heads then fit from cached features in seconds. The bank is
    returned so callers can persist it (`save_feature_bank`) and run the pooled-OOF
    CV protocol on it without re-decoding anything.

    Args:
        comp_root: Competition data root containing `train_series.csv` and
            `train_series/`.
        labels: One row per study to train on: `StudyInstanceUID` plus the 12 label
            columns as float probabilities — `load_blended_labels` output.
        out_dir: Directory for the per-plane checkpoints.
        series_types: The ensemble's planes, one specialist each.
        model: Model to train; a fresh pretrained `KneeModel` when omitted.
        input_size: Slice resize target fed to `load_volume`.
        label_source: Provenance string stamped into filenames and checkpoints.
        log: Progress sink (`print` in notebooks).

    Returns:
        The feature bank and one training result per plane.

    Raises:
        ValueError: Propagated from `collect_features` (bad series types) or
            `train_heads_from_bank` (a plane with zero usable studies).
    """
    model = model or KneeModel()
    bank = collect_features(
        comp_root,
        labels,
        series_types=series_types,
        model=model,
        input_size=input_size,
        log=log,
    )
    results = train_heads_from_bank(
        bank,
        out_dir,
        model,
        input_size=input_size,
        label_source=label_source,
        log=log,
    )
    return bank, results
