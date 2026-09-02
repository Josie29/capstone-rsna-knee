import math

import numpy as np
import torch

# sklearn ships no py.typed marker, so pyright sees partially-unknown types here.
from sklearn.metrics import roc_auc_score  # pyright: ignore[reportUnknownVariableType]
from torch import nn

from knee.labels import LABEL_COLUMNS
from knee.model import PerLabelAttentionHead, resolve_device

_HEAD_EPOCHS = 300
_HEAD_LR = 1e-3

# ~100k-param head on cached features: 100 epochs x (n/128) steps is minutes at
# 4.4k studies. Batch small enough that even tiny CV training folds get real
# step counts — 40 full-batch epochs measurably underfit the MIL toy problem.
_ATTENTION_EPOCHS = 100
_ATTENTION_LR = 1e-3
_ATTENTION_BATCH = 128
_ATTENTION_WEIGHT_DECAY = 1e-4


def _positive_weight(targets: torch.Tensor) -> torch.Tensor:
    """Per-label BCE pos_weight so rare findings aren't drowned out.

    On soft labels the counts become expected counts, which weights the same way.
    A label with no positive mass contributes no positive terms, so weight 1 is inert.
    """
    positives = targets.sum(dim=0)
    negatives = targets.shape[0] - positives
    return torch.where(positives > 0, negatives / positives.clamp(min=1), torch.ones_like(positives))


def pad_slice_features(features: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length slice matrices into one batch tensor plus a validity mask.

    Args:
        features: Per-study (n_slices_i, feature_dim) matrices, n_slices_i varying.

    Returns:
        Tuple of (batch, max_slices, feature_dim) zero-padded features and a
        (batch, max_slices) bool mask, True where a slice is real.

    Raises:
        ValueError: If `features` is empty.
    """
    if not features:
        raise ValueError("pad_slice_features needs at least one study")
    max_slices = max(matrix.shape[0] for matrix in features)
    padded = torch.zeros(len(features), max_slices, features[0].shape[1])
    mask = torch.zeros(len(features), max_slices, dtype=torch.bool)
    for row, matrix in enumerate(features):
        padded[row, : matrix.shape[0]] = matrix
        mask[row, : matrix.shape[0]] = True
    return padded, mask


def fit_head(head: nn.Linear, features: torch.Tensor, targets: torch.Tensor) -> None:
    """Train the linear head on cached study features.

    Valid only because the backbone is frozen: `model(volume)` equals
    `head(pool_features(volume))`, so training on cached features trains the same
    function the checkpoint will compute.

    Args:
        head: The model's head, trained in place.
        features: (n_studies, feature_dim) pooled study features.
        targets: (n_studies, 12) float labels — hard 0/1 or soft probabilities;
            BCE accepts both.
    """
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=_positive_weight(targets))

    optimizer = torch.optim.Adam(head.parameters(), lr=_HEAD_LR)
    head.train()
    for _ in range(_HEAD_EPOCHS):  # full-batch: even 4.4k rows x 1024 features is ~18 MB
        optimizer.zero_grad()
        loss = loss_fn(head(features), targets)
        loss.backward()  # pyright: ignore[reportUnknownMemberType]
        optimizer.step()  # pyright: ignore[reportUnknownMemberType]
    head.eval()


def fit_attention_head(
    head: PerLabelAttentionHead,
    slice_features: list[torch.Tensor],
    targets: torch.Tensor,
    *,
    seed: int = 0,
) -> None:
    """Train the per-label attention head on cached per-slice study features.

    Valid only because the backbone is frozen (same argument as `fit_head`): the
    head sees exactly the slice features the checkpointed model will compute.
    Mini-batched with per-batch padding because slice counts vary per study; small
    weight decay plus the head's own dropout guard against a learnable pooler
    fitting miner label noise. Trains on the GPU when one works (the T4 otherwise
    idles through the CV's ~dozens of fits) and returns the head on CPU, so
    checkpointing and CPU-side prediction are unaffected.

    Args:
        head: The attention head, trained in place; on CPU when this returns.
        slice_features: Per-study (n_slices_i, feature_dim) matrices, aligned with
            `targets` rows.
        targets: (n_studies, 12) float labels — hard 0/1 or soft probabilities.
        seed: Shuffling seed, so refits from the same bank are reproducible.

    Raises:
        ValueError: If `slice_features` and `targets` disagree on study count.
    """
    if len(slice_features) != targets.shape[0]:
        raise ValueError(f"{len(slice_features)} feature matrices vs {targets.shape[0]} target rows")
    device = resolve_device()
    head.to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=_positive_weight(targets).to(device))
    optimizer = torch.optim.Adam(head.parameters(), lr=_ATTENTION_LR, weight_decay=_ATTENTION_WEIGHT_DECAY)
    generator = torch.Generator().manual_seed(seed)
    head.train()
    for _ in range(_ATTENTION_EPOCHS):
        order = torch.randperm(len(slice_features), generator=generator)
        for start in range(0, len(slice_features), _ATTENTION_BATCH):
            batch = order[start : start + _ATTENTION_BATCH]
            padded, mask = pad_slice_features([slice_features[int(i)] for i in batch])
            optimizer.zero_grad()
            loss = loss_fn(head(padded.to(device), mask.to(device)), targets[batch].to(device))
            loss.backward()  # pyright: ignore[reportUnknownMemberType]
            optimizer.step()  # pyright: ignore[reportUnknownMemberType]
    head.to("cpu")
    head.eval()


def per_label_auc(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """Per-label AUC; NaN where a label has a single class.

    Args:
        targets: (n_studies, 12) binary array in `LABEL_COLUMNS` order — soft labels
            must be thresholded by the caller first (AUC is undefined on continuous
            targets).
        probabilities: (n_studies, 12) predicted probabilities.

    Returns:
        Label name -> AUC, NaN where the targets carry a single class.
    """
    scores: dict[str, float] = {}
    for column, label in enumerate(LABEL_COLUMNS):
        if len(np.unique(targets[:, column])) < 2:  # AUC undefined for a single class
            scores[label] = math.nan
            continue
        score = roc_auc_score(targets[:, column], probabilities[:, column])  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]
        scores[label] = float(score)  # pyright: ignore[reportUnknownArgumentType]
    return scores
