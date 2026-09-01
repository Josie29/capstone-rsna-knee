import math

import numpy as np
import torch

# sklearn ships no py.typed marker, so pyright sees partially-unknown types here.
from sklearn.metrics import roc_auc_score  # pyright: ignore[reportUnknownVariableType]
from torch import nn

from knee.labels import LABEL_COLUMNS

_HEAD_EPOCHS = 300
_HEAD_LR = 1e-3


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
    # Up-weight positives per label so rare findings aren't drowned out; on soft
    # labels the counts become expected counts, which weights the same way. A label
    # with no positive mass contributes no positive terms, so weight 1 is inert.
    positives = targets.sum(dim=0)
    negatives = targets.shape[0] - positives
    pos_weight = torch.where(positives > 0, negatives / positives.clamp(min=1), torch.ones_like(positives))
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(head.parameters(), lr=_HEAD_LR)
    head.train()
    for _ in range(_HEAD_EPOCHS):  # full-batch: even 4.4k rows x 1024 features is ~18 MB
        optimizer.zero_grad()
        loss = loss_fn(head(features), targets)
        loss.backward()  # pyright: ignore[reportUnknownMemberType]
        optimizer.step()  # pyright: ignore[reportUnknownMemberType]
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
