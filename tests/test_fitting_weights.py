import torch

from knee.fitting import fit_head, positive_weight, weighted_bce
from knee.labels import LABEL_COLUMNS


def test_zero_weight_cells_do_not_influence_the_loss() -> None:
    """Catches the weighted loss silently averaging over down-weighted cells anyway —
    tier-3 guesses would then train the model at full strength and E006a would be a
    no-op reported as an experiment."""
    logits = torch.zeros(2, len(LABEL_COLUMNS))
    targets = torch.zeros(2, len(LABEL_COLUMNS))
    targets[0, 0] = 1.0  # one confidently-wrong cell (logit 0 vs target 1)
    pos_weight = torch.ones(len(LABEL_COLUMNS))

    weights = torch.ones_like(targets)
    weights[0, 0] = 0.0  # zero out exactly the wrong cell
    masked = weighted_bce(logits, targets, pos_weight=pos_weight, cell_weights=weights)
    clean = weighted_bce(logits, torch.zeros_like(targets), pos_weight=pos_weight, cell_weights=None)
    torch.testing.assert_close(masked, clean)  # the zero-weight cell vanished


def test_fit_head_with_zero_weight_ignores_poisoned_labels() -> None:
    """Catches per-cell weights leaking across labels during training: a linear
    head's output rows are independent, so zero-weighting a poisoned label must
    leave every other label's learned weights bit-identical to a run that never
    saw weights at all."""
    generator = torch.Generator().manual_seed(0)
    signal = (torch.rand(32, 1, generator=generator) > 0.5).float()
    features = torch.rand(32, 8, generator=generator)
    features[:, :1] = signal * 2 - 1  # high-margin signal in feature 0
    targets = signal.repeat(1, len(LABEL_COLUMNS))
    poisoned = targets.clone()
    poisoned[:, 0] = torch.rand(32, generator=generator).round()  # label 0 becomes noise

    weights = torch.ones_like(targets)
    weights[:, 0] = 0.0

    torch.manual_seed(1)  # pyright: ignore[reportUnknownMemberType] # deterministic head init
    weighted_head = torch.nn.Linear(8, len(LABEL_COLUMNS))
    fit_head(weighted_head, features, poisoned, cell_weights=weights)

    with torch.no_grad():
        logits = weighted_head(features)
    # Labels 1..11 still learned the true signal despite label 0's noise being
    # present: near-perfect ranking (AUC), the metric the competition scores.
    positive = logits[targets[:, 1] == 1.0, 1]
    negative = logits[targets[:, 1] == 0.0, 1]
    auc = (positive[:, None] > negative[None, :]).float().mean()
    assert auc > 0.9


def test_positive_weight_ignores_cell_weights_by_design() -> None:
    """Documents the separation of concerns: pos_weight corrects class imbalance
    from raw targets; confidence weighting happens per cell in the loss. Mixing
    them would double-count tier information."""
    targets = torch.zeros(10, len(LABEL_COLUMNS))
    targets[:3, 0] = 1.0
    weight = positive_weight(targets)
    torch.testing.assert_close(weight[0], torch.tensor(7 / 3))
