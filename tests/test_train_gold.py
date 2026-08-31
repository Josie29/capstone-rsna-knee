from pathlib import Path

import pytest
import torch

from knee.labels import LABEL_COLUMNS
from knee.model import KneeModel, load_model, save_model

# Random-init (pretrained=False) so tests run offline; small volumes keep them fast.
_VOLUME = torch.rand(4, 64, 64, generator=torch.Generator().manual_seed(0))


def test_model_round_trips_through_checkpoint(tmp_path: Path) -> None:
    """Catches the bug where save/load drops or misaligns weights — predictions after
    reload on Kaggle would silently differ from what training validated, and the
    offline submission notebook has no way to notice."""
    model = KneeModel(pretrained=False)
    model.eval()
    before = model.predict_study(_VOLUME)

    save_model(model, tmp_path / "ck.pt", input_size=64)
    reloaded, input_size = load_model(tmp_path / "ck.pt")
    after = reloaded.predict_study(_VOLUME)

    assert input_size == 64
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
    save_model(model, tmp_path / "ck.pt", input_size=64)
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
