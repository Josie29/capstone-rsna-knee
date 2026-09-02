from pathlib import Path

import numpy as np
import pytest
import torch

from knee.finetune import FinetuneConfig, cache_path, finetune_unified
from knee.labels import LABEL_COLUMNS
from knee.model import (
    MultiPlaneModel,
    load_model,
    load_multiplane_model,
    save_multiplane_model,
)
from knee.series import SeriesType

_PLANES = [SeriesType.SAGITTAL_FLUID, SeriesType.AXIAL_FLUID]


def _small_model() -> MultiPlaneModel:
    model = MultiPlaneModel(series_types=_PLANES, pretrained=False)
    model.eval()
    return model


def test_bag_prediction_is_padding_invariant_and_handles_missing_planes() -> None:
    """Catches the two contracts the combiner used to provide: a study's prediction
    must not depend on other studies in the batch (masking), and a missing plane
    must shrink the bag rather than crash — the masked softmax renormalizes, which
    is exactly what combiner_weights did by hand."""
    model = _small_model()
    generator = torch.Generator().manual_seed(0)
    sagittal = torch.rand(6, 64, 64, generator=generator)
    axial = torch.rand(5, 64, 64, generator=generator)

    both = model.predict_study({_PLANES[0]: sagittal, _PLANES[1]: axial})
    one = model.predict_study({_PLANES[0]: sagittal})

    assert both.shape == (len(LABEL_COLUMNS),)
    assert one.shape == (len(LABEL_COLUMNS),)
    assert not torch.allclose(both, one)  # the axial items genuinely contribute

    with pytest.raises(ValueError, match="at least one plane"):
        model.predict_study({})
    with pytest.raises(ValueError, match="no plane embedding"):
        model.predict_study({SeriesType.CORONAL_FLUID: sagittal})


def test_plane_embedding_is_live() -> None:
    """Catches the "which camera" signal being dead — identical pixel content
    presented as sagittal vs axial must produce different logits, or the model has
    no way to learn per-label plane preferences and the combiner-retirement story
    is fiction."""
    model = _small_model()
    volume = torch.rand(6, 64, 64, generator=torch.Generator().manual_seed(1))

    as_sagittal = model.predict_study({_PLANES[0]: volume})
    as_axial = model.predict_study({_PLANES[1]: volume})

    assert not torch.allclose(as_sagittal, as_axial)


def test_multiplane_checkpoint_round_trips(tmp_path: Path) -> None:
    """Catches save/load dropping the plane embeddings, series order, or metadata —
    the offline submission notebook rebuilds this model from the checkpoint alone,
    and a silently reordered series_types list would swap every plane embedding."""
    model = _small_model()
    volume = torch.rand(5, 64, 64, generator=torch.Generator().manual_seed(2))
    before = model.predict_study({_PLANES[1]: volume})

    save_multiplane_model(model, tmp_path / "ck.pt", input_size=64, crop_mm=140.0, n_studies=7)
    loaded = load_multiplane_model(tmp_path / "ck.pt")

    assert loaded.series_types == _PLANES
    assert loaded.input_size == 64
    assert loaded.crop_mm == 140.0
    torch.testing.assert_close(loaded.model.predict_study({_PLANES[1]: volume}), before)


def test_legacy_loader_redirects_on_multiplane_checkpoints(tmp_path: Path) -> None:
    """Catches a multiplane checkpoint silently loading through the per-plane path —
    the error must name the right loader instead of failing with a shape mismatch
    deep inside load_state_dict at scoring time."""
    save_multiplane_model(_small_model(), tmp_path / "ck.pt", input_size=64)
    with pytest.raises(ValueError, match="use load_multiplane_model"):
        load_model(tmp_path / "ck.pt")
    with pytest.raises(ValueError, match="not a multiplane checkpoint"):
        # And the reverse: a per-plane checkpoint refuses the multiplane loader.
        from knee.model import KneeModel, save_model

        save_model(KneeModel(pretrained=False), tmp_path / "legacy.pt", input_size=64, series_type=_PLANES[0])
        load_multiplane_model(tmp_path / "legacy.pt")


def _two_plane_cache(cache_dir: Path, n_studies: int) -> np.ndarray:
    """Synthetic cache: positives carry a bright band; study 1 misses its axial plane."""
    rng = np.random.default_rng(4)
    for plane in _PLANES:
        (cache_dir / plane.value).mkdir(parents=True, exist_ok=True)
    targets = np.zeros((n_studies, len(LABEL_COLUMNS)), dtype=np.float32)
    for row in range(n_studies):
        for plane in _PLANES:
            if row == 1 and plane is _PLANES[1]:
                continue  # this study's axial plane sits out
            stack = rng.integers(40, 80, (6, 32, 32)).astype(np.uint8)
            if row % 2 == 0:
                stack[2:5, 8:24, 8:24] = 220
            np.save(cache_path(cache_dir, plane, row), stack)
        targets[row] = 0.9 if row % 2 == 0 else 0.1
    return targets


def test_finetune_unified_end_to_end(tmp_path: Path) -> None:
    """Catches integration bugs across bag assembly, padding/masking, staged
    unfreezing, best-epoch selection, and multiplane checkpoint export — including
    a study with a missing plane flowing through training AND evaluation. This
    composition otherwise only runs on Kaggle, the costliest place to debug."""
    targets = _two_plane_cache(tmp_path, 16)
    val_mask = np.zeros(16, dtype=bool)
    val_mask[[0, 1, 4, 5]] = True  # includes the missing-plane study; both classes

    model = MultiPlaneModel(series_types=_PLANES, pretrained=False)
    result = finetune_unified(
        tmp_path,
        targets,
        val_mask,
        model=model,
        out_path=tmp_path / "ck.pt",
        config=FinetuneConfig(epochs=2, frozen_epochs=1, batch_studies=8, seed=0),
        input_size=32,
        crop_mm=140.0,
        log=lambda _: None,
    )

    assert result.n_train == 12 and result.n_val == 4
    assert 0.0 <= result.best_val_macro_auc <= 1.0
    assert result.series_types == _PLANES

    loaded = load_multiplane_model(tmp_path / "ck.pt")
    assert loaded.crop_mm == 140.0
    probs = loaded.model.predict_study({_PLANES[0]: torch.rand(6, 32, 32)})
    assert probs.shape == (len(LABEL_COLUMNS),)
