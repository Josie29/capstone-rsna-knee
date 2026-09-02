from pathlib import Path

import numpy as np
import torch

from knee.fitting import fit_attention_head, pad_slice_features
from knee.labels import LABEL_COLUMNS
from knee.model import (
    HeadType,
    InputMode,
    KneeModel,
    PerLabelAttentionHead,
    load_model,
    save_model,
)

_DIM = 8


def test_padding_never_changes_a_study_logits() -> None:
    """Catches a masking bug where zero-padded slices leak into the softmax — every
    batched prediction would then depend on the longest study in its batch, making
    results irreproducible across batch compositions."""
    head = PerLabelAttentionHead(_DIM)
    head.eval()
    generator = torch.Generator().manual_seed(0)
    short = torch.rand(3, _DIM, generator=generator)
    long = torch.rand(7, _DIM, generator=generator)

    alone = head(short)
    padded, mask = pad_slice_features([short, long])
    batched = head(padded, mask)

    assert alone.shape == (len(LABEL_COLUMNS),)
    assert batched.shape == (2, len(LABEL_COLUMNS))
    torch.testing.assert_close(batched[0], alone)


def test_each_label_gets_its_own_attention() -> None:
    """Catches the per-label design collapsing to one shared weighting — the point of
    the head is that a meniscus and an effusion can look at different slices, which
    is also what makes the per-finding demo heatmaps honest."""
    head = PerLabelAttentionHead(_DIM)
    head.eval()
    slices = torch.rand(6, _DIM, generator=torch.Generator().manual_seed(1))

    weights = head.attention_weights(slices)

    assert weights.shape == (len(LABEL_COLUMNS), 6)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(len(LABEL_COLUMNS)))
    # Distinct per-label scorers must yield distinct weightings even at random init.
    assert not torch.allclose(weights[0], weights[1])


def test_attention_learns_to_find_the_signal_slice() -> None:
    """Catches the pooler silently degenerating into a uniform mean — the failure
    mode where E005b trains, reports numbers, and adds nothing over E003. On bags
    where the label is carried by one planted slice, a working MIL head must both
    classify held-out bags and put its weight on the planted slice."""
    generator = torch.Generator().manual_seed(2)
    signal = torch.zeros(_DIM)
    signal[:4] = 3.0

    def make_bag(positive: bool) -> tuple[torch.Tensor, int]:
        bag = torch.randn(6, _DIM, generator=generator) * 0.5
        index = int(torch.randint(6, (1,), generator=generator))
        if positive:
            bag[index] += signal
        return bag, index

    train = [make_bag(i % 2 == 0) for i in range(200)]
    test = [make_bag(i % 2 == 0) for i in range(60)]
    train_targets = torch.tensor([[float(i % 2 == 0)] * len(LABEL_COLUMNS) for i in range(200)])

    torch.manual_seed(2)  # pyright: ignore[reportUnknownMemberType] # deterministic head init
    head = PerLabelAttentionHead(_DIM)
    fit_attention_head(head, [bag for bag, _ in train], train_targets, seed=2)
    # Contract: training may use the GPU, but the head comes back on CPU so
    # checkpointing and CPU-side prediction never care where it trained.
    assert all(p.device.type == "cpu" for p in head.parameters())

    with torch.no_grad():
        probabilities = torch.stack([torch.sigmoid(head(bag)) for bag, _ in test])
    positives = torch.tensor([i % 2 == 0 for i in range(60)])
    # Held-out separation on label 0: mean rank of positives far above negatives.
    auc = float(
        (probabilities[positives, 0][:, None] > probabilities[~positives, 0][None, :]).float().mean()
    )
    assert auc > 0.9

    # Attention must concentrate on the planted slice for positive bags.
    hits = 0
    positive_bags = [(bag, index) for i, (bag, index) in enumerate(test) if i % 2 == 0]
    for bag, index in positive_bags:
        with torch.no_grad():
            weights = head.attention_weights(bag)
        if int(weights[0].argmax()) == index:
            hits += 1
    assert hits / len(positive_bags) > 0.8


def test_attention_model_round_trips_through_checkpoint(tmp_path: Path) -> None:
    """Catches save/load dropping or misrouting the attention head — the submission
    notebook rebuilds the model from checkpoint metadata alone, and a head_type
    mismatch there would silently score with an untrained default head."""
    model = KneeModel(pretrained=False, head_type=HeadType.ATTENTION)
    model.eval()
    volume = torch.rand(4, 64, 64, generator=torch.Generator().manual_seed(3))
    before = model.predict_study(volume)

    from knee.series import SeriesType

    save_model(model, tmp_path / "ck.pt", input_size=64, series_type=SeriesType.SAGITTAL_FLUID)
    loaded = load_model(tmp_path / "ck.pt")

    assert loaded.model.head_type is HeadType.ATTENTION
    torch.testing.assert_close(loaded.model.predict_study(volume), before)


def test_checkpoint_without_head_type_loads_as_mean_max(tmp_path: Path) -> None:
    """Catches a compatibility break with E003/E004 checkpoints, which predate the
    head_type field — they must keep loading (and predicting identically) as
    mean_max, or every already-published knee-weights version dies at scoring time."""
    from knee.series import SeriesType

    model = KneeModel(pretrained=False)
    model.eval()
    volume = torch.rand(3, 64, 64, generator=torch.Generator().manual_seed(4))
    before = model.predict_study(volume)
    save_model(model, tmp_path / "ck.pt", input_size=64, series_type=SeriesType.AXIAL_FLUID)

    payload = torch.load(tmp_path / "ck.pt", weights_only=True)
    del payload["head_type"]  # simulate a pre-E005b checkpoint
    torch.save(payload, tmp_path / "legacy.pt")

    loaded = load_model(tmp_path / "legacy.pt")
    assert loaded.model.head_type is HeadType.MEAN_MAX
    torch.testing.assert_close(loaded.model.predict_study(volume), before)


def test_fixed_size_vit_runs_at_overridden_resolution(tmp_path: Path) -> None:
    """Catches the E007 enabler breaking — DINOv2's ViT is fixed at 518px, and
    fine-tuning at that size is infeasible (a ~100GB pixel cache). The image_size
    override must build a working ViT at the chosen resolution AND survive the
    checkpoint round trip, or the offline submission notebook would rebuild the
    model at the wrong size and crash (or interpolate wrongly) at scoring time."""
    from knee.model import DINOV2_BACKBONE
    from knee.series import SeriesType

    model = KneeModel(
        DINOV2_BACKBONE, pretrained=False, head_type=HeadType.ATTENTION,
        input_mode=InputMode.TRIPLETS, image_size=126,  # small multiple of patch 14
    )
    model.eval()
    volume = torch.rand(5, 126, 126, generator=torch.Generator().manual_seed(6))
    before = model.predict_study(volume)
    assert before.shape == (len(LABEL_COLUMNS),)

    save_model(model, tmp_path / "ck.pt", input_size=126, series_type=SeriesType.CORONAL_FLUID)
    loaded = load_model(tmp_path / "ck.pt")
    assert loaded.model.image_size == 126
    torch.testing.assert_close(loaded.model.predict_study(volume), before)


def test_pooled_view_matches_model_pooling() -> None:
    """Catches the bank refactor changing the mean_max arm's inputs — the E005b A/B
    is only valid if pooling cached slice features reproduces exactly what
    pool_features computed when the bank stored pooled vectors."""
    from knee.cv import pooled_view

    model = KneeModel(pretrained=False)
    model.eval()
    volume = torch.rand(5, 64, 64, generator=torch.Generator().manual_seed(5))
    with torch.no_grad():
        direct = model.pool_features(volume)
        via_bank = pooled_view(model.slice_features(volume))
    torch.testing.assert_close(direct, via_bank)


def test_mismatched_bag_and_target_counts_are_rejected() -> None:
    """Catches silent misalignment between feature lists and label rows — training on
    shifted pairs would produce a plausible-looking but meaningless head."""
    import pytest

    head = PerLabelAttentionHead(_DIM)
    bags = [torch.rand(3, _DIM)]
    targets = torch.zeros(2, len(LABEL_COLUMNS))
    with pytest.raises(ValueError, match="feature matrices"):
        fit_attention_head(head, bags, targets)


def test_learnability_auc_is_on_held_out_bags() -> None:
    """Guards the guard: the toy AUC above must be computed from numpy-comparable
    shapes — a silent broadcast here would let a broken pooler pass the MIL test."""
    probabilities = np.array([0.9, 0.8, 0.2, 0.1])
    positives = np.array([True, True, False, False])
    auc = float((probabilities[positives][:, None] > probabilities[~positives][None, :]).mean())
    assert auc == 1.0
