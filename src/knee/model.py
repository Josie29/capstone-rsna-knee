import functools
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import timm
import torch
from pydantic import BaseModel, ConfigDict
from torch import nn

from knee.labels import LABEL_COLUMNS
from knee.series import SeriesType


class HeadType(StrEnum):
    """How per-slice backbone features become 12 study-level logits."""

    MEAN_MAX = "mean_max"  # E001-E004: mean+max pool across slices, one linear layer
    ATTENTION = "attention"  # E005b: per-label gated-attention MIL pooling


class InputMode(StrEnum):
    """What the backbone sees per study."""

    SLICES = "slices"  # E001-E005: every gray slice, repeated to 3 channels
    TRIPLETS = "triplets"  # E006: K anchor images, channels = adjacent slices (2.5D)


def sample_triplets(
    volume: torch.Tensor,
    *,
    n_anchors: int = 3,
    window: tuple[float, float] = (0.2, 0.8),
    jitter: float = 0.0,
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    """K anchor images whose channels are physically adjacent slices (2.5D input).

    Anchors are spread evenly across the central window of the (physically sorted)
    stack; each anchor's neighbors [i-1, i, i+1] become the R/G/B channels of one
    image, so a real finding — which persists across neighbors — is visible to the
    backbone in a single forward. Edge anchors clamp, so a 1-2 slice stack degrades
    to repeated slices (the old gray->3ch behavior) rather than failing.

    Args:
        volume: (n_slices, H, W) float tensor in [0, 1], physically sorted.
        n_anchors: Number of anchor images (K).
        window: Fractional range of the stack anchors are drawn from — findings
            live near the joint, edge slices are mostly muscle.
        jitter: Fractional anchor jitter; free augmentation during training.
        rng: Jitter source; None (inference) means deterministic anchors.

    Returns:
        (n_anchors, 3, H, W) tensor.

    Raises:
        ValueError: If `volume` is not 3-D or has zero slices.
    """
    if volume.ndim != 3 or volume.shape[0] == 0:
        raise ValueError(f"Expected non-empty (n_slices, H, W) volume, got {tuple(volume.shape)}")
    n_slices = volume.shape[0]
    positions = np.linspace(window[0], window[1], n_anchors) * (n_slices - 1)
    if rng is not None and jitter > 0:
        positions = positions + rng.uniform(-jitter, jitter, n_anchors) * n_slices
    anchors = np.clip(np.round(positions), 0, n_slices - 1).astype(int)
    return torch.stack(
        [
            torch.stack(
                [
                    volume[max(anchor - 1, 0)],
                    volume[anchor],
                    volume[min(anchor + 1, n_slices - 1)],
                ]
            )
            for anchor in anchors
        ]
    )

# ImageNet statistics — required because the backbone starts from ImageNet weights.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_BACKBONE = "resnet34"

# E004 production backbone: self-supervised DINOv2 features, built for the frozen
# linear-probe regime we run. Fixed 518px input (patch 14 x 37) — pair with
# input_size=518. Not the code default: tests and fixtures use small inputs that a
# fixed-size ViT rejects, so notebooks opt in via config.
DINOV2_BACKBONE = "vit_small_patch14_dinov2.lvd142m"

# Upper bound on slices per backbone forward inside pool_features. Slices are
# independent and the model is in eval mode, so chunking never changes the result;
# it only bounds peak memory (ViT attention at 518px on a long series would OOM a T4).
_SLICE_BATCH = 8


@functools.cache
def _cuda_works() -> bool:
    """Probe CUDA once per process; hardware does not change mid-run.

    `torch.cuda.is_available()` is not enough on Kaggle: the default-assigned P100
    predates the image's torch build (no sm_60 kernels), so the first CUDA op raises
    `cudaErrorNoKernelImageForDevice`. The `+ 1` forces an arithmetic kernel launch
    and `.item()` forces a sync — CUDA errors are asynchronous, so without the sync
    the failure would surface at some later unrelated call outside this try block.
    """
    if not torch.cuda.is_available():
        return False
    try:
        (torch.zeros(2, device="cuda") + 1).sum().item()
        return True
    except RuntimeError as exc:  # torch.AcceleratorError subclasses RuntimeError
        print(f"CUDA present but unusable ({exc}); falling back to CPU")
        return False


def resolve_device() -> str:
    """The device every model call should use: "cuda" only if it actually works."""
    return "cuda" if _cuda_works() else "cpu"


class PerLabelAttentionHead(nn.Module):
    """Gated-attention MIL head: per-label slice weighting, then per-label logits.

    Each of the 12 findings learns its own attention over slices (a tear lives at
    the joint line, an effusion above the kneecap), so few-slice findings stop being
    diluted by a whole-stack mean. The gate trunk (V, U) is shared across labels;
    only the scoring and classifier vectors are per-label, keeping parameters tiny
    relative to the backbone. Gated attention per Ilse et al. 2018.
    """

    def __init__(self, feature_dim: int, *, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        """Build the head.

        Args:
            feature_dim: Per-slice backbone feature size.
            hidden_dim: Gate trunk width.
            dropout: Dropout on slice features during training — the guard against
                a learnable pooler fitting miner label noise.
        """
        super().__init__()
        self.gate_value = nn.Linear(feature_dim, hidden_dim)
        self.gate_sigmoid = nn.Linear(feature_dim, hidden_dim)
        self.label_scorers = nn.Parameter(torch.empty(len(LABEL_COLUMNS), hidden_dim))
        self.label_classifiers = nn.Parameter(torch.empty(len(LABEL_COLUMNS), feature_dim))
        self.bias = nn.Parameter(torch.zeros(len(LABEL_COLUMNS)))
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.label_scorers)  # pyright: ignore[reportUnknownMemberType]
        nn.init.xavier_uniform_(self.label_classifiers)  # pyright: ignore[reportUnknownMemberType]

    def attention_weights(self, slices: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Per-label attention over slices; the "where it looked" signal.

        Args:
            slices: (n_slices, feature_dim) or padded (batch, n_slices, feature_dim).
            mask: Optional (batch, n_slices) bool, True for real slices. Padded
                positions get zero weight.

        Returns:
            (12, n_slices) or (batch, 12, n_slices) weights, summing to 1 over slices.
        """
        unbatched = slices.ndim == 2
        if unbatched:
            slices = slices.unsqueeze(0)
        gated = torch.tanh(self.gate_value(slices)) * torch.sigmoid(self.gate_sigmoid(slices))
        scores = (gated @ self.label_scorers.T).transpose(1, 2)  # (batch, 12, n_slices)
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        return weights.squeeze(0) if unbatched else weights

    def forward(self, slices: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Logits in `LABEL_COLUMNS` order from per-slice features.

        Args:
            slices: (n_slices, feature_dim) or padded (batch, n_slices, feature_dim).
            mask: Optional (batch, n_slices) bool, True for real slices.

        Returns:
            (12,) or (batch, 12) logits.
        """
        unbatched = slices.ndim == 2
        if unbatched:
            slices = slices.unsqueeze(0)
        slices = self.dropout(slices)
        weights = self.attention_weights(slices, mask)  # (batch, 12, n_slices)
        pooled = weights @ slices  # (batch, 12, feature_dim): per-label weighted sums
        logits = (pooled * self.label_classifiers).sum(dim=-1) + self.bias
        return logits.squeeze(0) if unbatched else logits


class KneeModel(nn.Module):
    """One CNN from volume to 12 findings: per-slice backbone, pooling, linear head.

    Deliberately the simplest end-to-end shape: each slice runs through the backbone,
    per-slice features are mean- and max-pooled across slices (so any slice count
    yields a fixed-size vector), and a single linear layer maps that to 12 logits.
    Smarter pooling/fusion layers slot in here later without changing callers.
    """

    # Registered buffers; declared so attribute access types as Tensor, not Tensor | Module.
    pixel_mean: torch.Tensor
    pixel_std: torch.Tensor

    def __init__(
        self,
        backbone: str = DEFAULT_BACKBONE,
        *,
        pretrained: bool = True,
        head_type: HeadType = HeadType.MEAN_MAX,
        input_mode: InputMode = InputMode.SLICES,
        n_anchors: int = 3,
        image_size: int | None = None,
    ) -> None:
        """Build the model.

        Args:
            backbone: A timm model name.
            pretrained: Load ImageNet backbone weights (True for training; False when
                the full state dict comes from a checkpoint, e.g. offline on Kaggle
                where downloads are impossible).
            head_type: How per-slice features become logits; must match the head the
                checkpoint was trained with when loading one.
            input_mode: What the backbone sees per study — every gray slice, or
                `n_anchors` adjacent-slice triplet images (2.5D).
            n_anchors: Triplet count when `input_mode` is TRIPLETS; ignored otherwise.
            image_size: Override the backbone's native input size — required to run
                fixed-size ViTs (DINOv2's 518) at an affordable resolution; timm
                interpolates the position embeddings. None keeps the backbone's
                default and must stay None for CNNs, which reject the kwarg.
        """
        super().__init__()
        self.backbone_name = backbone
        self.head_type = HeadType(head_type)
        self.input_mode = InputMode(input_mode)
        self.n_anchors = n_anchors
        self.image_size = image_size
        if image_size is not None:
            self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, img_size=image_size)
        else:
            self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        # nn.Module attribute access types as Tensor | Module; timm guarantees an int here.
        num_features = int(self.backbone.num_features)  # pyright: ignore[reportArgumentType]
        self.head: nn.Module = (
            PerLabelAttentionHead(num_features)
            if self.head_type is HeadType.ATTENTION
            else nn.Linear(2 * num_features, len(LABEL_COLUMNS))
        )
        self.register_buffer("pixel_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    def freeze_backbone(self) -> None:
        """Stop gradients into the backbone so only the head trains.

        The gold-58 prototype uses this: 58 studies can support a linear head but
        fine-tuning 21M backbone parameters would memorize noise.
        """
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def slice_features(self, volume: torch.Tensor) -> torch.Tensor:
        """Per-slice backbone features for one study's volume.

        Args:
            volume: (n_slices, H, W) float tensor in [0, 1], as produced by
                `knee.dicom.load_volume`.

        Returns:
            (n_slices, backbone.num_features) tensor, one row per slice in order.

        Raises:
            ValueError: If `volume` is not 3-D or has zero slices.
        """
        if volume.ndim != 3 or volume.shape[0] == 0:
            raise ValueError(f"Expected non-empty (n_slices, H, W) volume, got {tuple(volume.shape)}")
        slices = volume.unsqueeze(1).repeat(1, 3, 1, 1)  # gray -> 3ch
        slices = (slices - self.pixel_mean) / self.pixel_std
        # torch stubs leave Tensor.split partially unknown; it yields plain Tensors.
        return torch.cat(
            [self.backbone(chunk) for chunk in slices.split(_SLICE_BATCH)]  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        )

    def pool_features(self, volume: torch.Tensor) -> torch.Tensor:
        """Per-study feature vector: mean+max of `slice_features` across slices.

        Args:
            volume: (n_slices, H, W) float tensor in [0, 1].

        Returns:
            1-D tensor of length `2 * backbone.num_features`.

        Raises:
            ValueError: If `volume` is not 3-D or has zero slices.
        """
        per_slice = self.slice_features(volume)
        return torch.cat([per_slice.mean(dim=0), per_slice.max(dim=0).values])

    def triplet_features(self, images: torch.Tensor) -> torch.Tensor:
        """Backbone features for already-3-channel images (the 2.5D triplet path).

        Args:
            images: (n_images, 3, H, W) float tensor in [0, 1] — e.g. from
                `sample_triplets`. Unlike `slice_features`, channels carry three
                *different* adjacent slices, so no gray->3ch repeat happens here.

        Returns:
            (n_images, backbone.num_features) tensor.
        """
        return self.backbone((images - self.pixel_mean) / self.pixel_std)

    def _head_logits(self, per_item_features: torch.Tensor) -> torch.Tensor:
        """Route per-item (slice or triplet) features through the configured head."""
        if self.head_type is HeadType.ATTENTION:
            return self.head(per_item_features)
        pooled = torch.cat([per_item_features.mean(dim=0), per_item_features.max(dim=0).values])
        return self.head(pooled)

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        """Logits for one study's volume, in `LABEL_COLUMNS` order."""
        if self.input_mode is InputMode.TRIPLETS:
            images = sample_triplets(volume, n_anchors=self.n_anchors)  # deterministic anchors
            return self._head_logits(self.triplet_features(images))
        return self._head_logits(self.slice_features(volume))

    @torch.inference_mode()
    def predict_study(self, volume: torch.Tensor) -> torch.Tensor:
        """Probabilities for one study's volume, in `LABEL_COLUMNS` order.

        Runs the backbone under fp16 autocast on CUDA (~2x on T4 tensor cores);
        a no-op on CPU. Logits are cast back to fp32 before the sigmoid so callers
        always see full-precision probabilities.
        """
        on_cuda = self.pixel_mean.device.type == "cuda"
        with torch.autocast("cuda", enabled=on_cuda):
            logits = self.forward(volume)
        return torch.sigmoid(logits.float())


class LoadedModel(BaseModel):
    """A checkpointed model plus the metadata inference needs to feed it correctly."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: KneeModel
    input_size: int
    # The one series type this model consumes — specialist models must never be fed
    # another type (a fluid-trained model reads non-fluid contrast backwards).
    series_type: SeriesType
    # Fixed-mm crop the model was trained with (None = full frame); inference must
    # reproduce it or the model sees anatomy at the wrong physical scale.
    crop_mm: float | None


def save_model(
    model: KneeModel,
    path: Path,
    *,
    input_size: int,
    series_type: SeriesType,
    label_source: str = "unspecified",
    n_studies: int = 0,
    crop_mm: float | None = None,
) -> None:
    """Write the full model (backbone + head weights) plus reproduction metadata.

    The whole state dict is saved — not just the head — because the submission
    notebook runs offline and cannot re-download pretrained backbone weights.

    Args:
        model: The trained model.
        path: Destination .pt file.
        input_size: The slice resize target the model was trained with.
        series_type: The series type the model was trained on; stored so inference
            routes the right series to it without filename conventions.
        label_source: Which label set trained this checkpoint (e.g. "gold58",
            "blended_v1") — without it, checkpoints from different label regimes are
            byte-for-byte indistinguishable.
        n_studies: Number of studies the head actually trained on.
        crop_mm: Fixed-mm crop the training volumes used (None = full frame);
            stored so inference reproduces the same physical scale.
    """
    torch.save(
        {
            "state_dict": model.state_dict(),
            "backbone": model.backbone_name,
            "input_size": input_size,
            "series_type": series_type.value,
            "label_columns": list(LABEL_COLUMNS),
            "label_source": label_source,
            "n_studies": n_studies,
            "head_type": model.head_type.value,
            "crop_mm": crop_mm,
            "input_mode": model.input_mode.value,
            "n_anchors": model.n_anchors,
            "image_size": model.image_size,
        },
        path,
    )


def load_model(path: Path) -> LoadedModel:
    """Rebuild a model saved by `save_model`, without needing internet access.

    Args:
        path: The .pt checkpoint.

    Returns:
        The model in eval mode with its input size and native series type.

    Raises:
        ValueError: If the checkpoint's label order does not match `LABEL_COLUMNS` —
            a mismatch would silently scramble every submission column.
    """
    payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("model_kind") == "multiplane":
        raise ValueError(f"{path.name} is a multiplane checkpoint; use load_multiplane_model")
    label_columns = tuple(payload["label_columns"])
    if label_columns != LABEL_COLUMNS:
        raise ValueError(f"Checkpoint label order {label_columns} != expected {LABEL_COLUMNS}")
    # Checkpoints from before E005b/E006 carry no head_type/input_mode; they are
    # all mean_max over gray slices.
    head_type = HeadType(payload.get("head_type", HeadType.MEAN_MAX.value))
    input_mode = InputMode(payload.get("input_mode", InputMode.SLICES.value))
    image_size = payload.get("image_size")  # pre-E007 checkpoints used native sizes
    model = KneeModel(
        payload["backbone"],
        pretrained=False,
        head_type=head_type,
        input_mode=input_mode,
        n_anchors=int(payload.get("n_anchors", 3)),
        image_size=int(image_size) if image_size is not None else None,
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    crop_mm = payload.get("crop_mm")  # pre-E005a checkpoints trained on full frames
    return LoadedModel(
        model=model,
        input_size=int(payload["input_size"]),
        series_type=SeriesType(payload["series_type"]),
        crop_mm=float(crop_mm) if crop_mm is not None else None,
    )


class MultiPlaneModel(nn.Module):
    """One model for the whole study: a bag of triplets from every plane, per-label
    attention over the bag.

    Replaces the three per-plane specialists plus the hand-coded plane-prior
    combiner (E002: a null): per-label attention over the mixed bag IS the learned
    per-label plane weighting, and a missing plane simply contributes fewer bag
    items — the head's masked softmax renormalizes, which is what
    `combiner_weights` did by hand. A learned per-plane embedding added to each
    item's features carries the "which camera" signal the per-plane structure used
    to provide.
    """

    pixel_mean: torch.Tensor
    pixel_std: torch.Tensor

    def __init__(
        self,
        backbone: str = DEFAULT_BACKBONE,
        series_types: Sequence[SeriesType] = (
            SeriesType.SAGITTAL_FLUID,
            SeriesType.CORONAL_FLUID,
            SeriesType.AXIAL_FLUID,
        ),
        *,
        pretrained: bool = True,
        n_anchors: int = 3,
        image_size: int | None = None,
    ) -> None:
        """Build the model.

        Args:
            backbone: A timm model name.
            series_types: The planes this model consumes; order defines the plane
                embedding indices and is persisted in checkpoints.
            pretrained: Load pretrained backbone weights (False when the state dict
                comes from a checkpoint, e.g. offline on Kaggle).
            n_anchors: Triplet count per plane (see `sample_triplets`).
            image_size: Backbone input-size override (fixed-size ViTs at affordable
                resolutions); None keeps the backbone's native size.

        Raises:
            ValueError: If `series_types` is empty or has duplicates.
        """
        super().__init__()
        if not series_types or len(set(series_types)) != len(series_types):
            raise ValueError(f"series_types must be non-empty and unique, got {list(series_types)}")
        self.backbone_name = backbone
        self.series_types = list(series_types)
        self.n_anchors = n_anchors
        self.image_size = image_size
        if image_size is not None:
            self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, img_size=image_size)
        else:
            self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        num_features = int(self.backbone.num_features)  # pyright: ignore[reportArgumentType]
        self.plane_embeddings = nn.Embedding(len(self.series_types), num_features)
        # Small init: items start nearly plane-agnostic and the signal is learned,
        # which also keeps warm-started backbones/heads near their loaded behavior.
        nn.init.normal_(self.plane_embeddings.weight, std=0.02)  # pyright: ignore[reportUnknownMemberType]
        self.head = PerLabelAttentionHead(num_features)
        self.register_buffer("pixel_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    def freeze_backbone(self) -> None:
        """Stop gradients into the backbone (stage A of the staged unfreeze)."""
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def bag_features(self, images: torch.Tensor, plane_indices: torch.Tensor) -> torch.Tensor:
        """Per-item features for a bag: backbone + the item's plane embedding.

        Args:
            images: (n_items, 3, H, W) float triplet images in [0, 1].
            plane_indices: (n_items,) long indices into `series_types`.

        Returns:
            (n_items, num_features) tensor.
        """
        features = self.backbone((images - self.pixel_mean) / self.pixel_std)
        return features + self.plane_embeddings(plane_indices)

    def forward(self, images: torch.Tensor, plane_indices: torch.Tensor) -> torch.Tensor:
        """Logits for one study's bag, in `LABEL_COLUMNS` order."""
        return self.head(self.bag_features(images, plane_indices))

    @torch.inference_mode()
    def predict_study(self, volumes: dict[SeriesType, torch.Tensor]) -> torch.Tensor:
        """Probabilities for one study from whichever planes it has.

        Args:
            volumes: Physically-sorted (n_slices, H, W) volumes keyed by plane;
                absent planes are simply omitted — the bag shrinks.

        Returns:
            (12,) probabilities in `LABEL_COLUMNS` order.

        Raises:
            ValueError: If `volumes` is empty (the caller decides the fallback) or
                contains a plane this model was not built for.
        """
        if not volumes:
            raise ValueError("predict_study needs at least one plane's volume")
        images: list[torch.Tensor] = []
        plane_indices: list[int] = []
        for series_type, volume in volumes.items():
            if series_type not in self.series_types:
                raise ValueError(f"Model has no plane embedding for {series_type}")
            triplets = sample_triplets(volume, n_anchors=self.n_anchors)  # deterministic anchors
            images.append(triplets)
            plane_indices.extend([self.series_types.index(series_type)] * triplets.shape[0])
        bag = torch.cat(images).to(self.pixel_mean.device)
        indices = torch.tensor(plane_indices, dtype=torch.long, device=self.pixel_mean.device)
        on_cuda = self.pixel_mean.device.type == "cuda"
        with torch.autocast("cuda", enabled=on_cuda):
            logits = self.forward(bag, indices)
        return torch.sigmoid(logits.float())


def save_multiplane_model(
    model: MultiPlaneModel,
    path: Path,
    *,
    input_size: int,
    label_source: str = "unspecified",
    n_studies: int = 0,
    crop_mm: float | None = None,
) -> None:
    """Write a unified multi-plane checkpoint plus reproduction metadata.

    Args:
        model: The trained model.
        path: Destination .pt file.
        input_size: Slice resize target the model was trained with.
        label_source: Which label set trained this checkpoint.
        n_studies: Number of studies trained on.
        crop_mm: Fixed-mm crop the training volumes used (None = full frame).
    """
    torch.save(
        {
            "model_kind": "multiplane",
            "state_dict": model.state_dict(),
            "backbone": model.backbone_name,
            "series_types": [t.value for t in model.series_types],
            "input_size": input_size,
            "label_columns": list(LABEL_COLUMNS),
            "label_source": label_source,
            "n_studies": n_studies,
            "crop_mm": crop_mm,
            "n_anchors": model.n_anchors,
            "image_size": model.image_size,
        },
        path,
    )


class LoadedMultiPlaneModel(BaseModel):
    """A unified checkpoint plus the metadata inference needs to feed it correctly."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: MultiPlaneModel
    input_size: int
    series_types: list[SeriesType]
    crop_mm: float | None


def load_multiplane_model(path: Path) -> LoadedMultiPlaneModel:
    """Rebuild a model saved by `save_multiplane_model`, without internet access.

    Args:
        path: The .pt checkpoint.

    Returns:
        The model in eval mode with its input size, planes, and crop.

    Raises:
        ValueError: If the checkpoint is not a multiplane checkpoint or its label
            order does not match `LABEL_COLUMNS`.
    """
    payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("model_kind") != "multiplane":
        raise ValueError(f"{path.name} is not a multiplane checkpoint; use load_model")
    label_columns = tuple(payload["label_columns"])
    if label_columns != LABEL_COLUMNS:
        raise ValueError(f"Checkpoint label order {label_columns} != expected {LABEL_COLUMNS}")
    series_types = [SeriesType(value) for value in payload["series_types"]]
    image_size = payload.get("image_size")
    model = MultiPlaneModel(
        payload["backbone"],
        series_types,
        pretrained=False,
        n_anchors=int(payload.get("n_anchors", 3)),
        image_size=int(image_size) if image_size is not None else None,
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    crop_mm = payload.get("crop_mm")
    return LoadedMultiPlaneModel(
        model=model,
        input_size=int(payload["input_size"]),
        series_types=series_types,
        crop_mm=float(crop_mm) if crop_mm is not None else None,
    )
