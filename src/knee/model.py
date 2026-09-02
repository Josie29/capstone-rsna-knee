import functools
from enum import StrEnum
from pathlib import Path
from typing import Any

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
    ) -> None:
        """Build the model.

        Args:
            backbone: A timm model name.
            pretrained: Load ImageNet backbone weights (True for training; False when
                the full state dict comes from a checkpoint, e.g. offline on Kaggle
                where downloads are impossible).
            head_type: How per-slice features become logits; must match the head the
                checkpoint was trained with when loading one.
        """
        super().__init__()
        self.backbone_name = backbone
        self.head_type = HeadType(head_type)
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

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        """Logits for one study's volume, in `LABEL_COLUMNS` order."""
        if self.head_type is HeadType.ATTENTION:
            return self.head(self.slice_features(volume))
        return self.head(self.pool_features(volume))

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


def save_model(
    model: KneeModel,
    path: Path,
    *,
    input_size: int,
    series_type: SeriesType,
    label_source: str = "unspecified",
    n_studies: int = 0,
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
    label_columns = tuple(payload["label_columns"])
    if label_columns != LABEL_COLUMNS:
        raise ValueError(f"Checkpoint label order {label_columns} != expected {LABEL_COLUMNS}")
    # Checkpoints from before E005b carry no head_type; they are all mean_max.
    head_type = HeadType(payload.get("head_type", HeadType.MEAN_MAX.value))
    model = KneeModel(payload["backbone"], pretrained=False, head_type=head_type)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return LoadedModel(
        model=model,
        input_size=int(payload["input_size"]),
        series_type=SeriesType(payload["series_type"]),
    )
