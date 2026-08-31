from pathlib import Path
from typing import Any

import timm
import torch
from torch import nn

from knee.labels import LABEL_COLUMNS

# ImageNet statistics — required because the backbone starts from ImageNet weights.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_BACKBONE = "resnet34"


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

    def __init__(self, backbone: str = DEFAULT_BACKBONE, *, pretrained: bool = True) -> None:
        """Build the model.

        Args:
            backbone: A timm model name.
            pretrained: Load ImageNet backbone weights (True for training; False when
                the full state dict comes from a checkpoint, e.g. offline on Kaggle
                where downloads are impossible).
        """
        super().__init__()
        self.backbone_name = backbone
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        # nn.Module attribute access types as Tensor | Module; timm guarantees an int here.
        num_features = int(self.backbone.num_features)  # pyright: ignore[reportArgumentType]
        self.head = nn.Linear(2 * num_features, len(LABEL_COLUMNS))
        self.register_buffer("pixel_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    def freeze_backbone(self) -> None:
        """Stop gradients into the backbone so only the head trains.

        The gold-58 prototype uses this: 58 studies can support a linear head but
        fine-tuning 21M backbone parameters would memorize noise.
        """
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def pool_features(self, volume: torch.Tensor) -> torch.Tensor:
        """Per-study feature vector: backbone per slice, mean+max pooled across slices.

        Args:
            volume: (n_slices, H, W) float tensor in [0, 1], as produced by
                `knee.dicom.load_volume`.

        Returns:
            1-D tensor of length `2 * backbone.num_features`.

        Raises:
            ValueError: If `volume` is not 3-D or has zero slices.
        """
        if volume.ndim != 3 or volume.shape[0] == 0:
            raise ValueError(f"Expected non-empty (n_slices, H, W) volume, got {tuple(volume.shape)}")
        slices = volume.unsqueeze(1).repeat(1, 3, 1, 1)  # gray -> 3ch
        slices = (slices - self.pixel_mean) / self.pixel_std
        per_slice = self.backbone(slices)  # (n_slices, num_features)
        return torch.cat([per_slice.mean(dim=0), per_slice.max(dim=0).values])

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        """Logits for one study's volume, in `LABEL_COLUMNS` order."""
        return self.head(self.pool_features(volume))

    @torch.inference_mode()
    def predict_study(self, volume: torch.Tensor) -> torch.Tensor:
        """Probabilities for one study's volume, in `LABEL_COLUMNS` order."""
        return torch.sigmoid(self.forward(volume))


def save_model(model: KneeModel, path: Path, *, input_size: int) -> None:
    """Write the full model (backbone + head weights) plus reproduction metadata.

    The whole state dict is saved — not just the head — because the submission
    notebook runs offline and cannot re-download pretrained backbone weights.

    Args:
        model: The trained model.
        path: Destination .pt file.
        input_size: The slice resize target the model was trained with.
    """
    torch.save(
        {
            "state_dict": model.state_dict(),
            "backbone": model.backbone_name,
            "input_size": input_size,
            "label_columns": list(LABEL_COLUMNS),
        },
        path,
    )


def load_model(path: Path) -> tuple[KneeModel, int]:
    """Rebuild a model saved by `save_model`, without needing internet access.

    Args:
        path: The .pt checkpoint.

    Returns:
        The model in eval mode, and the input size it expects.

    Raises:
        ValueError: If the checkpoint's label order does not match `LABEL_COLUMNS` —
            a mismatch would silently scramble every submission column.
    """
    payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    label_columns = tuple(payload["label_columns"])
    if label_columns != LABEL_COLUMNS:
        raise ValueError(f"Checkpoint label order {label_columns} != expected {LABEL_COLUMNS}")
    model = KneeModel(payload["backbone"], pretrained=False)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, int(payload["input_size"])
