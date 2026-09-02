from pathlib import Path

import numpy as np
import pydicom
import torch
import torch.nn.functional as F

# Robust-range clip before scaling: MRI intensities are arbitrary units that vary by
# site/scanner/sequence, and hot pixels would otherwise compress the useful range.
_CLIP_PERCENTILES = (0.5, 99.5)


class DicomDecodeError(RuntimeError):
    """A slice failed to read or decode (missing handler, corrupt file, bad tags)."""


def _slice_sort_key(dataset: pydicom.Dataset, path: Path) -> tuple[int, float, str]:
    """Anatomical sort key for one slice, from the most trustworthy signal available.

    Tiers, best first:
      0. `ImagePositionPatient` projected onto the stack normal (the cross product of
         `ImageOrientationPatient`'s row and column direction cosines) — pure
         geometry, correct regardless of file naming or acquisition numbering.
      1. `SliceLocation` — the same scalar, precomputed by the scanner.
      2. `InstanceNumber` — acquisition order, which usually matches anatomy.
      3. Filename — arbitrary but deterministic (SOP-UID names carry no order).

    Malformed geometry tags fall through to the next tier rather than failing the
    slice: a readable image with a broken position tag is still worth feeding.
    """
    orientation = getattr(dataset, "ImageOrientationPatient", None)
    position = getattr(dataset, "ImagePositionPatient", None)
    if orientation is not None and position is not None:
        try:
            row = np.asarray(orientation[:3], dtype=np.float64)
            column = np.asarray(orientation[3:], dtype=np.float64)
            point = np.asarray(position, dtype=np.float64)
            if row.shape == (3,) and column.shape == (3,) and point.shape == (3,):
                normal = np.cross(row, column)
                # Near-zero normal = degenerate orientation (parallel axes); fall through.
                if float(np.linalg.norm(normal)) > 1e-6:
                    return (0, float(np.dot(point, normal)), "")
        except (TypeError, ValueError):
            pass  # malformed values in the geometry tags; use the next tier
    location = getattr(dataset, "SliceLocation", None)
    if location is not None:
        try:
            return (1, float(location), "")
        except (TypeError, ValueError):
            pass
    instance_number = getattr(dataset, "InstanceNumber", None)
    if instance_number is not None:
        return (2, float(instance_number), "")
    return (3, 0.0, path.name)


def _read_slice(path: Path) -> tuple[tuple[int, float, str], np.ndarray]:
    """Read one .dcm file to (sort key, float32 pixel array).

    Args:
        path: The slice file.

    Returns:
        Tuple of the anatomical sort key (see `_slice_sort_key`) and the rescaled
        pixel array.

    Raises:
        DicomDecodeError: If the file cannot be read or its pixel data cannot be
            decoded — names the file so a bad transfer syntax is diagnosable from logs.
    """
    try:
        # pydicom is not fully typed; dcmread's return covers the Dataset API we use.
        dataset = pydicom.dcmread(path)  # pyright: ignore[reportUnknownMemberType]
        pixels = dataset.pixel_array.astype(np.float32)
    except Exception as exc:  # pydicom raises a zoo of types; the file is what matters
        raise DicomDecodeError(f"Failed to decode {path}: {exc}") from exc

    slope = float(getattr(dataset, "RescaleSlope", 1.0))
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
    if slope != 1.0 or intercept != 0.0:
        pixels = pixels * slope + intercept

    return _slice_sort_key(dataset, path), pixels


def load_volume(series_dir: Path, *, size: int = 224) -> torch.Tensor:
    """Load one series directory into a normalized, resized volume.

    Slices are sorted by physical position along the stack normal (see
    `_slice_sort_key` for the fallback chain), intensity-clipped to the volume's
    0.5–99.5 percentile range, min-max scaled to [0, 1], and resized.

    Args:
        series_dir: Directory holding the series' `<SOPUID>.dcm` files.
        size: Output height/width per slice.

    Returns:
        Float32 tensor of shape (n_slices, size, size) with values in [0, 1].

    Raises:
        DicomDecodeError: If any slice fails to decode.
        ValueError: If the directory contains no .dcm files.
    """
    slice_paths = sorted(series_dir.glob("*.dcm"))
    if not slice_paths:
        raise ValueError(f"No .dcm files in {series_dir}")

    ordered = sorted((_read_slice(p) for p in slice_paths), key=lambda item: item[0])
    volume = np.stack([pixels for _, pixels in ordered])

    low, high = np.percentile(volume, _CLIP_PERCENTILES)
    volume = np.clip(volume, low, high)
    if high > low:
        volume = (volume - low) / (high - low)
    else:  # constant volume (blank series) — leave as zeros rather than divide by zero
        volume = np.zeros_like(volume)

    # torch stubs leave from_numpy partially unknown; the result is a plain Tensor.
    tensor = torch.from_numpy(volume.astype(np.float32)).unsqueeze(1)  # (n, 1, H, W)  # pyright: ignore[reportUnknownMemberType]
    resized = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
    return resized.squeeze(1)
