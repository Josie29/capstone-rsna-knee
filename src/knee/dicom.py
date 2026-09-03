from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
import pydicom
import torch
import torch.nn.functional as F

# Robust-range clip before scaling: MRI intensities are arbitrary units that vary by
# site/scanner/sequence, and hot pixels would otherwise compress the useful range.
_CLIP_PERCENTILES = (0.5, 99.5)

# Knee side is read off the volume center's patient-x coordinate (LPS: +x = patient
# left). Centers within this band of the midline are treated as side-unknown — this
# also covers the handful of bilateral series, where mirroring would be wrong.
_X_DEAD_ZONE_MM = 20.0


class LateralityOutcome(StrEnum):
    """How laterality canonicalization resolved one volume (for run-log tallies)."""

    LEFT_MIRRORED = "left_mirrored"
    RIGHT = "right"
    AMBIGUOUS = "ambiguous"
    NO_GEOMETRY = "no_geometry"


@dataclass(frozen=True)
class _SliceGeometry:
    """Patient-space geometry of one slice (LPS coordinates, millimeters)."""

    row: np.ndarray  # direction of increasing column index (IOP first triplet)
    column: np.ndarray  # direction of increasing row index (IOP second triplet)
    center: np.ndarray  # patient-space center of the slice


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


def _slice_geometry(dataset: pydicom.Dataset, shape: tuple[int, ...]) -> _SliceGeometry | None:
    """Patient-space orientation and center for one slice, or None when unusable.

    Args:
        dataset: The slice's DICOM dataset.
        shape: The decoded pixel array's (rows, cols) shape.

    Returns:
        The slice geometry, or None when `ImageOrientationPatient`,
        `ImagePositionPatient`, or `PixelSpacing` is absent, malformed, or degenerate
        — mirroring without a trustworthy frame would be worse than not mirroring.
    """
    orientation = getattr(dataset, "ImageOrientationPatient", None)
    position = getattr(dataset, "ImagePositionPatient", None)
    spacing = _pixel_spacing(dataset)
    if orientation is None or position is None or spacing is None:
        return None
    try:
        row = np.asarray(orientation[:3], dtype=np.float64)
        column = np.asarray(orientation[3:], dtype=np.float64)
        origin = np.asarray(position, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if row.shape != (3,) or column.shape != (3,) or origin.shape != (3,):
        return None
    if float(np.linalg.norm(np.cross(row, column))) <= 1e-6:
        return None
    n_rows, n_cols = shape[0], shape[1]
    # PixelSpacing is (between-rows, between-columns); rows advance along the column
    # direction cosine and columns along the row direction cosine.
    center = origin + row * spacing[1] * (n_cols - 1) / 2 + column * spacing[0] * (n_rows - 1) / 2
    return _SliceGeometry(row=row, column=column, center=center)


def _laterality_flip(geometry: _SliceGeometry, median_x: float) -> tuple[int, bool]:
    """Decide which volume axis mirrors patient-x, and whether to flip it.

    Two normalizations compose into one flip decision on the x-dominant axis
    (0 = slice order, 1 = image rows, 2 = image columns):
      1. Direction: make the axis increase toward patient-left, so acquisition
         direction (left-to-right vs right-to-left scans) stops mattering.
      2. Side: mirror LEFT knees across the sagittal plane onto a canonical RIGHT
         frame. In-plane that is a horizontal flip (coronal/axial); when the stack
         normal is x-dominant (sagittal) it reverses slice order instead.
    Two flips on the same axis cancel, so the net decision is their XOR.

    Args:
        geometry: Any slice's geometry (orientation is uniform within a series).
        median_x: Median patient-x of the slice centers, in millimeters.

    Returns:
        (volume axis index, whether to flip it).
    """
    normal = np.cross(geometry.row, geometry.column)
    x_by_axis = (float(normal[0]), float(geometry.column[0]), float(geometry.row[0]))
    axis = int(np.argmax(np.abs(np.asarray(x_by_axis))))
    points_left = x_by_axis[axis] > 0
    knee_is_left = median_x > _X_DEAD_ZONE_MM
    return axis, (not points_left) != knee_is_left  # != is XOR on bools


def _canonicalize_laterality(
    volume: np.ndarray, geometries: list[_SliceGeometry | None]
) -> tuple[np.ndarray, LateralityOutcome]:
    """Flip the volume onto the canonical (right-knee, left-increasing) frame.

    Args:
        volume: Stacked slices, (n, H, W), already in anatomical order.
        geometries: Per-slice geometry aligned with the volume's slice order;
            None entries are slices whose geometry tags were unusable.

    Returns:
        The (possibly flipped) volume and how the side was resolved. Volumes with
        no usable geometry pass through untouched; side-ambiguous volumes still get
        the direction normalization, just not the mirror.
    """
    usable = [g for g in geometries if g is not None]
    if not usable:
        return volume, LateralityOutcome.NO_GEOMETRY
    median_x = float(np.median([g.center[0] for g in usable]))
    axis, flip = _laterality_flip(usable[0], median_x)
    if flip:
        volume = np.flip(volume, axis=axis)
    if abs(median_x) <= _X_DEAD_ZONE_MM:
        return volume, LateralityOutcome.AMBIGUOUS
    return volume, LateralityOutcome.LEFT_MIRRORED if median_x > 0 else LateralityOutcome.RIGHT


def _pixel_spacing(dataset: pydicom.Dataset) -> tuple[float, float] | None:
    """(row_mm, col_mm) from `PixelSpacing`, or None when absent or malformed."""
    raw = getattr(dataset, "PixelSpacing", None)
    if raw is None:
        return None
    try:
        row_mm, col_mm = float(raw[0]), float(raw[1])
    except (TypeError, ValueError, IndexError):
        return None
    if row_mm <= 0 or col_mm <= 0:
        return None
    return (row_mm, col_mm)


def _crop_center_mm(volume: np.ndarray, spacing: tuple[float, float] | None, crop_mm: float) -> np.ndarray:
    """Centered fixed-millimeter window across the whole stack.

    Returns the volume untouched when spacing is unknown (no ruler, no crop) and
    clamps to the frame when it is smaller than the window — a 120mm frame cannot
    produce a 140mm crop, and padding would fabricate anatomy.
    """
    if spacing is None:
        return volume
    height = min(volume.shape[1], max(1, round(crop_mm / spacing[0])))
    width = min(volume.shape[2], max(1, round(crop_mm / spacing[1])))
    top = (volume.shape[1] - height) // 2
    left = (volume.shape[2] - width) // 2
    return volume[:, top : top + height, left : left + width]


def _read_slice(
    path: Path,
) -> tuple[tuple[int, float, str], np.ndarray, tuple[float, float] | None, _SliceGeometry | None]:
    """Read one .dcm file to (sort key, float32 pixel array, pixel spacing, geometry).

    Args:
        path: The slice file.

    Returns:
        Tuple of the anatomical sort key (see `_slice_sort_key`), the rescaled
        pixel array, the (row_mm, col_mm) pixel spacing when present, and the
        patient-space geometry when derivable (see `_slice_geometry`).

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

    return (
        _slice_sort_key(dataset, path),
        pixels,
        _pixel_spacing(dataset),
        _slice_geometry(dataset, pixels.shape),
    )


def load_volume(
    series_dir: Path,
    *,
    size: int = 224,
    crop_mm: float | None = None,
    canonicalize_laterality: bool = False,
    laterality_counts: Counter[LateralityOutcome] | None = None,
) -> torch.Tensor:
    """Load one series directory into a normalized, resized volume.

    Slices are sorted by physical position along the stack normal (see
    `_slice_sort_key` for the fallback chain), optionally cropped to a centered
    fixed-millimeter window, intensity-clipped to the (cropped) volume's
    0.5–99.5 percentile range, min-max scaled to [0, 1], and resized.

    Args:
        series_dir: Directory holding the series' `<SOPUID>.dcm` files.
        size: Output height/width per slice.
        crop_mm: Physical window edge in millimeters, converted per study via
            `PixelSpacing` so every study lands at the same mm-per-pixel scale.
            None (default) keeps the scanner's full frame. Series without a usable
            `PixelSpacing` fall back to the full frame; frames smaller than the
            window are used whole.
        canonicalize_laterality: When True, mirror the volume onto a canonical
            right-knee, patient-left-increasing frame (see `_canonicalize_laterality`)
            so medial/lateral anatomy lands on the same side of the frame for every
            study. Default False leaves output byte-identical to prior behavior.
        laterality_counts: Optional tally updated with this volume's
            `LateralityOutcome` — lets batch callers report flip/ambiguity rates.

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
    volume = np.stack([pixels for _, pixels, _, _ in ordered])
    if canonicalize_laterality:
        volume, outcome = _canonicalize_laterality(volume, [geometry for *_, geometry in ordered])
        if laterality_counts is not None:
            laterality_counts[outcome] += 1
    if crop_mm is not None:
        # Spacing is uniform within a series in practice; the first slice speaks for all.
        volume = _crop_center_mm(volume, ordered[0][2], crop_mm)

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
