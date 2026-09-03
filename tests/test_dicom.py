from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import write_dicom_slice

from knee.dicom import LateralityOutcome, load_volume

# Coronal frame: columns advance toward patient-left (+x), rows advance downward (-z).
_CORONAL = (1.0, 0.0, 0.0, 0.0, 0.0, -1.0)


def test_slices_are_sorted_by_physical_position_over_instance_number(tmp_path: Path) -> None:
    """Catches the teleported-slice bug: when acquisition numbering disagrees with
    geometry (renumbered exports, missing InstanceNumber on one file), number-sorted
    stacks put a mid-knee slice at the wrong depth — silently breaking any
    adjacency-based input (2.5D triplets) built on top."""
    sagittal = (0.0, 1.0, 0.0, 0.0, 0.0, -1.0)  # normal = (-1, 0, 0): key is -x
    # InstanceNumbers say a, b, c; geometry says b (x=45.2), c (x=41.7), a (x=38.2).
    write_dicom_slice(
        tmp_path / "a.dcm", np.full((4, 4), 300), instance_number=1,
        image_position=(38.2, -102.4, 88.1), image_orientation=sagittal,
    )
    write_dicom_slice(
        tmp_path / "b.dcm", np.zeros((4, 4)), instance_number=2,
        image_position=(45.2, -102.4, 88.1), image_orientation=sagittal,
    )
    write_dicom_slice(
        tmp_path / "c.dcm", np.full((4, 4), 150), instance_number=3,
        image_position=(41.7, -102.4, 88.1), image_orientation=sagittal,
    )
    volume = load_volume(tmp_path, size=4)
    assert volume[0].max() == 0.0  # x=45.2 first (key -45.2)
    assert volume[1].max() == 0.5  # x=41.7
    assert volume[2].max() == 1.0  # x=38.2 last


def test_slice_location_is_used_when_geometry_tags_are_absent(tmp_path: Path) -> None:
    """Catches the fallback chain regressing to InstanceNumber-first: SliceLocation
    is the scanner's own position scalar and must outrank acquisition numbering when
    the full geometry tags are missing."""
    # InstanceNumber order (1, 2) is the reverse of physical order by SliceLocation.
    write_dicom_slice(tmp_path / "a.dcm", np.full((4, 4), 300), instance_number=1, slice_location=12.0)
    write_dicom_slice(tmp_path / "b.dcm", np.zeros((4, 4)), instance_number=2, slice_location=8.5)
    volume = load_volume(tmp_path, size=4)
    assert volume[0].max() == 0.0
    assert volume[1].max() == 1.0


def test_slices_are_sorted_by_instance_number_not_filename(tmp_path: Path) -> None:
    """Catches the bug where slices feed the model in filename order — SOP UIDs are
    unordered, so anatomy would be shuffled and slice-adjacent context destroyed."""
    # Filename order (a, b) is the reverse of instance order (2, 1).
    write_dicom_slice(tmp_path / "a.dcm", np.full((4, 4), 300), instance_number=2)
    write_dicom_slice(tmp_path / "b.dcm", np.zeros((4, 4)), instance_number=1)
    volume = load_volume(tmp_path, size=4)
    # Instance 1 (all zeros) must come first; instance 2 holds the bright values.
    assert volume[0].max() == 0.0
    assert volume[1].max() == 1.0


def test_intensities_scale_to_unit_range_per_volume(tmp_path: Path) -> None:
    """Catches the bug where site-dependent raw intensity scales reach the model —
    a scanner emitting 0..4000 would swamp one emitting 0..400 in the same batch."""
    write_dicom_slice(tmp_path / "a.dcm", np.random.default_rng(0).integers(100, 4000, (8, 8)), 1)
    volume = load_volume(tmp_path, size=8)
    assert volume.dtype == torch.float32
    assert volume.min() >= 0.0
    assert volume.max() <= 1.0


def test_crop_mm_gives_every_study_the_same_physical_window(tmp_path: Path) -> None:
    """Catches the scale bug the crop exists to fix, inverted: the same crop_mm must
    select the same *millimeters* on scanners with different pixel spacings. If the
    crop were applied in pixels, a fine-spacing scanner would keep content a
    coarse-spacing scanner discards, and anatomy scale would still vary per study."""
    # Bright block 10+ columns right of center (big enough to survive the 0.5%
    # percentile clip); everything else zero.
    pixels = np.zeros((32, 32))
    pixels[12:20, 26:31] = 900

    coarse = tmp_path / "coarse"
    write_dicom_slice(coarse / "a.dcm", pixels, instance_number=1, pixel_spacing=(1.0, 1.0))
    fine = tmp_path / "fine"
    write_dicom_slice(fine / "a.dcm", pixels, instance_number=1, pixel_spacing=(0.5, 0.5))

    # 16mm window: 16px at 1.0mm/px (half-width 8 — pixel at +10 excluded),
    # 32px at 0.5mm/px (whole frame — pixel included).
    assert load_volume(coarse, size=16, crop_mm=16.0).max() == 0.0
    assert load_volume(fine, size=16, crop_mm=16.0).max() == 1.0


def test_crop_without_pixel_spacing_keeps_the_full_frame(tmp_path: Path) -> None:
    """Catches the fallback regressing to a pixel-unit crop or a hard failure — a
    series without PixelSpacing has no ruler, and silently cropping it by pixels
    would feed the model an arbitrary physical window."""
    pixels = np.zeros((32, 32))
    pixels[12:20, 26:31] = 900
    write_dicom_slice(tmp_path / "a.dcm", pixels, instance_number=1)  # no PixelSpacing

    volume = load_volume(tmp_path, size=16, crop_mm=16.0)
    assert volume.max() == 1.0  # edge content survives: full frame was kept


def test_crop_larger_than_frame_uses_whole_frame(tmp_path: Path) -> None:
    """Catches padding/indexing errors on frames smaller than the requested window —
    a 120mm frame cannot produce a 140mm crop, and fabricating padded anatomy or
    crashing would both be worse than using what exists."""
    pixels = np.zeros((32, 32))
    pixels[0:8, 0:8] = 900  # corner content
    write_dicom_slice(tmp_path / "a.dcm", pixels, instance_number=1, pixel_spacing=(1.0, 1.0))

    volume = load_volume(tmp_path, size=32, crop_mm=100.0)
    assert volume.shape == (1, 32, 32)
    assert volume.max() == 1.0  # corner survives: whole frame used


def _asymmetric_pixels() -> np.ndarray:
    """8x8 frame with a bright band on the patient-right (low-column) side."""
    pixels = np.zeros((8, 8))
    pixels[:, 0:3] = 900
    return pixels


def test_mirrored_knees_canonicalize_identically(tmp_path: Path) -> None:
    """The invariance the feature exists for: a left knee and its mirror-twin right
    knee must produce the same canonical volume. Without it, medial/lateral anatomy
    lands on opposite sides of the frame per knee side, corrupting the five
    side-specific findings (MCL, both menisci, both OA compartments)."""
    pixels = _asymmetric_pixels()
    right = tmp_path / "right"
    write_dicom_slice(
        right / "a.dcm", pixels, instance_number=1,
        image_position=(-53.5, 0.0, 3.5), image_orientation=_CORONAL, pixel_spacing=(1.0, 1.0),
    )
    left = tmp_path / "left"  # mirrored across the sagittal plane: flipped pixels at +x
    write_dicom_slice(
        left / "a.dcm", np.fliplr(pixels), instance_number=1,
        image_position=(46.5, 0.0, 3.5), image_orientation=_CORONAL, pixel_spacing=(1.0, 1.0),
    )

    counts: Counter[LateralityOutcome] = Counter()
    volume_right = load_volume(right, size=8, canonicalize_laterality=True, laterality_counts=counts)
    volume_left = load_volume(left, size=8, canonicalize_laterality=True, laterality_counts=counts)
    assert torch.allclose(volume_right, volume_left)
    assert counts == {LateralityOutcome.RIGHT: 1, LateralityOutcome.LEFT_MIRRORED: 1}


def test_left_knee_is_mirrored_and_right_knee_is_not(tmp_path: Path) -> None:
    """Catches the flip firing on the wrong side (or both sides): the canonical
    frame is the right knee's, so a right knee must pass through untouched while a
    left knee gets the horizontal flip."""
    pixels = _asymmetric_pixels()
    for side, origin_x in (("right", -53.5), ("left", 46.5)):
        write_dicom_slice(
            tmp_path / side / "a.dcm", pixels, instance_number=1,
            image_position=(origin_x, 0.0, 3.5), image_orientation=_CORONAL, pixel_spacing=(1.0, 1.0),
        )
    volume_right = load_volume(tmp_path / "right", size=8, canonicalize_laterality=True)
    volume_left = load_volume(tmp_path / "left", size=8, canonicalize_laterality=True)
    assert volume_right[0, 0, 0] == 1.0 and volume_right[0, 0, 7] == 0.0  # untouched
    assert volume_left[0, 0, 0] == 0.0 and volume_left[0, 0, 7] == 1.0  # mirrored


def test_sagittal_acquisition_direction_is_normalized(tmp_path: Path) -> None:
    """Catches stack-direction variance surviving canonicalization: the same knee
    scanned left-to-right vs right-to-left must yield the same slice order, or
    sagittal medial/lateral position stays scanner-dependent (the depth-axis
    equivalent of the horizontal flip)."""
    # Same right knee; orientation A's stack normal points to patient-right (-x),
    # orientation B's to patient-left (+x) — opposite acquisition directions.
    orientations = {
        "a": (0.0, 1.0, 0.0, 0.0, 0.0, -1.0),  # normal (-1, 0, 0)
        "b": (0.0, 1.0, 0.0, 0.0, 0.0, 1.0),  # normal (+1, 0, 0)
    }
    for name, orientation in orientations.items():
        for i, x in enumerate((-52.0, -50.0, -48.0)):
            write_dicom_slice(
                tmp_path / name / f"{i}.dcm", np.full((4, 4), 100.0 * (i + 1)), instance_number=i,
                image_position=(x, 0.0, 1.5), image_orientation=orientation, pixel_spacing=(1.0, 1.0),
            )
    volume_a = load_volume(tmp_path / "a", size=4, canonicalize_laterality=True)
    volume_b = load_volume(tmp_path / "b", size=4, canonicalize_laterality=True)
    assert torch.allclose(volume_a, volume_b)


def test_midline_center_skips_the_mirror(tmp_path: Path) -> None:
    """Catches over-eager mirroring of side-ambiguous volumes — notably the handful
    of bilateral series, where flipping would swap one real knee into the other."""
    pixels = _asymmetric_pixels()
    write_dicom_slice(
        tmp_path / "a.dcm", pixels, instance_number=1,
        image_position=(-3.5, 0.0, 3.5), image_orientation=_CORONAL, pixel_spacing=(1.0, 1.0),
    )
    counts: Counter[LateralityOutcome] = Counter()
    volume = load_volume(tmp_path, size=8, canonicalize_laterality=True, laterality_counts=counts)
    assert volume[0, 0, 0] == 1.0  # not mirrored
    assert counts == {LateralityOutcome.AMBIGUOUS: 1}


def test_missing_geometry_passes_through_unflipped(tmp_path: Path) -> None:
    """Catches the flag crashing or guessing on the SliceLocation-fallback path:
    without orientation tags there is no trustworthy frame, so the volume must load
    exactly as it does with canonicalization off."""
    pixels = _asymmetric_pixels()
    write_dicom_slice(tmp_path / "a.dcm", pixels, instance_number=1, slice_location=10.0)
    counts: Counter[LateralityOutcome] = Counter()
    canonical = load_volume(tmp_path, size=8, canonicalize_laterality=True, laterality_counts=counts)
    plain = load_volume(tmp_path, size=8)
    assert torch.equal(canonical, plain)
    assert counts == {LateralityOutcome.NO_GEOMETRY: 1}


def test_canonicalization_off_never_flips(tmp_path: Path) -> None:
    """Catches the default regressing: every pre-E009 caller (legacy ensemble
    checkpoints, feature banks) trained on un-mirrored volumes and must keep
    loading byte-identical inputs."""
    pixels = _asymmetric_pixels()
    write_dicom_slice(
        tmp_path / "a.dcm", pixels, instance_number=1,
        image_position=(46.5, 0.0, 3.5), image_orientation=_CORONAL, pixel_spacing=(1.0, 1.0),
    )  # a left knee, which WOULD mirror if the flag were on
    volume = load_volume(tmp_path, size=8)
    assert volume[0, 0, 0] == 1.0


def test_empty_series_dir_is_rejected(tmp_path: Path) -> None:
    """Catches a path-join bug (wrong study/series UID) surfacing as an empty stack
    crash instead of a clear error naming the directory."""
    with pytest.raises(ValueError, match="No .dcm files"):
        load_volume(tmp_path)
