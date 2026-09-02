from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import write_dicom_slice

from knee.dicom import load_volume


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


def test_empty_series_dir_is_rejected(tmp_path: Path) -> None:
    """Catches a path-join bug (wrong study/series UID) surfacing as an empty stack
    crash instead of a clear error naming the directory."""
    with pytest.raises(ValueError, match="No .dcm files"):
        load_volume(tmp_path)
