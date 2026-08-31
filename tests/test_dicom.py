from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import write_dicom_slice

from knee.dicom import load_volume


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
