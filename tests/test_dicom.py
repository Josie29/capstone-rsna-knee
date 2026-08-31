from pathlib import Path

import numpy as np
import pydicom
import pytest
import torch
from pydicom.dataset import FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from knee.dicom import load_volume


def _write_slice(
    path: Path,
    pixels: np.ndarray,
    instance_number: int | None,
) -> None:
    """Write a minimal uncompressed DICOM slice for tests (synthetic — no competition data)."""
    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = MRImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    dataset = pydicom.Dataset()
    dataset.file_meta = file_meta
    dataset.Rows, dataset.Columns = pixels.shape
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    if instance_number is not None:
        dataset.InstanceNumber = instance_number
    dataset.PixelData = pixels.astype(np.uint16).tobytes()
    dataset.save_as(path, enforce_file_format=True)


def test_slices_are_sorted_by_instance_number_not_filename(tmp_path: Path) -> None:
    """Catches the bug where slices feed the model in filename order — SOP UIDs are
    unordered, so anatomy would be shuffled and slice-adjacent context destroyed."""
    # Filename order (a, b) is the reverse of instance order (2, 1).
    _write_slice(tmp_path / "a.dcm", np.full((4, 4), 300), instance_number=2)
    _write_slice(tmp_path / "b.dcm", np.zeros((4, 4)), instance_number=1)
    volume = load_volume(tmp_path, size=4)
    # Instance 1 (all zeros) must come first; instance 2 holds the bright values.
    assert volume[0].max() == 0.0
    assert volume[1].max() == 1.0


def test_intensities_scale_to_unit_range_per_volume(tmp_path: Path) -> None:
    """Catches the bug where site-dependent raw intensity scales reach the model —
    a scanner emitting 0..4000 would swamp one emitting 0..400 in the same batch."""
    _write_slice(tmp_path / "a.dcm", np.random.default_rng(0).integers(100, 4000, (8, 8)), 1)
    volume = load_volume(tmp_path, size=8)
    assert volume.dtype == torch.float32
    assert volume.min() >= 0.0
    assert volume.max() <= 1.0


def test_empty_series_dir_is_rejected(tmp_path: Path) -> None:
    """Catches a path-join bug (wrong study/series UID) surfacing as an empty stack
    crash instead of a clear error naming the directory."""
    with pytest.raises(ValueError, match="No .dcm files"):
        load_volume(tmp_path)
