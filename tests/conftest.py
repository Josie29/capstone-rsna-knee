from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid


def write_dicom_slice(
    path: Path,
    pixels: np.ndarray,
    instance_number: int | None,
) -> None:
    """Write a minimal uncompressed DICOM slice for tests (synthetic — no competition data).

    Args:
        path: Destination .dcm file; parent directories are created.
        pixels: 2-D uint16-representable array.
        instance_number: `InstanceNumber` tag value, or None to omit the tag.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
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
