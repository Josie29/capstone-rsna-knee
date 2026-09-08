
import numpy as np
from pathlib import Path
import cv2
import pydicom


def _slice_order(datasets):
    """Sort by position along the slice normal. InstanceNumber is unreliable in this corpus."""
    try:
        iop = np.asarray(datasets[0].ImageOrientationPatient, dtype=np.float64)
        normal = np.cross(iop[:3], iop[3:])
        keys = [float(np.dot(np.asarray(d.ImagePositionPatient, dtype=np.float64), normal))
                for d in datasets]
    except Exception:
        keys = [float(getattr(d, "InstanceNumber", i)) for i, d in enumerate(datasets)]
    return list(np.argsort(keys))


def _to_hu(ds, arr):
    arr = arr.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1) or 1)
    inter = float(getattr(ds, "RescaleIntercept", 0) or 0)
    if slope != 1 or inter != 0:
        arr = arr * slope + inter
    if str(getattr(ds, "PhotometricInterpretation", "")) == "MONOCHROME1":
        arr = arr.max() - arr
    return arr


def load_volume(series_dir, size, depth):
    series_dir = Path(series_dir)
    files = sorted(p for p in series_dir.iterdir() if p.is_file() and not p.name.startswith("."))
    if not files:
        raise RuntimeError("empty series dir")

    if len(files) == 1:  # enhanced / multi-frame DICOM: one file, many frames
        ds = pydicom.dcmread(str(files[0]))
        vol = _to_hu(ds, ds.pixel_array)
        if vol.ndim == 2:
            vol = vol[None]
    else:
        dss = [pydicom.dcmread(str(f)) for f in files]
        dss = [d for d in dss if hasattr(d, "pixel_array") or "PixelData" in d]
        order = _slice_order(dss)
        planes = []
        for i in order:
            a = _to_hu(dss[i], dss[i].pixel_array)
            if a.ndim == 3:
                planes.extend(list(a))
            else:
                planes.append(a)
        vol = np.stack(planes)

    n = vol.shape[0]
    idx = np.linspace(0, n - 1, depth).round().astype(int) if n > 1 else np.zeros(depth, int)
    vol = vol[idx]

    out = np.empty((depth, size, size), dtype=np.uint8)
    lo, hi = np.percentile(vol, [1.0, 99.0])
    if hi <= lo:
        hi = lo + 1.0
    for i, plane in enumerate(vol):
        p = np.clip((plane - lo) / (hi - lo), 0, 1)
        out[i] = (cv2.resize(p, (size, size), interpolation=cv2.INTER_AREA) * 255).astype(np.uint8)
    return out


def decode_bucket(job):
    """Try candidate series dirs in order; first successful decode wins."""
    study, bucket, candidates, out_dir, size, depth = job
    errs = []
    for sid, sdir in candidates:
        try:
            vol = load_volume(sdir, size, depth)
            dst = Path(out_dir) / f"{sid}.npy"
            np.save(dst, vol)
            return (study, bucket, sid, 1, "")
        except Exception as e:
            errs.append(f"{sid}: {type(e).__name__}: {e}"[:120])
    return (study, bucket, "", 0, " | ".join(errs)[:300])
