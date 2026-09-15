"""Loaders for raw AFM scan formats, plus a fallback for plain raster images.

Ported from DNAO-Analysis-Tool/src/noahs_tools/afm_read_in.py. The raw-format
readers (.spm/.jpk/.ibw/.mi) need optional extra dependencies (pySPM, tifffile,
igor2), imported lazily so the rest of the package works without them.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_afm_image(path: str | Path) -> np.ndarray:
    """Load a raw height-map scan (or a plain raster image) as a 2D float array."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".spm" or (len(suffix) == 4 and suffix[1:].isdigit()):
        return _read_bruker_image(path)
    if suffix == ".ibw":
        return _read_ibw_image(path)
    if suffix == ".jpk":
        return _read_jpk_image(path)
    if suffix == ".mi":
        return _read_mi_image(path)
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unsupported or unreadable AFM file: {path}")
    return image.astype(np.float64)


def _read_bruker_image(path: Path) -> np.ndarray:
    import pySPM

    afm_file = pySPM.Bruker(str(path))
    channel_name = "Height Sensor" if path.suffix.lower() == ".spm" else "Height"
    height = afm_file.get_channel(channel_name)
    image = cv2.flip(height.pixels, 90)
    return cv2.rotate(image, cv2.ROTATE_180)


def _read_jpk_image(path: Path) -> np.ndarray:
    import tifffile

    tif = tifffile.TiffFile(str(path))
    image = tif.pages[1].asarray().astype(np.float64) * -1
    image = cv2.flip(image, 90)
    return cv2.rotate(image, cv2.ROTATE_180)


def _read_ibw_image(path: Path) -> np.ndarray:
    import igor2.binarywave as igor_binarywave

    ibw_file = igor_binarywave.load(str(path))
    image = ibw_file["wave"]["wData"][:, :, 1]
    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)


def _read_mi_image(path: Path) -> np.ndarray:
    x_pixels = y_pixels = None
    binary_index = 0
    with open(path, "rb") as f:
        lines = list(f)
    for idx, line in enumerate(lines):
        if line.startswith(b"xPixels"):
            x_pixels = int(line.split()[1])
        if line.startswith(b"yPixels"):
            y_pixels = int(line.split()[1])
        if line.strip() == b"data          BINARY_32":
            binary_index = idx
            break
    if x_pixels is None or y_pixels is None:
        raise ValueError(f"{path} contains no xPixels/yPixels header")
    if x_pixels != y_pixels:
        raise ValueError(f"{path} is not a square scan ({x_pixels}x{y_pixels})")

    binary_data = b"".join(lines[binary_index + 1 :])
    image = np.frombuffer(binary_data, dtype=np.int32) / (2**32 / 2.0)
    image = image[: x_pixels * y_pixels].reshape(x_pixels, y_pixels)
    image = cv2.flip(image, 90)
    return cv2.rotate(image, cv2.ROTATE_180)
