"""Turn a raw AFM height map into a clean, normalized grayscale image.

Ported from DNAO-Analysis-Tool/src/noahs_tools/afm_preprocessing.py. Dropped
the unused pySPM helpers (scar removal, percentile normalization, median-diff
correction) that the original preprocess_image() never actually called.
"""

from __future__ import annotations

import cv2
import numpy as np


def correct_lines(image: np.ndarray) -> np.ndarray:
    """Subtract the per-line mean, removing the classic AFM scanline offset."""
    return image - np.mean(image, axis=1, keepdims=True)


def correct_plane(image: np.ndarray) -> np.ndarray:
    """Subtract a least-squares-fitted plane, removing overall sample tilt."""
    y, x = np.mgrid[0 : image.shape[0], 0 : image.shape[1]]
    A = np.column_stack([np.ones(image.size), x.ravel(), y.ravel()])
    coeffs, *_ = np.linalg.lstsq(A, image.ravel(), rcond=None)
    plane = coeffs[0] + coeffs[1] * x + coeffs[2] * y
    return image - plane


def correct_bow(image: np.ndarray, degree: int = 2) -> np.ndarray:
    """Subtract a fitted 2D polynomial surface, removing residual scanner bow."""
    y, x = np.mgrid[0 : image.shape[0], 0 : image.shape[1]]
    terms = [np.ones(image.size)]
    for i in range(1, degree + 1):
        terms.append(x.ravel() ** i)
    for i in range(1, degree + 1):
        terms.append(y.ravel() ** i)
    A = np.column_stack(terms)
    coeffs, *_ = np.linalg.lstsq(A, image.ravel(), rcond=None)
    surface = (A @ coeffs).reshape(image.shape)
    return image - surface


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Min-max normalize a float height map to an 8-bit grayscale image."""
    return cv2.normalize(image.astype(np.float64), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def preprocess_afm(
    raw_image: np.ndarray,
    *,
    size: int | None = None,
    correct_tilt: bool = True,
    bow_degree: int | None = 2,
) -> np.ndarray:
    """Line/plane/bow-correct, normalize and (optionally) resize a raw AFM scan.

    Returns an 8-bit grayscale image ready for prompting/segmentation.
    """
    image = correct_lines(raw_image)
    if correct_tilt:
        image = correct_plane(image)
    if bow_degree:
        image = correct_bow(image, degree=bow_degree)
    image = to_uint8(image)
    if size is not None:
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)
    return image
