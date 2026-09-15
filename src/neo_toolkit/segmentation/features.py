"""Shape-descriptor features for scoring/classifying candidate masks.

Ported from DNAO-Analysis-Tool/src/client/pipeline/features_shared.py. Dropped
the SIFT-descriptor branch: filter_masks() there only ever called the
Fourier-only extract_mask_features(), the SIFT path was dead code. Revived
the area/aspect-ratio/solidity/circularity features -- present in the
original file as _extract_area_features()/_extract_shape_features(), but
never actually called by extract_mask_features(), which only used the
Fourier descriptors.
"""

from __future__ import annotations

import cv2
import numpy as np
from pyefd import elliptic_fourier_descriptors

from neo_toolkit.segmentation.types import Candidate

DESCRIPTOR_ORDER = 10
SHAPE_FEATURE_NAMES = ("circularity", "aspect_ratio", "extent", "solidity")
FEATURE_DIM = DESCRIPTOR_ORDER * 4 + len(SHAPE_FEATURE_NAMES)


def fourier_features(candidate: Candidate, *, order: int = DESCRIPTOR_ORDER) -> np.ndarray:
    """Normalized elliptic Fourier descriptors of a candidate's contour, flattened."""
    if len(candidate.contour) < 3:
        return np.zeros(order * 4, dtype=np.float32)
    descriptors = elliptic_fourier_descriptors(candidate.contour, order=order, normalize=True)
    return np.nan_to_num(descriptors.flatten().astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)


def shape_features(candidate: Candidate) -> np.ndarray:
    """Scale-invariant shape descriptors: circularity, aspect ratio, extent, solidity."""
    contour = candidate.contour
    if len(contour) < 3:
        return np.zeros(len(SHAPE_FEATURE_NAMES), dtype=np.float32)

    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    _, _, w, h = cv2.boundingRect(contour)
    hull_area = cv2.contourArea(cv2.convexHull(contour))

    circularity = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0.0
    aspect_ratio = w / h if h > 0 else 1.0
    extent = area / (w * h) if w > 0 and h > 0 else 0.0
    solidity = area / hull_area if hull_area > 0 else 0.0

    features = np.array([circularity, aspect_ratio, extent, solidity], dtype=np.float32)
    return np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)


def mask_features(candidate: Candidate, *, order: int = DESCRIPTOR_ORDER) -> np.ndarray:
    """Combined feature vector (Fourier descriptors + shape features) for a candidate."""
    return np.concatenate([fourier_features(candidate, order=order), shape_features(candidate)])


def features_batch(candidates: list[Candidate], *, order: int = DESCRIPTOR_ORDER) -> np.ndarray:
    if not candidates:
        return np.zeros((0, order * 4 + len(SHAPE_FEATURE_NAMES)), dtype=np.float32)
    return np.stack([mask_features(c, order=order) for c in candidates])
