import numpy as np
import pytest

from neo_toolkit.segmentation.features import FEATURE_DIM, mask_features, shape_features
from neo_toolkit.segmentation.types import Candidate


def _square_candidate(size: int = 10) -> Candidate:
    contour = np.array([[0, 0], [size, 0], [size, size], [0, size]])
    return Candidate(point=(0, 0), contour=contour, mask=np.zeros((size + 2, size + 2), dtype=bool), area=float(size * size), sam_score=0.9)


def test_mask_features_has_fixed_length_for_valid_contour():
    candidate = _square_candidate()

    features = mask_features(candidate)

    assert features.shape == (FEATURE_DIM,)
    assert np.isfinite(features).all()


def test_mask_features_degenerate_contour_returns_zeros_of_same_length():
    contour = np.array([[0, 0], [1, 1]])  # fewer than 3 points
    candidate = Candidate(point=(0, 0), contour=contour, mask=np.zeros((5, 5), dtype=bool), area=1.0, sam_score=0.5)

    features = mask_features(candidate)

    assert features.shape == (FEATURE_DIM,)
    assert np.all(features == 0)


def test_shape_features_of_a_square_are_high_extent_and_solidity():
    candidate = _square_candidate()

    circularity, aspect_ratio, extent, solidity = shape_features(candidate)

    assert aspect_ratio == pytest.approx(1.0, abs=0.1)
    assert extent > 0.8  # a square nearly fills its own bounding box
    assert solidity == 1.0  # a square is already convex


def test_shape_features_aspect_ratio_reflects_a_wide_rectangle():
    contour = np.array([[0, 0], [20, 0], [20, 5], [0, 5]])
    candidate = Candidate(point=(0, 0), contour=contour, mask=np.zeros((10, 25), dtype=bool), area=100.0, sam_score=0.9)

    _, aspect_ratio, _, _ = shape_features(candidate)

    assert aspect_ratio > 3.0  # clearly wide, exact value depends on cv2's pixel-inclusive bbox convention
