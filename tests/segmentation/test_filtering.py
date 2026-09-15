import numpy as np
import pytest

from neo_toolkit.segmentation.filtering import (
    _box_iou,
    _mask_iou,
    best_per_point,
    filter_by_area,
    non_max_suppression,
)
from neo_toolkit.segmentation.types import Candidate


def _square_candidate(x0: int, y0: int, size: int, *, sam_score: float = 0.9) -> Candidate:
    contour = np.array([[x0, y0], [x0 + size, y0], [x0 + size, y0 + size], [x0, y0 + size]])
    mask = np.zeros((100, 100), dtype=bool)
    mask[y0 : y0 + size, x0 : x0 + size] = True
    return Candidate(point=(x0, y0), contour=contour, mask=mask, area=float(size * size), sam_score=sam_score)


def _corners_candidate(x0: int, y0: int, size: int, corner: int, *, sam_score: float = 0.9) -> Candidate:
    """A mask whose bbox is a size x size square but which only actually
    covers small corner x corner blocks at its top-left and bottom-right."""
    contour = np.array([[x0, y0], [x0 + size, y0], [x0 + size, y0 + size], [x0, y0 + size]])
    mask = np.zeros((100, 100), dtype=bool)
    mask[y0 : y0 + corner, x0 : x0 + corner] = True
    mask[y0 + size - corner : y0 + size, x0 + size - corner : x0 + size] = True
    return Candidate(point=(x0, y0), contour=contour, mask=mask, area=float(2 * corner * corner), sam_score=sam_score)


def test_best_per_point_picks_highest_sam_score_without_classifier():
    low = _square_candidate(0, 0, 10, sam_score=0.3)
    high = _square_candidate(0, 0, 10, sam_score=0.8)

    best = best_per_point([[low, high]])

    assert best == [high]


def test_filter_by_area_drops_outliers():
    normal_a = _square_candidate(0, 0, 10)
    normal_b = _square_candidate(20, 0, 10)
    tiny = _square_candidate(40, 0, 1)

    kept = filter_by_area([normal_a, normal_b, tiny], min_ratio=0.5, max_ratio=2.0)

    assert tiny not in kept
    assert normal_a in kept and normal_b in kept


def test_non_max_suppression_removes_overlapping_lower_score():
    strong = _square_candidate(0, 0, 20, sam_score=0.9)
    strong.final_score = 0.9
    weak_overlap = _square_candidate(2, 2, 20, sam_score=0.4)
    weak_overlap.final_score = 0.4
    far_away = _square_candidate(80, 80, 10, sam_score=0.5)
    far_away.final_score = 0.5

    kept = non_max_suppression([strong, weak_overlap, far_away], iou_threshold=0.3)

    assert strong in kept
    assert far_away in kept
    assert weak_overlap not in kept


def test_non_max_suppression_uses_mask_iou_not_box_iou():
    # Two corner-blocks masks offset so their bounding boxes overlap heavily
    # but their actual masks don't touch at all -- a box-IoU-based NMS would
    # wrongly treat these as the same object.
    a = _corners_candidate(0, 0, 20, corner=3, sam_score=0.9)
    a.final_score = 0.9
    b = _corners_candidate(4, 4, 20, corner=3, sam_score=0.8)
    b.final_score = 0.8

    assert _box_iou(a.bbox, b.bbox) > 0.4  # sanity: boxes really do overlap a lot
    assert _mask_iou(a.mask, b.mask) == 0.0

    kept = non_max_suppression([a, b], iou_threshold=0.3)

    assert a in kept and b in kept
