import cv2
import numpy as np

from neo_toolkit.segmentation.evaluation import (
    aggregate,
    load_ground_truth,
    mask_iou,
    match_instances,
)
from neo_toolkit.segmentation.types import Candidate


def _candidate_from_mask(mask: np.ndarray) -> Candidate:
    ys, xs = np.where(mask)
    contour = np.array([[xs.min(), ys.min()], [xs.max(), ys.min()], [xs.max(), ys.max()], [xs.min(), ys.max()]])
    return Candidate(point=(0, 0), contour=contour, mask=mask, area=float(mask.sum()), sam_score=0.9)


def _square_mask(x0: int, y0: int, size: int, shape=(100, 100)) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[y0 : y0 + size, x0 : x0 + size] = True
    return mask


def test_mask_iou_identical_masks_is_one():
    mask = _square_mask(0, 0, 10)
    assert mask_iou(mask, mask) == 1.0


def test_mask_iou_disjoint_masks_is_zero():
    assert mask_iou(_square_mask(0, 0, 10), _square_mask(50, 50, 10)) == 0.0


def test_match_instances_perfect_prediction():
    gt_masks = [_square_mask(0, 0, 10), _square_mask(50, 50, 10)]
    predicted = [_candidate_from_mask(m) for m in gt_masks]

    result = match_instances(predicted, gt_masks, iou_threshold=0.5)

    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0
    assert result.mean_iou == 1.0
    assert result.n_matched == 2


def test_match_instances_false_positive_hurts_precision_not_recall():
    gt_masks = [_square_mask(0, 0, 10)]
    predicted = [_candidate_from_mask(_square_mask(0, 0, 10)), _candidate_from_mask(_square_mask(80, 80, 10))]

    result = match_instances(predicted, gt_masks, iou_threshold=0.5)

    assert result.precision == 0.5  # 1 of 2 predictions matched
    assert result.recall == 1.0  # the 1 ground truth was found
    assert result.n_matched == 1


def test_match_instances_missed_detection_hurts_recall_not_precision():
    gt_masks = [_square_mask(0, 0, 10), _square_mask(50, 50, 10)]
    predicted = [_candidate_from_mask(_square_mask(0, 0, 10))]

    result = match_instances(predicted, gt_masks, iou_threshold=0.5)

    assert result.precision == 1.0
    assert result.recall == 0.5
    assert result.n_matched == 1


def test_match_instances_both_empty_is_perfect():
    result = match_instances([], [], iou_threshold=0.5)

    assert result.precision == 1.0 and result.recall == 1.0 and result.f1 == 1.0


def test_match_instances_below_threshold_not_counted_as_match():
    # overlapping but not enough: predicted square only half-covers ground truth
    gt_masks = [_square_mask(0, 0, 10)]
    predicted = [_candidate_from_mask(_square_mask(5, 0, 10))]

    result = match_instances(predicted, gt_masks, iou_threshold=0.5)

    assert result.n_matched == 0
    assert result.precision == 0.0
    assert result.recall == 0.0


def test_aggregate_pools_counts_across_images():
    r1 = match_instances([_candidate_from_mask(_square_mask(0, 0, 10))], [_square_mask(0, 0, 10)])
    r2 = match_instances([], [_square_mask(0, 0, 10)])  # a missed image

    pooled = aggregate([r1, r2])

    assert pooled.n_predicted == 1
    assert pooled.n_ground_truth == 2
    assert pooled.n_matched == 1
    assert pooled.recall == 0.5


def test_load_ground_truth_reads_real_labeled_category(tmp_path):
    images_dir = tmp_path / "Images"
    labels_dir = tmp_path / "Contours_YOLO"
    images_dir.mkdir()
    labels_dir.mkdir()
    size = 50
    cv2.imwrite(str(images_dir / "a.png"), np.zeros((size, size), dtype=np.uint8))
    polygon = [(10, 10), (30, 10), (30, 20), (10, 20)]
    line = "0 " + " ".join(f"{x / size} {y / size}" for x, y in polygon)
    (labels_dir / "a_yolo.txt").write_text(line)

    items = load_ground_truth(tmp_path)

    assert len(items) == 1
    assert len(items[0].masks) == 1
    assert items[0].masks[0].sum() > 0
