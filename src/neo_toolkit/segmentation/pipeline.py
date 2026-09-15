"""End-to-end AFM segmentation: raw scan file -> list of origami instance masks."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from neo_toolkit.segmentation.classifier import MaskClassifier
from neo_toolkit.segmentation.filtering import filter_candidates
from neo_toolkit.segmentation.io import read_afm_image
from neo_toolkit.segmentation.preprocessing import preprocess_afm
from neo_toolkit.segmentation.prompting import foreground_component_points
from neo_toolkit.segmentation.sam_backend import SamSegmenter
from neo_toolkit.segmentation.types import Candidate

Instance = Candidate  # public alias: the final, filtered output of segment_image()


def segment_image(
    image_uint8: np.ndarray,
    sam: SamSegmenter,
    *,
    prompt_points: list[tuple[int, int]] | None = None,
    classifier: MaskClassifier | None = None,
    classifier_weight: float = 0.9,
    area_ratio: tuple[float, float] = (0.62, 1.49),
    nms_iou: float = 0.065,
    multimask: bool = True,
) -> list[Instance]:
    """Segment a preprocessed grayscale AFM image into DNA-origami instances.

    By default, prompts come from foreground_component_points() (one per
    connected foreground blob). Pass prompt_points explicitly (e.g. from
    foreground_grid_points()) to use a different prompting strategy.
    """
    points = prompt_points if prompt_points is not None else foreground_component_points(image_uint8)
    if not points:
        return []

    image_rgb = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2RGB)
    sam.set_image(image_rgb)
    point_candidates = sam.predict_points(points, multimask=multimask)

    return filter_candidates(
        point_candidates,
        classifier=classifier,
        classifier_weight=classifier_weight,
        area_ratio=area_ratio,
        nms_iou=nms_iou,
    )


def segment_afm_file(
    path: str | Path,
    sam: SamSegmenter,
    *,
    size: int | None = None,
    correct_tilt: bool = True,
    bow_degree: int | None = 2,
    **segment_kwargs,
) -> tuple[np.ndarray, list[Instance]]:
    """Load + preprocess a raw AFM scan file, then segment it.

    Returns (preprocessed_image, instances).
    """
    raw = read_afm_image(path)
    image = preprocess_afm(raw, size=size, correct_tilt=correct_tilt, bow_degree=bow_degree)
    instances = segment_image(image, sam, **segment_kwargs)
    return image, instances
