"""Shared candidate-mask type used across the segmentation stages."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(eq=False)
class Candidate:
    """A single candidate DNA-origami instance mask.

    eq=False: candidates carry numpy array fields (contour, mask), so the
    dataclass-generated structural __eq__ would crash with "truth value of
    an array is ambiguous" the moment two different candidates happened to
    share e.g. the same point. Falls back to identity-based equality, which
    is what every consumer of this type (filtering, tests) actually needs.
    """

    point: tuple[int, int]
    contour: np.ndarray  # (N, 2) int32 polygon
    mask: np.ndarray  # boolean HxW array
    area: float
    sam_score: float
    classifier_score: float | None = None
    final_score: float = field(init=False, repr=False, default=0.0)

    def __post_init__(self) -> None:
        self.final_score = self.sam_score

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Axis-aligned bounding box (x1, y1, x2, y2) of the contour."""
        x, y, w, h = cv2_bounding_rect(self.contour)
        return float(x), float(y), float(x + w), float(y + h)


def cv2_bounding_rect(contour: np.ndarray) -> tuple[int, int, int, int]:
    import cv2

    return cv2.boundingRect(contour)
