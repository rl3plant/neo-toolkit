"""Small debugging helper to preview segmentation results."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from neo_toolkit.segmentation.types import Candidate


def draw_candidates(image_uint8: np.ndarray, candidates: list[Candidate], *, color: tuple[int, int, int] = (0, 255, 0), alpha: float = 0.4) -> np.ndarray:
    """Overlay filled contours of each candidate on top of the image."""
    base = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2BGR) if image_uint8.ndim == 2 else image_uint8.copy()
    overlay = base.copy()
    for candidate in candidates:
        cv2.fillPoly(overlay, [candidate.contour.reshape(-1, 1, 2).astype(np.int32)], color)
    return cv2.addWeighted(base, 1 - alpha, overlay, alpha, 0)


def save_preview(image_uint8: np.ndarray, candidates: list[Candidate], out_path: str | Path) -> None:
    preview = draw_candidates(image_uint8, candidates)
    cv2.imwrite(str(out_path), preview)
