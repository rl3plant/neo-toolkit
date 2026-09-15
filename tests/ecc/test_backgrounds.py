import numpy as np
import pytest

from neo_toolkit.ecc.backgrounds import Crop, _letterbox, _warp_instance, load_bank, save_bank, split_bank_by_source
from neo_toolkit.segmentation.types import Candidate


def _rect_instance(cx: int, cy: int, w: int, h: int) -> Candidate:
    contour = np.array([[cx - w // 2, cy - h // 2], [cx + w // 2, cy - h // 2], [cx + w // 2, cy + h // 2], [cx - w // 2, cy + h // 2]])
    mask = np.zeros((200, 200), dtype=bool)
    mask[cy - h // 2 : cy + h // 2, cx - w // 2 : cx + w // 2] = True
    return Candidate(point=(cx, cy), contour=contour, mask=mask, area=float(w * h), sam_score=0.9)


def test_warp_instance_produces_expected_output_size():
    image = np.random.default_rng(0).integers(0, 255, size=(200, 200, 3), dtype=np.uint8)
    instance = _rect_instance(100, 100, 40, 20)

    result = _warp_instance(image, instance, out_size=64, rect_expand=1.0)

    assert result is not None
    warped_image, warped_mask = result
    assert warped_image.shape == (64, 64, 3)
    assert warped_mask.shape == (64, 64)
    assert warped_image.dtype == np.float32
    assert warped_mask.max() <= 1.0 and warped_mask.min() >= 0.0


def test_warp_instance_degenerate_contour_returns_none():
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    degenerate = Candidate(point=(0, 0), contour=np.array([[0, 0], [0, 0], [0, 0]]), mask=np.zeros((50, 50), dtype=bool), area=0.0, sam_score=0.5)

    assert _warp_instance(image, degenerate, out_size=32, rect_expand=1.0) is None


def test_letterbox_centers_content_and_preserves_aspect():
    image = np.ones((10, 40, 1), dtype=np.float32)  # wide rectangle

    canvas = _letterbox(image, 20)

    assert canvas.shape == (20, 20, 1)
    # content should be centered vertically (image is wider than tall)
    assert canvas[0, 10, 0] == 0.0  # top row, empty letterbox padding
    assert canvas[10, 10, 0] == 1.0  # middle row, actual content


def _fake_crops(sources: list[str], n_per_source: int = 3) -> list[Crop]:
    crops = []
    for source in sources:
        for _ in range(n_per_source):
            crops.append(Crop(image=np.zeros((4, 4, 3), dtype=np.float32), mask=np.zeros((4, 4), dtype=np.float32), source=source))
    return crops


def test_split_bank_by_source_no_crop_leaks_across_the_split():
    crops = _fake_crops([f"scan_{i}.png" for i in range(8)])

    train, val = split_bank_by_source(crops, val_fraction=0.25, seed=0)

    train_sources = {c.source for c in train}
    val_sources = {c.source for c in val}
    assert train_sources.isdisjoint(val_sources)
    assert len(train) + len(val) == len(crops)
    assert val_sources  # at least one source held out


def test_split_bank_by_source_requires_at_least_two_sources():
    crops = _fake_crops(["only_scan.png"])

    with pytest.raises(ValueError):
        split_bank_by_source(crops)


def test_save_and_load_bank_roundtrip(tmp_path):
    crops = _fake_crops(["a.png", "b.png"], n_per_source=2)
    path = tmp_path / "bank.npz"

    save_bank(crops, path)
    loaded = load_bank(path)

    assert len(loaded) == len(crops)
    assert {c.source for c in loaded} == {c.source for c in crops}
    assert loaded[0].image.shape == crops[0].image.shape
