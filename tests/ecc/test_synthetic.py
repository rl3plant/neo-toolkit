import numpy as np
import pytest

from neo_toolkit.ecc.backgrounds import Crop
from neo_toolkit.ecc.channel import ZChannel
from neo_toolkit.ecc.code import NEO_20_10
from neo_toolkit.ecc.detector import GRID_COLS, GRID_ROWS
from neo_toolkit.ecc.synthetic import (
    SyntheticConfig,
    _augment_crop,
    _grid_centers_in_mask_bbox,
    generate_dataset,
    render_blobs,
)


def _fake_crop(size: int = 32) -> Crop:
    mask = np.zeros((size, size), dtype=np.float32)
    mask[2 : size - 2, 2 : size - 2] = 1.0
    image = np.full((size, size, 3), 0.1, dtype=np.float32)
    return Crop(image=image, mask=mask, source="fake.png")


def test_grid_centers_in_mask_bbox_returns_one_per_grid_cell():
    crop = _fake_crop()

    centers = _grid_centers_in_mask_bbox(crop.mask)

    assert len(centers) == GRID_ROWS * GRID_COLS
    for cx, cy in centers:
        assert 0 <= cx < crop.mask.shape[1]
        assert 0 <= cy < crop.mask.shape[0]


def test_render_blobs_keeps_shape_and_range():
    crop = _fake_crop()
    bits = np.zeros(20, dtype=np.uint8)
    bits[[0, 5, 19]] = 1
    cfg = SyntheticConfig(jitter_std=0.0)
    rng = np.random.default_rng(0)

    rendered = render_blobs(crop, bits, cfg, rng)

    assert rendered.shape == crop.image.shape
    assert rendered.dtype == np.float32
    assert rendered.min() >= 0.0 and rendered.max() <= 1.0
    assert rendered.max() > crop.image.max()  # a blob was actually drawn


def test_render_blobs_all_zero_bits_leaves_image_unchanged():
    crop = _fake_crop()
    bits = np.zeros(20, dtype=np.uint8)
    cfg = SyntheticConfig()
    rng = np.random.default_rng(0)

    rendered = render_blobs(crop, bits, cfg, rng)

    assert np.allclose(rendered, crop.image)


def test_generate_dataset_shapes_and_group_mask():
    bank = [_fake_crop() for _ in range(3)]
    cfg = SyntheticConfig(num_messages=5, group_j_min=2, group_j_max=4, channel=ZChannel(p=0.0), seed=0)

    dataset = generate_dataset(bank, NEO_20_10, cfg)

    assert dataset.images.shape == (5, 4, 32, 32, 3)
    assert dataset.clean_bits.shape == (5, 20)
    assert dataset.corrupted_bits.shape == (5, 4, 20)
    assert dataset.group_mask.shape == (5, 4)
    for g in range(5):
        j = int(dataset.group_sizes[g])
        assert 2 <= j <= 4
        assert dataset.group_mask[g, :j].sum() == j
        assert dataset.group_mask[g, j:].sum() == 0


def test_generate_dataset_clean_bits_match_codebook():
    bank = [_fake_crop() for _ in range(2)]
    cfg = SyntheticConfig(num_messages=4, group_j_min=1, group_j_max=1, seed=1)

    dataset = generate_dataset(bank, NEO_20_10, cfg)

    for g, message_index in enumerate(dataset.message_indices):
        assert np.array_equal(dataset.clean_bits[g], NEO_20_10.codebook[message_index])


def test_generate_dataset_zero_noise_channel_reproduces_clean_bits_exactly():
    bank = [_fake_crop() for _ in range(2)]
    cfg = SyntheticConfig(num_messages=4, group_j_min=3, group_j_max=3, channel=ZChannel(p=0.0), seed=2)

    dataset = generate_dataset(bank, NEO_20_10, cfg)

    for g in range(4):
        for obs in range(3):
            assert np.array_equal(dataset.corrupted_bits[g, obs], dataset.clean_bits[g])


def test_generate_dataset_rejects_too_many_messages():
    bank = [_fake_crop() for _ in range(2)]
    cfg = SyntheticConfig(num_messages=2000)

    with pytest.raises(ValueError):
        generate_dataset(bank, NEO_20_10, cfg)


def test_generate_dataset_rejects_empty_bank():
    with pytest.raises(ValueError):
        generate_dataset([], NEO_20_10, SyntheticConfig(num_messages=1))


def test_augment_crop_flips_stay_consistent_between_image_and_mask():
    size = 16
    image = np.zeros((size, size, 3), dtype=np.float32)
    mask = np.zeros((size, size), dtype=np.float32)
    image[0, 0] = 1.0  # a distinctive marker in the top-left corner
    mask[0, 0] = 1.0
    crop = Crop(image=image, mask=mask, source="fake.png")
    rng = np.random.default_rng(0)

    augmented = _augment_crop(crop, rng, brightness_range=(1.0, 1.0))

    # wherever the marker ended up in the image, the mask must agree
    img_pos = np.argwhere(augmented.image[:, :, 0] > 0.5)
    mask_pos = np.argwhere(augmented.mask > 0.5)
    assert len(img_pos) == 1 and len(mask_pos) == 1
    assert tuple(img_pos[0]) == tuple(mask_pos[0])


def test_augment_crop_brightness_stays_in_valid_range():
    crop = _fake_crop()
    rng = np.random.default_rng(0)

    for _ in range(20):
        augmented = _augment_crop(crop, rng, brightness_range=(0.5, 1.5))
        assert augmented.image.min() >= 0.0
        assert augmented.image.max() <= 1.0


def test_generate_dataset_with_augment_still_produces_valid_shapes():
    bank = [_fake_crop() for _ in range(3)]
    cfg = SyntheticConfig(num_messages=4, group_j_min=2, group_j_max=2, augment=True, seed=0)

    dataset = generate_dataset(bank, NEO_20_10, cfg)

    assert dataset.images.shape == (4, 2, 32, 32, 3)
    assert dataset.images.min() >= 0.0 and dataset.images.max() <= 1.0
