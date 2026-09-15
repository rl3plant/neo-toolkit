import numpy as np

from neo_toolkit.segmentation.preprocessing import correct_lines, correct_plane, preprocess_afm, to_uint8


def test_correct_lines_removes_per_row_offset():
    image = np.zeros((10, 10))
    image += np.arange(10).reshape(-1, 1)  # each row has a different constant offset

    corrected = correct_lines(image)

    assert np.allclose(corrected, 0, atol=1e-9)


def test_correct_plane_removes_linear_tilt():
    y, x = np.mgrid[0:20, 0:20]
    image = 2.0 * x + 3.0 * y + 5.0

    corrected = correct_plane(image)

    assert np.allclose(corrected, 0, atol=1e-6)


def test_to_uint8_spans_full_range():
    image = np.array([[0.0, 5.0], [10.0, -10.0]])

    result = to_uint8(image)

    assert result.dtype == np.uint8
    assert result.min() == 0
    assert result.max() == 255


def test_preprocess_afm_resizes_when_size_given():
    raw = np.random.default_rng(0).normal(size=(64, 64))

    image = preprocess_afm(raw, size=32)

    assert image.shape == (32, 32)
    assert image.dtype == np.uint8
