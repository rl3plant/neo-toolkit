import numpy as np

from neo_toolkit.segmentation.prompting import foreground_component_points, foreground_grid_points


def test_foreground_grid_points_only_keeps_bright_region():
    image = np.zeros((100, 100), dtype=np.uint8)
    image[40:60, 40:60] = 255  # a single bright square in the middle

    points = foreground_grid_points(image, grid=(10, 10))

    assert points  # found at least one prompt
    for x, y in points:
        assert 30 <= x <= 70
        assert 30 <= y <= 70


def test_foreground_grid_points_empty_image_returns_no_points():
    image = np.zeros((50, 50), dtype=np.uint8)

    points = foreground_grid_points(image, grid=(5, 5))

    assert points == []


def test_foreground_component_points_one_point_per_separated_square():
    image = np.zeros((200, 200), dtype=np.uint8)
    squares = [(20, 20), (100, 30), (50, 120), (150, 150)]
    size = 15
    for x, y in squares:
        image[y : y + size, x : x + size] = 255

    points = foreground_component_points(image, min_component_area=5)

    assert len(points) == len(squares)
    for (sx, sy), (px, py) in zip(sorted(squares), sorted(points)):
        assert sx <= px <= sx + size
        assert sy <= py <= sy + size


def test_foreground_component_points_samples_a_grid_over_merged_components():
    image = np.zeros((200, 200), dtype=np.uint8)
    # several normal, isolated instances set the "typical" component size...
    image[50:65, 50:65] = 255
    image[80:95, 20:35] = 255
    image[20:35, 80:95] = 255
    # ...against which this one big blob (~3 instances merged side by side) stands out
    image[100:100 + 15, 100:100 + 45] = 255

    points = foreground_component_points(image, min_component_area=5, merge_area_ratio=1.8, dense_grid=(3, 1))

    isolated_points = [(x, y) for x, y in points if 50 <= x < 65 and 50 <= y < 65]
    merged_points = [(x, y) for x, y in points if 100 <= x < 145 and 100 <= y < 115]
    assert len(isolated_points) == 1
    assert len(merged_points) > 1


def test_foreground_component_points_ignores_tiny_noise_specks():
    image = np.zeros((100, 100), dtype=np.uint8)
    image[10, 10] = 255  # single-pixel speck

    points = foreground_component_points(image, min_component_area=5)

    assert points == []


def test_foreground_component_points_empty_image_returns_no_points():
    image = np.zeros((50, 50), dtype=np.uint8)

    assert foreground_component_points(image) == []


def test_foreground_component_points_auto_cutoff_ignores_noise_without_explicit_threshold():
    image = np.zeros((300, 300), dtype=np.uint8)
    rng = np.random.default_rng(0)
    # many 1-2px noise specks scattered around, well below any real instance
    for _ in range(40):
        x, y = rng.integers(0, 300, size=2)
        image[y, x] = 255
    # a handful of clearly-real, much larger square instances
    real_squares = [(30, 30), (150, 60), (80, 200), (220, 220)]
    for x, y in real_squares:
        image[y : y + 20, x : x + 20] = 255

    points = foreground_component_points(image)  # min_component_area left as None

    assert len(points) == len(real_squares)
