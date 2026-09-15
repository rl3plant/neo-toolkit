import cv2
import numpy as np

from neo_toolkit.ecc.yolo_dataset import load_category_as_crops, parse_yolo_seg_file


def _write_category(tmp_path, *, size: int = 100):
    images_dir = tmp_path / "Images"
    labels_dir = tmp_path / "Contours_YOLO"
    images_dir.mkdir()
    labels_dir.mkdir()

    image = np.random.default_rng(0).integers(0, 255, size=(size, size), dtype=np.uint8)
    cv2.imwrite(str(images_dir / "scan1.png"), image)

    # a 20x10 rectangle (class 0 = origami) and a small dot inside it (class 1 = protein)
    origami_polygon = [(10, 10), (30, 10), (30, 20), (10, 20)]
    protein_polygon = [(15, 13), (17, 13), (17, 15), (15, 15)]

    def normalize(polygon):
        return " ".join(f"{x / size} {y / size}" for x, y in polygon)

    label_lines = [f"0 {normalize(origami_polygon)}", f"1 {normalize(protein_polygon)}"]
    (labels_dir / "scan1_yolo.txt").write_text("\n".join(label_lines))
    return tmp_path


def test_parse_yolo_seg_file_converts_normalized_to_pixel_coords(tmp_path):
    category = _write_category(tmp_path)
    label_path = category / "Contours_YOLO" / "scan1_yolo.txt"

    instances = parse_yolo_seg_file(label_path, width=100, height=100)

    assert len(instances) == 2
    cls0, polygon0 = instances[0]
    assert cls0 == 0
    assert polygon0.shape == (4, 2)
    assert tuple(polygon0[0]) == (10, 10)


def test_load_category_as_crops_only_keeps_origami_class(tmp_path):
    category = _write_category(tmp_path)

    crops = load_category_as_crops(category, origami_class=0, out_size=32, min_area=1.0)

    assert len(crops) == 1  # only the class-0 polygon, not the class-1 protein dot
    assert crops[0].image.shape == (32, 32, 3)
    assert crops[0].source == f"{category.name}/scan1.png"


def test_load_category_as_crops_skips_images_without_labels(tmp_path):
    category = _write_category(tmp_path)
    cv2.imwrite(str(category / "Images" / "unlabeled.png"), np.zeros((50, 50), dtype=np.uint8))

    crops = load_category_as_crops(category, min_area=1.0)

    assert all("unlabeled" not in c.source for c in crops)
