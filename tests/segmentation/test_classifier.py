import numpy as np
import pytest

from neo_toolkit.segmentation.classifier import MaskClassifier


def test_untrained_classifier_returns_neutral_score():
    classifier = MaskClassifier()

    assert classifier.score(np.zeros(8)) == 0.5


def test_fit_then_score_inliers_higher_than_outliers():
    rng = np.random.default_rng(0)
    inliers = rng.normal(loc=0.0, scale=0.1, size=(50, 4))
    classifier = MaskClassifier(nu=0.05)

    classifier.fit(inliers)

    inlier_score = classifier.score(np.zeros(4))
    outlier_score = classifier.score(np.full(4, 50.0))
    assert inlier_score > outlier_score


def test_save_and_load_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    features = rng.normal(size=(30, 4))
    classifier = MaskClassifier()
    classifier.fit(features)
    path = tmp_path / "classifier.joblib"

    classifier.save(path)
    loaded = MaskClassifier.load(path)

    assert loaded.score(features[0]) == pytest.approx(classifier.score(features[0]))


def test_save_untrained_raises():
    classifier = MaskClassifier()

    with pytest.raises(RuntimeError):
        classifier.save("unused.joblib")
