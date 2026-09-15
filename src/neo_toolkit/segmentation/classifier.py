"""One-class SVM that scores whether a candidate mask looks like a real DNA origami.

Ported from DNAO-Analysis-Tool's SVMClassifier; dropped the unused
BaseClassifier ABC since this is the only classifier implementation.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


class MaskClassifier:
    """Trained only on known-good masks; scores new masks as inlier/outlier."""

    def __init__(self, *, nu: float = 0.05):
        self.model = OneClassSVM(kernel="rbf", gamma="scale", nu=nu)
        self.scaler = StandardScaler()
        self.is_trained = False

    def fit(self, features: np.ndarray) -> None:
        if len(features) == 0:
            raise ValueError("no training features provided")
        features_scaled = self.scaler.fit_transform(features)
        self.model.fit(features_scaled)
        self.is_trained = True

    def score(self, features: np.ndarray) -> float:
        """Confidence in [0, 1] that this mask is a valid origami (0.5 = untrained)."""
        if not self.is_trained:
            return 0.5
        features = np.asarray(features).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        decision = self.model.decision_function(features_scaled)[0]
        return float(1.0 / (1.0 + np.exp(-decision)))

    def save(self, path: str | Path) -> None:
        if not self.is_trained:
            raise RuntimeError("cannot save an untrained classifier")
        joblib.dump({"model": self.model, "scaler": self.scaler}, path)

    @classmethod
    def load(cls, path: str | Path) -> "MaskClassifier":
        data = joblib.load(path)
        classifier = cls()
        classifier.model = data["model"]
        classifier.scaler = data["scaler"]
        classifier.is_trained = True
        return classifier
