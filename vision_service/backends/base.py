"""What a backend has to provide, and the shapes it returns.

Keeping this narrow is what lets the whole API be tested with no models
installed and no GPU: the test suite's stub (tests/stub_backend.py) satisfies
the same protocols as the real backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class Detection:
    cls: str
    confidence: float
    bbox: tuple[float, float, float, float]   # x1, y1, x2, y2 in sent-image pixels


@dataclass(frozen=True)
class FaceEmbedding:
    embedding: np.ndarray                      # float32, L2-normalised
    face_bbox: tuple[float, float, float, float]
    quality: float                             # 0..1; the Pi discards below ~0.6
    face_count: int = 1                        # faces in the image; >1 means ambiguous


class Detector(Protocol):
    model_version: str

    def detect(self, image: np.ndarray, min_confidence: float) -> list[Detection]: ...


class FaceEncoder(Protocol):
    model_version: str

    def embed(self, image: np.ndarray) -> FaceEmbedding | None:
        """Largest face in the image, or None if there isn't one."""


class PlantEncoder(Protocol):
    model_version: str

    def embed(self, image: np.ndarray) -> np.ndarray:
        """Global appearance embedding of a plant crop (float32, L2-normalised)."""


class DepthEstimator(Protocol):
    model_version: str
    metric: bool          # False means the numbers order correctly but are not metres

    def depth_at(
        self, image: np.ndarray, points: Sequence[tuple[float, float]]
    ) -> tuple[list[float], list[float]]:
        """Returns (depths, confidences), one per requested point."""


@dataclass
class Backend:
    detector: Detector
    face: FaceEncoder
    depth: DepthEstimator
    kind: str             # "real" in the service; surfaced by /healthz and /v1/models
    plant: PlantEncoder | None = None

    def versions(self) -> dict[str, str]:
        v = {
            "detect": self.detector.model_version,
            "face": self.face.model_version,
            "depth": self.depth.model_version,
        }
        if self.plant is not None:
            v["plant"] = self.plant.model_version
        return v
