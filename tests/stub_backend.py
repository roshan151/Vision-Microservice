"""A test double for the model backend. Lives in tests/ only — the service
itself always loads the real models.

It satisfies the same protocols as `backends/real.py`, so every route,
validation path and error branch is exercised with no models, no GPU and no
network. Outputs are derived from the image bytes: stable across runs, but
different between images, which is what the assertions need.
"""

from __future__ import annotations

import zlib
from typing import Sequence

import numpy as np

from vision_service.backends.base import Backend, Detection, FaceEmbedding

_MIN_FACE_PX = 32   # crops smaller than this "contain no face"


def _seed(image: np.ndarray) -> int:
    return zlib.crc32(np.ascontiguousarray(image).tobytes())


class StubDetector:
    model_version = "stub-detector"
    _CONFIDENCES = (0.95, 0.8, 0.6, 0.45, 0.3, 0.15)

    def detect(self, image: np.ndarray, min_confidence: float) -> list[Detection]:
        h, w = image.shape[:2]
        rng = np.random.default_rng(_seed(image))
        out = []
        for conf in self._CONFIDENCES:
            x1, y1 = rng.uniform(0, w * 0.5), rng.uniform(0, h * 0.5)
            x2, y2 = rng.uniform(x1 + 1, w), rng.uniform(y1 + 1, h)
            if conf >= min_confidence:
                out.append(Detection("object", conf, (float(x1), float(y1), float(x2), float(y2))))
        return out


class StubFaceEncoder:
    model_version = "stub-face"

    def embed(self, image: np.ndarray) -> FaceEmbedding | None:
        h, w = image.shape[:2]
        if min(h, w) < _MIN_FACE_PX:
            return None
        v = np.random.default_rng(_seed(image)).normal(size=512).astype(np.float32)
        return FaceEmbedding(
            embedding=v / np.linalg.norm(v),
            face_bbox=(0.0, 0.0, float(w), float(h)),
            quality=0.9,
            face_count=2 if w > 2 * h else 1,   # wide crops "contain two faces"
        )


class StubPlantEncoder:
    """Embedding keyed on the image's mean colour: two photos of the "same
    plant" (same colour) match, a differently coloured plant does not."""
    model_version = "stub-plant"

    def embed(self, image: np.ndarray) -> np.ndarray:
        key = tuple(int(c) // 16 for c in image.reshape(-1, 3).mean(axis=0))
        v = np.random.default_rng(abs(hash(key)) % (2**32)).normal(size=384).astype(np.float32)
        return v / np.linalg.norm(v)


class StubDepthEstimator:
    model_version = "stub-depth"
    metric = False

    def depth_at(
        self, image: np.ndarray, points: Sequence[tuple[float, float]]
    ) -> tuple[list[float], list[float]]:
        return [1.0 + i for i in range(len(points))], [0.5] * len(points)


def build_stub_backend() -> Backend:
    return Backend(
        detector=StubDetector(), face=StubFaceEncoder(), depth=StubDepthEstimator(), kind="test",
        plant=StubPlantEncoder(),
    )
