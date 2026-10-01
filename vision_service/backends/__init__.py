"""Backend construction.

The service always loads the real models (`real.py`). A missing dependency or
unloadable model is a startup failure, never a silent degradation.
"""

from __future__ import annotations

from .base import (
    Backend, Detection, DepthEstimator, Detector, FaceEmbedding, FaceEncoder, PlantEncoder,
)

__all__ = [
    "Backend", "Detection", "FaceEmbedding",
    "Detector", "FaceEncoder", "DepthEstimator", "PlantEncoder",
    "build_backend",
]


def build_backend() -> Backend:
    # Imported here so that importing the app (e.g. in tests) never pulls in
    # torch and friends.
    from . import real
    return real.build()
