"""Backend selection.

`VS_BACKEND` decides: "real" insists on models, "fake" insists on stubs, and
"auto" prefers real but degrades to fake with a loud warning. Auto is the
default so a fresh clone runs; production sets "real" so a missing dependency
is a startup failure rather than a robot that silently sees imaginary furniture.
"""

from __future__ import annotations

import logging

from ..settings import settings
from .base import Backend, Detection, DepthEstimator, Detector, FaceEmbedding, FaceEncoder

log = logging.getLogger(__name__)

__all__ = [
    "Backend", "Detection", "FaceEmbedding",
    "Detector", "FaceEncoder", "DepthEstimator",
    "build_backend",
]


def build_backend() -> Backend:

    from . import real
    return real.build()
