"""The real models: Ultralytics YOLO, InsightFace, Depth Anything V2.

Everything here imports lazily and loads once at startup, because a model
loaded per request is the difference between a 20 ms answer and a 3 s one, and
the Pi's DETECT_TIMEOUT_S is 0.6 s.

Licensing, so it is written down somewhere: Ultralytics is AGPL-3.0 and
InsightFace's pretrained packs are research/non-commercial. Fine for a robot in
your house. If this ever ships, swap both — the Backend protocol is the seam
that makes that a contained change.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from ..settings import settings
from .base import Backend, Detection, FaceEmbedding

log = logging.getLogger(__name__)


def _resolve_device(requested: str) -> str:
    """mps if it is really there, else cpu, and say which."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if requested == "mps" and torch.backends.mps.is_available():
        return "mps"
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    if requested not in ("mps", "cuda", "cpu"):
        log.warning("unknown VS_DEVICE=%r, using cpu", requested)
        return "cpu"
    if requested != "cpu":
        log.warning("%s unavailable, falling back to cpu — expect ~10x slower inference", requested)
    return "cpu"


class YoloDetector:
    def __init__(self, weights: str, imgsz: int, device: str) -> None:
        from ultralytics import YOLO

        self._imgsz = imgsz
        self._device = device
        self._model = YOLO(weights)
        self._names: dict[int, str] = self._model.names
        self.model_version = f"{weights.removesuffix('.pt')}-{device}"
        # Warm the graph. The first inference on MPS compiles kernels and takes
        # far longer than steady state; doing it here keeps it out of the Pi's
        # timeout budget.
        self._model.predict(
            np.zeros((imgsz, imgsz, 3), dtype=np.uint8),
            imgsz=imgsz, device=device, verbose=False,
        )
        log.info("detector ready: %s on %s", weights, device)

    def detect(self, image: np.ndarray, min_confidence: float) -> list[Detection]:
        results = self._model.predict(
            image, imgsz=self._imgsz, conf=min_confidence,
            device=self._device, verbose=False,
        )
        out: list[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                out.append(
                    Detection(
                        cls=self._names[int(box.cls[0])],
                        confidence=round(float(box.conf[0]), 4),
                        bbox=(x1, y1, x2, y2),
                    )
                )
        return out


class InsightFaceEncoder:
    def __init__(self, pack: str, device: str) -> None:
        from insightface.app import FaceAnalysis

        providers = (
            ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            if device == "mps" else ["CPUExecutionProvider"]
        )
        self._app = FaceAnalysis(name=pack, providers=providers)
        self._app.prepare(ctx_id=0, det_size=(320, 320))
        self.model_version = f"insightface-{pack}"
        log.info("face encoder ready: %s via %s", pack, providers[0])

    def embed(self, image: np.ndarray) -> FaceEmbedding | None:
        faces = self._app.get(image)
        if not faces:
            return None
        # Largest face wins. The Pi sends a crop it already believes is one
        # person, so this is a tiebreak, not a selection policy.
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        v = np.asarray(face.normed_embedding, dtype=np.float32)
        x1, y1, x2, y2 = (float(c) for c in face.bbox)
        return FaceEmbedding(
            embedding=v,
            face_bbox=(x1, y1, x2, y2),
            quality=_face_quality(face, image, (x1, y1, x2, y2)),
        )


def _face_quality(face, image: np.ndarray, bbox: tuple[float, ...]) -> float:
    """A cheap 0..1 usability score, so the Pi can skip a doomed match call.

    Three things make a crop useless: it is tiny, it is blurry, or the head is
    turned away. Detector confidence covers some of that; size and Laplacian
    variance cover the rest. Not calibrated — it only has to rank.
    """
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    area_frac = max(0.0, ((x2 - x1) * (y2 - y1)) / float(w * h))
    size_score = min(1.0, area_frac / 0.08)          # ~8% of frame is plenty

    crop = image[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
    if crop.size == 0:
        return 0.0
    grey = crop.mean(axis=2) if crop.ndim == 3 else crop
    # Laplacian variance without a cv2 round trip.
    lap = (
        -4 * grey[1:-1, 1:-1]
        + grey[:-2, 1:-1] + grey[2:, 1:-1] + grey[1:-1, :-2] + grey[1:-1, 2:]
    ) if min(grey.shape) > 2 else np.zeros((1, 1))
    sharp_score = min(1.0, float(lap.var()) / 500.0)

    det_score = float(getattr(face, "det_score", 1.0))
    return round(min(1.0, 0.4 * size_score + 0.3 * sharp_score + 0.3 * det_score), 3)


class DepthAnythingEstimator:
    metric = False   # relative inverse depth, scaled — see the README caveat

    def __init__(self, model_id: str, device: str) -> None:
        import torch
        from transformers import pipeline

        self._torch = torch
        self._pipe = pipeline("depth-estimation", model=model_id, device=device)
        self.model_version = model_id.split("/")[-1]
        log.info("depth estimator ready: %s on %s (relative, not metric)", model_id, device)

    def depth_at(
        self, image: np.ndarray, points: Sequence[tuple[float, float]]
    ) -> tuple[list[float], list[float]]:
        from PIL import Image

        out = self._pipe(Image.fromarray(image))
        d = np.asarray(out["predicted_depth"] if "predicted_depth" in out else out["depth"],
                       dtype=np.float32)
        if d.ndim == 3:
            d = d[0]
        # The map comes back at the model's own resolution; sample in normalised
        # coordinates so the caller's pixel points land in the right place.
        h_img, w_img = image.shape[:2]
        h_d, w_d = d.shape[:2]
        depths, confs = [], []
        for cx, cy in points:
            xi = int(np.clip(cx / max(w_img - 1, 1) * (w_d - 1), 0, w_d - 1))
            yi = int(np.clip(cy / max(h_img - 1, 1) * (h_d - 1), 0, h_d - 1))
            # Median of a small patch: a single pixel on an object edge is a
            # coin flip between the object and whatever is behind it.
            patch = d[max(0, yi - 2):yi + 3, max(0, xi - 2):xi + 3]
            depths.append(round(float(np.median(patch)), 4))
            confs.append(round(float(1.0 / (1.0 + patch.std())), 3))
        return depths, confs


def build() -> Backend:
    device = _resolve_device(settings.device)
    return Backend(
        detector=YoloDetector(settings.detect_model, settings.detect_imgsz, device),
        face=InsightFaceEncoder(settings.face_model, device),
        depth=DepthAnythingEstimator(settings.depth_model, device),
        kind="real",
    )
