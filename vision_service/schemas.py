"""Response shapes. These are the contract — the Pi's gateway client mirrors them.

Every inference response carries `seq`. The Pi discards any result whose seq it
does not recognise, because a result it cannot place in time is worse than no
result (VISION-DESIGN.md Part 2.1).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class DetectionOut(BaseModel):
    cls: str = Field(serialization_alias="class", validation_alias="class")
    confidence: float
    bbox: list[float] = Field(min_length=4, max_length=4, description="x1, y1, x2, y2")

    model_config = {"populate_by_name": True}


class DetectResponse(BaseModel):
    seq: int
    detections: list[DetectionOut]
    model_version: str
    # Echoed so the Pi never has to assume what resolution its bbox is in. A
    # bbox silently in the wrong scale looks exactly like bad calibration.
    image_size: list[int] = Field(min_length=2, max_length=2, description="width, height as received")
    inference_ms: float


class EmbedResponse(BaseModel):
    seq: int
    embedding: list[float]
    face_bbox: list[float] = Field(min_length=4, max_length=4)
    quality: float
    face_count: int = 1
    model_version: str
    inference_ms: float


class MatchOut(BaseModel):
    label: str
    similarity: float


class MatchResponse(BaseModel):
    matches: list[MatchOut]
    threshold: float
    gallery_size: int


class EnrollResponse(BaseModel):
    label: str
    embeddings_added: int
    images_rejected: int
    gallery_size: int


class DepthResponse(BaseModel):
    seq: int
    depths_m: list[float]
    confidence: list[float]
    # False means the values order correctly but are NOT metres. The Pi must
    # branch on this, not on the field name.
    metric: bool
    model_version: str
    inference_ms: float


class ModelsResponse(BaseModel):
    backend: str
    versions: dict[str, str]
    depth_is_metric: bool


class HealthResponse(BaseModel):
    status: str
    backend: str
    versions: dict[str, str]
    gallery: dict[str, int]
    plants: dict[str, int] = {}
    vlm: list[str] = []
    queue_depth: int
    uptime_s: float


class ErrorResponse(BaseModel):
    error: str
    detail: str
