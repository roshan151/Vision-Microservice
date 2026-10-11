"""The HTTP surface. One client (the Pi), five inference endpoints, two for ops;
explore mode's plant, ask and SLAM endpoints live in explore_api.py, follow mode's
in follow.py.

Transport rules that every endpoint obeys, from VISION-DESIGN.md Part 3:
  * binary JPEG via multipart/form-data. Never base64 — it costs 33% for nothing
  * every inference request carries `seq`; every response echoes it
  * models are resident; nothing is loaded per request
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from . import images, photos
from .backends import build_backend
from .explore_api import router as explore_router
from .follow import FollowSession, router as follow_router
from .gallery import Gallery
from .plants import BadLabel, PlantStore, check_label
from .runner import Busy, InferenceRunner, InferenceTimeout
from .schemas import (
    DepthResponse, DetectResponse, DetectionOut, EmbedResponse, EnrollResponse,
    HealthResponse, MatchOut, MatchResponse, ModelsResponse,
)
from .settings import settings
from .slam import SlamConfig, SlamSession
from .vlm import VLM

log = logging.getLogger("vision_service")


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.monotonic()
    log.info("loading models on VS_DEVICE=%s...", settings.device)
    app.state.backend = build_backend()
    app.state.gallery = Gallery(settings.gallery_path)
    app.state.runner = InferenceRunner(settings.max_queue_depth, settings.inference_timeout_s)
    app.state.plants = PlantStore(settings.plant_dir)
    app.state.vlm = VLM(settings)
    app.state.slam = SlamSession(SlamConfig(
        size_m=settings.slam_size_m, res=settings.slam_res_m, robot_radius=settings.robot_radius_m))
    app.state.follow = FollowSession()
    app.state.started_at = time.monotonic()
    log.info(
        "ready in %.1fs — backend=%s versions=%s gallery=%s",
        time.monotonic() - t0, app.state.backend.kind,
        app.state.backend.versions(), app.state.gallery.stats(),
    )
    try:
        yield
    finally:
        app.state.runner.shutdown()


app = FastAPI(
    title="Robot Vision Service",
    version="1.0.0",
    description="Off-Pi perception for Robot-Computer-Vision. See README.md.",
    lifespan=lifespan,
)
app.include_router(explore_router)
app.include_router(follow_router)


# --------------------------------------------------------------------------
# error handling — one shape for every failure, so the Pi parses one thing
# --------------------------------------------------------------------------

def _err(status: int, error: str, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": error, "detail": detail})


@app.exception_handler(images.ImageTooLarge)
async def _too_large(_: Request, exc: images.ImageTooLarge):
    return _err(413, "image_too_large", str(exc))


@app.exception_handler(images.ImageUndecodable)
async def _undecodable(_: Request, exc: images.ImageUndecodable):
    return _err(400, "image_undecodable", str(exc))


@app.exception_handler(Busy)
async def _busy(_: Request, exc: Busy):
    # 503 + Retry-After is the honest answer: try again, don't queue.
    return JSONResponse(
        status_code=503,
        content={"error": "busy", "detail": str(exc)},
        headers={"Retry-After": "1"},
    )


@app.exception_handler(InferenceTimeout)
async def _timeout(_: Request, exc: InferenceTimeout):
    return _err(504, "inference_timeout", str(exc))


# --------------------------------------------------------------------------
# ops
# --------------------------------------------------------------------------

@app.get("/healthz", response_model=HealthResponse, tags=["ops"])
async def healthz(request: Request) -> HealthResponse:
    """Liveness plus the two things worth alerting on: backend kind and queue depth."""
    st = request.app.state
    return HealthResponse(
        status="ok",
        backend=st.backend.kind,
        versions=st.backend.versions(),
        gallery=st.gallery.stats(),
        plants=st.plants.gallery.stats(),
        vlm=st.vlm.available(),
        queue_depth=st.runner.queue_depth,
        uptime_s=round(time.monotonic() - st.started_at, 1),
    )


@app.get("/v1/models", response_model=ModelsResponse, tags=["ops"])
async def models(request: Request) -> ModelsResponse:
    """What is actually loaded. The Pi logs this on change (PLAN rule #6)."""
    st = request.app.state
    return ModelsResponse(
        backend=st.backend.kind,
        versions=st.backend.versions(),
        depth_is_metric=bool(getattr(st.backend.depth, "metric", False)),
    )


# --------------------------------------------------------------------------
# inference
# --------------------------------------------------------------------------

@app.post("/v1/detect", response_model=DetectResponse, tags=["inference"])
async def detect(
    request: Request,
    image: UploadFile = File(..., description="JPEG, ~640 px long edge"),
    seq: int = Form(..., description="Pi's monotonic capture counter; echoed back"),
    min_confidence: float = Form(0.4, ge=0.0, le=1.0),
) -> DetectResponse:
    """Object detection. The Pi's hot path, ~2 Hz.

    `bbox` is in the pixel space of the image **as sent**, and `image_size`
    echoes that size so the Pi never has to assume a scale.
    """
    st = request.app.state
    frame = images.decode(await image.read())
    dets, ms = await st.runner.run(st.backend.detector.detect, frame, min_confidence)
    h, w = frame.shape[:2]
    return DetectResponse(
        seq=seq,
        detections=[
            DetectionOut(**{"class": d.cls, "confidence": d.confidence, "bbox": list(d.bbox)})
            for d in dets
        ],
        model_version=st.backend.detector.model_version,
        image_size=[w, h],
        inference_ms=ms,
    )


@app.post("/v1/faces/embed", response_model=EmbedResponse, tags=["inference"])
async def embed(
    request: Request,
    background: BackgroundTasks,
    image: UploadFile = File(..., description="JPEG crop of the person/head region"),
    seq: int = Form(...),
) -> EmbedResponse:
    """Face embedding for one crop.

    Called **once per person track**, not per frame (VISION-DESIGN.md Part 6).
    `quality` lets the caller drop a blurry or steeply-angled crop before
    spending a match call on it; below ~0.6 is not worth matching.

    404 when there is no face in the crop — that is a normal outcome for a
    person seen from behind, not an error to retry.
    """
    st = request.app.state
    crop = images.decode(await image.read())
    result, ms = await st.runner.run(st.backend.face.embed, crop)
    if result is None:
        raise HTTPException(status_code=404, detail="no face found in crop")
    if result.quality < settings.face_min_quality or result.face_count > 1:
        background.add_task(_save_rejected, crop, result, seq)
    return EmbedResponse(
        seq=seq,
        embedding=[round(float(v), 6) for v in result.embedding.tolist()],
        face_bbox=list(result.face_bbox),
        quality=result.quality,
        face_count=result.face_count,
        model_version=st.backend.face.model_version,
        inference_ms=ms,
    )


def _save_rejected(rgb: np.ndarray, result, seq: int) -> None:
    """Background task: keep a face frame the Pi will drop (quality below the cut-off,
    or several faces) as faces/_rejected/photo_<epoch ms>.png + bbox, and append its
    quality to quality.csv. The Pi never sends these to /enroll, so without this a
    failed enroll leaves nothing behind to look at."""
    folder = settings.face_photo_dir / "_rejected"
    i = int(time.time() * 1000)
    try:
        photos.write(folder, i, rgb, result.face_bbox)
        csv = folder / "quality.csv"
        new = not csv.exists()
        with csv.open("a") as f:
            if new:
                f.write("photo,time,seq,quality,face_count\n")
            f.write(f"photo_{i}.png,{time.strftime('%Y-%m-%d %H:%M:%S')},{seq},"
                    f"{result.quality:.3f},{result.face_count}\n")
    except Exception:                                       # noqa: BLE001
        log.exception("could not save rejected face photo_%d", i)


@app.post("/v1/faces/match", response_model=MatchResponse, tags=["inference"])
async def match(request: Request, body: dict) -> MatchResponse:
    """Identity lookup against the gallery. No GPU, no model — pure numpy.

    Below `threshold`, the caller must treat the result as `unknown`. The
    threshold is returned rather than assumed so it can be tuned server-side
    without a Pi deploy.
    """
    st = request.app.state
    raw = body.get("embedding")
    if not isinstance(raw, list) or not raw:
        raise HTTPException(status_code=422, detail="embedding must be a non-empty list of floats")
    top_k = int(body.get("top_k", 1))
    if not 1 <= top_k <= 25:
        raise HTTPException(status_code=422, detail="top_k must be between 1 and 25")
    try:
        matches = st.gallery.match(np.asarray(raw, dtype=np.float32), top_k=top_k)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return MatchResponse(
        matches=[MatchOut(label=m.label, similarity=m.similarity) for m in matches],
        threshold=settings.match_threshold,
        gallery_size=st.gallery.size(),
    )


@app.post("/v1/faces/enroll", response_model=EnrollResponse, tags=["admin"])
async def enroll(
    request: Request,
    background: BackgroundTasks,
    label: str = Form(..., min_length=1, max_length=64),
    images_: list[UploadFile] = File(..., alias="images"),
) -> EnrollResponse:
    """Add a person to the gallery. Admin/offline — never on the hot path.

    The accepted photos are saved as faces/<label>/photo_i.png + bbox_i.txt by a
    background task; the response does not wait for them.

    Images with no face, or more than one (whose face would be ambiguous), are
    counted in `images_rejected` rather than
    failing the call, so one bad photo in a batch of ten does not lose the other
    nine. A call where every image is rejected is a 422.
    """
    st = request.app.state
    try:
        check_label(label)              # it becomes a folder name
    except BadLabel as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if len(images_) > settings.max_enroll_images:
        raise HTTPException(
            status_code=413,
            detail=f"{len(images_)} images exceeds the {settings.max_enroll_images} limit",
        )
    embeddings, shots, rejected = [], [], 0
    for upload in images_:
        frame = images.decode(await upload.read())
        result, _ = await st.runner.run(st.backend.face.embed, frame)
        if result is None or result.face_count > 1:
            rejected += 1
            continue
        embeddings.append(result.embedding)
        shots.append((frame, result.face_bbox))
    if not embeddings:
        raise HTTPException(status_code=422, detail="no single-face image among those supplied")
    try:
        added = st.gallery.add(label, embeddings)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    folder = settings.face_photo_dir / label
    background.add_task(photos.write_all, folder, photos.next_index(folder), shots)
    log.info("enrolled %s: +%d embeddings (%d rejected)", label, added, rejected)
    return EnrollResponse(
        label=label, embeddings_added=added,
        images_rejected=rejected, gallery_size=st.gallery.size(),
    )


@app.get("/v1/faces", tags=["admin"])
async def list_faces(request: Request) -> dict:
    """Who is in the gallery. Labels only — embeddings are not browsable."""
    st = request.app.state
    return {"labels": st.gallery.labels(), **st.gallery.stats()}


@app.delete("/v1/faces/{label}", tags=["admin"])
async def delete_face(request: Request, label: str) -> dict:
    """Forget a person entirely. 404 if they were not enrolled."""
    st = request.app.state
    removed = st.gallery.remove(label)
    if removed == 0:
        raise HTTPException(status_code=404, detail=f"{label!r} is not in the gallery")
    return {"label": label, "embeddings_removed": removed, "gallery_size": st.gallery.size()}


@app.post("/v1/depth", response_model=DepthResponse, tags=["inference"])
async def depth(
    request: Request,
    image: UploadFile = File(...),
    seq: int = Form(...),
    points: str = Form(..., description='JSON array of [cx, cy] pixel points, e.g. [[320,240]]'),
) -> DepthResponse:
    """Monocular depth at specific points. **Fallback only.**

    Prefer the lidar: the camera gives an accurate bearing and `/scan` gives an
    accurate metric range at that bearing, locally and for free. Call this only
    when the scan sees past the object because it sits above or below the
    lidar's horizontal plane (VISION-DESIGN.md Part 3).

    `points` are bbox centres, so the response is sparse rather than a full
    depth map. **Check `metric`** — when false the values order correctly but
    are not metres, and using them as distances will place objects wrongly.
    """
    st = request.app.state
    try:
        parsed = json.loads(points)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"points is not valid JSON: {exc}") from exc
    if not isinstance(parsed, list) or not parsed:
        raise HTTPException(status_code=422, detail="points must be a non-empty array")
    if len(parsed) > 64:
        raise HTTPException(status_code=422, detail="at most 64 points per call")
    try:
        pts = [(float(p[0]), float(p[1])) for p in parsed]
    except (TypeError, ValueError, IndexError) as exc:
        raise HTTPException(status_code=422, detail="each point must be [cx, cy]") from exc

    frame = images.decode(await image.read())
    (depths, confs), ms = await st.runner.run(st.backend.depth.depth_at, frame, pts)
    return DepthResponse(
        seq=seq, depths_m=depths, confidence=confs,
        metric=bool(getattr(st.backend.depth, "metric", False)),
        model_version=st.backend.depth.model_version,
        inference_ms=ms,
    )
