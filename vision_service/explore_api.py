"""Endpoints behind the robot's explore mode: plants, the image+text model,
and keyframe SLAM. Same transport rules as app.py (binary JPEG multipart, one
error shape); state lives on app.state, built in app.lifespan.

    POST   /v1/plants/embed        crop -> embedding            (GPU runner)
    POST   /v1/plants/match        embedding [+ position] -> best plants
    POST   /v1/plants/enroll       photos + meta -> new plant or new visit
    GET    /v1/plants              every plant, with its map position
    GET    /v1/plants/{label}      one plant's full record (visits, views)
    GET    /v1/plants/{label}/{file}   a stored photo (png)
    DELETE /v1/plants/{label}      forget (photos move to _deleted/)

    POST   /v1/ask                 images + question -> answer  (Gemini / local)

    POST   /v1/slam/keyframe       odom + rays -> corrected pose + next goal
    POST   /v1/slam/plan           path between two map points
    POST   /v1/slam/viewpoints     reachable poses around a target
    POST   /v1/slam/reset          new run (optionally on a saved map)
    POST   /v1/slam/save           save the map (also done after every keyframe)
    GET    /v1/slam/state
    GET    /v1/slam/map.png        the map, with plants marked
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from . import images, photos
from .plants import BadLabel, check_label
from .settings import settings
from .vlm import VLMUnavailable

log = logging.getLogger(__name__)
router = APIRouter()

PLANT_CLASS = "potted plant"        # the detector class whose box goes in bbox_{i}.txt
PLANT_BBOX_MIN_CONF = 0.25


def _encoder(request: Request):
    enc = request.app.state.backend.plant
    if enc is None:
        raise HTTPException(status_code=503, detail="plant encoder not loaded")
    return enc


# --------------------------------------------------------------------------
# plants
# --------------------------------------------------------------------------

@router.post("/v1/plants/embed", tags=["plants"])
async def plant_embed(request: Request, image: UploadFile = File(...), seq: int = Form(0)) -> dict:
    """Embedding of a plant crop (the detector's box, padded ~10 %)."""
    st = request.app.state
    enc = _encoder(request)
    crop = images.decode(await image.read())
    v, ms = await st.runner.run(enc.embed, crop)
    return {"seq": seq, "embedding": [round(float(x), 6) for x in v.tolist()],
            "model_version": enc.model_version, "inference_ms": ms}


class PlantMatchIn(BaseModel):
    embedding: list[float] = Field(min_length=1)
    top_k: int = Field(3, ge=1, le=25)
    position: Optional[list[float]] = Field(None, description="[x, y] on the map, if known")


@router.post("/v1/plants/match", tags=["plants"])
async def plant_match(request: Request, body: PlantMatchIn) -> dict:
    """Best plants by appearance, each with its distance from `position`.

    Appearance and place are returned separately on purpose: the robot decides.
    Same place + similar look = the same plant. Similar look but 3 m away =
    moved, or a sibling — the case worth asking /v1/ask about.
    """
    store = request.app.state.plants
    try:
        matches = store.gallery.match(np.asarray(body.embedding, np.float32), top_k=body.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    where = dict((l, (x, y)) for l, x, y in store.positions())
    out = []
    for m in matches:
        p = where.get(m.label)
        d = (math.hypot(p[0] - body.position[0], p[1] - body.position[1])
             if p and body.position else None)
        out.append({"label": m.label, "similarity": m.similarity, "position": list(p) if p else None,
                    "distance_m": None if d is None else round(d, 3)})
    return {"matches": out, "threshold": settings.plant_match_threshold,
            "gallery_size": store.gallery.size()}


@router.post("/v1/plants/enroll", tags=["plants"])
async def plant_enroll(
    request: Request,
    background: BackgroundTasks,
    images_: list[UploadFile] = File(..., alias="images"),
    label: str = Form("", description="empty = new plant, auto-named plant_NNN"),
    meta: str = Form("{}", description='JSON: {"position": [x, y], "views": [{"pose": [...]}, ...]}'),
) -> dict:
    """Save a visit: every photo's embedding to the gallery now, every photo to
    plants/map_N/<label>/photo_i.png (+ bbox_i.txt) in a background task — the
    response does not wait for PNG encoding or the bbox detection. An existing
    label adds a visit (growth tracking)."""
    st = request.app.state
    enc = _encoder(request)
    if len(images_) > settings.max_enroll_images:
        raise HTTPException(status_code=413, detail=f"at most {settings.max_enroll_images} images")
    try:
        meta_d = json.loads(meta)
        label = check_label(label) if label else st.plants.next_label()
    except (ValueError, BadLabel) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    raw, embeddings = [], []
    for up in images_:
        data = await up.read()
        v, _ = await st.runner.run(enc.embed, images.decode(data))
        raw.append(data)
        embeddings.append(v)
    try:
        res = await run_in_threadpool(st.plants.add_visit, label, raw, embeddings, meta_d)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    background.add_task(_save_plant_photos, st, Path(res["folder"]), res["files"], raw)
    return res


async def _save_plant_photos(st, folder: Path, files: list, raw: list) -> None:
    for name, data in zip(files, raw):
        try:
            frame = images.decode(data)
            bbox = None
            try:
                dets, _ = await st.runner.run(st.backend.detector.detect, frame, PLANT_BBOX_MIN_CONF)
                plants = [d for d in dets if d.cls == PLANT_CLASS]
                bbox = max(plants, key=lambda d: d.confidence).bbox if plants else None
            except Exception as exc:                        # noqa: BLE001  (busy/timeout: keep the photo)
                log.warning("bbox detection skipped for %s: %s", name, exc)
            await run_in_threadpool(photos.write, folder, int(name[6:-4]), frame, bbox)
        except Exception:                                   # noqa: BLE001
            log.exception("could not save %s/%s", folder, name)


@router.get("/v1/plants", tags=["plants"])
async def plant_list(request: Request) -> dict:
    store = request.app.state.plants
    return {"plants": store.list(), "stats": store.gallery.stats()}


@router.get("/v1/plants/{label}", tags=["plants"])
async def plant_get(request: Request, label: str) -> dict:
    try:
        m = request.app.state.plants.meta(label)
    except BadLabel as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if m is None:
        raise HTTPException(status_code=404, detail=f"{label!r} is not a known plant")
    return m


@router.get("/v1/plants/{label}/{file}", tags=["plants"])
async def plant_photo(request: Request, label: str, file: str):
    try:
        check_label(label)
    except BadLabel as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    folder = request.app.state.plants.folder(label)
    path = (folder / file).resolve()
    if path.parent != folder.resolve() or not path.is_file() or path.suffix != ".png":
        raise HTTPException(status_code=404, detail="no such photo")
    return FileResponse(path, media_type="image/png")


@router.delete("/v1/plants/{label}", tags=["plants"])
async def plant_delete(request: Request, label: str) -> dict:
    try:
        removed = request.app.state.plants.remove(label)
    except BadLabel as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not removed:
        raise HTTPException(status_code=404, detail=f"{label!r} is not a known plant")
    return {"label": label, "removed": True}


# --------------------------------------------------------------------------
# image + text model
# --------------------------------------------------------------------------

@router.post("/v1/ask", tags=["reasoning"])
async def ask(
    request: Request,
    question: str = Form(..., min_length=1, max_length=4000),
    images_: list[UploadFile] = File(default=[], alias="images"),
    json_mode: bool = Form(False, alias="json", description="ask for a JSON answer"),
) -> dict:
    """Free-form visual question. With json=true the answer is also parsed into
    `parsed` when it is valid JSON — the robot's canned questions use this."""
    if len(images_) > 6:
        raise HTTPException(status_code=413, detail="at most 6 images per question")
    blobs = []
    for up in images_:
        data = await up.read()
        images.decode(data)                  # validate size/format up front
        blobs.append(data)
    try:
        return await run_in_threadpool(request.app.state.vlm.ask, blobs, question, json_mode)
    except VLMUnavailable as exc:
        raise HTTPException(status_code=503, detail=f"no VLM answered: {exc}") from exc


# --------------------------------------------------------------------------
# SLAM
# --------------------------------------------------------------------------

class KeyframeIn(BaseModel):
    odom: list[float] = Field(min_length=3, max_length=3, description="x, y, theta in odom")
    rays: list[list[Optional[float]]] = Field(description="[[bearing_rad, range_m|null], ...] in base_link")
    sensor_xy: list[float] = Field([0.0, 0.0], min_length=2, max_length=2)
    obstacles: list[list[float]] = Field([], description="[[x, y]] in base_link: low obstacles / drops")
    plan: bool = True


class PlanIn(BaseModel):
    start: list[float] = Field(min_length=2)
    goal: list[float] = Field(min_length=2)


class ViewpointsIn(BaseModel):
    start: list[float] = Field(min_length=2)
    target: list[float] = Field(min_length=2, max_length=2)
    radius: float = Field(0.8, gt=0.2, le=3.0)
    n: int = Field(4, ge=1, le=12)
    start_angle: Optional[float] = None


class ResetIn(BaseModel):
    name: Optional[str] = None
    initial_pose: list[float] = Field([0.0, 0.0, 0.0], min_length=3, max_length=3)
    load: bool = Field(False, description="continue on the saved map `name`")
    explore_radius_m: Optional[float] = Field(None, description="only explore within this radius of initial_pose; null/0 = unlimited")
    no_behind: bool = Field(False, description="only explore ahead of initial_pose's heading")


@router.post("/v1/slam/keyframe", tags=["slam"])
async def slam_keyframe(request: Request, body: KeyframeIn, background: BackgroundTasks) -> dict:
    s = request.app.state.slam
    for r in body.rays:
        if len(r) != 2 or r[0] is None:
            raise HTTPException(status_code=422, detail="each ray is [bearing_rad, range_m|null]")
    out = await run_in_threadpool(s.add_keyframe, body.odom, body.rays, body.sensor_xy,
                                  body.obstacles, body.plan)
    # The Pi keeps nothing: the map is written to the Mac after every keyframe, so a
    # crash or a dead battery mid-run loses at most the scan in flight.
    background.add_task(_save_map, request.app.state)
    return out


def _save_map(st) -> str:
    """maps/<name>.npz (reloadable) and maps/<name>.png (plants marked)."""
    path = st.slam.save(settings.slam_dir)
    path.with_suffix(".png").write_bytes(st.slam.render(2, st.plants.positions()))
    return str(path)


@router.post("/v1/slam/plan", tags=["slam"])
async def slam_plan(request: Request, body: PlanIn) -> dict:
    path = await run_in_threadpool(request.app.state.slam.plan, body.start[:2], body.goal[:2])
    if path is None:
        raise HTTPException(status_code=422, detail="no path: goal unknown, blocked or unreachable")
    return {"path": path}


@router.post("/v1/slam/viewpoints", tags=["slam"])
async def slam_viewpoints(request: Request, body: ViewpointsIn) -> dict:
    vps = await run_in_threadpool(request.app.state.slam.viewpoints, body.start[:2], body.target,
                                  body.radius, body.n, body.start_angle)
    return {"viewpoints": vps, "requested": body.n}


@router.post("/v1/slam/reset", tags=["slam"])
async def slam_reset(request: Request, body: ResetIn) -> dict:
    s = request.app.state.slam
    if body.name:
        try:
            s.name = check_label(body.name)
        except BadLabel as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    if body.load:
        try:
            await run_in_threadpool(s.load, settings.slam_dir)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"no saved map {s.name!r}") from exc
    s.reset(body.initial_pose, keep_map=body.load, radius_m=body.explore_radius_m, no_behind=body.no_behind)
    # Plant photos are filed under plants/map_{index}/: a new map starts a new index.
    return {**s.state(), "map_index": request.app.state.plants.use_map(s.name, new=not body.load)}


@router.post("/v1/slam/save", tags=["slam"])
async def slam_save(request: Request) -> dict:
    return {"saved": await run_in_threadpool(_save_map, request.app.state)}


@router.get("/v1/slam/state", tags=["slam"])
async def slam_state(request: Request) -> dict:
    return request.app.state.slam.state()


@router.get("/v1/slam/map.png", tags=["slam"])
async def slam_map(request: Request, scale: int = 2) -> Response:
    st = request.app.state
    png = await run_in_threadpool(st.slam.render, max(1, min(scale, 8)), st.plants.positions())
    return Response(png, media_type="image/png")
