"""The global navigation layer's endpoints (the Pi's local layer: robot_core/nav).

    POST /v1/nav/frame     the camera stream (~10 fps): JPEG, capture time, odometry pose, head pan
                           and real tilt -> visual SLAM (vslam.py)
    POST /v1/nav/plan      {pose, goal, radius_m, local} -> {path: waypoints in the Pi's odometry
                           frame}. The Pi's rolling costmap (`local`) is folded into the global map
                           first; Theta* plans in the map frame (corrected by visual SLAM)
    GET  /v1/nav/state     map size, SLAM state, odom_to_map
    GET  /v1/nav/map.png   the map as a picture, with the robot and its last path
    POST /v1/nav/reset     a new, empty map

`project_detections` puts what the follow step's detector sees (chairs, sofas, bins... not
people) on the map, from the box's bottom edge on the floor: follow.py calls it when the Pi
sends its pose with a frame.
"""

from __future__ import annotations

import json
import math
import time
from typing import Optional

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from .nav_map import GlobalMap, apply, compose, inverse
from .settings import settings
from .vslam import MonoVSLAM

router = APIRouter()

# COCO classes that stand on the floor and block a small robot (people are followed, not avoided).
OBSTACLES = {"chair", "couch", "bed", "dining table", "toilet", "potted plant", "bench", "suitcase",
             "backpack", "handbag", "bicycle", "motorcycle", "dog", "cat", "tv", "refrigerator",
             "oven", "sink", "vase", "sports ball", "skateboard", "umbrella"}
SEMANTIC_MAX_M = 3.0             # floor projection gets poor beyond this


class NavSession:
    def __init__(self) -> None:
        self.map = GlobalMap(settings.slam_size_m, settings.slam_res_m)
        self.vslam = MonoVSLAM(settings.orb_vocab, settings.camera_hfov_deg,
                               (settings.head_x_m, settings.head_y_m, settings.head_height_m),
                               enabled=settings.vslam_enabled)
        self.plans = self.failed = 0
        self.last_plan_ms = 0.0

    @property
    def odom_to_map(self) -> tuple:
        return self.vslam.odom_to_map

    def reset(self) -> None:
        self.map = GlobalMap(settings.slam_size_m, settings.slam_res_m)


def project_detections(nav: Optional[NavSession], dets, W: int, H: int, pan_deg: float, tilt_deg: float,
                       hfov_deg: float, camera_height_m: float, pose) -> int:
    """Floor-standing obstacles the detector found, onto the global map. tilt_deg: the head's real
    tilt above level (as follow gets it). Returns how many were placed."""
    if nav is None or pose is None:
        return 0
    f = (W / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    to_map = compose(nav.odom_to_map, pose)
    n = 0
    for d in dets:
        if d.cls not in OBSTACLES or d.bbox[3] >= H * 0.97:      # bottom cut off: no floor contact seen
            continue
        below = math.radians(-tilt_deg) + math.atan2(d.bbox[3] - H / 2.0, f)
        if below <= math.radians(2.0):
            continue
        dist = camera_height_m / math.tan(below)
        if dist > SEMANTIC_MAX_M:
            continue
        bearing = math.radians(pan_deg) + math.atan2(W / 2.0 - (d.bbox[0] + d.bbox[2]) / 2.0, f)
        width = (d.bbox[2] - d.bbox[0]) / f * dist
        r = dist + width / 4.0                              # the box's bottom edge is its near side
        xy = apply(to_map, [(r * math.cos(bearing), r * math.sin(bearing))])[0]
        nav.map.add_semantic((float(xy[0]), float(xy[1])), max(0.12, min(0.6, width / 2.0)), d.cls)
        n += 1
    return n


def _nav(request: Request) -> NavSession:
    st = request.app.state
    if getattr(st, "nav", None) is None:
        st.nav = NavSession()
    return st.nav


@router.post("/v1/nav/frame", tags=["nav"])
async def nav_frame(request: Request, image: UploadFile = File(...), t: float = Form(...),
                    pose: Optional[str] = Form(None), pan_deg: float = Form(0.0),
                    tilt_deg: float = Form(0.0)) -> dict:
    """One frame of the robot's camera stream, for visual SLAM. tilt_deg is the head's REAL tilt."""
    nav = _nav(request)
    nav.vslam.feed(await image.read(), t, json.loads(pose) if pose else None, pan_deg, tilt_deg)
    return {"vslam": nav.vslam.backend, "tracking": nav.vslam.tracking}


class PlanIn(BaseModel):
    pose: list                   # (x, y, heading) in the Pi's odometry frame
    goal: list                   # (x, y) in the same frame
    radius_m: float = 0.19
    local: Optional[dict] = None  # RollingCostmap.snapshot()


@router.post("/v1/nav/plan", tags=["nav"])
async def nav_plan(request: Request, body: PlanIn) -> dict:
    """Waypoints from pose to goal, in the Pi's odometry frame (empty: no way found)."""
    nav = _nav(request)
    t0 = time.monotonic()
    o2m = nav.odom_to_map
    if body.local:
        nav.map.fold_local(body.local, o2m)
    start = compose(o2m, body.pose)
    nav.map.robot = start
    goal = apply(o2m, [body.goal[:2]])[0]
    path = await run_in_threadpool(nav.map.plan, (start[0], start[1]), (float(goal[0]), float(goal[1])),
                                   body.radius_m)          # not the inference queue: planning is CPU only
    nav.last_plan_ms = (time.monotonic() - t0) * 1000.0
    nav.plans += 1
    if not path:
        nav.failed += 1
        return {"path": [], "odom_to_map": list(o2m), "ms": round(nav.last_plan_ms, 1)}
    back = apply(inverse(o2m), path)
    return {"path": [[round(float(x), 3), round(float(y), 3)] for x, y in back],
            "odom_to_map": [round(v, 4) for v in o2m], "ms": round(nav.last_plan_ms, 1),
            "vslam": nav.vslam.backend}


@router.get("/v1/nav/state", tags=["nav"])
async def nav_state(request: Request) -> dict:
    nav = _nav(request)
    return {"map": nav.map.stats(), "vslam": nav.vslam.state(), "plans": nav.plans,
            "failed": nav.failed, "last_plan_ms": round(nav.last_plan_ms, 1), "robot": nav.map.robot}


@router.get("/v1/nav/map.png", tags=["nav"])
async def nav_map_png(request: Request) -> Response:
    return Response(_nav(request).map.png(), media_type="image/png")


@router.post("/v1/nav/reset", tags=["nav"])
async def nav_reset(request: Request) -> dict:
    _nav(request).reset()
    return {"ok": True}
