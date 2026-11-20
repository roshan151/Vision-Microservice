"""The global navigation layer's endpoints (the Pi's local layer: robot_core/nav).

    POST /v1/nav/frame     the camera stream (~10 fps): JPEG, capture time, odometry pose, head pan
                           and real tilt, the ToF grid nearest it -> visual SLAM (vslam.py)
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
from .nav_record import NavRecorder
from .settings import settings
from .vslam import MonoVSLAM

router = APIRouter()

# COCO classes that stand on the floor and block a small robot (people are followed, not avoided). Left out:
# what stands against a wall (refrigerator, oven, sink, toilet, tv: the ToF sees the wall anyway, and a white door
# was a "refrigerator" 15 times in one doorway on Nov 19), animals (they move), and what feet and legs were taken
# for (skateboard, sports ball, umbrella).
OBSTACLES = {"chair", "couch", "bed", "dining table", "potted plant", "bench", "suitcase",
             "backpack", "handbag", "bicycle", "motorcycle", "vase"}
SEMANTIC_MIN_CONF = 0.68         # a guess is not an obstacle; doorposts must not be mistaken for furniture
SEMANTIC_MAX_M = 1.8             # floor projection gets poor beyond this


class NavSession:
    def __init__(self) -> None:
        self.map = GlobalMap(settings.slam_size_m, settings.slam_res_m)
        self.vslam = MonoVSLAM(settings.orb_vocab, settings.camera_hfov_deg,
                               (settings.head_x_m, settings.head_y_m, settings.head_height_m),
                               enabled=settings.vslam_enabled)
        self.plans = self.failed = 0
        self.last_plan_ms = 0.0
        self.record = NavRecorder(status=lambda: {"vslam": self.vslam.state(), "map": self.map.stats(),
                                                  "map_origin": self.map.origin, "orb_vocab": settings.orb_vocab,
                                                  "camera_hfov_deg": settings.camera_hfov_deg,
                                                  "mount": [settings.head_x_m, settings.head_y_m, settings.head_height_m]})
        self.vslam.record = self.record

    @property
    def odom_to_map(self) -> tuple:
        return self.vslam.odom_to_map

    def reset(self) -> None:
        self.map = GlobalMap(settings.slam_size_m, settings.slam_res_m)


def _inside(box, other) -> float:
    """How much of `box` lies inside `other` (0..1)."""
    w = max(0.0, min(box[2], other[2]) - max(box[0], other[0]))
    h = max(0.0, min(box[3], other[3]) - max(box[1], other[1]))
    area = max(1e-6, (box[2] - box[0]) * (box[3] - box[1]))
    return w * h / area


def project_detections(nav: Optional[NavSession], dets, W: int, H: int, pan_deg: float, tilt_deg: float,
                       hfov_deg: float, camera_height_m: float, pose) -> int:
    """Floor-standing obstacles the detector found, onto the global map. tilt_deg: the head's real
    tilt above level (as follow gets it). Anything mostly inside a person's box is part of them, not
    in their way: on Nov 19 the followed person's feet were taken for a skateboard and a cat, the
    planner went round them, and the robot turned away and lost them. Returns how many were placed."""
    if nav is None or pose is None:
        return 0
    f = (W / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    to_map = compose(nav.odom_to_map, pose)
    people = [d.bbox for d in dets if d.cls == "person"]
    n = 0
    for d in dets:
        if d.cls not in OBSTACLES or d.confidence < SEMANTIC_MIN_CONF or d.bbox[3] >= H * 0.97:   # no floor contact seen
            continue
        if any(_inside(d.bbox, p) > 0.4 for p in people):       # their feet, bag, a pet in their arms
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
        obs_r = max(0.10, min(0.35, width / 2.0))
        nav.map.add_semantic((float(xy[0]), float(xy[1])), obs_r, d.cls)
        nav.record("semantic", cls=d.cls, xy=[float(xy[0]), float(xy[1])], radius=obs_r,
                   dist=dist, bearing_deg=math.degrees(bearing), pose=list(pose), odom_to_map=list(nav.odom_to_map))
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
                    tilt_deg: float = Form(0.0), tof: Optional[str] = Form(None)) -> dict:
    """One frame of the robot's camera stream, for visual SLAM. tilt_deg is the head's REAL tilt;
    tof: {"rows", "age_s"}, the ToF grid nearest the frame (gives SLAM its metric scale)."""
    nav = _nav(request)
    nav.vslam.feed(await image.read(), t, json.loads(pose) if pose else None, pan_deg, tilt_deg,
                   json.loads(tof) if tof else None)
    return {"vslam": nav.vslam.backend, "tracking": nav.vslam.tracking}


class PlanIn(BaseModel):
    pose: list                   # (x, y, heading) in the Pi's odometry frame
    goal: list                   # (x, y) in the same frame
    target: Optional[list] = None  # (x, y): what the goal is for (the person): the goal stays on its side of walls
    radius_m: float = 0.19
    local: Optional[dict] = None  # RollingCostmap.snapshot()


@router.post("/v1/nav/plan", tags=["nav"])
async def nav_plan(request: Request, body: PlanIn) -> dict:
    """Waypoints from pose to goal, in the Pi's odometry frame (empty: no way found)."""
    nav = _nav(request)
    t0 = time.monotonic()
    o2m = nav.odom_to_map
    folded = nav.map.fold_local(body.local, o2m) if body.local else 0
    start = compose(o2m, body.pose)
    nav.map.robot = start
    goal = apply(o2m, [body.goal[:2]])[0]
    target = apply(o2m, [body.target[:2]])[0] if body.target else None
    path = await run_in_threadpool(nav.map.plan, (start[0], start[1]), (float(goal[0]), float(goal[1])),
                                   body.radius_m, 0.05, 2.0,
                                   None if target is None else (float(target[0]), float(target[1])))
    # (not the inference queue: planning is CPU only)
    nav.last_plan_ms = (time.monotonic() - t0) * 1000.0
    nav.plans += 1
    map_file = nav.record.save_map(nav.map) if nav.record.map_due() else None
    nav.record("plan", pose=body.pose, goal=body.goal, radius_m=body.radius_m, odom_to_map=list(o2m),
               start_map=list(start), goal_map=[float(goal[0]), float(goal[1])], folded_cells=folded,
               target=body.target, target_map=None if target is None else [float(target[0]), float(target[1])],
               path_map=path or [], path_odom=apply(inverse(o2m), path).tolist() if path else [],
               margin_m=getattr(nav.map, "last_margin", None), ms=nav.last_plan_ms, ok=bool(path),
               vslam_tracking=nav.vslam.tracking, map=map_file)
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
    nav = _nav(request)
    nav.reset()
    nav.record("reset")
    return {"ok": True}
