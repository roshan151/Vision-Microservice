"""Follow mode: lock onto an enrolled person, keep finding them, say how to move.

The Pi is a stop-and-go loop: capture, POST /v1/follow/step, wait, turn, drive,
repeat. Everything that looks at pixels or decides happens here; the robot just
does what the answer says. One follower at a time, like the SLAM session.

    lock    a person box whose head crop matches the target in the face gallery,
            CONFIRM_FRAMES frames in a row (VISION-DESIGN Part 6: identity is
            bound to a track, not a frame). Their clothes' colour histogram is
            kept as the track's signature.
    track   the person box whose histogram is closest to the signature — the
            face is usually not visible from behind. A face match refreshes it.
    plan    bearing from the box centre; distance from the TF-Luna when its beam
            (the image centre column) lands on the box, otherwise from the box's
            bottom edge on the floor plane. Turn to face them, then close to the
            standoff in steps of at most MAX_STEP_M. Never reverses.
    lose    a few frames without them: hold. Longer: turn toward the side they
            were last seen on, SEARCH_TURN_DEG at a time; after a full circle
            the person is "gone" and the Pi ends the task.

Angles: bearing_deg follows the head (REP-103, + is LEFT); turn_deg follows the
robot's drive/turn tools (+ is RIGHT).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from PIL import Image

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

from . import images
from .backends.base import Detection
from .settings import settings


@dataclass
class FollowConfig:
    standoff_m: float = 1.2          # VISION-DESIGN 7.2 FOLLOW_STANDOFF_M
    max_step_m: float = 0.5          # longest drive per step; then look again
    drive_max_bearing_deg: float = 25.0   # turning further than this: turn only, look again
    turn_deadband_deg: float = 8.0
    confirm_frames: int = 2
    min_similarity: float = 0.5      # histogram intersection, 0..1
    hold_frames: int = 3             # misses before searching
    search_turn_deg: float = 30.0
    max_range_m: float = 6.0         # TF-Luna readings beyond this are ignored


@dataclass
class Geometry:
    pan_deg: float = 0.0
    tilt_deg: float = 0.0
    hfov_deg: float = 66.0
    camera_height_m: float = 0.25
    range_m: Optional[float] = None   # TF-Luna along the camera axis


def signature(rgb: np.ndarray, box: Sequence[float]) -> Optional[np.ndarray]:
    """Colour histogram (HSV 8x4x4) of the middle of a person box: clothes, not background."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    crop = rgb[int(y1 + 0.15 * h):int(y2 - 0.1 * h), int(x1 + 0.2 * w):int(x2 - 0.2 * w)]
    if crop.size < 3 * 16:
        return None
    hsv = np.asarray(Image.fromarray(np.ascontiguousarray(crop)).convert("HSV")).reshape(-1, 3)
    hist, _ = np.histogramdd(hsv, bins=(8, 4, 4), range=((0, 256),) * 3)
    hist = hist.ravel()
    return hist / max(hist.sum(), 1.0)


def head_crop(rgb: np.ndarray, box: Sequence[float]) -> np.ndarray:
    """Top 40 % of a person box, padded, for the face encoder."""
    H, W = rgb.shape[:2]
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return rgb[max(0, int(y1 - 0.1 * h)):min(H, int(y1 + 0.4 * h)),
               max(0, int(x1 - 0.1 * w)):min(W, int(x2 + 0.1 * w))]


class FollowSession:
    def __init__(self, cfg: FollowConfig = FollowConfig()) -> None:
        self.cfg = cfg
        self.start(None)
        self.active = False

    def start(self, label: Optional[str]) -> None:
        """label None = whichever enrolled person is seen first ("follow me")."""
        self.label, self.target = label, None     # target: who we locked onto
        self.active = True
        self.sig: Optional[np.ndarray] = None
        self.confirm, self.misses, self.searched_deg = 0, 0, 0.0
        self.last_bearing = 0.0
        self.followed = False                     # locked at least once
        self._candidate: Optional[str] = None     # "follow me": who is being confirmed

    def stop(self) -> None:
        self.active = False

    @property
    def locked(self) -> bool:
        return self.sig is not None

    def wants(self, label: str) -> bool:
        want = self.target or self.label or self._candidate
        return want is None or label == want

    # ------------------------------------------------------------------ step

    def step(self, rgb: np.ndarray, persons: Sequence[Detection],
             face_labels: Sequence[Optional[str]], geo: Geometry) -> dict:
        """One frame. `face_labels[i]` is the gallery label (above threshold)
        of person i's face, or None (no face, unknown, or not checked)."""
        cfg = self.cfg
        H, W = rgb.shape[:2]
        pick = self._pick(rgb, persons, face_labels)

        if pick is None:
            self.misses += 1
            if self.locked and self.misses <= cfg.hold_frames:
                return self._out("holding", geo)
            if not self.locked and self.misses % 2:      # two looks per heading before turning
                return self._out("searching", geo)
            if self.searched_deg >= 360.0:
                return self._out("gone", geo)
            self.searched_deg += cfg.search_turn_deg
            # last seen left (bearing > 0) -> turn left (negative turn_deg)
            turn = -cfg.search_turn_deg if self.last_bearing >= 0 else cfg.search_turn_deg
            return self._out("searching", geo, turn_deg=turn)

        if not self.locked:          # still confirming who it is: don't move yet
            self.misses = 0
            return self._out("searching", geo, box=persons[pick].bbox)

        self.misses, self.searched_deg, self.followed = 0, 0.0, True
        box = persons[pick].bbox
        f = (W / 2.0) / math.tan(math.radians(geo.hfov_deg) / 2.0)
        cx = (box[0] + box[2]) / 2.0
        bearing = geo.pan_deg + math.degrees(math.atan2(W / 2.0 - cx, f))
        self.last_bearing = bearing
        dist = self._distance(box, W, H, f, geo)

        turn = -bearing if abs(bearing) > cfg.turn_deadband_deg else 0.0
        drive = 0.0
        if dist is not None and abs(bearing) <= cfg.drive_max_bearing_deg \
                and dist > cfg.standoff_m + 0.1:
            drive = min(dist - cfg.standoff_m, cfg.max_step_m)
        return self._out("tracking", geo, box=box, bearing=bearing, dist=dist,
                         turn_deg=turn, drive_m=drive)

    def _pick(self, rgb, persons, face_labels) -> Optional[int]:
        cfg = self.cfg
        # A face match for the target always wins, and is how the lock is made.
        for i, label in enumerate(face_labels):
            if label is not None and self.wants(label):
                sig = signature(rgb, persons[i].bbox)
                if self.locked:
                    self.sig = sig if sig is not None else self.sig
                    return i
                self.confirm += 1
                self._candidate = label
                if self.confirm >= cfg.confirm_frames and sig is not None:
                    self.sig, self.target = sig, label
                return i
        if not self.locked:
            self.confirm, self._candidate = 0, None
            return None
        W = rgb.shape[1]
        best, best_score = None, -1.0
        for i, p in enumerate(persons):
            sig = signature(rgb, p.bbox)
            if sig is None:
                continue
            sim = float(np.minimum(sig, self.sig).sum())
            if sim < cfg.min_similarity:
                continue
            # Prefer the box nearest the centre: the robot keeps its target there.
            score = sim - 0.2 * abs((p.bbox[0] + p.bbox[2]) / 2.0 / W - 0.5)
            if score > best_score:
                best, best_score = i, score
        return best

    def _distance(self, box, W, H, f, geo: Geometry) -> Optional[float]:
        r = geo.range_m
        if r is not None and 0.1 < r <= self.cfg.max_range_m and box[0] <= W / 2.0 <= box[2]:
            return r * math.cos(math.radians(geo.tilt_deg))
        if box[3] >= H - 2:          # feet out of frame: no floor point
            return None
        below = math.radians(-geo.tilt_deg) + math.atan2(box[3] - H / 2.0, f)
        if below <= math.radians(1.0):
            return None
        return geo.camera_height_m / math.tan(below)

    def _out(self, state: str, geo: Geometry, box=None, bearing=None, dist=None,
             turn_deg: float = 0.0, drive_m: float = 0.0) -> dict:
        return {
            "state": state,                       # searching | tracking | holding | gone
            "target": self.target,
            "locked": self.locked,
            "followed": self.followed,
            "bbox": [round(float(v), 1) for v in box] if box is not None else None,
            "bearing_deg": None if bearing is None else round(bearing, 1),
            "distance_m": None if dist is None else round(dist, 2),
            "turn_deg": round(float(turn_deg), 1),
            "drive_m": round(float(drive_m), 2),
        }


# --------------------------------------------------------------------------
# endpoints — state is app.state.follow, built in app.lifespan
# --------------------------------------------------------------------------

router = APIRouter()
MAX_PERSONS = 3          # face-check at most this many people per frame, largest first


class FollowStartIn(BaseModel):
    label: str = ""      # empty = any enrolled person


@router.post("/v1/follow/start", tags=["follow"])
async def follow_start(request: Request, body: FollowStartIn) -> dict:
    """Begin following `label` (case-insensitive), or the first enrolled face
    seen when it is empty. 404 if that person is not enrolled."""
    st = request.app.state
    labels = st.gallery.labels()
    want = body.label.strip().lower()
    label = next((l for l in labels if l.lower() == want), None) if want else None
    if (want and label is None) or not labels:
        raise HTTPException(status_code=404, detail=f"{body.label or 'nobody'!r} is not enrolled")
    st.follow.start(label)
    return {"label": label, "state": "searching"}


@router.post("/v1/follow/step", tags=["follow"])
async def follow_step(
    request: Request,
    image: UploadFile = File(..., description="JPEG, ~640 px long edge"),
    seq: int = Form(...),
    pan_deg: float = Form(0.0),
    tilt_deg: float = Form(0.0),
    hfov_deg: float = Form(66.0),
    camera_height_m: float = Form(0.25),
    range_m: Optional[float] = Form(None, description="TF-Luna along the camera axis, if read"),
) -> dict:
    """One frame of follow mode -> where the person is and how to move:
    turn `turn_deg` (+ right) then drive `drive_m`. state `gone` = give up."""
    st = request.app.state
    if not st.follow.active:
        raise HTTPException(status_code=409, detail="follow is not started")
    rgb = images.decode(await image.read())
    dets, _ = await st.runner.run(st.backend.detector.detect, rgb, 0.4)
    persons = sorted((d for d in dets if d.cls == "person"),
                     key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]),
                     reverse=True)[:MAX_PERSONS]
    labels = [await _face_label(st, rgb, p.bbox) for p in persons]
    out = st.follow.step(rgb, persons, labels, Geometry(
        pan_deg, tilt_deg, hfov_deg, camera_height_m, range_m))
    return {"seq": seq, "persons": len(persons), **out}


@router.post("/v1/follow/stop", tags=["follow"])
async def follow_stop(request: Request) -> dict:
    st = request.app.state
    st.follow.stop()
    return {"target": st.follow.target, "followed": st.follow.followed}


async def _face_label(st, rgb: np.ndarray, box) -> Optional[str]:
    """Gallery label of the face in a person's head crop, if above threshold."""
    crop = head_crop(rgb, box)
    if min(crop.shape[:2]) < 32:
        return None
    face, _ = await st.runner.run(st.backend.face.embed, crop)
    if face is None or face.face_count > 1:
        return None
    best = st.gallery.match(face.embedding, top_k=1)
    return best[0].label if best and best[0].similarity >= settings.match_threshold else None
