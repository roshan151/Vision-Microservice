"""Follow mode: lock onto an enrolled person, keep finding them, say how to move.

The Pi is a stop-and-go loop: capture, POST /v1/follow/step, wait, turn, drive,
repeat. Everything that looks at pixels or decides happens here; the robot just
does what the answer says. One follower at a time, like the SLAM session.

    lock    a person box whose head crop matches the target in the face gallery,
            CONFIRM_FRAMES frames in a row (VISION-DESIGN Part 6: identity is
            bound to a track, not a frame). Their clothes (torso and legs colour
            histograms) become the first "look" in the track's bank.
    track   identity is carried by continuity, not by the face, which is usually
            hidden from behind:
              - where: the robot knows where the person should be next frame
                (last bearing + the turn it just made). Someone inside that gate,
                alone, is the target even if their look changed (they turned
                round), and that new look is learned into the bank (up to
                bank_size looks: front, side, back...).
              - what: otherwise the best clothes match to any look in the bank.
                Far from where they should be (lost, or moved fast) needs a
                stronger match, and is not learned.
              - who it is not: a person whose face is someone else's (another
                enrolled person, or a clear face that does not match the target)
                is never tracked by their clothes. The target's face always wins
                and is always learned.
    plan    bearing from the box centre; distance from the TF-Luna when its beam
            (the image centre column) lands on the box, otherwise from the box's
            bottom edge on the floor plane. Turn to face them, then close to the
            standoff in steps of at most MAX_STEP_M. Never reverses.
    lose    a few frames without them: hold. Longer: sweep the HEAD (the robot
            stays put) through SEARCH_SWEEP - pan left to 90 in 30-degree steps,
            tilt up 45, pan right across to -90, tilt back down, pan back to
            centre; after the whole sweep the person is "gone" and the Pi ends
            the task. While anyone is in view the sweep pauses, so the head
            stays on them until the face is checked. `head` in every answer is where the head should look:
            (pan_deg, tilt_up_deg above the follow tilt); (0, 0) once tracking.

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
    min_similarity: float = 0.5      # histogram intersection, 0..1, near where they should be
    continuity_similarity: float = 0.35   # alone where they should be: enough to keep them
    reacquire_similarity: float = 0.6     # anywhere else in view: needed to take them back
    learn_similarity: float = 0.5    # a clear, gated match at least this good is learned
    bank_size: int = 8               # clothes looks remembered per track
    same_look: float = 0.9           # this close to a kept look: refresh it, don't add
    gate_deg: float = 30.0           # how far from the expected bearing they may be
    gate_grow_deg: float = 15.0      # wider for every frame they were missed
    motion_weight: float = 0.3       # closeness to the expected bearing, as a score bonus
    ambiguity: float = 0.05          # two candidates this close: follow the better, learn neither
    hold_frames: int = 3             # misses before searching
    max_range_m: float = 6.0         # TF-Luna readings beyond this are ignored


# Head sweep while searching: (pan_deg, tilt_up_deg). + pan is LEFT.
SEARCH_SWEEP = [(30, 0), (60, 0), (90, 0), (90, 45),
                (60, 45), (30, 45), (0, 45), (-30, 45), (-60, 45), (-90, 45),
                (-90, 0), (-60, 0), (-30, 0), (0, 0)]


@dataclass
class Geometry:
    pan_deg: float = 0.0
    tilt_deg: float = 0.0
    hfov_deg: float = 66.0
    camera_height_m: float = 0.25
    range_m: Optional[float] = None   # TF-Luna along the camera axis


STRANGER = "?"   # face_labels value: a clear face that is not the target


def _hist(rgb: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> Optional[np.ndarray]:
    crop = rgb[int(y1):int(y2), int(x1):int(x2)]
    if crop.size < 3 * 16:
        return None
    hsv = np.asarray(Image.fromarray(np.ascontiguousarray(crop)).convert("HSV")).reshape(-1, 3)
    hist, _ = np.histogramdd(hsv, bins=(8, 4, 4), range=((0, 256),) * 3)
    hist = hist.ravel()
    return hist / max(hist.sum(), 1.0)


def signature(rgb: np.ndarray, box: Sequence[float]) -> Optional[np.ndarray]:
    """Clothes, not background: colour histograms (HSV 8x4x4) of the torso and the
    legs, the middle of the box width. Two parts tell a red shirt over jeans from
    a red dress. Intersection with another signature is 0..1."""
    H, W = rgb.shape[:2]
    x1, y1, x2, y2 = (max(0.0, box[0]), max(0.0, box[1]), min(float(W), box[2]), min(float(H), box[3]))
    w, h = x2 - x1, y2 - y1
    xa, xb = x1 + 0.2 * w, x2 - 0.2 * w
    torso = _hist(rgb, xa, y1 + 0.15 * h, xb, y1 + 0.5 * h)
    legs = _hist(rgb, xa, y1 + 0.5 * h, xb, y2 - 0.1 * h)
    if torso is None:
        return None
    return np.concatenate([torso, legs if legs is not None else torso]) / 2.0


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
        self.bank: list[np.ndarray] = []          # clothes looks; [0] is the one seen with the face at lock
        self.confirm, self.misses, self.sweep = 0, 0, 0
        self.last_bearing = 0.0
        self.expect: Optional[float] = None       # bearing they should be at in the next frame
        self.followed = False                     # locked at least once
        self._candidate: Optional[str] = None     # "follow me": who is being confirmed

    def stop(self) -> None:
        self.active = False

    @property
    def locked(self) -> bool:
        return bool(self.bank)

    def similarity(self, sig: np.ndarray) -> float:
        """Best histogram intersection with any look in the bank."""
        return max((float(np.minimum(sig, b).sum()) for b in self.bank), default=0.0)

    def learn(self, sig: Optional[np.ndarray]) -> None:
        """Add a look to the bank; one nearly the same as a kept look replaces it
        (keeps up with the lighting). The first look, seen with the face, stays."""
        if sig is None:
            return
        if not self.bank:
            self.bank = [sig]
            return
        sims = [float(np.minimum(sig, b).sum()) for b in self.bank]
        i = int(np.argmax(sims))
        if sims[i] >= self.cfg.same_look:
            if i > 0:
                self.bank[i] = sig
            return
        self.bank.append(sig)
        if len(self.bank) > self.cfg.bank_size:
            del self.bank[1]                      # oldest learned look

    def wants(self, label: str) -> bool:
        want = self.target or self.label or self._candidate
        return want is None or label == want

    # ------------------------------------------------------------------ step

    def step(self, rgb: np.ndarray, persons: Sequence[Detection],
             face_labels: Sequence[Optional[str]], geo: Geometry) -> dict:
        """One frame. `face_labels[i]` is the gallery label (above threshold)
        of person i's face, STRANGER for a clear face that is not the target, or
        None (no face, too unclear to say, or not checked)."""
        cfg = self.cfg
        H, W = rgb.shape[:2]
        f = (W / 2.0) / math.tan(math.radians(geo.hfov_deg) / 2.0)
        bearings = [geo.pan_deg + math.degrees(math.atan2(W / 2.0 - (p.bbox[0] + p.bbox[2]) / 2.0, f))
                    for p in persons]
        pick = self._pick(rgb, persons, face_labels, bearings)

        if pick is None:
            self.misses += 1
            if self.locked and self.misses <= cfg.hold_frames:
                return self._out("holding", geo)
            if persons:                                   # someone in view: stop sweeping, keep looking here
                return self._out("searching", geo, box=persons[0].bbox)
            if not self.locked and self.misses % 2:      # two looks per heading before turning
                return self._out("searching", geo)
            if self.sweep >= len(SEARCH_SWEEP):
                return self._out("gone", geo)
            self.sweep += 1                               # next head position
            return self._out("searching", geo)

        if not self.locked:          # still confirming who it is: don't move yet
            self.misses = 0
            return self._out("searching", geo, box=persons[pick].bbox)

        self.misses, self.sweep, self.followed = 0, 0, True
        box = persons[pick].bbox
        bearing = bearings[pick]
        self.last_bearing = bearing
        dist = self._distance(box, W, H, f, geo)

        turn = -bearing if abs(bearing) > cfg.turn_deadband_deg else 0.0
        drive = 0.0
        if dist is not None and abs(bearing) <= cfg.drive_max_bearing_deg \
                and dist > cfg.standoff_m + 0.1:
            drive = min(dist - cfg.standoff_m, cfg.max_step_m)
        # Turning right by `turn` moves every bearing left (+) by as much.
        self.expect = bearing + turn
        return self._out("tracking", geo, box=box, bearing=bearing, dist=dist,
                         turn_deg=turn, drive_m=drive)

    def _pick(self, rgb, persons, face_labels, bearings) -> Optional[int]:
        cfg = self.cfg
        # A face match for the target always wins, and is how the lock is made.
        for i, label in enumerate(face_labels):
            if label not in (None, STRANGER) and self.wants(label):
                sig = signature(rgb, persons[i].bbox)
                if self.locked:
                    self.learn(sig)
                    return i
                self.confirm += 1
                self._candidate = label
                if self.confirm >= cfg.confirm_frames and sig is not None:
                    self.bank, self.target = [sig], label
                return i
        if not self.locked:
            self.confirm, self._candidate = 0, None
            return None

        # Everyone whose face does not say they are someone else.
        cands = []                                       # (index, signature, similarity)
        for i, p in enumerate(persons):
            if face_labels[i] is not None:               # another enrolled face, or a stranger's
                continue
            sig = signature(rgb, p.bbox)
            if sig is not None:
                cands.append((i, sig, self.similarity(sig)))

        # 1. Where they should be (tracking, or only just missed).
        if self.expect is not None and self.sweep == 0 and self.misses <= cfg.hold_frames:
            gate = cfg.gate_deg + cfg.gate_grow_deg * self.misses
            near = [(i, sig, sim, abs(bearings[i] - self.expect)) for i, sig, sim in cands
                    if abs(bearings[i] - self.expect) <= gate]
            if len(near) == 1 and near[0][2] >= cfg.continuity_similarity:
                i, sig, _, _ = near[0]                   # alone where expected: still them, learn the new look
                self.learn(sig)
                return i
            ok = sorted(((sim + cfg.motion_weight * (1.0 - miss / gate), sim, i, sig)
                         for i, sig, sim, miss in near if sim >= cfg.min_similarity),
                        key=lambda t: t[0], reverse=True)
            if ok:
                score, sim, i, sig = ok[0]
                if sim >= cfg.learn_similarity and (len(ok) == 1 or score - ok[1][0] > cfg.ambiguity):
                    self.learn(sig)
                return i

        # 2. Anywhere in view: only a strong clothes match, and nothing is learned.
        W = rgb.shape[1]
        best, best_score = None, -1.0
        for i, sig, sim in cands:
            if sim < cfg.reacquire_similarity:
                continue
            # Prefer the box nearest the centre: the robot keeps its target there.
            box = persons[i].bbox
            score = sim - 0.2 * abs((box[0] + box[2]) / 2.0 / W - 0.5)
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
            "head": list(SEARCH_SWEEP[self.sweep - 1]) if self.sweep else [0, 0],
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
    """Gallery label of the face in a person's head crop if above threshold;
    STRANGER for a clear face that is plainly not the follow target (or, before
    a "follow me" lock, nobody enrolled); None when there is no telling."""
    crop = head_crop(rgb, box)
    if min(crop.shape[:2]) < 32:
        return None
    face, _ = await st.runner.run(st.backend.face.embed, crop)
    if face is None or face.face_count > 1:
        return None
    matches = st.gallery.match(face.embedding, top_k=max(1, len(st.gallery.labels())))
    if matches and matches[0].similarity >= settings.match_threshold:
        return matches[0].label
    want = st.follow.target or st.follow.label
    to_want = max((m.similarity for m in matches if want is None or m.label == want), default=0.0)
    if face.quality >= settings.face_min_quality and to_want < settings.face_reject_threshold:
        return STRANGER
    return None
