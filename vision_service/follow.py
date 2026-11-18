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
    range   with the 8x8 ToF the Pi sends its whole grid (tof_grid): every person in
            view is ranged by the zones on their box. That range is their distance,
            and a check on who is who: once the target's range is known, someone
            whose range is further from where the target should be than range_tol_m
            (plus range_tol_grow_m per missed frame) is not taken by their clothes
            (a lookalike across the room). A face match is never overruled.
    plan    bearing from the box centre; distance from the ToF zones on the box, else
            from the TF-Luna when its beam
            (the image centre column) lands on the box, otherwise from the box's
            bottom edge on the floor plane when the feet are in view, otherwise
            from the size of their face, otherwise from the width of their box
            (an average adult's); a box cut off at both the top and the bottom of
            the frame is "near" and is not driven on. Turn part of the way to face them (turn_gain:
            an error in the angle then shrinks instead of swinging the robot back
            and forth), then close to the standoff in steps of at most
            max_step_m. Never reverses.
    lose    a few frames without them: hold. Longer: sweep the HEAD (the robot
            stays put) through SEARCH_SWEEP - pan left to 90 in 45-degree steps,
            tilt up 30, pan right across to -90, tilt back down, pan back to
            centre; after the whole sweep, or search_s (20 s) after they were
            lost, whichever comes first, the person is "gone" and the Pi ends the
            task (and its voice session reopens). While anyone is in view the sweep pauses, so the head
            stays on them until the face is checked. The only person in view,
            in clothes like the target's and without someone else's face, is
            taken back. With `hint_deg` (where the Pi's odometry says they were last
            seen, from its present heading, + left) the search starts there: the
            head looks that way first, or past +-90 degrees the robot turns that way.
            `head` in every answer is where the head should look:
            (pan_deg, tilt_up_deg above the follow tilt); (0, 0) once tracking.

Angles: bearing_deg follows the head (REP-103, + is LEFT); turn_deg follows the
robot's drive/turn tools (+ is RIGHT).

Measurements for the Pi (it decides how to move; turn_deg and drive_m are still sent,
and the Pi uses them only while searching): bearing_deg is already in the ROBOT's
frame (the head's pan is added); distance_sigma_m is how far off distance_m may be for
the way it was found (ToF zones a few cm; feet on the floor d^2/h per degree of tilt
error; face size 15 %; box width 30 %); ground_xy is where they stand, metres ahead
and to the left of the robot when the frame was taken; confidence is the face match
(1.0) or how well their clothes match the track's looks (0..1).
"""

from __future__ import annotations

import json
import math
import time
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
    standoff_m: float = 0.8          # stop this far from them (VISION-DESIGN 7.2 FOLLOW_STANDOFF_M)
    max_step_m: float = 0.5          # longest drive per step; then look again
    drive_max_bearing_deg: float = 25.0   # turning further than this: turn only, look again
    turn_deadband_deg: float = 8.0
    turn_gain: float = 0.6           # turn this share of the bearing each step, never all of it
    bottom_margin: float = 0.03      # a box this close to the bottom edge has its feet out of frame
    face_width_m: float = 0.16       # an adult's face box, for distance when the feet are not in view
    body_width_m: float = 0.45       # an adult's person box, the last resort
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
    search_s: float = 20.0           # once locked, lost this long (holding + sweep) is "gone"
    max_range_m: float = 6.0         # TF-Luna readings beyond this are ignored
    range_tol_m: float = 1.2         # ToF range this far from where the target should be: not them by clothes
    range_tol_grow_m: float = 0.5    # ...plus this per missed frame (they walk on while lost)


# Head sweep while searching: (pan_deg, tilt_up_deg). + pan is LEFT.
# 45-degree steps: the camera sees 80 degrees, so neighbouring looks still overlap.
SEARCH_SWEEP = [(45, 0), (90, 0), (90, 30), (0, 30), (-90, 30), (-90, 0), (-45, 0), (0, 0)]


@dataclass
class Geometry:
    pan_deg: float = 0.0
    tilt_deg: float = 0.0
    hfov_deg: float = 66.0
    camera_height_m: float = 0.25
    range_m: Optional[float] = None   # TF-Luna along the camera axis
    tof_grid: Optional[list] = None   # 8x8 ToF ranges (m, None = no return), row 0 the top, as the camera sees
    tof_fov_deg: float = 60.0         # the ToF's square field of view
    hint_deg: Optional[float] = None  # where they were last seen, from the robot's heading now (+ left)


def range_in_box(grid, box, W: int, H: int, hfov_deg: float, tof_fov_deg: float = 60.0) -> Optional[float]:
    """Median range of the ToF zones whose centres fall in a camera box; a box too small
    for any centre takes the nearest zone it touches (it stands in front of what is
    behind it). The ToF sits beside the camera and looks the same way. Same rule as the
    Pi's robot_core.sensors.tof8x8.range_in_box. None if no zone on it has a range."""
    if not grid or box is None:
        return None
    n = len(grid)
    zone = tof_fov_deg / n
    f = (W / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    left = math.degrees(math.atan2(W / 2.0 - box[0], f))
    right = math.degrees(math.atan2(W / 2.0 - box[2], f))
    top = math.degrees(math.atan2(H / 2.0 - box[1], f))
    bottom = math.degrees(math.atan2(H / 2.0 - box[3], f))
    good, touched, half, mid = [], [], zone / 2.0, (n - 1) / 2.0
    for r in range(n):
        for c in range(len(grid[r])):
            v = grid[r][c]
            if v is None:
                continue
            up, lf = (mid - r) * zone, (mid - c) * zone
            if right <= lf <= left and bottom <= up <= top:
                good.append(v)
            elif right - half < lf < left + half and bottom - half < up < top + half:
                touched.append(v)
    if good:
        return float(np.median(good))
    return min(touched) if touched else None


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
    def __init__(self, cfg: FollowConfig = FollowConfig(), clock=time.monotonic) -> None:
        self.cfg, self.clock = cfg, clock
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
        self.expect_dist: Optional[float] = None  # their ToF distance next frame (after the drive)
        self.path = SEARCH_SWEEP                  # this search's head positions
        self.hinted = False                       # the hint was used for this loss
        self.lost_t: Optional[float] = None       # when they were last lost (first miss)
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
             face_labels: Sequence[Optional[str]], geo: Geometry,
             face_px: Sequence[Optional[float]] = ()) -> dict:
        """One frame. `face_labels[i]` is the gallery label (above threshold)
        of person i's face, STRANGER for a clear face that is not the target, or
        None (no face, too unclear to say, or not checked). `face_px[i]` is the
        width of person i's face in pixels, when one was seen."""
        cfg = self.cfg
        H, W = rgb.shape[:2]
        f = (W / 2.0) / math.tan(math.radians(geo.hfov_deg) / 2.0)
        bearings = [geo.pan_deg + math.degrees(math.atan2(W / 2.0 - (p.bbox[0] + p.bbox[2]) / 2.0, f))
                    for p in persons]
        ranges = [range_in_box(geo.tof_grid, p.bbox, W, H, geo.hfov_deg, geo.tof_fov_deg) for p in persons]
        ranges = [None if r is None or not 0.1 < r <= cfg.max_range_m
                  else r * math.cos(math.radians(geo.tilt_deg)) for r in ranges]
        pick = self._pick(rgb, persons, face_labels, bearings, ranges)

        if pick is None:
            self.misses += 1
            if self.misses == 1:
                self.lost_t = self.clock()
            if self.locked and self.lost_t is not None and self.clock() - self.lost_t > cfg.search_s:
                return self._out("gone", geo)             # lost too long: give up, the Pi ends the task
            if self.locked and self.misses <= cfg.hold_frames:
                return self._out("holding", geo)
            if persons:                                   # someone in view: stop sweeping, keep looking here
                return self._out("searching", geo, box=persons[0].bbox)
            if not self.locked and self.misses % 2:      # two looks per heading before turning
                return self._out("searching", geo)
            if self.sweep == 0:                           # a new search: start where they were last seen
                turn = self._start_search(geo.hint_deg)
                if turn:
                    return self._out("searching", geo, turn_deg=turn)
            if self.sweep >= len(self.path):
                return self._out("gone", geo)
            self.sweep += 1                               # next head position
            return self._out("searching", geo)

        if not self.locked:          # still confirming who it is: don't move yet
            self.misses = 0
            return self._out("searching", geo, box=persons[pick].bbox)

        self.misses, self.sweep, self.followed, self.hinted = 0, 0, True, False
        box = persons[pick].bbox
        bearing = bearings[pick]
        self.last_bearing = bearing
        dist, how = self._distance(box, W, H, f, geo, face_px[pick] if pick < len(face_px) else None,
                                   ranges[pick])

        turn = -bearing * cfg.turn_gain if abs(bearing) > cfg.turn_deadband_deg else 0.0
        drive = 0.0
        if dist is not None and how != "near" and abs(bearing) <= cfg.drive_max_bearing_deg \
                and dist > cfg.standoff_m + 0.1:
            drive = min(dist - cfg.standoff_m, cfg.max_step_m)
        # Turning right by `turn` moves every bearing left (+) by as much.
        self.expect = bearing + turn
        self.expect_dist = dist - drive if how == "tof" else None
        conf = 1.0 if face_labels[pick] == self.target and self.target else \
            (lambda sig: 0.0 if sig is None else self.similarity(sig))(signature(rgb, box))
        return self._out("tracking", geo, box=box, bearing=bearing, dist=dist, how=how,
                         turn_deg=turn, drive_m=drive, confidence=conf)

    def _start_search(self, hint: Optional[float]) -> float:
        """Order this search's head positions: where they were last seen first. Further
        round than the head can look: turn the robot that way instead (the turn, + right)."""
        self.path = SEARCH_SWEEP
        if hint is None or abs(hint) < 10.0 or self.hinted:
            return 0.0
        self.hinted = True                        # once per loss: after a turn the hint is stale
        if abs(hint) > 90.0:
            return -max(-180.0, min(180.0, hint))
        self.path = [(int(round(hint)), 0)] + SEARCH_SWEEP
        return 0.0

    def _pick(self, rgb, persons, face_labels, bearings, ranges=None) -> Optional[int]:
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

        # Everyone whose face does not say they are someone else, nor their range.
        ranges = ranges or [None] * len(persons)
        tol = cfg.range_tol_m + cfg.range_tol_grow_m * self.misses
        cands = []                                       # (index, signature, similarity)
        for i, p in enumerate(persons):
            if face_labels[i] is not None:               # another enrolled face, or a stranger's
                continue
            if self.expect_dist is not None and ranges[i] is not None and abs(ranges[i] - self.expect_dist) > tol:
                continue                                 # too near or too far to be them
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
        # 1b. Lost and searching: the only person in view, in clothes like theirs and without
        # someone else's face, is taken back (not learned).
        elif len(persons) == 1 and cands and cands[0][2] >= cfg.continuity_similarity:
            return cands[0][0]

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

    def _distance(self, box, W, H, f, geo: Geometry, face_px: Optional[float] = None,
                  tof: Optional[float] = None):
        """(metres, how): the ToF zones on the box; else the TF-Luna on the box; else
        the feet on the floor; else the face's size; else the box's width. (None, None)
        if none of them works."""
        cfg = self.cfg
        if tof is not None:
            return tof, "tof"
        r = geo.range_m
        if r is not None and 0.1 < r <= cfg.max_range_m and box[0] <= W / 2.0 <= box[2]:
            return r * math.cos(math.radians(geo.tilt_deg)), "range"
        # A box ending at (or a few px short of) the bottom edge is cut off there: the
        # "feet" would be the frame edge, always the same distance whoever stands where.
        near = False
        if box[3] < H * (1.0 - cfg.bottom_margin):
            below = math.radians(-geo.tilt_deg) + math.atan2(box[3] - H / 2.0, f)
            if below > math.radians(1.0):
                return geo.camera_height_m / math.tan(below), "floor"
        else:                                          # cut off at the top as well: they fill the frame
            near = box[1] <= H * cfg.bottom_margin
        if face_px:
            dist, how = f * cfg.face_width_m / face_px, "face"
        elif box[0] > 1.0 and box[2] < W - 1.0:        # their whole width is in view
            dist, how = f * cfg.body_width_m / max(box[2] - box[0], 1.0), "width"
        else:
            return None, None
        # Filling the frame top to bottom they are close, whatever the size guess says (legs
        # alone are narrower than a body and read as far): "near" is not driven on.
        return dist, ("near" if near else how)

    def _sigma(self, dist: Optional[float], how: Optional[str], geo: Geometry) -> Optional[float]:
        """How far off `dist` may be (m), by the way it was found."""
        if dist is None:
            return None
        if how in ("tof", "range"):
            return 0.05 + 0.02 * dist
        if how == "floor":
            return max(0.05, dist * dist / max(geo.camera_height_m, 0.05) * math.radians(1.0))
        return (0.15 if how == "face" else 0.3) * dist           # face | width | near

    def _out(self, state: str, geo: Geometry, box=None, bearing=None, dist=None, how=None,
             turn_deg: float = 0.0, drive_m: float = 0.0, confidence: Optional[float] = None) -> dict:
        ground = None
        if bearing is not None and dist is not None:
            ground = [round(dist * math.cos(math.radians(bearing)), 3), round(dist * math.sin(math.radians(bearing)), 3)]
        sigma = self._sigma(dist, how, geo)
        return {
            "state": state,                       # searching | tracking | holding | gone
            "target": self.target,
            "locked": self.locked,
            "followed": self.followed,
            "bbox": [round(float(v), 1) for v in box] if box is not None else None,
            "bearing_deg": None if bearing is None else round(bearing, 1),
            "distance_m": None if dist is None else round(dist, 2),
            "distance_from": how,                 # tof | range | floor | face | width | near (they fill the frame: no drive)
            "distance_sigma_m": None if sigma is None else round(sigma, 3),
            "ground_xy": ground,                  # [ahead, left] m from the robot when the frame was taken
            "confidence": None if confidence is None else round(float(confidence), 2),
            "turn_deg": round(float(turn_deg), 1),
            "drive_m": round(float(drive_m), 2),
            "head": list(self.path[self.sweep - 1]) if self.sweep else [0, 0],
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
    tof_grid: Optional[str] = Form(None, description="JSON 8x8 ToF ranges in m (null = no return), row 0 the top"),
    tof_fov_deg: float = Form(60.0),
    hint_deg: Optional[float] = Form(None, description="where they were last seen, from the robot's heading (+ left)"),
) -> dict:
    """One frame of follow mode -> where the person is and how to move:
    turn `turn_deg` (+ right) then drive `drive_m`. state `gone` = give up."""
    st = request.app.state
    if not st.follow.active:
        raise HTTPException(status_code=409, detail="follow is not started")
    grid = None
    if tof_grid:
        try:
            grid = json.loads(tof_grid)
            assert isinstance(grid, list) and len(grid) == 8 and all(isinstance(r, list) and len(r) == 8 for r in grid)
            grid = [[None if v is None else float(v) for v in r] for r in grid]
        except (ValueError, AssertionError, TypeError) as exc:
            raise HTTPException(status_code=422, detail="tof_grid must be 8 rows of 8 ranges (m) or null") from exc
    rgb = images.decode(await image.read())
    dets, _ = await st.runner.run(st.backend.detector.detect, rgb, 0.4)
    persons = sorted((d for d in dets if d.cls == "person"),
                     key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]),
                     reverse=True)[:MAX_PERSONS]
    faces = [await _face_label(st, rgb, p.bbox) for p in persons]
    out = st.follow.step(rgb, persons, [label for label, _ in faces], Geometry(
        pan_deg, tilt_deg, hfov_deg, camera_height_m, range_m, grid, tof_fov_deg, hint_deg),
        face_px=[w for _, w in faces])
    return {"seq": seq, "persons": len(persons), **out}


@router.post("/v1/follow/stop", tags=["follow"])
async def follow_stop(request: Request) -> dict:
    st = request.app.state
    st.follow.stop()
    return {"target": st.follow.target, "followed": st.follow.followed}


async def _face_label(st, rgb: np.ndarray, box) -> tuple[Optional[str], Optional[float]]:
    """(label, face width in px). Label: the gallery label of the face in a person's
    head crop if above threshold; STRANGER for a clear face that is plainly not the
    follow target (or, before a "follow me" lock, nobody enrolled); None when there
    is no telling. Width: None when no single face was found."""
    crop = head_crop(rgb, box)
    if min(crop.shape[:2]) < 32:
        return None, None
    face, _ = await st.runner.run(st.backend.face.embed, crop)
    if face is None or face.face_count > 1:
        return None, None
    width = float(face.face_bbox[2] - face.face_bbox[0])
    matches = st.gallery.match(face.embedding, top_k=max(1, len(st.gallery.labels())))
    if matches and matches[0].similarity >= settings.match_threshold:
        return matches[0].label, width
    want = st.follow.target or st.follow.label
    to_want = max((m.similarity for m in matches if want is None or m.label == want), default=0.0)
    if face.quality >= settings.face_min_quality and to_want < settings.face_reject_threshold:
        return STRANGER, width
    return None, width
