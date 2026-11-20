"""What the global layer computed, saved next to the robot's debug uploads, so a session can be
replayed later with the Mac's side of it (Monitoring-Service tof_camera_overlaid.py --nav).

Into the debug session that is receiving uploads right now (debug/latest.json, updated within
ACTIVE_S), under debug/<session>/nav/:

    nav.jsonl           one JSON object per line, each with "wall" (this Mac's clock, epoch s, the
                        same clock as the uploads' "received") and "kind":
        status          once per session: SLAM backend, why not running, map size and resolution
        slam_frame      a frame ORB-SLAM3 tracked: capture time t (Pi), odometry pose, head pan and
                        real tilt, tracking state, frames in / tracked / skipped
        slam_pose       a SLAM camera pose read back: R_wc, p_wc (SLAM units), the odometry pose then,
                        the base pose in the map from it, and odom_to_map before and after
        slam_tof_scale  a scale step: tracked points, how many fell on flat ToF zones, their metres
                        per SLAM unit (a sample), this frame's estimate, the running scale and spread
        slam_fit        the odometry fit (Umeyama): scale and where it came from (tof / odometry),
                        rotation, translation, residual, and the
                        point pairs it used
        slam_newmap     ORB-SLAM3 started a new map (tracking lost for good): the fit starts over
        plan            a planning request: pose and goal (odometry frame), odom_to_map, start and goal
                        in the map, the path in both frames, the margin it took, ms, and the map file
        semantic        an object the detector placed on the map (class, map xy, radius)
    follow.jsonl        follow mode (vision_service/follow.py), same "wall" clock:
        follow_start    the label asked for
        follow_step     one per /v1/follow/step, with the Pi's seq: head pan/real tilt, hint, the Pi's
                        range_m; every person in view (box, face label, bearing, ToF range from the zones
                        on the box and how many zones were inside / touching / skipped as floor, why
                        that range was refused, where their feet put them); who was picked and why
                        (face, lock, continuity, clothes near / far, searching); the distance used, how,
                        sigma, and why a ToF/range reading was passed over; misses, sweep, looks banked,
                        expected bearing / distance; the state answered
        follow_stop     who was followed
    map-000001.npz      the global map when a plan was made (at most every MAP_EVERY_S): log-odds
                        (int8, x40), camera-found cells, cropped to what is known (ix0, iy0), origin, res

Nothing is recorded when no debug session is active (robot debug_mode off). Old sessions go with
their folder (VS_DEBUG_KEEP).
"""

from __future__ import annotations

import io
import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .settings import settings

log = logging.getLogger(__name__)

ACTIVE_S = 30.0                  # a session counts as live this long after its last upload
MAP_EVERY_S = 2.0


def _round(v, nd=4):
    if isinstance(v, float):
        return round(v, nd)
    if isinstance(v, np.ndarray):
        return _round(v.tolist(), nd)
    if isinstance(v, (list, tuple)):
        return [_round(x, nd) for x in v]
    if isinstance(v, dict):
        return {k: _round(x, nd) for k, x in v.items()}
    if isinstance(v, np.generic):
        return _round(v.item(), nd)
    return v


class NavRecorder:
    def __init__(self, root: Optional[Path] = None, status=None, jsonl: str = "nav.jsonl") -> None:
        self.root = Path(root or settings.debug_dir)
        self.status = status                      # () -> dict, written once per session
        self.jsonl = jsonl                        # nav.jsonl; follow.py records into follow.jsonl
        self._lock = threading.Lock()
        self._session: Optional[str] = None
        self._checked = 0.0
        self._map_t = 0.0
        self._maps = 0

    def session(self) -> Optional[str]:
        """The debug session receiving uploads now, or None."""
        now = time.monotonic()
        if now - self._checked < (1.0 if self._session else 0.2):   # notice a new session quickly
            return self._session
        self._checked = now
        p = self.root / "latest.json"
        try:
            fresh = time.time() - p.stat().st_mtime < ACTIVE_S
            name = json.loads(p.read_text())["session"] if fresh else None
        except (OSError, ValueError, KeyError):
            name = None
        if name and name != self._session:
            self._session, self._maps = name, 0
            (self.root / name / "nav").mkdir(parents=True, exist_ok=True)
            if self.status is not None:
                try:
                    self._append(name, {"kind": "status", **self.status()})
                except Exception as exc:          # noqa: BLE001
                    log.info("nav_record: status: %s", exc)
        elif not name:
            self._session = None
        return self._session

    def __call__(self, kind: str, **data) -> None:
        """Record one computation; never raises."""
        try:
            name = self.session()
            if name:
                self._append(name, {"kind": kind, **data})
        except Exception as exc:                  # noqa: BLE001 — debugging aid only
            log.info("nav_record: %s: %s", kind, exc)

    def map_due(self) -> bool:
        return self.session() is not None and time.monotonic() - self._map_t >= MAP_EVERY_S

    def save_map(self, gmap) -> Optional[str]:
        """The global map as it is now; returns its file name (for the plan record)."""
        name = self.session()
        if not name:
            return None
        self._map_t = time.monotonic()
        with gmap._lock:
            lo = gmap.lo.copy()
            sem = (gmap.sem_t > 0) & (time.monotonic() - gmap.sem_t < gmap.semantic_keep_s)
        known = (np.abs(lo) > 0.05) | sem
        if not known.any():
            return None
        ii, jj = np.nonzero(known)
        x0, x1, y0, y1 = ii.min(), ii.max() + 1, jj.min(), jj.max() + 1
        self._maps += 1
        fn = f"map-{self._maps:06d}.npz"
        buf = io.BytesIO()
        np.savez_compressed(buf, lo=np.clip(lo[x0:x1, y0:y1] * 40, -127, 127).astype(np.int8),
                            sem=sem[x0:x1, y0:y1], ix0=x0, iy0=y0, origin=gmap.origin, res=gmap.res, n=gmap.n,
                            occ_t=gmap.OCC_T, free_t=gmap.FREE_T)
        (self.root / name / "nav" / fn).write_bytes(buf.getvalue())
        return fn

    def _append(self, name: str, rec: dict) -> None:
        rec = {"wall": round(time.time(), 3), **_round(rec)}
        line = json.dumps(rec, separators=(",", ":")) + "\n"
        with self._lock:
            with open(self.root / name / "nav" / self.jsonl, "a") as f:
                f.write(line)
