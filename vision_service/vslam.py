"""Monocular visual SLAM (ORB-SLAM3) as the global map's pose source.

The robot streams its camera (~10 fps) with, for every frame, the odometry pose and where
the head pointed (POST /v1/nav/frame). Here each frame goes to ORB-SLAM3 in monocular mode,
which tracks the camera against a map of visual features and closes loops, so its poses do
not drift the way wheel odometry does. Three things make its poses usable for a ground robot
with the camera on a pan/tilt head:

  * the head: ORB-SLAM3 tracks the CAMERA, which pans and tilts on the head. The robot's base
    pose is the camera pose with the head's transform (pan, real tilt, mount offset) taken off;
  * scale and level: monocular SLAM has no metric scale and its world frame is wherever the
    first camera looked (tilted). A similarity (scale, rotation, translation; Umeyama) is fitted
    between the SLAM base positions and the odometry positions once the robot has moved enough;
    that turns SLAM into a level, metric "map" frame. Mono-inertial would give scale too, but
    the IMU is on the chassis and the camera on the moving head;
  * the scale from the ToF, not the tracks: the Pi sends the ToF grid nearest each frame. The
    points ORB-SLAM3 tracks in a frame have a depth in SLAM units; where one falls in a ToF zone
    that sees a flat surface (the zone and its four neighbours fit a plane), the zone's metric
    range there, over that depth, is metres per SLAM unit. The zone reports the nearest point in
    its 7.5 deg cone, so the range is first corrected to the point's own direction with the plane.
    A running median of these replaces odometry's scale, which slipping tracks would stretch;
    odometry then only levels and places the map. Until the ToF has enough flat views, or with no
    ToF, the scale comes from odometry as before;
  * the correction: odom_to_map = (base pose in the map) composed with the inverse of the
    odometry pose at the same frame. The Pi stays in its odometry frame; the Mac uses
    odom_to_map to put the Pi's costmap and its plans in the right place.

When tracking is lost the last correction is kept (odometry carries on). If ORB-SLAM3 starts a
new map, the similarity is fitted again, against the map-frame poses, so the map frame stays put.

ORB-SLAM3 is a native library. Install it with deploy/install_orbslam3_macos.sh; until it is,
`backend` is "none" and odom_to_map stays the identity: the global map is then odometry-based.
"""

from __future__ import annotations

import io
import logging
import math
import os
import threading
import time
from collections import OrderedDict, deque
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)

MIN_PAIRS = 30                  # frames with both a SLAM pose and an odometry pose before fitting
MIN_SPREAD_M = 0.8              # ...over at least this much odometry spread
SMOOTH = 0.3                    # how fast odom_to_map follows a new estimate (per tracked frame)
TOF_FOV_DEG = 60.0              # VL53L7CX: 8 x 8 zones over 60 deg, looking along the camera's axis
TOF_MAX_AGE_S = 0.1             # a grid further than this from the frame (either side) is not used
TOF_RANGE_M = (0.3, 3.0)        # ranges trusted for scale
TOF_MIN_POINTS = 4              # flat-zone points a frame needs to give an estimate
TOF_FRAMES = 60                 # the running scale is the median of this many frame estimates...
TOF_MIN_FRAMES = 8              # ...and is used once there are this many, agreeing within
TOF_MAX_SPREAD = 0.10           # this (median absolute deviation over the median)
TOF_REFIT = 0.03                # the map frame is fitted again when the ToF scale moves this much


def umeyama(src: np.ndarray, dst: np.ndarray, scale: Optional[float] = None):
    """Similarity (s, R, t) minimising |dst - (s R src + t)|^2. src, dst: (N, 3). `scale`: known
    (from the ToF): only R and t are fitted."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var = (xs ** 2).sum() / len(src)
    s = float(np.trace(np.diag(D) @ S) / var) if var > 1e-12 else 1.0
    if scale is not None:
        s = float(scale)
    t = mu_d - s * R @ mu_s
    return s, R, t


def _ray(left_deg: float, up_deg: float) -> np.ndarray:
    """Unit direction, camera axes (x right, y down, z ahead), of a ray left_deg left and up_deg up."""
    v = np.array([-math.tan(math.radians(left_deg)), -math.tan(math.radians(up_deg)), 1.0])
    return v / np.linalg.norm(v)


def _nearest(nrm: np.ndarray, left_deg: float, up_deg: float, half_deg: float) -> float:
    """max of nrm . u over a zone's cone (centre left_deg, up_deg, half_deg either way): a zone reads
    the nearest point of a plane in its cone, which is where the cone comes closest to the normal."""
    nl = math.degrees(math.atan2(-nrm[0], nrm[2])) if nrm[2] > 1e-6 else left_deg
    nu = math.degrees(math.atan2(-nrm[1], nrm[2])) if nrm[2] > 1e-6 else up_deg
    return float(nrm @ _ray(min(left_deg + half_deg, max(left_deg - half_deg, nl)),
                            min(up_deg + half_deg, max(up_deg - half_deg, nu))))


def tof_scale_samples(points_w, R_wc: np.ndarray, p_wc: np.ndarray, rows, fov_deg: float = TOF_FOV_DEG) -> list:
    """Metres per SLAM unit, one per tracked map point (world, SLAM units) that falls near the
    centre of a ToF zone seeing a flat surface. rows: the 8 x 8 ranges (m, None = none), row 0 at
    the top, column 0 at the left, as the camera sees them."""
    P = np.asarray(points_w, float).reshape(-1, 3)
    if not len(P) or not rows:
        return []
    cam = (P - np.asarray(p_wc, float).reshape(3)) @ np.asarray(R_wc, float)    # R_wc^T (P - p): camera frame
    n = len(rows)
    zone, mid, half = fov_deg / n, (n - 1) / 2.0, fov_deg / n / 2.0
    planes = {}

    def plane(r: int, c: int):
        """(normal, the zone's centre ray) of the surface zone (r, c) sees, or None if not flat."""
        if (r, c) in planes:
            return planes[(r, c)]
        out = None
        cells = [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        vals = [rows[i][j] for i, j in cells]
        if all(v is not None and TOF_RANGE_M[0] <= v <= TOF_RANGE_M[1] for v in vals):
            angs = [((mid - j) * zone, (mid - i) * zone) for i, j in cells]
            rays = [_ray(lf, u) for lf, u in angs]
            uc = rays[0]
            rng = list(vals)
            for _ in range(4):                           # fit, correct each reading to its centre ray, refit
                X = np.array([v * u for v, u in zip(rng, rays)])
                mu = X.mean(0)
                _, sv, vt = np.linalg.svd(X - mu)
                nrm = vt[2] if vt[2] @ uc >= 0 else -vt[2]
                if nrm @ uc <= 0.3:
                    break
                rng = [v * _nearest(nrm, lf, u, half) / max(1e-6, float(nrm @ ray))
                       for v, (lf, u), ray in zip(vals, angs, rays)]
            if sv[2] / math.sqrt(5) < 0.015 + 0.01 * vals[0] and nrm @ uc > 0.5:   # flat, not grazing
                out = (nrm, uc, (mid - c) * zone, (mid - r) * zone)
        planes[(r, c)] = out
        return out

    out = []
    for x, y, z in cam:
        if z <= 0.05:
            continue
        left, up = math.degrees(math.atan2(-x, z)), math.degrees(math.atan2(-y, z))
        fc, fr = mid - left / zone, mid - up / zone
        c, r = int(round(fc)), int(round(fr))
        if not (1 <= r < n - 1 and 1 <= c < n - 1) or abs(fc - c) > 0.4 or abs(fr - r) > 0.4:
            continue                                     # needs four neighbours, and not at a zone's edge
        pl = plane(r, c)
        if pl is None:
            continue
        nrm, _, lc, uc_deg = pl
        up_ = np.array([x, y, z]) / math.sqrt(x * x + y * y + z * z)
        if nrm @ up_ < 0.5:
            continue
        rng = rows[r][c] * _nearest(nrm, lc, uc_deg, half) / float(nrm @ up_)   # range along the point's direction
        out.append(rng / math.sqrt(x * x + y * y + z * z))
    return out


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])


# camera optical axes (x right, y down, z ahead) as robot axes (x ahead, y left, z up)
R_OPT = np.array([[0, 0, 1.0], [-1.0, 0, 0], [0, -1.0, 0]])


def base_from_camera(R_wc: np.ndarray, p_wc: np.ndarray, pan_deg: float, tilt_deg: float,
                     mount: Sequence[float], scale: float = 1.0):
    """The robot base pose in the SLAM world from the camera's, given the head's pan and REAL
    tilt (degrees above level) and where the head sits on the robot (x, y, z in metres; divided
    by `scale`, the metres per SLAM unit, to be in SLAM units)."""
    R_bc = _rz(math.radians(pan_deg)) @ _ry(-math.radians(tilt_deg)) @ R_OPT
    t_bc = np.asarray(mount, float) / max(scale, 1e-9)
    R_wb = R_wc @ R_bc.T
    return R_wb, p_wc - R_wb @ t_bc


def settings_yaml(path: Path, width: int, height: int, hfov_deg: float, fps: float = 10.0,
                  dist: Sequence[float] = (0.0, 0.0, 0.0, 0.0)) -> Path:
    """An ORB-SLAM3 settings file for the robot's camera. Pinhole from the horizontal field of
    view; replace fx/fy/cx/cy and k1 k2 p1 p2 with a real calibration (the USB HD lens bends)."""
    fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    path.write_text(f"""%YAML:1.0
File.version: "1.0"
Camera.type: "PinHole"
Camera1.fx: {fx:.3f}
Camera1.fy: {fx:.3f}
Camera1.cx: {width / 2.0:.3f}
Camera1.cy: {height / 2.0:.3f}
Camera1.k1: {dist[0]}
Camera1.k2: {dist[1]}
Camera1.p1: {dist[2]}
Camera1.p2: {dist[3]}
Camera.width: {width}
Camera.height: {height}
Camera.fps: {int(fps)}
Camera.RGB: 1
ORBextractor.nFeatures: 1500
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7
Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500.0
""")
    return path


def parse_pose(entry) -> Optional[tuple]:
    """(t, R_wc, p_wc) from one get_trajectory_points() entry: a timestamp and a 3x4 [R | t],
    row by row (ORB_SLAM PythonBindings), or the 9 rotation values then the 3 translation ones."""
    vals = list(entry)
    if len(vals) == 2:                                    # (t, 3x4 nested)
        vals = [vals[0]] + list(np.asarray(vals[1], float).ravel())
    if len(vals) != 13:
        return None
    t, v = float(vals[0]), np.asarray(vals[1:], float)
    m = v.reshape(3, 4)
    R, p = m[:, :3], m[:, 3]
    if abs(np.linalg.det(R) - 1.0) > 0.05 or not np.allclose(R @ R.T, np.eye(3), atol=0.05):
        R, p = v[:9].reshape(3, 3), v[9:]
        if abs(np.linalg.det(R) - 1.0) > 0.05:
            return None
    return t, R, p


class MonoVSLAM:
    def __init__(self, vocab: Optional[str], hfov_deg: float = 80.0, mount=(0.05, -0.0275, 0.22),
                 work_dir: Optional[Path] = None, enabled: bool = True) -> None:
        self.hfov, self.mount = hfov_deg, tuple(mount)
        self.work_dir = Path(work_dir or Path.home() / ".robot-vision" / "vslam")
        self.backend, self.reason = "none", ""
        self._orb = None
        self._sys = None
        self._frames: OrderedDict = OrderedDict()          # t -> (odom pose, pan, tilt)
        self._pairs: deque = deque(maxlen=600)             # (slam R_wc, p_wc, pan, tilt, odom pose in map frame)
        self._sim = None                                   # (s, R, t): SLAM world -> map frame
        self.odom_to_map = (0.0, 0.0, 0.0)
        self.tracking = False
        self.frames_in = self.frames_tracked = self.frames_skipped = 0
        self._tof: deque = deque(maxlen=TOF_FRAMES)        # frame estimates of metres per SLAM unit
        self.tof_scale: Optional[float] = None             # their median, once they agree
        self.scale_source = None                           # "tof" or "odometry": what the fit used
        self.record = lambda kind, **data: None            # nav_record.NavRecorder: what was computed, for replay
        self._last_traj_t, self._last_len = 0.0, 0
        self._slot = None
        self._ev = threading.Event()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        if not enabled:
            self.reason = "disabled (VS_VSLAM=0)"
            return
        # Only look for it here: loading it into this process (beside torch and the venv's OpenCV) aborts
        # the service on a second OpenMP runtime. It runs in a process of its own (_OrbProcess).
        import importlib.util
        if importlib.util.find_spec("orbslam3") is None:
            self.reason = "orbslam3 not installed: run deploy/install_orbslam3_macos.sh"
            return
        if not vocab or not os.path.exists(vocab):
            self.reason = f"ORB vocabulary not found at {vocab!r} (VS_ORB_VOCAB)"
            return
        self.vocab = vocab
        self.backend = "orbslam3"
        threading.Thread(target=self._work, name="vslam", daemon=True).start()

    # ---------------------------------------------------------------- input

    def feed(self, jpeg: bytes, t: float, odom_pose: Optional[Sequence[float]], pan_deg: float,
             tilt_deg: float, tof: Optional[dict] = None) -> None:
        """One streamed frame (tilt_deg: the head's REAL tilt above level; tof: {"rows", "age_s"},
        the ToF grid nearest it). Never blocks: a frame that finds the previous one still being
        tracked replaces it."""
        self.frames_in += 1
        if self.backend == "none" or odom_pose is None:
            return
        if self._slot is not None:                         # the last one was never tracked
            self.frames_skipped += 1
        self._slot = (jpeg, t, tuple(odom_pose), pan_deg, tilt_deg, tof)
        self._ev.set()

    def state(self) -> dict:
        return {"backend": self.backend, "reason": self.reason or None, "tracking": self.tracking,
                "scaled": self._sim is not None, "scale_m_per_unit": None if self._sim is None else round(self._sim[0], 4),
                "scale_source": self.scale_source, "tof_scale": None if self.tof_scale is None else round(self.tof_scale, 4),
                "tof_frames": len(self._tof),
                "odom_to_map": [round(v, 4) for v in self.odom_to_map],
                "frames_in": self.frames_in, "frames_tracked": self.frames_tracked}

    # ----------------------------------------------------------------- work

    def _work(self) -> None:
        while not self._stop.is_set():
            self._ev.wait(1.0)
            self._ev.clear()
            item, self._slot = self._slot, None
            if item is None:
                continue
            try:
                self._track(*item)
            except Exception as exc:                       # noqa: BLE001
                log.warning("vslam: %s: %s", type(exc).__name__, exc)
                if "process exited" in str(exc) or "did not start" in str(exc):   # gone: carry on without it
                    self.backend, self.reason, self.tracking = "none", f"ORB-SLAM3 stopped: {exc}", False
                    self.record("slam_stopped", reason=str(exc))
                    return

    def _start(self, w: int, h: int) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        cfg = settings_yaml(self.work_dir / "camera.yaml", w, h, self.hfov)
        if os.environ.get("VS_VSLAM_INPROC") == "1":       # tests (a stand-in module): in this process
            import orbslam3
            sys_ = orbslam3.System(self.vocab, str(cfg), orbslam3.Sensor.MONOCULAR)
            if hasattr(sys_, "set_use_viewer"):
                sys_.set_use_viewer(False)
            sys_.initialize()
        else:
            sys_ = _OrbProcess(self.vocab, str(cfg))
        self._sys = sys_
        log.info("vslam: ORB-SLAM3 monocular started (%dx%d, hfov %.0f)", w, h, self.hfov)

    def _track(self, jpeg: bytes, t: float, odom, pan: float, tilt: float, tof: Optional[dict] = None) -> None:
        from PIL import Image
        img = np.asarray(Image.open(io.BytesIO(jpeg)).convert("L"))
        if self._sys is None:
            self._start(img.shape[1], img.shape[0])
        self._frames[round(t, 4)] = (odom, pan, tilt)
        while len(self._frames) > 2000:
            self._frames.popitem(last=False)
        self._sys.process_image_mono(img, t)
        st = self._sys.get_tracking_state() if hasattr(self._sys, "get_tracking_state") else None
        self.tracking = st is None or "OK" in str(st).upper() or st == 2
        if self.tracking:
            self.frames_tracked += 1
        self.record("slam_frame", t=t, odom=list(odom), pan=pan, tilt=tilt, state=str(st), tracking=self.tracking,
                    frames_in=self.frames_in, tracked=self.frames_tracked, skipped=self.frames_skipped)
        now = time.monotonic()
        if now - self._last_traj_t < 0.5:                 # the trajectory grows: read it twice a second
            return
        self._last_traj_t = now
        traj = list(self._sys.get_trajectory_points())
        if len(traj) < self._last_len * 0.5:              # it started a new map: fit the frame again
            log.info("vslam: new SLAM map (trajectory %d -> %d): refitting", self._last_len, len(traj))
            self.record("slam_newmap", before=self._last_len, after=len(traj))
            self._sim, self._pairs = None, deque(maxlen=600)
            self._tof.clear()                             # a new map has a new scale
            self.tof_scale = None
        self._last_len = len(traj)
        if tof and self.tracking and traj:
            self._tof_step(t, tof, traj[-1])
        self.update_from(traj)

    def update_from(self, traj) -> None:
        """Fold SLAM poses (get_trajectory_points() entries) into the fit and the correction."""
        seen = {q[0] for q in self._pairs}
        for e in traj[-40:]:
            parsed = parse_pose(e)
            if parsed is None:
                continue
            t, R, p = parsed
            key = round(t, 4)
            if key in seen or key not in self._frames:
                continue
            odom, pan, tilt = self._frames[key]
            in_map = _compose(self.odom_to_map, odom)
            self._pairs.append((key, R, p, pan, tilt, in_map, odom))
            self.record("slam_pose", t=key, R_wc=np.asarray(R).reshape(-1), p_wc=np.asarray(p).reshape(-1), pan=pan,
                        tilt=tilt, odom=list(odom), odom_in_map=list(in_map),
                        base_in_map=list(self.base_in_map(R, p, pan, tilt)) if self._sim is not None else None)
        if len(self._pairs) < MIN_PAIRS:
            return
        if self._sim is None:
            self._fit()
            if self._sim is None:
                return
        key, R, p, pan, tilt, _, odom = self._pairs[-1]
        est = self.base_in_map(R, p, pan, tilt)
        new = _compose(est, _inverse(odom))
        before = self.odom_to_map
        with self._lock:
            self.odom_to_map = _blend(self.odom_to_map, new, SMOOTH)
        self.record("slam_correction", t=key, base_in_map=list(est), odom=list(odom), target=list(new),
                    before=list(before), after=list(self.odom_to_map), smooth=SMOOTH)

    def _tof_step(self, t: float, tof: dict, last) -> None:
        """Metres per SLAM unit from this frame's tracked points and ToF grid (see the module notes)."""
        parsed = parse_pose(last)
        age = float(tof.get("age_s") or 0.0)
        if parsed is None or abs(parsed[0] - t) > 1e-3 or abs(age) > TOF_MAX_AGE_S:
            return                                        # not this frame's pose, or the grid too far off
        pts = list(self._sys.get_tracked_mappoints()) if hasattr(self._sys, "get_tracked_mappoints") else []
        samples = tof_scale_samples(pts, parsed[1], parsed[2], tof.get("rows"))
        est = float(np.median(samples)) if len(samples) >= TOF_MIN_POINTS else None
        if est is not None:
            self._tof.append(est)
        spread, prev = None, self.tof_scale
        if len(self._tof) >= TOF_MIN_FRAMES:
            vals = np.array(self._tof)
            med = float(np.median(vals))
            spread = float(np.median(np.abs(vals - med)) / med)
            self.tof_scale = med if spread < TOF_MAX_SPREAD else None
        self.record("slam_tof_scale", t=t, age_s=age, points=len(pts), samples=len(samples),
                    ratios=sorted(samples)[:: max(1, len(samples) // 40)], frame_scale=est, frames=len(self._tof),
                    scale=self.tof_scale, spread=spread)
        if self.tof_scale is not None and self._sim is not None and (
                self.scale_source != "tof" or abs(self.tof_scale / self._sim[0] - 1.0) > TOF_REFIT):
            log.info("vslam: ToF scale %.3f m/unit (was %.3f, %s): refitting the map frame",
                     self.tof_scale, self._sim[0], self.scale_source)
            self._sim = None
            self._fit()
        elif prev is not None and self.tof_scale is None:
            log.info("vslam: ToF scale estimates disagree (spread %.2f): not using them", spread or 0.0)

    def _fit(self) -> None:
        pairs = list(self._pairs)
        dst = np.array([[q[5][0], q[5][1], 0.0] for q in pairs])
        if np.ptp(dst[:, 0]) + np.ptp(dst[:, 1]) < MIN_SPREAD_M:
            return
        known = self.tof_scale                              # metric scale from the ToF, if it has one
        scale = known or 1.0
        for _ in range(1 if known else 2):                  # base positions need the scale: iterate once
            src = np.array([base_from_camera(q[1], q[2], q[3], q[4], self.mount, scale)[1] for q in pairs])
            s, R, t = umeyama(src, dst, known)
            scale = s
        self.scale_source = "tof" if known else "odometry"
        self._sim = (s, R, t)
        res = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1)
        log.info("vslam: map frame fitted: %.3f m per SLAM unit (%s), residual %.3f m (n=%d)", s, self.scale_source,
                 float(res.mean()), len(pairs))
        self.record("slam_fit", scale=float(s), scale_source=self.scale_source, R=R, t=t, residual_mean=float(res.mean()), residual_max=float(res.max()),
                    n=len(pairs), spread_m=float(np.ptp(dst[:, 0]) + np.ptp(dst[:, 1])), slam_base=src,
                    odom_xy=dst[:, :2], residuals=res)

    def base_in_map(self, R_wc: np.ndarray, p_wc: np.ndarray, pan: float, tilt: float) -> tuple:
        s, Rs, ts = self._sim
        R_wb, p_wb = base_from_camera(R_wc, p_wc, pan, tilt, self.mount, s)
        p = s * Rs @ p_wb + ts
        fwd = Rs @ R_wb @ np.array([1.0, 0.0, 0.0])
        return float(p[0]), float(p[1]), math.atan2(fwd[1], fwd[0])


def _plain(v):
    """A binding's result as plain Python, to cross the pipe (an enum becomes its name, or its number)."""
    if v is None or type(v) in (bool, int, float, str):      # exact types: an enum is an int subclass, and
        return v                                            # unpickling it would load the binding here
    if isinstance(v, (list, tuple)) or (hasattr(v, "__iter__") and not hasattr(v, "keys")):
        return [_plain(x) for x in v]
    if isinstance(v, (int, float)):                         # an enum or a numpy scalar: plain number
        return int(v) if isinstance(v, int) else float(v)
    text = str(v)
    try:
        return int(v) if not any(ch.isalpha() for ch in text) else text.rsplit(".", 1)[-1]
    except (TypeError, ValueError):
        return text


def _orb_child(conn, vocab: str, cfg: str) -> None:
    """The ORB-SLAM3 process: builds the System, then answers calls on it until told to stop."""
    import orbslam3
    sys_ = orbslam3.System(vocab, cfg, orbslam3.Sensor.MONOCULAR)
    if hasattr(sys_, "set_use_viewer"):
        sys_.set_use_viewer(False)
    sys_.initialize()
    conn.send(("ready", None))
    while True:
        name, args = conn.recv()
        if name == "shutdown":
            sys_.shutdown()
            conn.send(("ok", None))
            return
        try:
            conn.send(("ok", _plain(getattr(sys_, name)(*args))))
        except Exception as exc:                           # noqa: BLE001 — reported in the service's log
            conn.send(("err", f"{type(exc).__name__}: {exc}"))


class _OrbProcess:
    """ORB-SLAM3 in a process of its own, called like the binding's System. Loaded into the service
    itself, its OpenMP (Homebrew's, through OpenCV 4) and torch's own clash ("OMP: Error #15") and the
    service aborts at start; apart, neither sees the other. A crash there stops visual SLAM, not the service."""

    METHODS = ("process_image_mono", "get_tracking_state", "get_trajectory_points", "get_tracked_mappoints")

    def __init__(self, vocab: str, cfg: str, timeout_s: float = 120.0) -> None:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")                      # a fresh interpreter: none of torch's libraries
        self._conn, child = ctx.Pipe()
        self._proc = ctx.Process(target=_orb_child, args=(child, vocab, cfg), name="orbslam3", daemon=True)
        self._proc.start()
        self._lock = threading.Lock()
        end = time.monotonic() + timeout_s                 # the vocabulary takes a few seconds to load
        while not self._conn.poll(0.2):
            if not self._proc.is_alive() or time.monotonic() > end:
                raise RuntimeError(f"ORB-SLAM3 process did not start (exit code {self._proc.exitcode})")
        self._conn.recv()

    def _call(self, name: str, *args):
        with self._lock:
            try:
                self._conn.send((name, args))
                kind, value = self._conn.recv()
            except (EOFError, OSError, BrokenPipeError) as exc:
                raise RuntimeError(f"ORB-SLAM3 process exited (code {self._proc.exitcode})") from exc
        if kind == "err":
            raise RuntimeError(value)
        return value

    def __getattr__(self, name: str):
        if name in self.METHODS:
            return lambda *args: self._call(name, *args)
        raise AttributeError(name)

    def shutdown(self) -> None:
        try:
            self._call("shutdown")
        finally:
            self._proc.join(timeout=5.0)


def _compose(a, b) -> tuple:
    c, s = math.cos(a[2]), math.sin(a[2])
    return a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], math.atan2(math.sin(a[2] + b[2]), math.cos(a[2] + b[2]))


def _inverse(a) -> tuple:
    c, s = math.cos(a[2]), math.sin(a[2])
    return -c * a[0] - s * a[1], s * a[0] - c * a[1], -a[2]


def _blend(a, b, k: float) -> tuple:
    d = math.atan2(math.sin(b[2] - a[2]), math.cos(b[2] - a[2]))
    return a[0] + k * (b[0] - a[0]), a[1] + k * (b[1] - a[1]), a[2] + k * d
