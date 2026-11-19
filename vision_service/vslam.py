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


def umeyama(src: np.ndarray, dst: np.ndarray):
    """Similarity (s, R, t) minimising |dst - (s R src + t)|^2. src, dst: (N, 3)."""
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
    t = mu_d - s * R @ mu_s
    return s, R, t


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
        self.frames_in = self.frames_tracked = 0
        self._last_traj_t, self._last_len = 0.0, 0
        self._slot = None
        self._ev = threading.Event()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        if not enabled:
            self.reason = "disabled (VS_VSLAM=0)"
            return
        try:
            import orbslam3                                # noqa: F401
            self._orb = orbslam3
        except Exception as exc:                           # noqa: BLE001
            self.reason = f"orbslam3 not installed ({type(exc).__name__}): run deploy/install_orbslam3_macos.sh"
            return
        if not vocab or not os.path.exists(vocab):
            self.reason = f"ORB vocabulary not found at {vocab!r} (VS_ORB_VOCAB)"
            return
        self.vocab = vocab
        self.backend = "orbslam3"
        threading.Thread(target=self._work, name="vslam", daemon=True).start()

    # ---------------------------------------------------------------- input

    def feed(self, jpeg: bytes, t: float, odom_pose: Optional[Sequence[float]], pan_deg: float,
             tilt_deg: float) -> None:
        """One streamed frame (tilt_deg: the head's REAL tilt above level). Never blocks: a frame
        that finds the previous one still being tracked replaces it."""
        self.frames_in += 1
        if self.backend == "none" or odom_pose is None:
            return
        self._slot = (jpeg, t, tuple(odom_pose), pan_deg, tilt_deg)
        self._ev.set()

    def state(self) -> dict:
        return {"backend": self.backend, "reason": self.reason or None, "tracking": self.tracking,
                "scaled": self._sim is not None, "scale_m_per_unit": None if self._sim is None else round(self._sim[0], 4),
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

    def _start(self, w: int, h: int) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        cfg = settings_yaml(self.work_dir / "camera.yaml", w, h, self.hfov)
        sys_ = self._orb.System(self.vocab, str(cfg), self._orb.Sensor.MONOCULAR)
        if hasattr(sys_, "set_use_viewer"):
            sys_.set_use_viewer(False)
        sys_.initialize()
        self._sys = sys_
        log.info("vslam: ORB-SLAM3 monocular started (%dx%d, hfov %.0f)", w, h, self.hfov)

    def _track(self, jpeg: bytes, t: float, odom, pan: float, tilt: float) -> None:
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
        now = time.monotonic()
        if now - self._last_traj_t < 0.5:                 # the trajectory grows: read it twice a second
            return
        self._last_traj_t = now
        traj = list(self._sys.get_trajectory_points())
        if len(traj) < self._last_len * 0.5:              # it started a new map: fit the frame again
            log.info("vslam: new SLAM map (trajectory %d -> %d): refitting", self._last_len, len(traj))
            self._sim, self._pairs = None, deque(maxlen=600)
        self._last_len = len(traj)
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
        if len(self._pairs) < MIN_PAIRS:
            return
        if self._sim is None:
            self._fit()
            if self._sim is None:
                return
        key, R, p, pan, tilt, _, odom = self._pairs[-1]
        est = self.base_in_map(R, p, pan, tilt)
        new = _compose(est, _inverse(odom))
        with self._lock:
            self.odom_to_map = _blend(self.odom_to_map, new, SMOOTH)

    def _fit(self) -> None:
        pairs = list(self._pairs)
        dst = np.array([[q[5][0], q[5][1], 0.0] for q in pairs])
        if np.ptp(dst[:, 0]) + np.ptp(dst[:, 1]) < MIN_SPREAD_M:
            return
        scale = 1.0
        for _ in range(2):                                  # base positions need the scale: iterate once
            src = np.array([base_from_camera(q[1], q[2], q[3], q[4], self.mount, scale)[1] for q in pairs])
            s, R, t = umeyama(src, dst)
            scale = s
        self._sim = (s, R, t)
        res = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1)
        log.info("vslam: map frame fitted: %.3f m per SLAM unit, residual %.3f m (n=%d)", s, float(res.mean()), len(pairs))

    def base_in_map(self, R_wc: np.ndarray, p_wc: np.ndarray, pan: float, tilt: float) -> tuple:
        s, Rs, ts = self._sim
        R_wb, p_wb = base_from_camera(R_wc, p_wc, pan, tilt, self.mount, s)
        p = s * Rs @ p_wb + ts
        fwd = Rs @ R_wb @ np.array([1.0, 0.0, 0.0])
        return float(p[0]), float(p[1]), math.atan2(fwd[1], fwd[0])


def _compose(a, b) -> tuple:
    c, s = math.cos(a[2]), math.sin(a[2])
    return a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], math.atan2(math.sin(a[2] + b[2]), math.cos(a[2] + b[2]))


def _inverse(a) -> tuple:
    c, s = math.cos(a[2]), math.sin(a[2])
    return -c * a[0] - s * a[1], s * a[0] - c * a[1], -a[2]


def _blend(a, b, k: float) -> tuple:
    d = math.atan2(math.sin(b[2] - a[2]), math.cos(b[2] - a[2]))
    return a[0] + k * (b[0] - a[0]), a[1] + k * (b[1] - a[1]), a[2] + k * d
