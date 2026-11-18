"""Keyframe SLAM for a slow robot with a single-beam rangefinder.

Why this runs on the Mac and why it can be this simple
------------------------------------------------------
The robot has no 2-D lidar. It has a TF-Luna on the pan/tilt head, so a
"scan" is the head sweeping ±45° while the base turns in 90° steps: a sparse
360° ring of ranges taken over ~20 s, at one height. That is far too slow for
a real-time SLAM package — but the environment is static and nobody is in a
hurry, so the problem becomes *batch* SLAM on keyframes:

    robot stops ─► sweeps ─► POST keyframe (odom pose + rays)
                                   │
       Mac: predict pose from odometry, correct it by matching the rays
            against the map so far, fold the rays into the map, find the
            nearest unexplored edge, plan a path to it
                                   │
    robot drives the path ◄── pose correction + next goal

Everything is numpy on a ~500x500 occupancy grid: a keyframe costs tens of
milliseconds, against ~20 s for the sweep that produced it.

Pieces (each small, each tested in tests/test_slam.py):
  * `OccupancyGrid`  — log-odds grid; rays carve free space, endpoints mark walls
  * `match_scan`     — correlative scan matcher: brute-force search of
                       (dx, dy, dθ) around the odometry prediction, scoring
                       ray endpoints against a blurred copy of the walls
  * `plan_path`      — BFS over free, obstacle-inflated cells, then shortcut
  * `frontiers`      — free cells next to unknown ones, clustered
  * `viewpoints`     — N reachable poses around a target, facing it

With the 8x8 ToF head the robot sends more than wall rays, and all of it is kept:
  * `clear`  — rays along which the floor was seen and nothing stood: free space
               with no wall at the end (the wall is beyond the sensor's range)
  * `low`    — things too low for a level beam (a step, a box): never driven over
  * `drops`  — where the floor falls away (stairs down): never driven near
  * `cloud`  — every reading as an (x, y, height) point: the room in 3-D, saved
               with the map

Loop closure (`_close_loop`): drift between keyframes is corrected by matching
against the map, which stops it growing without bound in a room, but a long loop
can still come back a little off and leave a seam (a wall drawn twice) - which
shrinks or stretches the room. So when a keyframe is back near one taken at least
`loop_gap` keyframes and `loop_min_travel` metres ago, it is matched twice: against
the last few keyframes only (where the drifted chain says the robot is) and against
the old keyframes only (where it really is). If the old map agrees well enough and
the two differ, the difference is spread along the keyframes in between (none for
the old one, all of it for this one) and the map is rebuilt from every keyframe's
stored readings at the corrected poses.

Paths keep clear of walls where they can (`pref_clearance`) and come with
`clear_m`: how close each leg passes to anything mapped, for the robot to size its
legs. `walls()` fits straight lines to the wall cells: wall lengths, and the
explored area's size along its main directions.
"""

from __future__ import annotations

import json
import logging
import math
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)

L_FREE, L_OCC, L_MIN, L_MAX = -0.4, 0.85, -4.0, 4.0
OCC_T, FREE_T = 0.5, -0.6     # one hit is enough to call a wall: scans are sparse


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def compose(a: Sequence[float], b: Sequence[float]) -> tuple:
    c, s = math.cos(a[2]), math.sin(a[2])
    return (a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], wrap(a[2] + b[2]))


def between(a: Sequence[float], b: Sequence[float]) -> tuple:
    """Motion from a to b, in a's frame."""
    c, s = math.cos(a[2]), math.sin(a[2])
    dx, dy = b[0] - a[0], b[1] - a[1]
    return (c * dx + s * dy, -s * dx + c * dy, wrap(b[2] - a[2]))


def transform(pose: Sequence[float], pts: np.ndarray) -> np.ndarray:
    c, s = math.cos(pose[2]), math.sin(pose[2])
    r = np.array([[c, -s], [s, c]])
    return pts @ r.T + np.array(pose[:2])


# --------------------------------------------------------------------------
# grid
# --------------------------------------------------------------------------

class OccupancyGrid:
    def __init__(self, size_m: float = 24.0, res: float = 0.05) -> None:
        self.res = res
        self.n = int(round(size_m / res))
        self.origin = -size_m / 2.0          # world coord of cell (0, 0)'s corner
        self.logodds = np.zeros((self.n, self.n), np.float32)   # [iy, ix]

    def cells(self, xy: np.ndarray) -> np.ndarray:
        return np.floor((np.asarray(xy) - self.origin) / self.res).astype(np.int64)

    def world(self, ij: np.ndarray) -> np.ndarray:
        return (np.asarray(ij) + 0.5) * self.res + self.origin

    def inside(self, ij: np.ndarray) -> np.ndarray:
        return (ij[..., 0] >= 0) & (ij[..., 0] < self.n) & (ij[..., 1] >= 0) & (ij[..., 1] < self.n)

    def occupied(self) -> np.ndarray:
        return self.logodds > OCC_T

    def free(self) -> np.ndarray:
        return self.logodds < FREE_T

    def unknown(self) -> np.ndarray:
        return ~(self.occupied() | self.free())

    def _bump(self, ij: np.ndarray, delta: float) -> None:
        ij = ij[self.inside(ij)]
        if len(ij):
            ij = np.unique(ij, axis=0)
            self.logodds[ij[:, 1], ij[:, 0]] = np.clip(
                self.logodds[ij[:, 1], ij[:, 0]] + delta, L_MIN, L_MAX)

    def integrate(self, origin_xy: np.ndarray, hits_xy: np.ndarray, max_range: float) -> None:
        """origin_xy: (n,2) sensor positions; hits_xy: (n,2) ray endpoints (world).
        Rays shorter than max_range end in an obstacle."""
        if not len(hits_xy):
            return
        free = []
        occ = []
        step = self.res / 2.0
        for o, h in zip(origin_xy, hits_xy):
            d = h - o
            length = float(np.hypot(*d))
            if length < 1e-6:
                continue
            ts = np.arange(0.0, max(0.0, length - self.res), step)
            free.append(self.cells(o + np.outer(ts / length, d)))
            if length < max_range - 1e-6:
                occ.append(self.cells(h[None, :]))
        if free:
            fc = np.concatenate(free)
            if occ:
                # Never carve through the endpoint cells of this same scan.
                oc = np.concatenate(occ)
                key = lambda a: a[:, 0] * (self.n + 7) + a[:, 1]
                fc = fc[~np.isin(key(fc), key(oc))]
            self._bump(fc, L_FREE)
        if occ:
            self._bump(np.concatenate(occ), L_OCC)

    def mark_free_disc(self, xy: Sequence[float], radius: float) -> None:
        """The robot is standing here, so it is certainly traversable."""
        c = self.cells(np.array(xy)[None, :])[0]
        r = int(math.ceil(radius / self.res))
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        m = xx ** 2 + yy ** 2 <= r * r
        ij = np.stack([xx[m] + c[0], yy[m] + c[1]], axis=1)
        ij = ij[self.inside(ij)]
        self.logodds[ij[:, 1], ij[:, 0]] = np.minimum(self.logodds[ij[:, 1], ij[:, 0]], FREE_T - 0.4)

    def mark_occupied(self, xy: np.ndarray) -> None:
        if len(xy):
            self._bump(self.cells(np.asarray(xy, float).reshape(-1, 2)), L_OCC * 2)


def _blur(mask: np.ndarray, sigma_cells: float) -> np.ndarray:
    r = max(1, int(3 * sigma_cells))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_cells) ** 2).astype(np.float32)
    out = mask.astype(np.float32)
    out = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, out)
    out = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, out)
    m = out.max()
    return out / m if m > 0 else out


def _dilate(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    out = mask.copy()
    r = radius_cells
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy > r * r or (dx == 0 and dy == 0):
                continue
            shifted = np.zeros_like(mask)
            ys = slice(max(dy, 0), mask.shape[0] + min(dy, 0))
            yd = slice(max(-dy, 0), mask.shape[0] + min(-dy, 0))
            xs = slice(max(dx, 0), mask.shape[1] + min(dx, 0))
            xd = slice(max(-dx, 0), mask.shape[1] + min(-dx, 0))
            shifted[ys, xs] = mask[yd, xd]
            out |= shifted
    return out


# --------------------------------------------------------------------------
# scan matching
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class MatchResult:
    pose: tuple
    score: float          # mean likelihood of endpoints, 0..1
    accepted: bool


def match_scan(grid: OccupancyGrid, field_: np.ndarray, prior: Sequence[float],
               pts_local: np.ndarray, xy_window: float = 0.4, th_window: float = math.radians(12),
               th_step: float = math.radians(1), min_score: float = 0.35,
               prior_weight: float = 0.08) -> MatchResult:
    """Correlative search around `prior` for the pose that puts the scan's
    endpoints on the walls already in the map.

    `field_` is the blurred wall map (see `_blur`); a small penalty on distance
    from the odometry prior keeps symmetric rooms from pulling the pose to a
    look-alike spot.
    """
    if len(pts_local) < 15 or field_.max() <= 0:
        return MatchResult(tuple(prior), 0.0, False)
    k = int(round(xy_window / grid.res))
    offs = np.stack(np.meshgrid(np.arange(-k, k + 1), np.arange(-k, k + 1)), -1).reshape(-1, 2)
    best = (-1e9, 0.0, tuple(prior))
    for dth in np.arange(-th_window, th_window + 1e-9, th_step):
        pose = (prior[0], prior[1], wrap(prior[2] + dth))
        base = grid.cells(transform(pose, pts_local))                 # (n,2)
        idx = base[None, :, :] + offs[:, None, :]                      # (m,n,2)
        ok = grid.inside(idx)
        ix = np.clip(idx[..., 0], 0, grid.n - 1)
        iy = np.clip(idx[..., 1], 0, grid.n - 1)
        vals = np.where(ok, field_[iy, ix], 0.0).mean(axis=1)          # (m,)
        pen = prior_weight * ((np.hypot(offs[:, 0], offs[:, 1]) * grid.res / xy_window) ** 2
                              + (dth / th_window) ** 2)
        j = int(np.argmax(vals - pen))
        if vals[j] - pen[j] > best[0]:
            dx, dy = offs[j] * grid.res
            best = (vals[j] - pen[j], float(vals[j]), (prior[0] + dx, prior[1] + dy, pose[2]))
    _, score, pose = best
    return MatchResult(pose, round(score, 3), score >= min_score)


# --------------------------------------------------------------------------
# planning
# --------------------------------------------------------------------------

_N8 = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]


def traversable(grid: OccupancyGrid, robot_radius: float) -> np.ndarray:
    blocked = _dilate(grid.occupied(), int(math.ceil(robot_radius / grid.res)))
    return grid.free() & ~blocked


def _bfs(trav: np.ndarray, start: tuple, goal_mask: Optional[np.ndarray] = None,
         goal: Optional[tuple] = None) -> tuple:
    """8-connected BFS from `start` (ix, iy). Returns (reached_cell, parents) or (None, parents)."""
    n_y, n_x = trav.shape
    parent = -np.ones(trav.shape, np.int64)
    sx, sy = start
    if not (0 <= sx < n_x and 0 <= sy < n_y):
        return None, parent
    parent[sy, sx] = sy * n_x + sx
    q = deque([(sx, sy)])
    while q:
        x, y = q.popleft()
        if (goal is not None and (x, y) == goal) or (goal_mask is not None and goal_mask[y, x]):
            return (x, y), parent
        for dx, dy in _N8:
            nx, ny = x + dx, y + dy
            if 0 <= nx < n_x and 0 <= ny < n_y and parent[ny, nx] < 0 and trav[ny, nx]:
                parent[ny, nx] = y * n_x + x
                q.append((nx, ny))
    return None, parent


def _unwind(parent: np.ndarray, end: tuple) -> list:
    n_x = parent.shape[1]
    x, y = end
    out = [(x, y)]
    while True:
        p = int(parent[y, x])
        px, py = p % n_x, p // n_x
        if (px, py) == (x, y):
            break
        x, y = px, py
        out.append((x, y))
    return out[::-1]


def _line_clear(trav: np.ndarray, a: tuple, b: tuple) -> bool:
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) * 2 + 1
    xs = np.rint(np.linspace(a[0], b[0], n)).astype(int)
    ys = np.rint(np.linspace(a[1], b[1], n)).astype(int)
    return bool(trav[ys, xs].all())


def _shortcut(trav: np.ndarray, cells: list) -> list:
    """Greedy line-of-sight smoothing: fewest straight segments, i.e. fewest
    turn-then-drive steps for the robot."""
    if len(cells) <= 2:
        return cells
    out = [cells[0]]
    i = 0
    while i < len(cells) - 1:
        j = len(cells) - 1
        while j > i + 1 and not _line_clear(trav, cells[i], cells[j]):
            j -= 1
        out.append(cells[j])
        i = j
    return out


def _start_cell(grid: OccupancyGrid, trav: np.ndarray, xy: Sequence[float]) -> Optional[tuple]:
    """Robot's cell, or the nearest traversable one (it may be parked close to
    a wall, inside the inflation)."""
    c = grid.cells(np.array(xy[:2])[None, :])[0]
    for r in range(0, 8):
        ys, xs = np.nonzero(trav[max(0, c[1] - r):c[1] + r + 1, max(0, c[0] - r):c[0] + r + 1])
        if len(xs):
            k = int(np.argmin((xs + max(0, c[0] - r) - c[0]) ** 2 + (ys + max(0, c[1] - r) - c[1]) ** 2))
            return (int(xs[k] + max(0, c[0] - r)), int(ys[k] + max(0, c[1] - r)))
    return None


def plan_path(grid: OccupancyGrid, trav: np.ndarray, start_xy, goal_xy) -> Optional[list]:
    s = _start_cell(grid, trav, start_xy)
    g = _start_cell(grid, trav, goal_xy)
    if s is None or g is None:
        return None
    end, parent = _bfs(trav, s, goal=g)
    if end is None:
        return None
    cells = _shortcut(trav, _unwind(parent, end))
    return [tuple(map(float, grid.world(np.array(c)))) for c in cells]


def frontiers(grid: OccupancyGrid, min_cells: int = 6) -> list:
    """Clusters of free cells that touch unknown space: list of (n_cells, [ix, iy] cells)."""
    free, unk = grid.free(), grid.unknown()
    touch = np.zeros_like(free)
    touch[1:, :] |= unk[:-1, :]
    touch[:-1, :] |= unk[1:, :]
    touch[:, 1:] |= unk[:, :-1]
    touch[:, :-1] |= unk[:, 1:]
    front = free & touch
    seen = np.zeros_like(front)
    out = []
    for y, x in zip(*np.nonzero(front)):
        if seen[y, x]:
            continue
        comp, q = [], deque([(x, y)])
        seen[y, x] = True
        while q:
            cx, cy = q.popleft()
            comp.append((cx, cy))
            for dx, dy in _N8:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < grid.n and 0 <= ny < grid.n and front[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    q.append((nx, ny))
        if len(comp) >= min_cells:
            out.append((len(comp), comp))
    return out


def fit_walls(xy: np.ndarray, res: float, min_len: float = 0.5, gap: float = 0.25,
              max_walls: int = 40) -> list:
    """Straight walls through wall points (x, y): a Hough vote for the strongest line,
    its points (within one cell) split where there is a gap, each run of at least
    min_len kept; repeat on what is left. Returns [(x1, y1, x2, y2), ...], longest first."""
    pts = np.asarray(xy, float).reshape(-1, 2)
    thetas = np.radians(np.arange(0.0, 180.0, 1.0))
    cs, sn = np.cos(thetas), np.sin(thetas)
    out = []
    while len(pts) >= max(3, int(min_len / res)) and len(out) < max_walls:
        rho = pts[:, :1] * cs + pts[:, 1:] * sn                        # (n, 180)
        q = np.round(rho / res).astype(np.int64)
        best = (0, 0, 0)
        for j in range(len(thetas)):
            v, c = np.unique(q[:, j], return_counts=True)
            i = int(np.argmax(c))
            if c[i] > best[0]:
                best = (int(c[i]), j, int(v[i]))
        votes, j, r = best
        if votes < min_len / res * 0.6:
            break
        on = np.abs(rho[:, j] - r * res) <= res * 1.0
        line = pts[on]
        d = np.array([-sn[j], cs[j]])                                  # along the line
        t = np.sort(line @ d)
        cuts = np.nonzero(np.diff(t) > gap)[0]
        p0 = np.array([cs[j], sn[j]]) * r * res
        for seg in np.split(t, cuts + 1):
            if seg[-1] - seg[0] >= min_len:
                a, b = p0 + d * seg[0], p0 + d * seg[-1]
                out.append((float(a[0]), float(a[1]), float(b[0]), float(b[1])))
        pts = pts[~on]
    return sorted(out, key=lambda w: -math.hypot(w[2] - w[0], w[3] - w[1]))


def extent(walls: list) -> Optional[dict]:
    """Size of what the walls enclose along its main directions: the length-weighted
    wall angle (folded to 90 degrees) and the spread of the wall ends across it."""
    if not walls:
        return None
    w = np.array(walls, float)
    ln = np.hypot(w[:, 2] - w[:, 0], w[:, 3] - w[:, 1])
    ang = np.arctan2(w[:, 3] - w[:, 1], w[:, 2] - w[:, 0])
    a = math.atan2(float((ln * np.sin(4 * ang)).sum()), float((ln * np.cos(4 * ang)).sum())) / 4.0
    ends = np.concatenate([w[:, :2], w[:, 2:]])
    u = ends @ np.array([math.cos(a), math.sin(a)])
    v = ends @ np.array([-math.sin(a), math.cos(a)])
    dims = sorted([float(u.max() - u.min()), float(v.max() - v.min())], reverse=True)
    return {"length_m": round(dims[0], 2), "width_m": round(dims[1], 2), "angle_deg": round(math.degrees(a), 1)}


def in_scope(start: Sequence[float], radius_m: Optional[float], no_behind: bool, xy: np.ndarray) -> np.ndarray:
    """Which points (N x 2, map frame) the run may explore: within `radius_m` of
    the start pose (None / <= 0 = no limit) and, with `no_behind`, on the side
    the start pose faces (the half-plane ahead of it)."""
    d = np.asarray(xy, float).reshape(-1, 2) - np.asarray(start[:2], float)
    ok = np.ones(len(d), bool)
    if radius_m and radius_m > 0:
        ok &= np.hypot(d[:, 0], d[:, 1]) <= radius_m
    if no_behind:
        ok &= d[:, 0] * math.cos(start[2]) + d[:, 1] * math.sin(start[2]) >= 0.0
    return ok


# --------------------------------------------------------------------------
# the session
# --------------------------------------------------------------------------

@dataclass
class SlamConfig:
    size_m: float = 24.0
    res: float = 0.05
    max_range: float = 7.5          # TF-Luna is good to ~8 m indoors
    robot_radius: float = 0.18
    clearance: float = 0.05         # extra margin on top of the radius
    min_frontier_cells: int = 6
    max_cloud_points: int = 400_000   # 3-D points kept (one per 5 cm cube), oldest dropped first
    pref_clearance: float = 0.15    # paths keep this much more room where the map allows it
    loop_closure: bool = True
    loop_gap: int = 5               # keyframes between this one and an old one before it is a loop
    loop_radius: float = 1.5        # m: an old keyframe this near counts as a revisit
    loop_min_travel: float = 3.0    # m driven since that old keyframe
    loop_window_m: float = 0.8      # the loop match searches this far...
    loop_window_deg: float = 20.0   # ...and this far round
    loop_min_score: float = 0.5     # and needs at least this good a fit (0..1)


@dataclass
class Keyframe:
    kf_id: int
    odom: tuple
    pose: tuple
    score: float
    accepted: bool


@dataclass
class SlamSession:
    config: SlamConfig = field(default_factory=SlamConfig)
    name: str = "default"

    def __post_init__(self) -> None:
        self.lock = threading.Lock()
        self.grid = OccupancyGrid(self.config.size_m, self.config.res)
        self.keyframes: list[Keyframe] = []
        self.initial_pose: tuple = (0.0, 0.0, 0.0)
        self.scope: tuple = (None, False)          # (radius_m, no_behind), see in_scope
        self.goal: Optional[dict] = None
        self._new_layers()
        self._kf_data: dict = {}                   # kf_id -> its readings in base_link, for rebuilds
        self._base = None                          # the loaded map a rebuild starts from (None: empty)

    # --- helpers --------------------------------------------------------

    def _new_layers(self) -> None:
        """What the ToF grid adds to the wall map. `low` and `drops` are kept apart
        from the walls (they are not what the wall rays are matched against) and are
        sticky: a later ray passing over a step does not erase it."""
        n = self.grid.n
        self.low = np.zeros((n, n), bool)            # low obstacles           [iy, ix]
        self.drops = np.zeros((n, n), bool)          # the floor falls away
        self.cloud = np.zeros((0, 3), np.float32)    # x, y (map), height above the floor

    def _mark(self, layer: np.ndarray, xy: np.ndarray) -> None:
        ij = self.grid.cells(xy)
        ij = ij[self.grid.inside(ij)]
        layer[ij[:, 1], ij[:, 0]] = True

    def _trav(self, extra: float = 0.0) -> np.ndarray:
        r = self.config.robot_radius + self.config.clearance + extra
        nogo = _dilate(self.low | self.drops, int(math.ceil(r / self.grid.res)))
        return traversable(self.grid, r) & ~nogo

    def _plan(self, start_xy, goal_xy) -> Optional[list]:
        """A path with room to spare where the map allows (pref_clearance), else the tightest."""
        if self.config.pref_clearance > 0:
            p = plan_path(self.grid, self._trav(self.config.pref_clearance), start_xy, goal_xy)
            if p is not None:
                return p
        return plan_path(self.grid, self._trav(), start_xy, goal_xy)

    def clearances(self, path: Sequence, reach: float = 1.0) -> list:
        """For each path point, the nearest mapped wall, low obstacle or drop (m, capped at
        `reach`) along the leg that arrives there (the first: around the start)."""
        with self.lock:
            return self._clearances(path, reach)

    def _clearances(self, path: Sequence, reach: float = 1.0) -> list:
        g = self.grid
        hard = g.occupied() | self.low | self.drops
        k = int(math.ceil(reach / g.res))

        def near(xy) -> float:
            c = g.cells(np.array([xy[:2]], float))[0]
            y0, x0 = max(0, c[1] - k), max(0, c[0] - k)
            ys, xs = np.nonzero(hard[y0:c[1] + k + 1, x0:c[0] + k + 1])
            if not len(xs):
                return reach
            return min(reach, float(np.min(np.hypot(xs + x0 - c[0], ys + y0 - c[1]))) * g.res)
        out = []
        for i, p in enumerate(path):
            a = path[i - 1] if i else p
            n = max(2, int(math.hypot(p[0] - a[0], p[1] - a[1]) / g.res) + 1)
            out.append(round(min(near((a[0] + (p[0] - a[0]) * t, a[1] + (p[1] - a[1]) * t))
                                 for t in np.linspace(0.0, 1.0, n)), 3))
        return out

    def _integrate(self, pose: tuple, d: dict) -> None:
        """Fold one keyframe's readings (base_link, see add_keyframe) into the map at `pose`."""
        cfg = self.config
        pts, sensor_xy = d["pts"], d["sensor_xy"]
        world = transform(pose, pts) if len(pts) else pts
        origin = transform(pose, np.array([sensor_xy], float))
        self.grid.integrate(np.repeat(origin, len(world), 0), world, cfg.max_range)
        self.grid.mark_free_disc(pose, cfg.robot_radius)
        if len(d["obstacles"]):
            self.grid.mark_occupied(transform(pose, d["obstacles"]))
        if len(d["clear"]):
            # max_range 0: no ray "ends in an obstacle", so these only carve free space.
            # Twice: the floor was SEEN there, which is worth more than a beam passing over.
            for _ in range(2):
                self.grid.integrate(np.repeat(origin, len(d["clear"]), 0), transform(pose, d["clear"]), 0.0)
        for layer, pts_ in ((self.low, d["low"]), (self.drops, d["drops"])):
            if len(pts_):
                self._mark(layer, transform(pose, pts_))
        if len(d["cloud"]):
            p3 = d["cloud"]
            p3 = np.column_stack([transform(pose, p3[:, :2].astype(float)), p3[:, 2]]).astype(np.float32)
            both = np.concatenate([self.cloud, p3])
            _, first = np.unique(np.round(both / 0.05).astype(np.int32), axis=0, return_index=True)
            self.cloud = both[np.sort(first)][-cfg.max_cloud_points:]

    def _submap(self, ids) -> OccupancyGrid:
        """Walls only, from just these keyframes: what a scan is matched against in a loop check."""
        g = OccupancyGrid(self.config.size_m, self.config.res)
        for i in ids:
            kf, d = self.keyframes[i], self._kf_data.get(i)
            if d is None or not len(d["pts"]):
                continue
            origin = transform(kf.pose, np.array([d["sensor_xy"]], float))
            g.integrate(np.repeat(origin, len(d["pts"]), 0), transform(kf.pose, d["pts"]), self.config.max_range)
        return g

    def _rebuild(self) -> None:
        """The map again, from the loaded base (if any) and every keyframe at its pose now."""
        self.grid = OccupancyGrid(self.config.size_m, self.config.res)
        self._new_layers()
        if self._base is not None:
            self.grid.logodds, self.low, self.drops, self.cloud = (a.copy() for a in self._base)
        for kf in self.keyframes:
            if kf.kf_id in self._kf_data:
                self._integrate(kf.pose, self._kf_data[kf.kf_id])

    def _close_loop(self, prior: tuple, pts_hit: np.ndarray) -> Optional[dict]:
        """See the module docstring. Corrects the keyframes in place and returns what it did,
        or None (no loop here, or the old map does not agree well enough)."""
        cfg, k = self.config, len(self.keyframes)
        if not cfg.loop_closure or k <= cfg.loop_gap or len(pts_hit) < 15:
            return None
        old = [kf.kf_id for kf in self.keyframes[:k - cfg.loop_gap]
               if math.hypot(kf.pose[0] - prior[0], kf.pose[1] - prior[1]) < cfg.loop_radius
               and kf.kf_id in self._kf_data]
        if not old:
            return None
        anchor = max(old)
        chain = [kf.pose for kf in self.keyframes[anchor:]] + [prior]
        if sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(chain, chain[1:])) < cfg.loop_min_travel:
            return None
        recent = self._submap(range(max(anchor + 1, k - 4), k))
        mr = match_scan(recent, _blur(recent.occupied(), 2.0), prior, pts_hit)
        here = mr.pose if mr.accepted else prior
        past = self._submap(old)
        mo = match_scan(past, _blur(past.occupied(), 2.0), here, pts_hit, xy_window=cfg.loop_window_m,
                        th_window=math.radians(cfg.loop_window_deg), min_score=cfg.loop_min_score)
        if not mo.accepted:
            return None
        fix = compose(mo.pose, between(here, (0.0, 0.0, 0.0)))    # map-frame correction: fix ⊕ here = mo.pose
        if math.hypot(fix[0], fix[1]) < 0.05 and abs(fix[2]) < math.radians(1.0) and \
                math.hypot(mo.pose[0] - here[0], mo.pose[1] - here[1]) < 0.05:
            return None                                           # already consistent
        n = k - anchor
        for kf in self.keyframes[anchor + 1:]:
            w = (kf.kf_id - anchor) / n
            kf.pose = compose((w * fix[0], w * fix[1], w * fix[2]), kf.pose)
        self._rebuild()
        return {"closed": True, "with_kf": anchor, "score": mo.score, "pose": mo.pose,
                "correction_m": round(math.hypot(mo.pose[0] - here[0], mo.pose[1] - here[1]), 3),
                "correction_deg": round(math.degrees(wrap(mo.pose[2] - here[2])), 2)}

    def map_to_odom(self) -> tuple:
        if not self.keyframes:
            return self.initial_pose
        kf = self.keyframes[-1]
        # map_T_odom = map_T_base ⊕ (odom_T_base)⁻¹
        return compose(kf.pose, between(kf.odom, (0.0, 0.0, 0.0)))

    def pose_of(self, odom: Sequence[float]) -> tuple:
        return compose(self.map_to_odom(), tuple(odom))

    # --- api ------------------------------------------------------------

    def reset(self, initial_pose: Sequence[float] = (0.0, 0.0, 0.0), keep_map: bool = False,
              radius_m: Optional[float] = None, no_behind: bool = False) -> None:
        """Start a new run. With keep_map (after `load`), the robot is told
        where it is on the saved map — the "user points at the start" case."""
        with self.lock:
            if not keep_map:
                self.grid = OccupancyGrid(self.config.size_m, self.config.res)
                self._new_layers()
            self._base = (self.grid.logodds.copy(), self.low.copy(), self.drops.copy(),
                          self.cloud.copy()) if keep_map else None
            self.keyframes = []
            self._kf_data = {}
            self.initial_pose = tuple(map(float, initial_pose))
            self.scope = (radius_m, bool(no_behind))
            self.goal = None

    def add_keyframe(self, odom: Sequence[float], rays: Sequence, sensor_xy=(0.0, 0.0),
                     obstacles: Sequence = (), plan: bool = True, low: Sequence = (),
                     drops: Sequence = (), clear: Sequence = (), cloud: Sequence = ()) -> dict:
        """rays: [[bearing_rad, range_m], ...] in base_link at `odom`, measured
        from `sensor_xy` (the pan axis) in base_link. obstacles: [[x, y]] in
        base_link — low obstacles / drop-offs found by the floor probe.
        From the ToF grid, all in base_link too: low, drops: [[x, y]]; clear:
        [[bearing_rad, range_m]] free space with no wall at the end; cloud:
        [[x, y, height_m]] every reading (see the module docstring)."""
        odom = tuple(map(float, odom))
        cfg = self.config
        r = np.array([[float(b), float(d)] for b, d in rays if d is not None and d > 0], float).reshape(-1, 2)
        pts = np.stack([sensor_xy[0] + r[:, 1] * np.cos(r[:, 0]),
                        sensor_xy[1] + r[:, 1] * np.sin(r[:, 0])], 1) if len(r) else np.zeros((0, 2))
        hit = r[:, 1] < cfg.max_range if len(r) else np.zeros(0, bool)

        c = np.asarray(clear, float).reshape(-1, 2)
        data = {"pts": pts, "sensor_xy": tuple(map(float, sensor_xy)),
                "obstacles": np.asarray(obstacles, float).reshape(-1, 2),
                "clear": np.stack([sensor_xy[0] + c[:, 1] * np.cos(c[:, 0]),
                                   sensor_xy[1] + c[:, 1] * np.sin(c[:, 0])], 1) if len(c) else np.zeros((0, 2)),
                "low": np.asarray(low, float).reshape(-1, 2), "drops": np.asarray(drops, float).reshape(-1, 2),
                "cloud": np.asarray(cloud, np.float32).reshape(-1, 3)}

        with self.lock:
            if self.keyframes:
                last = self.keyframes[-1]
                prior = compose(last.pose, between(last.odom, odom))
            else:
                prior = compose(self.initial_pose, odom)
            loop = self._close_loop(prior, pts[hit])
            if loop:
                m = MatchResult(tuple(loop.pop("pose")), loop["score"], True)
            elif self.keyframes or self.grid.occupied().any():
                field_ = _blur(self.grid.occupied(), sigma_cells=2.0)
                m = match_scan(self.grid, field_, prior, pts[hit])
            else:
                m = MatchResult(prior, 0.0, False)
            pose = m.pose if m.accepted else prior
            self._integrate(pose, data)
            kf = Keyframe(len(self.keyframes), odom, pose, m.score, m.accepted)
            self.keyframes.append(kf)
            self._kf_data[kf.kf_id] = data
            out = {
                "kf_id": kf.kf_id,
                "pose": [round(v, 4) for v in pose],
                "map_to_odom": [round(v, 4) for v in self.map_to_odom()],
                "match": {"score": m.score, "accepted": m.accepted,
                          "prior": [round(v, 4) for v in prior]},
                "loop": loop or {"closed": False},
            }
            if plan:
                out.update(self._next_goal(pose))
            return out

    def _next_goal(self, pose: Sequence[float]) -> dict:
        trav = self._trav()
        s = _start_cell(self.grid, trav, pose)
        fr = frontiers(self.grid, self.config.min_frontier_cells)
        explored = float(self.grid.free().sum()) * self.grid.res ** 2
        base = {"explored_m2": round(explored, 2), "frontiers": len(fr)}
        if s is None or not fr:
            self.goal = None
            return {**base, "goal": None, "done": True}
        # One BFS from the robot reaches every frontier; pick the nearest large
        # one (size breaks ties so we don't chase slivers).
        goal_mask = np.zeros_like(trav)
        owner = {}
        for i, (size, cells) in enumerate(fr):
            ok = in_scope(self.initial_pose, *self.scope, self.grid.world(np.array(cells)))
            for c, inside in zip(cells, ok):
                if inside and trav[c[1], c[0]]:
                    goal_mask[c[1], c[0]] = True
                    owner[c] = i
        if not goal_mask.any():
            # Frontier cells are all inside the inflation (e.g. a gap narrower
            # than the robot), or outside the radius / behind the start.
            # Nothing reachable left.
            self.goal = None
            return {**base, "goal": None, "done": True}
        end, parent = _bfs(trav, s, goal_mask=goal_mask)
        if end is None:
            self.goal = None
            return {**base, "goal": None, "done": True}
        cells = _shortcut(trav, _unwind(parent, end))
        path = [tuple(round(float(v), 3) for v in self.grid.world(np.array(c))) for c in cells]
        roomy = plan_path(self.grid, self._trav(self.config.pref_clearance), pose, path[-1]) \
            if self.config.pref_clearance > 0 else None
        if roomy is not None:                       # the same goal, with room to spare
            path = [(round(a, 3), round(b, 3)) for a, b in roomy]
        self.goal = {"xy": list(path[-1]), "path": [list(p) for p in path],
                     "clear_m": self._clearances(path), "frontier_cells": fr[owner[end]][0]}
        return {**base, "goal": self.goal, "done": False}

    def walls(self, min_len: float = 0.5) -> dict:
        """Straight walls fitted to the map's wall cells, and the size of the area they
        enclose (the whole explored area: several rooms make one extent)."""
        with self.lock:
            g = self.grid
            ys, xs = np.nonzero(g.occupied())
            xy = g.world(np.stack([xs, ys], 1)) if len(xs) else np.zeros((0, 2))
        ws = fit_walls(xy, g.res, min_len)
        return {"walls": [{"from": [round(a, 3), round(b, 3)], "to": [round(c, 3), round(d, 3)],
                           "length_m": round(math.hypot(c - a, d - b), 2)} for a, b, c, d in ws],
                "extent": extent(ws)}

    def plan(self, start_xy, goal_xy) -> Optional[list]:
        with self.lock:
            p = self._plan(start_xy, goal_xy)
        return None if p is None else [[round(a, 3), round(b, 3)] for a, b in p]

    def viewpoints(self, start_xy, target_xy, radius: float = 0.8, n: int = 4,
                   start_angle: Optional[float] = None) -> list:
        """Up to n reachable poses spread around target_xy, each facing it,
        visited in a walk-around order, each with the path from the previous.

        The target itself (a plant) is an obstacle, so line of sight is only
        required up to `radius * 0.5` from it.
        """
        with self.lock:
            trav = self._trav()
            occ = self.grid.occupied()
            tx, ty = map(float, target_xy)
            if start_angle is None:
                start_angle = math.atan2(start_xy[1] - ty, start_xy[0] - tx)
            cands = []
            for k in range(n):
                ang = start_angle + 2 * math.pi * k / n
                for rr in (radius, radius * 1.25, radius * 0.8, radius * 1.5):
                    p = (tx + rr * math.cos(ang), ty + rr * math.sin(ang))
                    c = self.grid.cells(np.array([p]))[0]
                    if not self.grid.inside(c) or not trav[c[1], c[0]]:
                        continue
                    stop = (tx + rr * 0.5 * math.cos(ang), ty + rr * 0.5 * math.sin(ang))
                    a = tuple(c)
                    b = tuple(self.grid.cells(np.array([stop]))[0])
                    if not _line_clear(~occ, a, b):
                        continue
                    cands.append({"angle_deg": round(math.degrees(ang), 1),
                                  "pose": [round(p[0], 3), round(p[1], 3),
                                           round(math.atan2(ty - p[1], tx - p[0]), 4)]})
                    break
            out, cur = [], tuple(start_xy[:2])
            for v in cands:           # already in walk-around order
                path = plan_path(self.grid, trav, cur, v["pose"][:2])
                if path is None:
                    continue
                v["path"] = [[round(a, 3), round(b, 3)] for a, b in path]
                out.append(v)
                cur = tuple(v["pose"][:2])
            return out

    # --- persistence ----------------------------------------------------

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.name}.npz"
        with self.lock, open(path.with_suffix(".tmp"), "wb") as fh:
            np.savez(fh, logodds=self.grid.logodds, res=self.grid.res, origin=self.grid.origin,
                     low=self.low, drops=self.drops, cloud=self.cloud,
                     keyframes=json.dumps([kf.__dict__ for kf in self.keyframes]))
        path.with_suffix(".tmp").replace(path)
        return path

    def load(self, directory: Path) -> None:
        data = np.load(directory / f"{self.name}.npz", allow_pickle=False)
        with self.lock:
            self.grid.logodds = data["logodds"].astype(np.float32)
            self.grid.n = self.grid.logodds.shape[0]
            self.grid.res = float(data["res"])
            self.grid.origin = float(data["origin"])
            self._new_layers()
            if "low" in data:                        # maps saved before the ToF layers have none
                self.low, self.drops = data["low"].astype(bool), data["drops"].astype(bool)
                self.cloud = data["cloud"].astype(np.float32)
            self.keyframes = []

    def state(self) -> dict:
        with self.lock:
            last = self.keyframes[-1] if self.keyframes else None
            return {
                "name": self.name,
                "keyframes": len(self.keyframes),
                "pose": list(last.pose) if last else None,
                "map_to_odom": list(self.map_to_odom()),
                "explored_m2": round(float(self.grid.free().sum()) * self.grid.res ** 2, 2),
                "goal": self.goal,
                "low_cells": int(self.low.sum()), "drop_cells": int(self.drops.sum()),
                "cloud_points": int(len(self.cloud)),
            }

    def render(self, scale: int = 2, marks: Sequence = ()) -> bytes:
        """PNG of the map: unknown grey, free white, walls black, low obstacles
        orange, drops (stairs down) magenta, keyframe trail blue, goal path green,
        marks (e.g. plants) red. marks: (label, x, y) or (label, x, y, height_m);
        marks at one spot (a plant stand) are listed one under the other, lowest
        last, each with its height."""
        import io

        from PIL import Image, ImageDraw
        with self.lock:
            g = self.grid
            img = np.full((g.n, g.n, 3), 160, np.uint8)
            img[g.free()] = (250, 250, 250)
            img[g.occupied()] = (20, 20, 20)
            img[self.low] = (240, 150, 30)
            img[self.drops] = (200, 40, 200)
            kfs = [kf.pose for kf in self.keyframes]
            goal = self.goal
        im = Image.fromarray(img[::-1]).resize((g.n * scale, g.n * scale), Image.NEAREST)
        d = ImageDraw.Draw(im)

        def px(x, y):
            c = (np.array([x, y]) - g.origin) / g.res
            return (c[0] * scale, (g.n - c[1]) * scale)
        if goal:
            d.line([px(*p) for p in goal["path"]], fill=(0, 170, 0), width=2)
        if len(kfs) > 1:
            d.line([px(p[0], p[1]) for p in kfs], fill=(40, 90, 220), width=2)
        for p in kfs[-1:]:
            x, y = px(p[0], p[1])
            d.ellipse([x - 5, y - 5, x + 5, y + 5], outline=(40, 90, 220), width=2)
            d.line([(x, y), (x + 12 * math.cos(p[2]), y - 12 * math.sin(p[2]))], fill=(40, 90, 220), width=2)
        rows: dict = {}                                  # text lines already written beside each spot
        for label, x, y, *z in sorted(marks, key=lambda m: -(m[3] or 0.0) if len(m) > 3 else 0.0):
            cx, cy = px(x, y)
            d.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(210, 40, 40))
            spot = (round(cx / 12), round(cy / 12))
            text = str(label) + (f" {z[0]:.2f}m up" if z and z[0] else "")
            d.text((cx + 6, cy - 6 + 11 * rows.get(spot, 0)), text, fill=(210, 40, 40))
            rows[spot] = rows.get(spot, 0) + 1
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()
