"""The global layer's map and planner: Theta* over a 2-D occupancy grid on the Mac.

The Pi navigates locally (DWA on a rolling 4 x 4 m costmap) and asks here, about once a
second, for waypoints to its goal (POST /v1/nav/plan). Each request carries the Pi's whole
local costmap, which is folded into this map first, so the map grows as the robot drives:

  * the Pi's cells are in its odometry frame; `odom_to_map` (from visual SLAM, vslam.py; the
    identity until it has one) puts them where they really are, so a room driven round twice
    lines up instead of being drawn twice as odometry drifts;
  * what the Pi saw recently wins: each fold pulls the cells it covers toward its view
    (occupied, seen free) and leaves the cells it has not seen alone;
  * things the camera recognises (a chair, a sofa, a bin: `add_semantic`) are kept in a
    separate layer that fades over a couple of minutes; the ToF misses chair legs and glass
    tables, the camera does not.

`plan()` runs Theta* (any-angle A*: paths are a few straight legs, not a staircase of cells)
over the cells that keep the robot's radius plus a margin from anything occupied, in a window
round the start and goal, with a cost for passing near things so paths keep to the middle of
gaps. Unknown cells count as free (with a small cost): the Pi's DWA and reflex keep it safe in
what neither has seen yet.
"""

from __future__ import annotations

import base64
import heapq
import math
import threading
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np

XY = Tuple[float, float]


def compose(a: Sequence[float], b: Sequence[float]) -> tuple:
    """Pose a then b (x, y, heading)."""
    c, s = math.cos(a[2]), math.sin(a[2])
    return a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], a[2] + b[2]


def inverse(a: Sequence[float]) -> tuple:
    c, s = math.cos(a[2]), math.sin(a[2])
    return -c * a[0] - s * a[1], s * a[0] - c * a[1], -a[2]


def apply(t: Sequence[float], xy: np.ndarray) -> np.ndarray:
    """Transform points (N, 2) by pose t."""
    c, s = math.cos(t[2]), math.sin(t[2])
    xy = np.asarray(xy, float).reshape(-1, 2)
    return np.stack([t[0] + c * xy[:, 0] - s * xy[:, 1], t[1] + s * xy[:, 0] + c * xy[:, 1]], axis=1)


def distance_field(occ: np.ndarray, res: float, cap_m: float = 1.0) -> np.ndarray:
    """Metres from every cell to the nearest occupied one (capped at cap_m)."""
    if not occ.any():
        return np.full(occ.shape, cap_m, np.float32)
    try:
        import cv2
        d = cv2.distanceTransform((~occ).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32) * res
        return np.minimum(d, cap_m)
    except Exception:                                        # noqa: BLE001 — no OpenCV: rings by FFT
        out = np.full(occ.shape, cap_m, np.float32)
        f_occ = np.fft.rfft2(occ.astype(np.float32))
        for r_m in np.arange(cap_m, 0.0, -res)[::-1]:        # nearest ring first
            r = int(math.ceil(r_m / res))
            yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
            k = np.zeros(occ.shape, np.float32)
            disc = (xx ** 2 + yy ** 2) * res * res <= r_m * r_m
            k[yy[disc] % occ.shape[0], xx[disc] % occ.shape[1]] = 1.0
            near = np.fft.irfft2(f_occ * np.fft.rfft2(k), s=occ.shape) > 0.5
            out = np.where(near & (out >= cap_m), r_m, out)
        out[occ] = 0.0
        return out


def _los(free: np.ndarray, a, b) -> bool:
    n = max(abs(b[0] - a[0]), abs(b[1] - a[1])) * 2 + 1
    xs = np.rint(np.linspace(a[0], b[0], n)).astype(int)
    ys = np.rint(np.linspace(a[1], b[1], n)).astype(int)
    return bool(free[xs, ys].all())


def theta_star(free: np.ndarray, start, goal, penalty: Optional[np.ndarray] = None,
               max_expand: int = 200_000) -> Optional[list]:
    """Any-angle A* on a bool grid [ix, iy]: cells [start, ..., goal] or None."""
    nx, ny = free.shape
    if not free[goal]:
        return None
    g, parent = {start: 0.0}, {start: start}
    todo, closed = [(math.dist(start, goal), start)], set()
    steps = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    while todo and len(closed) < max_expand:
        _, c = heapq.heappop(todo)
        if c in closed:
            continue
        if c == goal:
            path = [c]
            while parent[path[-1]] != path[-1]:
                path.append(parent[path[-1]])
            return path[::-1]
        closed.add(c)
        for dx, dy in steps:
            n = (c[0] + dx, c[1] + dy)
            if not (0 <= n[0] < nx and 0 <= n[1] < ny) or not free[n] or n in closed:
                continue
            extra = 0.0 if penalty is None else float(penalty[n])
            p = parent[c]
            if p != c and _los(free, p, n):
                cost, via = g[p] + math.dist(p, n) + extra, p
            else:
                cost, via = g[c] + math.dist(c, n) + extra, c
            if cost < g.get(n, math.inf):
                g[n], parent[n] = cost, via
                heapq.heappush(todo, (cost + math.dist(n, goal), n))
    return None


class GlobalMap:
    OCC_T, FREE_T = 0.6, -0.5
    TOP, BOTTOM = 3.0, -2.0

    def __init__(self, size_m: float = 24.0, res: float = 0.05, semantic_keep_s: float = 120.0) -> None:
        self.res, self.n = res, int(round(size_m / res))
        self.origin = -size_m / 2.0                          # map coord of cell (0, 0)'s corner, both axes
        self.lo = np.zeros((self.n, self.n), np.float32)     # [ix, iy]
        self.sem_t = np.zeros((self.n, self.n), np.float64)  # when the camera last saw something here (0: never)
        self.sem_label = {}                                  # cell -> label, for the picture
        self.semantic_keep_s = semantic_keep_s
        self.last_path: List[XY] = []
        self.robot: Optional[tuple] = None
        self._lock = threading.Lock()

    def cell(self, xy) -> Tuple[int, int]:
        return int((xy[0] - self.origin) / self.res), int((xy[1] - self.origin) / self.res)

    def centre(self, c) -> XY:
        return self.origin + (c[0] + 0.5) * self.res, self.origin + (c[1] + 0.5) * self.res

    def _inside(self, ix, iy):
        return (ix >= 0) & (ix < self.n) & (iy >= 0) & (iy < self.n)

    def fold_local(self, local: dict, odom_to_map: Sequence[float]) -> int:
        """The Pi's costmap snapshot (RollingCostmap.snapshot: corner ox, oy in the odometry frame,
        res, n, cells as bytes 2 occupied / 1 seen free / 0 unknown), put into this map."""
        n, res = int(local["n"]), float(local["res"])
        cells = np.frombuffer(base64.b64decode(local["cells"]), np.uint8).reshape(n, n)
        ii, jj = np.nonzero(cells)
        if not len(ii):
            return 0
        pts = np.stack([local["ox"] + (ii + 0.5) * res, local["oy"] + (jj + 0.5) * res], axis=1)
        m = apply(odom_to_map, pts)
        ix = np.floor((m[:, 0] - self.origin) / self.res).astype(int)
        iy = np.floor((m[:, 1] - self.origin) / self.res).astype(int)
        ok = self._inside(ix, iy)
        ix, iy, kind = ix[ok], iy[ok], cells[ii, jj][ok]
        with self._lock:
            target = np.where(kind == 2, 2.5, -1.5).astype(np.float32)
            self.lo[ix, iy] = np.clip(0.5 * self.lo[ix, iy] + 0.5 * target, self.BOTTOM, self.TOP)
        return int(ok.sum())

    def add_semantic(self, xy: XY, radius_m: float, label: str) -> None:
        """Something the camera recognised standing at map xy, about radius_m across."""
        c = self.cell(xy)
        r = max(1, int(math.ceil(radius_m / self.res)))
        # The same thing seen again replaces where it was put before: each sighting lands a little elsewhere
        # (0.4 m further at 2 m), and their union grew a suitcase into a 1 m blob over the person beside it,
        # so every way to them went round it (Nov 19).
        k = r + int(math.ceil(0.3 / self.res))
        oi, oj = np.mgrid[c[0] - k:c[0] + k + 1, c[1] - k:c[1] + k + 1]
        old = ((oi - c[0]) ** 2 + (oj - c[1]) ** 2 <= k * k) & self._inside(oi, oj)
        ii, jj = np.mgrid[c[0] - r:c[0] + r + 1, c[1] - r:c[1] + r + 1]
        disc = ((ii - c[0]) ** 2 + (jj - c[1]) ** 2 <= r * r) & self._inside(ii, jj)
        with self._lock:
            self.sem_t[oi[old], oj[old]] = 0.0
            self.sem_t[ii[disc], jj[disc]] = time.monotonic()
            self.sem_label[c] = label

    def occupied(self) -> np.ndarray:
        fresh = (self.sem_t > 0) & (time.monotonic() - self.sem_t < self.semantic_keep_s)
        return (self.lo > self.OCC_T) | fresh

    def plan(self, start: XY, goal: XY, radius_m: float, margin_m: float = 0.05,
             border_m: float = 2.0, target: Optional[XY] = None) -> Optional[List[XY]]:
        """Waypoints (map frame) from start to goal, the start first; None: no way. `target`: what the
        goal is for (the person the robot stops short of): the goal must be on the target's side of any
        wall. Nov 19: the goal 0.5 m short of a person standing just behind a doorway wall fell in the
        wall's margin, was moved to the robot's side, and the plan flipped between "through the doorway"
        and "straight at the wall" as their tracked spot moved."""
        s, g = self.cell(start), self.cell(goal)
        t = self.cell(target) if target is not None else g
        if not all(0 <= c[0] < self.n and 0 <= c[1] < self.n for c in (s, g, t)):
            return None
        b = int(border_m / self.res)
        x0, x1 = max(0, min(s[0], g[0], t[0]) - b), min(self.n, max(s[0], g[0], t[0]) + b + 1)
        y0, y1 = max(0, min(s[1], g[1], t[1]) - b), min(self.n, max(s[1], g[1], t[1]) + b + 1)
        with self._lock:
            occ = self.occupied()[x0:x1, y0:y1]
            unknown = (np.abs(self.lo[x0:x1, y0:y1]) < 0.3) & ~occ
        field = distance_field(occ, self.res, cap_m=radius_m + 0.6)
        ls, lg, lt = (s[0] - x0, s[1] - y0), (g[0] - x0, g[1] - y0), (t[0] - x0, t[1] - y0)
        rb = 0.15 / self.res                                   # the target's own body: its last 0.15 m

        def sees(a, b) -> bool:
            """No occupied cell on the line a -> b, except within rb of b (the person's own body)."""
            n = max(abs(b[0] - a[0]), abs(b[1] - a[1])) * 2 + 1
            xs = np.rint(np.linspace(a[0], b[0], n)).astype(int)
            ys = np.rint(np.linspace(a[1], b[1], n)).astype(int)
            far = np.hypot(xs - b[0], ys - b[1]) > rb
            return not bool(occ[xs[far], ys[far]].any())
        penalty = np.clip((radius_m + 0.4 - field) / 0.4, 0.0, 1.0) * 2.0 + unknown * 0.3
        for margin in (margin_m, 0.03):
            self.last_margin = margin
            free = field >= radius_m + margin
            strict = free.copy()                               # where it may END: never the relaxed area below
            if not free[ls]:                                   # starting in the margin: may leave it, but only
                ii, jj = np.ogrid[:free.shape[0], :free.shape[1]]   # near the start: relaxed everywhere, the path's
                near = np.hypot(ii - ls[0], jj - ls[1]) <= 0.4 / self.res   # end hugged a wall too (Nov 19 sim)
                free |= (field >= max(0.0, field[ls] - 1e-3)) & near
                free[ls] = True
            goal_c = lg
            wrong_side = target is not None and not sees(lg, lt)
            if not strict[goal_c] or wrong_side:               # a person's spot is "occupied" by them; or the goal
                r = int(0.6 / self.res)                        # is across a wall from what it is for
                cx, cy = lt if wrong_side else lg              # wrong side: look round the target instead
                ii, jj = np.mgrid[max(0, cx - r):min(free.shape[0], cx + r + 1),
                                  max(0, cy - r):min(free.shape[1], cy + r + 1)]
                ok = strict[ii, jj]                            # a place to stop with the full margin round it: a goal
                if not ok.any():                               # in the start's relaxed area (0.22 m off a wall) was one
                    continue                                   # DWA could not reach (Nov 19 sim, doorway2)
                d = np.where(ok, np.hypot(ii - lg[0], jj - lg[1]), np.inf)   # nearest to the asked goal...
                goal_c = None
                for k in np.argsort(d, axis=None)[:int(np.isfinite(d).sum())]:
                    c = np.unravel_index(k, d.shape)
                    cand = (int(ii[c]), int(jj[c]))
                    if target is None or sees(cand, lt):       # ...that can see the target
                        goal_c = cand
                        break
                if goal_c is None:
                    continue
            cells = theta_star(free, ls, goal_c, penalty)
            if cells:
                path = [start] + [self.centre((c[0] + x0, c[1] + y0)) for c in cells[1:]]
                if goal_c == lg:
                    path[-1] = tuple(goal)
                self.last_path = path
                return path
        self.last_path = []
        return None

    def stats(self) -> dict:
        with self._lock:
            occ = self.occupied()
            return {"occupied_cells": int(occ.sum()), "free_cells": int((self.lo < self.FREE_T).sum()),
                    "semantic_cells": int((self.sem_t > 0).sum()), "size_m": self.n * self.res, "res_m": self.res}

    def png(self) -> bytes:
        """The map as a picture: occupied black, seen free white, unknown grey, camera-found objects
        orange, the last path blue, the robot red. North (map +y) up."""
        from io import BytesIO

        from PIL import Image, ImageDraw
        with self._lock:
            occ = self.occupied()
            img = np.full((self.n, self.n, 3), 160, np.uint8)
            img[self.lo < self.FREE_T] = 255
            img[self.lo > self.OCC_T] = 0
            sem = (self.sem_t > 0) & (time.monotonic() - self.sem_t < self.semantic_keep_s)
            img[sem] = (255, 140, 0)
            del occ
        pic = Image.fromarray(np.ascontiguousarray(np.flipud(img.transpose(1, 0, 2))))
        d = ImageDraw.Draw(pic)
        to_px = lambda xy: ((xy[0] - self.origin) / self.res, self.n - (xy[1] - self.origin) / self.res)  # noqa: E731
        if len(self.last_path) > 1:
            d.line([to_px(p) for p in self.last_path], fill=(0, 90, 255), width=2)
        if self.robot is not None:
            x, y = to_px(self.robot)
            d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(220, 0, 0))
            d.line([x, y, x + 10 * math.cos(self.robot[2]), y - 10 * math.sin(self.robot[2])], fill=(220, 0, 0), width=2)
        buf = BytesIO()
        pic.save(buf, "PNG")
        return buf.getvalue()
