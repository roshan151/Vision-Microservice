"""SLAM on a simulated room: a 6 x 4 m box with a 1 x 0.6 m cabinet in it."""

from __future__ import annotations

import math

import numpy as np
import pytest

from vision_service.slam import SlamConfig, SlamSession, between, compose

WALLS = [((-1, -1), (5, -1)), ((5, -1), (5, 3)), ((5, 3), (-1, 3)), ((-1, 3), (-1, -1)),
         ((2, 0.4), (3, 0.4)), ((3, 0.4), (3, 1.0)), ((3, 1.0), (2, 1.0)), ((2, 1.0), (2, 0.4))]


def cast(pose, bearing, max_r=7.5):
    ox, oy = pose[0], pose[1]
    a = pose[2] + bearing
    dx, dy = math.cos(a), math.sin(a)
    best = max_r
    for (x1, y1), (x2, y2) in WALLS:
        ex, ey = x2 - x1, y2 - y1
        den = dx * ey - dy * ex
        if abs(den) < 1e-12:
            continue
        t = ((x1 - ox) * ey - (y1 - oy) * ex) / den
        u = ((x1 - ox) * dy - (y1 - oy) * dx) / den
        if t > 0 and 0 <= u <= 1:
            best = min(best, t)
    return best


def scan(pose, step_deg=1.0, noise=0.01, rng=None):
    rng = rng or np.random.default_rng(0)
    return [[math.radians(b), cast(pose, math.radians(b)) + rng.normal(0, noise)]
            for b in np.arange(-180, 180, step_deg)]


def test_keyframes_correct_odometry_drift():
    s = SlamSession(SlamConfig(size_m=16))
    truth = [(0, 0, 0), (1.0, 0.1, 0.2), (1.2, 1.8, 1.4), (0.2, 2.0, 3.0), (4.0, 2.0, -0.5)]
    rng = np.random.default_rng(1)
    odom = truth[0]
    s.add_keyframe(odom, scan(truth[0], rng=rng))
    for prev, cur in zip(truth, truth[1:]):
        d = between(prev, cur)
        # odometry lies: 8 % distance error and 4° heading error per leg
        odom = compose(odom, (d[0] * 1.08, d[1] * 1.08, d[2] + math.radians(4)))
        out = s.add_keyframe(odom, scan(cur, rng=rng))
        assert out["match"]["accepted"], out
        x, y, th = out["pose"]
        assert math.hypot(x - cur[0], y - cur[1]) < 0.08, (out, cur)
        assert abs(math.degrees(th - cur[2])) < 2.5


def test_exploration_finishes_and_plans_around_the_cabinet():
    s = SlamSession(SlamConfig(size_m=16))
    pose = (0.0, 0.0, 0.0)
    for _ in range(25):
        out = s.add_keyframe(pose, scan(pose))
        if out["done"]:
            break
        gx, gy = out["goal"]["xy"]
        # drive the path perfectly; face along the last leg
        path = out["goal"]["path"]
        a = path[-2] if len(path) > 1 else pose
        pose = (gx, gy, math.atan2(gy - a[1], gx - a[0]))
    else:
        pytest.fail("exploration did not finish")
    assert out["explored_m2"] > 0.7 * 24
    path = s.plan((0.0, 0.7), (4.0, 0.7))            # straight line crosses the cabinet
    assert path is not None and len(path) >= 3


def test_viewpoints_surround_a_target_and_are_reachable():
    s = SlamSession(SlamConfig(size_m=16))
    for p in [(0, 0, 0), (2, 2, 0), (4, 0, 0), (0.5, 2.2, 0)]:
        s.add_keyframe(p, scan(p), plan=False)
    vps = s.viewpoints((0.0, 0.0), (1.2, 1.2), radius=0.8, n=4)
    assert len(vps) >= 3
    angles = sorted(v["angle_deg"] % 360 for v in vps)
    assert max(np.diff(angles + [angles[0] + 360])) <= 180 + 1e-6
    for v in vps:
        x, y, th = v["pose"]
        assert abs(math.atan2(1.2 - y, 1.2 - x) - th) < 1e-3     # faces the plant
        assert v["path"][-1] == pytest.approx([x, y], abs=0.05)


def test_render_and_save_load(tmp_path):
    s = SlamSession(SlamConfig(size_m=16), name="room")
    s.add_keyframe((0, 0, 0), scan((0, 0, 0)))
    png = s.render(marks=[("plant_001", 1.0, 1.0)])
    assert png[:4] == b"\x89PNG"
    s.save(tmp_path)
    t = SlamSession(SlamConfig(size_m=16), name="room")
    t.load(tmp_path)
    t.reset(initial_pose=(0, 0, 0), keep_map=True)
    out = t.add_keyframe((0.05, -0.03, 0.03), scan((0, 0, 0)))
    assert out["match"]["accepted"]
    assert math.hypot(*out["pose"][:2]) < 0.06


def test_in_scope_radius_and_half_plane():
    from vision_service.slam import in_scope
    pts = np.array([[2.0, 0.0], [-2.0, 0.0], [0.0, 2.0], [12.0, 0.0]])
    assert in_scope((0, 0, 0), 10, False, pts).tolist() == [True, True, True, False]
    assert in_scope((0, 0, 0), 10, True, pts).tolist() == [True, False, True, False]
    assert in_scope((0, 0, math.pi), None, True, pts).tolist() == [False, True, True, False]
    assert in_scope((0, 0, 0), 0, False, pts).all()                # 0 = unlimited


def test_exploration_respects_radius():
    def first_keyframe(**scope):
        s = SlamSession(SlamConfig(size_m=16))
        s.reset((0.0, 0.0, 0.0), **scope)
        return s.add_keyframe((0.0, 0.0, 0.0), scan((0.0, 0.0, 0.0)))

    assert not first_keyframe()["done"]
    assert not first_keyframe(radius_m=10, no_behind=True)["done"]
    assert first_keyframe(radius_m=0.05)["done"]        # every frontier is farther than that
