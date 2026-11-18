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


def test_tof_layers_low_drops_clear_and_cloud(tmp_path):
    """What the 8x8 ToF adds: free space without a wall, places never to drive, 3-D points."""
    s = SlamSession(SlamConfig(size_m=16), name="tof")
    short = [[b, min(r, 1.0)] for b, r in scan((0, 0, 0))]       # a sensor that only reaches 1 m ...
    walls = [r for r in short if r[1] < 1.0]
    clear = [r for r in short if r[1] >= 1.0]                     # ... sees only floor in most directions
    s.add_keyframe((0, 0, 0), walls, clear=clear,
                   low=[[0.6, 0.0]], drops=[[0.0, 0.7]],
                   cloud=[[0.6, 0.0, 0.08], [0.6, 0.0, 0.08], [1.0, 0.2, 1.4]])
    g = s.grid
    cell = lambda x, y: tuple(g.cells(np.array([[x, y]]))[0][::-1])
    assert g.free()[cell(0.8, 0.0)] and not g.occupied()[cell(1.0, 0.0)], "clear rays carve, and leave no wall"
    trav = s._trav()
    assert trav[cell(0.0, -0.5)]
    assert not trav[cell(0.6, 0.0)] and not trav[cell(0.45, 0.0)], "a low obstacle, and the robot's width around it"
    assert not trav[cell(0.0, 0.7)] and not trav[cell(0.0, 0.55)], "a drop, kept clear of"
    assert s.plan((0, 0), (0.0, 0.7)) is None or s.plan((0, 0), (0.0, 0.7))[-1] != pytest.approx([0.0, 0.7], abs=0.1)
    st = s.state()
    assert st["low_cells"] == 1 and st["drop_cells"] == 1 and st["cloud_points"] == 2   # same 5 cm cube: one point

    # Later rays passing over the step do not erase it.
    for _ in range(5):
        s.add_keyframe((0, 0, 0), walls, clear=clear)
    assert not s._trav()[cell(0.6, 0.0)]

    png = s.render(marks=[("plant_001", 1.0, 1.0, 0.0), ("plant_002", 1.0, 1.0, 0.7), ("plant_003", 0.5, 0.5)])
    assert png[:4] == b"\x89PNG"
    s.save(tmp_path)
    t = SlamSession(SlamConfig(size_m=16), name="tof")
    t.load(tmp_path)
    assert t.low.sum() == 1 and t.drops.sum() == 1 and len(t.cloud) == 2
    assert t.cloud[:, 2].max() == pytest.approx(1.4)
    t.reset()
    assert not t.low.any() and len(t.cloud) == 0, "a new map starts empty"


def test_maps_saved_before_the_tof_layers_still_load(tmp_path):
    s = SlamSession(SlamConfig(size_m=16), name="old")
    s.add_keyframe((0, 0, 0), scan((0, 0, 0)))
    np.savez(tmp_path / "old.npz", logodds=s.grid.logodds, res=s.grid.res, origin=s.grid.origin, keyframes="[]")
    t = SlamSession(SlamConfig(size_m=16), name="old")
    t.load(tmp_path)
    assert not t.low.any() and t.grid.occupied().any()


def _drive_round(s, truth, odom_drift_deg=0.0, odom_scale=1.0, rng=None):
    """Keyframes along `truth` with odometry that lies by odom_scale and odom_drift_deg per leg."""
    rng = rng or np.random.default_rng(3)
    odom, outs = truth[0], [s.add_keyframe(truth[0], scan(truth[0], rng=rng))]
    for prev, cur in zip(truth, truth[1:]):
        d = between(prev, cur)
        odom = compose(odom, (d[0] * odom_scale, d[1] * odom_scale, d[2] + math.radians(odom_drift_deg)))
        outs.append(s.add_keyframe(odom, scan(cur, rng=rng)))
    return outs


def test_a_loop_coming_back_off_is_closed_and_the_poses_in_between_corrected(monkeypatch):
    """With keyframe-to-keyframe matching switched off, the chain drifts; back at the start,
    the loop match against the old keyframes puts it right and spreads the fix back."""
    import vision_service.slam as slam
    real = slam.match_scan

    def no_sequential(grid, field_, prior, pts, xy_window=0.4, **kw):
        if xy_window == 0.4:                       # the ordinary and the "recent" matches: drift on
            return slam.MatchResult(tuple(prior), 0.0, False)
        return real(grid, field_, prior, pts, xy_window=xy_window, **kw)
    monkeypatch.setattr(slam, "match_scan", no_sequential)
    s = SlamSession(SlamConfig(size_m=16))
    truth = [(0, 0, 0), (1, 0, 0), (1.8, 0, 0), (1.8, 1.8, 1.57), (0.8, 2.2, 3.14),
             (0, 2.2, 3.14), (-0.4, 1.2, -1.57), (0.1, 0.1, 0.0)]
    outs = _drive_round(s, truth, odom_drift_deg=2.0)
    assert outs[-1]["loop"]["closed"], outs[-1]["loop"]
    x, y, th = outs[-1]["pose"]
    assert math.hypot(x - truth[-1][0], y - truth[-1][1]) < 0.08
    drifted = [o["match"]["prior"] for o in outs]
    for kf, true, before in zip(s.keyframes[5:7], truth[5:7], drifted[5:7]):
        err_now = math.hypot(kf.pose[0] - true[0], kf.pose[1] - true[1])
        err_before = math.hypot(before[0] - true[0], before[1] - true[1])
        assert err_now < err_before, (kf.kf_id, err_now, err_before)


def test_no_loop_without_enough_travel_and_ordinary_runs_unchanged():
    s = SlamSession(SlamConfig(size_m=16))
    outs = _drive_round(s, [(0, 0, 0), (0.5, 0, 0), (0.5, 0.5, 1.0), (0, 0.5, 2.0), (0, 0, 3.0), (0.2, 0.1, 0.0)])
    assert not any(o["loop"]["closed"] for o in outs)


def test_walls_and_room_size_from_the_map():
    s = SlamSession(SlamConfig(size_m=16))
    _drive_round(s, [(0, 0, 0), (2, 1.8, 1.0), (4, 0, 2.0), (0.5, 2.2, -1.0), (4.2, 2.2, 3.0)])
    out = s.walls()
    lengths = [w["length_m"] for w in out["walls"]]
    assert lengths[0] > 5.0 and sum(1 for L in lengths if L > 3.0) >= 3, lengths
    ext = out["extent"]
    assert abs(ext["length_m"] - 6.0) < 0.2 and abs(ext["width_m"] - 4.0) < 0.2, ext


def test_paths_keep_clear_where_they_can_and_say_how_clear():
    s = SlamSession(SlamConfig(size_m=16))
    _drive_round(s, [(0, 0, 0), (2, 1.8, 1.0), (4, 0, 2.0), (0.5, 2.2, -1.0), (4.2, 2.2, 3.0)])
    path = s.plan((0.0, 0.0), (4.2, 1.6))
    assert path is not None
    clear = s.clearances(path)
    assert len(clear) == len(path) and min(clear[1:]) >= s.config.robot_radius + s.config.clearance - 0.05
