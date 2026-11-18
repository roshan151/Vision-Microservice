"""Follow mode: the session logic on synthetic frames, then the endpoints."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from vision_service.backends.base import Detection, FaceEmbedding
from vision_service.follow import SEARCH_SWEEP, STRANGER, FollowConfig, FollowSession, Geometry, signature

W, H = 640, 480
RED, BLUE = (200, 30, 30), (30, 30, 200)
GREEN, BLACK = (30, 200, 30), (10, 10, 10)


def frame(*people):
    """people: (box, colour). Grey background, each person a filled box."""
    img = np.full((H, W, 3), 128, np.uint8)
    for (x1, y1, x2, y2), c in people:
        img[y1:y2, x1:x2] = c
    return img


def person(box):
    return Detection("person", 0.9, tuple(float(v) for v in box))


def locked_session(box=(280, 100, 360, 400), colour=RED, label="roshan"):
    s = FollowSession()
    s.start(label)
    img = frame((box, colour))
    for _ in range(2):                                   # two face matches lock
        s.step(img, [person(box)], [label], Geometry())
    assert s.locked
    return s


def test_two_face_matches_lock_and_the_robot_does_not_move_while_confirming():
    s = FollowSession()
    s.start("roshan")
    box = (280, 100, 360, 400)
    first = s.step(frame((box, RED)), [person(box)], ["roshan"], Geometry())
    assert first["state"] == "searching" and not first["locked"]
    assert first["turn_deg"] == 0 and first["drive_m"] == 0
    assert s.step(frame((box, RED)), [person(box)], ["roshan"], Geometry())["locked"]


def test_someone_else_does_not_lock():
    s = FollowSession()
    s.start("roshan")
    box = (280, 100, 360, 400)
    for _ in range(3):
        out = s.step(frame((box, RED)), [person(box)], ["sam"], Geometry())
    assert not out["locked"]


def test_follow_me_locks_on_any_enrolled_face():
    s = FollowSession()
    s.start(None)
    box = (280, 100, 360, 400)
    for _ in range(2):
        out = s.step(frame((box, RED)), [person(box)], ["sam"], Geometry())
    assert out["locked"] and out["target"] == "sam"


def test_tracks_by_clothes_from_behind_and_turns_toward_them():
    s = locked_session()
    box = (40, 100, 120, 400)                            # walked to the left, face hidden
    out = s.step(frame((box, RED), ((500, 100, 580, 400), BLUE)),
                 [person(box), person((500, 100, 580, 400))], [None, None], Geometry())
    assert out["state"] == "tracking"
    assert out["bearing_deg"] > 20 and out["turn_deg"] < 0          # left = negative turn
    assert out["drive_m"] == 0                                      # big turn: turn only


def test_drives_to_the_standoff_and_never_reverses():
    s = locked_session()
    box = (290, 150, 350, 400)
    far = s.step(frame((box, RED)), [person(box)], [None], Geometry(range_m=2.0))
    assert far["distance_m"] == 2.0 and far["drive_m"] == pytest.approx(0.5)
    mid = s.step(frame((box, RED)), [person(box)], [None], Geometry(range_m=1.2))
    assert mid["drive_m"] == pytest.approx(0.4)                     # to the 0.8 m standoff
    near = s.step(frame((box, RED)), [person(box)], [None], Geometry(range_m=0.6))
    assert near["drive_m"] == 0 and near["turn_deg"] == 0


def test_distance_from_feet_on_the_floor_without_a_range():
    s = locked_session()
    # Feet 60 px below centre with fx = 320/tan(33°) ≈ 493 px: 6.9° down at 0.25 m high ≈ 2.06 m
    box = (290, 100, 350, 300)
    out = s.step(frame((box, RED)), [person(box)], [None], Geometry())
    assert out["distance_m"] == pytest.approx(2.06, abs=0.05) and out["distance_from"] == "floor"


def test_a_box_cut_by_the_bottom_edge_is_not_read_as_feet():
    # Boxes cut off by the frame end a few px short of it: the old check took that
    # edge for the feet and read ~0.9 m every time, so the robot never drove.
    s = locked_session()
    for bottom in (H, H - 8):
        clipped = (265, 100, 375, bottom)                # head up 20 deg: the floor shows only beyond ~2.4 m
        out = s.step(frame((clipped, RED)), [person(clipped)], [None], Geometry(tilt_deg=20.0))
        assert out["distance_from"] == "width"           # 0.45 m across 110 px: ~2.0 m
        assert out["distance_m"] == pytest.approx(2.0, abs=0.1) and out["drive_m"] == pytest.approx(0.5)


def test_legs_filling_the_frame_are_not_read_as_far_away():
    # Cut off at the top and the bottom they are close. Legs are narrower than a body, so the
    # width said ~1.5 m and the robot crept in to the knees.
    s = locked_session()
    legs = (245, 0, 395, H)
    out = s.step(frame((legs, RED)), [person(legs)], [None], Geometry(tilt_deg=10.0))
    assert out["state"] == "tracking" and out["distance_from"] == "near"
    assert out["drive_m"] == 0
    far = (265, 250, 375, H)                             # feet cut off but not the top: they are not near
    out = s.step(frame((far, RED)), [person(far)], [None], Geometry(tilt_deg=10.0))
    assert out["distance_from"] == "width" and out["drive_m"] == pytest.approx(0.5)


def test_a_visible_face_gives_the_distance_when_the_feet_do_not():
    s = locked_session()
    clipped = (290, 100, 350, H)
    out = s.step(frame((clipped, RED)), [person(clipped)], [None], Geometry(tilt_deg=20.0), face_px=[80.0])
    assert out["distance_from"] == "face"                # 0.16 m across 80 px: ~0.99 m
    assert out["distance_m"] == pytest.approx(0.99, abs=0.03) and out["drive_m"] == pytest.approx(0.19, abs=0.03)


def test_no_distance_when_nothing_tells_it():
    s = locked_session()
    huge = (0, 100, W, H)                                # cut at the sides and the bottom, no face
    out = s.step(frame((huge, RED)), [person(huge)], [None], Geometry())
    assert out["distance_m"] is None and out["drive_m"] == 0


def test_turns_only_part_of_the_way_so_it_does_not_swing_back_and_forth():
    s = locked_session()
    box = (40, 100, 120, 400)
    out = s.step(frame((box, RED)), [person(box)], [None], Geometry())
    assert out["turn_deg"] == pytest.approx(-0.6 * out["bearing_deg"], abs=0.1)


def test_lost_holds_then_sweeps_the_head_then_gives_up():
    s = locked_session()
    box = (40, 100, 120, 400)
    s.step(frame((box, RED)), [person(box)], [None], Geometry())    # last seen on the left
    empty = frame()
    n = len(SEARCH_SWEEP)
    states = [s.step(empty, [], [], Geometry()) for _ in range(3 + n + 1)]
    assert [o["state"] for o in states[:3]] == ["holding"] * 3
    assert all(o["state"] == "searching" and o["turn_deg"] == 0 for o in states[3:3 + n])
    assert [tuple(o["head"]) for o in states[3:3 + n]] == SEARCH_SWEEP
    assert states[-1]["state"] == "gone" and states[-1]["followed"]


def test_sweep_pauses_while_someone_is_in_view():
    s = FollowSession()
    s.start("sam")
    box = (280, 100, 360, 400)
    outs = [s.step(frame((box, RED)), [person(box)], [None], Geometry()) for _ in range(5)]
    assert all(o["state"] == "searching" and o["head"] == [0, 0] for o in outs)


def test_a_differently_dressed_person_is_not_followed():
    s = locked_session()
    box = (280, 100, 360, 400)
    out = s.step(frame((box, BLUE)), [person(box)], [None], Geometry())
    assert out["state"] == "holding"


def dressed(img, box, top, legs, legs_other=None, share=1.0):
    """Paint a person: torso colour over legs colour (legs `share` of it, rest `legs_other`)."""
    x1, y1, x2, y2 = box
    mid = y1 + (y2 - y1) // 2
    img[y1:mid, x1:x2] = top
    img[mid:y2, x1:x2] = legs
    if legs_other is not None:
        cut = mid + int((y2 - mid) * share)
        img[cut:y2, x1:x2] = legs_other
    return img


def test_keeps_them_when_they_turn_round_and_knows_their_back_later():
    box = (280, 100, 360, 400)
    s = FollowSession()
    s.start("roshan")
    front = dressed(frame(), box, RED, BLUE)
    for _ in range(2):
        s.step(front, [person(box)], ["roshan"], Geometry())
    assert s.locked and len(s.bank) == 1
    back = dressed(frame(), box, GREEN, BLUE, BLACK, share=0.6)     # backpack over the shirt
    sim = s.similarity(signature(back, box))
    assert s.cfg.continuity_similarity <= sim < s.cfg.min_similarity  # a weak match on looks alone
    out = s.step(back, [person(box)], [None], Geometry())
    assert out["state"] == "tracking", "alone where expected: still them"
    assert len(s.bank) == 2, "the back view is learned"
    empty = frame()                                                 # lost: hold, then the sweep starts
    while s.step(empty, [], [], Geometry())["state"] != "searching":
        pass
    far = (40, 100, 120, 400)                                       # seen again, from behind, elsewhere
    out = s.step(dressed(frame(), far, GREEN, BLUE, BLACK, share=0.6), [person(far)], [None], Geometry())
    assert out["state"] == "tracking" and out["target"] == "roshan"


def test_while_searching_a_weak_match_is_taken_back_only_when_they_are_alone_in_view():
    far, other = (40, 100, 120, 400), (500, 100, 580, 400)
    weak = dressed(frame(), far, GREEN, RED, BLACK, share=0.6)      # an unlearned back view, elsewhere

    s = locked_session()
    while s.step(frame(), [], [], Geometry())["state"] != "searching":
        pass
    sim = s.similarity(signature(weak, far))
    assert s.cfg.continuity_similarity <= sim < s.cfg.reacquire_similarity
    both = dressed(weak.copy(), other, BLUE, BLUE)
    assert s.step(both, [person(far), person(other)], [None, None], Geometry())["state"] != "tracking"
    assert s.step(weak, [person(far)], [STRANGER], Geometry())["state"] != "tracking"
    assert s.step(weak, [person(far)], [None], Geometry())["state"] == "tracking", "alone: it is them"


def test_same_clothes_but_a_strangers_face_is_not_followed():
    s = locked_session()
    box = (280, 100, 360, 400)
    out = s.step(frame((box, RED)), [person(box)], [STRANGER], Geometry())
    assert out["state"] == "holding"


def test_same_clothes_but_another_enrolled_face_is_not_followed():
    s = locked_session()
    box = (280, 100, 360, 400)
    out = s.step(frame((box, RED)), [person(box)], ["sam"], Geometry())
    assert out["state"] == "holding"


def test_two_lookalikes_the_one_where_they_should_be_wins():
    s = locked_session()                                   # locked straight ahead: expected bearing 0
    target, decoy = (570, 100, 640, 400), (285, 100, 355, 400)
    # The head has panned 30 degrees left: straight ahead is now at the right edge of
    # the image, and the decoy is in the middle (where the old tie-break looked).
    out = s.step(frame((target, RED), (decoy, RED)), [person(decoy), person(target)], [None, None],
                 Geometry(pan_deg=30.0))
    assert out["state"] == "tracking" and out["bbox"][0] == 570


# ------------------------------------------------------------------ endpoints

def jpeg(img):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


class PersonDetector:
    model_version = "fake"

    def __init__(self, box): self.box = box

    def detect(self, image, min_confidence):
        return [person(self.box), Detection("chair", 0.9, (0, 0, 10, 10))]


class OneFace:
    model_version = "fake"
    v = np.eye(512, dtype=np.float32)[0]

    def embed(self, image):
        return FaceEmbedding(self.v, (0, 0, 10, 10), 0.9, 1)


@pytest.fixture()
def follow_client(client):
    st = client.app.state
    box = (280, 100, 360, 400)
    st.backend.detector, st.backend.face = PersonDetector(box), OneFace()
    st.gallery.add("Roshan", [OneFace.v])
    client.img = jpeg(frame((box, RED)))
    return client


def _step(c, seq=1, **form):
    return c.post("/v1/follow/step", files={"image": ("f.jpg", c.img, "image/jpeg")},
                  data={"seq": seq, **form})


def test_follow_endpoints_round_trip(follow_client):
    c = follow_client
    assert c.post("/v1/follow/start", json={"label": "nobody"}).status_code == 404
    assert _step(c).status_code == 409                                  # not started
    r = c.post("/v1/follow/start", json={"label": "roshan"})            # case-insensitive
    assert r.status_code == 200 and r.json()["label"] == "Roshan"
    assert _step(c, 1).json()["state"] == "searching"
    body = _step(c, 2, range_m="2.5", tilt_deg="20").json()
    assert body["seq"] == 2 and body["persons"] == 1
    assert body["state"] == "tracking" and body["target"] == "Roshan"
    assert body["drive_m"] == pytest.approx(0.5)
    assert c.post("/v1/follow/stop").json() == {"target": "Roshan", "followed": True}


def test_follow_me_needs_someone_enrolled(client):
    assert client.post("/v1/follow/start", json={}).status_code == 404


class OtherFace(OneFace):
    v = np.eye(512, dtype=np.float32)[1]          # orthogonal to Roshan's: similarity 0


def test_a_clear_face_that_is_not_the_target_is_a_stranger(follow_client):
    import asyncio
    from vision_service.follow import _face_label
    c = follow_client
    st = c.app.state
    c.post("/v1/follow/start", json={"label": "roshan"})
    rgb = frame(((280, 100, 360, 400), RED))
    assert asyncio.run(_face_label(st, rgb, (280, 100, 360, 400)))[0] == "Roshan"
    st.backend.face = OtherFace()
    assert asyncio.run(_face_label(st, rgb, (280, 100, 360, 400)))[0] == STRANGER
    assert _step(c, 1).json()["locked"] is False


def grid(v=None, box=None, inside=None):
    """An 8x8 ToF grid of `v`, with `inside` in the zones a camera box covers (W x H, hfov 66)."""
    from vision_service.follow import range_in_box
    g = [[v] * 8 for _ in range(8)]
    if box is not None:
        for r in range(8):
            for c in range(8):
                one = [[None] * 8 for _ in range(8)]
                one[r][c] = 1.0
                if range_in_box(one, box, W, H, 66.0) is not None:
                    g[r][c] = inside
    return g


def test_range_in_box_takes_the_zones_on_the_box_and_the_nearest_for_a_small_one():
    from vision_service.follow import range_in_box
    g = grid(3.0, (280, 100, 360, 400), 1.5)
    assert range_in_box(g, (280, 100, 360, 400), W, H, 66.0) == 1.5
    assert range_in_box(g, (0, 0, 10, 10), W, H, 66.0) is None          # outside the ToF's 60 degrees
    assert range_in_box(None, (280, 100, 360, 400), W, H, 66.0) is None


def test_the_tof_zones_on_the_person_give_the_distance():
    s = locked_session()
    box = (280, 100, 360, 400)
    out = s.step(frame((box, RED)), [person(box)], [None], Geometry(tof_grid=grid(3.0, box, 2.0)))
    assert out["distance_from"] == "tof" and out["distance_m"] == 2.0
    assert out["drive_m"] == 0.5


def test_a_lookalike_at_the_wrong_range_is_not_taken_back():
    box = (280, 100, 360, 400)
    s = locked_session(box)
    s.step(frame((box, RED)), [person(box)], [None], Geometry(tof_grid=grid(3.0, box, 1.4)))   # at 1.4 m
    assert s.expect_dist == pytest.approx(0.9)                       # 1.4, less the 0.5 it drove
    img = frame((box, RED))
    far = s.step(img, [person(box)], [None], Geometry(tof_grid=grid(None, box, 4.5)))
    assert far["state"] != "tracking", "same clothes, but 4.5 m away where they should be at 0.9"
    near = s.step(img, [person(box)], [None], Geometry(tof_grid=grid(None, box, 1.2)))
    assert near["state"] == "tracking"
    # A face is never overruled by the range.
    s2 = locked_session(box)
    s2.step(img, [person(box)], [None], Geometry(tof_grid=grid(3.0, box, 1.4)))
    assert s2.step(img, [person(box)], ["roshan"], Geometry(tof_grid=grid(None, box, 4.5)))["state"] == "tracking"


def test_the_search_starts_where_the_robot_says_they_were_last_seen():
    s = locked_session()
    outs = [s.step(frame(), [], [], Geometry(hint_deg=45.0)) for _ in range(5)]
    assert [o["state"] for o in outs[:3]] == ["holding"] * 3
    assert outs[3]["head"] == [45, 0] and outs[4]["head"] == list(SEARCH_SWEEP[0])
    s = locked_session()
    outs = [s.step(frame(), [], [], Geometry(hint_deg=-150.0)) for _ in range(5)]
    assert outs[3]["state"] == "searching" and outs[3]["turn_deg"] == 150.0 and outs[3]["head"] == [0, 0], \
        "behind the robot, past what the head reaches: the robot turns (right, 150) to look"
    assert outs[4]["turn_deg"] == 0                                  # once: then the usual sweep


def test_follow_step_takes_the_tof_grid_and_the_hint(follow_client):
    import json
    c = follow_client
    c.post("/v1/follow/start", json={"label": "roshan"})
    _step(c, 1)
    ok = _step(c, 2, tof_grid=json.dumps([[2.0] * 8 for _ in range(8)]), hint_deg="30")
    assert ok.status_code == 200, ok.text
    assert ok.json()["distance_from"] in ("tof", "range", "floor", "face", "width", "near")
    bad = _step(c, 3, tof_grid=json.dumps([[1.0] * 4]))
    assert bad.status_code == 422


def test_a_lost_search_gives_up_after_search_s_even_mid_sweep():
    t = [0.0]
    s = FollowSession(clock=lambda: t[0])
    s.start("roshan")
    box = (280, 100, 360, 400)
    for _ in range(2):
        s.step(frame((box, RED)), [person(box)], ["roshan"], Geometry())
    s.step(frame((box, RED)), [person(box)], [None], Geometry())          # tracking
    outs = []
    for _ in range(7):                                                   # lost: a step every 4 s
        outs.append(s.step(frame(), [], [], Geometry()))
        t[0] += 4.0
    assert [o["state"] for o in outs[:3]] == ["holding"] * 3
    assert [o["state"] for o in outs[3:6]] == ["searching"] * 3          # up to 20 s
    assert outs[6]["state"] == "gone", "lost at 0 s, past 20 s: gone, though the sweep was not done"
    assert len(SEARCH_SWEEP) <= 8


def test_tracking_answers_carry_measurements_for_the_pi():
    """bearing in the robot frame, distance uncertainty by method, ground position, confidence."""
    from vision_service.follow import FollowSession, Geometry
    s = FollowSession()
    g = Geometry(camera_height_m=0.25)
    assert s._sigma(1.0, "tof", g) == pytest.approx(0.07)
    assert s._sigma(2.0, "floor", g) == pytest.approx(4.0 / 0.25 * 0.017453, rel=1e-3)
    assert s._sigma(2.0, "width", g) == pytest.approx(0.6)
    out = s._out("tracking", g, bearing=30.0, dist=2.0, how="tof", confidence=0.8)
    assert out["ground_xy"] == pytest.approx([1.732, 1.0], abs=1e-3)
    assert out["distance_sigma_m"] == pytest.approx(0.09) and out["confidence"] == 0.8
    assert s._out("searching", g)["ground_xy"] is None
