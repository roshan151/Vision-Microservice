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
    mid = s.step(frame((box, RED)), [person(box)], [None], Geometry(range_m=1.5))
    assert mid["drive_m"] == pytest.approx(0.3)
    near = s.step(frame((box, RED)), [person(box)], [None], Geometry(range_m=0.6))
    assert near["drive_m"] == 0 and near["turn_deg"] == 0


def test_distance_from_feet_on_the_floor_without_a_range():
    s = locked_session()
    # Feet 60 px below centre with fx = 320/tan(33°) ≈ 493 px: 6.9° down at 0.25 m high ≈ 2.06 m
    box = (290, 100, 350, 300)
    out = s.step(frame((box, RED)), [person(box)], [None], Geometry())
    assert out["distance_m"] == pytest.approx(2.06, abs=0.05)
    clipped = (290, 100, 350, H)                          # feet out of frame: unknown, so no drive
    out = s.step(frame((clipped, RED)), [person(clipped)], [None], Geometry())
    assert out["distance_m"] is None and out["drive_m"] == 0


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


def test_an_unlearned_back_view_far_from_where_they_were_is_not_taken():
    s = locked_session()
    while s.step(frame(), [], [], Geometry())["state"] != "searching":
        pass
    far = (40, 100, 120, 400)
    out = s.step(dressed(frame(), far, GREEN, RED, BLACK, share=0.6), [person(far)], [None], Geometry())
    assert out["state"] != "tracking"


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
    assert asyncio.run(_face_label(st, rgb, (280, 100, 360, 400))) == "Roshan"
    st.backend.face = OtherFace()
    assert asyncio.run(_face_label(st, rgb, (280, 100, 360, 400))) == STRANGER
    assert _step(c, 1).json()["locked"] is False
