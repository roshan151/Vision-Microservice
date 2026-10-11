"""Follow mode: the session logic on synthetic frames, then the endpoints."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from vision_service.backends.base import Detection, FaceEmbedding
from vision_service.follow import FollowConfig, FollowSession, Geometry

W, H = 640, 480
RED, BLUE = (200, 30, 30), (30, 30, 200)


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


def test_lost_holds_then_searches_toward_the_last_side_then_gives_up():
    s = locked_session()
    box = (40, 100, 120, 400)
    s.step(frame((box, RED)), [person(box)], [None], Geometry())    # last seen on the left
    empty = frame()
    states = [s.step(empty, [], [], Geometry()) for _ in range(3 + 12 + 1)]
    assert [o["state"] for o in states[:3]] == ["holding"] * 3
    assert all(o["state"] == "searching" and o["turn_deg"] == -30 for o in states[3:15])
    assert states[-1]["state"] == "gone" and states[-1]["followed"]


def test_a_differently_dressed_person_is_not_followed():
    s = locked_session()
    box = (280, 100, 360, 400)
    out = s.step(frame((box, BLUE)), [person(box)], [None], Geometry())
    assert out["state"] == "holding"


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
