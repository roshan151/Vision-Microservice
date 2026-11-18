from __future__ import annotations

import json
import math

from conftest import jpeg


def _embed(client, img):
    r = client.post("/v1/plants/embed", files={"image": ("p.jpg", img, "image/jpeg")}, data={"seq": 3})
    assert r.status_code == 200, r.text
    return r.json()["embedding"]


def test_plant_enroll_match_and_revisit(client):
    fern = [jpeg(320, 240, (40, 140, 60)) for _ in range(3)]
    meta = {"position": [1.0, 2.0], "views": [{"pose": [0, 0, 0]}, {"pose": [1, 0, 3.1]}, {}]}
    r = client.post("/v1/plants/enroll", data={"meta": json.dumps(meta)},
                    files=[("images", (f"{i}.jpg", b, "image/jpeg")) for i, b in enumerate(fern)])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["label"] == "plant_001" and body["photos_saved"] == 3 and body["visits"] == 1

    m = client.post("/v1/plants/match", json={"embedding": _embed(client, fern[0]),
                                              "position": [1.1, 2.0]}).json()
    assert m["matches"][0]["label"] == "plant_001"
    assert m["matches"][0]["similarity"] > 0.99
    assert m["matches"][0]["distance_m"] == 0.1

    other = _embed(client, jpeg(320, 240, (200, 60, 200)))
    m2 = client.post("/v1/plants/match", json={"embedding": other}).json()
    assert m2["matches"][0]["similarity"] < m2["threshold"]

    # a second visit to the same plant appends, it doesn't create plant_002
    r = client.post("/v1/plants/enroll", data={"label": "plant_001", "meta": "{}"},
                    files=[("images", ("x.jpg", fern[0], "image/jpeg"))])
    assert r.json()["visits"] == 2 and r.json()["views_total"] == 4

    listing = client.get("/v1/plants").json()
    assert listing["plants"][0]["position"] == [1.0, 2.0]
    rec = client.get("/v1/plants/plant_001").json()
    photo = rec["visits"][0]["views"][1]
    assert photo["pose"] == [1, 0, 3.1]
    img = client.get(f"/v1/plants/plant_001/{photo['file']}")
    assert img.status_code == 200 and img.headers["content-type"] == "image/png"

    # photos are filed plants/map_{index}/plant_{NNN}/photo_{i}.png with a bbox_{i}.txt twin
    from vision_service.settings import settings
    folder = settings.plant_dir / "map_1" / "plant_001"
    assert sorted(p.name for p in folder.glob("photo_*.png")) == ["photo_1.png", "photo_2.png", "photo_3.png", "photo_4.png"]
    assert sorted(p.name for p in folder.glob("bbox_*.txt")) == [f"bbox_{i}.txt" for i in range(1, 5)]

    assert client.delete("/v1/plants/plant_001").status_code == 200
    assert client.get("/v1/plants").json()["plants"] == []


def test_plant_labels_cannot_escape_the_folder(client):
    r = client.post("/v1/plants/enroll", data={"label": "../etc", "meta": "{}"},
                    files=[("images", ("x.jpg", jpeg(), "image/jpeg"))])
    assert r.status_code == 422
    assert client.get("/v1/plants/plant_001/..%2Fgallery.npz").status_code in (404, 422)


def test_ask_without_providers_is_503(client):
    r = client.post("/v1/ask", data={"question": "what is this?"},
                    files=[("images", ("x.jpg", jpeg(), "image/jpeg"))])
    assert r.status_code == 503


def test_ask_uses_first_working_provider(client, monkeypatch):
    import vision_service.app as app_mod
    vlm = app_mod.app.state.vlm
    vlm.providers = ["gemini", "ollama"]
    monkeypatch.setattr(vlm, "_gemini", lambda *a: (_ for _ in ()).throw(TimeoutError("offline")))
    monkeypatch.setattr(vlm, "_ollama", lambda imgs, q, j: ('```json\n{"same": true}\n```', "qwen3-vl:8b"))
    r = client.post("/v1/ask", data={"question": "same plant?", "json": "true"},
                    files=[("images", ("a.jpg", jpeg(), "image/jpeg")),
                           ("images", ("b.jpg", jpeg(), "image/jpeg"))])
    assert r.status_code == 200, r.text
    assert r.json()["provider"] == "ollama" and r.json()["parsed"] == {"same": True}


def test_slam_round_trip(client):
    rays = [[math.radians(b), 1.5] for b in range(-180, 180, 2)]      # round room, r = 1.5 m
    out = client.post("/v1/slam/keyframe", json={"odom": [0, 0, 0], "rays": rays}).json()
    assert out["kf_id"] == 0 and "goal" in out
    out = client.post("/v1/slam/keyframe", json={"odom": [0.02, 0, 0.01], "rays": rays}).json()
    assert out["match"]["accepted"]
    assert client.post("/v1/slam/plan", json={"start": [0, 0], "goal": [0.5, 0.3]}).status_code == 200
    assert client.post("/v1/slam/plan", json={"start": [0, 0], "goal": [5, 5]}).status_code == 422
    vp = client.post("/v1/slam/viewpoints", json={"start": [0, 0], "target": [0.3, 0.0],
                                                  "radius": 0.6, "n": 4}).json()
    assert vp["requested"] == 4
    png = client.get("/v1/slam/map.png")
    assert png.status_code == 200 and png.content[:4] == b"\x89PNG"
    assert client.post("/v1/slam/save").status_code == 200
    st = client.post("/v1/slam/reset", json={"load": True, "initial_pose": [0, 0, 0]}).json()
    assert st["keyframes"] == 0 and st["explored_m2"] > 1.0
    assert client.post("/v1/slam/keyframe", json={"odom": [0, 0, 0], "rays": [[0.0]]}).status_code == 422


def test_a_new_slam_map_starts_a_new_map_folder(client):
    from vision_service.settings import settings
    body = {"name": "home", "initial_pose": [0, 0, 0], "load": False}
    assert client.post("/v1/slam/reset", json=body).json()["map_index"] == 1
    client.post("/v1/plants/enroll", data={"meta": "{}"}, files=[("images", ("x.jpg", jpeg(), "image/jpeg"))])
    assert (settings.plant_dir / "map_1" / "plant_001" / "photo_1.png").is_file()

    assert client.post("/v1/slam/reset", json={**body, "name": "garage"}).json()["map_index"] == 2
    r = client.post("/v1/plants/enroll", data={"meta": "{}"}, files=[("images", ("x.jpg", jpeg(), "image/jpeg"))])
    assert r.json()["label"] == "plant_002"            # labels stay unique across maps
    assert (settings.plant_dir / "map_2" / "plant_002" / "photo_1.png").is_file()
    assert client.post("/v1/slam/reset", json={**body, "load": False}).json()["map_index"] == 3


def test_every_keyframe_saves_the_map_on_the_mac(client):
    from vision_service.settings import settings
    client.post("/v1/slam/reset", json={"name": "home", "initial_pose": [0, 0, 0]})
    rays = [[i * 0.1, 2.0] for i in range(-30, 30)]
    r = client.post("/v1/slam/keyframe", json={"odom": [0, 0, 0], "rays": rays})
    assert r.status_code == 200, r.text
    assert (settings.slam_dir / "home.npz").is_file() and (settings.slam_dir / "home.png").read_bytes()[:4] == b"\x89PNG"


def test_two_plants_on_one_stand_keep_their_heights(client):
    """Same x, y; different shelves. The match gives each one's height back so the robot can tell them apart."""
    low, high = jpeg(320, 240, (40, 140, 60)), jpeg(320, 240, (45, 150, 70))
    for img, z in ((low, 0.0), (high, 0.7)):
        meta = {"position": [1.0, 2.0, z], "height_m": z, "on_stand": True}
        r = client.post("/v1/plants/enroll", data={"meta": json.dumps(meta)},
                        files=[("images", ("p.jpg", img, "image/jpeg"))])
        assert r.status_code == 200, r.text
    listing = {p["label"]: p["position"] for p in client.get("/v1/plants").json()["plants"]}
    assert listing == {"plant_001": [1.0, 2.0, 0.0], "plant_002": [1.0, 2.0, 0.7]}
    m = client.post("/v1/plants/match", json={"embedding": _embed(client, high), "position": [1.0, 2.0, 0.7],
                                              "top_k": 2}).json()
    by = {x["label"]: x for x in m["matches"]}
    assert by["plant_002"]["position"] == [1.0, 2.0, 0.7] and by["plant_001"]["position"] == [1.0, 2.0, 0.0]
    assert by["plant_002"]["distance_m"] == 0.0 and by["plant_001"]["distance_m"] == 0.0     # along the floor
    assert client.get("/v1/slam/map.png").content[:4] == b"\x89PNG"


def test_keyframe_takes_the_tof_layers(client):
    rays = [[math.radians(a), 2.0] for a in range(0, 360, 10)]
    r = client.post("/v1/slam/keyframe", json={"odom": [0, 0, 0], "rays": rays, "low": [[0.6, 0.0]],
                                               "drops": [[0.0, 0.7]], "clear": [[0.1, 1.5]],
                                               "cloud": [[0.6, 0.0, 0.08]]})
    assert r.status_code == 200, r.text
    st = client.get("/v1/slam/state").json()
    assert st["low_cells"] == 1 and st["drop_cells"] == 1 and st["cloud_points"] == 1
    bad = client.post("/v1/slam/keyframe", json={"odom": [0, 0, 0], "rays": rays, "cloud": [[1.0, 2.0]]})
    assert bad.status_code == 422


def test_plant_position_is_the_median_of_its_visits():
    from vision_service.plants import combine_positions
    pos, spread = combine_positions([[1.0, 2.0, 0.3], [1.1, 2.1, 0.32], [3.0, 5.0, 0.3]])
    assert pos == [1.1, 2.1, 0.3] and abs(spread - ((1.9 ** 2 + 2.9 ** 2) ** 0.5)) < 1e-3
    assert combine_positions([[1.0, 2.0]]) == ([1.0, 2.0], 0.0)
