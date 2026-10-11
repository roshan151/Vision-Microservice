from __future__ import annotations

from conftest import jpeg


def _embed(client, img, seq=1):
    r = client.post("/v1/faces/embed", files={"image": ("c.jpg", img, "image/jpeg")},
                    data={"seq": seq})
    assert r.status_code == 200, r.text
    return r.json()


def test_embed_shape_and_quality(client):
    body = _embed(client, jpeg(160, 160, (200, 180, 170)))
    assert body["seq"] == 1
    assert len(body["embedding"]) == 512
    assert 0.0 <= body["quality"] <= 1.0
    assert len(body["face_bbox"]) == 4


def test_embed_404s_when_there_is_no_face(client):
    r = client.post("/v1/faces/embed", files={"image": ("c.jpg", jpeg(8, 8), "image/jpeg")},
                    data={"seq": 5})
    assert r.status_code == 404


def test_enroll_then_match_round_trip(client):
    face = jpeg(160, 160, (200, 180, 170))
    r = client.post("/v1/faces/enroll", data={"label": "roshan"},
                    files=[("images", ("a.jpg", face, "image/jpeg"))])
    assert r.status_code == 200, r.text
    assert r.json() == {"label": "roshan", "embeddings_added": 1,
                        "images_rejected": 0, "gallery_size": 1}

    # The same crop must match itself at ~1.0 — this is the check that catches a
    # normalisation bug, which otherwise shows up as "recognition just doesn't work".
    emb = _embed(client, face)["embedding"]
    body = client.post("/v1/faces/match", json={"embedding": emb, "top_k": 1}).json()
    assert body["matches"][0]["label"] == "roshan"
    assert body["matches"][0]["similarity"] > 0.99
    assert body["threshold"] == 0.45


def test_a_stranger_scores_below_threshold(client):
    client.post("/v1/faces/enroll", data={"label": "roshan"},
                files=[("images", ("a.jpg", jpeg(160, 160, (200, 180, 170)), "image/jpeg"))])
    stranger = _embed(client, jpeg(160, 160, (20, 90, 240)), seq=9)["embedding"]
    body = client.post("/v1/faces/match", json={"embedding": stranger}).json()
    assert body["matches"][0]["similarity"] < body["threshold"]


def test_match_on_empty_gallery_returns_no_matches(client):
    emb = _embed(client, jpeg(160, 160))["embedding"]
    body = client.post("/v1/faces/match", json={"embedding": emb}).json()
    assert body["matches"] == []
    assert body["gallery_size"] == 0


def test_enroll_counts_rejects_without_failing_the_batch(client):
    good = jpeg(160, 160, (200, 180, 170))
    bad = jpeg(8, 8)
    r = client.post("/v1/faces/enroll", data={"label": "mixed"}, files=[
        ("images", ("a.jpg", good, "image/jpeg")),
        ("images", ("b.jpg", bad, "image/jpeg")),
    ])
    assert r.status_code == 200
    assert r.json()["embeddings_added"] == 1
    assert r.json()["images_rejected"] == 1


def test_enroll_422s_when_every_image_is_rejected(client):
    r = client.post("/v1/faces/enroll", data={"label": "nobody"},
                    files=[("images", ("b.jpg", jpeg(8, 8), "image/jpeg"))])
    assert r.status_code == 422


def test_list_and_delete(client):
    client.post("/v1/faces/enroll", data={"label": "roshan"},
                files=[("images", ("a.jpg", jpeg(160, 160, (200, 180, 170)), "image/jpeg"))])
    assert client.get("/v1/faces").json()["labels"] == ["roshan"]
    assert client.delete("/v1/faces/roshan").json()["embeddings_removed"] == 1
    assert client.get("/v1/faces").json()["labels"] == []
    assert client.delete("/v1/faces/roshan").status_code == 404


def test_match_rejects_malformed_bodies(client):
    assert client.post("/v1/faces/match", json={}).status_code == 422
    assert client.post("/v1/faces/match", json={"embedding": []}).status_code == 422
    assert client.post("/v1/faces/match",
                       json={"embedding": [0.1] * 512, "top_k": 0}).status_code == 422


def test_match_rejects_wrong_dimension(client):
    client.post("/v1/faces/enroll", data={"label": "roshan"},
                files=[("images", ("a.jpg", jpeg(160, 160, (200, 180, 170)), "image/jpeg"))])
    r = client.post("/v1/faces/match", json={"embedding": [0.1] * 128})
    assert r.status_code == 422
    assert "dim" in r.json()["detail"]


def test_multiple_faces_are_reported_and_refused_for_enrollment(client):
    two = jpeg(400, 160, (200, 180, 170))       # stub: wide crop == two faces
    r = client.post("/v1/faces/embed", files={"image": ("c.jpg", two, "image/jpeg")},
                    data={"seq": 1})
    assert r.json()["face_count"] == 2
    r = client.post("/v1/faces/enroll", data={"label": "crowd"},
                    files=[("images", ("a.jpg", two, "image/jpeg"))])
    assert r.status_code == 422


def test_enrolled_face_photos_are_saved_with_bboxes(client):
    from vision_service.settings import settings
    for _ in range(2):        # a second enroll continues the numbering
        r = client.post("/v1/faces/enroll", data={"label": "roshan"},
                        files=[("images", ("a.jpg", jpeg(200, 200, (10, 20, 30)), "image/jpeg"))])
        assert r.status_code == 200
    folder = settings.face_photo_dir / "roshan"
    assert sorted(p.name for p in folder.glob("photo_*.png")) == ["photo_1.png", "photo_2.png"]
    assert (folder / "bbox_1.txt").read_text() == "0.0 0.0 200.0 200.0"


def test_face_label_cannot_escape_the_folder(client):
    r = client.post("/v1/faces/enroll", data={"label": "../x"},
                    files=[("images", ("a.jpg", jpeg(200, 200), "image/jpeg"))])
    assert r.status_code == 422


def test_low_quality_embed_is_kept_for_inspection(client, monkeypatch):
    from vision_service import settings as settings_mod
    monkeypatch.setattr(settings_mod.settings, "face_min_quality", 0.95)   # stub quality is 0.9
    _embed(client, jpeg(160, 160, (200, 180, 170)), seq=7)
    folder = settings_mod.settings.face_photo_dir / "_rejected"
    rows = (folder / "quality.csv").read_text().splitlines()
    assert rows[0] == "photo,time,seq,quality,face_count"
    photo, _, seq, quality, count = rows[1].split(",")
    assert (seq, quality, count) == ("7", "0.900", "1")
    assert (folder / photo).exists()


def test_good_embed_is_not_kept(client):
    from vision_service import settings as settings_mod
    _embed(client, jpeg(160, 160, (200, 180, 170)))
    assert not (settings_mod.settings.face_photo_dir / "_rejected").exists()
