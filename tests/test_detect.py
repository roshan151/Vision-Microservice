from __future__ import annotations

from conftest import jpeg


def test_detect_echoes_seq_and_image_size(client, frame):
    r = client.post("/v1/detect", files={"image": ("f.jpg", frame, "image/jpeg")},
                    data={"seq": 10432, "min_confidence": 0.4})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["seq"] == 10432
    assert body["image_size"] == [640, 360]
    assert body["model_version"]
    assert body["inference_ms"] >= 0


def test_detect_bboxes_lie_inside_the_image(client, frame):
    body = client.post("/v1/detect", files={"image": ("f.jpg", frame, "image/jpeg")},
                       data={"seq": 1}).json()
    w, h = body["image_size"]
    for d in body["detections"]:
        x1, y1, x2, y2 = d["bbox"]
        assert 0 <= x1 < x2 <= w
        assert 0 <= y1 < y2 <= h
        assert "class" in d and 0.0 <= d["confidence"] <= 1.0


def test_min_confidence_filters(client, frame):
    low = client.post("/v1/detect", files={"image": ("f.jpg", frame, "image/jpeg")},
                      data={"seq": 1, "min_confidence": 0.1}).json()
    high = client.post("/v1/detect", files={"image": ("f.jpg", frame, "image/jpeg")},
                       data={"seq": 2, "min_confidence": 0.85}).json()
    assert len(high["detections"]) < len(low["detections"])


def test_seq_is_required(client, frame):
    r = client.post("/v1/detect", files={"image": ("f.jpg", frame, "image/jpeg")})
    assert r.status_code == 422


def test_rejects_undecodable_body(client):
    r = client.post("/v1/detect", files={"image": ("f.jpg", b"not a jpeg", "image/jpeg")},
                    data={"seq": 1})
    assert r.status_code == 400
    assert r.json()["error"] == "image_undecodable"


def test_rejects_oversized_image(client, monkeypatch):
    from vision_service import settings as settings_mod

    monkeypatch.setattr(settings_mod.settings, "max_upload_bytes", 2048, raising=False)
    big = jpeg(1920, 1080, (7, 200, 30))
    r = client.post("/v1/detect", files={"image": ("f.jpg", big, "image/jpeg")}, data={"seq": 1})
    assert r.status_code == 413
    assert r.json()["error"] == "image_too_large"


def test_same_image_gives_same_answer(client, frame):
    a = client.post("/v1/detect", files={"image": ("f.jpg", frame, "image/jpeg")}, data={"seq": 1}).json()
    b = client.post("/v1/detect", files={"image": ("f.jpg", frame, "image/jpeg")}, data={"seq": 2}).json()
    assert a["detections"] == b["detections"]
