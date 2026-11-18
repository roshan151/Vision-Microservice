"""Debug uploads: stored as sent, latest.json points at the newest, old sessions pruned."""

import json


def _post(client, session, n, image=None):
    meta = {"session": session, "n": n, "ts": "2026-11-18T18:00:00.000+00:00",
            "tof": {"age_ms": 40, "rows": [[1.0] * 8] * 8}}
    files = {"image": ("f.jpg", image, "image/jpeg")} if image else None
    return client.post("/v1/debug/frame", data={"meta": json.dumps(meta)}, files=files)


def test_frame_is_stored_as_sent_and_latest_points_at_it(client, frame, tmp_path, monkeypatch):
    from vision_service.settings import settings
    monkeypatch.setattr(settings, "debug_dir", tmp_path / "debug")
    assert client.get("/v1/debug/latest").status_code == 404
    r = _post(client, "20261118-100000", 1, frame)
    assert r.status_code == 200 and r.json()["jpg"] == "20261118-100000/000001.jpg"
    assert (tmp_path / "debug/20261118-100000/000001.jpg").read_bytes() == frame
    r = _post(client, "20261118-100000", 2)                  # a ToF grid on its own
    assert r.status_code == 200 and r.json()["jpg"] is None
    latest = client.get("/v1/debug/latest").json()
    assert latest["n"] == 2 and latest["image"] is None and latest["tof"]["rows"][0][0] == 1.0
    assert "received" in latest


def test_bad_meta_is_refused_and_old_sessions_are_pruned(client, frame, tmp_path, monkeypatch):
    from vision_service.settings import settings
    monkeypatch.setattr(settings, "debug_dir", tmp_path / "debug")
    monkeypatch.setattr(settings, "debug_keep", 2)
    assert client.post("/v1/debug/frame", data={"meta": "{}"}).status_code == 422
    assert _post(client, "../etc", 1, frame).status_code == 422
    for s in ("20261118-100000", "20261118-110000", "20261118-120000"):
        assert _post(client, s, 1, frame).status_code == 200
    assert sorted(p.name for p in (tmp_path / "debug").iterdir() if p.is_dir()) == \
        ["20261118-110000", "20261118-120000"]
