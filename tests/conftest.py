"""Every test runs against a stub backend (tests/stub_backend.py): no models,
no GPU, no network.

If a change breaks the API contract, this suite catches it in about a second.
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # Point the gallery at a temp file BEFORE the app's lifespan builds it, so
    # tests never touch a real gallery and never see each other's enrollments.
    from vision_service import settings as settings_mod

    monkeypatch.setattr(settings_mod.settings, "gallery_path", tmp_path / "gallery.npz",
                        raising=False)
    monkeypatch.setattr(settings_mod.settings, "plant_dir", tmp_path / "plants", raising=False)
    monkeypatch.setattr(settings_mod.settings, "face_photo_dir", tmp_path / "faces", raising=False)
    monkeypatch.setattr(settings_mod.settings, "slam_dir", tmp_path / "maps", raising=False)
    monkeypatch.setattr(settings_mod.settings, "vlm_providers", "", raising=False)
    import vision_service.app as app_mod
    from stub_backend import build_stub_backend

    monkeypatch.setattr(app_mod, "build_backend", build_stub_backend)
    app = app_mod.app

    with TestClient(app) as c:
        yield c


def jpeg(width: int = 640, height: int = 360, colour: tuple[int, int, int] = (120, 140, 160)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(buf, format="JPEG", quality=75)
    return buf.getvalue()


@pytest.fixture()
def frame() -> bytes:
    return jpeg()
