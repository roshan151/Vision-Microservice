"""Every test runs against the fake backend: no models, no GPU, no network.

That is the point of the backend seam. If a change breaks the contract, this
suite catches it on a laptop in under a second.
"""

from __future__ import annotations

import io
import os

import pytest
from fastapi.testclient import TestClient
from PIL import Image

os.environ["VS_BACKEND"] = "fake"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # Point the gallery at a temp file BEFORE the app's lifespan builds it, so
    # tests never touch a real gallery and never see each other's enrollments.
    from vision_service import settings as settings_mod

    monkeypatch.setattr(settings_mod.settings, "gallery_path", tmp_path / "gallery.npz",
                        raising=False)
    from vision_service.app import app

    with TestClient(app) as c:
        yield c


def jpeg(width: int = 640, height: int = 360, colour: tuple[int, int, int] = (120, 140, 160)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(buf, format="JPEG", quality=75)
    return buf.getvalue()


@pytest.fixture()
def frame() -> bytes:
    return jpeg()
