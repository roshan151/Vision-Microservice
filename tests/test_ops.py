from __future__ import annotations


def test_healthz(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["backend"] == "fake"
    assert body["queue_depth"] == 0
    assert set(body["versions"]) == {"detect", "face", "depth"}


def test_models_endpoint(client):
    body = client.get("/v1/models").json()
    assert body["backend"] == "fake"
    assert isinstance(body["depth_is_metric"], bool)


def test_openapi_documents_every_endpoint(client):
    paths = client.get("/openapi.json").json()["paths"]
    for p in ("/v1/detect", "/v1/faces/embed", "/v1/faces/match",
              "/v1/faces/enroll", "/v1/depth", "/healthz", "/v1/models"):
        assert p in paths, f"{p} missing from the OpenAPI schema"
