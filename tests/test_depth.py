from __future__ import annotations


def test_depth_returns_one_value_per_point(client, frame):
    r = client.post("/v1/depth", files={"image": ("f.jpg", frame, "image/jpeg")},
                    data={"seq": 77, "points": "[[320,180],[100,90]]"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["seq"] == 77
    assert len(body["depths_m"]) == len(body["confidence"]) == 2
    # The Pi branches on this flag, not on the field name.
    assert isinstance(body["metric"], bool)


def test_depth_rejects_bad_points(client, frame):
    def post(points):
        return client.post("/v1/depth", files={"image": ("f.jpg", frame, "image/jpeg")},
                           data={"seq": 1, "points": points})

    assert post("not json").status_code == 422
    assert post("[]").status_code == 422
    assert post('[["a","b"]]').status_code == 422
    assert post("[[1,2]]" .replace("[[1,2]]", "[" + ",".join(["[1,2]"] * 65) + "]")).status_code == 422
