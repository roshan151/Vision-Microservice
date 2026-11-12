"""reqlog: one line per request with inputs and outputs summarised."""

from __future__ import annotations

import logging

from conftest import jpeg


def test_each_request_logs_its_input_and_output(client, caplog):
    caplog.set_level(logging.INFO, logger="vision_service.req")
    logging.getLogger("vision_service").propagate = True       # let caplog see it
    try:
        r = client.post("/v1/faces/embed", files={"image": ("frame.jpg", jpeg(1920, 1080), "image/jpeg")},
                        data={"seq": "7"})
    finally:
        logging.getLogger("vision_service").propagate = False
    lines = {rec.getMessage() for rec in caplog.records if rec.name == "vision_service.req"}
    assert len(lines) == 1                                     # one line per request
    line = lines.pop()
    assert "POST /v1/faces/embed" in line and f"-> {r.status_code}" in line
    assert "image=<frame.jpg" in line and "1920x1080" in line and "seq=7" in line
    if r.status_code == 200:
        assert '"embedding": "<512 numbers>"' in line                     # embedding squashed
