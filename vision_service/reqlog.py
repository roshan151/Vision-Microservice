"""One timestamped log line per request: what came in, what went out, how long it took.

    2026-11-10 12:30:01.482 INFO  vision_service.req: 192.168.1.40 POST /v1/faces/embed -> 404 in 212 ms
        | in: image=<frame.jpg 412803 B 1920x1080> seq=7 | out: {"error": "...", "detail": "no face found in crop"}

Plain ASGI middleware: it copies the body chunks as they pass and changes nothing.
Inputs are summarised (an uploaded image becomes its name, size and pixel size) and
outputs are shortened (an embedding becomes "<512 numbers>"), so a line stays
readable at follow mode's 2 requests a second. /healthz is logged at DEBUG only.
"""

from __future__ import annotations

import io
import json
import logging
import re
import time

log = logging.getLogger("vision_service.req")

MAX_TEXT = 600           # a summary longer than this is cut


def setup() -> None:
    """Timestamps on every vision_service.* line (this module's and the existing
    log.info calls, which had no handler under uvicorn and went nowhere)."""
    root = logging.getLogger("vision_service")
    if any(getattr(h, "_vs", False) for h in root.handlers):
        return
    h = logging.StreamHandler()
    h._vs = True                                          # type: ignore[attr-defined]
    h.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)-5s %(name)s: %(message)s",
                                     "%Y-%m-%d %H:%M:%S"))
    root.addHandler(h)
    root.setLevel(logging.INFO)
    root.propagate = False


class RequestLog:
    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        t0, body_in, body_out, meta = time.monotonic(), bytearray(), bytearray(), {}

        async def recv():
            msg = await receive()
            if msg["type"] == "http.request":
                body_in.extend(msg.get("body", b""))
            return msg

        async def snd(msg):
            if msg["type"] == "http.response.start":
                meta["status"] = msg["status"]
                meta["ctype"] = _header(msg.get("headers", []), b"content-type")
            elif msg["type"] == "http.response.body":
                body_out.extend(msg.get("body", b""))
            await send(msg)

        try:
            await self.app(scope, recv, snd)
        finally:
            try:
                path = scope.get("path", "")
                if scope.get("query_string"):
                    path += "?" + scope["query_string"].decode("latin-1")
                client = (scope.get("client") or ("?",))[0]
                ctype_in = _header(scope.get("headers", []), b"content-type")
                log.log(logging.DEBUG if scope.get("path") == "/healthz" else logging.INFO,
                        "%s %s %s -> %s in %.0f ms | in: %s | out: %s",
                        client, scope.get("method", ""), path, meta.get("status", "no response"),
                        (time.monotonic() - t0) * 1000,
                        _summary(bytes(body_in), ctype_in), _summary(bytes(body_out), meta.get("ctype", "")))
            except Exception:                              # noqa: BLE001 — logging must never break a request
                log.exception("could not log request")


def _header(headers, name: bytes) -> str:
    for k, v in headers:
        if k.lower() == name:
            return v.decode("latin-1")
    return ""


def _summary(body: bytes, ctype: str) -> str:
    if not body:
        return "-"
    if "json" in ctype:
        try:
            return _cut(json.dumps(_squash(json.loads(body))))
        except ValueError:
            pass
    if "multipart/form-data" in ctype:
        return _cut(_multipart(body, ctype))
    if ctype.startswith("text/") or "x-www-form-urlencoded" in ctype:
        return _cut(body.decode("utf-8", "replace"))
    return f"<{ctype or 'bytes'} {len(body)} B{_dims(body)}>"


def _multipart(body: bytes, ctype: str) -> str:
    m = re.search(r'boundary="?([^";]+)"?', ctype)
    if not m:
        return f"<multipart {len(body)} B>"
    out = []
    for part in body.split(b"--" + m.group(1).encode())[1:]:
        head, sep, data = part.partition(b"\r\n\r\n")
        if not sep:
            continue                                     # the closing "--"
        data = data[:-2] if data.endswith(b"\r\n") else data
        h = head.decode("latin-1")
        name = re.search(r'\bname="([^"]*)"', h)
        fname = re.search(r'filename="([^"]*)"', h)
        key = name.group(1) if name else "?"
        if fname:
            out.append(f"{key}=<{fname.group(1)} {len(data)} B{_dims(data)}>")
        else:
            out.append(f"{key}={data.decode('utf-8', 'replace')[:120]}")
    return " ".join(out)


def _dims(data: bytes) -> str:
    try:
        from PIL import Image
        w, h = Image.open(io.BytesIO(data)).size
        return f" {w}x{h}"
    except Exception:                                      # noqa: BLE001 — not an image
        return ""


def _squash(v):
    if isinstance(v, list):
        if len(v) > 8 and all(isinstance(x, (int, float)) for x in v):
            return f"<{len(v)} numbers>"
        return [_squash(x) for x in v[:20]] + ([f"...+{len(v) - 20}"] if len(v) > 20 else [])
    if isinstance(v, dict):
        return {k: _squash(x) for k, x in v.items()}
    if isinstance(v, str) and len(v) > 120:
        return v[:120] + "..."
    return v


def _cut(text: str) -> str:
    return text if len(text) <= MAX_TEXT else text[:MAX_TEXT] + "..."
