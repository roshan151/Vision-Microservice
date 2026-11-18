"""Debug uploads from the robot (robot.yaml /voice debug_mode): the camera frames and
ToF grids a task collects, stored as they come for Monitoring-Service to show. No
inference here, and nothing is decoded: the JPEG is written exactly as it arrived.

    POST /v1/debug/frame     meta (JSON form field) [+ image (JPEG)] -> stored
    GET  /v1/debug/latest    the newest stored upload's meta

On disk, under VS_DEBUG_DIR (default: debug/ next to the code), one folder per robot
session (the robot names it by its start time):

    debug/<session>/000001.json     meta + "received" (this Mac's clock)
    debug/<session>/000001.jpg      the frame, if the upload had one
    debug/latest.json               {"session", "n", "json", "jpg"} of the newest

The .jpg is written before the .json and latest.json is replaced atomically, so a
reader never finds a half-written frame. Only the newest VS_DEBUG_KEEP sessions are
kept: starting a new one deletes the oldest beyond that.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from .settings import settings

log = logging.getLogger("vision_service.debug")
router = APIRouter()
_SESSION = re.compile(r"^[0-9A-Za-z_-]{1,40}$")


def _write(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _prune(root: Path, keep: int) -> None:
    sessions = sorted(p for p in root.iterdir() if p.is_dir() and _SESSION.match(p.name))
    for old in sessions[:-keep] if keep > 0 else []:
        shutil.rmtree(old, ignore_errors=True)
        log.info("debug: removed old session %s (keeping %d)", old.name, keep)


def _store(meta: dict, image: Optional[bytes]) -> dict:
    root: Path = settings.debug_dir
    folder = root / meta["session"]
    if not folder.exists():
        folder.mkdir(parents=True)
        _prune(root, settings.debug_keep)
    stem = f"{int(meta['n']):06d}"
    meta["received"] = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    if image:
        _write(folder / f"{stem}.jpg", image)
    meta["image"] = f"{stem}.jpg" if image else None
    _write(folder / f"{stem}.json", json.dumps(meta, separators=(",", ":")).encode())
    latest = {"session": meta["session"], "n": meta["n"], "json": f"{meta['session']}/{stem}.json",
              "jpg": f"{meta['session']}/{stem}.jpg" if image else None}
    _write(root / "latest.json", json.dumps(latest).encode())
    return latest


@router.post("/v1/debug/frame", tags=["debug"])
async def debug_frame(meta: str = Form(...), image: Optional[UploadFile] = File(None)) -> dict:
    try:
        m = json.loads(meta)
        if not _SESSION.match(str(m.get("session", ""))) or int(m.get("n", -1)) < 0:
            raise ValueError
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail="meta must be JSON with a session name and n >= 0") from exc
    data = await image.read() if image is not None else None
    if data and len(data) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="image too large")
    return await run_in_threadpool(_store, m, data)


@router.get("/v1/debug/latest", tags=["debug"])
async def debug_latest() -> dict:
    p = settings.debug_dir / "latest.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="no debug uploads yet")
    latest = json.loads(p.read_text())
    return json.loads((settings.debug_dir / latest["json"]).read_text())
