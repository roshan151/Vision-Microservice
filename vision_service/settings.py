"""Configuration, all of it from the environment.

Defaults are the ones VISION-DESIGN.md's appendix assumes. Anything you change
here is a number the Pi side also believes, so change both or neither.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_f(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_i(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


# Not frozen: tests override fields directly, and a config object you cannot
# point at a temp directory is a config object that makes tests touch real state.
@dataclass
class Settings:
    # --- serving ---------------------------------------------------------
    host: str = field(default_factory=lambda: _env("VS_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env_i("VS_PORT", 8080))

    # --- backend selection ----------------------------------------------
    # "auto"  -> real models if importable, else fake (with a loud warning)
    # "real"  -> real models, fail loudly at startup if they will not load
    # "fake"  -> deterministic stubs. No models, no GPU. What CI uses.
    backend: str = field(default_factory=lambda: _env("VS_BACKEND", "auto"))

    # --- models ----------------------------------------------------------
    detect_model: str = field(default_factory=lambda: _env("VS_DETECT_MODEL", "yolov8n.pt"))
    detect_imgsz: int = field(default_factory=lambda: _env_i("VS_DETECT_IMGSZ", 640))
    face_model: str = field(default_factory=lambda: _env("VS_FACE_MODEL", "buffalo_l"))
    depth_model: str = field(
        default_factory=lambda: _env("VS_DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Small-hf")
    )
    device: str = field(default_factory=lambda: _env("VS_DEVICE", "mps"))

    # --- gallery ---------------------------------------------------------
    gallery_path: Path = field(
        default_factory=lambda: Path(_env("VS_GALLERY_PATH", "~/.robot-vision/gallery.npz")).expanduser()
    )
    match_threshold: float = field(default_factory=lambda: _env_f("VS_MATCH_THRESHOLD", 0.75))

    # --- limits ----------------------------------------------------------
    # Bounded intake. The Pi sends ~30 KB frames; anything near this cap means
    # the Pi is misconfigured and it is cheaper to say so than to process it.
    max_upload_bytes: int = field(default_factory=lambda: _env_i("VS_MAX_UPLOAD_BYTES", 4 * 1024 * 1024))
    max_enroll_images: int = field(default_factory=lambda: _env_i("VS_MAX_ENROLL_IMAGES", 16))

    # Server-side backpressure. One GPU means inference is serialized; if more
    # than this many requests are already waiting, new ones are refused with 503
    # rather than queued. This is the server half of PLAN.md standing rule #8 —
    # a queue in front of a slow resource turns a hiccup into stale answers.
    max_queue_depth: int = field(default_factory=lambda: _env_i("VS_MAX_QUEUE_DEPTH", 4))
    inference_timeout_s: float = field(default_factory=lambda: _env_f("VS_INFERENCE_TIMEOUT_S", 10.0))


settings = Settings()
