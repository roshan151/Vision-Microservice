"""Configuration, all of it from the environment.

Defaults are the ones VISION-DESIGN.md's appendix assumes. Anything you change
here is a number the Pi side also believes, so change both or neither.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Photos live next to the code (plants/map_N/..., faces/<name>/...), not in ~/.robot-vision.
_REPO = Path(__file__).resolve().parent.parent


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

    # --- models ----------------------------------------------------------
    detect_model: str = field(default_factory=lambda: _env("VS_DETECT_MODEL", "yolov8m.pt"))
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
    # buffalo_l (ArcFace) cosine. Same person across lighting/pose scores ~0.4-0.7, so
    # 0.75 only ever matched near-duplicates of the enrolment frames. Follow also needs
    # CONFIRM_FRAMES agreeing frames, which is what makes a lower cut-off safe.
    match_threshold: float = field(default_factory=lambda: _env_f("VS_MATCH_THRESHOLD", 0.45))
    # A clear face scoring below this against the follow target is someone else:
    # follow mode will not track that person by their clothes.
    face_reject_threshold: float = field(default_factory=lambda: _env_f("VS_FACE_REJECT_THRESHOLD", 0.25))
    # Face detector input size. 640 keeps small faces findable in 1080p frames.
    face_det_size: int = field(default_factory=lambda: _env_i("VS_FACE_DET_SIZE", 640))
    # Same cut-off the Pi uses (FACE_MIN_QUALITY): /v1/faces/embed keeps frames below it
    # in faces/_rejected/ so a failed enroll can be inspected afterwards.
    face_min_quality: float = field(default_factory=lambda: _env_f("VS_FACE_MIN_QUALITY", 0.6))

    # --- plants ----------------------------------------------------------
    # DINOv2 is trained for instance-level visual similarity, which is exactly
    # "is this the same plant from another side" — CLIP-style models match the
    # *category* (any fern) far more readily than the individual.
    plant_model: str = field(default_factory=lambda: _env("VS_PLANT_MODEL", "facebook/dinov2-small"))
    plant_dir: Path = field(
        default_factory=lambda: Path(_env("VS_PLANT_DIR", str(_REPO / "plants"))).expanduser()
    )
    face_photo_dir: Path = field(
        default_factory=lambda: Path(_env("VS_FACE_PHOTO_DIR", str(_REPO / "faces"))).expanduser()
    )
    plant_match_threshold: float = field(default_factory=lambda: _env_f("VS_PLANT_MATCH_THRESHOLD", 0.62))

    # --- image + text model (POST /v1/ask) -----------------------------------
    # Tried in order until one answers. "gemini" needs GEMINI_API_KEY; "ollama"
    # needs `ollama serve` with the model pulled (a 4-bit 7-8B VLM, e.g.
    # `ollama pull qwen3-vl:8b`).
    vlm_providers: str = field(default_factory=lambda: _env("VS_VLM_PROVIDERS", "gemini,ollama"))
    gemini_model: str = field(default_factory=lambda: _env("VS_GEMINI_MODEL", "gemini-3.6-flash"))
    gemini_api_key: str = field(
        default_factory=lambda: os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))
    )
    ollama_url: str = field(default_factory=lambda: _env("VS_OLLAMA_URL", "http://127.0.0.1:11434"))
    ollama_model: str = field(default_factory=lambda: _env("VS_OLLAMA_MODEL", "qwen3-vl:8b"))
    vlm_timeout_s: float = field(default_factory=lambda: _env_f("VS_VLM_TIMEOUT_S", 45.0))

    # --- SLAM ------------------------------------------------------------
    slam_dir: Path = field(
        default_factory=lambda: Path(_env("VS_SLAM_DIR", str(_REPO / "maps"))).expanduser()
    )
    slam_size_m: float = field(default_factory=lambda: _env_f("VS_SLAM_SIZE_M", 24.0))
    slam_res_m: float = field(default_factory=lambda: _env_f("VS_SLAM_RES_M", 0.05))
    robot_radius_m: float = field(default_factory=lambda: _env_f("VS_ROBOT_RADIUS_M", 0.18))

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
