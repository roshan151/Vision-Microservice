"""Image intake: bytes in, RGB ndarray out, with the size limit enforced early."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, UnidentifiedImageError

from .settings import settings


class ImageTooLarge(ValueError):
    pass


class ImageUndecodable(ValueError):
    pass


def decode(data: bytes, *, max_bytes: int | None = None) -> np.ndarray:
    """Decode a JPEG (or anything Pillow reads) to an HxWx3 uint8 RGB array."""
    limit = settings.max_upload_bytes if max_bytes is None else max_bytes
    if len(data) > limit:
        raise ImageTooLarge(
            f"{len(data)} bytes exceeds the {limit}-byte limit. The Pi should be "
            f"sending ~30 KB frames — check DETECT_LONG_EDGE_PX on that side."
        )
    if not data:
        raise ImageUndecodable("empty body")
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.draft("RGB", im.size)     # lets Pillow skip full-res JPEG decode where it can
            return np.asarray(im.convert("RGB"), dtype=np.uint8)
    except UnidentifiedImageError as exc:
        raise ImageUndecodable("not a decodable image") from exc
    except OSError as exc:               # truncated JPEG, most often a cut-off upload
        raise ImageUndecodable(f"corrupt image data: {exc}") from exc
