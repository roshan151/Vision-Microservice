"""Photo folders on the Mac, written by background tasks.

    plants/map_{index}/plant_{NNN}/photo_{i}.png   + bbox_{i}.txt
    faces/{name}/photo_{i}.png                     + bbox_{i}.txt

`bbox_{i}.txt` is one line, `x1 y1 x2 y2` in pixels of photo_{i}.png (the same
order /v1/detect uses), or an empty file when nothing was found — so photo_{i}
and bbox_{i} always come in pairs. Enroll endpoints pick the file names up front
(so the records they return are already final), answer, and leave the PNG
encoding and disk writes to a FastAPI background task.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


def next_index(folder: Path) -> int:
    """One past the highest photo_{i}.png already in `folder` (1 when empty)."""
    nums = [int(p.stem.split("_")[1]) for p in folder.glob("photo_*.png") if p.stem.split("_")[1].isdigit()]
    return max(nums, default=0) + 1


def write_all(folder: Path, first: int, shots: Sequence[tuple]) -> None:
    """Background task: write (rgb, bbox) shots as photo_{first}, photo_{first+1}, ..."""
    for i, (rgb, bbox) in enumerate(shots, first):
        try:
            write(folder, i, rgb, bbox)
        except Exception:                                   # noqa: BLE001
            log.exception("could not save %s/photo_%d.png", folder, i)


def write(folder: Path, index: int, rgb: np.ndarray, bbox: Optional[Sequence[float]]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(folder / f"photo_{index}.png")
    (folder / f"bbox_{index}.txt").write_text(" ".join(f"{v:.1f}" for v in bbox) if bbox else "")
