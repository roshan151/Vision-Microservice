"""The plant register: identity embeddings plus a folder of photos per plant.

    plants/                         (VS_PLANT_DIR; default: next to the code)
        gallery.npz                 DINOv2 embeddings, one row per view
        maps.json                   SLAM map name -> map index
        map_1/
            plant_001/
                meta.json           position on the map, every visit, every view
                photo_1.png  bbox_1.txt        (written in the background, see photos.py)
                ...

Same shape as the face gallery on purpose — enroll = add views under a label,
match = best cosine similarity over that label's views. Several views per plant
is what makes a *rotated* plant still match: the side the robot sees today was
probably photographed from some angle last time.

Re-visiting a known plant appends a new visit (with its photos) rather than a
new plant, which is what growth tracking reads later.
"""

from __future__ import annotations

import json
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

from . import photos
from .gallery import Gallery

LABEL_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class BadLabel(ValueError):
    pass


def check_label(label: str) -> str:
    # Labels become directory names: refuse anything that could escape the root.
    if not LABEL_RE.match(label or ""):
        raise BadLabel(f"label {label!r} must match {LABEL_RE.pattern}")
    return label


class PlantStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.gallery = Gallery(root / "gallery.npz", noun="plants")
        self._lock = threading.Lock()
        self.map_index = self._latest_map() or 1

    def _latest_map(self) -> int:
        return max((int(m.group(1)) for p in self.root.iterdir() if (m := re.match(r"^map_(\d+)$", p.name))),
                   default=0)

    def use_map(self, name: str, new: bool) -> int:
        """Point the store at a SLAM map. A fresh map gets the next index; a
        reloaded one gets the index it had (or the latest, if never recorded)."""
        f = self.root / "maps.json"
        with self._lock:
            known = json.loads(f.read_text()) if f.exists() else {}
            if new or name not in known:
                known[name] = (max([self._latest_map(), *known.values()]) + 1) if new else (self._latest_map() or 1)
                f.write_text(json.dumps(known, indent=2))
            self.map_index = known[name]
        return self.map_index

    def folder(self, label: str) -> Path:
        return self.root / f"map_{self.map_index}" / check_label(label)

    def next_label(self) -> str:
        with self._lock:
            nums = [int(m.group(1)) for p in self.root.glob("map_*/plant_*")
                    if (m := re.match(r"^plant_(\d+)$", p.name))]
            nums += [int(m.group(1)) for l in self.gallery.labels()
                     if (m := re.match(r"^plant_(\d+)$", l))]
            return f"plant_{(max(nums) + 1) if nums else 1:03d}"

    def meta(self, label: str) -> Optional[dict]:
        f = self.folder(label) / "meta.json"
        return json.loads(f.read_text()) if f.exists() else None

    def add_visit(self, label: str, jpegs: list[bytes], embeddings: list[np.ndarray],
                  meta: dict) -> dict:
        """Store one visit: embeddings in the gallery, the visit record (position,
        per-view poses, anything else the robot sent) in meta.json, and the photo
        file names reserved. The photos themselves are written afterwards by a
        background task (photos.write) — the caller gets `files` back for that."""
        folder = self.folder(label)
        stamp = time.strftime("%Y%m%dT%H%M%S")
        with self._lock:
            folder.mkdir(parents=True, exist_ok=True)
            first = (self.meta(label) or {}).get("n_views", 0) + 1
            files = [f"photo_{first + i}.png" for i in range(len(jpegs))]
            m = self.meta(label) or {"label": label, "created": stamp, "visits": []}
            views = meta.get("views") or [{} for _ in files]
            m["visits"].append({
                "t": stamp,
                "position": meta.get("position"),
                "views": [{"file": f, **(views[i] if i < len(views) else {})}
                          for i, f in enumerate(files)],
                **{k: v for k, v in meta.items() if k not in ("views", "position")},
            })
            if meta.get("position") is not None:
                m["position"] = meta["position"]
            m["updated"] = stamp
            m["n_views"] = sum(len(v["views"]) for v in m["visits"])
            tmp = folder / "meta.json.tmp"
            tmp.write_text(json.dumps(m, indent=2))
            tmp.replace(folder / "meta.json")
        added = self.gallery.add(label, embeddings)
        return {"label": label, "folder": str(folder), "files": files, "photos_saved": len(files),
                "embeddings_added": added, "views_total": m["n_views"], "visits": len(m["visits"])}

    def list(self) -> list[dict]:
        out = []
        for p in sorted((self.root / f"map_{self.map_index}").glob("*")):
            if p.is_dir() and LABEL_RE.match(p.name) and (p / "meta.json").exists():
                m = json.loads((p / "meta.json").read_text())
                out.append({k: m.get(k) for k in ("label", "position", "created", "updated", "n_views")}
                           | {"visits": len(m.get("visits", []))})
        return out

    def positions(self) -> list[tuple]:
        return [(m["label"], *m["position"][:2]) for m in self.list() if m.get("position")]

    def remove(self, label: str) -> int:
        """Forget a plant. Photos are moved to _deleted/, never destroyed."""
        check_label(label)
        removed = self.gallery.remove(label)
        folder = self.folder(label)
        if folder.exists():
            dest = self.root / "_deleted" / f"{label}_{time.strftime('%Y%m%dT%H%M%S')}"
            dest.parent.mkdir(exist_ok=True)
            shutil.move(str(folder), str(dest))
            removed = removed or 1
        return removed
