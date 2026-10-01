"""The face gallery: labelled embeddings on disk, cosine match in memory.

Deliberately not a vector database. A household gallery is a handful of people
with a few embeddings each; a brute-force dot product over a (N, 512) float32
array is microseconds, and the whole store fits in one .npz you can delete to
start over. Introducing an index here would add a dependency and a failure mode
to save nothing measurable.

Embeddings are L2-normalised on the way in, so cosine similarity is a dot
product and similarity is in [-1, 1].
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

EMBED_DIM_DEFAULT = 512


@dataclass(frozen=True)
class Match:
    label: str
    similarity: float


class Gallery:
    def __init__(self, path: Path, noun: str = "people") -> None:
        self._path = path
        self._noun = noun                  # what stats() calls the distinct labels
        self._lock = threading.Lock()
        self._vectors: np.ndarray = np.zeros((0, EMBED_DIM_DEFAULT), dtype=np.float32)
        self._labels: list[str] = []
        self._load()

    # --- persistence -----------------------------------------------------

    def _load(self) -> None:
        if not self._path.exists():
            log.info("no gallery at %s — starting empty", self._path)
            return
        try:
            data = np.load(self._path, allow_pickle=False)
            self._vectors = data["vectors"].astype(np.float32)
            self._labels = [str(x) for x in data["labels"]]
            log.info("gallery loaded: %d embeddings, %d people",
                     len(self._labels), len(set(self._labels)))
        except Exception as exc:                       # noqa: BLE001
            # A corrupt gallery must not stop the service booting. Everyone
            # becomes "unknown", which is a degraded robot, not a dead one.
            log.error("gallery at %s unreadable (%s) — starting empty", self._path, exc)

    def _save_locked(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_name(self._path.name + ".tmp")
        # Write through an open handle: np.savez helpfully appends ".npz" to a
        # *path* that lacks it, which would leave the rename below pointing at a
        # file that was never created.
        with open(tmp, "wb") as fh:
            np.savez(fh, vectors=self._vectors, labels=np.array(self._labels, dtype=str))
        tmp.replace(self._path)                        # atomic: never a half-written gallery

    # --- api -------------------------------------------------------------

    def add(self, label: str, embeddings: list[np.ndarray]) -> int:
        if not embeddings:
            return 0
        stacked = np.stack([_normalise(e) for e in embeddings]).astype(np.float32)
        with self._lock:
            if len(self._labels) and stacked.shape[1] != self._vectors.shape[1]:
                raise ValueError(
                    f"embedding dim {stacked.shape[1]} != gallery dim {self._vectors.shape[1]} "
                    "— the face model changed; delete the gallery and re-enroll"
                )
            if not len(self._labels):
                self._vectors = np.zeros((0, stacked.shape[1]), dtype=np.float32)
            self._vectors = np.concatenate([self._vectors, stacked], axis=0)
            self._labels.extend([label] * len(stacked))
            self._save_locked()
        return len(stacked)

    def match(self, embedding: np.ndarray, top_k: int = 1) -> list[Match]:
        q = _normalise(embedding)
        with self._lock:
            if not len(self._labels):
                return []
            if q.shape[0] != self._vectors.shape[1]:
                raise ValueError(
                    f"query dim {q.shape[0]} != gallery dim {self._vectors.shape[1]}"
                )
            sims = self._vectors @ q
            labels = list(self._labels)

        # Best score per person, not per embedding — otherwise four photos of
        # the same face crowd out everyone else in the top_k.
        best: dict[str, float] = {}
        for label, sim in zip(labels, sims.tolist()):
            if sim > best.get(label, -2.0):
                best[label] = sim
        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
        return [Match(label=l, similarity=round(float(s), 4)) for l, s in ranked[:top_k]]

    def remove(self, label: str) -> int:
        with self._lock:
            keep = [i for i, l in enumerate(self._labels) if l != label]
            removed = len(self._labels) - len(keep)
            if removed:
                self._vectors = self._vectors[keep]
                self._labels = [self._labels[i] for i in keep]
                self._save_locked()
        return removed

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "embeddings": len(self._labels),
                self._noun: len(set(self._labels)),
            }

    def labels(self) -> list[str]:
        with self._lock:
            return sorted(set(self._labels))

    def size(self) -> int:
        with self._lock:
            return len(self._labels)


def _normalise(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).ravel()
    n = float(np.linalg.norm(v))
    if n == 0.0:
        raise ValueError("zero-length embedding")
    return v / n
