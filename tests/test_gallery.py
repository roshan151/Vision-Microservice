"""Gallery unit tests. Pure numpy — no app, no client, no models."""

from __future__ import annotations

import numpy as np
import pytest

from vision_service.gallery import Gallery


def vec(seed: int, dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


def test_add_and_match(tmp_path):
    g = Gallery(tmp_path / "g.npz")
    assert g.add("roshan", [vec(1), vec(2)]) == 2
    m = g.match(vec(1))
    assert m[0].label == "roshan" and m[0].similarity > 0.99


def test_best_score_per_person_not_per_embedding(tmp_path):
    """Four photos of one face must not crowd everyone else out of top_k."""
    g = Gallery(tmp_path / "g.npz")
    g.add("roshan", [vec(i) for i in range(1, 5)])
    g.add("sam", [vec(50)])
    labels = [m.label for m in g.match(vec(1), top_k=2)]
    assert labels == ["roshan", "sam"]


def test_persists_across_instances(tmp_path):
    path = tmp_path / "g.npz"
    Gallery(path).add("roshan", [vec(1)])
    assert Gallery(path).match(vec(1))[0].label == "roshan"


def test_unnormalised_input_is_normalised(tmp_path):
    g = Gallery(tmp_path / "g.npz")
    g.add("roshan", [vec(1) * 17.0])
    assert g.match(vec(1))[0].similarity == pytest.approx(1.0, abs=1e-3)


def test_zero_vector_rejected(tmp_path):
    g = Gallery(tmp_path / "g.npz")
    with pytest.raises(ValueError):
        g.add("x", [np.zeros(512, dtype=np.float32)])


def test_dimension_change_is_a_clear_error(tmp_path):
    g = Gallery(tmp_path / "g.npz")
    g.add("roshan", [vec(1, 512)])
    with pytest.raises(ValueError, match="re-enroll"):
        g.add("sam", [vec(2, 128)])


def test_remove(tmp_path):
    g = Gallery(tmp_path / "g.npz")
    g.add("roshan", [vec(1), vec(2)])
    g.add("sam", [vec(3)])
    assert g.remove("roshan") == 2
    assert g.labels() == ["sam"]
    assert g.remove("nobody") == 0


def test_corrupt_gallery_starts_empty_rather_than_crashing(tmp_path):
    """A bad gallery file degrades recognition. It must not stop the service booting."""
    path = tmp_path / "g.npz"
    path.write_bytes(b"this is not an npz")
    g = Gallery(path)
    assert g.size() == 0
    assert g.match(vec(1)) == []


def test_empty_gallery_matches_nothing(tmp_path):
    assert Gallery(tmp_path / "g.npz").match(vec(1)) == []
