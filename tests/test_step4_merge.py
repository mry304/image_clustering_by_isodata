"""Kiểm thử Bước 4: bước gộp cụm (merge) trong ISODATA."""

import numpy as np

from core.isodata import IsodataParams, _merge_clusters, isodata


def _make_two_close_blobs(seed: int = 42) -> np.ndarray:
    """Tạo 2 đám điểm 2D gần nhau (tâm cách nhau ~1.6)."""
    rng = np.random.default_rng(seed)
    c1 = rng.normal(loc=(0.0, 0.0), scale=0.2, size=(40, 2))
    c2 = rng.normal(loc=(1.6, 0.0), scale=0.2, size=(40, 2))
    return np.vstack([c1, c2])  # (80, 2)


def test_merge_occurs_when_theta_c_is_large():
    """Hai đám điểm gần nhau phải được gộp thành 1 khi theta_c đủ lớn."""
    X = _make_two_close_blobs()
    params = IsodataParams(K0=2, theta_N=5, theta_c=3.0, L=2, max_iter=15, seed=42)
    res = isodata(X, params)

    assert res.centers.shape[0] == 1
    assert any("merge" in ev for step in res.history for ev in step.events)


def test_no_merge_when_theta_c_is_small():
    """Hai đám điểm gần nhau không bị gộp khi theta_c đủ nhỏ."""
    X = _make_two_close_blobs()
    params = IsodataParams(K0=2, theta_N=5, theta_c=0.5, L=2, max_iter=15, seed=42)
    res = isodata(X, params)

    assert res.centers.shape[0] == 2
    assert all("merge" not in ev for step in res.history for ev in step.events)


def test_merge_respects_L_and_single_participation_per_round():
    """Mỗi cụm chỉ tham gia tối đa 1 lần gộp và không vượt quá L cặp mỗi vòng."""
    centers = np.array([[0.0, 0.0], [1.0, 0.0], [2.2, 0.0], [3.5, 0.0]])  # (4, 2)
    counts = np.array([10, 30, 20, 10])

    # Cặp (0,1) khoảng cách 1.0 < 2.0; cặp (1,2) khoảng cách 1.2 nhưng cụm 1 đã gộp rồi.
    # Cặp (2,3) khoảng cách 1.3 < 2.0 -> được gộp nếu L=2.
    merged, events = _merge_clusters(centers, counts, theta_c=2.0, max_merges=2, iteration=1)
    assert len(events) == 2
    assert merged.shape == (2, 2)
    # Tâm của (0, 1) với trọng số 10 và 30 phải là (0.75, 0.0).
    assert np.allclose(merged[0], np.array([0.75, 0.0]))

    # Với L=1, chỉ gộp đúng 1 cặp gần nhất là (0, 1), còn lại cụm 2 và 3.
    merged_l1, events_l1 = _merge_clusters(centers, counts, theta_c=2.0, max_merges=1, iteration=1)
    assert len(events_l1) == 1
    assert merged_l1.shape == (3, 2)
