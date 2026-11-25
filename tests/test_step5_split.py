"""Kiểm thử Bước 5: bước tách cụm (split) trong ISODATA."""

import numpy as np

from core.isodata import IsodataParams, isodata


def _make_elongated_blob(seed: int = 42) -> np.ndarray:
    """Tạo một đám điểm 2D kéo dài dọc theo trục x (std_x >> std_y)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-10.0, 10.0, size=120)
    y = rng.normal(0.0, 0.25, size=120)
    return np.column_stack([x, y])  # (120, 2)


def test_split_elongated_cluster_along_major_axis():
    """Đám điểm kéo dài theo trục x phải bị tách đôi dọc theo trục x và ghi event split."""
    X = _make_elongated_blob()
    params = IsodataParams(
        K0=1,
        K_max=2,
        theta_N=10,
        theta_S=2.0,
        theta_c=1.0,
        L=1,
        max_iter=15,
        seed=42,
    )
    res = isodata(X, params)

    assert res.centers.shape == (2, 2)
    assert any("split" in ev for step in res.history for ev in step.events)

    # Hai tâm sau tách phải phân tách rõ theo trục x và gần 0 theo trục y.
    x_sorted = np.sort(res.centers[:, 0])
    assert x_sorted[0] < -2.0 and x_sorted[1] > 2.0
    assert np.all(np.abs(res.centers[:, 1]) < 0.5)


def test_split_respects_k_max():
    """Không được tách thêm nếu số cụm hiện tại đã đạt K_max."""
    X = _make_elongated_blob()
    params = IsodataParams(
        K0=1,
        K_max=1,
        theta_N=10,
        theta_S=2.0,
        theta_c=1.0,
        L=1,
        max_iter=10,
        seed=42,
    )
    res = isodata(X, params)

    assert res.centers.shape == (1, 2)
    assert all("split" not in ev for step in res.history for ev in step.events)
