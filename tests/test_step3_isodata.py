"""Kiểm thử Bước 3: khung ISODATA cơ bản với bước loại cụm nhỏ và ghi lịch sử."""

import numpy as np
from sklearn.datasets import make_blobs

from core.isodata import IsodataParams, isodata
from core.kmeans import kmeans


def test_isodata_small_theta_n_matches_kmeans():
    """Với theta_N đủ nhỏ, không cụm nào bị loại và kết quả tương đương K-Means."""
    X, _ = make_blobs(
        n_samples=300,
        centers=[(-6, -6), (0, 6), (6, -2)],
        cluster_std=0.8,
        random_state=42,
    )  # (300, 2)

    km_res = kmeans(X, K=3, max_iter=50, tol=1e-4, seed=42)
    iso_params = IsodataParams(K0=3, theta_N=1, max_iter=50, seed=42)
    iso_res = isodata(X, iso_params, tol=1e-4)

    assert iso_res.centers.shape == (3, 2)
    assert iso_res.labels.shape == (300,)
    assert np.isclose(iso_res.history[-1].sse, km_res.sse_history[-1], rtol=1e-6)
    assert all(len(step.events) == 0 for step in iso_res.history)


def test_isodata_large_theta_n_drops_small_clusters():
    """Với theta_N lớn, cụm có ít hơn theta_N mẫu bị loại và số cụm giảm xuống."""
    rng = np.random.default_rng(7)
    c1 = rng.normal(loc=(0.0, 0.0), scale=0.3, size=(50, 2))
    c2 = rng.normal(loc=(10.0, 0.0), scale=0.3, size=(50, 2))
    c3_small = rng.normal(loc=(20.0, 0.0), scale=0.2, size=(5, 2))
    X = np.vstack([c1, c2, c3_small])  # (105, 2)

    # Dùng theta_N=10: cụm nhỏ (5 mẫu) nếu được khởi tạo riêng sẽ bị loại, còn <= 2 cụm.
    params = IsodataParams(K0=4, theta_N=10, max_iter=20, seed=0)
    res = isodata(X, params)

    assert res.centers.shape[0] < params.K0
    assert res.centers.shape[0] <= 2
    assert any("Loại" in ev for step in res.history for ev in step.events)


def test_isodata_history_structure():
    """Mỗi phần tử trong history phải ghi đủ số cụm, tâm, SSE và event log."""
    X, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.7, random_state=1)
    params = IsodataParams(K0=5, theta_N=15, max_iter=10, seed=1)
    res = isodata(X, params)

    assert len(res.history) == res.n_iter >= 1
    for idx, step in enumerate(res.history, start=1):
        assert step.iteration == idx
        assert step.centers.shape[0] == step.n_clusters
        assert isinstance(step.sse, float) and step.sse >= 0.0
        assert isinstance(step.events, list)
