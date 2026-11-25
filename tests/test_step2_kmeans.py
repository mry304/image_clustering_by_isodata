"""Kiểm thử Bước 2: assign_clusters, update_centers và K-Means tự cài đối chiếu sklearn."""

import numpy as np
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.datasets import make_blobs

from core.common import assign_clusters, compute_sse, update_centers
from core.kmeans import kmeans


def test_assign_and_update_centers():
    """Kiểm tra gán nhãn và cập nhật tâm trên dữ liệu nhỏ tính tay được."""
    X = np.array([[0.0, 0.0], [0.0, 2.0], [10.0, 0.0], [10.0, 2.0]])  # (4, 2)
    centers = np.array([[0.5, 0.5], [9.5, 0.5]])  # (2, 2)

    labels = assign_clusters(X, centers)
    assert labels.shape == (4,)
    assert np.array_equal(labels, np.array([0, 0, 1, 1]))

    new_centers = update_centers(X, labels, K=2, prev_centers=centers)
    assert new_centers.shape == (2, 2)
    assert np.allclose(new_centers, np.array([[0.0, 1.0], [10.0, 1.0]]))


def test_kmeans_matches_sklearn_on_3_blobs():
    """Trên 3 đám điểm 2D tách rõ, SSE của K-Means tự cài phải gần với sklearn.KMeans."""
    X, _ = make_blobs(
        n_samples=300,
        centers=[(-6, -6), (0, 6), (6, -2)],
        cluster_std=0.8,
        random_state=42,
    )  # (300, 2)

    ours = kmeans(X, K=3, max_iter=100, tol=1e-6, seed=42)
    sk = SklearnKMeans(n_clusters=3, init="random", n_init=10, random_state=42).fit(X)

    assert ours.labels.shape == (300,)
    assert ours.centers.shape == (3, 2)
    assert len(ours.sse_history) == ours.n_iter >= 1
    assert np.isclose(ours.sse_history[-1], compute_sse(X, ours.labels, ours.centers))

    # Cho phép sai khác tương đối nhỏ (< 1%) do khởi tạo ngẫu nhiên khác nhau.
    rel_diff = abs(ours.sse_history[-1] - sk.inertia_) / sk.inertia_
    assert rel_diff < 0.01, f"SSE lệch quá lớn: ours={ours.sse_history[-1]:.3f}, sklearn={sk.inertia_:.3f}"


def test_kmeans_sse_non_increasing():
    """SSE qua các vòng lặp K-Means không được tăng (trừ sai số số học rất nhỏ)."""
    X, _ = make_blobs(n_samples=200, centers=4, cluster_std=1.2, random_state=7)
    res = kmeans(X, K=4, max_iter=50, tol=1e-6, seed=7)

    diffs = np.diff(res.sse_history)
    assert np.all(diffs <= 1e-8)
