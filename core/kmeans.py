"""Thuật toán K-Means tự cài đặt bằng NumPy."""

from dataclasses import dataclass
import numpy as np

from core.common import assign_clusters, compute_sse, update_centers


@dataclass
class KMeansResult:
    """Kết quả chạy K-Means gồm nhãn (N,), tâm (K, d), lịch sử SSE và số vòng lặp."""

    labels: np.ndarray
    centers: np.ndarray
    sse_history: list[float]
    n_iter: int


def kmeans(
    X: np.ndarray,
    K: int,
    max_iter: int = 100,
    tol: float = 1e-4,
    seed: int = 42,
) -> KMeansResult:
    """Phân cụm dữ liệu X (N, d) thành K cụm bằng thuật toán K-Means thuần NumPy.

    Dừng khi độ dịch chuyển lớn nhất của tâm cụm nhỏ hơn tol hoặc đạt max_iter vòng.
    """
    X_arr = np.asarray(X, dtype=np.float64)  # (N, d)
    if X_arr.ndim != 2:
        raise ValueError(f"X phải là mảng 2 chiều (N, d), nhận được {X_arr.shape}")

    n_samples = X_arr.shape[0]
    if K <= 0 or K > n_samples:
        raise ValueError(f"Số cụm K={K} không hợp lệ với N={n_samples} điểm dữ liệu.")

    rng = np.random.default_rng(seed)
    init_indices = rng.choice(n_samples, size=K, replace=False)  # (K,)
    centers = X_arr[init_indices].copy()  # (K, d)

    sse_history: list[float] = []
    labels = assign_clusters(X_arr, centers)  # (N,)
    n_iter = 0

    for it in range(1, max_iter + 1):
        n_iter = it
        new_centers = update_centers(X_arr, labels, K, prev_centers=centers)  # (K, d)
        labels = assign_clusters(X_arr, new_centers)  # (N,)
        sse = compute_sse(X_arr, labels, new_centers)
        sse_history.append(sse)

        # Kiểm tra độ dịch chuyển lớn nhất theo chuẩn Euclid để quyết định hội tụ.
        max_shift = float(np.max(np.linalg.norm(new_centers - centers, axis=1)))
        centers = new_centers
        if max_shift < tol:
            break

    return KMeansResult(
        labels=labels,
        centers=centers,
        sse_history=sse_history,
        n_iter=n_iter,
    )
