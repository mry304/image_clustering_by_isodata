"""Các hàm dùng chung cho phân cụm (gán nhãn, cập nhật tâm, tính SSE)."""

import numpy as np


def assign_clusters(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Gán mỗi điểm dữ liệu trong X (N, d) về tâm cụm gần nhất trong centers (K, d).

    Trả về mảng nhãn nguyên labels có shape (N,).
    """
    X_arr = np.asarray(X, dtype=np.float64)  # (N, d)
    C_arr = np.asarray(centers, dtype=np.float64)  # (K, d)

    # Khai triển ||x - c||^2 = ||x||^2 + ||c||^2 - 2 x.c để tránh tạo mảng (N, K, d) tốn RAM.
    x_sq = np.sum(X_arr ** 2, axis=1, keepdims=True)  # (N, 1)
    c_sq = np.sum(C_arr ** 2, axis=1, keepdims=True).T  # (1, K)
    sq_dists = x_sq + c_sq - 2.0 * (X_arr @ C_arr.T)  # (N, K)

    # Kẹp về >= 0 để loại sai số dấu phẩy động âm rất nhỏ khi x trùng c.
    sq_dists = np.maximum(sq_dists, 0.0)  # (N, K)
    return np.argmin(sq_dists, axis=1)  # (N,)


def update_centers(
    X: np.ndarray,
    labels: np.ndarray,
    K: int,
    prev_centers: np.ndarray | None = None,
) -> np.ndarray:
    """Tính lại tọa độ tâm của K cụm từ X (N, d) và nhãn labels (N,).

    Nếu một cụm bị rỗng, giữ nguyên tâm cũ (nếu có prev_centers) để tránh NaN.
    Trả về mảng tâm mới có shape (K, d).
    """
    X_arr = np.asarray(X, dtype=np.float64)  # (N, d)
    labels_arr = np.asarray(labels)  # (N,)
    n_features = X_arr.shape[1]
    new_centers = np.zeros((K, n_features), dtype=np.float64)  # (K, d)

    for k in range(K):
        mask = labels_arr == k  # (N,)
        if np.any(mask):
            new_centers[k] = np.mean(X_arr[mask], axis=0)  # (d,)
        elif prev_centers is not None:
            new_centers[k] = prev_centers[k]
        else:
            new_centers[k] = X_arr[0]

    return new_centers


def compute_sse(X: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> float:
    """Tính tổng bình phương sai số (SSE) giữa các điểm X (N, d) và tâm cụm tương ứng."""
    X_arr = np.asarray(X, dtype=np.float64)  # (N, d)
    C_arr = np.asarray(centers, dtype=np.float64)  # (K, d)
    assigned_centers = C_arr[labels]  # (N, d)
    diff = X_arr - assigned_centers  # (N, d)
    return float(np.sum(diff ** 2))
