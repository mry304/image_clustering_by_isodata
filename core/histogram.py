"""Tính toán histogram cho ảnh xám và từng kênh màu RGB / Lab."""

import cv2
import matplotlib.pyplot as plt
import numpy as np


def compute_gray_histogram(img: np.ndarray, bins: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Tính histogram mức xám của ảnh (H, W) hoặc RGB (H, W, 3).

    Trả về (hist, bin_edges) với hist có shape (bins,) và bin_edges có shape (bins + 1,).
    """
    if img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # (H, W)
    elif img.ndim == 2:
        gray = img  # (H, W)
    else:
        raise ValueError(f"Shape ảnh không hợp lệ: {img.shape}")

    hist, bin_edges = np.histogram(gray.ravel(), bins=bins, range=(0, 256))
    return hist, bin_edges


def compute_channel_histograms(
    img: np.ndarray,
    mode: str = "rgb",
    bins: int = 256,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Tính histogram từng kênh cho ảnh RGB (H, W, 3) theo không gian 'rgb' hoặc 'lab'.

    Trả về dict ánh xạ tên kênh -> (hist, bin_edges), mỗi hist có shape (bins,).
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Ảnh đầu vào phải có shape (H, W, 3), nhận được {img.shape}")

    if mode == "rgb":
        converted = img  # (H, W, 3)
        channel_names = ("R", "G", "B")
    elif mode == "lab":
        converted = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)  # (H, W, 3)
        channel_names = ("L", "a", "b")
    else:
        raise ValueError(f"mode không hợp lệ: {mode}. Chỉ hỗ trợ 'rgb' hoặc 'lab'.")

    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for idx, name in enumerate(channel_names):
        channel_data = converted[:, :, idx]  # (H, W)
        hist, bin_edges = np.histogram(channel_data.ravel(), bins=bins, range=(0, 256))
        result[name] = (hist, bin_edges)
    return result


def smooth_histogram_gaussian(hist: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Làm mượt mảng histogram 1D (bins,) bằng nhân Gaussian tự tạo với NumPy."""
    hist_arr = np.asarray(hist, dtype=np.float64).ravel()  # (bins,)
    if sigma <= 0:
        return hist_arr.copy()

    radius = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)  # (2*radius + 1,)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= np.sum(kernel)

    # mode='same' giữ nguyên kích thước (bins,) để vị trí bin sau làm mượt khớp với chỉ số gốc.
    return np.convolve(hist_arr, kernel, mode="same")  # (bins,)


def count_histogram_peaks(
    hist: np.ndarray,
    sigma: float = 2.0,
    min_distance: int = 10,
    min_height_ratio: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Đếm và trả về chỉ số các đỉnh nổi bật của histogram sau khi làm mượt Gaussian.

    Trả về (peak_indices, smoothed_hist) với peak_indices sắp xếp tăng dần.
    """
    smoothed = smooth_histogram_gaussian(hist, sigma=sigma)  # (bins,)
    if smoothed.size < 3 or float(np.max(smoothed)) <= 0.0:
        return np.array([], dtype=np.int64), smoothed

    # Tìm cực đại địa phương (bỏ qua 2 biên để tránh nhiễu biên nhân chập).
    candidates = (
        np.where((smoothed[1:-1] > smoothed[:-2]) & (smoothed[1:-1] >= smoothed[2:]))[0]
        + 1
    )
    min_height = min_height_ratio * float(np.max(smoothed))
    candidates = candidates[smoothed[candidates] >= min_height]
    if candidates.size == 0:
        return np.array([], dtype=np.int64), smoothed

    # Ưu tiên giữ đỉnh cao hơn khi hai đỉnh nằm gần nhau hơn min_distance.
    order = np.argsort(smoothed[candidates])[::-1]
    selected: list[int] = []
    for idx in candidates[order]:
        idx_int = int(idx)
        if all(abs(idx_int - kept) >= min_distance for kept in selected):
            selected.append(idx_int)

    return np.array(sorted(selected), dtype=np.int64), smoothed


def plot_histogram_with_centers(
    values_1d: np.ndarray,
    centers_1d: np.ndarray,
    bins: int = 256,
    title: str = "Histogram toàn ảnh và tâm cụm",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Vẽ histogram toàn ảnh (mức xám hoặc kênh L) kèm các vạch đứng tại tâm cụm."""
    vals = np.asarray(values_1d, dtype=np.float64).ravel()  # (N,)
    ctrs = np.asarray(centers_1d, dtype=np.float64).ravel()  # (K,)
    hist, bin_edges = np.histogram(vals, bins=bins, range=(0, 256))

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    ax.bar(bin_edges[:-1], hist, width=1.0, color="steelblue", alpha=0.65, label="Toàn ảnh")
    for k, c in enumerate(ctrs):
        ax.axvline(c, color="crimson", linestyle="--", linewidth=1.5, label="Tâm cụm" if k == 0 else None)

    ax.set_title(title)
    ax.set_xlabel("Giá trị cường độ (0–255)")
    ax.set_ylabel("Số pixel")
    ax.set_xlim(0, 255)
    ax.legend()
    return ax


def plot_cluster_histograms(
    values_1d: np.ndarray,
    labels: np.ndarray,
    bins: int = 256,
    title: str = "Histogram từng cụm chồng trên toàn ảnh",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Vẽ histogram toàn ảnh làm nền và chồng histogram từng cụm (mỗi cụm một màu)."""
    vals = np.asarray(values_1d, dtype=np.float64).ravel()  # (N,)
    lbls = np.asarray(labels).reshape(-1)  # (N,)
    total_hist, bin_edges = np.histogram(vals, bins=bins, range=(0, 256))
    x_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # (bins,)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    ax.fill_between(x_centers, total_hist, color="lightgray", alpha=0.6, label="Toàn ảnh")

    unique_ids = np.unique(lbls)
    cmap = plt.get_cmap("tab10" if len(unique_ids) <= 10 else "tab20")
    for idx, cid in enumerate(unique_ids):
        c_vals = vals[lbls == cid]
        c_hist, _ = np.histogram(c_vals, bins=bins, range=(0, 256))
        ax.plot(x_centers, c_hist, color=cmap(idx % cmap.N), linewidth=1.5, label=f"Cụm {int(cid)}")

    ax.set_title(title)
    ax.set_xlabel("Giá trị cường độ (0–255)")
    ax.set_ylabel("Số pixel")
    ax.set_xlim(0, 255)
    ax.legend(ncol=2, fontsize=8)
    return ax


def compute_cluster_stats(X: np.ndarray, labels: np.ndarray) -> list[dict]:
    """Tính thống kê cho từng cụm: số pixel, tỉ lệ %, giá trị trung bình và độ lệch chuẩn."""
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim == 1:
        X_arr = X_arr[:, None]  # (N, 1)
    labels_arr = np.asarray(labels).reshape(-1)  # (N,)

    n_total = X_arr.shape[0]
    if n_total == 0:
        return []

    stats: list[dict] = []
    for cid in np.unique(labels_arr):
        pts = X_arr[labels_arr == cid]  # (n_c, d)
        n_c = int(pts.shape[0])
        stats.append(
            {
                "cluster_id": int(cid),
                "n_pixels": n_c,
                "percentage": 100.0 * n_c / n_total,
                "mean": np.mean(pts, axis=0),  # (d,)
                "std": np.std(pts, axis=0),  # (d,)
            }
        )
    return stats
