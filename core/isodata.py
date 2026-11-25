"""Thuật toán phân cụm ISODATA tự cài đặt bằng NumPy (Bước 3: chưa có tách/gộp)."""

from dataclasses import dataclass, field
import numpy as np

from core.common import assign_clusters, compute_sse, update_centers


@dataclass
class IsodataParams:
    """Bộ tham số cấu hình cho thuật toán ISODATA."""

    K0: int = 5
    K_max: int = 12
    theta_N: int = 10
    theta_S: float = 10.0
    theta_c: float = 5.0
    L: int = 2
    max_iter: int = 20
    seed: int = 42


@dataclass
class IsodataStepLog:
    """Thông tin trạng thái sau mỗi vòng lặp của ISODATA."""

    iteration: int
    n_clusters: int
    centers: np.ndarray
    sse: float
    events: list[str] = field(default_factory=list)


@dataclass
class IsodataResult:
    """Kết quả cuối cùng khi chạy ISODATA."""

    labels: np.ndarray
    centers: np.ndarray
    history: list[IsodataStepLog]
    n_iter: int


def isodata(X: np.ndarray, params: IsodataParams, tol: float = 1e-4) -> IsodataResult:
    """Phân cụm dữ liệu X (N, d) bằng ISODATA (phiên bản Bước 3: lọc cụm nhỏ, chưa tách/gộp).

    Mỗi vòng lặp thực hiện: gán nhãn -> loại cụm có ít hơn theta_N mẫu (và gán lại)
    -> cập nhật tâm -> ghi lịch sử.
    """
    X_arr = np.asarray(X, dtype=np.float64)  # (N, d)
    if X_arr.ndim != 2:
        raise ValueError(f"X phải là mảng 2 chiều (N, d), nhận được {X_arr.shape}")

    n_samples = X_arr.shape[0]
    if params.K0 <= 0 or params.K0 > n_samples:
        raise ValueError(f"K0={params.K0} không hợp lệ với N={n_samples} điểm dữ liệu.")

    rng = np.random.default_rng(params.seed)
    init_indices = rng.choice(n_samples, size=params.K0, replace=False)  # (K0,)
    centers = X_arr[init_indices].copy()  # (K0, d)

    history: list[IsodataStepLog] = []
    labels = np.zeros(n_samples, dtype=np.int64)  # (N,)
    n_iter = 0

    for it in range(1, params.max_iter + 1):
        n_iter = it
        events: list[str] = []
        prev_centers = centers.copy()  # (K_prev, d)

        # 1. Gán mỗi điểm dữ liệu về tâm gần nhất.
        labels = assign_clusters(X_arr, centers)  # (N,)
        k_curr = centers.shape[0]

        # 2. Loại bỏ các cụm có số mẫu < theta_N và gán lại các mẫu thuộc cụm bị loại.
        counts = np.bincount(labels, minlength=k_curr)  # (K_curr,)
        keep_mask = counts >= params.theta_N  # (K_curr,)

        if not np.any(keep_mask):
            # Tránh trường hợp tất cả cụm đều bị loại dẫn đến 0 cụm.
            largest_idx = int(np.argmax(counts))
            keep_mask[largest_idx] = True
            events.append(
                f"Tất cả {k_curr} cụm đều < {params.theta_N} mẫu; giữ lại cụm lớn nhất ({counts[largest_idx]} mẫu)."
            )

        if not np.all(keep_mask):
            dropped_indices = np.where(~keep_mask)[0].tolist()
            centers = centers[keep_mask]  # (K_kept, d)
            labels = assign_clusters(X_arr, centers)  # (N,)
            events.append(
                f"Loại {len(dropped_indices)} cụm (chỉ số {dropped_indices}) do < {params.theta_N} mẫu; còn {centers.shape[0]} cụm."
            )

        # 3. Cập nhật tâm cụm và đồng bộ nhãn / SSE cho vòng hiện tại.
        k_kept = centers.shape[0]
        new_centers = update_centers(X_arr, labels, k_kept, prev_centers=centers)  # (K_kept, d)
        labels = assign_clusters(X_arr, new_centers)  # (N,)
        sse = compute_sse(X_arr, labels, new_centers)

        history.append(
            IsodataStepLog(
                iteration=it,
                n_clusters=k_kept,
                centers=new_centers.copy(),
                sse=sse,
                events=events,
            )
        )

        # Chỉ kiểm tra hội tụ khi số cụm không thay đổi so với đầu vòng lặp.
        if k_kept == prev_centers.shape[0]:
            max_shift = float(np.max(np.linalg.norm(new_centers - prev_centers, axis=1)))
            if max_shift < tol:
                centers = new_centers
                break

        centers = new_centers

    return IsodataResult(
        labels=labels,
        centers=centers,
        history=history,
        n_iter=n_iter,
    )
