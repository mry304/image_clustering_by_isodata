"""Đánh giá chỉ số phân cụm và đối chiếu độ ổn định giữa ISODATA và K-Means tự cài."""

from dataclasses import replace
from pathlib import Path
import time
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sklearn.metrics import davies_bouldin_score, silhouette_score

from core.common import compute_sse
from core.features import features_to_image, image_to_features
from core.isodata import IsodataParams, isodata
from core.kmeans import kmeans


def evaluate_clustering(
    X: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    img_orig: np.ndarray,
    mode: str,
    elapsed_sec: float,
    n_iter: int,
    algo_name: str,
    metric_seed: int = 42,
    max_silhouette_samples: int = 5000,
) -> dict:
    """Tính các chỉ số đánh giá chất lượng phân cụm và chất lượng tái tạo ảnh."""
    X_arr = np.asarray(X, dtype=np.float64)  # (N, d)
    labels_arr = np.asarray(labels).reshape(-1)  # (N,)
    centers_arr = np.asarray(centers, dtype=np.float64)  # (K, d)

    n_samples = X_arr.shape[0]
    k_clusters = int(centers_arr.shape[0])
    unique_labels = np.unique(labels_arr)

    sse = compute_sse(X_arr, labels_arr, centers_arr)

    if len(unique_labels) >= 2:
        db_score = float(davies_bouldin_score(X_arr, labels_arr))
    else:
        db_score = float("nan")

    # Lấy mẫu con cố định seed vì silhouette có độ phức tạp O(N^2) khi N lớn.
    sample_n = min(max_silhouette_samples, n_samples)
    rng = np.random.default_rng(metric_seed)
    sample_idx = rng.choice(n_samples, size=sample_n, replace=False)  # (sample_n,)
    sample_labels = labels_arr[sample_idx]
    n_unique_sample = len(np.unique(sample_labels))
    if 2 <= n_unique_sample < sample_n:
        sil_score = float(silhouette_score(X_arr[sample_idx], sample_labels))
    else:
        sil_score = float("nan")

    reconstructed = features_to_image(labels_arr, centers_arr, img_orig.shape, mode=mode)
    if mode == "gray" and img_orig.ndim == 3:
        ref_img = cv2.cvtColor(img_orig, cv2.COLOR_RGB2GRAY)  # (H, W)
    else:
        ref_img = img_orig  # (H, W, 3) hoặc (H, W)

    psnr_val = float(peak_signal_noise_ratio(ref_img, reconstructed, data_range=255))
    channel_axis = -1 if ref_img.ndim == 3 else None
    ssim_val = float(
        structural_similarity(ref_img, reconstructed, data_range=255, channel_axis=channel_axis)
    )

    return {
        "Thuật toán": algo_name,
        "K": k_clusters,
        "SSE": sse,
        "Davies-Bouldin": db_score,
        "Silhouette": sil_score,
        "PSNR": psnr_val,
        "SSIM": ssim_val,
        "Thời gian (s)": elapsed_sec,
        "Số vòng": n_iter,
    }


def compare_isodata_and_kmeans(
    img: np.ndarray,
    mode: str,
    params: IsodataParams,
    kmeans_k_range: list[int] | range | None = None,
    max_iter_kmeans: int = 100,
    metric_seed: int = 42,
) -> pd.DataFrame:
    """Chạy ISODATA và K-Means (tại K cuối của ISODATA và tùy chọn dải K) rồi tổng hợp chỉ số."""
    X = image_to_features(img, mode=mode)  # (N, d)
    rows: list[dict] = []

    t0 = time.perf_counter()
    iso_res = isodata(X, params, record_labels_every=0)
    iso_time = time.perf_counter() - t0
    k_iso = int(iso_res.centers.shape[0])

    rows.append(
        evaluate_clustering(
            X,
            iso_res.labels,
            iso_res.centers,
            img,
            mode,
            iso_time,
            iso_res.n_iter,
            algo_name=f"ISODATA (K={k_iso})",
            metric_seed=metric_seed,
        )
    )

    # Luôn chạy K-Means tại K = K cuối của ISODATA để so sánh trực tiếp.
    k_list = [k_iso]
    if kmeans_k_range is not None:
        for k in kmeans_k_range:
            if k not in k_list:
                k_list.append(int(k))

    for k in k_list:
        t1 = time.perf_counter()
        km_res = kmeans(X, K=k, max_iter=max_iter_kmeans, seed=params.seed)
        km_time = time.perf_counter() - t1
        suffix = " [cùng K]" if k == k_iso else ""
        rows.append(
            evaluate_clustering(
                X,
                km_res.labels,
                km_res.centers,
                img,
                mode,
                km_time,
                km_res.n_iter,
                algo_name=f"K-Means (K={k}){suffix}",
                metric_seed=metric_seed,
            )
        )

    return pd.DataFrame(rows)


def stability_study(
    img: np.ndarray,
    mode: str,
    params: IsodataParams,
    n_runs: int = 10,
    base_seed: int = 0,
    max_iter_kmeans: int = 100,
    metric_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chạy lặp n_runs lần với các seed khác nhau để đo trung bình và độ lệch chuẩn của K và chỉ số.

    Trả về (summary_df, raw_runs_df).
    """
    X = image_to_features(img, mode=mode)  # (N, d)
    raw_rows: list[dict] = []

    # Chạy ISODATA trước để lấy K_iso cho mỗi seed và dùng K của lần đầu cho K-Means.
    k_iso_first: int | None = None
    for run_idx in range(n_runs):
        seed_val = base_seed + run_idx
        run_params = replace(params, seed=seed_val)

        t0 = time.perf_counter()
        iso_res = isodata(X, run_params, record_labels_every=0)
        iso_time = time.perf_counter() - t0
        if k_iso_first is None:
            k_iso_first = int(iso_res.centers.shape[0])

        iso_metrics = evaluate_clustering(
            X,
            iso_res.labels,
            iso_res.centers,
            img,
            mode,
            iso_time,
            iso_res.n_iter,
            algo_name="ISODATA",
            metric_seed=metric_seed,
        )
        iso_metrics["Run"] = run_idx + 1
        iso_metrics["Seed"] = seed_val
        raw_rows.append(iso_metrics)

    assert k_iso_first is not None
    for run_idx in range(n_runs):
        seed_val = base_seed + run_idx
        t1 = time.perf_counter()
        km_res = kmeans(X, K=k_iso_first, max_iter=max_iter_kmeans, seed=seed_val)
        km_time = time.perf_counter() - t1

        km_metrics = evaluate_clustering(
            X,
            km_res.labels,
            km_res.centers,
            img,
            mode,
            km_time,
            km_res.n_iter,
            algo_name=f"K-Means (K={k_iso_first})",
            metric_seed=metric_seed,
        )
        km_metrics["Run"] = run_idx + 1
        km_metrics["Seed"] = seed_val
        raw_rows.append(km_metrics)

    raw_df = pd.DataFrame(raw_rows)
    metric_cols = ["K", "SSE", "Davies-Bouldin", "Silhouette", "PSNR", "SSIM", "Thời gian (s)", "Số vòng"]

    summary_rows: list[dict] = []
    for algo_name, group in raw_df.groupby("Thuật toán", sort=False):
        row: dict = {"Thuật toán": algo_name, "Số lần chạy": len(group)}
        for col in metric_cols:
            row[f"{col} (mean)"] = float(group[col].mean())
            row[f"{col} (std)"] = float(group[col].std(ddof=0))
        summary_rows.append(row)

    return pd.DataFrame(summary_rows), raw_df


def export_comparison_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Xuất bảng kết quả đối chiếu ra file CSV (mã hóa UTF-8 có BOM để mở đẹp trên Excel)."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path
