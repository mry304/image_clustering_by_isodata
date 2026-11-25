"""Kiểm thử Bước 8: so sánh chỉ số ISODATA vs K-Means, kiểm tra độ ổn định và xuất CSV."""

import numpy as np
import pandas as pd

from core.compare import (
    compare_isodata_and_kmeans,
    export_comparison_csv,
    stability_study,
)
from core.isodata import IsodataParams


def _make_synthetic_rgb_image() -> np.ndarray:
    """Tạo ảnh RGB tổng hợp (40, 60, 3) gồm 3 vùng màu tách biệt."""
    rng = np.random.default_rng(42)
    img = np.zeros((40, 60, 3), dtype=np.uint8)
    img[:, :20] = np.clip(rng.normal((220, 40, 40), 8, size=(40, 20, 3)), 0, 255).astype(np.uint8)
    img[:, 20:40] = np.clip(rng.normal((40, 200, 50), 8, size=(40, 20, 3)), 0, 255).astype(np.uint8)
    img[:, 40:] = np.clip(rng.normal((40, 60, 220), 8, size=(40, 20, 3)), 0, 255).astype(np.uint8)
    return img


def test_compare_isodata_and_kmeans_columns_and_k_range():
    """Bảng đối chiếu phải có đủ chỉ số cho ISODATA, K-Means cùng K và dải K tùy chọn."""
    img = _make_synthetic_rgb_image()
    params = IsodataParams(
        K0=3,
        K_max=6,
        theta_N=50,
        theta_S=25.0,
        theta_c=20.0,
        L=2,
        max_iter=15,
        seed=42,
    )

    df = compare_isodata_and_kmeans(img, mode="lab", params=params, kmeans_k_range=[2, 3, 4])
    assert isinstance(df, pd.DataFrame)

    expected_cols = {
        "Thuật toán",
        "K",
        "SSE",
        "Davies-Bouldin",
        "Silhouette",
        "PSNR",
        "SSIM",
        "Thời gian (s)",
        "Số vòng",
    }
    assert expected_cols.issubset(set(df.columns))
    assert df["Thuật toán"].iloc[0].startswith("ISODATA")
    # Có ít nhất dòng ISODATA + các dòng K-Means trong dải K=[2, 3, 4].
    assert len(df) >= 3
    assert np.all(df["PSNR"] > 10.0)
    assert np.all((df["SSIM"] >= -1.0) & (df["SSIM"] <= 1.0))


def test_stability_study_and_csv_export(tmp_path):
    """stability_study phải trả về mean/std của K và các chỉ số, đồng thời xuất được CSV."""
    img = _make_synthetic_rgb_image()
    params = IsodataParams(
        K0=3,
        K_max=6,
        theta_N=50,
        theta_S=25.0,
        theta_c=20.0,
        L=2,
        max_iter=10,
        seed=0,
    )

    summary_df, raw_df = stability_study(img, mode="lab", params=params, n_runs=3, base_seed=10)
    assert len(summary_df) == 2
    assert len(raw_df) == 6
    assert "K (mean)" in summary_df.columns
    assert "K (std)" in summary_df.columns
    assert "SSE (mean)" in summary_df.columns
    assert "Silhouette (std)" in summary_df.columns

    csv_path = export_comparison_csv(summary_df, tmp_path / "stability.csv")
    assert csv_path.exists()
    reloaded = pd.read_csv(csv_path)
    assert len(reloaded) == 2
