"""Kiểm thử Bước 7: đếm đỉnh histogram, thống kê cụm và vẽ histogram theo cụm."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.histogram import (
    compute_cluster_stats,
    count_histogram_peaks,
    plot_cluster_histograms,
    plot_histogram_with_centers,
)


def test_count_histogram_peaks_detects_bimodal_and_unimodal():
    """Đếm đúng 2 đỉnh trên phân bố 2 đỉnh rõ rệt và 1 đỉnh trên phân bố 1 đỉnh."""
    rng = np.random.default_rng(42)
    bimodal = np.concatenate([rng.normal(60, 6, size=1000), rng.normal(180, 6, size=1000)])
    hist_bi, _ = np.histogram(np.clip(bimodal, 0, 255), bins=256, range=(0, 256))
    peaks_bi, smoothed_bi = count_histogram_peaks(hist_bi, sigma=3.0, min_distance=20)
    assert smoothed_bi.shape == (256,)
    assert len(peaks_bi) == 2
    assert 45 <= peaks_bi[0] <= 75
    assert 165 <= peaks_bi[1] <= 195

    unimodal = rng.normal(128, 8, size=1500)
    hist_uni, _ = np.histogram(np.clip(unimodal, 0, 255), bins=256, range=(0, 256))
    peaks_uni, _ = count_histogram_peaks(hist_uni, sigma=3.0, min_distance=20)
    assert len(peaks_uni) == 1


def test_compute_cluster_stats_values():
    """Kiểm tra số pixel, tổng tỉ lệ %, trung bình và độ lệch chuẩn từng cụm."""
    X = np.array([[10.0, 20.0], [10.0, 40.0], [100.0, 200.0], [100.0, 200.0]])  # (4, 2)
    labels = np.array([0, 0, 1, 1])

    stats = compute_cluster_stats(X, labels)
    assert len(stats) == 2
    assert sum(row["n_pixels"] for row in stats) == 4
    assert np.isclose(sum(row["percentage"] for row in stats), 100.0)
    assert np.allclose(stats[0]["mean"], np.array([10.0, 30.0]))
    assert np.allclose(stats[0]["std"], np.array([0.0, 10.0]))
    assert np.allclose(stats[1]["mean"], np.array([100.0, 200.0]))


def test_plot_histogram_functions_render_without_error():
    """Các hàm vẽ histogram với tâm cụm và theo cụm phải chạy được trên Agg."""
    vals = np.concatenate([np.full(50, 40.0), np.full(50, 200.0)])
    labels = np.array([0] * 50 + [1] * 50)
    centers = np.array([40.0, 200.0])

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    plot_histogram_with_centers(vals, centers, ax=axes[0])
    plot_cluster_histograms(vals, labels, ax=axes[1])
    plt.close(fig)
