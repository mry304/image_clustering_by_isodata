"""Script minh họa Bước 7: phân tích histogram theo cụm và đếm đỉnh đối chiếu ISODATA."""

from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.features import image_to_features, load_image
from core.histogram import (
    compute_cluster_stats,
    count_histogram_peaks,
    plot_cluster_histograms,
    plot_histogram_with_centers,
)
from core.isodata import IsodataParams, isodata


def find_first_image(images_dir: Path) -> Path:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in exts)
    if not images:
        raise FileNotFoundError(f"Không tìm thấy ảnh trong {images_dir}.")
    return images[0]


def main() -> None:
    img_path = find_first_image(ROOT_DIR / "data" / "images")
    img = load_image(img_path, max_size=300)  # (H, W, 3)
    X = image_to_features(img, mode="lab")  # (N, 3)

    params = IsodataParams(
        K0=5,
        K_max=10,
        theta_N=300,
        theta_S=18.0,
        theta_c=22.0,
        L=2,
        max_iter=15,
        seed=42,
    )
    res = isodata(X, params, record_labels_every=0)

    # Phân tích trên kênh L (độ sáng, cột 0 của Lab).
    l_values = X[:, 0]  # (N,)
    l_centers = res.centers[:, 0]  # (K,)
    l_hist, _ = np.histogram(l_values, bins=256, range=(0, 256))
    peaks, _ = count_histogram_peaks(l_hist, sigma=3.0, min_distance=15, min_height_ratio=0.05)

    print(f"Ảnh: {img_path.name} | K cuối của ISODATA = {res.centers.shape[0]}")
    print(f"Số đỉnh histogram kênh L (sau làm mượt Gaussian): {len(peaks)} (tại mức {peaks.tolist()})")

    stats = compute_cluster_stats(l_values, res.labels)
    print("\nThống kê kênh L theo từng cụm:")
    print(f"{'Cụm':>4} | {'Số pixel':>10} | {'Tỉ lệ (%)':>10} | {'Trung bình':>11} | {'Độ lệch chuẩn':>14}")
    print("-" * 60)
    for row in stats:
        print(
            f"{row['cluster_id']:>4} | {row['n_pixels']:>10} | {row['percentage']:>9.2f}% | "
            f"{row['mean'][0]:>11.2f} | {row['std'][0]:>14.2f}"
        )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    plot_histogram_with_centers(
        l_values,
        l_centers,
        title=f"Histogram kênh L & tâm cụm (K={len(l_centers)}, số đỉnh={len(peaks)})",
        ax=axes[0],
    )
    plot_cluster_histograms(
        l_values,
        res.labels,
        title="Histogram từng cụm (kênh L) chồng trên toàn ảnh",
        ax=axes[1],
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
