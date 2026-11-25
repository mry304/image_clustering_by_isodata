"""Script chạy thực nghiệm hàng loạt cho ISODATA và K-Means trên tập ảnh data/images."""

from dataclasses import replace
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.compare import compare_isodata_and_kmeans, export_comparison_csv
from core.features import features_to_image, image_to_features, load_image
from core.histogram import plot_cluster_histograms, plot_histogram_with_centers
from core.isodata import IsodataParams, isodata
from core.kmeans import kmeans


def _list_images(images_dir: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    if not images_dir.exists():
        return []
    return sorted(p for p in images_dir.iterdir() if p.suffix.lower() in exts)


def _extract_1d_channel(X, centers, mode: str):
    if mode in ("gray", "lab"):
        return X[:, 0], centers[:, 0], "Mức xám" if mode == "gray" else "Kênh L"
    weights = [0.299, 0.587, 0.114]
    return X @ weights, centers @ weights, "Độ sáng (RGB)"


def _save_figures(
    img,
    X,
    iso_res,
    km_res,
    mode: str,
    stem: str,
    figures_dir: Path,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    k_iso = int(iso_res.centers.shape[0])

    iso_img = features_to_image(iso_res.labels, iso_res.centers, img.shape, mode=mode)
    km_img = features_to_image(km_res.labels, km_res.centers, img.shape, mode=mode)

    fig_cmp, axes_cmp = plt.subplots(1, 3, figsize=(13, 4.2))
    axes_cmp[0].imshow(img)
    axes_cmp[0].set_title("Ảnh gốc")
    axes_cmp[0].axis("off")

    axes_cmp[1].imshow(iso_img)
    axes_cmp[1].set_title(f"ISODATA (K={k_iso})")
    axes_cmp[1].axis("off")

    axes_cmp[2].imshow(km_img)
    axes_cmp[2].set_title(f"K-Means (K={k_iso})")
    axes_cmp[2].axis("off")

    plt.tight_layout()
    fig_cmp.savefig(figures_dir / f"{stem}_comparison.png", dpi=150)
    plt.close(fig_cmp)

    vals_1d, ctrs_1d, ch_name = _extract_1d_channel(X, iso_res.centers, mode)
    fig_hist, axes_hist = plt.subplots(1, 2, figsize=(12, 4.2))
    plot_histogram_with_centers(
        vals_1d, ctrs_1d, title=f"{stem} - Histogram {ch_name} & tâm cụm", ax=axes_hist[0]
    )
    plot_cluster_histograms(
        vals_1d, iso_res.labels, title=f"{stem} - Histogram từng cụm ({ch_name})", ax=axes_hist[1]
    )
    plt.tight_layout()
    fig_hist.savefig(figures_dir / f"{stem}_histogram.png", dpi=150)
    plt.close(fig_hist)


def main() -> None:
    config_path = ROOT_DIR / "config" / "experiment_config.json"
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    images = _list_images(ROOT_DIR / "data" / "images")
    if not images:
        raise FileNotFoundError("Không tìm thấy ảnh nào trong data/images.")

    mode = cfg["mode"]
    max_size = int(cfg["max_size"])
    params = IsodataParams(**cfg["isodata_params"])
    k_range = cfg.get("kmeans_k_range", [])

    results_dir = ROOT_DIR / "results"
    figures_dir = results_dir / "figures"

    metric_frames: list[pd.DataFrame] = []
    loaded_cache: list[tuple[Path, object, object]] = []

    for img_path in images:
        img = load_image(img_path, max_size=max_size)  # (H, W, 3)
        X = image_to_features(img, mode=mode)  # (N, d)
        loaded_cache.append((img_path, img, X))

        df_img = compare_isodata_and_kmeans(
            img, mode=mode, params=params, kmeans_k_range=k_range
        )
        df_img.insert(0, "Ảnh", img_path.name)
        metric_frames.append(df_img)

        iso_res = isodata(X, params, record_labels_every=0)
        km_res = kmeans(X, K=int(iso_res.centers.shape[0]), seed=params.seed)
        _save_figures(img, X, iso_res, km_res, mode, img_path.stem, figures_dir)

    metrics_df = pd.concat(metric_frames, ignore_index=True)
    metrics_csv = export_comparison_csv(metrics_df, results_dir / "metrics_summary.csv")

    sens_cfg = cfg["sensitivity"]
    sens_images = loaded_cache[: int(sens_cfg["max_images"])]
    sens_rows: list[dict] = []

    for img_path, _, X in sens_images:
        for theta_s in sens_cfg["theta_S_values"]:
            for theta_c in sens_cfg["theta_c_values"]:
                p_run = replace(params, theta_S=float(theta_s), theta_c=float(theta_c))
                res = isodata(X, p_run, record_labels_every=0)
                sens_rows.append(
                    {
                        "Ảnh": img_path.name,
                        "theta_S": float(theta_s),
                        "theta_c": float(theta_c),
                        "K_cuối": int(res.centers.shape[0]),
                        "Số_vòng": int(res.n_iter),
                        "SSE": round(res.history[-1].sse, 2),
                    }
                )

    sens_df = pd.DataFrame(sens_rows)
    sens_csv = export_comparison_csv(sens_df, results_dir / "sensitivity_k.csv")

    print(f"Số ảnh đã xử lý: {len(images)} | Số ảnh thí nghiệm độ nhạy: {len(sens_images)}")
    print(f"Đã lưu bảng chỉ số: {metrics_csv}")
    print(f"Đã lưu bảng độ nhạy: {sens_csv}")
    print(f"Đã lưu hình tại: {figures_dir}")
    print("\nBảng chỉ số:")
    print(metrics_df.to_string(index=False))
    print("\nBảng độ nhạy K_cuối theo (theta_S, theta_c):")
    print(sens_df.to_string(index=False))


if __name__ == "__main__":
    main()
