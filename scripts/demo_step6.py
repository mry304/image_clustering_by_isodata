"""Script minh họa Bước 6: chạy ISODATA trên ảnh thật (mode lab) và in bảng summarize_history."""

from pathlib import Path
import sys
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.features import features_to_image, image_to_features, load_image
from core.isodata import IsodataParams, isodata, summarize_history


def find_first_image(images_dir: Path) -> Path:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in exts)
    if not images:
        raise FileNotFoundError(
            f"Không tìm thấy ảnh nào trong {images_dir}. Hãy chạy demo_step1.py trước hoặc thêm ảnh."
        )
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
    res = isodata(X, params, record_labels_every=3)
    df = summarize_history(res.history)

    print(f"Ảnh: {img_path.name}, shape = {img.shape}, N = {X.shape[0]} pixel")
    print(f"Số vòng chạy: {res.n_iter}, Số cụm cuối: {res.centers.shape[0]}")
    print("\nBảng tóm tắt lịch sử ISODATA:")
    print(df.to_string(index=False))

    segmented = features_to_image(res.labels, res.centers, img.shape, mode="lab")  # (H, W, 3)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].imshow(img)
    axes[0].set_title("Ảnh gốc")
    axes[0].axis("off")

    axes[1].imshow(segmented)
    axes[1].set_title(f"ISODATA (mode=lab, K={res.centers.shape[0]})")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
