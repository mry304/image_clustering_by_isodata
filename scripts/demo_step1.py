"""Script minh họa Bước 1: nạp ảnh, kiểm tra shape đặc trưng và vẽ histogram."""

from pathlib import Path
import sys
import cv2
import matplotlib.pyplot as plt
from skimage import data

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.features import image_to_features, load_image
from core.histogram import compute_channel_histograms, compute_gray_histogram


def ensure_sample_image(images_dir: Path) -> Path:
    """Tìm ảnh đầu tiên trong data/images; nếu trống thì tạo ảnh mẫu từ scikit-image."""
    images_dir.mkdir(parents=True, exist_ok=True)
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    existing = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in exts)
    if existing:
        return existing[0]

    sample_path = images_dir / "astronaut.png"
    sample_rgb = data.astronaut()  # (512, 512, 3) uint8
    cv2.imwrite(str(sample_path), cv2.cvtColor(sample_rgb, cv2.COLOR_RGB2BGR))
    return sample_path


def main() -> None:
    images_dir = ROOT_DIR / "data" / "images"
    img_path = ensure_sample_image(images_dir)

    img = load_image(img_path, max_size=300)  # (H, W, 3)
    print(f"Ảnh: {img_path.name}")
    print(f"Shape ảnh sau load_image: {img.shape}, dtype={img.dtype}")

    for mode in ("gray", "rgb", "lab"):
        feats = image_to_features(img, mode=mode)
        print(f"Mode '{mode}': features shape = {feats.shape}, dtype = {feats.dtype}")

    gray_hist, gray_edges = compute_gray_histogram(img)
    rgb_hists = compute_channel_histograms(img, mode="rgb")
    lab_hists = compute_channel_histograms(img, mode="lab")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    axes[0, 0].imshow(img)
    axes[0, 0].set_title(f"Ảnh gốc ({img.shape[1]}x{img.shape[0]})")
    axes[0, 0].axis("off")

    axes[0, 1].plot(gray_edges[:-1], gray_hist, color="black")
    axes[0, 1].set_title("Histogram ảnh xám")
    axes[0, 1].set_xlabel("Mức xám (0-255)")
    axes[0, 1].set_ylabel("Số điểm ảnh")

    for name, color in (("R", "red"), ("G", "green"), ("B", "blue")):
        hist, edges = rgb_hists[name]
        axes[1, 0].plot(edges[:-1], hist, color=color, label=name)
    axes[1, 0].set_title("Histogram các kênh RGB")
    axes[1, 0].set_xlabel("Giá trị kênh (0-255)")
    axes[1, 0].legend()

    for name, color in (("L", "gray"), ("a", "magenta"), ("b", "orange")):
        hist, edges = lab_hists[name]
        axes[1, 1].plot(edges[:-1], hist, color=color, label=name)
    axes[1, 1].set_title("Histogram các kênh Lab")
    axes[1, 1].set_xlabel("Giá trị kênh (0-255)")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
