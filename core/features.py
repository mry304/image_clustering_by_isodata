"""Đọc ảnh và chuyển đổi qua lại giữa ảnh và ma trận đặc trưng."""

from pathlib import Path
import cv2
import numpy as np

VALID_MODES = ("gray", "rgb", "lab")


def load_image(path: str | Path, max_size: int = 300) -> np.ndarray:
    """Đọc ảnh từ đường dẫn và resize giữ tỉ lệ nếu cạnh dài vượt quá max_size.

    Trả về mảng ảnh RGB kiểu uint8 có shape (H, W, 3).
    """
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Không thể đọc ảnh từ đường dẫn: {path}")

    # OpenCV đọc mặc định theo BGR; chuyển sang RGB để nhất quán với matplotlib và Lab.
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # (H, W, 3)
    h, w = img_rgb.shape[:2]

    long_edge = max(h, w)
    if long_edge > max_size:
        scale = max_size / long_edge
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        # INTER_AREA giảm hiện tượng răng cưa khi thu nhỏ ảnh.
        img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)  # (new_h, new_w, 3)

    return img_rgb


def image_to_features(img: np.ndarray, mode: str) -> np.ndarray:
    """Chuyển ảnh RGB (H, W, 3) sang ma trận đặc trưng (N, d) kiểu float64.

    mode thuộc {"gray", "rgb", "lab"}:
    - "gray": trả về (N, 1) với N = H * W
    - "rgb":  trả về (N, 3)
    - "lab":  trả về (N, 3) trong không gian CIE Lab (thang 0..255 của OpenCV)
    """
    if mode not in VALID_MODES:
        raise ValueError(f"mode không hợp lệ: {mode}. Chỉ hỗ trợ {VALID_MODES}.")

    if mode == "gray":
        if img.ndim == 2:
            gray = img  # (H, W)
        elif img.ndim == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # (H, W)
        else:
            raise ValueError(f"Ảnh đầu vào có shape không hợp lệ cho mode='gray': {img.shape}")
        return gray.reshape(-1, 1).astype(np.float64)  # (N, 1)

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Ảnh đầu vào phải có shape (H, W, 3), nhận được {img.shape}")

    if mode == "rgb":
        return img.reshape(-1, 3).astype(np.float64)  # (N, 3)

    # Dùng thang uint8 (0..255) của OpenCV để đồng nhất miền giá trị với features_to_image.
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)  # (H, W, 3)
    return lab.reshape(-1, 3).astype(np.float64)  # (N, 3)


def features_to_image(
    labels: np.ndarray,
    centers: np.ndarray,
    image_shape: tuple[int, ...],
    mode: str,
) -> np.ndarray:
    """Dựng lại ảnh uint8 từ mảng nhãn (N,) và ma trận tâm cụm (K, d).

    Trả về ảnh (H, W) nếu mode == "gray", hoặc ảnh RGB (H, W, 3) nếu mode in {"rgb", "lab"}.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"mode không hợp lệ: {mode}. Chỉ hỗ trợ {VALID_MODES}.")

    h, w = image_shape[:2]
    expected_n = h * w
    labels_flat = np.asarray(labels).reshape(-1)  # (N,)
    if labels_flat.shape[0] != expected_n:
        raise ValueError(
            f"Số nhãn ({labels_flat.shape[0]}) không khớp H*W ({expected_n})."
        )

    centers_arr = np.asarray(centers, dtype=np.float64)  # (K, d)
    reconstructed = centers_arr[labels_flat]  # (N, d)
    reconstructed_u8 = np.clip(np.rint(reconstructed), 0, 255).astype(np.uint8)  # (N, d)

    if mode == "gray":
        return reconstructed_u8.reshape(h, w)  # (H, W)

    img_3ch = reconstructed_u8.reshape(h, w, 3)  # (H, W, 3)
    if mode == "rgb":
        return img_3ch  # (H, W, 3)

    return cv2.cvtColor(img_3ch, cv2.COLOR_LAB2RGB)  # (H, W, 3)
