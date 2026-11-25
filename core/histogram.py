"""Tính toán histogram cho ảnh xám và từng kênh màu RGB / Lab."""

import cv2
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
