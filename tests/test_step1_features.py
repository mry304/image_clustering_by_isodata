"""Kiểm thử Bước 1: chuyển đổi ảnh <-> đặc trưng và tính histogram."""

import numpy as np
import pytest

from core.features import features_to_image, image_to_features, load_image
from core.histogram import compute_channel_histograms, compute_gray_histogram


def test_features_roundtrip_shape():
    """Ảnh dựng lại từ đặc trưng phải có kích thước đúng như ảnh ban đầu cho cả 3 mode."""
    rng = np.random.default_rng(42)
    img = rng.integers(0, 256, size=(40, 60, 3), dtype=np.uint8)  # (H, W, 3)
    n_pixels = img.shape[0] * img.shape[1]

    for mode, expected_dim in (("gray", 1), ("rgb", 3), ("lab", 3)):
        feats = image_to_features(img, mode=mode)
        assert feats.shape == (n_pixels, expected_dim)
        assert np.issubdtype(feats.dtype, np.floating)

        # Dùng chính từng pixel làm tâm cụm riêng biệt (K = N) để kiểm tra dựng lại shape.
        labels = np.arange(n_pixels)
        reconstructed = features_to_image(labels, feats, img.shape, mode=mode)

        expected_shape = (40, 60) if mode == "gray" else (40, 60, 3)
        assert reconstructed.shape == expected_shape
        assert reconstructed.dtype == np.uint8

        if mode == "rgb":
            assert np.array_equal(reconstructed, img)


def test_load_image_resizes_keeping_aspect_ratio(tmp_path):
    """load_image phải thu nhỏ cạnh dài về tối đa max_size và giữ tỉ lệ."""
    import cv2

    big_img = np.zeros((400, 600, 3), dtype=np.uint8)
    img_path = tmp_path / "big.png"
    cv2.imwrite(str(img_path), big_img)

    loaded = load_image(img_path, max_size=300)
    assert loaded.shape == (200, 300, 3)
    assert loaded.dtype == np.uint8


def test_histograms_sum_to_pixel_count():
    """Tổng tần suất của mỗi histogram phải bằng số điểm ảnh H * W."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(30, 50, 3), dtype=np.uint8)
    n_pixels = 30 * 50

    gray_hist, gray_edges = compute_gray_histogram(img)
    assert gray_hist.shape == (256,)
    assert gray_edges.shape == (257,)
    assert gray_hist.sum() == n_pixels

    for mode, expected_keys in (("rgb", ("R", "G", "B")), ("lab", ("L", "a", "b"))):
        ch_hists = compute_channel_histograms(img, mode=mode)
        assert tuple(ch_hists.keys()) == expected_keys
        for hist, edges in ch_hists.values():
            assert hist.shape == (256,)
            assert edges.shape == (257,)
            assert hist.sum() == n_pixels


def test_invalid_mode_raises():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        image_to_features(img, mode="hsv")
