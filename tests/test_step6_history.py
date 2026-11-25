"""Kiểm thử Bước 6: lưu nhãn định kỳ trong history và hàm summarize_history."""

import numpy as np
import pandas as pd

from core.features import features_to_image, image_to_features
from core.isodata import IsodataParams, isodata, summarize_history


def test_history_records_labels_periodically_and_reconstructs_image():
    """Kiểm tra record_labels_every lưu nhãn đúng chu kỳ và dựng lại được ảnh từ history."""
    rng = np.random.default_rng(123)
    img = rng.integers(0, 256, size=(20, 30, 3), dtype=np.uint8)  # (20, 30, 3)
    X = image_to_features(img, mode="lab")  # (600, 3)

    params = IsodataParams(
        K0=3,
        K_max=6,
        theta_N=20,
        theta_S=15.0,
        theta_c=10.0,
        L=2,
        max_iter=6,
        seed=42,
    )
    res = isodata(X, params, tol=1e-9, record_labels_every=2)

    assert len(res.history) == 6
    # Vòng lẻ (1, 3, 5) không lưu nhãn; vòng chẵn (2, 4, 6) có lưu nhãn.
    assert res.history[0].labels is None
    assert res.history[1].labels is not None
    assert res.history[2].labels is None
    assert res.history[-1].labels is not None

    reconstructed = features_to_image(
        res.history[-1].labels,
        res.history[-1].centers,
        img.shape,
        mode="lab",
    )
    assert reconstructed.shape == img.shape
    assert reconstructed.dtype == np.uint8


def test_summarize_history_returns_expected_dataframe():
    """summarize_history phải trả về bảng pandas đúng cột và đúng số lượng sự kiện."""
    rng = np.random.default_rng(7)
    x = rng.uniform(-10.0, 10.0, size=100)
    y = rng.normal(0.0, 0.2, size=100)
    X = np.column_stack([x, y])  # (100, 2)

    params = IsodataParams(
        K0=1,
        K_max=3,
        theta_N=10,
        theta_S=2.0,
        theta_c=0.5,
        L=1,
        max_iter=6,
        seed=42,
    )
    res = isodata(X, params)
    df = summarize_history(res.history)

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["Vòng", "K", "Số split", "Số merge", "Số cụm bị loại", "SSE"]
    assert len(df) == len(res.history)
    assert df["Số split"].sum() >= 1
    assert df["Vòng"].tolist() == [step.iteration for step in res.history]
    assert df["K"].tolist() == [step.n_clusters for step in res.history]
