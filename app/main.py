"""Giao diện Streamlit cho ứng dụng Phân cụm ảnh bằng thuật toán ISODATA."""

from pathlib import Path
import sys
import tempfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.compare import compare_isodata_and_kmeans, stability_study
from core.features import features_to_image, image_to_features, load_image
from core.histogram import (
    compute_cluster_stats,
    count_histogram_peaks,
    plot_cluster_histograms,
    plot_histogram_with_centers,
)
from core.isodata import IsodataParams, isodata, summarize_history
from core.kmeans import kmeans

st.set_page_config(page_title="Phân cụm ảnh bằng ISODATA", layout="wide")


def _list_sample_images() -> list[Path]:
    """Liệt kê các file ảnh mẫu trong thư mục data/images."""
    images_dir = ROOT_DIR / "data" / "images"
    if not images_dir.exists():
        return []
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted(p for p in images_dir.iterdir() if p.suffix.lower() in exts)


@st.cache_data(show_spinner=False)
def _load_sample_cached(path_str: str, max_size: int = 300) -> np.ndarray:
    return load_image(path_str, max_size=max_size)


@st.cache_data(show_spinner=False)
def _load_uploaded_cached(file_bytes: bytes, suffix: str, max_size: int = 300) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)
    try:
        return load_image(tmp_path, max_size=max_size)
    finally:
        tmp_path.unlink(missing_ok=True)


@st.cache_data(show_spinner=False)
def _run_isodata_cached(
    img_bytes: bytes,
    img_shape: tuple[int, ...],
    mode: str,
    K0: int,
    K_max: int,
    theta_N: int,
    theta_S: float,
    theta_c: float,
    L: int,
    max_iter: int,
    seed: int,
):
    img = np.frombuffer(img_bytes, dtype=np.uint8).reshape(img_shape)
    X = image_to_features(img, mode=mode)
    params = IsodataParams(
        K0=K0,
        K_max=K_max,
        theta_N=theta_N,
        theta_S=theta_S,
        theta_c=theta_c,
        L=L,
        max_iter=max_iter,
        seed=seed,
    )
    return isodata(X, params, record_labels_every=1)


@st.cache_data(show_spinner=False)
def _run_kmeans_cached(
    img_bytes: bytes,
    img_shape: tuple[int, ...],
    mode: str,
    K: int,
    seed: int,
):
    img = np.frombuffer(img_bytes, dtype=np.uint8).reshape(img_shape)
    X = image_to_features(img, mode=mode)
    return kmeans(X, K=K, seed=seed)


@st.cache_data(show_spinner=False)
def _run_compare_cached(
    img_bytes: bytes,
    img_shape: tuple[int, ...],
    mode: str,
    K0: int,
    K_max: int,
    theta_N: int,
    theta_S: float,
    theta_c: float,
    L: int,
    max_iter: int,
    seed: int,
    k_range: tuple[int, ...],
) -> pd.DataFrame:
    img = np.frombuffer(img_bytes, dtype=np.uint8).reshape(img_shape)
    params = IsodataParams(
        K0=K0,
        K_max=K_max,
        theta_N=theta_N,
        theta_S=theta_S,
        theta_c=theta_c,
        L=L,
        max_iter=max_iter,
        seed=seed,
    )
    return compare_isodata_and_kmeans(
        img,
        mode=mode,
        params=params,
        kmeans_k_range=list(k_range) if k_range else None,
    )


@st.cache_data(show_spinner=False)
def _run_stability_cached(
    img_bytes: bytes,
    img_shape: tuple[int, ...],
    mode: str,
    K0: int,
    K_max: int,
    theta_N: int,
    theta_S: float,
    theta_c: float,
    L: int,
    max_iter: int,
    seed: int,
    n_runs: int,
):
    img = np.frombuffer(img_bytes, dtype=np.uint8).reshape(img_shape)
    params = IsodataParams(
        K0=K0,
        K_max=K_max,
        theta_N=theta_N,
        theta_S=theta_S,
        theta_c=theta_c,
        L=L,
        max_iter=max_iter,
        seed=seed,
    )
    return stability_study(img, mode=mode, params=params, n_runs=n_runs, base_seed=seed)


def _get_intensity_channel(X: np.ndarray, centers: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray, str]:
    """Lấy kênh cường độ 1D để vẽ histogram (mức xám cho gray, L cho lab, độ sáng xấp xỉ cho rgb)."""
    if mode == "gray":
        return X[:, 0], centers[:, 0], "Mức xám"
    if mode == "lab":
        return X[:, 0], centers[:, 0], "Kênh L (Lab)"
    # Với RGB, dùng kênh R hoặc độ sáng trung bình; dùng công thức độ sáng chuẩn BT.601 để hiển thị 1D.
    weights = np.array([0.299, 0.587, 0.114])
    return X @ weights, centers @ weights, "Độ sáng (RGB)"


def main() -> None:
    st.title("Phân cụm ảnh bằng thuật toán ISODATA")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["1. Ảnh & tham số", "2. Quá trình lặp", "3. Histogram", "4. So sánh K-Means"]
    )

    sample_images = _list_sample_images()
    img: np.ndarray | None = None
    img_name = ""

    with tab1:
        col_src, col_params = st.columns([1, 1])

        with col_src:
            st.subheader("Nguồn ảnh đầu vào")
            source_type = st.radio(
                "Chọn nguồn ảnh",
                options=["Ảnh mẫu (data/images)", "Tải ảnh lên"],
                horizontal=True,
            )
            if source_type == "Tải ảnh lên":
                uploaded = st.file_uploader(
                    "Chọn file ảnh", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
                )
                if uploaded is not None:
                    suffix = Path(uploaded.name).suffix or ".png"
                    img = _load_uploaded_cached(uploaded.getvalue(), suffix=suffix, max_size=300)
                    img_name = uploaded.name
            else:
                if sample_images:
                    selected = st.selectbox(
                        "Chọn ảnh mẫu",
                        options=sample_images,
                        format_func=lambda p: p.name,
                    )
                    img = _load_sample_cached(str(selected), max_size=300)
                    img_name = selected.name
                else:
                    st.warning("Thư mục data/images hiện chưa có ảnh. Hãy tải ảnh lên.")

            if img is not None:
                st.image(img, caption=f"{img_name} ({img.shape[1]}x{img.shape[0]})", use_container_width=True)

        with col_params:
            st.subheader("Không gian đặc trưng & tham số ISODATA")
            mode = st.selectbox("Không gian đặc trưng", options=["lab", "rgb", "gray"], index=0)
            k0 = st.slider("K0 (số cụm ban đầu)", min_value=2, max_value=15, value=5)
            k_max = st.slider("K_max (số cụm tối đa)", min_value=max(2, k0), max_value=25, value=max(10, k0))
            theta_n = st.slider("theta_N (số pixel tối thiểu mỗi cụm)", min_value=10, max_value=2000, value=300, step=10)
            theta_s = st.slider("theta_S (ngưỡng độ lệch chuẩn để tách)", min_value=1.0, max_value=60.0, value=18.0, step=0.5)
            theta_c = st.slider("theta_c (ngưỡng khoảng cách để gộp)", min_value=1.0, max_value=80.0, value=22.0, step=0.5)
            max_merges = st.slider("L (số cặp gộp tối đa mỗi vòng)", min_value=1, max_value=10, value=2)
            max_iter = st.slider("max_iter (số vòng tối đa)", min_value=2, max_value=50, value=15)
            seed = st.number_input("seed (ngẫu nhiên)", min_value=0, max_value=99999, value=42, step=1)

            run_clicked = st.button("Chạy ISODATA", type="primary", use_container_width=True)

        if img is not None:
            current_config = {
                "img_bytes": img.tobytes(),
                "img_shape": img.shape,
                "img_name": img_name,
                "mode": mode,
                "K0": int(k0),
                "K_max": int(k_max),
                "theta_N": int(theta_n),
                "theta_S": float(theta_s),
                "theta_c": float(theta_c),
                "L": int(max_merges),
                "max_iter": int(max_iter),
                "seed": int(seed),
            }
            if run_clicked or "active_run" not in st.session_state:
                st.session_state["active_run"] = current_config

            cfg = st.session_state["active_run"]
            active_img = np.frombuffer(cfg["img_bytes"], dtype=np.uint8).reshape(cfg["img_shape"])
            res = _run_isodata_cached(
                cfg["img_bytes"],
                cfg["img_shape"],
                cfg["mode"],
                cfg["K0"],
                cfg["K_max"],
                cfg["theta_N"],
                cfg["theta_S"],
                cfg["theta_c"],
                cfg["L"],
                cfg["max_iter"],
                cfg["seed"],
            )
            seg_img = features_to_image(res.labels, res.centers, active_img.shape, mode=cfg["mode"])

            st.divider()
            c1, c2 = st.columns(2)
            c1.image(active_img, caption="Ảnh gốc", use_container_width=True)
            c2.image(
                seg_img,
                caption=f"Kết quả ISODATA (mode={cfg['mode']}, K={res.centers.shape[0]}, {res.n_iter} vòng)",
                use_container_width=True,
            )

    if "active_run" not in st.session_state:
        for t in (tab2, tab3, tab4):
            with t:
                st.info("Hãy chọn hoặc tải ảnh ở Tab 1 và nhấn 'Chạy ISODATA'.")
        return

    cfg = st.session_state["active_run"]
    active_img = np.frombuffer(cfg["img_bytes"], dtype=np.uint8).reshape(cfg["img_shape"])
    res = _run_isodata_cached(
        cfg["img_bytes"],
        cfg["img_shape"],
        cfg["mode"],
        cfg["K0"],
        cfg["K_max"],
        cfg["theta_N"],
        cfg["theta_S"],
        cfg["theta_c"],
        cfg["L"],
        cfg["max_iter"],
        cfg["seed"],
    )

    with tab2:
        st.subheader("Theo dõi quá trình lặp ISODATA")
        if res.n_iter > 1:
            iter_idx = st.slider("Chọn vòng lặp", min_value=1, max_value=res.n_iter, value=res.n_iter)
        else:
            iter_idx = 1

        step = res.history[iter_idx - 1]
        col_img, col_info = st.columns([1, 1])
        with col_img:
            if step.labels is not None:
                step_img = features_to_image(step.labels, step.centers, active_img.shape, mode=cfg["mode"])
                st.image(step_img, caption=f"Vòng {step.iteration} — K = {step.n_clusters}", use_container_width=True)
            else:
                st.warning("Vòng này không lưu mảng nhãn.")

        with col_info:
            m1, m2, m3 = st.columns(3)
            m1.metric("Vòng lặp", step.iteration)
            m2.metric("Số cụm (K)", step.n_clusters)
            m3.metric("SSE", f"{step.sse:,.0f}")

            st.markdown("**Sự kiện của vòng:**")
            if step.events:
                for ev in step.events:
                    st.write(f"- {ev}")
            else:
                st.write("Không có sự kiện tách / gộp / loại cụm.")

        df_hist = summarize_history(res.history)
        fig_k, ax_k = plt.subplots(figsize=(8, 3))
        ax_k.plot(df_hist["Vòng"], df_hist["K"], marker="o", color="teal")
        ax_k.set_title("Biến thiên số cụm K theo vòng lặp")
        ax_k.set_xlabel("Vòng")
        ax_k.set_ylabel("Số cụm K")
        ax_k.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig_k)
        plt.close(fig_k)

        st.dataframe(df_hist, use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("Phân tích Histogram & thống kê cụm")
        X = image_to_features(active_img, mode=cfg["mode"])
        vals_1d, ctrs_1d, ch_label = _get_intensity_channel(X, res.centers, cfg["mode"])

        hist_1d, _ = np.histogram(vals_1d, bins=256, range=(0, 256))
        peaks, _ = count_histogram_peaks(hist_1d, sigma=3.0, min_distance=15, min_height_ratio=0.05)

        st.write(
            f"**K cuối của ISODATA:** `{res.centers.shape[0]}` | "
            f"**Số đỉnh histogram ({ch_label}, làm mượt Gaussian):** `{len(peaks)}` "
            f"(tại mức `{peaks.tolist()}`)"
        )

        col_h1, col_h2 = st.columns(2)
        with col_h1:
            fig1, ax1 = plt.subplots(figsize=(6.5, 4))
            plot_histogram_with_centers(vals_1d, ctrs_1d, title=f"Histogram {ch_label} & tâm cụm", ax=ax1)
            st.pyplot(fig1)
            plt.close(fig1)

        with col_h2:
            fig2, ax2 = plt.subplots(figsize=(6.5, 4))
            plot_cluster_histograms(vals_1d, res.labels, title=f"Histogram từng cụm ({ch_label})", ax=ax2)
            st.pyplot(fig2)
            plt.close(fig2)

        stats = compute_cluster_stats(vals_1d, res.labels)
        stats_df = pd.DataFrame(
            [
                {
                    "Cụm": row["cluster_id"],
                    "Số pixel": row["n_pixels"],
                    "Tỉ lệ (%)": round(row["percentage"], 2),
                    f"Trung bình ({ch_label})": round(float(row["mean"][0]), 2),
                    f"Độ lệch chuẩn ({ch_label})": round(float(row["std"][0]), 2),
                }
                for row in stats
            ]
        )
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

    with tab4:
        st.subheader("Đối chiếu ISODATA và K-Means tự cài")
        k_iso = int(res.centers.shape[0])
        km_res = _run_kmeans_cached(
            cfg["img_bytes"], cfg["img_shape"], cfg["mode"], k_iso, cfg["seed"]
        )

        iso_seg = features_to_image(res.labels, res.centers, active_img.shape, mode=cfg["mode"])
        km_seg = features_to_image(km_res.labels, km_res.centers, active_img.shape, mode=cfg["mode"])

        c1, c2, c3 = st.columns(3)
        c1.image(active_img, caption="Ảnh gốc", use_container_width=True)
        c2.image(iso_seg, caption=f"ISODATA (K={k_iso})", use_container_width=True)
        c3.image(km_seg, caption=f"K-Means tự cài (K={k_iso})", use_container_width=True)

        use_k_range = st.checkbox("Chạy thêm K-Means với dải K = 2..8", value=False)
        k_range = tuple(range(2, 9)) if use_k_range else ()

        df_cmp = _run_compare_cached(
            cfg["img_bytes"],
            cfg["img_shape"],
            cfg["mode"],
            cfg["K0"],
            cfg["K_max"],
            cfg["theta_N"],
            cfg["theta_S"],
            cfg["theta_c"],
            cfg["L"],
            cfg["max_iter"],
            cfg["seed"],
            k_range,
        )
        st.dataframe(df_cmp, use_container_width=True, hide_index=True)

        csv_bytes = df_cmp.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            "Tải bảng so sánh (CSV)",
            data=csv_bytes,
            file_name="isodata_vs_kmeans.csv",
            mime="text/csv",
        )

        st.divider()
        st.markdown("**Kiểm tra độ ổn định theo nhiều seed**")
        n_runs = st.slider("Số lần chạy (n_runs)", min_value=3, max_value=15, value=10)
        if st.button("Chạy kiểm tra độ ổn định"):
            summary_df, _ = _run_stability_cached(
                cfg["img_bytes"],
                cfg["img_shape"],
                cfg["mode"],
                cfg["K0"],
                cfg["K_max"],
                cfg["theta_N"],
                cfg["theta_S"],
                cfg["theta_c"],
                cfg["L"],
                cfg["max_iter"],
                cfg["seed"],
                n_runs,
            )
            st.dataframe(summary_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
