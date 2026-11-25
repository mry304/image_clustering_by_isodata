# Phân cụm ảnh bằng ISODATA

Ứng dụng phân cụm / phân vùng ảnh sử dụng thuật toán **ISODATA** (tự cài đặt bằng NumPy) và đối chiếu với **K-Means** (tự cài đặt bằng NumPy). `scikit-learn` chỉ dùng để tính các chỉ số đánh giá và đối chiếu K-Means trong kiểm thử.

## Cấu trúc thư mục

```text
├── app/
│   └── main.py                  # Giao diện Streamlit (4 tab tương tác)
├── config/
│   └── experiment_config.json   # Cấu hình cố định cho thực nghiệm hàng loạt
├── core/
│   ├── common.py                # assign_clusters, update_centers, compute_sse
│   ├── features.py              # load_image, image_to_features, features_to_image
│   ├── histogram.py             # Histogram toàn ảnh/kênh/cụm, đếm đỉnh Gaussian, thống kê cụm
│   ├── kmeans.py                # Thuật toán K-Means tự cài bằng NumPy
│   ├── isodata.py               # Thuật toán ISODATA đầy đủ (loại nhỏ, tách, gộp, history)
│   └── compare.py               # Chỉ số đánh giá, đối chiếu K-Means, kiểm tra độ ổn định
├── data/images/                 # Thư mục chứa ảnh đầu vào
├── results/                     # Kết quả CSV và hình vẽ xuất từ run_experiments.py
├── scripts/                     # Các script demo theo bước và script chạy thực nghiệm
├── tests/                       # Bộ kiểm thử tự động (pytest)
└── requirements.txt
```

## Cài đặt

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Chạy ứng dụng Streamlit

```powershell
.\.venv\Scripts\streamlit.exe run app/main.py
```

## Chạy thực nghiệm & xuất kết quả

Chép các ảnh thử nghiệm vào `data/images/`, chỉnh tham số trong `config/experiment_config.json` (nếu cần), rồi chạy:

```powershell
.\.venv\Scripts\python.exe scripts\run_experiments.py
```

Kết quả được lưu tại:
- `results/metrics_summary.csv`
- `results/sensitivity_k.csv`
- `results/figures/<tên_ảnh>_comparison.png` và `results/figures/<tên_ảnh>_histogram.png`

## Chạy kiểm thử tự động

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```
