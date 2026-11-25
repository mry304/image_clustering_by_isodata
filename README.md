# Phân cụm ảnh bằng ISODATA

Bài tập cuối học phần Xử lý ảnh: xây dựng ứng dụng phân cụm ảnh bằng thuật toán **ISODATA** (tự cài đặt bằng NumPy) và đối chiếu với K-Means.

## Cấu trúc thư mục

```text
├── core/          # Xử lý ảnh & thuật toán phân cụm (thuần NumPy/OpenCV, không import streamlit)
├── app/           # Giao diện Streamlit
├── scripts/       # Các script chạy minh họa theo từng bước
├── tests/         # Kiểm thử tự động với pytest
├── data/images/   # Ảnh đầu vào để thử nghiệm
└── requirements.txt
```

## Chạy kiểm thử và demo Bước 1

```bash
pytest -v
python scripts/demo_step1.py
```
