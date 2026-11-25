"""Script minh họa Bước 2: chạy K-Means tự cài trên dữ liệu 2D tổng hợp và đối chiếu sklearn."""

from pathlib import Path
import sys
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.datasets import make_blobs

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.kmeans import kmeans


def main() -> None:
    X, _ = make_blobs(
        n_samples=300,
        centers=[(-6, -6), (0, 6), (6, -2)],
        cluster_std=0.9,
        random_state=42,
    )  # (300, 2)

    ours = kmeans(X, K=3, max_iter=100, tol=1e-5, seed=42)
    sk = SklearnKMeans(n_clusters=3, init="k-means++", n_init=10, random_state=42).fit(X)

    print(f"K-Means tự cài: số vòng lặp = {ours.n_iter}, SSE cuối = {ours.sse_history[-1]:.4f}")
    print(f"sklearn.KMeans: SSE cuối = {sk.inertia_:.4f}")
    print(f"Tâm cụm tìm được:\n{ours.centers}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].scatter(X[:, 0], X[:, 1], c=ours.labels, cmap="viridis", s=25, alpha=0.8)
    axes[0].scatter(
        ours.centers[:, 0],
        ours.centers[:, 1],
        c="red",
        marker="X",
        s=180,
        edgecolor="black",
        label="Tâm cụm",
    )
    axes[0].set_title(f"K-Means tự cài (K=3, {ours.n_iter} vòng)")
    axes[0].set_xlabel("x1")
    axes[0].set_ylabel("x2")
    axes[0].legend()

    axes[1].plot(range(1, len(ours.sse_history) + 1), ours.sse_history, marker="o")
    axes[1].set_title("Lịch sử SSE theo vòng lặp")
    axes[1].set_xlabel("Vòng lặp")
    axes[1].set_ylabel("SSE")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
