import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Make sure we can import the provided course package "tc_shape"
# Folder structure assumed:
#   project_root/
#     ├── data/face_points45.txt
#     ├── results/
#     └── src/
#         ├── main.py   (this file)
#         └── tc_shape/ (__init__.py, apply_pca.py, plot.py, sim_transform.py, tests.py)
# ---------------------------------------------------------------------
CUR_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(CUR_DIR)
if CUR_DIR not in sys.path:
    sys.path.insert(0, CUR_DIR)

from tc_shape import sim_transform, apply_pca, plot  # your tutor's package

DATA_PATH = os.path.join(ROOT_DIR, "data", "face_points45.txt")
RES_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(RES_DIR, exist_ok=True)


# ---------------------------- 45-point connectivity ----------------------------
# This is exactly the connectivity you used in your original code (ass2.py).
# It draws clean facial parts instead of simply linking points in sequence.
CURVES_45 = [
    range(0, 15),                 # chin
    range(15, 19), range(19, 23), # eyebrows
    [23, 24, 25, 26, 23],         # left eye (closed)
    [27, 28, 29, 30, 27],         # right eye (closed)
    [31, 32],                     # nose bridge (short)
    range(33, 40),                # upper lip
    [39, 40, 41, 42, 43, 44, 33]  # lower lip (closed back to 33)
]


# ------------------------------- helper functions -------------------------------
def load_shapes_txt(path: str) -> np.ndarray:
    """
    Load shapes from plain text.
    Each row is a flattened shape: x0 y0 x1 y1 ... xN yN
    Returns:
        S: (n_samples, 2*n_points)
    """
    S = np.loadtxt(path)
    if S.ndim == 1:
        S = S[None, :]
    return S


def draw_shape_with_curves(ax, flat, curves, color="k", lw=1.5, ms=2.5, label=None):
    """
    Draw a flattened shape using a list of curve index sequences.
    Uses the same API idea as your original tc_shape.plot.plot_curves,
    but here we keep everything explicit and also optionally show points.
    """
    pts = flat.reshape(-1, 2)
    for seq in curves:
        seg = pts[list(seq)]
        ax.plot(seg[:, 0], -seg[:, 1], color=color, lw=lw)  # flip y for classroom convention
    # points on top (optional, match your original style)
    ax.plot(pts[:, 0], -pts[:, 1], "o", color=color, ms=ms, label=label)


def save_fig(fig, path, dpi=220):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"Saved: {path}")


# ------------------------------------ main ------------------------------------
def main():
    # 1) Load data
    S = load_shapes_txt(DATA_PATH)              # (n_samples, 2*n_points)
    n_samples, d = S.shape
    n_points = d // 2
    print(f"Loaded shapes: {n_samples} samples, {n_points} points each.")

    # 2) Align all shapes (translation + rotation + scale)
    S_aligned = sim_transform.align_set(S.copy())

    # 3) Mean shape (aligned)
    mean_shape = S_aligned.mean(axis=0)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    draw_shape_with_curves(ax, mean_shape, CURVES_45, color="k", label="mean")
    ax.set_aspect("equal"); ax.set_title("Mean Shape (Aligned)")
    save_fig(fig, os.path.join(RES_DIR, "mean_shape.png"))

    # 4) PCA (keep top-6 as typical)
    t = 6
    mean_vec, P, mode_var = apply_pca.apply_pca(S_aligned, t=t)
    mode_std = np.sqrt(mode_var)
    explained = mode_var / mode_var.sum()

    # Print mode standard deviations with 3 decimals (as assignment requires)
    std_str = ", ".join([f"{s:.3f}" for s in mode_std])
    exp_str = ", ".join([f"{r:.3f}" for r in explained])
    print(f"Mode STD (top-{t}): [{std_str}]")
    print(f"Explained ratio (within top-{t}): [{exp_str}]")

    # 5) Visualize ±3σ for the first three modes (using your curves)
    for k in range(min(3, P.shape[1])):
        b3 = 3.0 * mode_std[k]
        shape_minus = mean_vec - P[:, k] * b3
        shape_plus  = mean_vec + P[:, k] * b3

        fig = plt.figure(figsize=(5.2, 5.2))
        ax = fig.add_subplot(111)

        draw_shape_with_curves(ax, shape_minus, CURVES_45, color="g", label=f"-3σ (mode{k+1})")
        draw_shape_with_curves(ax, mean_vec,    CURVES_45, color="k", label="mean")
        draw_shape_with_curves(ax, shape_plus,  CURVES_45, color="m", label=f"+3σ (mode{k+1})")

        ax.legend(); ax.set_aspect("equal")
        ax.set_title(f"Mode {k+1} Variation (±3σ)")
        save_fig(fig, os.path.join(RES_DIR, f"mode{k+1}_variation.png"))

    # 6) Shape parameters b for every sample:
    #    b = (X - mean) @ P
    B = (S_aligned - mean_vec) @ P   # (n_samples, t)
    np.savetxt(os.path.join(RES_DIR, "b_matrix.csv"), B, delimiter=",", fmt="%.6f")
    print(f"Saved: {os.path.join(RES_DIR, 'b_matrix.csv')}")

    # 7) Scatter plot of (b1, b2): first half = Neutral, second half = Smile
    mid = n_samples // 2
    fig = plt.figure(figsize=(5.4, 4.2))
    ax = fig.add_subplot(111)
    ax.scatter(B[:mid, 0], B[:mid, 1], c="tab:blue", alpha=0.75, label="Neutral")
    ax.scatter(B[mid:, 0], B[mid:, 1], c="tab:orange", alpha=0.75, label="Smile")
    ax.set_xlabel("b1"); ax.set_ylabel("b2"); ax.legend()
    ax.set_title("Scatter of (b1, b2)")
    save_fig(fig, os.path.join(RES_DIR, "scatter_b1_b2.png"))

    # 8) Mean neutral vs mean smiling (overlay)
    mean_neutral = S_aligned[:mid].mean(axis=0)
    mean_smile   = S_aligned[mid:].mean(axis=0)
    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_subplot(111)
    draw_shape_with_curves(ax, mean_neutral, CURVES_45, color="tab:blue", label="Neutral (mean)")
    draw_shape_with_curves(ax, mean_smile,   CURVES_45, color="tab:red",  label="Smile (mean)")
    ax.legend(); ax.set_aspect("equal")
    ax.set_title("Mean Shape: Neutral vs Smiling")
    save_fig(fig, os.path.join(RES_DIR, "smile_vs_neutral.png"))

    # 9) Projection on "smile direction" and histogram + simple threshold accuracy
    direction = mean_smile - mean_neutral
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    proj_all = (S_aligned @ direction)              # (n_samples,)
    proj_neutral = proj_all[:mid]
    proj_smile   = proj_all[mid:]

    fig = plt.figure(figsize=(5.6, 4.2))
    ax = fig.add_subplot(111)
    ax.hist(proj_neutral, bins=20, alpha=0.65, label="Neutral")
    ax.hist(proj_smile,   bins=20, alpha=0.65, label="Smile")
    ax.set_xlabel("Projection on Smile Direction"); ax.set_ylabel("Count")
    ax.legend(); ax.set_title("Histogram on Smile Direction")
    save_fig(fig, os.path.join(RES_DIR, "histogram_projection.png"))

    thr = 0.5 * (proj_neutral.mean() + proj_smile.mean())
    preds = (proj_all > thr).astype(int)            # neutral=0, smile=1
    labels = np.zeros(n_samples, dtype=int); labels[mid:] = 1
    acc = (preds == labels).mean()

    # 10) Class-wise mean of b (first t modes)
    mean_b_neutral = B[:mid].mean(axis=0)
    mean_b_smile   = B[mid:].mean(axis=0)
    mean_b_neutral_str = ", ".join([f"{x:.3f}" for x in mean_b_neutral[:t]])
    mean_b_smile_str   = ", ".join([f"{x:.3f}" for x in mean_b_smile[:t]])

    # Save metrics
    metrics_path = os.path.join(RES_DIR, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Mode STD (top-{t}): [{std_str}]\n")
        f.write(f"Explained ratio (within top-{t}): [{exp_str}]\n")
        f.write(f"Mean b (neutral, first {t}): [{mean_b_neutral_str}]\n")
        f.write(f"Mean b (smile,   first {t}): [{mean_b_smile_str}]\n")
        f.write(f"Smile/Neutral threshold: {thr:.4f}\n")
        f.write(f"Accuracy: {acc*100:.2f}%\n")
    print(f"Saved: {metrics_path} (Accuracy: {acc*100:.2f}%)")


if __name__ == "__main__":
    main()
