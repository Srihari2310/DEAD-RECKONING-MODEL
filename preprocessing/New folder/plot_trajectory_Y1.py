"""
Sanity-check plot: Trip Y1's reconstructed trajectory from V-file ground truth
(V-Y1_labels.csv, produced by derive_labels() in common.py).

This does NOT use the IMU/model at all -- it plots the ground-truth
displacement (dx, dy from V-file Velocity+Heading, cumulatively summed)
to check whether Y1's alignment/labels look like a real driving route,
given the weak lag-detection correlation flagged for this trip.

Run from project root:
    python plot_trajectory_Y1.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

LABELS_PATH = os.path.join("preprocessing", "output", "V-Y1_labels.csv")


def find_cum_cols(df):
    """Find cumulative x/y columns regardless of exact naming."""
    candidates_x = ["x_cum", "x_cumulative", "cum_x", "X_cum"]
    candidates_y = ["y_cum", "y_cumulative", "cum_y", "Y_cum"]

    x_col = next((c for c in candidates_x if c in df.columns), None)
    y_col = next((c for c in candidates_y if c in df.columns), None)

    if x_col and y_col:
        return x_col, y_col

    # fallback: derive from dx/dy if cumulative columns aren't present
    dx_col = next((c for c in ["dx", "Dx", "DX"] if c in df.columns), None)
    dy_col = next((c for c in ["dy", "Dy", "DY"] if c in df.columns), None)
    if dx_col and dy_col:
        df["x_cum_derived"] = df[dx_col].cumsum()
        df["y_cum_derived"] = df[dy_col].cumsum()
        return "x_cum_derived", "y_cum_derived"

    raise KeyError(
        f"Could not find cumulative or dx/dy columns in {LABELS_PATH}. "
        f"Available columns: {list(df.columns)}"
    )


def main():
    if not os.path.exists(LABELS_PATH):
        print(f"ERROR: {LABELS_PATH} not found. Run derive_labels('Y1') first.")
        return

    df = pd.read_csv(LABELS_PATH)
    df.columns = [c.strip() for c in df.columns]

    x_col, y_col = find_cum_cols(df)
    x, y = df[x_col].to_numpy(), df[y_col].to_numpy()

    total_dist = ((df[x_col].diff().fillna(0)) ** 2 + (df[y_col].diff().fillna(0)) ** 2) ** 0.5
    print(f"Points: {len(df)}")
    print(f"Net displacement (start->end): {((x[-1]-x[0])**2 + (y[-1]-y[0])**2)**0.5:.1f} m")
    print(f"Total path length: {total_dist.sum():.1f} m")
    print(f"Bounding box: x=[{x.min():.1f}, {x.max():.1f}]  y=[{y.min():.1f}, {y.max():.1f}]")

    plt.figure(figsize=(10, 8))
    plt.plot(x, y, color="blue", linewidth=1.2, label="Y1 reconstructed route (ground truth)")
    plt.scatter(x[0], y[0], color="green", s=100, zorder=5, label="Start")
    plt.scatter(x[-1], y[-1], color="red", s=100, zorder=5, label="End")
    plt.title("Trip Y1 — reconstructed trajectory from V-file ground truth")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="best")
    print("Displaying plot...")
    plt.show()


if __name__ == "__main__":
    main()
