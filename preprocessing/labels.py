"""
preprocessing/labels.py

Step 5: displacement label derivation (dx, dy, cumulative trajectory)
for any trip.
"""

import os
import numpy as np
import pandas as pd

from .utils import find_col

LABELS_DIR = os.path.join("preprocessing/output", "05_labels")
os.makedirs(LABELS_DIR, exist_ok=True)


def derive_labels(trip_name, output_dir="preprocessing/output", verbose=True,
                   plot_path=None):
    """
    Step 5: derive (dx, dy) displacement labels per row from V-file
    Velocity + Heading, using the confirmed compass convention
    (dx = v*sin(heading), dy = v*cos(heading)), scaled by sample period.
    Also computes cumulative (x_cum, y_cum) trajectory for visual
    validation against the trip's real route thumbnail.

    Reads V-{trip}_aligned.csv. Writes V-{trip}_labels.csv with
    dx, dy, x_cum, y_cum, heading_rad columns. Optionally saves a
    trajectory plot to plot_path (PNG) for visual comparison against
    the route thumbnail.
    """
    v_path = os.path.join(output_dir, "01_aligned", f"V-{trip_name}_aligned.csv")
    if not os.path.exists(v_path):
        raise FileNotFoundError(f"Aligned V-file not found for trip '{trip_name}': {v_path}")

    v_df = pd.read_csv(v_path)

    v_speed_col = find_col(v_df.columns, ["velocity"])
    v_heading_col = find_col(v_df.columns, ["heading"])
    v_period_col = find_col(v_df.columns, ["sample", "period"])
    missing = [name for name, col in [
        ("speed", v_speed_col), ("heading", v_heading_col), ("sample_period", v_period_col),
    ] if col is None]
    if missing:
        raise ValueError(f"[{trip_name}] Could not auto-detect columns: {missing}")

    speed_kmh = v_df[v_speed_col].values.astype(float)
    speed_ms = speed_kmh / 3.6
    heading_rad = np.deg2rad(v_df[v_heading_col].values.astype(float))
    period_s = v_df[v_period_col].values.astype(float)
    # guard against missing/zero sample period (fall back to 0.1s @ 10Hz)
    period_s = np.where((period_s <= 0) | np.isnan(period_s), 0.1, period_s)

    dx = speed_ms * np.sin(heading_rad) * period_s
    dy = speed_ms * np.cos(heading_rad) * period_s
    x_cum = np.cumsum(dx)
    y_cum = np.cumsum(dy)

    v_df = v_df.copy()
    v_df["dx"] = dx
    v_df["dy"] = dy
    v_df["x_cum"] = x_cum
    v_df["y_cum"] = y_cum
    v_df["heading_rad"] = heading_rad
    v_df["speed_ms"] = speed_ms
    # NOTE: dx/dy/x_cum/y_cum stay GLOBAL/compass-frame -- correct and needed
    # for route-thumbnail validation and final trajectory reconstruction.
    # They are NOT the ML training target. heading_rad is saved so Step 6
    # (windowing) can rotate each window's *summed* global displacement into
    # that window's local/start-heading frame -- see window_trip().
    # A per-row local-frame value would be meaningless (always ~0, since a
    # row's own heading trivially cancels its own displacement).

    total_path_length = np.sum(np.sqrt(dx ** 2 + dy ** 2))
    net_displacement = np.sqrt(x_cum[-1] ** 2 + y_cum[-1] ** 2)

    if verbose:
        print(f"[{trip_name}] Total path length (sum of |dx,dy|): {total_path_length:.1f} m")
        print(f"[{trip_name}] Net displacement (start to end): {net_displacement:.1f} m")
        print(f"[{trip_name}] Trajectory bounding box: x=[{x_cum.min():.1f}, {x_cum.max():.1f}] "
              f"y=[{y_cum.min():.1f}, {y_cum.max():.1f}]")

    out_path = os.path.join(LABELS_DIR, f"V-{trip_name}_labels.csv")
    v_df.to_csv(out_path, index=False)
    if verbose:
        print(f"[{trip_name}] Saved: {out_path}")

    if plot_path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 8))
        plt.plot(x_cum, y_cum, linewidth=0.8)
        plt.scatter([x_cum[0]], [y_cum[0]], color="green", label="start", zorder=5)
        plt.scatter([x_cum[-1]], [y_cum[-1]], color="red", label="end", zorder=5)
        plt.axis("equal")
        plt.xlabel("x (m, east+)")
        plt.ylabel("y (m, north+)")
        plt.title(f"Trip {trip_name} — reconstructed trajectory from V-file labels")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(plot_path, dpi=120, bbox_inches="tight")
        plt.close()
        if verbose:
            print(f"[{trip_name}] Saved trajectory plot: {plot_path}")

    return {
        "v_labels": v_df,
        "total_path_length_m": total_path_length,
        "net_displacement_m": net_displacement,
        "out_path": out_path,
    }
