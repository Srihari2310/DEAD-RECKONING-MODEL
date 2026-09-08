"""
Step 5: Displacement label derivation (trip M).
Converts V-M's Velocity + Heading into per-timestep (dx, dy) displacement
in a local ENU-ish frame (meters), using the corrected alignment.
This is the regression target for training — GPS/CAN is the answer key only.

Reads V-M_aligned.csv, writes V-M_labels.csv (adds dx, dy, cumulative x, y).
"""

import pandas as pd
import numpy as np

V_PATH = "preprocessing/output/V-M_aligned.csv"
OUT_PATH = "preprocessing/output/V-M_labels.csv"

VELOCITY_COL = "Velocity (km/hr)"
HEADING_COL = "Heading (degrees)"
DT = 0.1  # seconds, confirmed exact 10Hz sample period


def main():
    v = pd.read_csv(V_PATH)

    speed_mps = v[VELOCITY_COL].values / 3.6
    heading_rad = np.deg2rad(v[HEADING_COL].values)

    # per-timestep displacement in local frame (x=east, y=north; adjust if heading convention differs)
    dx = speed_mps * np.sin(heading_rad) * DT
    dy = speed_mps * np.cos(heading_rad) * DT

    v["dx"] = dx
    v["dy"] = dy
    v["x_cum"] = np.cumsum(dx)
    v["y_cum"] = np.cumsum(dy)

    total_dist = np.sqrt(dx**2 + dy**2).sum()
    straight_line_dist = np.sqrt(v["x_cum"].iloc[-1]**2 + v["y_cum"].iloc[-1]**2)

    print(f"Total path length (sum of |displacement|): {total_dist:.1f} m")
    print(f"Straight-line start->end distance:          {straight_line_dist:.1f} m")
    print(f"Rows: {len(v)}")

    v.to_csv(OUT_PATH, index=False)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()