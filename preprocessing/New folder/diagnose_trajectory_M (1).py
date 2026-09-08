"""
preprocessing/diagnose_trajectory_M.py

Diagnostic for the M trajectory mismatch. Builds a second, independent
trajectory directly from V-M's own Latitude/Longitude columns (simple
equirectangular projection, centered at the trip's start point) and
compares it against the dx/dy-integrated trajectory from Step 5.

If the two match closely -> Step 5's dx/dy math is correct; the
visual mismatch against the satellite map was likely a plotting/scale
issue, not a data bug.

If the two diverge -> there's a real bug in the dx/dy derivation
(heading convention, sample period, or similar) that needs fixing.

Also prints diagnostics on Sample Period and Heading to catch any
anomalies (gaps, non-constant period, etc.) specific to trip M.

Run from project root:
    python preprocessing/diagnose_trajectory_M.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

v = pd.read_csv("preprocessing/output/V-M_labels.csv")

# --- Sanity checks on Sample Period and Heading ---
period_col = [c for c in v.columns if "sample" in c.lower() and "period" in c.lower()][0]
heading_col = [c for c in v.columns if "heading" in c.lower()][0]
lat_col = [c for c in v.columns if c.lower().strip() == "latitude (degrees)"][0]
lon_col = [c for c in v.columns if c.lower().strip() == "longitude (degrees)"][0]
speed_col = [c for c in v.columns if "velocity" in c.lower()][0]

print("=== Sample period sanity ===")
print(v[period_col].describe())
n_nonstandard = (v[period_col] != 0.1).sum()
print(f"Rows where sample period != 0.1: {n_nonstandard} / {len(v)}")

print("\n=== Heading sanity ===")
print(v[heading_col].describe())

# --- Build ground-truth trajectory directly from lat/lon ---
lat = v[lat_col].values.astype(float)
lon = v[lon_col].values.astype(float)
lat0, lon0 = lat[0], lon[0]
# equirectangular approx (fine for a ~10km-scale trip)
R_earth = 6371000.0
lat0_rad = np.deg2rad(lat0)
x_gps = np.deg2rad(lon - lon0) * R_earth * np.cos(lat0_rad)
y_gps = np.deg2rad(lat - lat0) * R_earth

# --- dx/dy-integrated trajectory (already in the labels file) ---
x_cum = v["x_cum"].values
y_cum = v["y_cum"].values

# --- Compare ---
end_gap = np.sqrt((x_gps[-1] - x_cum[-1]) ** 2 + (y_gps[-1] - y_cum[-1]) ** 2)
gps_total_dist = np.sum(np.sqrt(np.diff(x_gps) ** 2 + np.diff(y_gps) ** 2))
print(f"\n=== Trajectory comparison ===")
print(f"GPS-derived total path length: {gps_total_dist:.1f} m")
print(f"dx/dy-integrated total path length: {v.attrs.get('total_path_length', 'n/a')}")
print(f"Endpoint gap between the two methods: {end_gap:.1f} m")

plt.figure(figsize=(9, 9))
plt.plot(x_gps, y_gps, label="From V-M lat/lon (ground truth)", linewidth=1.2, color="black")
plt.plot(x_cum, y_cum, label="From dx/dy integration (Step 5)", linewidth=1.0, color="red", alpha=0.7)
plt.scatter([x_gps[0]], [y_gps[0]], color="green", zorder=5, label="start")
plt.axis("equal")
plt.legend()
plt.grid(True, alpha=0.3)
plt.title("Trip M: lat/lon-derived vs dx/dy-integrated trajectory")
plt.savefig("preprocessing/output/M_trajectory_comparison.png", dpi=120, bbox_inches="tight")
print("\nSaved: preprocessing/output/M_trajectory_comparison.png")
