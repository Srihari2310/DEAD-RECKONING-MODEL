"""
Step 3b: Sanity-check remaining extremes in filtered accel signals
against V-M ground truth (velocity, heading change, braking).

Reads:
  preprocessing/output/S-M_filtered.csv
  preprocessing/output/V-M_aligned.csv
(both already row-aligned 1:1 from Step 1)
"""

import pandas as pd
import numpy as np

S_PATH = "preprocessing/output/S-M_filtered.csv"
V_PATH = "preprocessing/output/V-M_aligned.csv"

TOP_N = 10
WINDOW = 5

VELOCITY_COL = "Velocity (km/hr)"
YAW_RATE_COL = "Yaw Rate (deg/sec)"
BRAKE_POS_COL = "Brake Position (0 or 1)"
BRAKE_PRESSURE_COL = "Brake Pressure (psi)"
LAT_ACCEL_COL = "Indicated Lateral Acceleration (g)"
LONG_ACCEL_COL = "Indicated Longitudinal Acceleration (g)"

CHECK_CHANNELS = ["accel_lateral_filtered", "accel_vertical_filtered"]


def main():
    s = pd.read_csv(S_PATH)
    v = pd.read_csv(V_PATH)

    if len(s) != len(v):
        raise ValueError(f"Row count mismatch: S={len(s)} V={len(v)} (expected equal, already aligned)")

    for col in CHECK_CHANNELS:
        if col not in s.columns:
            print(f"Skipping {col} - not found")
            continue

        vals = s[col]
        extreme_idx = vals.abs().sort_values(ascending=False).index[:TOP_N]

        print(f"\n=== Top {TOP_N} extremes: {col} ===")
        for idx in extreme_idx:
            row_val = s.loc[idx, col]
            v_speed = v.loc[idx, VELOCITY_COL]
            v_yaw = v.loc[idx, YAW_RATE_COL]
            v_brake_pos = v.loc[idx, BRAKE_POS_COL]
            v_brake_psi = v.loc[idx, BRAKE_PRESSURE_COL]
            v_lat_g = v.loc[idx, LAT_ACCEL_COL]
            v_long_g = v.loc[idx, LONG_ACCEL_COL]

            print(
                f"row {idx:>7} | {col}={row_val:7.2f} | speed={v_speed:6.1f} km/h | "
                f"yaw_rate={v_yaw:7.2f} deg/s | lat_g={v_lat_g:6.2f} | long_g={v_long_g:6.2f} | "
                f"brake_pos={v_brake_pos} brake_psi={v_brake_psi}"
            )

    print("\nInterpretation guide:")
    print("- accel_lateral extremes should line up with high |heading_rate| (turning).")
    print("- accel_vertical extremes with no clear speed/brake correlation -> likely pothole/bump noise, not a real event.")


if __name__ == "__main__":
    main()