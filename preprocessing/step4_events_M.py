"""
Step 4: Event detection (idling, harsh braking, sharp turns) on trip M.
Reads S-M_filtered.csv + V-M_aligned.csv, adds flag columns, writes S-M_events.csv.
"""

import pandas as pd
import numpy as np

S_PATH = "preprocessing/output/S-M_filtered.csv"
V_PATH = "preprocessing/output/V-M_aligned.csv"
OUT_PATH = "preprocessing/output/S-M_events.csv"

VELOCITY_COL = "Velocity (km/hr)"
YAW_RATE_COL = "Yaw Rate (deg/sec)"
LONG_ACCEL_COL = "Indicated Longitudinal Acceleration (g)"

IDLE_SPEED_THRESH = 2.0     # km/h
HARSH_BRAKE_G = -0.3        # g, longitudinal
SHARP_TURN_YAW = 20.0       # deg/s


def main():
    s = pd.read_csv(S_PATH)
    v = pd.read_csv(V_PATH)
    if len(s) != len(v):
        raise ValueError(f"Row mismatch: S={len(s)} V={len(v)}")

    idling = v[VELOCITY_COL] < IDLE_SPEED_THRESH
    harsh_braking = v[LONG_ACCEL_COL] < HARSH_BRAKE_G
    sharp_turn = v[YAW_RATE_COL].abs() > SHARP_TURN_YAW

    s["flag_idling"] = idling.astype(int)
    s["flag_harsh_braking"] = harsh_braking.astype(int)
    s["flag_sharp_turn"] = sharp_turn.astype(int)
    s["flag_non_navigation"] = (idling).astype(int)  # gate for training windows

    print(f"Idling:        {idling.sum():>7} rows ({100*idling.mean():.1f}%)")
    print(f"Harsh braking: {harsh_braking.sum():>7} rows ({100*harsh_braking.mean():.1f}%)")
    print(f"Sharp turn:    {sharp_turn.sum():>7} rows ({100*sharp_turn.mean():.1f}%)")

    s.to_csv(OUT_PATH, index=False)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()