"""
Step 3: Noise filtering on calibrated IMU signals (trip M).
Reads S-M_calibrated.csv + V-M_aligned.csv, writes S-M_filtered.csv.
"""

import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt

S_PATH = "preprocessing/output/S-M_calibrated.csv"
V_PATH = "preprocessing/output/V-M_aligned.csv"
OUT_PATH = "preprocessing/output/S-M_filtered.csv"

FS = 10.0  # Hz
YAW_RATE_COL = "Yaw Rate (deg/sec)"
LAT_ACCEL_COL = "Indicated Lateral Acceleration (g)"

YAW_GATE_THRESH = 5.0      # deg/s
LATG_GATE_THRESH = 0.1     # g
LATERAL_CLIP = 2.0         # m/s^2, when gated
FWD_VERT_CLIP = 10.0       # m/s^2, simple despike

HP_FREQ = 0.1   # Hz
LP_FREQ = 4.5   # Hz (Nyquist=5Hz at 10Hz sampling)


def bandpass(sig, fs, hp, lp, order=4):
    nyq = fs / 2
    b, a = butter(order, [hp / nyq, lp / nyq], btype="band")
    return filtfilt(b, a, sig)


def main():
    s = pd.read_csv(S_PATH)
    v = pd.read_csv(V_PATH)
    if len(s) != len(v):
        raise ValueError(f"Row mismatch: S={len(s)} V={len(v)}")

    # --- forward / vertical: simple despike + bandpass ---
    for col, out in [("accel_forward", "accel_forward_filtered"),
                      ("accel_vertical", "accel_vertical_filtered")]:
        clipped = s[col].clip(-FWD_VERT_CLIP, FWD_VERT_CLIP)
        s[out] = bandpass(clipped.values, FS, HP_FREQ, LP_FREQ)

    # --- lateral: yaw-rate gate AND lat_g gate, then bandpass ---
    yaw_not_turning = v[YAW_RATE_COL].abs() < YAW_GATE_THRESH
    latg_not_turning = v[LAT_ACCEL_COL].abs() < LATG_GATE_THRESH
    gate = yaw_not_turning | latg_not_turning  # gate clips if EITHER says "no real lateral force"

    lateral = s["accel_lateral"].copy()
    lateral[gate] = lateral[gate].clip(-LATERAL_CLIP, LATERAL_CLIP)
    s["accel_lateral_filtered"] = bandpass(lateral.values, FS, HP_FREQ, LP_FREQ)

    gated_pct = 100 * gate.sum() / len(gate)
    print(f"Lateral: {gated_pct:.1f}% rows gated (yaw OR lat_g says not turning)")

    s.to_csv(OUT_PATH, index=False)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()