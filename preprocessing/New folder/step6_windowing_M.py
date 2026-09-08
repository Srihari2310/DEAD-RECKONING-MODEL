"""
Step 6: Windowing (trip M).
Slices aligned/filtered/labeled data into overlapping windows.
X = conditioned IMU channels over the window.
Y = total (dx, dy) displacement summed over the window (from V-M labels).
Non-navigation (idling) windows are flagged, not dropped (filter at train time).

Reads S-M_filtered.csv, S-M_events.csv, V-M_labels.csv.
Writes: preprocessing/output/M_windows.npz  (X, Y, flags arrays)
"""

import pandas as pd
import numpy as np

S_FILT_PATH = "preprocessing/output/S-M_filtered.csv"
S_EVENTS_PATH = "preprocessing/output/S-M_events.csv"
V_LABELS_PATH = "preprocessing/output/V-M_labels.csv"
OUT_PATH = "preprocessing/output/M_windows.npz"

FS = 10.0
WINDOW_SEC = 4.0
STRIDE_SEC = 1.0
WINDOW_LEN = int(WINDOW_SEC * FS)   # 40 samples
STRIDE = int(STRIDE_SEC * FS)       # 10 samples

IMU_CHANNELS = [
    "accel_forward_filtered",
    "accel_lateral_filtered",
    "accel_vertical_filtered",
    "GYROSCOPE Yaw (rad/s)",
    "GYROSCOPE Pitch (rad/s)",
    "GYROSCOPE Roll (rad/s)",
]


def main():
    s_filt = pd.read_csv(S_FILT_PATH)
    s_ev = pd.read_csv(S_EVENTS_PATH)
    v_lab = pd.read_csv(V_LABELS_PATH)

    n = len(s_filt)
    if not (len(s_ev) == n and len(v_lab) == n):
        raise ValueError(f"Row mismatch: filt={n} events={len(s_ev)} labels={len(v_lab)}")

    missing = [c for c in IMU_CHANNELS if c not in s_filt.columns]
    if missing:
        raise ValueError(f"Missing IMU columns: {missing}. Check actual header names in S-M_filtered.csv")

    imu = s_filt[IMU_CHANNELS].values          # (n, 6)
    dx = v_lab["dx"].values
    dy = v_lab["dy"].values
    idling_flag = s_ev["flag_idling"].values

    X, Y, non_nav_frac = [], [], []
    for start in range(0, n - WINDOW_LEN + 1, STRIDE):
        end = start + WINDOW_LEN
        X.append(imu[start:end, :])
        Y.append([dx[start:end].sum(), dy[start:end].sum()])
        non_nav_frac.append(idling_flag[start:end].mean())

    X = np.array(X)               # (num_windows, 40, 6)
    Y = np.array(Y)                # (num_windows, 2)
    non_nav_frac = np.array(non_nav_frac)

    print(f"Windows created: {X.shape[0]}")
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")
    print(f"Windows >50% idling: {(non_nav_frac > 0.5).sum()} ({100*(non_nav_frac>0.5).mean():.1f}%)")

    np.savez(OUT_PATH, X=X, Y=Y, non_nav_frac=non_nav_frac)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()