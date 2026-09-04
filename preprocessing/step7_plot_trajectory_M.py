"""
Step 7: Plot IMU-only dead-reckoned path vs true path (trip M).

IMU-only path = naive baseline (heading from gyro yaw integration,
speed from accel_forward integration, then position integration).
This is NOT the AI model output - it's the classical double-integration
baseline mentioned in the training plan, useful here as a quick visual
sanity check of how much naive IMU-only drifts vs the true GPS/CAN path.

True path = from V-M_labels.csv (x_cum, y_cum), derived from CAN velocity+heading.

Reads: S-M_filtered.csv, V-M_labels.csv
Shows: matplotlib plot, both paths overlaid.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

S_PATH = "preprocessing/output/S-M_filtered.csv"
V_LABELS_PATH = "preprocessing/output/V-M_labels.csv"

DT = 0.1  # seconds, 10Hz

ACCEL_FWD_COL = "accel_forward_filtered"
GYRO_YAW_COL = "GYROSCOPE Yaw (rad/s)"   # verify against actual column name if this errors


def main():
    s = pd.read_csv(S_PATH)
    v = pd.read_csv(V_LABELS_PATH)

    if len(s) != len(v):
        raise ValueError(f"Row mismatch: S={len(s)} V={len(v)}")

    # --- IMU-only naive dead reckoning ---
    # heading: integrate gyro yaw rate
    heading = np.cumsum(s[GYRO_YAW_COL].values * DT)  # radians, relative to start

    # speed: integrate forward accel (naive - no zero-velocity correction, will drift)
    speed = np.cumsum(s[ACCEL_FWD_COL].values * DT)
    speed = np.clip(speed, 0, None)  # vehicle doesn't go backward; crude clamp

    # position: integrate velocity components
    dx_imu = speed * np.sin(heading) * DT
    dy_imu = speed * np.cos(heading) * DT
    x_imu = np.cumsum(dx_imu)
    y_imu = np.cumsum(dy_imu)

    # --- true path (already computed in Step 5) ---
    x_true = v["x_cum"].values
    y_true = v["y_cum"].values

    # --- plot ---
    plt.figure(figsize=(10, 8))
    plt.plot(x_true, y_true, label="True path (CAN/GPS)", color="green", linewidth=2)
    plt.plot(x_imu, y_imu, label="IMU-only dead reckoning (naive)", color="red", linewidth=1, alpha=0.8)
    plt.scatter([x_true[0]], [y_true[0]], color="black", marker="o", s=80, label="Start", zorder=5)
    plt.scatter([x_true[-1]], [y_true[-1]], color="green", marker="x", s=100, label="True end", zorder=5)
    plt.scatter([x_imu[-1]], [y_imu[-1]], color="red", marker="x", s=100, label="IMU-only end", zorder=5)
    plt.xlabel("X (m, east)")
    plt.ylabel("Y (m, north)")
    plt.title("Trip M: True path vs naive IMU-only dead reckoning")
    plt.legend()
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    end_drift = np.sqrt((x_imu[-1] - x_true[-1])**2 + (y_imu[-1] - y_true[-1])**2)
    print(f"End-point drift (IMU-only vs true): {end_drift:.1f} m")


if __name__ == "__main__":
    main()