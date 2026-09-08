"""
preprocessing/windowing.py

Step 6: windowing with the local-frame label rotation fix.
"""

import os
import numpy as np
import pandas as pd

WINDOWS_DIR = os.path.join("preprocessing/output", "06_windows")
os.makedirs(WINDOWS_DIR, exist_ok=True)


def window_trip(trip_name, output_dir="preprocessing/output", verbose=True,
                 window_size=40, stride=10,
                 imu_cols=("accel_forward_filt", "accel_lateral_filt",
                           "accel_vertical_filt",
                           "GYROSCOPE Yaw (rad/s)", "GYROSCOPE Pitch (rad/s)",
                           "GYROSCOPE Roll (rad/s)")):
    """
    Step 6: slice S-{trip}_events.csv (IMU, body-frame) and
    V-{trip}_labels.csv (dx, dy, heading_rad, global-frame) into
    overlapping windows.

    THE FIX: the model input (IMU) is body-frame -- "forward" means
    whichever way the car currently points, no compass reference. A
    stateless per-window model cannot recover absolute heading from
    body-frame input alone. So the training TARGET must also be in the
    vehicle's local frame at the window's start heading, not global
    compass frame -- otherwise two physically identical windows occurring
    at different points in the trip (different absolute heading) get
    different "correct" labels with nothing in X to distinguish them.

    For each window:
      1. Sum global dx, dy across the window -> (dx_g, dy_g).
      2. Rotate by -heading_start (heading_rad at the window's first row)
         to express that same displacement in the frame where "forward"
         = the car's heading at window start.
         (Global convention here is dx=sin(heading), dy=cos(heading),
         i.e. heading measured from north/y-axis, so the inverse
         rotation matches that convention -- see code below.)
      3. Also keep dx_g, dy_g, heading_start per window so predictions can
         be rotated back to global frame later for trajectory
         reconstruction/evaluation (position integrator, stage 7).

    Returns dict with:
        X            : (n_windows, window_size, len(imu_cols)) float32
        y_local      : (n_windows, 2) -- (dx_local, dy_local), THE TRAINING TARGET
        y_global     : (n_windows, 2) -- (dx_global, dy_global), for reconstruction
        heading_start: (n_windows,) heading_rad at each window's first row
        out_path     : saved .npz path
    """
    s_path = os.path.join("preprocessing/output/04_events", f"S-{trip_name}_events.csv")
    v_path = os.path.join("preprocessing/output/05_labels", f"V-{trip_name}_labels.csv")
    if not os.path.exists(s_path):
        raise FileNotFoundError(f"Events file not found for trip '{trip_name}': {s_path}")
    if not os.path.exists(v_path):
        raise FileNotFoundError(f"Labels file not found for trip '{trip_name}': {v_path}")

    s_df = pd.read_csv(s_path)
    v_df = pd.read_csv(v_path)
    n = min(len(s_df), len(v_df))
    s_df = s_df.iloc[:n].reset_index(drop=True)
    v_df = v_df.iloc[:n].reset_index(drop=True)

    missing_imu = [c for c in imu_cols if c not in s_df.columns]
    if missing_imu:
        raise ValueError(f"[{trip_name}] Missing IMU columns for windowing: {missing_imu}")
    if "heading_rad" not in v_df.columns:
        raise ValueError(
            f"[{trip_name}] V-labels file has no heading_rad column -- "
            f"re-run derive_labels() with the updated version first."
        )

    imu = s_df[list(imu_cols)].values.astype(np.float32)
    dx = v_df["dx"].values.astype(float)
    dy = v_df["dy"].values.astype(float)
    heading = v_df["heading_rad"].values.astype(float)

    X_list, y_local_list, y_global_list, h0_list = [], [], [], []

    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        X_list.append(imu[start:end])

        dx_g = dx[start:end].sum()
        dy_g = dy[start:end].sum()
        h0 = heading[start]  # heading at window START, not average/end

        # Global convention (derive_labels): dx = v*sin(h), dy = v*cos(h),
        # i.e. h measured clockwise from the y-axis (compass-style).
        # Rotating a compass-convention vector INTO the frame where the
        # car's heading-at-start points along local "forward" (+y_local)
        # is the inverse of that same rotation:
        #   dx_local =  dx_g*cos(h0) - dy_g*sin(h0)
        #   dy_local =  dx_g*sin(h0) + dy_g*cos(h0)
        dx_local = dx_g * np.cos(h0) - dy_g * np.sin(h0)
        dy_local = dx_g * np.sin(h0) + dy_g * np.cos(h0)

        y_local_list.append([dx_local, dy_local])
        y_global_list.append([dx_g, dy_g])
        h0_list.append(h0)

    X = np.stack(X_list, axis=0)
    y_local = np.array(y_local_list, dtype=np.float32)
    y_global = np.array(y_global_list, dtype=np.float32)
    heading_start = np.array(h0_list, dtype=np.float32)

    n_windows = X.shape[0]
    if verbose:
        local_mag = np.sqrt((y_local ** 2).sum(axis=1))
        global_mag = np.sqrt((y_global ** 2).sum(axis=1))
        print(f"[{trip_name}] Windows: {n_windows} (size={window_size}, stride={stride})")
        print(f"[{trip_name}] X shape: {X.shape}")
        print(f"[{trip_name}] |y_local| mean={local_mag.mean():.2f}m  "
              f"|y_global| mean={global_mag.mean():.2f}m  "
              f"(should match closely -- rotation preserves magnitude)")

    out_path = os.path.join(WINDOWS_DIR, f"{trip_name}_windows.npz")
    np.savez(out_path, X=X, y_local=y_local, y_global=y_global,
             heading_start=heading_start)
    if verbose:
        print(f"[{trip_name}] Saved: {out_path}")

    return {
        "X": X,
        "y_local": y_local,
        "y_global": y_global,
        "heading_start": heading_start,
        "out_path": out_path,
    }
