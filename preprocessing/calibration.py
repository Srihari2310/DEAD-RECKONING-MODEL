"""
preprocessing/calibration.py

Step 2: static leveling + dynamic yaw alignment for any trip.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd

from .utils import find_col

CALIBRATED_DIR = Path("preprocessing/output/02_calibrated")


def _rotation_between_vectors(a, b):
    """
    Return the 3x3 rotation matrix that rotates unit vector a onto unit vector b.
    Uses Rodrigues' rotation formula. Handles the near-parallel / near-anti-parallel
    edge cases safely.
    """
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)

    if s < 1e-10:
        # a and b are already (anti-)parallel
        if c > 0:
            return np.eye(3)
        else:
            # 180-degree rotation: pick any perpendicular axis
            axis = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            axis = axis - a * np.dot(axis, a)
            axis = axis / np.linalg.norm(axis)
            return 2 * np.outer(axis, axis) - np.eye(3)

    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])
    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))
    return R


def _rotation_about_z(angle_rad):
    """3x3 rotation matrix about the Z axis by angle_rad."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1],
    ])


def calibrate_trip(trip_name, output_dir="preprocessing/output", verbose=True,
                    speed_threshold_kmh=15.0, yaw_rate_threshold_degs=3.0,
                    min_accel_mag=0.3):
    """
    Step 2: calibration for any trip. Reads the Step-1-aligned S/V CSVs,
    performs static leveling (gravity -> true vertical) and dynamic yaw
    alignment (phone mounting offset estimated from horizontal accel
    direction vs GPS heading during straight, moderate-speed driving),
    and writes accel_forward/lateral/vertical columns.

    Returns dict with the calibrated dataframe, rotation matrices, and
    diagnostic stats (gravity vector, yaw offset, sample counts).
    """
    s_path = Path("preprocessing/output/01_aligned") / f"S-{trip_name}_aligned.csv"
    v_path = Path("preprocessing/output/01_aligned") / f"V-{trip_name}_aligned.csv"

    if not os.path.exists(s_path) or not os.path.exists(v_path):
        raise FileNotFoundError(
            f"Aligned files not found for trip '{trip_name}'. "
            f"Expected {s_path} and {v_path} — run Step 1 first."
        )

    s_df = pd.read_csv(s_path)
    v_df = pd.read_csv(v_path)
    n = min(len(s_df), len(v_df))
    s_df = s_df.iloc[:n].reset_index(drop=True)
    v_df = v_df.iloc[:n].reset_index(drop=True)

    accel_x_col = find_col(s_df.columns, ["accelerometer", "x"])
    accel_y_col = find_col(s_df.columns, ["accelerometer", "y"])
    accel_z_col = find_col(s_df.columns, ["accelerometer", "z"])
    grav_x_col = find_col(s_df.columns, ["gravity", "x"])
    grav_y_col = find_col(s_df.columns, ["gravity", "y"])
    grav_z_col = find_col(s_df.columns, ["gravity", "z"])
    v_heading_col = find_col(v_df.columns, ["heading"])
    v_speed_col = find_col(v_df.columns, ["velocity"])
    v_yawrate_col = find_col(v_df.columns, ["yaw", "rate"])

    missing = [name for name, col in [
        ("accel_x", accel_x_col), ("accel_y", accel_y_col), ("accel_z", accel_z_col),
        ("gravity_x", grav_x_col), ("gravity_y", grav_y_col), ("gravity_z", grav_z_col),
        ("heading", v_heading_col), ("speed", v_speed_col), ("yaw_rate", v_yawrate_col),
    ] if col is None]
    if missing:
        raise ValueError(f"[{trip_name}] Could not auto-detect columns: {missing}")

    accel = s_df[[accel_x_col, accel_y_col, accel_z_col]].values.astype(float)
    gravity = s_df[[grav_x_col, grav_y_col, grav_z_col]].values.astype(float)
    linear_accel = accel - gravity

    # --- Static leveling ---
    gravity_mean = np.nanmean(gravity, axis=0)
    gravity_mean_unit = gravity_mean / np.linalg.norm(gravity_mean)
    target_vertical = np.array([0.0, 0.0, 1.0])
    R_level = _rotation_between_vectors(gravity_mean_unit, target_vertical)

    leveled = (R_level @ linear_accel.T).T  # shape (n, 3): x', y', z'

    # --- Dynamic yaw alignment ---
    speed = v_df[v_speed_col].values.astype(float)
    yaw_rate = v_df[v_yawrate_col].values.astype(float)
    heading_deg = v_df[v_heading_col].values.astype(float)
    heading_rad = np.deg2rad(heading_deg)

    horiz_mag = np.sqrt(leveled[:, 0] ** 2 + leveled[:, 1] ** 2)

    mask = (
        (speed > speed_threshold_kmh)
        & (np.abs(yaw_rate) < yaw_rate_threshold_degs)
        & (horiz_mag > min_accel_mag)
    )
    n_qualified = int(mask.sum())

    if n_qualified < 50:
        raise ValueError(
            f"[{trip_name}] Only {n_qualified} qualifying samples for yaw alignment "
            f"(need >=50). Consider loosening thresholds."
        )

    angle_accel = np.arctan2(leveled[mask, 0], leveled[mask, 1])  # atan2(x', y') -> compass-style
    angle_diff = heading_rad[mask] - angle_accel
    weights = horiz_mag[mask]

    # magnitude-weighted circular mean of angle_diff
    sin_sum = np.sum(weights * np.sin(angle_diff))
    cos_sum = np.sum(weights * np.cos(angle_diff))
    yaw_offset_rad = np.arctan2(sin_sum, cos_sum)
    yaw_offset_deg = np.rad2deg(yaw_offset_rad)

    R_yaw = _rotation_about_z(yaw_offset_rad)
    R_total = R_yaw @ R_level

    vehicle_frame = (R_total @ linear_accel.T).T  # x=lateral, y=forward, z=vertical

    s_df = s_df.copy()
    s_df["accel_lateral"] = vehicle_frame[:, 0]
    s_df["accel_forward"] = vehicle_frame[:, 1]
    s_df["accel_vertical"] = vehicle_frame[:, 2]

    if verbose:
        print(f"[{trip_name}] Gravity mean (unit): {gravity_mean_unit}")
        print(f"[{trip_name}] Yaw alignment: {n_qualified}/{n} samples qualified "
              f"(speed>{speed_threshold_kmh}km/h, |yaw_rate|<{yaw_rate_threshold_degs}deg/s)")
        print(f"[{trip_name}] Estimated phone mounting yaw offset: {yaw_offset_deg:.2f} deg")
        for name in ["accel_forward", "accel_lateral", "accel_vertical"]:
            col = s_df[name].values
            print(f"[{trip_name}] {name}: mean={np.nanmean(col):.3f} std={np.nanstd(col):.3f} "
                  f"min={np.nanmin(col):.3f} max={np.nanmax(col):.3f}")

    CALIBRATED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CALIBRATED_DIR / f"S-{trip_name}_calibrated.csv"
    s_df.to_csv(out_path, index=False)
    if verbose:
        print(f"[{trip_name}] Saved: {out_path}")

    return {
        "s_calibrated": s_df,
        "gravity_mean_unit": gravity_mean_unit,
        "R_level": R_level,
        "R_yaw": R_yaw,
        "R_total": R_total,
        "yaw_offset_deg": yaw_offset_deg,
        "n_qualified": n_qualified,
        "out_path": out_path,
    }
