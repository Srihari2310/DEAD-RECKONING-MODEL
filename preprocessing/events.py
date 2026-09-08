"""
preprocessing/events.py

Step 4: event detection (idling / harsh braking / sharp turns) for any trip.
"""

import os
import numpy as np
import pandas as pd

from .utils import find_col

EVENTS_DIR = os.path.join("preprocessing/output", "04_events")
os.makedirs(EVENTS_DIR, exist_ok=True)


def detect_events(trip_name, output_dir="preprocessing/output", verbose=True,
                   idle_speed_kmh=1.0, harsh_brake_g=0.3, sharp_turn_degs=15.0):
    """
    Step 4: flag idling, harsh-braking, and sharp-turn samples using
    V-file ground truth (training-time only -- the deployed IMU-only
    pipeline needs a separate detector, not yet designed).

    Defaults are standard automotive thresholds (tunable per trip):
      - idling: speed < idle_speed_kmh
      - harsh braking: indicated longitudinal accel < -harsh_brake_g (g's)
      - sharp turn: |yaw rate| > sharp_turn_degs (deg/s)

    Reads S-{trip}_filtered.csv and V-{trip}_aligned.csv, writes
    S-{trip}_events.csv with is_idle / is_harsh_brake / is_sharp_turn
    boolean columns.
    """
    s_path = os.path.join("preprocessing/output/03_filtered", f"S-{trip_name}_filtered.csv")
    v_path = os.path.join(output_dir, "01_aligned", f"V-{trip_name}_aligned.csv")
    if not os.path.exists(s_path):
        raise FileNotFoundError(f"Filtered file not found for trip '{trip_name}': {s_path}")
    if not os.path.exists(v_path):
        raise FileNotFoundError(f"Aligned V-file not found for trip '{trip_name}': {v_path}")

    s_df = pd.read_csv(s_path)
    v_df = pd.read_csv(v_path)
    n = min(len(s_df), len(v_df))
    s_df = s_df.iloc[:n].reset_index(drop=True)
    v_df = v_df.iloc[:n].reset_index(drop=True)

    v_speed_col = find_col(v_df.columns, ["velocity"])
    v_long_accel_col = find_col(v_df.columns, ["longitudinal", "acceleration"])
    v_yawrate_col = find_col(v_df.columns, ["yaw", "rate"])
    missing = [name for name, col in [
        ("speed", v_speed_col), ("longitudinal_accel", v_long_accel_col), ("yaw_rate", v_yawrate_col),
    ] if col is None]
    if missing:
        raise ValueError(f"[{trip_name}] Could not auto-detect columns: {missing}")

    speed = v_df[v_speed_col].values.astype(float)
    long_accel_g = v_df[v_long_accel_col].values.astype(float)
    yaw_rate = v_df[v_yawrate_col].values.astype(float)

    is_idle = speed < idle_speed_kmh
    is_harsh_brake = long_accel_g < -harsh_brake_g
    is_sharp_turn = np.abs(yaw_rate) > sharp_turn_degs

    s_df = s_df.copy()
    s_df["is_idle"] = is_idle
    s_df["is_harsh_brake"] = is_harsh_brake
    s_df["is_sharp_turn"] = is_sharp_turn

    if verbose:
        print(f"[{trip_name}] Idling: {is_idle.sum()}/{n} ({100*is_idle.mean():.1f}%) "
              f"[speed < {idle_speed_kmh} km/h]")
        print(f"[{trip_name}] Harsh braking: {is_harsh_brake.sum()}/{n} ({100*is_harsh_brake.mean():.2f}%) "
              f"[long_accel < -{harsh_brake_g}g]")
        print(f"[{trip_name}] Sharp turns: {is_sharp_turn.sum()}/{n} ({100*is_sharp_turn.mean():.1f}%) "
              f"[|yaw_rate| > {sharp_turn_degs} deg/s]")

    out_path = os.path.join(EVENTS_DIR, f"S-{trip_name}_events.csv")
    s_df.to_csv(out_path, index=False)
    if verbose:
        print(f"[{trip_name}] Saved: {out_path}")

    return {
        "s_events": s_df,
        "idle_pct": float(is_idle.mean()),
        "harsh_brake_pct": float(is_harsh_brake.mean()),
        "sharp_turn_pct": float(is_sharp_turn.mean()),
        "out_path": out_path,
    }
