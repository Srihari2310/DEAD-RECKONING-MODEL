"""
preprocessing/common.py

Shared, trip-agnostic preprocessing functions for the IDR project.
Step 1: load + verify + alignment check, generalized to work on any
IO-VNBD trip (M, Vta1a, Y1, etc.) by passing a trip_name.

Usage:
    from common import load_and_verify_trip
    result = load_and_verify_trip("Vta1a")
"""

import os
import glob
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def _find_trip_file(root_dir, trip_name, prefix_letter):
    """
    Search root_dir recursively for a file matching this trip.
    Handles naming variations seen so far:
        S-M.csv       (hyphen, trip M)
        SVta1a.csv    (no hyphen, trip Vta1a)
    Tries both '<prefix>-<trip>' and '<prefix><trip>' patterns,
    case-insensitive, .csv or .xlsx.
    """
    patterns = [
        f"{prefix_letter}-{trip_name}.csv",
        f"{prefix_letter}-{trip_name}.xlsx",
        f"{prefix_letter}{trip_name}.csv",
        f"{prefix_letter}{trip_name}.xlsx",
    ]
    for pattern in patterns:
        matches = glob.glob(os.path.join(root_dir, "**", pattern), recursive=True)
        if matches:
            return matches[0]
        # case-insensitive fallback
        all_files = glob.glob(os.path.join(root_dir, "**", "*.*"), recursive=True)
        for f in all_files:
            if os.path.basename(f).lower() == pattern.lower():
                return f
    return None


def _load_csv_with_fallback(path):
    """Try utf-8, then cp1252, then latin-1. Return (df, encoding_used)."""
    encodings_to_try = ["utf-8", "cp1252", "latin-1"]
    last_err = None
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(path, encoding=enc)
            return df, enc
        except UnicodeDecodeError as e:
            last_err = e
            continue
    raise UnicodeDecodeError(
        f"Could not decode {path} with any of {encodings_to_try}: {last_err}"
    )


# ---------------------------------------------------------------------------
# Column detection (bug-fixed: full-token match, not substring)
# ---------------------------------------------------------------------------

def find_col(columns, keywords):
    """
    Find the column whose name contains ALL given keywords as whole tokens.
    Fixes the bug where 'y' matched inside 'gravity' — keywords are matched
    against tokens split on spaces, parentheses, and common punctuation.
    """
    def tokenize(name):
        cleaned = name.replace("(", " ").replace(")", " ").replace(",", " ")
        return [t.strip().lower() for t in cleaned.split() if t.strip()]

    keywords_lower = [k.lower() for k in keywords]
    for col in columns:
        tokens = tokenize(col)
        if all(kw in tokens for kw in keywords_lower):
            return col
    return None


# ---------------------------------------------------------------------------
# Alignment / lag detection
# ---------------------------------------------------------------------------

def detect_lag(s_df, v_df, s_lat_col, v_lat_col, v_speed_col, max_lag=50):
    """
    Cross-correlate latitude sequences between S- and V- files to find
    a fixed row-shift (lag). Also checks correlation between the naive
    (unshifted) GPS offset and vehicle speed, as a sanity check to
    distinguish a true fixed time-lag from random misalignment.

    IMPORTANT: phone GPS chips (S-file) commonly refresh at ~1Hz even
    though the IMU log is 10Hz, producing repeated/step-like lat values
    across consecutive rows. Raw cross-correlation on step-like data is
    unreliable (can lock onto quantization plateaus and report bogus
    large lags with spuriously perfect correlation). To guard against
    this, we linearly interpolate away repeated/stale GPS values in the
    S-file before correlating, so both sequences vary smoothly row-to-row.

    Returns dict with: best_lag_rows, best_lag_seconds (@10Hz),
    naive_offset_speed_corr.
    """
    n = min(len(s_df), len(v_df))
    s_lat_raw = s_df[s_lat_col].values[:n].astype(float)
    v_lat = v_df[v_lat_col].values[:n].astype(float)

    # De-stale the S-file GPS: mark repeated-consecutive values as NaN
    # (keep only the first row of each run), then linearly interpolate.
    s_lat_dedup = s_lat_raw.copy()
    repeat_mask = np.zeros(n, dtype=bool)
    repeat_mask[1:] = s_lat_raw[1:] == s_lat_raw[:-1]
    s_lat_dedup[repeat_mask] = np.nan
    s_series = pd.Series(s_lat_dedup)
    s_interp = s_series.interpolate(method="linear", limit_direction="both").values

    # Use the DERIVATIVE (row-to-row change) rather than raw latitude.
    # Raw latitude over a long drive is close to monotonic, so correlating
    # raw values is direction-invariant to shift and gives spuriously high
    # correlation (~1.0) at almost any lag. The derivative has real peaks/
    # troughs from turns, stops, and direction changes, which pins down a
    # genuine best-alignment lag instead of matching on the overall trend.
    s_diff = np.diff(s_interp, prepend=s_interp[0])
    v_diff = np.diff(v_lat, prepend=v_lat[0])

    s_norm = s_diff - np.nanmean(s_diff)
    v_norm = v_diff - np.nanmean(v_diff)

    best_lag = 0
    best_corr = -np.inf
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a = s_norm[-lag:]
            b = v_norm[: len(a)]
        else:
            a = s_norm[: n - lag]
            b = v_norm[lag: lag + len(a)]
        if len(a) < 10:
            continue
        # guard against zero-variance segments
        if np.nanstd(a) == 0 or np.nanstd(b) == 0:
            continue
        corr = np.corrcoef(a, b)[0, 1]
        if corr > best_corr:
            best_corr = corr
            best_lag = lag

    # naive (unshifted) offset vs speed correlation, in meters (approx, degrees->m)
    # uses de-staled/interpolated S-latitude, same reasoning as above
    naive_offset_m = np.abs(s_interp - v_lat) * 111_000  # rough deg->m at these latitudes
    speed = v_df[v_speed_col].values[:n]
    valid = ~np.isnan(naive_offset_m) & ~np.isnan(speed)
    if valid.sum() > 10:
        offset_speed_corr = np.corrcoef(naive_offset_m[valid], speed[valid])[0, 1]
    else:
        offset_speed_corr = np.nan

    hit_boundary = abs(best_lag) == max_lag
    if hit_boundary:
        print(f"WARNING: detected lag ({best_lag}) is at the search boundary "
              f"(max_lag={max_lag}). The true optimum may lie beyond this window — "
              f"re-run with a larger max_lag before trusting this result.")

    return {
        "best_lag_rows": best_lag,
        "best_lag_seconds": best_lag / 10.0,  # assumes 10Hz
        "best_corr": best_corr,
        "naive_offset_speed_corr": offset_speed_corr,
        "hit_search_boundary": hit_boundary,
    }


def align_shift(s_df, v_df, lag_rows):
    """
    Apply a detected lag to align S- and V- dataframes.
    Positive lag_rows means S- lags V- (S- reports later),
    matching the convention used for trip M (S lags V by 23 rows).
    """
    if lag_rows > 0:
        v_aligned = v_df.iloc[lag_rows:].reset_index(drop=True)
        s_aligned = s_df.iloc[: len(s_df) - lag_rows].reset_index(drop=True)
    elif lag_rows < 0:
        s_aligned = s_df.iloc[-lag_rows:].reset_index(drop=True)
        v_aligned = v_df.iloc[: len(v_df) - (-lag_rows)].reset_index(drop=True)
    else:
        s_aligned = s_df.reset_index(drop=True)
        v_aligned = v_df.reset_index(drop=True)

    n = min(len(s_aligned), len(v_aligned))
    return s_aligned.iloc[:n].reset_index(drop=True), v_aligned.iloc[:n].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main reusable entry point
# ---------------------------------------------------------------------------

def load_and_verify_trip(trip_name, s_root="S-Dataset", v_root="V-Dataset", max_lag=200, verbose=True):
    """
    Full Step 1 for any trip: locate files, load with encoding fallback,
    clean columns, check row counts, detect and report alignment lag.

    Returns a dict:
        {
            's_df', 'v_df'            : raw loaded dataframes (whitespace-stripped columns)
            's_path', 'v_path'        : file paths found
            's_encoding', 'v_encoding': encodings used
            'row_counts_match'        : bool
            'lag_info'                : dict from detect_lag()
            's_aligned', 'v_aligned'  : dataframes after align_shift()
        }
    """
    s_path = _find_trip_file(s_root, trip_name, "S")
    v_path = _find_trip_file(v_root, trip_name, "V")

    if s_path is None:
        raise FileNotFoundError(f"Could not find S-file for trip '{trip_name}' under {s_root}")
    if v_path is None:
        raise FileNotFoundError(f"Could not find V-file for trip '{trip_name}' under {v_root}")

    s_df, s_enc = _load_csv_with_fallback(s_path)
    v_df, v_enc = _load_csv_with_fallback(v_path)

    s_df.columns = [c.strip() for c in s_df.columns]
    v_df.columns = [c.strip() for c in v_df.columns]

    if verbose:
        print(f"[{trip_name}] S-file: {s_path} (encoding={s_enc}, rows={len(s_df)}, cols={len(s_df.columns)})")
        print(f"[{trip_name}] V-file: {v_path} (encoding={v_enc}, rows={len(v_df)}, cols={len(v_df.columns)})")

    row_counts_match = len(s_df) == len(v_df)
    if verbose and not row_counts_match:
        print(f"[{trip_name}] WARNING: row counts differ ({len(s_df)} vs {len(v_df)})")

    # locate key columns for lag detection
    s_lat_col = find_col(s_df.columns, ["gps", "latitude"])
    v_lat_col = find_col(v_df.columns, ["latitude"])
    v_speed_col = find_col(v_df.columns, ["velocity"])

    if s_lat_col is None or v_lat_col is None or v_speed_col is None:
        raise ValueError(
            f"[{trip_name}] Could not auto-detect required columns. "
            f"s_lat_col={s_lat_col}, v_lat_col={v_lat_col}, v_speed_col={v_speed_col}. "
            f"S columns: {list(s_df.columns)} | V columns: {list(v_df.columns)}"
        )

    lag_info = detect_lag(s_df, v_df, s_lat_col, v_lat_col, v_speed_col, max_lag=max_lag)

    if verbose:
        print(f"[{trip_name}] Detected lag: {lag_info['best_lag_rows']} rows "
              f"({lag_info['best_lag_seconds']:.1f}s), corr={lag_info['best_corr']:.3f}")
        print(f"[{trip_name}] Naive offset-vs-speed correlation: {lag_info['naive_offset_speed_corr']:.3f}")

    s_aligned, v_aligned = align_shift(s_df, v_df, lag_info["best_lag_rows"])

    if verbose:
        print(f"[{trip_name}] Aligned row count: {len(s_aligned)} (lost {len(s_df) - len(s_aligned)} rows)")

    return {
        "s_df": s_df,
        "v_df": v_df,
        "s_path": s_path,
        "v_path": v_path,
        "s_encoding": s_enc,
        "v_encoding": v_enc,
        "row_counts_match": row_counts_match,
        "lag_info": lag_info,
        "s_aligned": s_aligned,
        "v_aligned": v_aligned,
    }


# ---------------------------------------------------------------------------
# Step 2: Calibration (static leveling + dynamic yaw alignment)
# ---------------------------------------------------------------------------

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
    s_path = os.path.join(output_dir, f"S-{trip_name}_aligned.csv")
    v_path = os.path.join(output_dir, f"V-{trip_name}_aligned.csv")

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

    out_path = os.path.join(output_dir, f"S-{trip_name}_calibrated.csv")
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


# ---------------------------------------------------------------------------
# Step 3: Noise filtering (band-pass + despike, dual-gated lateral)
# ---------------------------------------------------------------------------

def filter_trip(trip_name, output_dir="preprocessing/output", verbose=True,
                 highpass_hz=0.1, lowpass_hz=4.5, sample_rate_hz=10.0,
                 despike_threshold=10.0, lateral_yawrate_gate_degs=5.0,
                 lateral_g_gate=0.1, lateral_clip=2.0):
    """
    Step 3: noise filtering for any calibrated trip. Reads
    S-{trip}_calibrated.csv, applies a Butterworth band-pass
    (highpass_hz-lowpass_hz, default 0.1-4.5Hz -- correct for 10Hz
    sampling / 5Hz Nyquist, NOT the originally-scoped 0.1-15Hz which is
    infeasible above Nyquist) to accel_forward/lateral/vertical, with
    despiking:
      - forward/vertical: hard despike at +-despike_threshold m/s^2,
        then band-pass.
      - lateral: dual-gated despike -- only clip to +-lateral_clip when
        BOTH yaw_rate is low (<lateral_yawrate_gate_degs deg/s) AND
        indicated lateral g is low (<lateral_g_gate g), i.e. only
        suppress spikes that occur when the vehicle is NOT actually
        turning hard (real turns are preserved).
    """
    from scipy.signal import butter, filtfilt

    s_path = os.path.join(output_dir, f"S-{trip_name}_calibrated.csv")
    v_path = os.path.join(output_dir, f"V-{trip_name}_aligned.csv")
    if not os.path.exists(s_path):
        raise FileNotFoundError(f"Calibrated file not found for trip '{trip_name}': {s_path}")
    if not os.path.exists(v_path):
        raise FileNotFoundError(f"Aligned V-file not found for trip '{trip_name}': {v_path}")

    s_df = pd.read_csv(s_path)
    v_df = pd.read_csv(v_path)
    n = min(len(s_df), len(v_df))
    s_df = s_df.iloc[:n].reset_index(drop=True)
    v_df = v_df.iloc[:n].reset_index(drop=True)

    v_yawrate_col = find_col(v_df.columns, ["yaw", "rate"])
    v_lat_g_col = find_col(v_df.columns, ["lateral", "acceleration"])
    if v_yawrate_col is None or v_lat_g_col is None:
        raise ValueError(f"[{trip_name}] Could not find yaw_rate/lateral_g columns for gating.")

    yaw_rate = v_df[v_yawrate_col].values.astype(float)
    lateral_g = v_df[v_lat_g_col].values.astype(float)

    nyquist = sample_rate_hz / 2.0
    if lowpass_hz >= nyquist:
        raise ValueError(
            f"lowpass_hz={lowpass_hz} must be below Nyquist ({nyquist}Hz) for "
            f"sample_rate_hz={sample_rate_hz}."
        )
    b, a = butter(N=4, Wn=[highpass_hz / nyquist, lowpass_hz / nyquist], btype="band")

    def despike_and_filter(signal, clip_val):
        clipped = np.clip(signal, -clip_val, clip_val)
        return filtfilt(b, a, clipped)

    forward_raw = s_df["accel_forward"].values.astype(float)
    lateral_raw = s_df["accel_lateral"].values.astype(float)
    vertical_raw = s_df["accel_vertical"].values.astype(float)

    forward_filtered = despike_and_filter(forward_raw, despike_threshold)
    vertical_filtered = despike_and_filter(vertical_raw, despike_threshold)

    # dual-gated lateral despike: only clip where NOT a real turn
    safe_to_clip = (np.abs(yaw_rate) < lateral_yawrate_gate_degs) & (np.abs(lateral_g) < lateral_g_gate)
    lateral_gated = lateral_raw.copy()
    clip_mask = safe_to_clip & (np.abs(lateral_raw) > lateral_clip)
    lateral_gated[clip_mask] = np.sign(lateral_raw[clip_mask]) * lateral_clip
    lateral_filtered = filtfilt(b, a, lateral_gated)

    s_df = s_df.copy()
    s_df["accel_forward_filt"] = forward_filtered
    s_df["accel_lateral_filt"] = lateral_filtered
    s_df["accel_vertical_filt"] = vertical_filtered

    n_lateral_clipped = int(clip_mask.sum())

    if verbose:
        print(f"[{trip_name}] Band-pass: {highpass_hz}-{lowpass_hz}Hz (Nyquist={nyquist}Hz)")
        print(f"[{trip_name}] Lateral dual-gate clipped {n_lateral_clipped}/{n} rows "
              f"({100*n_lateral_clipped/n:.2f}%)")
        for name, arr in [("accel_forward_filt", forward_filtered),
                           ("accel_lateral_filt", lateral_filtered),
                           ("accel_vertical_filt", vertical_filtered)]:
            print(f"[{trip_name}] {name}: mean={np.nanmean(arr):.3f} std={np.nanstd(arr):.3f} "
                  f"min={np.nanmin(arr):.3f} max={np.nanmax(arr):.3f}")

    out_path = os.path.join(output_dir, f"S-{trip_name}_filtered.csv")
    s_df.to_csv(out_path, index=False)
    if verbose:
        print(f"[{trip_name}] Saved: {out_path}")

    return {
        "s_filtered": s_df,
        "n_lateral_clipped": n_lateral_clipped,
        "out_path": out_path,
    }


# ---------------------------------------------------------------------------
# Step 4: Event detection (idling / harsh braking / sharp turns)
# ---------------------------------------------------------------------------

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
    s_path = os.path.join(output_dir, f"S-{trip_name}_filtered.csv")
    v_path = os.path.join(output_dir, f"V-{trip_name}_aligned.csv")
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

    out_path = os.path.join(output_dir, f"S-{trip_name}_events.csv")
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


# ---------------------------------------------------------------------------
# Step 5: Displacement label derivation (dx, dy, cumulative trajectory)
# ---------------------------------------------------------------------------

def derive_labels(trip_name, output_dir="preprocessing/output", verbose=True,
                   plot_path=None):
    """
    Step 5: derive (dx, dy) displacement labels per row from V-file
    Velocity + Heading, using the confirmed compass convention
    (dx = v*sin(heading), dy = v*cos(heading)), scaled by sample period.
    Also computes cumulative (x_cum, y_cum) trajectory for visual
    validation against the trip's real route thumbnail.

    Reads V-{trip}_aligned.csv. Writes V-{trip}_labels.csv with
    dx, dy, x_cum, y_cum columns. Optionally saves a trajectory plot
    to plot_path (PNG) for visual comparison against the route thumbnail.
    """
    v_path = os.path.join(output_dir, f"V-{trip_name}_aligned.csv")
    if not os.path.exists(v_path):
        raise FileNotFoundError(f"Aligned V-file not found for trip '{trip_name}': {v_path}")

    v_df = pd.read_csv(v_path)

    v_speed_col = find_col(v_df.columns, ["velocity"])
    v_heading_col = find_col(v_df.columns, ["heading"])
    v_period_col = find_col(v_df.columns, ["sample", "period"])
    missing = [name for name, col in [
        ("speed", v_speed_col), ("heading", v_heading_col), ("sample_period", v_period_col),
    ] if col is None]
    if missing:
        raise ValueError(f"[{trip_name}] Could not auto-detect columns: {missing}")

    speed_kmh = v_df[v_speed_col].values.astype(float)
    speed_ms = speed_kmh / 3.6
    heading_rad = np.deg2rad(v_df[v_heading_col].values.astype(float))
    period_s = v_df[v_period_col].values.astype(float)
    # guard against missing/zero sample period (fall back to 0.1s @ 10Hz)
    period_s = np.where((period_s <= 0) | np.isnan(period_s), 0.1, period_s)

    dx = speed_ms * np.sin(heading_rad) * period_s
    dy = speed_ms * np.cos(heading_rad) * period_s
    x_cum = np.cumsum(dx)
    y_cum = np.cumsum(dy)

    v_df = v_df.copy()
    v_df["dx"] = dx
    v_df["dy"] = dy
    v_df["x_cum"] = x_cum
    v_df["y_cum"] = y_cum

    total_path_length = np.sum(np.sqrt(dx ** 2 + dy ** 2))
    net_displacement = np.sqrt(x_cum[-1] ** 2 + y_cum[-1] ** 2)

    if verbose:
        print(f"[{trip_name}] Total path length (sum of |dx,dy|): {total_path_length:.1f} m")
        print(f"[{trip_name}] Net displacement (start to end): {net_displacement:.1f} m")
        print(f"[{trip_name}] Trajectory bounding box: x=[{x_cum.min():.1f}, {x_cum.max():.1f}] "
              f"y=[{y_cum.min():.1f}, {y_cum.max():.1f}]")

    out_path = os.path.join(output_dir, f"V-{trip_name}_labels.csv")
    v_df.to_csv(out_path, index=False)
    if verbose:
        print(f"[{trip_name}] Saved: {out_path}")

    if plot_path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 8))
        plt.plot(x_cum, y_cum, linewidth=0.8)
        plt.scatter([x_cum[0]], [y_cum[0]], color="green", label="start", zorder=5)
        plt.scatter([x_cum[-1]], [y_cum[-1]], color="red", label="end", zorder=5)
        plt.axis("equal")
        plt.xlabel("x (m, east+)")
        plt.ylabel("y (m, north+)")
        plt.title(f"Trip {trip_name} — reconstructed trajectory from V-file labels")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(plot_path, dpi=120, bbox_inches="tight")
        plt.close()
        if verbose:
            print(f"[{trip_name}] Saved trajectory plot: {plot_path}")

    return {
        "v_labels": v_df,
        "total_path_length_m": total_path_length,
        "net_displacement_m": net_displacement,
        "out_path": out_path,
    }