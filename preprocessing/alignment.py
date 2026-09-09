"""
preprocessing/alignment.py

Step 1: load, verify, and align S-/V- files for any trip.
Output folder: preprocessing/output/01_aligned/
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

from .utils import (
    IOVNBD_ROOT,
    _find_trip_file,
    _load_csv_with_fallback,
    find_col,
    find_iovnbd_trip_files,
)

ALIGNED_DIR = Path("preprocessing/output/01_aligned")
MIN_ROWS_FOR_RELIABLE_LAG = 300
MIN_OVERLAP = 30


# ---------------------------------------------------------------------------
# Alignment / lag detection
# ---------------------------------------------------------------------------

def detect_lag(s_df, v_df, s_lat_col, v_lat_col, v_speed_col, max_lag=50):
    """
    Cross-correlate latitude sequences between S- and V- files to find
    a fixed row-shift (lag). Also checks correlation between the naive
    (unshifted) GPS offset and vehicle speed, as a sanity check to
    distinguish a true fixed time-lag from random misalignment.

    Uses derivative-based correlation (not raw values) to avoid locking
    onto quantization plateaus / monotonic-trend artifacts.
    """
    n = min(len(s_df), len(v_df))
    if n < MIN_ROWS_FOR_RELIABLE_LAG:
        raise ValueError(
            f"Trip too short ({n} rows) for reliable lag detection "
            f"(minimum {MIN_ROWS_FOR_RELIABLE_LAG})"
        )

    s_lat_raw = s_df[s_lat_col].values[:n].astype(float)
    v_lat = v_df[v_lat_col].values[:n].astype(float)

    s_lat_dedup = s_lat_raw.copy()
    repeat_mask = np.zeros(n, dtype=bool)
    repeat_mask[1:] = s_lat_raw[1:] == s_lat_raw[:-1]
    s_lat_dedup[repeat_mask] = np.nan
    s_series = pd.Series(s_lat_dedup)
    s_interp = s_series.interpolate(method="linear", limit_direction="both").values

    s_diff = np.diff(s_interp, prepend=s_interp[0])
    v_diff = np.diff(v_lat, prepend=v_lat[0])

    s_norm = s_diff - np.nanmean(s_diff)
    v_norm = v_diff - np.nanmean(v_diff)

    effective_max_lag = min(max_lag, n - 1 - MIN_OVERLAP)
    if effective_max_lag < 1:
        raise ValueError(f"Trip too short ({n} rows) for reliable lag detection")

    best_lag = 0
    best_corr = -np.inf
    for lag in range(-effective_max_lag, effective_max_lag + 1):
        if lag < 0:
            a = s_norm[-lag:]
            b = v_norm[: len(a)]
        else:
            a = s_norm[: n - lag]
            b = v_norm[lag: lag + len(a)]
        if len(a) < 10:
            continue
        if np.nanstd(a) == 0 or np.nanstd(b) == 0:
            continue
        corr = np.corrcoef(a, b)[0, 1]
        if corr > best_corr:
            best_corr = corr
            best_lag = lag

    naive_offset_m = np.abs(s_interp - v_lat) * 111_000
    speed = v_df[v_speed_col].values[:n]
    valid = ~np.isnan(naive_offset_m) & ~np.isnan(speed)
    if valid.sum() > 10:
        offset_speed_corr = np.corrcoef(naive_offset_m[valid], speed[valid])[0, 1]
    else:
        offset_speed_corr = np.nan

    hit_boundary = abs(best_lag) == effective_max_lag
    if hit_boundary:
        print(f"WARNING: detected lag ({best_lag}) is at the search boundary "
              f"(max_lag={max_lag}). The true optimum may lie beyond this window — "
              f"re-run with a larger max_lag before trusting this result.")

    return {
        "best_lag_rows": best_lag,
        "best_lag_seconds": best_lag / 10.0,
        "best_corr": best_corr,
        "naive_offset_speed_corr": offset_speed_corr,
        "hit_search_boundary": hit_boundary,
    }


def align_shift(s_df, v_df, lag_rows):
    """
    Apply a detected lag to align S- and V- dataframes.
    Positive lag_rows means S- lags V- (S- reports later).
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
    Full Step 1 for any trip. Saves aligned CSVs to
    preprocessing/output/01_aligned/.
    """
    s_path = _find_trip_file(s_root, trip_name, "S")
    v_path = _find_trip_file(v_root, trip_name, "V")

    # Fall back to the original IO-VNBD layout, where each trip has its own
    # folder and filenames may use zero-padding or lowercase "v".
    if s_path is None or v_path is None:
        io_s_path, io_v_path = find_iovnbd_trip_files(trip_name, IOVNBD_ROOT)
        s_path = s_path or io_s_path
        v_path = v_path or io_v_path

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

    # --- Trim trailing frozen/duplicate GPS rows ---
    v_lat_col_check = find_col(v_df.columns, ["latitude"])
    v_lon_col_check = find_col(v_df.columns, ["longitude"])
    if v_lat_col_check and v_lon_col_check:
        frozen = (v_df[v_lat_col_check].diff() == 0) & (v_df[v_lon_col_check].diff() == 0)
        if frozen.any():
            last_good = frozen[~frozen].index.max() if (~frozen).any() else len(v_df) - 1
            trailing_frozen = len(v_df) - 1 - last_good
            if trailing_frozen > 50:
                print(f"[{trip_name}] Trimming {trailing_frozen} trailing frozen GPS rows "
                      f"from V-file before alignment.")
                v_df = v_df.iloc[: last_good + 1].reset_index(drop=True)
                s_df = s_df.iloc[: len(v_df)].reset_index(drop=True)

    # --- Trim after any single-row GPS teleport (corrupt/dropped fix) ---
    v_lat_col_check2 = find_col(v_df.columns, ["latitude"])
    v_lon_col_check2 = find_col(v_df.columns, ["longitude"])
    if v_lat_col_check2 and v_lon_col_check2:
        dlat = v_df[v_lat_col_check2].diff() * 111000
        dlon = v_df[v_lat_col_check2].diff() * 0  # placeholder, overwritten below
        dlon = v_df[v_lon_col_check2].diff() * 111000 * np.cos(np.radians(v_df[v_lat_col_check2]))
        jump_dist = np.sqrt(dlat**2 + dlon**2)
        JUMP_THRESHOLD_M = 200
        bad_jumps = jump_dist[jump_dist > JUMP_THRESHOLD_M]
        if len(bad_jumps) > 0:
            first_bad_idx = bad_jumps.index[0]
            print(f"[{trip_name}] Trimming at row {first_bad_idx}: GPS jump of "
                  f"{jump_dist[first_bad_idx]:.0f}m detected (threshold={JUMP_THRESHOLD_M}m).")
            v_df = v_df.iloc[:first_bad_idx].reset_index(drop=True)
            s_df = s_df.iloc[:len(v_df)].reset_index(drop=True)

    row_counts_match = len(s_df) == len(v_df)
    if verbose and not row_counts_match:
        print(f"[{trip_name}] WARNING: row counts differ ({len(s_df)} vs {len(v_df)})")

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

    ALIGNED_DIR.mkdir(parents=True, exist_ok=True)
    s_aligned.to_csv(ALIGNED_DIR / f"S-{trip_name}_aligned.csv", index=False)
    v_aligned.to_csv(ALIGNED_DIR / f"V-{trip_name}_aligned.csv", index=False)
    if verbose:
        print(f"[{trip_name}] Saved: {ALIGNED_DIR / f'S-{trip_name}_aligned.csv'}")
        print(f"[{trip_name}] Saved: {ALIGNED_DIR / f'V-{trip_name}_aligned.csv'}")

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
