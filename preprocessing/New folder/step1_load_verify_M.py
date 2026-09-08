"""
Step 1: Load, verify, and check alignment for trip M (S-M / V-M) — IO-VNBD dataset
Run from project_root/, or adjust BASE_DIR below.
"""

import os
import glob
import pandas as pd
import numpy as np

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
S_DIR = os.path.join(BASE_DIR, "S-Dataset")
V_DIR = os.path.join(BASE_DIR, "V-Dataset")

EXPECTED_ROWS = 105974
EXPECTED_S_COLS = 24
EXPECTED_V_COLS = 29


def find_trip_file(root_dir, prefix):
    patterns = [
        os.path.join(root_dir, "**", f"{prefix}*.csv"),
        os.path.join(root_dir, "**", f"{prefix}*.xlsx"),
    ]
    matches = []
    for pattern in patterns:
        matches.extend(glob.glob(pattern, recursive=True))
    if not matches:
        raise FileNotFoundError(f"No file found matching '{prefix}*' under {root_dir}.")
    if len(matches) > 1:
        print(f"WARNING: multiple matches for '{prefix}*', using first:")
        for m in matches:
            print("   ", m)
    return matches[0]


def load_file(path):
    if path.lower().endswith(".csv"):
        for encoding in ["utf-8", "cp1252", "latin-1"]:
            try:
                return pd.read_csv(path, encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise UnicodeDecodeError(f"Could not decode {path} with utf-8, cp1252, or latin-1")
    elif path.lower().endswith(".xlsx"):
        return pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")


def find_col(df, keywords):
    """
    Match all keywords[:-1] as substrings, and keywords[-1] as a WHOLE WORD/token
    (fixes the 'gravity' containing 'y' bug — axis letter must be standalone).
    """
    for col in df.columns:
        col_lower = col.lower()
        tokens = col_lower.replace("(", " ").replace(")", " ").split()
        if all(kw.lower() in col_lower for kw in keywords[:-1]) and keywords[-1].lower() in tokens:
            return col
    return None

def align_shift(s_df, v_df, shift):
    if shift > 0:
        v_a = v_df.iloc[shift:].reset_index(drop=True)
        s_a = s_df.iloc[: len(v_a)].reset_index(drop=True)
    elif shift < 0:
        s_a = s_df.iloc[-shift:].reset_index(drop=True)
        v_a = v_df.iloc[: len(s_a)].reset_index(drop=True)
    else:
        s_a, v_a = s_df.reset_index(drop=True), v_df.reset_index(drop=True)
    n = min(len(s_a), len(v_a))
    return s_a.iloc[:n].reset_index(drop=True), v_a.iloc[:n].reset_index(drop=True)


def haversine_m(lat1, lon1, lat2, lon2):
    """Approximate flat-earth distance in meters — fine for small offsets."""
    dy = (lat1 - lat2) * 111000
    dx = (lon1 - lon2) * 111000 * np.cos(np.radians(lat1))
    return np.sqrt(dx**2 + dy**2)


def main():
    print("=" * 70)
    print("STEP 1: LOAD + VERIFY + ALIGNMENT CHECK — TRIP M")
    print("=" * 70)

    # --- Locate + load ---
    s_path = find_trip_file(S_DIR, "S-M")
    v_path = find_trip_file(V_DIR, "V-M")
    print(f"\nS-M file: {s_path}")
    print(f"V-M file: {v_path}")

    s_df = load_file(s_path)
    v_df = load_file(v_path)

    # Strip whitespace from column names to avoid " GPS LATITUDE" vs "GPS LATITUDE" issues
    s_df.columns = [c.strip() for c in s_df.columns]
    v_df.columns = [c.strip() for c in v_df.columns]

    # --- Shape check ---
    print("\n--- SHAPE CHECK ---")
    print(f"S-M shape: {s_df.shape}  (expected: {EXPECTED_ROWS} rows, {EXPECTED_S_COLS} cols)")
    print(f"V-M shape: {v_df.shape}  (expected: {EXPECTED_ROWS} rows, {EXPECTED_V_COLS} cols)")
    if s_df.shape[0] != v_df.shape[0]:
        print("!! ROW COUNT MISMATCH — investigate before proceeding.")
    else:
        print("OK: row counts match.")

    # --- Auto-detect key columns (now with fixed axis matching) ---
    s_lat_col = find_col(s_df, ["gps", "latitude"])
    s_lon_col = find_col(s_df, ["gps", "longitude"])
    v_lat_col = find_col(v_df, ["latitude"])
    v_lon_col = find_col(v_df, ["longitude"])
    v_speed_col = find_col(v_df, ["velocity"])

    s_accel_cols = {
        "x": find_col(s_df, ["accelerometer", "x"]),
        "y": find_col(s_df, ["accelerometer", "y"]),
        "z": find_col(s_df, ["accelerometer", "z"]),
    }
    s_gravity_cols = {
        "x": find_col(s_df, ["gravity", "x"]),
        "y": find_col(s_df, ["gravity", "y"]),
        "z": find_col(s_df, ["gravity", "z"]),
    }

    print("\n--- AUTO-DETECTED COLUMNS ---")
    print(f"S-M lat/lon: {s_lat_col} / {s_lon_col}")
    print(f"V-M lat/lon: {v_lat_col} / {v_lon_col}")
    print(f"V-M speed:   {v_speed_col}")
    print(f"S-M accel x/y/z:   {s_accel_cols}")
    print(f"S-M gravity x/y/z: {s_gravity_cols}")

    if None in [s_lat_col, s_lon_col, v_lat_col, v_lon_col, v_speed_col]:
        print("\n!! Could not auto-detect one or more required columns. Stopping.")
        print("   Set them manually near the top of main() and rerun.")
        return

    # --- Dense alignment check: does offset correlate with speed? ---
    print("\n--- ALIGNMENT CHECK: does GPS offset scale with vehicle speed? ---")
    n = min(len(s_df), len(v_df))
    idx = np.arange(0, n, 200)  # dense sample every 200 rows (~20s at 10Hz)

    offsets, speeds = [], []
    for i in idx:
        s_lat, s_lon = s_df.loc[i, s_lat_col], s_df.loc[i, s_lon_col]
        v_lat, v_lon = v_df.loc[i, v_lat_col], v_df.loc[i, v_lon_col]
        if pd.isna(s_lat) or pd.isna(v_lat):
            continue
        dist_m = haversine_m(s_lat, s_lon, v_lat, v_lon)
        offsets.append(dist_m)
        speeds.append(v_df.loc[i, v_speed_col])

    offsets = np.array(offsets)
    speeds = np.array(speeds)

    print(f"Samples checked: {len(offsets)}")
    print(f"Offset stats: mean={offsets.mean():.1f}m  max={offsets.max():.1f}m  std={offsets.std():.1f}m")

    corr = np.corrcoef(offsets, speeds)[0, 1]
    print(f"Correlation(offset, speed) = {corr:.3f}")

    if corr > 0.5:
        print(">> Offset scales with speed. This looks like a small fixed TIME LAG")
        print("   (e.g. S-M lags V-M by ~0.2-0.5s), NOT random misalignment.")
        print("   Row-index alignment is usable, but consider estimating/correcting the lag")
        print("   via cross-correlation before final windowing.")
    else:
        print(">> Offset does NOT correlate with speed — this suggests real row misalignment")
        print("   (dropped/duplicated rows, not just a timing lag). Needs deeper investigation")
        print("   before trusting row-index-based label alignment.")

    # --- Estimate best time-shift via cross-correlation on lat sequences ---
    print("\n--- ESTIMATING OPTIMAL ROW SHIFT (cross-correlation on latitude) ---")
    s_lat_series = s_df[s_lat_col].to_numpy()
    v_lat_series = v_df[v_lat_col].to_numpy()

    max_shift = 30  # search +/- 30 rows (=3s at 10Hz)
    best_shift, best_err = 0, np.inf
    for shift in range(-max_shift, max_shift + 1):
        if shift >= 0:
            a = s_lat_series[shift:]
            b = v_lat_series[: len(a)]
        else:
            b = v_lat_series[-shift:]
            a = s_lat_series[: len(b)]
        m = min(len(a), len(b))
        if m < 1000:
            continue
        err = np.nanmean((a[:m] - b[:m]) ** 2)
        if err < best_err:
            best_err, best_shift = err, shift

    print(f"Best-fit row shift: {best_shift} rows ({best_shift/10:.2f}s at 10Hz)")
    if best_shift == 0:
        print("OK: zero-shift is optimal — S-M and V-M are already row-aligned.")
    else:
        print(f"!! Suggests S-M should be shifted by {best_shift} rows to best match V-M.")
        print("   Apply this shift before deriving displacement labels in Step 2.")

    # --- Linear acceleration sanity check (now with corrected gravity mapping) ---
    if None not in s_accel_cols.values() and None not in s_gravity_cols.values():
        print("\n--- LINEAR ACCELERATION CHECK (ACCEL - GRAVITY, corrected axis mapping) ---")
        for axis in ["x", "y", "z"]:
            lin = s_df[s_accel_cols[axis]] - s_df[s_gravity_cols[axis]]
            print(f"  {axis}: mean={lin.mean():.4f}  std={lin.std():.4f}  min={lin.min():.4f}  max={lin.max():.4f}")
    else:
        print("\nSkipping linear acceleration check — accel/gravity columns not fully detected.")

    print("\n" + "=" * 70)
    print("STEP 1 COMPLETE")
    print("=" * 70)
    s_df, v_df = align_shift(s_df, v_df, best_shift)
    print(f"\nApplied shift={best_shift}. New aligned shapes: S-M={s_df.shape}, V-M={v_df.shape}")


if __name__ == "__main__":
    main()