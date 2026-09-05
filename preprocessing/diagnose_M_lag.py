"""
preprocessing/diagnose_M_lag.py

One-off diagnostic to resolve the discrepancy between the old Step 1
script's lag result for trip M (23 rows) and the new common.py result
(17 rows, corr=0.962 -- lower confidence than Vta1a's 0.998).

Checks:
1. Is M's S-file GPS stale/repeated (like Vta1a's ~1Hz refresh), or
   genuinely dense at 10Hz?
2. What does the correlation-vs-lag curve look like around both
   candidate lags (17 and 23)? Is there a clear single peak, or two
   close competing peaks (which would explain the discrepancy)?

Run from project root:
    python preprocessing/diagnose_M_lag.py
"""

import numpy as np
import pandas as pd
from common import _load_csv_with_fallback, _find_trip_file, find_col

s_path = _find_trip_file("S-Dataset", "M", "S")
v_path = _find_trip_file("V-Dataset", "M", "V")

s_df, s_enc = _load_csv_with_fallback(s_path)
v_df, v_enc = _load_csv_with_fallback(v_path)
s_df.columns = [c.strip() for c in s_df.columns]
v_df.columns = [c.strip() for c in v_df.columns]

s_lat_col = find_col(s_df.columns, ["gps", "latitude"])
v_lat_col = find_col(v_df.columns, ["latitude"])

s_lat = s_df[s_lat_col].values.astype(float)
v_lat = v_df[v_lat_col].values.astype(float)
n = min(len(s_lat), len(v_lat))
s_lat, v_lat = s_lat[:n], v_lat[:n]

# --- Check 1: staleness / update-rate of S-file GPS ---
repeat_mask = np.zeros(n, dtype=bool)
repeat_mask[1:] = s_lat[1:] == s_lat[:-1]
changes = np.where(~repeat_mask)[0]
run_lengths = np.diff(np.concatenate([[0], changes, [n]]))
print("=== S-file GPS update-rate check ===")
print(f"Unique lat values: {len(np.unique(s_lat))} / {n} rows")
print(f"Run-length stats: mean={run_lengths.mean():.2f} median={np.median(run_lengths):.1f} "
      f"max={run_lengths.max()} min={run_lengths.min()}")
print("(median run-length ~1 means dense/real 10Hz GPS; ~10 means ~1Hz stale-repeat like Vta1a)\n")

# --- Check 2: correlation curve around both candidate lags ---
s_dedup = s_lat.copy()
s_dedup[repeat_mask] = np.nan
s_interp = pd.Series(s_dedup).interpolate(limit_direction="both").values

s_diff = np.diff(s_interp, prepend=s_interp[0])
v_diff = np.diff(v_lat, prepend=v_lat[0])
s_norm = s_diff - np.nanmean(s_diff)
v_norm = v_diff - np.nanmean(v_diff)

print("=== Correlation curve, derivative-based (new method), lags -10..40 ===")
for lag in range(-10, 41):
    if lag < 0:
        a = s_norm[-lag:]
        b = v_norm[: len(a)]
    else:
        a = s_norm[: n - lag]
        b = v_norm[lag: lag + len(a)]
    if len(a) < 10 or np.nanstd(a) == 0 or np.nanstd(b) == 0:
        continue
    c = np.corrcoef(a, b)[0, 1]
    marker = "  <-- 17" if lag == 17 else ("  <-- 23 (old script's answer)" if lag == 23 else "")
    print(f"{lag:4d}  {c:.4f}{marker}")

print("\n=== Correlation curve, RAW latitude (old method), lags -10..40 ===")
s_raw_norm = s_lat - np.nanmean(s_lat)
v_raw_norm = v_lat - np.nanmean(v_lat)
for lag in range(-10, 41):
    if lag < 0:
        a = s_raw_norm[-lag:]
        b = v_raw_norm[: len(a)]
    else:
        a = s_raw_norm[: n - lag]
        b = v_raw_norm[lag: lag + len(a)]
    if len(a) < 10 or np.nanstd(a) == 0 or np.nanstd(b) == 0:
        continue
    c = np.corrcoef(a, b)[0, 1]
    marker = "  <-- 17" if lag == 17 else ("  <-- 23 (old script's answer)" if lag == 23 else "")
    print(f"{lag:4d}  {c:.4f}{marker}")
