"""
preprocessing/step1_run.py

Thin runner for Step 1 (load + verify + alignment check) on any trip.
Saves aligned S/V CSVs to preprocessing/output/ for downstream steps.

Usage:
    python preprocessing/step1_run.py Vta1a
    python preprocessing/step1_run.py M
"""

import sys
import os
from common import load_and_verify_trip

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python step1_run.py <trip_name>")
        sys.exit(1)

    trip_name = sys.argv[1]
    result = load_and_verify_trip(trip_name)

    print("\n--- Summary ---")
    print(f"Trip: {trip_name}")
    print(f"S columns ({len(result['s_df'].columns)}):", list(result["s_df"].columns))
    print(f"V columns ({len(result['v_df'].columns)}):", list(result["v_df"].columns))
    print(f"Row counts match: {result['row_counts_match']}")
    print(f"Lag: {result['lag_info']['best_lag_rows']} rows "
          f"({result['lag_info']['best_lag_seconds']:.1f}s)")
    print(f"Aligned shape: S={result['s_aligned'].shape}, V={result['v_aligned'].shape}")

    out_dir = "preprocessing/output"
    os.makedirs(out_dir, exist_ok=True)
    s_out_path = os.path.join(out_dir, f"S-{trip_name}_aligned.csv")
    v_out_path = os.path.join(out_dir, f"V-{trip_name}_aligned.csv")
    result["s_aligned"].to_csv(s_out_path, index=False)
    result["v_aligned"].to_csv(v_out_path, index=False)
    print(f"\nSaved: {s_out_path}")
    print(f"Saved: {v_out_path}")