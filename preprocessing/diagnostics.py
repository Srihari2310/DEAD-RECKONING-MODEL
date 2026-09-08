"""
preprocessing/diagnostics.py

Data-quality diagnostics, usable on any trip before trusting its
pipeline output (frozen GPS, jumps, etc.).
"""

from .utils import _find_trip_file, _load_csv_with_fallback, find_col


def check_gps_freezing(trip_name, v_root="V-Dataset", tail_n=20):
    """
    Diagnostic: checks V-file for frozen/duplicate consecutive GPS lat/lon
    rows, which corrupts lag detection and trajectory reconstruction.
    Works for any trip name.
    """
    v_path = _find_trip_file(v_root, trip_name, "V")
    v_df, _ = _load_csv_with_fallback(v_path)
    v_df.columns = [c.strip() for c in v_df.columns]

    lat_col = find_col(v_df.columns, ["latitude"])
    lon_col = find_col(v_df.columns, ["longitude"])

    frozen = (v_df[lat_col].diff() == 0) & (v_df[lon_col].diff() == 0)
    print(f"[{trip_name}] Frozen/duplicate GPS rows: {frozen.sum()} / {len(v_df)}")
    print(f"[{trip_name}] Last {tail_n} lat/lon values:")
    print(v_df[[lat_col, lon_col]].tail(tail_n))

    return frozen
