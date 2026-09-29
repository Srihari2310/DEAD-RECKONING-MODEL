"""Numerical sanity check for one converted phone S-file."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    args = parser.parse_args()

    # New converter output is UTF-8; legacy IO-VNBD files are commonly
    # cp1252/latin-1.  Try the lossless encoding first, then the legacy one.
    try:
        df = pd.read_csv(args.csv_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(args.csv_path, encoding="latin-1")
    time_col = "TIME SINCE START (ms)"
    accel_cols = [
        "ACCELEROMETER X (m/s²)",
        "ACCELEROMETER Y (m/s²)",
        "ACCELEROMETER Z (m/s²)",
    ]
    gyro_cols = [
        "GYROSCOPE Yaw (rad/s)",
        "GYROSCOPE Pitch (rad/s)",
        "GYROSCOPE Roll (rad/s)",
    ]
    gps_cols = [
        "GPS SPEED (Kmh)",
        "GPS LATITUDE (degrees)",
        "GPS LONGITUDE (degrees)",
    ]

    required = [time_col, *accel_cols, *gyro_cols, *gps_cols]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print(f"File: {args.csv_path}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Duration: {df[time_col].min() / 1000:.3f} to {df[time_col].max() / 1000:.3f} seconds")
    print("\nAccelerometer statistics:")
    print(df[accel_cols].describe().loc[["min", "mean", "max"]].to_string())
    print("\nGyroscope statistics:")
    print(df[gyro_cols].describe().loc[["min", "mean", "max"]].to_string())
    print("\nGPS ranges:")
    for column in gps_cols:
        values = pd.to_numeric(df[column], errors="coerce")
        print(f"  {column}: {values.min()} to {values.max()}")

    numeric = df[required].apply(pd.to_numeric, errors="coerce")
    nan_counts = numeric.isna().sum()
    print("\nNaN counts in required numeric columns:")
    print(nan_counts[nan_counts > 0].to_string() if nan_counts.any() else "  none")
    print("\nResult: PASS" if not nan_counts.any() else "\nResult: REVIEW NaNs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
