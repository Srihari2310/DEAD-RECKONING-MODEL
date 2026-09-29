"""
run_pipeline.py — runs Steps 1-6 automatically for a single trip.

Usage: python preprocessing/run_pipeline.py <trip_name>
Example: python preprocessing/run_pipeline.py Y1

Assumes preprocessing/common.py exposes (per v9 restructure):
  load_and_verify_trip(trip_name)  -> Step 1: load, align, save S-{trip}_aligned.csv / V-{trip}_aligned.csv
  calibrate_trip(trip_name)        -> Step 2: save S-{trip}_calibrated.csv
  filter_trip(trip_name)           -> Step 3: save S-{trip}_filtered.csv
  detect_events(trip_name)         -> Step 4: save S-{trip}_events.csv
  derive_labels(trip_name)         -> Step 5: save V-{trip}_labels.csv

Step 6 (windowing) is included directly in this file (same logic as step6_run.py)
so the whole thing runs end-to-end in one command.

If your common.py function names/signatures differ slightly, edit the
STEP_FUNCS calls below to match — everything else stays the same.
"""

import sys
import traceback
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

WINDOW_SAMPLES = 10   # 1.0s at 10Hz
STRIDE_SAMPLES = 10   # 1.0s at 10Hz
CHANNELS = [
    "accel_forward", "accel_lateral", "accel_vertical",
    "gyro_yaw", "gyro_pitch", "gyro_roll",
]


def find_col(df, keyword_tokens):
    for col in df.columns:
        tokens = col.lower().replace("(", " ").replace(")", " ").replace("/", " ").split()
        if all(any(kw.lower() in t for t in tokens[:-1]) for kw in keyword_tokens[:-1]) and \
           keyword_tokens[-1].lower() in tokens:
            return col
    return None


def run_step6(trip_name):
    out_dir = Path("preprocessing/output")
    filtered_path = out_dir / f"S-{trip_name}_filtered.csv"
    labels_path = out_dir / f"V-{trip_name}_labels.csv"

    df = pd.read_csv(filtered_path)
    df.columns = [c.strip() for c in df.columns]
    resolved = {}
    for ch in CHANNELS:
        if ch in df.columns:
            resolved[ch] = ch
        else:
            match = find_col(df, ch.split("_"))
            if match is None:
                raise KeyError(f"Column for '{ch}' not found in {filtered_path}: {list(df.columns)}")
            resolved[ch] = match
    chan_df = df[[resolved[ch] for ch in CHANNELS]].copy()
    chan_df.columns = CHANNELS

    lab_df = pd.read_csv(labels_path)
    lab_df.columns = [c.strip() for c in lab_df.columns]

    n = min(len(chan_df), len(lab_df))
    X_chan = chan_df.iloc[:n][CHANNELS].to_numpy(dtype=np.float32)
    Y_dxdy = lab_df.iloc[:n][["dx", "dy"]].to_numpy(dtype=np.float32)

    starts = list(range(0, n - WINDOW_SAMPLES + 1, STRIDE_SAMPLES))
    X = np.zeros((len(starts), WINDOW_SAMPLES, len(CHANNELS)), dtype=np.float32)
    Y = np.zeros((len(starts), 2), dtype=np.float32)
    idx = np.zeros((len(starts),), dtype=np.int64)
    for i, s in enumerate(starts):
        e = s + WINDOW_SAMPLES
        X[i] = X_chan[s:e]
        Y[i] = Y_dxdy[s:e].sum(axis=0)
        idx[i] = s

    out_path = out_dir / f"{trip_name}_windows.npz"
    np.savez(out_path, X=X, Y=Y, idx=idx)

    disp_mag = np.sqrt(Y[:, 0] ** 2 + Y[:, 1] ** 2)
    print(f"[Step 6] X shape: {X.shape}, Y shape: {Y.shape}")
    print(f"[Step 6] Saved: {out_path}")
    print(f"[Step 6] Displacement magnitude -> mean={disp_mag.mean():.2f}m max={disp_mag.max():.2f}m")


def run(trip_name):
    steps = [
        ("Step 1: load/verify/align", lambda: common.load_and_verify_trip(trip_name)),
        ("Step 2: calibrate",         lambda: common.calibrate_trip(trip_name)),
        ("Step 3: filter",            lambda: common.filter_trip(trip_name)),
        ("Step 4: detect events",     lambda: common.detect_events(trip_name)),
        ("Step 5: derive labels",     lambda: common.derive_labels(trip_name)),
        ("Step 6: windowing",         lambda: run_step6(trip_name)),
    ]

    for name, fn in steps:
        print(f"\n===== {name} ({trip_name}) =====")
        try:
            fn()
        except Exception as e:
            print(f"[FAILED] {name} raised an error:\n{e}")
            traceback.print_exc()
            print(f"\nPipeline stopped at '{name}'. Fix the issue above and re-run "
                  f"(you can resume by editing the 'steps' list to skip completed steps).")
            sys.exit(1)

    print(f"\n===== Pipeline complete for trip '{trip_name}' =====")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python preprocessing/run_pipeline.py <trip_name>")
        sys.exit(1)
    run(sys.argv[1])
