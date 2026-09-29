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

try:
    from . import common
except ImportError:  # direct execution: python preprocessing/run_pipeline.py ...
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from preprocessing import common

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
    # Keep the CLI pipeline on the canonical implementation. This preserves
    # the local-frame target rotation and metadata written by window_trip().
    return common.window_trip(trip_name)


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
