"""
Step 3: Noise filtering on calibrated IMU signals (trip M).
Reads preprocessing/output/S-M_calibrated.csv
Writes preprocessing/output/S-M_filtered.csv

Pipeline per channel (accel_forward, accel_lateral, accel_vertical):
  1. Despike: clip samples beyond a physically plausible threshold.
  2. Band-pass via Butterworth (zero-phase, filtfilt).
     - Sample rate is 10Hz -> Nyquist = 5Hz, so low-pass cutoff
       capped at 4.5Hz instead of the original 15Hz spec.
"""

import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt

FS = 10.0                # sample rate (Hz)
HP_CUTOFF = 0.1           # high-pass cutoff (Hz) - removes slow drift
LP_CUTOFF = 4.5           # low-pass cutoff (Hz) - capped below Nyquist (5Hz)
DESPIKE_THRESH = 10.0     # m/s^2, clip beyond this before filtering
FILTER_ORDER = 4

CHANNELS = ["accel_forward", "accel_lateral", "accel_vertical"]

IN_PATH = "preprocessing/output/S-M_calibrated.csv"
OUT_PATH = "preprocessing/output/S-M_filtered.csv"


def despike(series, thresh):
    return series.clip(lower=-thresh, upper=thresh)


def bandpass_filtfilt(series, fs, hp_cutoff, lp_cutoff, order):
    nyq = fs / 2.0
    low = hp_cutoff / nyq
    high = lp_cutoff / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, series.values)


def main():
    df = pd.read_csv(IN_PATH)

    for ch in CHANNELS:
        if ch not in df.columns:
            raise KeyError(f"Expected column '{ch}' not found. Columns present: {list(df.columns)}")

        despiked = despike(df[ch], DESPIKE_THRESH)
        filtered = bandpass_filtfilt(despiked, FS, HP_CUTOFF, LP_CUTOFF, FILTER_ORDER)
        df[f"{ch}_filtered"] = filtered

        print(f"--- {ch} ---")
        print(f"  before: min={df[ch].min():.2f} max={df[ch].max():.2f} std={df[ch].std():.2f}")
        print(f"  after : min={filtered.min():.2f} max={filtered.max():.2f} std={filtered.std():.2f}")

    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
