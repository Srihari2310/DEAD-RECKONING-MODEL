"""
Step 6 — Windowing
Usage: python preprocessing/step6_run.py <trip_name>

Reads:
  preprocessing/output/S-{trip}_filtered.csv   (accel_forward/lateral/vertical + gyro yaw/pitch/roll)
  preprocessing/output/S-{trip}_events.csv     (idle/harsh_brake/sharp_turn flags, optional passthrough)
  preprocessing/output/V-{trip}_labels.csv     (dx, dy per row from Step 5)

Writes:
  preprocessing/output/{trip}_windows.npz
    X: (n_windows, 40, 6)  -> accel_forward, accel_lateral, accel_vertical, gyro_yaw, gyro_pitch, gyro_roll
    Y: (n_windows, 2)      -> summed dx, dy over the window
    idx: (n_windows,)      -> start row index of each window (for traceability back to source rows)

Params (fixed, consistent across trips per v8/v9 handoff):
  WINDOW_SEC = 4.0  -> 40 samples at 10Hz
  STRIDE_SEC = 1.0  -> 10 samples at 10Hz
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

WINDOW_SAMPLES = 40   # 4.0s at 10Hz
STRIDE_SAMPLES = 10   # 1.0s at 10Hz

CHANNELS = [
    "accel_forward", "accel_lateral", "accel_vertical",
    "gyro_yaw", "gyro_pitch", "gyro_roll",
]


def find_col(df, keyword_tokens):
    """Match a column by requiring the LAST keyword token to match a full token
    (not substring) — same fix as Step 1's find_col bug fix."""
    for col in df.columns:
        tokens = col.lower().replace("(", " ").replace(")", " ").replace("/", " ").split()
        if all(any(kw.lower() in t for t in tokens[:-1]) for kw in keyword_tokens[:-1]) and \
           keyword_tokens[-1].lower() in tokens:
            return col
    return None


def load_channel_frame(filtered_path):
    df = pd.read_csv(filtered_path)
    df.columns = [c.strip() for c in df.columns]

    # Try exact expected names first (Step 2/3 output should already use these),
    # fall back to fuzzy match if column naming differs.
    resolved = {}
    for ch in CHANNELS:
        if ch in df.columns:
            resolved[ch] = ch
        else:
            # crude fallback: split "accel_forward" -> ["accel","forward"]
            tokens = ch.split("_")
            match = find_col(df, tokens)
            if match is None:
                raise KeyError(f"Could not find column for channel '{ch}' in {filtered_path}. "
                                f"Available columns: {list(df.columns)}")
            resolved[ch] = match

    chan_df = df[[resolved[ch] for ch in CHANNELS]].copy()
    chan_df.columns = CHANNELS
    return chan_df


def load_labels(labels_path):
    df = pd.read_csv(labels_path)
    df.columns = [c.strip() for c in df.columns]
    if "dx" not in df.columns or "dy" not in df.columns:
        raise KeyError(f"'dx'/'dy' columns not found in {labels_path}. "
                        f"Available columns: {list(df.columns)}")
    return df[["dx", "dy"]].copy()


def build_windows(chan_df, label_df):
    n = min(len(chan_df), len(label_df))
    chan_df = chan_df.iloc[:n].reset_index(drop=True)
    label_df = label_df.iloc[:n].reset_index(drop=True)

    X_chan = chan_df[CHANNELS].to_numpy(dtype=np.float32)
    Y_dxdy = label_df[["dx", "dy"]].to_numpy(dtype=np.float32)

    starts = list(range(0, n - WINDOW_SAMPLES + 1, STRIDE_SAMPLES))
    X = np.zeros((len(starts), WINDOW_SAMPLES, len(CHANNELS)), dtype=np.float32)
    Y = np.zeros((len(starts), 2), dtype=np.float32)
    idx = np.zeros((len(starts),), dtype=np.int64)

    for i, s in enumerate(starts):
        e = s + WINDOW_SAMPLES
        X[i] = X_chan[s:e]
        Y[i] = Y_dxdy[s:e].sum(axis=0)
        idx[i] = s

    return X, Y, idx


def main(trip_name):
    out_dir = Path("preprocessing/output")
    filtered_path = out_dir / f"S-{trip_name}_filtered.csv"
    labels_path = out_dir / f"V-{trip_name}_labels.csv"

    if not filtered_path.exists():
        raise FileNotFoundError(f"Missing {filtered_path} — run Step 3 (filter_trip) first.")
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing {labels_path} — run Step 5 (derive_labels) first.")

    print(f"[Step 6] Loading channels from {filtered_path}")
    chan_df = load_channel_frame(filtered_path)

    print(f"[Step 6] Loading labels from {labels_path}")
    label_df = load_labels(labels_path)

    print(f"[Step 6] Building windows: {WINDOW_SAMPLES} samples ({WINDOW_SAMPLES/10:.1f}s), "
          f"stride {STRIDE_SAMPLES} samples ({STRIDE_SAMPLES/10:.1f}s)")
    X, Y, idx = build_windows(chan_df, label_df)

    out_path = out_dir / f"{trip_name}_windows.npz"
    np.savez(out_path, X=X, Y=Y, idx=idx)

    print(f"[Step 6] Done. X shape: {X.shape}, Y shape: {Y.shape}")
    print(f"[Step 6] Saved: {out_path}")
    print(f"[Step 6] Y stats -> dx: mean={Y[:,0].mean():.3f} std={Y[:,0].std():.3f} | "
          f"dy: mean={Y[:,1].mean():.3f} std={Y[:,1].std():.3f}")
    disp_mag = np.sqrt(Y[:, 0] ** 2 + Y[:, 1] ** 2)
    print(f"[Step 6] Displacement magnitude per window -> mean={disp_mag.mean():.2f}m "
          f"max={disp_mag.max():.2f}m")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python preprocessing/step6_run.py <trip_name>")
        sys.exit(1)
    main(sys.argv[1])
