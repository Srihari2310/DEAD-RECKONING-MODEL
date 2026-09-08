"""
preprocessing/filtering.py

Step 3: noise filtering (band-pass + dual-gated despike) for any trip.
"""

import os
import numpy as np
import pandas as pd

from .utils import find_col

FILTERED_DIR = os.path.join("preprocessing/output", "03_filtered")
os.makedirs(FILTERED_DIR, exist_ok=True)


def filter_trip(trip_name, output_dir="preprocessing/output", verbose=True,
                 highpass_hz=0.1, lowpass_hz=4.5, sample_rate_hz=10.0,
                 despike_threshold=10.0, lateral_yawrate_gate_degs=5.0,
                 lateral_g_gate=0.1, lateral_clip=2.0):
    """
    Step 3: noise filtering for any calibrated trip. Reads
    S-{trip}_calibrated.csv, applies a Butterworth band-pass
    (highpass_hz-lowpass_hz, default 0.1-4.5Hz -- correct for 10Hz
    sampling / 5Hz Nyquist, NOT the originally-scoped 0.1-15Hz which is
    infeasible above Nyquist) to accel_forward/lateral/vertical, with
    despiking:
      - forward/vertical: hard despike at +-despike_threshold m/s^2,
        then band-pass.
      - lateral: dual-gated despike -- only clip to +-lateral_clip when
        BOTH yaw_rate is low (<lateral_yawrate_gate_degs deg/s) AND
        indicated lateral g is low (<lateral_g_gate g), i.e. only
        suppress spikes that occur when the vehicle is NOT actually
        turning hard (real turns are preserved).
    """
    from scipy.signal import butter, filtfilt

    s_path = os.path.join("preprocessing/output/02_calibrated", f"S-{trip_name}_calibrated.csv")
    v_path = os.path.join(output_dir, "01_aligned", f"V-{trip_name}_aligned.csv")
    if not os.path.exists(s_path):
        raise FileNotFoundError(f"Calibrated file not found for trip '{trip_name}': {s_path}")
    if not os.path.exists(v_path):
        raise FileNotFoundError(f"Aligned V-file not found for trip '{trip_name}': {v_path}")

    s_df = pd.read_csv(s_path)
    v_df = pd.read_csv(v_path)
    n = min(len(s_df), len(v_df))
    s_df = s_df.iloc[:n].reset_index(drop=True)
    v_df = v_df.iloc[:n].reset_index(drop=True)

    v_yawrate_col = find_col(v_df.columns, ["yaw", "rate"])
    v_lat_g_col = find_col(v_df.columns, ["lateral", "acceleration"])
    if v_yawrate_col is None or v_lat_g_col is None:
        raise ValueError(f"[{trip_name}] Could not find yaw_rate/lateral_g columns for gating.")

    yaw_rate = v_df[v_yawrate_col].values.astype(float)
    lateral_g = v_df[v_lat_g_col].values.astype(float)

    nyquist = sample_rate_hz / 2.0
    if lowpass_hz >= nyquist:
        raise ValueError(
            f"lowpass_hz={lowpass_hz} must be below Nyquist ({nyquist}Hz) for "
            f"sample_rate_hz={sample_rate_hz}."
        )
    b, a = butter(N=4, Wn=[highpass_hz / nyquist, lowpass_hz / nyquist], btype="band")

    def despike_and_filter(signal, clip_val):
        clipped = np.clip(signal, -clip_val, clip_val)
        return filtfilt(b, a, clipped)

    forward_raw = s_df["accel_forward"].values.astype(float)
    lateral_raw = s_df["accel_lateral"].values.astype(float)
    vertical_raw = s_df["accel_vertical"].values.astype(float)

    forward_filtered = despike_and_filter(forward_raw, despike_threshold)
    vertical_filtered = despike_and_filter(vertical_raw, despike_threshold)

    # dual-gated lateral despike: only clip where NOT a real turn
    safe_to_clip = (np.abs(yaw_rate) < lateral_yawrate_gate_degs) & (np.abs(lateral_g) < lateral_g_gate)
    lateral_gated = lateral_raw.copy()
    clip_mask = safe_to_clip & (np.abs(lateral_raw) > lateral_clip)
    lateral_gated[clip_mask] = np.sign(lateral_raw[clip_mask]) * lateral_clip
    lateral_filtered = filtfilt(b, a, lateral_gated)

    s_df = s_df.copy()
    s_df["accel_forward_filt"] = forward_filtered
    s_df["accel_lateral_filt"] = lateral_filtered
    s_df["accel_vertical_filt"] = vertical_filtered

    n_lateral_clipped = int(clip_mask.sum())

    if verbose:
        print(f"[{trip_name}] Band-pass: {highpass_hz}-{lowpass_hz}Hz (Nyquist={nyquist}Hz)")
        print(f"[{trip_name}] Lateral dual-gate clipped {n_lateral_clipped}/{n} rows "
              f"({100*n_lateral_clipped/n:.2f}%)")
        for name, arr in [("accel_forward_filt", forward_filtered),
                           ("accel_lateral_filt", lateral_filtered),
                           ("accel_vertical_filt", vertical_filtered)]:
            print(f"[{trip_name}] {name}: mean={np.nanmean(arr):.3f} std={np.nanstd(arr):.3f} "
                  f"min={np.nanmin(arr):.3f} max={np.nanmax(arr):.3f}")

    out_path = os.path.join(FILTERED_DIR, f"S-{trip_name}_filtered.csv")
    s_df.to_csv(out_path, index=False)
    if verbose:
        print(f"[{trip_name}] Saved: {out_path}")

    return {
        "s_filtered": s_df,
        "n_lateral_clipped": n_lateral_clipped,
        "out_path": out_path,
    }
