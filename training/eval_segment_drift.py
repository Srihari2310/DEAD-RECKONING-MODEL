"""
training/eval_segment_drift.py

Realistic drift evaluation, replacing the flawed "one giant open-loop
cumsum over the whole held-out tail" approach used in train_multitrip.py.

WHY the old method was misleading:
    The old eval integrated predictions open-loop over ~35 minutes (M)
    or ~8.5 minutes (Vta1a) of continuous windows with zero correction.
    Real GNSS blackouts in this dataset are 0.1-3.0 seconds (confirmed
    on trip M). The SIH benchmark's own example targets imply blackouts
    of tens of seconds, not tens of minutes ("<5m drift over 50m/<1min",
    "<100m drift over 1km at 60km/h"). Dead reckoning ALWAYS compounds
    over time in any system -- that's why the full architecture has
    GNSS+map-matching fusion to correct it periodically (stages 7-9).
    Measuring pure open-loop drift over 35 minutes makes any model look
    catastrophic regardless of quality, and isn't the number the
    benchmark actually cares about.

WHAT this script does instead:
    Simulates many independent short "blackout" segments (default:
    10s, 30s, 60s) by sampling many start points across the held-out
    tail of each trip. For each segment: reconstruct the path via
    cumulative sum of predicted (dx,dy) ONLY within that segment
    (resetting to the true position at the segment start -- this
    mimics losing GNSS for exactly that duration, then checking drift
    at reacquisition), compare to the true displacement over the same
    segment, and compute drift % = endpoint_error / true_distance.
    Reports the MEAN and DISTRIBUTION (median, p90) of drift % across
    all simulated segments per duration, per trip -- this is the
    number that's actually comparable to the <10% benchmark.

Requires: preprocessing/output/dr_model_multitrip.pt
          preprocessing/output/dr_model_multitrip_norm.npz
          preprocessing/output/{trip}_windows.npz

Run from project root:
    python training/eval_segment_drift.py
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

OUTPUT_DIR = Path("preprocessing/output")
MODEL_PATH = OUTPUT_DIR / "dr_model_multitrip.pt"
NORM_PATH = OUTPUT_DIR / "dr_model_multitrip_norm.npz"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_FRACTION = 0.8  # must match train_multitrip.py's split point

# Windows are 1.0s each (10 samples @ 10Hz) with 1.0s stride (see run_pipeline.py).
WINDOW_SECONDS = 1.0
STRIDE_SECONDS = 1.0

# Blackout durations to simulate, in seconds.
BLACKOUT_DURATIONS_S = [10, 30, 60]
N_SEGMENTS_PER_DURATION = 200  # random start points sampled per duration, per trip


# ---------------------------------------------------------------------------
# Model definition (must match train_multitrip.py exactly)
# ---------------------------------------------------------------------------

class DRNet(nn.Module):
    def __init__(self, in_channels=6):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)


def load_model():
    data = np.load(NORM_PATH)
    mean, std = data["mean"], data["std"]
    model = DRNet(in_channels=len(mean)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model, mean, std


# ---------------------------------------------------------------------------
# Data loading -- reproduce the same held-out tail split as training
# ---------------------------------------------------------------------------

def load_holdout_tail(trip_name):
    path = OUTPUT_DIR / f"{trip_name}_windows.npz"
    data = np.load(path)
    X, Y, idx = data["X"], data["Y"], data["idx"]
    cut = int(len(X) * TRAIN_FRACTION)
    return X[cut:], Y[cut:], idx[cut:]


# ---------------------------------------------------------------------------
# Segment-based drift simulation
# ---------------------------------------------------------------------------

def predict_all(model, X, mean, std):
    X_n = (X - mean) / std
    with torch.no_grad():
        X_t = torch.from_numpy(X_n.astype(np.float32)).to(DEVICE)
        Y_pred = model(X_t).cpu().numpy()
    return Y_pred


def simulate_segments(Y_pred, Y_true, duration_s, n_segments, rng):
    """
    Windows are spaced STRIDE_SECONDS apart. A blackout of duration_s
    seconds spans roughly (duration_s / STRIDE_SECONDS) consecutive
    windows. We sample random start indices and sum predicted vs true
    displacement over that many consecutive windows, then compute
    drift % for each simulated segment.
    """
    n_windows_per_segment = max(1, round(duration_s / STRIDE_SECONDS))
    n_total = len(Y_pred)
    max_start = n_total - n_windows_per_segment
    if max_start <= 0:
        return np.array([])  # trip too short for this duration

    starts = rng.integers(0, max_start, size=min(n_segments, max_start))
    drift_pcts = []
    for s in starts:
        e = s + n_windows_per_segment
        pred_disp = Y_pred[s:e].sum(axis=0)
        true_disp = Y_true[s:e].sum(axis=0)
        true_dist = np.sqrt(np.sum(true_disp ** 2))
        if true_dist < 1e-3:
            continue  # skip near-stationary segments, drift % undefined/meaningless
        endpoint_error = np.sqrt(np.sum((pred_disp - true_disp) ** 2))
        drift_pcts.append(100 * endpoint_error / true_dist)

    return np.array(drift_pcts)


def summarize(drift_pcts, label):
    if len(drift_pcts) == 0:
        print(f"  {label}: no valid segments (trip too short or all near-stationary)")
        return
    print(f"  {label}: n={len(drift_pcts)}  "
          f"mean={drift_pcts.mean():.1f}%  median={np.median(drift_pcts):.1f}%  "
          f"p90={np.percentile(drift_pcts, 90):.1f}%  "
          f"pass_rate(<10%)={100*np.mean(drift_pcts < 10):.1f}%")


def main():
    model, mean, std = load_model()
    rng = np.random.default_rng(42)

    for trip_name in ["M", "Vta1a"]:
        X, Y_true, idx = load_holdout_tail(trip_name)
        Y_pred = predict_all(model, X, mean, std)

        print(f"\n=== Trip '{trip_name}' held-out tail ({len(Y_true)} windows, "
              f"~{len(Y_true) * STRIDE_SECONDS / 60:.1f} min) ===")

        for duration_s in BLACKOUT_DURATIONS_S:
            drift_pcts = simulate_segments(Y_pred, Y_true, duration_s, N_SEGMENTS_PER_DURATION, rng)
            summarize(drift_pcts, f"{duration_s}s blackout")


if __name__ == "__main__":
    main()
