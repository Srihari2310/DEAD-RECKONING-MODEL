"""
training/train_multitrip.py  (v2 -- corrected split + normalization)

Fixes the v1 bug: training on M ONLY and testing on Vta1a ONLY meant the
model never saw Vta1a's faster/aggressive driving regime during training
at all -- pure out-of-distribution failure (61% drift), not a fair test
of generalization.

Corrected approach (stopgap for 2 trips):
    - Each trip is split TIME-ORDERED into first 80% / last 20% (a
      contiguous block per trip, not randomly-by-window, so no
      overlapping-window leakage across the split boundary).
    - TRAIN = first 80% of M windows + first 80% of Vta1a windows,
      combined and shuffled together. Both driving styles are now
      represented during learning.
    - VAL = last 20% of M + last 20% of Vta1a windows (for early
      stopping / loss tracking only).
    - DRIFT EVAL = full-trajectory reconstruction on the held-out LAST
      20% contiguous segment of EACH trip separately (this is real
      unseen driving, just from a trip the model has partially seen
      earlier segments of -- the honest test once more trips are added
      is to hold out an ENTIRE trip/driver, not just a tail segment).

    Inputs are normalized (per-channel mean/std fit on TRAIN only) since
    Vta1a's IMU amplitude is ~3x M's (aggressive vs defensive driving) --
    v1 trained on unnormalized data dominated by M's smaller scale.

Once a 3rd+ trip is available, switch to true trip-level holdout:
    Train: trips A, B      Val: trip C (partial)      Test: trip D (full, unseen driver)

Run from project root:
    python training/train_multitrip.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

OUTPUT_DIR = Path("preprocessing/output")
MODEL_OUT = OUTPUT_DIR / "dr_model_multitrip.pt"
NORM_OUT = OUTPUT_DIR / "dr_model_multitrip_norm.npz"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
LR = 1e-3
MAX_EPOCHS = 200
EARLY_STOP_PATIENCE = 10
PHYSICAL_MAX_SPEED_MPS = 40.0
PHYSICAL_REG_WEIGHT = 0.01
TRAIN_FRACTION = 0.8  # per-trip time-ordered split point


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_windows(trip_name):
    path = OUTPUT_DIR / f"{trip_name}_windows.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path} -- run preprocessing/run_pipeline.py {trip_name} first.")
    data = np.load(path)
    return data["X"], data["Y"], data["idx"]


def split_trip(X, Y, idx, train_fraction=TRAIN_FRACTION):
    cut = int(len(X) * train_fraction)
    return (X[:cut], Y[:cut], idx[:cut]), (X[cut:], Y[cut:], idx[cut:])


def build_datasets():
    trips = {}
    for name in ["M", "Vta1a"]:
        X, Y, idx = load_windows(name)
        trips[name] = split_trip(X, Y, idx)
        print(f"{name} windows: total={len(X)}  train={len(trips[name][0][0])}  holdout={len(trips[name][1][0])}")

    X_train = np.concatenate([trips[n][0][0] for n in trips], axis=0)
    Y_train = np.concatenate([trips[n][0][1] for n in trips], axis=0)

    X_val = np.concatenate([trips[n][1][0] for n in trips], axis=0)
    Y_val = np.concatenate([trips[n][1][1] for n in trips], axis=0)

    print(f"\nCombined train: {X_train.shape} (M + Vta1a mixed, both driving styles present)")
    print(f"Combined val:   {X_val.shape} (held-out tail of each trip)")

    # per-trip holdout segments, kept separate for drift evaluation
    holdouts = {n: trips[n][1] for n in trips}

    return (X_train, Y_train), (X_val, Y_val), holdouts


def compute_normalization(X_train):
    mean = X_train.mean(axis=(0, 1))
    std = X_train.std(axis=(0, 1))
    std[std < 1e-6] = 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


def apply_normalization(X, mean, std):
    return (X - mean) / std


# ---------------------------------------------------------------------------
# Model (same architecture as v8/v1's DRNet)
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
        x = x.permute(0, 2, 1)  # (batch, timesteps, channels) -> (batch, channels, timesteps)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)


def physical_regularizer(pred, window_seconds=1.0):
    disp_mag = torch.sqrt(torch.sum(pred ** 2, dim=1) + 1e-8)
    implied_speed = disp_mag / window_seconds
    excess = torch.clamp(implied_speed - PHYSICAL_MAX_SPEED_MPS, min=0)
    return torch.mean(excess ** 2)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train():
    (X_train, Y_train), (X_val, Y_val), holdouts = build_datasets()

    mean, std = compute_normalization(X_train)
    print(f"\nNormalization (fit on train only): mean={mean}  std={std}")
    X_train_n = apply_normalization(X_train, mean, std)
    X_val_n = apply_normalization(X_val, mean, std)

    train_ds = TensorDataset(torch.from_numpy(X_train_n), torch.from_numpy(Y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val_n), torch.from_numpy(Y_val))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = DRNet(in_channels=X_train.shape[2]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    mse = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = mse(pred, yb) + PHYSICAL_REG_WEIGHT * physical_regularizer(pred)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(xb)
        train_loss = train_loss_sum / len(train_ds)

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                loss = mse(pred, yb)
                val_loss_sum += loss.item() * len(xb)
        val_loss = val_loss_sum / len(val_ds)

        print(f"Epoch {epoch:3d}  train_loss={train_loss:.2f}  val_loss={val_loss:.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch} (best val_loss={best_val_loss:.2f})")
                break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), MODEL_OUT)
    np.savez(NORM_OUT, mean=mean, std=std)
    print(f"\nSaved best model: {MODEL_OUT} (val_loss={best_val_loss:.2f})")
    print(f"Saved normalization stats: {NORM_OUT}")

    for trip_name, (Xh, Yh, idxh) in holdouts.items():
        Xh_n = apply_normalization(Xh, mean, std)
        evaluate_drift(model, Xh_n, Yh, trip_name=f"{trip_name}_holdout_tail")


# ---------------------------------------------------------------------------
# Drift evaluation: reconstruct held-out tail trajectory, compare to true path
# ---------------------------------------------------------------------------

def evaluate_drift(model, X, Y_true, trip_name):
    model.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X).to(DEVICE)
        Y_pred = model(X_t).cpu().numpy()

    pred_cum = np.cumsum(Y_pred, axis=0)
    true_cum = np.cumsum(Y_true, axis=0)

    true_dist = np.sum(np.sqrt(np.sum(Y_true ** 2, axis=1)))
    endpoint_error = np.sqrt(np.sum((pred_cum[-1] - true_cum[-1]) ** 2))
    drift_pct = 100 * endpoint_error / true_dist if true_dist > 0 else float("nan")

    print(f"\n=== Drift evaluation: {trip_name} (held-out tail, {len(Y_true)} windows) ===")
    print(f"True total distance:    {true_dist:.1f} m")
    print(f"Endpoint error (drift): {endpoint_error:.1f} m")
    print(f"Drift %% of distance:    {drift_pct:.1f}%%  (benchmark: <10%%)")

    pred_mean = np.sqrt(np.sum(Y_pred ** 2, axis=1)).mean()
    true_mean = np.sqrt(np.sum(Y_true ** 2, axis=1)).mean()
    print(f"Per-window displacement magnitude -- pred mean={pred_mean:.2f}m  true mean={true_mean:.2f}m")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 8))
        plt.plot(true_cum[:, 0], true_cum[:, 1], label="True", linewidth=1.2)
        plt.plot(pred_cum[:, 0], pred_cum[:, 1], label="Predicted", linewidth=1.0, alpha=0.8)
        plt.axis("equal")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.title(f"{trip_name}: true vs predicted (drift={drift_pct:.1f}%)")
        out_png = OUTPUT_DIR / f"{trip_name}_drift_eval.png"
        plt.savefig(out_png, dpi=120, bbox_inches="tight")
        print(f"Saved plot: {out_png}")
    except ImportError:
        pass


if __name__ == "__main__":
    train()
