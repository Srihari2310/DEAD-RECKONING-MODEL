"""
Step 8: Train the AI Dead-Reckoning model (stage [4]) on the configured trips.
1D-CNN backbone -> predicts (dx, dy) displacement per window.

Each training trip is split by contiguous time blocks (first 85% train, last
15% validation) to avoid leakage between overlapping windows. Vta1a, Vtb11,
and Vw11 are held out for evaluation.
"""

import numpy as np
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DATA_PATHS = [
    "preprocessing/output/06_windows/M_windows.npz",
    "preprocessing/output/06_windows/Vta2_windows.npz",
    "preprocessing/output/06_windows/Vfa01_windows.npz",
    "preprocessing/output/06_windows/Vtb1_windows.npz",
    "preprocessing/output/06_windows/Vtb4_windows.npz",
    "preprocessing/output/06_windows/Vtb5_windows.npz",
    "preprocessing/output/06_windows/Vtb6_windows.npz",
    "preprocessing/output/06_windows/Vtb7_windows.npz",
    "preprocessing/output/06_windows/Vtb9_windows.npz",
    # Vtb11 removed -- now a held-out test trip
    "preprocessing/output/06_windows/Vtb12_windows.npz",
    "preprocessing/output/06_windows/S1_windows.npz",
    "preprocessing/output/06_windows/S2_windows.npz",
    "preprocessing/output/06_windows/S3a_windows.npz",
    "preprocessing/output/06_windows/s3b_windows.npz",
    "preprocessing/output/06_windows/s3c_windows.npz",
    # Vw series, added for the expanded training experiment; Vw11 remains held out.
    
]
TEST_PATH = "preprocessing/output/06_windows/Vta1a_windows.npz"

MODEL_OUT = "preprocessing/output/models/dr_model_v0noise.pt"
TEST_SPLIT_OUT = "preprocessing/output/test_splits/Vta1a_test_split.npz"

VAL_FRACTION = 0.15

BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
PATIENCE = 8  # early stopping


class WindowDataset(Dataset):
    def __init__(self, X, Y, V0, VEND):
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)  # (N, C, T) for Conv1d
        self.Y = torch.tensor(Y, dtype=torch.float32)          # (N, 2) dx, dy
        self.V0 = torch.tensor(V0, dtype=torch.float32)        # (N,)
        self.VEND = torch.tensor(VEND, dtype=torch.float32)    # (N,)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.V0[idx], self.Y[idx], self.VEND[idx]


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
            nn.AdaptiveAvgPool1d(1),  # global pooling -> (N, 64, 1)
        )
        self.head = nn.Sequential(
            nn.Linear(64 + 1, 32),   # +1 for v0
            nn.ReLU(),
            nn.Linear(32, 3),        # (dx, dy, v_end)
        )

    def forward(self, x, v0):
        feat = self.conv(x).flatten(1)          # (N, 64)
        v0 = v0.unsqueeze(1)                    # (N, 1)
        combined = torch.cat([feat, v0], dim=1) # (N, 65)
        return self.head(combined)


def physical_regularizer(pred, dt_window=4.0):
    # penalize implausible speed implied by predicted displacement
    speed = torch.sqrt((pred ** 2).sum(dim=1)) / dt_window
    implausible = torch.clamp(speed - 40.0, min=0)  # >40 m/s (~144km/h) is implausible for this dataset
    return (implausible ** 2).mean()


mse = nn.MSELoss()


def compute_loss(pred, yb, vend_true):
    pred_disp, pred_vend = pred[:, :2], pred[:, 2]
    disp_loss = mse(pred_disp, yb)
    vend_loss = mse(pred_vend, vend_true)
    reg = physical_regularizer(pred_disp)
    return disp_loss + 4.0 * vend_loss + 0.01 * reg


def split_trip(d, val_fraction=VAL_FRACTION):
    """Split one trip chronologically, reserving its final windows for validation."""
    n = len(d["X"])
    split = int(n * (1 - val_fraction))
    X_train, Y_train = d["X"][:split], d["y_local"][:split]
    X_val, Y_val = d["X"][split:], d["y_local"][split:]
    V0_train, VEND_train = d["v_start"][:split], d["v_end"][:split]
    V0_val, VEND_val = d["v_start"][split:], d["v_end"][split:]
    return X_train, Y_train, V0_train, VEND_train, X_val, Y_val, V0_val, VEND_val


def main():
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    os.makedirs(os.path.dirname(TEST_SPLIT_OUT), exist_ok=True)

    train_X_parts, train_Y_parts, train_V0_parts, train_VEND_parts = [], [], [], []
    val_X_parts, val_Y_parts, val_V0_parts, val_VEND_parts = [], [], [], []

    for path in DATA_PATHS:
        d = np.load(path)
        (X_train_trip, Y_train_trip, V0_train_trip, VEND_train_trip,
         X_val_trip, Y_val_trip, V0_val_trip, VEND_val_trip) = split_trip(d)
        train_X_parts.append(X_train_trip)
        train_Y_parts.append(Y_train_trip)
        train_V0_parts.append(V0_train_trip)
        train_VEND_parts.append(VEND_train_trip)
        val_X_parts.append(X_val_trip)
        val_Y_parts.append(Y_val_trip)
        val_V0_parts.append(V0_val_trip)
        val_VEND_parts.append(VEND_val_trip)

    X_train = np.concatenate(train_X_parts)
    Y_train = np.concatenate(train_Y_parts)
    V0_train = np.concatenate(train_V0_parts)
    VEND_train = np.concatenate(train_VEND_parts)
    X_val = np.concatenate(val_X_parts)
    Y_val = np.concatenate(val_Y_parts)
    V0_val = np.concatenate(val_V0_parts)
    VEND_val = np.concatenate(val_VEND_parts)

    d_test = np.load(TEST_PATH)
    X_test, Y_test = d_test["X"], d_test["y_local"]
    V0_test, VEND_test = d_test["v_start"], d_test["v_end"]

    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(
        WindowDataset(X_train, Y_train, V0_train, VEND_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=g,
    )
    val_loader = DataLoader(WindowDataset(X_val, Y_val, V0_val, VEND_val), batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DRNet(in_channels=X_train.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for xb, v0b, yb, vend_true in train_loader:
            xb, v0b = xb.to(device), v0b.to(device)
            yb, vend_true = yb.to(device), vend_true.to(device)
            optimizer.zero_grad()
            sigma = torch.rand_like(v0b) * 2.0
            v0_in = torch.clamp(v0b + torch.randn_like(v0b) * sigma, min=0.0)
            pred = model(xb, v0_in)
            loss = compute_loss(pred, yb, vend_true)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(X_train)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, v0b, yb, vend_true in val_loader:
                xb, v0b = xb.to(device), v0b.to(device)
                yb, vend_true = yb.to(device), vend_true.to(device)
                pred = model(xb, v0b)
                loss = compute_loss(pred, yb, vend_true)
                val_loss += loss.item() * len(xb)
        val_loss /= len(X_val)

        print(f"Epoch {epoch:>3} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "preprocessing/output/models/dr_model_v0noise.pt")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"Best val_loss: {best_val_loss:.4f}")
    print(f"Saved best checkpoint: {MODEL_OUT}")

    # save test set indices/arrays for later evaluation (position integrator, drift check)
    np.savez(TEST_SPLIT_OUT,
             X=X_test, Y=Y_test,
             y_global=d_test["y_global"], heading_start=d_test["heading_start"],
             v_start=V0_test, v_end=VEND_test)
    print(f"Saved test split: {TEST_SPLIT_OUT}")


if __name__ == "__main__":
    main()
