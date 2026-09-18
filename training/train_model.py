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

MODEL_OUT = "preprocessing/output/models/dr_model.pt"
TEST_SPLIT_OUT = "preprocessing/output/test_splits/Vta1a_test_split.npz"

VAL_FRACTION = 0.15

BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
PATIENCE = 8  # early stopping


class WindowDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)  # (N, C, T) for Conv1d
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


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
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),  # (dx, dy)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.head(x)


def physical_regularizer(pred, dt_window=4.0):
    # penalize implausible speed implied by predicted displacement
    speed = torch.sqrt((pred ** 2).sum(dim=1)) / dt_window
    implausible = torch.clamp(speed - 40.0, min=0)  # >40 m/s (~144km/h) is implausible for this dataset
    return (implausible ** 2).mean()


def split_trip(d, val_fraction=VAL_FRACTION):
    """Split one trip chronologically, reserving its final windows for validation."""
    n = len(d["X"])
    split = int(n * (1 - val_fraction))
    X_train, Y_train = d["X"][:split], d["y_local"][:split]
    X_val, Y_val = d["X"][split:], d["y_local"][split:]
    return X_train, Y_train, X_val, Y_val


def main():
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    os.makedirs(os.path.dirname(TEST_SPLIT_OUT), exist_ok=True)

    train_X_parts, train_Y_parts = [], []
    val_X_parts, val_Y_parts = [], []

    for path in DATA_PATHS:
        d = np.load(path)
        X_train_trip, Y_train_trip, X_val_trip, Y_val_trip = split_trip(d)
        train_X_parts.append(X_train_trip)
        train_Y_parts.append(Y_train_trip)
        val_X_parts.append(X_val_trip)
        val_Y_parts.append(Y_val_trip)

    X_train = np.concatenate(train_X_parts)
    Y_train = np.concatenate(train_Y_parts)
    X_val = np.concatenate(val_X_parts)
    Y_val = np.concatenate(val_Y_parts)

    d_test = np.load(TEST_PATH)
    X_test, Y_test = d_test["X"], d_test["y_local"]

    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(
        WindowDataset(X_train, Y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=g,
    )
    val_loader = DataLoader(WindowDataset(X_val, Y_val), batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DRNet(in_channels=X_train.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    mse = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = mse(pred, yb) + 0.01 * physical_regularizer(pred)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(X_train)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = mse(pred, yb)
                val_loss += loss.item() * len(xb)
        val_loss /= len(X_val)

        print(f"Epoch {epoch:>3} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_OUT)
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
             y_global=d_test["y_global"], heading_start=d_test["heading_start"])
    print(f"Saved test split: {TEST_SPLIT_OUT}")


if __name__ == "__main__":
    main()
