"""
Step 8: Train the AI Dead-Reckoning model (stage [4]) on trip M windows.
1D-CNN backbone -> predicts (dx, dy) displacement per window.

Reads: preprocessing/output/M_windows.npz
Saves: preprocessing/output/dr_model_M.pt (best checkpoint)

NOTE: only trip M available so far, so split is by CONTIGUOUS time block
(first 70% train, next 15% val, last 15% test) - NOT random window split,
to avoid leakage between overlapping windows. Once more trips are added,
switch to proper trip-level splitting per the training plan.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DATA_PATH = "preprocessing/output/M_windows.npz"
MODEL_OUT = "preprocessing/output/dr_model_M.pt"

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


def main():
    data = np.load(DATA_PATH)
    X, Y = data["X"], data["Y"]
    n = len(X)

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train, Y_train = X[:train_end], Y[:train_end]
    X_val, Y_val = X[train_end:val_end], Y[train_end:val_end]
    X_test, Y_test = X[val_end:], Y[val_end:]

    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    train_loader = DataLoader(WindowDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(WindowDataset(X_val, Y_val), batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DRNet(in_channels=X.shape[2]).to(device)
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
    np.savez("preprocessing/output/M_test_split.npz", X=X_test, Y=Y_test)
    print("Saved test split: preprocessing/output/M_test_split.npz")


if __name__ == "__main__":
    main()