"""
Step 9b: Diagnostic - check model's predicted vs true displacement
magnitude across TRAINING windows (not test). Tells us if the model
underpredicts speed everywhere, or only on the unseen test segment.

Reads: preprocessing/output/dr_model_M.pt, preprocessing/output/M_windows.npz
Shows: scatter of predicted vs true per-window displacement magnitude,
       plus a histogram comparing true displacement magnitude in
       train portion vs test portion (checks for distribution shift).
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

MODEL_PATH = "preprocessing/output/dr_model_M.pt"
WINDOWS_PATH = "preprocessing/output/M_windows.npz"


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
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        x = self.conv(x)
        return self.head(x)


def main():
    data = np.load(WINDOWS_PATH)
    X, Y = data["X"], data["Y"]
    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DRNet(in_channels=X.shape[2]).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    X_tensor = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1).to(device)
    with torch.no_grad():
        pred_all = model(X_tensor).cpu().numpy()

    true_mag = np.sqrt((Y ** 2).sum(axis=1))
    pred_mag = np.sqrt((pred_all ** 2).sum(axis=1))

    # split regions
    train_true, train_pred = true_mag[:train_end], pred_mag[:train_end]
    test_true, test_pred = true_mag[val_end:], pred_mag[val_end:]

    print("--- Displacement magnitude per window (meters, 4s window) ---")
    print(f"Train: true mean={train_true.mean():.2f}  pred mean={train_pred.mean():.2f}  "
          f"true max={train_true.max():.2f}  pred max={train_pred.max():.2f}")
    print(f"Test:  true mean={test_true.mean():.2f}  pred mean={test_pred.mean():.2f}  "
          f"true max={test_true.max():.2f}  pred max={test_pred.max():.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(train_true, train_pred, s=4, alpha=0.3, label="Train windows", color="blue")
    axes[0].scatter(test_true, test_pred, s=4, alpha=0.5, label="Test windows", color="red")
    max_val = max(true_mag.max(), pred_mag.max())
    axes[0].plot([0, max_val], [0, max_val], "k--", label="Perfect prediction")
    axes[0].set_xlabel("True displacement magnitude (m)")
    axes[0].set_ylabel("Predicted displacement magnitude (m)")
    axes[0].set_title("Predicted vs true per-window displacement")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(train_true, bins=50, alpha=0.5, label="Train true", color="blue", density=True)
    axes[1].hist(test_true, bins=50, alpha=0.5, label="Test true", color="red", density=True)
    axes[1].set_xlabel("True displacement magnitude (m) per window")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Distribution shift check: train vs test")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()