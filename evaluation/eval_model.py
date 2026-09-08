"""
Step 9: Evaluate trained model on held-out test set (Vta1a).
Rotates predicted local-frame (dx,dy) back to global frame using
heading_start, reconstructs path via cumulative sum, compares to
true path, computes drift metrics against the SIH <10% benchmark.

Reads: preprocessing/output/dr_model.pt, preprocessing/output/Vta1a_test_split.npz
Shows: plot of predicted path vs true path (test segment only).
Prints: drift in meters and as % of distance traveled.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

MODEL_PATH = "preprocessing/output/models/dr_model.pt"
TEST_PATH = "preprocessing/output/test_splits/Vta1a_test_split.npz"


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


def rotate_local_to_global(vec_local, heading):
    """Inverse of the -heading_start rotation used in window_trip()."""
    dx_l, dy_l = vec_local[:, 0], vec_local[:, 1]
    dx_g = dx_l * np.cos(heading) + dy_l * np.sin(heading)
    dy_g = -dx_l * np.sin(heading) + dy_l * np.cos(heading)
    return np.stack([dx_g, dy_g], axis=1)


def main():
    data = np.load(TEST_PATH)
    X_test, Y_test = data["X"], data["Y"]           # Y_test = y_local (truth, local frame)
    heading_start = data["heading_start"]            # per-window heading, radians
    y_global_true = data["y_global"]                 # global-frame truth, for sanity check

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DRNet(in_channels=X_test.shape[2]).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    X_tensor = torch.tensor(X_test, dtype=torch.float32).permute(0, 2, 1).to(device)
    with torch.no_grad():
        pred_local = model(X_tensor).cpu().numpy()  # (N, 2) predicted local (dx, dy)

    # rotate predictions and truth back to global frame
    pred_global = rotate_local_to_global(pred_local, heading_start)
    true_global = rotate_local_to_global(Y_test, heading_start)

    # sanity check: true_global (rotated back) should closely match saved y_global_true
    sanity_diff = np.abs(true_global - y_global_true).mean()
    print(f"Sanity check -- mean |rotated_true - saved_y_global|: {sanity_diff:.4f} m (should be ~0)")

    # reconstruct path: cumulative sum of per-window global displacement
    x_pred = np.cumsum(pred_global[:, 0])
    y_pred = np.cumsum(pred_global[:, 1])
    x_true = np.cumsum(true_global[:, 0])
    y_true = np.cumsum(true_global[:, 1])

    # drift metrics
    endpoint_drift = np.sqrt((x_pred[-1] - x_true[-1])**2 + (y_pred[-1] - y_true[-1])**2)
    total_true_dist = np.sqrt(np.diff(x_true, prepend=0)**2 + np.diff(y_true, prepend=0)**2).sum()
    drift_pct = 100 * endpoint_drift / total_true_dist

    # per-window position error (running, not just endpoint) - shows if drift grows over time
    pos_err = np.sqrt((x_pred - x_true)**2 + (y_pred - y_true)**2)
    for idx in [50, 200, 500, 1000, 1500, 2000, len(pos_err)-1]:
        print(f"window {idx}: running error = {pos_err[idx]:.1f} m")

    print(f"Test windows: {len(X_test)}")
    print(f"Total true distance covered (test segment): {total_true_dist:.1f} m")
    print(f"Endpoint drift: {endpoint_drift:.1f} m")
    print(f"Drift as % of distance: {drift_pct:.2f}%  (SIH benchmark: <10%)")
    print(f"Max running position error: {pos_err.max():.1f} m")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].plot(x_true, y_true, label="True path", color="green", linewidth=2)
    axes[0].plot(x_pred, y_pred, label="Model-predicted path", color="blue", linewidth=1.5, alpha=0.8)
    axes[0].scatter([x_true[0]], [y_true[0]], color="black", s=60, label="Start", zorder=5)
    axes[0].scatter([x_true[-1]], [y_true[-1]], color="green", marker="x", s=100, label="True end", zorder=5)
    axes[0].scatter([x_pred[-1]], [y_pred[-1]], color="blue", marker="x", s=100, label="Predicted end", zorder=5)
    axes[0].set_xlabel("X (m, east)")
    axes[0].set_ylabel("Y (m, north)")
    axes[0].set_title(f"Test segment: model vs true path\nDrift: {drift_pct:.1f}% of distance")
    axes[0].legend()
    axes[0].axis("equal")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(pos_err, color="red")
    axes[1].set_xlabel("Window index (time)")
    axes[1].set_ylabel("Running position error (m)")
    axes[1].set_title("Drift growth over test segment")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
