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


def integrate_trajectory_with_tracked_heading(pred_local, gyro_yaw,
                                              heading_start_first, dt=0.1,
                                              stride_samples=10):
    """
    Reconstruct a trajectory using tracked heading from integrated gyro yaw
    rate, rather than ground-truth heading_start for each window.

    pred_local: (N, 2) predicted local-frame displacement per window
    gyro_yaw: (N, 40) gyro yaw-rate in rad/s per window
    heading_start_first: scalar heading used only to bootstrap window 0
    dt: sample period in seconds
    stride_samples: number of samples elapsed between consecutive windows

    Returns (N, 2) global-frame displacement per window.
    """
    n_windows = pred_local.shape[0]
    global_disp = np.zeros((n_windows, 2), dtype=np.float32)
    heading_estimate = float(heading_start_first)

    for i in range(n_windows):
        dx_local, dy_local = pred_local[i]
        global_disp[i] = [
            dx_local * np.cos(heading_estimate) + dy_local * np.sin(heading_estimate),
            -dx_local * np.sin(heading_estimate) + dy_local * np.cos(heading_estimate),
        ]

        # Track heading for the next window using only the elapsed stride,
        # not the entire overlapping window.
        heading_estimate += np.sum(gyro_yaw[i, :stride_samples]) * dt

    return global_disp


def integrate_trajectory_with_periodic_correction(pred_local, gyro_yaw,
                                                  heading_start_true,
                                                  correction_interval,
                                                  dt=0.1, stride_samples=10):
    """
    Re-anchor tracked heading to heading_start_true periodically, simulating
    periodic GNSS or map-match heading corrections.
    """
    n_windows = pred_local.shape[0]
    global_disp = np.zeros((n_windows, 2), dtype=np.float32)
    heading_estimate = float(heading_start_true[0])

    for i in range(n_windows):
        if correction_interval is not None and i % correction_interval == 0:
            heading_estimate = float(heading_start_true[i])

        dx_local, dy_local = pred_local[i]
        global_disp[i] = [
            dx_local * np.cos(heading_estimate) + dy_local * np.sin(heading_estimate),
            -dx_local * np.sin(heading_estimate) + dy_local * np.cos(heading_estimate),
        ]

        heading_estimate += np.sum(gyro_yaw[i, :stride_samples]) * dt

    return global_disp



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
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    X_tensor = torch.tensor(X_test, dtype=torch.float32).permute(0, 2, 1).to(device)
    with torch.no_grad():
        pred_local = model(X_tensor).cpu().numpy()  # (N, 2) predicted local (dx, dy)

    # Existing evaluation: rotate each window using its saved ground-truth
    # heading_start. This is retained as the reference/baseline.
    pred_global = rotate_local_to_global(pred_local, heading_start)
    true_global = rotate_local_to_global(Y_test, heading_start)

    # sanity check: true_global (rotated back) should closely match saved y_global_true
    sanity_diff = np.abs(true_global - y_global_true).mean()
    print(f"Sanity check -- mean |rotated_true - saved_y_global|: {sanity_diff:.4f} m (should be ~0)")

    # Deployment-style evaluation: bootstrap from the first heading only, then
    # track heading using the per-window gyro yaw-rate channel (X[:, :, 3]).
    gyro_yaw_test = X_test[:, :, 3]
    pred_global_tracked = integrate_trajectory_with_tracked_heading(
        pred_local, gyro_yaw_test, heading_start[0], dt=0.1, stride_samples=10
    )

    # Reconstruct both paths by cumulative sum of per-window global displacement.
    pred_trajectory = np.cumsum(pred_global, axis=0)
    tracked_trajectory = np.cumsum(pred_global_tracked, axis=0)
    true_trajectory = np.cumsum(true_global, axis=0)

    x_pred, y_pred = pred_trajectory[:, 0], pred_trajectory[:, 1]
    x_tracked, y_tracked = tracked_trajectory[:, 0], tracked_trajectory[:, 1]
    x_true, y_true = true_trajectory[:, 0], true_trajectory[:, 1]

    # Drift metrics for both the ground-truth-heading reference and tracked
    # heading deployment simulation.
    endpoint_drift = np.linalg.norm(pred_trajectory[-1] - true_trajectory[-1])
    tracked_endpoint_drift = np.linalg.norm(tracked_trajectory[-1] - true_trajectory[-1])
    total_true_dist = np.linalg.norm(np.diff(true_trajectory, axis=0, prepend=0), axis=1).sum()
    drift_pct = 100 * endpoint_drift / total_true_dist
    tracked_drift_pct = 100 * tracked_endpoint_drift / total_true_dist

    pos_err = np.linalg.norm(pred_trajectory - true_trajectory, axis=1)
    tracked_pos_err = np.linalg.norm(tracked_trajectory - true_trajectory, axis=1)

    correction_intervals_windows = {
        "every 1s (1 window)": 1,
        "every 10s (10 windows)": 10,
        "every 30s (30 windows)": 30,
        "every 60s (60 windows)": 60,
        "every 120s (120 windows)": 120,
        "every 300s (300 windows)": 300,
        "NEVER (pure gyro)": None,
    }
    print("Periodic heading-correction sweep:")
    print(f"{'Interval':<26}{'Drift %':>10}")
    for label, interval in correction_intervals_windows.items():
        corrected_disp = integrate_trajectory_with_periodic_correction(
            pred_local, gyro_yaw_test, heading_start, interval,
            dt=0.1, stride_samples=10,
        )
        corrected_trajectory = np.cumsum(corrected_disp, axis=0)
        corrected_endpoint_drift = np.linalg.norm(
            corrected_trajectory[-1] - true_trajectory[-1]
        )
        corrected_drift_pct = 100 * corrected_endpoint_drift / total_true_dist
        print(f"{label:<26}{corrected_drift_pct:>9.2f}%")

    print(f"Reference drift (ground-truth heading_start): {drift_pct:.2f}%")
    print(f"Tracked-heading drift (gyro integration): {tracked_drift_pct:.2f}%")
    print(f"Tracked vs reference: {'better' if tracked_drift_pct < drift_pct else 'worse' if tracked_drift_pct > drift_pct else 'same'}")

    # Keep the existing reference metrics/reporting below, and add tracked values.
    print(f"Reference endpoint drift: {endpoint_drift:.1f} m")
    print(f"Tracked-heading endpoint drift: {tracked_endpoint_drift:.1f} m")
    print(f"Total true distance covered (test segment): {total_true_dist:.1f} m")
    print(f"Max reference running position error: {pos_err.max():.1f} m")
    print(f"Max tracked running position error: {tracked_pos_err.max():.1f} m")

    # Existing reference metrics retained for compatibility.
    for idx in [50, 200, 500, 1000, 1500, 2000, len(pos_err)-1]:
        if idx < len(pos_err):
            print(f"window {idx}: reference error = {pos_err[idx]:.1f} m | "
                  f"tracked error = {tracked_pos_err[idx]:.1f} m")

    print(f"Test windows: {len(X_test)}")
    print(f"Reference drift as % of distance: {drift_pct:.2f}%  (SIH benchmark: <10%)")
    print(f"Tracked drift as % of distance: {tracked_drift_pct:.2f}%  (SIH benchmark: <10%)")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].plot(x_true, y_true, label="True path", color="green", linewidth=2)
    axes[0].plot(x_pred, y_pred, label="Model path (reference heading)", color="blue", linewidth=1.5, alpha=0.8)
    axes[0].plot(x_tracked, y_tracked, label="Model path (tracked heading)", color="orange", linewidth=1.5, alpha=0.8)
    axes[0].scatter([x_true[0]], [y_true[0]], color="black", s=60, label="Start", zorder=5)
    axes[0].scatter([x_true[-1]], [y_true[-1]], color="green", marker="x", s=100, label="True end", zorder=5)
    axes[0].scatter([x_pred[-1]], [y_pred[-1]], color="blue", marker="x", s=100, label="Reference end", zorder=5)
    axes[0].scatter([x_tracked[-1]], [y_tracked[-1]], color="orange", marker="x", s=100, label="Tracked end", zorder=5)
    axes[0].set_xlabel("X (m, east)")
    axes[0].set_ylabel("Y (m, north)")
    axes[0].set_title(f"Test segment\nReference: {drift_pct:.1f}% | Tracked: {tracked_drift_pct:.1f}%")
    axes[0].legend()
    axes[0].axis("equal")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(pos_err, color="red", label="Reference heading")
    axes[1].plot(tracked_pos_err, color="orange", label="Tracked heading")
    axes[1].set_xlabel("Window index (time)")
    axes[1].set_ylabel("Running position error (m)")
    axes[1].set_title("Drift growth over test segment")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
