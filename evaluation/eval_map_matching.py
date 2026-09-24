"""
Map-matching evaluation: compares dead-reckoning drift with map-matching
correction against the existing reference (ground-truth heading) and
tracked (pure gyro) baselines from eval_model.py.

Run standalone, after eval_model.py's baseline numbers are known:
    python evaluation/eval_map_matching.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))

from eval_model import DRNet, rotate_local_to_global, integrate_trajectory_with_tracked_heading
from preprocessing.map_matching import RoadMatcher, integrate_trajectory_with_map_matching

MODEL_PATH = "preprocessing/output/models/dr_model.pt"
TEST_PATH = "preprocessing/output/test_splits/Vta1a_test_split.npz"
GRAPH_PATH = "preprocessing/output/osm_cache/vta1a_roads.graphml"
ORIGIN_PATH = "preprocessing/output/osm_cache/vta1a_origin.json"


def main():
    data = np.load(TEST_PATH)
    X_test, Y_test = data["X"], data["Y"]
    heading_start = data["heading_start"]
    y_global_true = data["y_global"]
    v_start = data["v_start"]
    GYRO_W = -np.array([0.568, 0.831, 1.123])   # heading rate = -(V YawRate convention); try -np.array([0.271, 0.965, 1.374]) too
    gyro_yaw_test = X_test[:, :, 3:6] @ GYRO_W  # (N, 40) heading rate, rad/s

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DRNet(in_channels=X_test.shape[2]).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    X_tensor = torch.tensor(X_test, dtype=torch.float32).permute(0, 2, 1).to(device)
    v0_tensor = torch.tensor(v_start, dtype=torch.float32).to(device)
    with torch.no_grad():
        pred = model(X_tensor, v0_tensor).cpu().numpy()
        pred_local = pred[:, :2]

    h = float(heading_start[0])
    for i in range(60):
        err = (h - heading_start[i] + np.pi) % (2*np.pi) - np.pi
        if i % 5 == 0:
            print(f"w{i}: tracked={np.degrees(h):.1f} true={np.degrees(heading_start[i]):.1f} err={np.degrees(err):.1f}")
        h += np.sum(gyro_yaw_test[i, :10]) * 0.1

    true_global = rotate_local_to_global(Y_test, heading_start)
    true_trajectory = np.cumsum(true_global, axis=0)
    total_true_dist = np.linalg.norm(np.diff(true_trajectory, axis=0, prepend=0), axis=1).sum()

    # Reference (ground-truth heading every window) -- same as eval_model.py
    pred_global_ref = rotate_local_to_global(pred_local, heading_start)
    ref_trajectory = np.cumsum(pred_global_ref, axis=0)
    ref_drift = 100 * np.linalg.norm(ref_trajectory[-1] - true_trajectory[-1]) / total_true_dist

    # Tracked (pure gyro, never corrected) -- same as eval_model.py
    pred_global_tracked = integrate_trajectory_with_tracked_heading(
        pred_local, gyro_yaw_test, heading_start[0], dt=0.1, stride_samples=10
    )
    tracked_trajectory = np.cumsum(pred_global_tracked, axis=0)
    tracked_drift = 100 * np.linalg.norm(tracked_trajectory[-1] - true_trajectory[-1]) / total_true_dist

    # Map-matching
    print("Loading road network / building matcher...")
    road_matcher = RoadMatcher(graph_path=GRAPH_PATH, origin_path=ORIGIN_PATH, snap_gate_m=20.0)

    mm_trajectory = integrate_trajectory_with_map_matching(
        pred_local, gyro_yaw_test, heading_start, road_matcher,
        dt=0.1, stride_samples=10
    )
    matched_path = Path("preprocessing/output/diagnostics/matched_trajectory_Vta1a.npy")
    matched_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(matched_path, mm_trajectory)
    print(f"Saved matched trajectory: {matched_path}")
    mm_drift = 100 * np.linalg.norm(mm_trajectory[-1] - true_trajectory[-1]) / total_true_dist

    mm_pos_err = np.linalg.norm(mm_trajectory - true_trajectory, axis=1)
    tracked_pos_err = np.linalg.norm(tracked_trajectory - true_trajectory, axis=1)
    ref_pos_err = np.linalg.norm(ref_trajectory - true_trajectory, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: trajectory comparison
    ax = axes[0]
    ax.plot(true_trajectory[:, 0], true_trajectory[:, 1],
            label="True path", color="black", linewidth=2)
    ax.plot(mm_trajectory[:, 0], mm_trajectory[:, 1],
            label="Map-matched", color="tab:green", linewidth=1.5)
    ax.plot(tracked_trajectory[:, 0], tracked_trajectory[:, 1],
            label="Tracked (pure gyro)", color="tab:red", linewidth=1, alpha=0.6)
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    trip_name = globals().get("TEST_TRIP_NAME", "Vta1a")
    ax.set_title(f"Trajectory comparison — {trip_name}")
    ax.legend()
    ax.axis("equal")

    # Right: running position error over windows
    ax2 = axes[1]
    ax2.plot(mm_pos_err, label="Map-matched error", color="tab:green")
    ax2.plot(tracked_pos_err, label="Tracked error", color="tab:red", alpha=0.6)
    ax2.set_xlabel("Window index")
    ax2.set_ylabel("Position error (m)")
    ax2.set_title("Running position error")
    ax2.legend()

    plt.tight_layout()
    out_path = Path("preprocessing/output/diagnostics/map_matching_comparison.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")
    plt.show()

    print(f"\n=== Map-matching comparison (test trip: Vta1a) ===")
    print(f"Reference (GT heading, ~1s):   {ref_drift:.2f}%")
    print(f"Map-matched (no GPS, snapped): {mm_drift:.2f}%")
    print(f"Tracked (pure gyro, never):    {tracked_drift:.2f}%")
    print(f"Total true distance: {total_true_dist:.1f} m")

    for idx in [50, 200, 500, 1000, 1500, 2000, len(mm_pos_err) - 1]:
        if idx < len(mm_pos_err):
            print(f"window {idx}: ref={ref_pos_err[idx]:.1f}m | "
                  f"map-matched={mm_pos_err[idx]:.1f}m | tracked={tracked_pos_err[idx]:.1f}m")

if __name__ == "__main__":
    main()
