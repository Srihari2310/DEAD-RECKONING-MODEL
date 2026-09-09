import sys
import numpy as np
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from training.train_model import DRNet  # adjust import if your model class lives elsewhere

MODEL_PATH = "preprocessing/output/models/dr_model.pt"

def evaluate(trip_name):
    windows_path = f"preprocessing/output/06_windows/{trip_name}_windows.npz"
    d = np.load(windows_path)
    X, y_local, heading_start, y_global = d["X"], d["y_local"], d["heading_start"], d["y_global"]

    model = DRNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        x_tensor = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
        pred_local = model(x_tensor).numpy()

    # rotate local -> global using ground-truth heading_start (reference/optimistic eval)
    cos_h, sin_h = np.cos(heading_start), np.sin(heading_start)
    dx_l, dy_l = pred_local[:, 0], pred_local[:, 1]
    pred_global = np.stack([
        dx_l * cos_h + dy_l * sin_h,
        -dx_l * sin_h + dy_l * cos_h
    ], axis=1)

    pred_traj = np.cumsum(pred_global, axis=0)
    true_traj = np.cumsum(y_global, axis=0)

    total_true_distance = np.sum(np.linalg.norm(y_global, axis=1))
    endpoint_drift = np.linalg.norm(pred_traj[-1] - true_traj[-1])
    drift_pct = 100 * endpoint_drift / total_true_distance

    print(f"[{trip_name}] windows: {len(X)}")
    print(f"Total true distance: {total_true_distance:.1f} m")
    print(f"Endpoint drift: {endpoint_drift:.1f} m")
    print(f"Drift %: {drift_pct:.2f}%")

if __name__ == "__main__":
    trip = sys.argv[1]
    evaluate(trip)
