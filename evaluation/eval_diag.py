"""
Diagnostic (v30): where does the ~16% ceiling come from?
 1) speed-binned bias of the displacement model
 2) non-overlapping integration (windows i, i+4, ...) using the FULL 4 s
    displacement, no 0.25 scaling -> exact consistency with training labels.
Run: python evaluation/eval_diag.py
"""
import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))
from eval_model import rotate_local_to_global


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
            nn.Linear(64 + 1, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, x, v0):
        feat = self.conv(x).flatten(1)
        v0 = v0.unsqueeze(1)
        combined = torch.cat([feat, v0], dim=1)
        return self.head(combined)

MODEL_PATH = "preprocessing/output/models/dr_model.pt"
TEST_PATH = "preprocessing/output/test_splits/Vta1a_test_split.npz"
W = -np.array([0.568, 0.831, 1.123])
ACC_T = 1.0
GYR_T = 0.05

d = np.load(TEST_PATH)
X, Y, hs, v0 = d["X"], d["Y"], d["heading_start"], d["v_start"]
v_end = d["v_end"]


def stop_features(X):
    tail = X[:, -10:, :]
    acc = np.sqrt((tail[:, :, 0:3] ** 2).sum(-1)).mean(1)
    gyr = np.abs(tail[:, :, 3:6] @ W).mean(1)
    return acc, gyr


acc, gyr = stop_features(X)
stopped = v_end < 0.3
print("Stop-feature calibration:")
for name, feature in (("acc", acc), ("gyr", gyr)):
    print(name, "stopped p50/p90:", np.percentile(feature[stopped], [50, 90]),
          "moving p10/p50:", np.percentile(feature[~stopped], [10, 50]))
print("fraction of windows that are stops:", stopped.mean())
print(f"selected thresholds: acc<{ACC_T:.2f}, gyr<{GYR_T:.2f}")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m = DRNet(in_channels=X.shape[2]).to(dev)
m.load_state_dict(torch.load(MODEL_PATH, map_location=dev)); m.eval()
with torch.no_grad():
    X_tensor = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1).to(dev)
    v0_tensor = torch.tensor(v0, dtype=torch.float32).to(dev)
    pred = m(X_tensor, v0_tensor).cpu().numpy()
    P = pred[:, :2]
    pred_vend = pred[:, 2]

tn, pn = np.linalg.norm(Y, axis=1), np.linalg.norm(P, axis=1)
print(f"Overall: sum|pred|/sum|true| = {pn.sum()/tn.sum():.3f}   "
      f"mean vector err = {np.linalg.norm(P-Y,axis=1).mean():.1f} m "
      f"(mean true {tn.mean():.1f} m)")
kmh = tn / 4 * 3.6
print(f"\n{'speed bin km/h':>15} {'n':>5} {'true mean m':>12} {'pred mean m':>12} {'ratio':>6} {'vec err %':>10}")
for lo, hi in [(0,10),(10,30),(30,50),(50,80),(80,200)]:
    k = (kmh >= lo) & (kmh < hi)
    if k.sum() == 0: continue
    e = 100*np.linalg.norm(P[k]-Y[k],axis=1).sum()/tn[k].sum()
    print(f"{lo:>6}-{hi:<8} {k.sum():>5} {tn[k].mean():>12.1f} {pn[k].mean():>12.1f} "
          f"{pn[k].sum()/tn[k].sum():>6.2f} {e:>9.1f}%")

# Non-overlapping integration
rate = X[:, :, 3:6] @ W
Yg = rotate_local_to_global(Y, hs)
print("\nNon-overlapping (step=1 window, full 1s displacement), median drift %:")
print(f"{'blackout':>9} {'#runs':>6} {'GT heading':>11} {'gyro heading':>13}")
for dur in (32, 60, 120, 300):
    steps = dur
    gt, gy = [], []
    for s in range(1, len(X) - dur, 30):
        if kmh[s] < 20: continue
        idx = [s + j for j in range(steps)]
        target = Yg[idx].sum(0); dist = np.linalg.norm(Yg[idx], axis=1).sum()
        if dist < 50: continue
        # GT heading
        p = np.zeros(2)
        for j in idx:
            p += rotate_local_to_global(P[j:j+1], hs[j:j+1])[0]
        gt.append(100*np.linalg.norm(p-target)/dist)
        # gyro heading (start from true heading, integrate full 1 s of rate)
        h = hs[s]; p = np.zeros(2)
        for j in idx:
            p += rotate_local_to_global(P[j:j+1], np.array([h]))[0]
            h += rate[j].sum()*0.1
        gy.append(100*np.linalg.norm(p-target)/dist)
    print(f"{dur:>7}s {len(gt):>6} {np.median(gt):>10.1f}% {np.median(gy):>12.1f}%")
