"""
Multi-blackout evaluation (v30). Matches the SIH scenario: GPS is on, then
drops for N seconds. Position + heading start from ground truth, then we
dead-reckon (gyro-mix heading) with and without map-matching.

Run:  python evaluation/eval_blackout.py
"""
import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import argparse

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))

from eval_model import rotate_local_to_global
from preprocessing.map_matching import RoadMatcher, dead_reckon_blackout, DISP_SCALE

MODEL_PATH = "preprocessing/output/models/dr_model_v0noise.pt"
TEST_PATH = "preprocessing/output/test_splits/Vta1a_test_split.npz"

# heading-rate mix (sign: compass heading = -(V YawRate convention))
WEIGHT_SETS = {
    "full-fit mix":     -np.array([0.568, 0.831, 1.123]),
    "first-10% mix":    -np.array([0.271, 0.965, 1.374]),
}
DURATIONS_S = [30, 60]     # 1 window = 1 s
START_STEP = 30                      # candidate start every 30 windows
MIN_SPEED_KMH = 20.0
RESET_S = (10, 15, 20, 30)           # seconds; windows advance 1s each
ACC_T = 1.0
GYR_T = 0.05
GYRO_W = -np.array([0.568, 0.831, 1.123])
V0_EMA_ALPHAS = (0.25, 0.5, 0.75)
V0_NOISE_STDS = (0.25, 0.5, 1.0)  # m/s


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


def prepare_test_split(trip):
    """Load or create the evaluation split for a windowed trip."""
    split_path = Path(f"preprocessing/output/test_splits/{trip}_test_split.npz")
    if split_path.exists():
        return split_path

    windows_path = Path(f"preprocessing/output/06_windows/{trip}_windows.npz")
    if not windows_path.exists():
        raise FileNotFoundError(
            f"No test split or window file for trip {trip}: {windows_path}"
        )
    w = np.load(windows_path)
    y_local = w["Y"] if "Y" in w.files else w["y_local"]
    required = ("X", "y_global", "heading_start", "v_start", "v_end")
    missing = [key for key in required if key not in w.files]
    if missing:
        raise ValueError(f"Window file {windows_path} is missing: {missing}")
    split_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(split_path, X=w["X"], Y=y_local, y_global=w["y_global"],
             heading_start=w["heading_start"], v_start=w["v_start"],
             v_end=w["v_end"])
    print(f"Created test split: {split_path}")
    return split_path


def reliable_heading(heading_start, v_start, s, min_speed=3.0,
                     lookback=60, max_change=5.0):
    """Use a recent heading recorded while moving steadily."""
    def wrap(a):
        return (a + 180) % 360 - 180

    for j in range(s, max(2, s - lookback), -1):
        if (v_start[j] >= min_speed and
                abs(wrap(heading_start[j] - heading_start[j - 2])) < max_change):
            return heading_start[j]
    return heading_start[s]


def main(trip="Vta1a"):
    global TEST_PATH
    TEST_PATH = str(prepare_test_split(trip))
    graph_path = Path(f"preprocessing/output/osm_cache/{trip.lower()}_roads.graphml")
    origin_path = Path(f"preprocessing/output/osm_cache/{trip.lower()}_origin.json")
    if not graph_path.exists() or not origin_path.exists():
        raise FileNotFoundError(
            f"Missing OSM cache for {trip}. Expected:\n"
            f"  {graph_path}\n  {origin_path}"
        )

    d = np.load(TEST_PATH)
    X, Y = d["X"], d["Y"]
    heading_start, y_global = d["heading_start"], d["y_global"]
    v_start = d["v_start"]
    tail = X[:, -10:, :]
    acc_feature = np.sqrt((tail[:, :, 0:3] ** 2).sum(-1)).mean(1)
    gyr_feature = np.abs(tail[:, :, 3:6] @ GYRO_W).mean(1)
    stop_mask = (acc_feature < ACC_T) & (gyr_feature < GYR_T)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DRNet(in_channels=X.shape[2]).to(dev)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=dev))
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1).to(dev)

    def predict_window(index, v0):
        """Predict one window and return displacement plus chained v_end."""
        v0_tensor = torch.tensor([v0], dtype=torch.float32, device=dev)
        with torch.no_grad():
            pred = model(X_tensor[index:index + 1], v0_tensor)[0].cpu().numpy()
        return pred[:2], float(pred[2])

    true_disp = rotate_local_to_global(Y, heading_start) * DISP_SCALE
    true_traj = np.cumsum(true_disp, axis=0)                 # true metres, origin = trip start
    speed_kmh = np.linalg.norm(true_disp, axis=1) * 3.6      # per-second displacement
    print(f"True trip distance (scaled): {np.linalg.norm(true_disp,axis=1).sum():.0f} m "
          f"(V-file says ~40,700 m)")
    print(f"ZUPT thresholds: acc<{ACC_T:.2f}, gyro<{GYR_T:.2f}; "
          f"calibrated stop-feature windows: {stop_mask.sum()}/{len(stop_mask)} "
          f"({100 * stop_mask.mean():.2f}%)")

    print("Loading road network...")
    matcher = RoadMatcher(
        graph_path=graph_path, origin_path=origin_path, snap_gate_m=20.0
    )
    n = len(X)

    for wname, W in WEIGHT_SETS.items():
        rate = X[:, :, 3:6] @ W
        print(f"\n=== Heading source: {wname}  W={W} ===")
        print(f"{'Blackout':>9} {'#runs':>6} | {'GT-heading ceiling':>19} | "
              f"{'GT + chained v0':>16} | {'gyro only':>10} | "
              f"{'gyro+map':>9} {'gyro+lock':>10} | "
              f"{'map<10%':>8} {'lock<10%':>9} {'gyro<10%':>9} | matched map/lock")
        for dur in DURATIONS_S:
            res = {"gt": [], "gt_heading_chained_v0": [], "gyro": [],
                   "mm": [], "mm_lock": []}
            frac = []
            frac_lock = []
            for s in range(1, n - dur, START_STEP):
                if speed_kmh[s] < MIN_SPEED_KMH:
                    continue
                p0 = true_traj[s - 1]
                h0 = reliable_heading(heading_start, v_start, s)
                dist = np.linalg.norm(true_disp[s:s + dur], axis=1).sum()
                if dist < 50:
                    continue
                target = true_traj[s + dur - 1]
                for key, kw in (("gt", dict(heading_true=heading_start,
                                             use_true_v0=True, v0_true=v_start)),
                                ("gt_heading_chained_v0", dict(
                                    heading_true=heading_start, use_true_v0=False,
                                    v0_true=v_start)),
                                ("gyro", dict(v0_true=v_start)),
                                ("mm", dict(matcher=matcher, v0_true=v_start,
                                             use_viterbi_lite=False)),
                                ("mm_lock", dict(matcher=matcher,
                                                  v0_true=v_start,
                                                  use_viterbi_lite=False,
                                                  road_lock=True))):
                    p, mf = dead_reckon_blackout(
                        None, rate, p0, h0, s, dur,
                        predict_fn=predict_window, v0_start=v_start,
                        low_speed_thresh=3.0, bootstrap_window_count=24, **kw)
                    res[key].append(100 * np.linalg.norm(p - target) / dist)
                    if key == "mm":
                        frac.append(mf)
                    elif key == "mm_lock":
                        frac_lock.append(mf)
            g, gc, y, m, ml = (np.array(res[k]) for k in
                               ("gt", "gt_heading_chained_v0", "gyro",
                                "mm", "mm_lock"))
            print(f"{dur:>7}s {len(g):>6} | {np.median(g):>17.1f}% | "
                  f"{np.median(gc):>14.1f}% | {np.median(y):>9.1f}% | "
                  f"{np.median(m):>8.1f}% {np.median(ml):>9.1f}% | "
                  f"{100*(m<10).mean():>7.0f}% "
                  f"{100*(ml<10).mean():>8.0f}% "
                  f"{100*(y<10).mean():>8.0f}% | "
                  f"{100*np.mean(frac):.0f}%/{100*np.mean(frac_lock):.0f}%")
            print(f"gyro+map MEAN: {np.mean(m):.1f}%  (median: {np.median(m):.1f}%)")
            print(f"gyro+lock MEAN: {np.mean(ml):.1f}%  (median: {np.median(ml):.1f}%)")
            print(f"runs with matched<20%: {sum(1 for f in frac if f < 0.2)}/{len(frac)}")
            print(f"runs with matched>80%: {sum(1 for f in frac if f > 0.8)}/{len(frac)}")
            print(f"lock runs matched<20%: {sum(1 for f in frac_lock if f < 0.2)}/{len(frac_lock)}")
            print(f"lock runs matched>80%: {sum(1 for f in frac_lock if f > 0.8)}/{len(frac_lock)}")

        # Speed re-anchoring controls, evaluated with both GT and gyro
        # heading. Keep this diagnostic on the primary heading-rate mix to
        # limit run time.
        if wname == "full-fit mix":
            def speed_control_sweep(heading_mode="gt", **controls):
                medians = []
                use_zupt = controls.pop("use_zupt", False)
                for dur in DURATIONS_S:
                    errors = []
                    for s in range(1, n - dur, START_STEP):
                        if speed_kmh[s] < MIN_SPEED_KMH:
                            continue
                        p0 = true_traj[s - 1]
                        dist = np.linalg.norm(true_disp[s:s + dur], axis=1).sum()
                        if dist < 50:
                            continue
                        target = true_traj[s + dur - 1]
                        rng = np.random.default_rng(1000 + s + dur)
                        zupt_controls = (dict(use_zupt=True,
                                              acc_feature=acc_feature,
                                              gyr_feature=gyr_feature,
                                              acc_threshold=ACC_T,
                                              gyr_threshold=GYR_T)
                                         if use_zupt
                                         else {})
                        h0 = reliable_heading(heading_start, v_start, s)
                        p, _ = dead_reckon_blackout(
                            None, rate, p0, h0, s, dur,
                            heading_true=heading_start if heading_mode == "gt" else None,
                            predict_fn=predict_window, v0_start=v_start,
                            rng=rng, **zupt_controls,
                            low_speed_thresh=3.0, bootstrap_window_count=24,
                            **controls)
                        errors.append(100 * np.linalg.norm(p - target) / dist)
                    medians.append(np.median(errors))
                return medians

            print("\nPeriodic true-speed re-anchoring (median drift %):")
            print(f"{'ZUPT/mode/reset interval':>27} " +
                  " ".join(f"{d:>8}s" for d in DURATIONS_S))
            for heading_mode in ("gt", "gyro"):
                for use_zupt in (False, True):
                    for interval in RESET_S:
                        vals = speed_control_sweep(
                            heading_mode=heading_mode,
                            speed_reset_interval=interval,
                            v0_true=v_start, use_zupt=use_zupt)
                        label = "on" if use_zupt else "off"
                        print(f"{label:>4} {heading_mode:>8} {interval:>5}s   " +
                              " ".join(f"{v:>8.1f}" for v in vals))

            print("\nDetected stops inside evaluated Vta1a blackout windows:")
            for dur in DURATIONS_S:
                stop_hits = blackout_windows = 0
                for s in range(1, n - dur, START_STEP):
                    if speed_kmh[s] < MIN_SPEED_KMH:
                        continue
                    dist = np.linalg.norm(true_disp[s:s + dur], axis=1).sum()
                    if dist < 50:
                        continue
                    blackout_windows += 1
                    stop_hits += int(stop_mask[s:s + dur].sum())
                print(f"{dur:>7}s: {stop_hits} detected stop windows across "
                      f"{blackout_windows} blackout windows")

            print("\nGT heading + v0 EMA smoothing (median drift %):")
            print(f"{'EMA alpha':>16} " + " ".join(f"{d:>8}s" for d in DURATIONS_S))
            for alpha in V0_EMA_ALPHAS:
                vals = speed_control_sweep(v0_ema_alpha=alpha)
                print(f"{alpha:>16.2f} " + " ".join(f"{v:>8.1f}" for v in vals))

            print("\nGT heading + v0 input noise (median drift %):")
            print(f"{'noise std m/s':>16} " + " ".join(f"{d:>8}s" for d in DURATIONS_S))
            for noise_std in V0_NOISE_STDS:
                vals = speed_control_sweep(v0_noise_std=noise_std)
                print(f"{noise_std:>16.2f} " + " ".join(f"{v:>8.1f}" for v in vals))
    print("\n(values = MEDIAN drift as % of distance travelled during the blackout)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate blackout runs for a trip")
    parser.add_argument("--trip", default="Vta1a", help="trip name, e.g. Vfa01")
    args = parser.parse_args()
    main(args.trip)
