"""
Map-matching module (v30): densified segments + blackout simulation.

Frame: x = east, y = north, metres, origin = trip's first GPS point
(translation-only from osmnx's UTM projection).

Changes vs v29:
  * Edges use their real geometry (curves) and are sampled every ~10 m, so the
    KDTree finds the correct segment even on long roads (old code indexed
    segment MIDPOINTS only).
  * DISP_SCALE: model predicts displacement over a 40-sample (4 s) window but
    windows advance only 10 samples (1 s) -> per-window displacement must be
    scaled by 10/40 = 0.25 before integrating (old eval overcounted 4x).
  * dead_reckon_blackout(): simulates a GPS blackout of N windows starting
    from a known position/heading.
"""

import json
import itertools
import numpy as np
import osmnx as ox
import pyproj
from pathlib import Path
from scipy.spatial import cKDTree

CACHE_DIR = Path("preprocessing/output/osm_cache")
GRAPH_PATH = CACHE_DIR / "vta1a_roads.graphml"
ORIGIN_PATH = CACHE_DIR / "vta1a_origin.json"

DISP_SCALE = 10.0 / 40.0   # stride / window length
SAMPLE_SPACING_M = 10.0


class RoadMatcher:
    def __init__(self, graph_path=GRAPH_PATH, origin_path=ORIGIN_PATH,
                 snap_gate_m=20.0):
        with open(origin_path) as f:
            origin = json.load(f)
        olat, olon = origin["origin_lat"], origin["origin_lon"]

        G = ox.load_graphml(graph_path)
        G_proj = ox.project_graph(G)
        crs = G_proj.graph["crs"]
        tf = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        oe, on = tf.transform(olon, olat)

        self.snap_gate_m = snap_gate_m
        segs, pts, pt_seg = [], [], []

        for u, v, data in G_proj.edges(data=True):
            geom = data.get("geometry")
            if geom is not None:
                coords = [(x - oe, y - on) for x, y in geom.coords]
            else:
                coords = [(G_proj.nodes[u]["x"] - oe, G_proj.nodes[u]["y"] - on),
                          (G_proj.nodes[v]["x"] - oe, G_proj.nodes[v]["y"] - on)]
            for (x1, y1), (x2, y2) in zip(coords[:-1], coords[1:]):
                idx = len(segs)
                segs.append((x1, y1, x2, y2))
                length = np.hypot(x2 - x1, y2 - y1)
                n = max(1, int(np.ceil(length / SAMPLE_SPACING_M)))
                for t in np.linspace(0.0, 1.0, n + 1):
                    pts.append((x1 + t * (x2 - x1), y1 + t * (y2 - y1)))
                    pt_seg.append(idx)

        self.segments = np.array(segs)
        self.pt_seg = np.array(pt_seg)
        self.tree = cKDTree(np.array(pts))
        print(f"RoadMatcher: {len(self.segments)} segments, {len(pts)} sample points")

    def nearest_segment(self, x, y, k=40, heading_estimate=None,
                        snap_gate_m=20.0, max_angle_deg=60.0):
        _, pidx = self.tree.query([x, y], k=k)
        seg_ids = np.unique(self.pt_seg[np.atleast_1d(pidx)])

        best_dist, best = np.inf, None
        for sid in seg_ids:
            x1, y1, x2, y2 = self.segments[sid]
            dist, px, py = self._point_segment_distance(x, y, x1, y1, x2, y2)
            if dist > snap_gate_m:
                continue
            bearing = np.arctan2(x2 - x1, y2 - y1)
            if heading_estimate is not None:
                d = (bearing - heading_estimate + np.pi) % (2 * np.pi) - np.pi
                if abs(d) > np.pi / 2:          # OSM edge may be stored reversed
                    bearing = (bearing + np.pi) % (2 * np.pi) - np.pi
                d = (bearing - heading_estimate + np.pi) % (2 * np.pi) - np.pi
                if abs(d) > np.radians(max_angle_deg):
                    continue
            if dist < best_dist:
                best_dist, best = dist, (px, py, bearing)
        if best is None:
            return None
        return best[0], best[1], best[2], best_dist

    @staticmethod
    def _point_segment_distance(px, py, x1, y1, x2, y2):
        dx, dy = x2 - x1, y2 - y1
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / L2))
        qx, qy = x1 + t * dx, y1 + t * dy
        return np.hypot(px - qx, py - qy), qx, qy


def dead_reckon_blackout(pred_local, gyro_rate, start_pos, start_heading,
                         start, n_windows, matcher=None,
                         disp_scale=DISP_SCALE, dt=0.1, stride_samples=10,
                         heading_true=None, predict_fn=None, v0_start=None,
                         v0_true=None, use_true_v0=False,
                         speed_reset_interval=None, v0_ema_alpha=0.25,
                         v0_noise_std=0.0, rng=None, use_zupt=False,
                         acc_feature=None, gyr_feature=None,
                         acc_threshold=None, gyr_threshold=None):
    """
    Simulate a GPS blackout over windows [start, start+n_windows).
    Starts from a KNOWN position/heading (what the app has from GPS just
    before the blackout).
      matcher=None        -> pure gyro heading
      matcher=RoadMatcher -> gyro heading + map-matching snaps
      heading_true given  -> ceiling: ground-truth heading every window
      predict_fn given    -> callback(index, v0) returning (local_dxdy, v_end);
                            v_end is chained into the next window's v0 unless
                            use_true_v0 is enabled
      speed_reset_interval -> reset v0 from v0_true every N windows
      v0_ema_alpha       -> smooth chained v_end with prior v0
      v0_noise_std       -> Gaussian noise std added to the model's v0 input
      use_zupt           -> set the next v0 to zero when stop features pass
                            both thresholds
    Returns (final_xy, matched_fraction).
    """
    pos = np.array(start_pos, dtype=float)
    heading = float(start_heading)
    if predict_fn is not None and v0_start is None:
        raise ValueError("v0_start is required when predict_fn is provided")
    if (speed_reset_interval is not None or use_true_v0) and v0_true is None:
        raise ValueError("v0_true is required for true-speed resets")
    if not 0.0 <= v0_ema_alpha <= 1.0:
        raise ValueError("v0_ema_alpha must be between 0 and 1")
    if use_zupt:
        if acc_feature is None or gyr_feature is None:
            raise ValueError("acc_feature and gyr_feature are required for ZUPT")
        if acc_threshold is None or gyr_threshold is None:
            raise ValueError("acc_threshold and gyr_threshold are required for ZUPT")
    if rng is None:
        rng = np.random.default_rng()
    v0 = float(v0_start[start] if np.ndim(v0_start) else v0_start) if predict_fn else None
    matched = 0
    for i in range(start, start + n_windows):
        if heading_true is not None:
            heading = float(heading_true[i])
        if predict_fn is not None:
            should_reset = (speed_reset_interval is not None and
                            (i - start) % speed_reset_interval == 0)
            if (use_true_v0 or should_reset) and v0_true is not None:
                v0 = float(v0_true[i])
            model_v0 = v0 + rng.normal(0.0, v0_noise_std) if v0_noise_std else v0
            disp, v_end = predict_fn(i, model_v0)
            if not use_true_v0:
                smoothed_vend = (v0_ema_alpha * v_end +
                                 (1.0 - v0_ema_alpha) * v0)
                is_stop = (use_zupt and
                           acc_feature[i] < acc_threshold and
                           gyr_feature[i] < gyr_threshold)
                v0 = 0.0 if is_stop else smoothed_vend
            dxl, dyl = np.asarray(disp, dtype=float) * disp_scale
        else:
            dxl, dyl = pred_local[i] * disp_scale
        pos[0] += dxl * np.cos(heading) + dyl * np.sin(heading)
        pos[1] += -dxl * np.sin(heading) + dyl * np.cos(heading)

        if matcher is not None:
            m = matcher.nearest_segment(pos[0], pos[1], heading_estimate=heading)
            if m is not None:
                pos[0], pos[1], heading, _ = m
                matched += 1
        if heading_true is None:
            heading += np.sum(gyro_rate[i, :stride_samples]) * dt
    return pos, matched / max(1, n_windows)


def integrate_trajectory_with_map_matching(pred_local, gyro_yaw, heading_start_true,
                                            road_matcher, dt=0.1, stride_samples=10,
                                            disp_scale=DISP_SCALE):
    """Full-trip version (kept for the old eval). Now applies disp_scale."""
    n = pred_local.shape[0]
    traj = np.zeros((n, 2), dtype=np.float32)
    heading = float(heading_start_true[0])
    cx = cy = 0.0
    flags = []
    for i in range(n):
        dxl, dyl = pred_local[i] * disp_scale
        cx += dxl * np.cos(heading) + dyl * np.sin(heading)
        cy += -dxl * np.sin(heading) + dyl * np.cos(heading)
        m = road_matcher.nearest_segment(cx, cy, heading_estimate=heading)
        if m is not None:
            cx, cy, heading, _ = m
        traj[i] = [cx, cy]
        flags.append(m is not None)
        heading += np.sum(gyro_yaw[i, :stride_samples]) * dt
    f = np.array(flags)
    runs, pos = [], 0
    for v, g in itertools.groupby(f):
        L = len(list(g))
        if not v:
            runs.append((pos, L))
        pos += L
    if runs:
        s, L = max(runs, key=lambda r: r[1])
        print(f"Longest unmatched streak: {L} windows, starting at window {s}")
    return traj
