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

MAIN_HIGHWAYS = {
    "motorway", "motorway_link", "trunk", "trunk_link",
    "primary", "primary_link", "secondary", "secondary_link",
    "tertiary", "tertiary_link",
}


def is_main_road(hw):
    if isinstance(hw, (list, tuple, set)):
        return any(h in MAIN_HIGHWAYS for h in hw)
    return hw in MAIN_HIGHWAYS


CACHE_DIR = Path("preprocessing/output/osm_cache")
GRAPH_PATH = CACHE_DIR / "vta1a_roads.graphml"
ORIGIN_PATH = CACHE_DIR / "vta1a_origin.json"

DISP_SCALE = 10.0 / 40.0   # stride / window length
SAMPLE_SPACING_M = 10.0


def circular_mean(angles):
    """Return the wrap-aware mean of angles in radians."""
    angles = np.asarray(angles, dtype=float)
    if angles.size == 0:
        raise ValueError("circular_mean requires at least one angle")
    return float(np.arctan2(np.mean(np.sin(angles)),
                            np.mean(np.cos(angles))))


def yaw_rate_from_accel(accel_lateral, speed, min_speed=2.0,
                        max_yaw_rate=1.5):
    """Estimate yaw rate from lateral acceleration and speed, when reliable."""
    if speed < min_speed:
        return None
    return float(np.clip(accel_lateral / speed,
                         -max_yaw_rate, max_yaw_rate))


# Chosen from the Vta1a 95th-percentile signal distribution.
SHARP_TURN_GYRO_THRESH = 0.14
SHARP_TURN_ACCEL_THRESH = 0.93


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
            if not is_main_road(data.get("highway")):
                continue
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
        # build segment adjacency (segments sharing an endpoint = adjacent)
        from collections import defaultdict
        endpoint_map = defaultdict(list)
        for sid, (x1, y1, x2, y2) in enumerate(segs):
            endpoint_map[(round(x1, 1), round(y1, 1))].append(sid)
            endpoint_map[(round(x2, 1), round(y2, 1))].append(sid)
        self.adjacent = defaultdict(set)
        for pts_ in endpoint_map.values():
            for a in pts_:
                for b in pts_:
                    if a != b:
                        self.adjacent[a].add(b)
        self.last_seg = None
        self.recent_bearings = []  # last few matched segment bearings
        self.tree = cKDTree(np.array(pts))
        print(f"RoadMatcher: {len(self.segments)} segments, {len(pts)} sample points")

    HEADING_W = 0.4        # metres of penalty per degree of disagreement
    AMBIG_MARGIN = 6.0     # score gap below which two branches count as a tie
    AMBIG_ANGLE_DEG = 15.0 # branches closer than this are the same road

    def nearest_segment(self, x, y, k=40, heading_estimate=None,
                        snap_gate_m=None, max_angle_deg=60.0):
        gate = self.snap_gate_m if snap_gate_m is None else snap_gate_m
        _, pidx = self.tree.query([x, y], k=k)
        seg_ids = np.unique(self.pt_seg[np.atleast_1d(pidx)])

        cands = []
        for sid in seg_ids:
            x1, y1, x2, y2 = self.segments[sid]
            dist, px, py = self._point_segment_distance(x, y, x1, y1, x2, y2)
            if dist > gate:
                continue
            bearing = np.arctan2(x2 - x1, y2 - y1)
            ang_deg = 0.0
            if heading_estimate is not None:
                d = (bearing - heading_estimate + np.pi) % (2 * np.pi) - np.pi
                if abs(d) > np.pi / 2:
                    bearing = (bearing + np.pi) % (2 * np.pi) - np.pi
                d = (bearing - heading_estimate + np.pi) % (2 * np.pi) - np.pi
                if abs(d) > np.radians(max_angle_deg):
                    continue
                ang_deg = abs(np.degrees(d))
            cands.append((dist + self.HEADING_W * ang_deg,
                          dist, px, py, bearing))

        if not cands:
            return None
        cands.sort(key=lambda c: c[0])
        best = cands[0]
        bearing_out = best[4]

        if heading_estimate is not None:
            for c in cands[1:]:
                if c[0] - best[0] >= self.AMBIG_MARGIN:
                    break
                dd = abs((c[4] - best[4] + np.pi) % (2 * np.pi) - np.pi)
                if dd > np.radians(self.AMBIG_ANGLE_DEG):
                    bearing_out = heading_estimate
                    break
        return best[2], best[3], bearing_out, best[1]

    def top_k_segments(self, x, y, k=3, heading_estimate=None,
                       snap_gate_m=None):
        """Return up to k candidates as (dist, px, py, bearing, segment_id)."""
        gate = self.snap_gate_m if snap_gate_m is None else snap_gate_m
        _, pidx = self.tree.query([x, y], k=40)
        seg_ids = np.unique(self.pt_seg[np.atleast_1d(pidx)])
        candidates = []
        for sid in seg_ids:
            x1, y1, x2, y2 = self.segments[sid]
            dist, px, py = self._point_segment_distance(x, y, x1, y1, x2, y2)
            if dist > gate:
                continue
            bearing = np.arctan2(x2 - x1, y2 - y1)
            ang_deg = 0.0
            if heading_estimate is not None:
                d = (bearing - heading_estimate + np.pi) % (2 * np.pi) - np.pi
                if abs(d) > np.pi / 2:
                    bearing = (bearing + np.pi) % (2 * np.pi) - np.pi
                d = (bearing - heading_estimate + np.pi) % (2 * np.pi) - np.pi
                if abs(d) > np.radians(60.0):
                    continue
                ang_deg = abs(np.degrees(d))
            candidates.append((dist + self.HEADING_W * ang_deg,
                               dist, px, py, bearing, sid))
        candidates.sort(key=lambda c: c[0])
        return [(c[1], c[2], c[3], c[4], c[5]) for c in candidates[:k]]

    @staticmethod
    def _point_segment_distance(px, py, x1, y1, x2, y2):
        dx, dy = x2 - x1, y2 - y1
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / L2))
        qx, qy = x1 + t * dx, y1 + t * dy
        return np.hypot(px - qx, py - qy), qx, qy


def viterbi_lite_match(matcher, positions, heading_estimates, k=3,
                       lookback=3, w_dist=1.0, w_bearing_deg=1.0,
                       w_switch_penalty=15.0, previous_segment_id=None):
    """Choose the current map candidate using a short rolling DP window."""
    if not positions:
        return None
    start = max(0, len(positions) - lookback)
    step_candidates = []
    for i in range(start, len(positions)):
        cands = matcher.top_k_segments(
            positions[i][0], positions[i][1], k=k,
            heading_estimate=heading_estimates[i]
        )
        step_candidates.append(cands if cands else [None])

    first_costs = []
    for cand in step_candidates[0]:
        cost = cand[0] * w_dist if cand is not None else 50.0
        if (cand is not None and previous_segment_id is not None and
                cand[4] != previous_segment_id):
            cost += w_switch_penalty
        first_costs.append(cost)
    dp = [first_costs]
    backptr = [[None] * len(step_candidates[0])]
    for i in range(1, len(step_candidates)):
        dp.append([])
        backptr.append([])
        for j, cand in enumerate(step_candidates[i]):
            best_score, best_prev = np.inf, None
            cand_bearing = cand[3] if cand is not None else None
            for pj, prev_cand in enumerate(step_candidates[i - 1]):
                prev_bearing = prev_cand[3] if prev_cand is not None else None
                transition_penalty = 0.0
                if cand_bearing is not None and prev_bearing is not None:
                    db = abs((cand_bearing - prev_bearing + np.pi) %
                             (2 * np.pi) - np.pi)
                    transition_penalty = np.degrees(db) * w_bearing_deg
                switch_penalty = 0.0
                if cand is not None and prev_cand is not None:
                    if cand[4] != prev_cand[4]:
                        switch_penalty = w_switch_penalty
                score = dp[i - 1][pj] + transition_penalty + switch_penalty
                if score < best_score:
                    best_score, best_prev = score, pj
            step_cost = cand[0] * w_dist if cand is not None else 50.0
            dp[i].append(best_score + step_cost)
            backptr[i].append(best_prev)

    best_j = int(np.argmin(dp[-1]))
    return step_candidates[-1][best_j]


def dead_reckon_blackout(pred_local, gyro_rate, start_pos, start_heading,
                         start, n_windows, matcher=None,
                         disp_scale=DISP_SCALE, dt=0.1, stride_samples=10,
                         heading_true=None, predict_fn=None, v0_start=None,
                         v0_true=None, use_true_v0=False,
                         speed_reset_interval=None, v0_ema_alpha=0.25,
                         v0_noise_std=0.0, rng=None, use_zupt=False,
                         acc_feature=None, gyr_feature=None,
                         acc_threshold=None, gyr_threshold=None,
                         low_speed_thresh=3.0, yaw_deadband=0.15,
                         bootstrap_window_count=24,
                         return_trace=False, use_viterbi_lite=False,
                         viterbi_k=3, viterbi_lookback=3,
                         viterbi_w_dist=1.0, viterbi_w_bearing_deg=1.0,
                         viterbi_w_switch_penalty=15.0,
                         accel_lateral=None, use_accel_yaw_fusion=False,
                         accel_min_speed=2.0, accel_max_yaw_rate=1.5,
                         accel_turn_scale=0.3, accel_max_weight=0.5,
                         accel_yaw_scale=1.0):
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
      low_speed_thresh   -> suppress gyro heading updates when the true speed
                            is below this threshold (m/s), treating yaw as
                            bias/noise during crawling starts
      bootstrap_window_count -> number of initial windows eligible for the
                                trip-start low-speed artifact exception
      use_viterbi_lite -> use the optional rolling top-k map candidate scorer
                          when matcher is provided
      accel_lateral -> lateral acceleration samples aligned with gyro_rate
      use_accel_yaw_fusion -> blend accel-derived yaw during stronger turns
    Returns (final_xy, matched_fraction), or additionally the per-window
    position/heading trace when return_trace=True.
    """
    pos = np.array(start_pos, dtype=float)
    heading = float(start_heading)
    if predict_fn is not None and v0_start is None:
        raise ValueError("v0_start is required when predict_fn is provided")
    if (speed_reset_interval is not None or use_true_v0) and v0_true is None:
        raise ValueError("v0_true is required for true-speed resets")
    if low_speed_thresh < 0.0:
        raise ValueError("low_speed_thresh must be non-negative")
    if bootstrap_window_count < 0:
        raise ValueError("bootstrap_window_count must be non-negative")
    if use_accel_yaw_fusion and accel_lateral is None:
        raise ValueError("accel_lateral is required for accel yaw fusion")
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
    trace = [] if return_trace else None
    recent_positions = []
    recent_headings = []
    last_segment_id = None
    turn_hist = []
    uturn_cooldown = 0
    if matcher is not None:
        matcher.last_seg = None
        matcher.recent_bearings = []
    for i in range(start, start + n_windows):
        if uturn_cooldown > 0:
            uturn_cooldown -= 1
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
            if use_viterbi_lite:
                recent_positions.append(pos.copy())
                recent_headings.append(heading)
                recent_positions = recent_positions[-viterbi_lookback:]
                recent_headings = recent_headings[-viterbi_lookback:]
                m = viterbi_lite_match(
                    matcher, recent_positions, recent_headings,
                    k=viterbi_k, lookback=viterbi_lookback,
                    w_dist=viterbi_w_dist,
                    w_bearing_deg=viterbi_w_bearing_deg,
                    w_switch_penalty=viterbi_w_switch_penalty,
                    previous_segment_id=last_segment_id,
                )
            else:
                m = matcher.nearest_segment(pos[0], pos[1], heading_estimate=heading)
            if m is not None:
                if use_viterbi_lite:
                    _, pos[0], pos[1], matched_heading, _ = m
                    if uturn_cooldown == 0:
                        heading = matched_heading
                    last_segment_id = m[4]
                else:
                    pos[0], pos[1], matched_heading, _ = m
                    if uturn_cooldown == 0:
                        heading = matched_heading
                matched += 1
        if heading_true is None:
            if use_accel_yaw_fusion:
                gyro_yaw_rate = np.mean(gyro_rate[i, :stride_samples])
                accel_lat_mean = np.mean(accel_lateral[i, :stride_samples])
                accel_yaw_rate = yaw_rate_from_accel(
                    accel_lat_mean,
                    float(v0_true[i]) if v0_true is not None else 0.0,
                    min_speed=accel_min_speed,
                    max_yaw_rate=accel_max_yaw_rate,
                )
                if accel_yaw_rate is not None:
                    accel_yaw_rate *= accel_yaw_scale
                    accel_yaw_rate = float(np.clip(
                        accel_yaw_rate, -accel_max_yaw_rate,
                        accel_max_yaw_rate
                    ))
                    is_sharp_turn = (
                        abs(gyro_yaw_rate) > SHARP_TURN_GYRO_THRESH and
                        abs(accel_lat_mean) > SHARP_TURN_ACCEL_THRESH
                    )
                    if is_sharp_turn:
                        w_accel = 0.3
                        yaw_rate = ((1.0 - w_accel) * gyro_yaw_rate +
                                    w_accel * accel_yaw_rate)
                    else:
                        yaw_rate = gyro_yaw_rate
                else:
                    yaw_rate = gyro_yaw_rate
                heading += yaw_rate * stride_samples * dt
            else:
                r = gyro_rate[i, :stride_samples].copy()
                if (v0_true is not None and
                        float(v0_true[i]) < low_speed_thresh):
                    r[np.abs(r) < yaw_deadband] = 0.0
                heading += np.sum(r) * dt
            turn_hist.append(heading)
            uturn = (len(turn_hist) > 10 and
                     abs(turn_hist[-1] - turn_hist[-11]) > np.radians(150))
            if uturn:
                uturn_cooldown = 5
        if return_trace:
            trace.append([pos.copy(), heading])
    result = (pos, matched / max(1, n_windows))
    if return_trace:
        result += (np.array([x[0] for x in trace]),
                   np.array([x[1] for x in trace]))
    return result


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
