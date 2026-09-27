"""Create an interactive OpenStreetMap overlay for blackout trajectories.

The output compares:
  * true path
  * tracked/reference path (model displacement + true heading)
  * gyro-only path (model displacement + integrated gyro heading)
  * map-matching path (true heading + road snapping)
  * gyro + map-matching path

Run from the project root:
    python evaluation/plot_blackout_paths_osm.py

Output:
    preprocessing/output/diagnostics/Vta1a_blackout_paths_osm.html
"""

import argparse
import json
import sys
from pathlib import Path

import folium
import numpy as np
import osmnx as ox
import pandas as pd
import pyproj
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "evaluation"))
sys.path.append(str(ROOT))

from eval_blackout import DRNet, GYRO_W
from preprocessing.map_matching import RoadMatcher, dead_reckon_blackout


TEST_PATH = ROOT / "preprocessing/output/test_splits/Vta1a_test_split.npz"
MODEL_PATH = ROOT / "preprocessing/output/models/dr_model_v0noise.pt"
GRAPH_PATH = ROOT / "preprocessing/output/osm_cache/vta1a_roads.graphml"
ORIGIN_PATH = ROOT / "preprocessing/output/osm_cache/vta1a_origin.json"
OUTPUT_PATH = ROOT / "preprocessing/output/diagnostics/Vta1a_blackout_paths_osm.html"
def make_predict_window(X, device):
    """Create the same per-window predictor used by eval_blackout."""
    model = DRNet(in_channels=X.shape[2]).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1).to(device)

    def predict_window(index, v0):
        with torch.no_grad():
            v0_tensor = torch.tensor([v0], dtype=torch.float32, device=device)
            pred = model(X_tensor[index:index + 1], v0_tensor)[0].cpu().numpy()
        return pred[:2], float(pred[2])

    return predict_window


def integrate_path(pred_local, gyro_rate, heading_start, matcher=None,
                   use_true_heading=False, v0_true=None,
                   low_speed_thresh=3.0, bootstrap_window_count=24):
    """Integrate one full path through dead_reckon_blackout."""
    kwargs = dict(
        matcher=matcher,
        predict_fn=pred_local,
        v0_start=v0_true,
        v0_true=v0_true,
        low_speed_thresh=low_speed_thresh,
        bootstrap_window_count=bootstrap_window_count,
        return_trace=True,
    )
    if use_true_heading:
        kwargs["heading_true"] = heading_start
    _, matched, path, _ = dead_reckon_blackout(
        None, gyro_rate, np.zeros(2), heading_start[0], 0,
        len(gyro_rate), **kwargs
    )
    return path, matched


def relative_xy_to_latlon(path):
    """Convert the cached OSM projected coordinates back to WGS84."""
    with ORIGIN_PATH.open() as f:
        origin = json.load(f)

    graph = ox.load_graphml(GRAPH_PATH)
    graph_projected = ox.project_graph(graph)
    projected_crs = graph_projected.graph["crs"]
    to_projected = pyproj.Transformer.from_crs(
        "EPSG:4326", projected_crs, always_xy=True
    )
    to_wgs84 = pyproj.Transformer.from_crs(
        projected_crs, "EPSG:4326", always_xy=True
    )
    origin_e, origin_n = to_projected.transform(
        origin["origin_lon"], origin["origin_lat"]
    )
    lon, lat = to_wgs84.transform(
        path[:, 0] + origin_e, path[:, 1] + origin_n
    )
    return list(zip(lat.tolist(), lon.tolist()))


def load_true_gps_coords(trip, n_windows):
    """Load the true GPS route at the 1-second window output timestamps."""
    path = ROOT / f"preprocessing/output/01_aligned/V-{trip}_aligned.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing aligned GPS file: {path}")

    df = pd.read_csv(path)
    lat = df["Latitude (degrees)"].to_numpy(dtype=float)
    lon = df["Longitude (degrees)"].to_numpy(dtype=float)

    # Windows advance by 10 samples (1 second). Include the actual origin,
    # then the GPS position reached after each model output window.
    sample_indices = np.minimum(np.arange(n_windows) * 10 + 10, len(df) - 1)
    indices = np.concatenate(([0], sample_indices))
    return list(zip(lat[indices].tolist(), lon[indices].tolist()))


def add_path(feature_group, coords, name, color, weight=4, opacity=0.85,
             dash_array=None):
    folium.PolyLine(
        coords,
        color=color,
        weight=weight,
        opacity=opacity,
        dash_array=dash_array,
        tooltip=name,
    ).add_to(feature_group)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trip", default="Vta1a", help="processed trip name, e.g. Vfa01")
    args = parser.parse_args()
    trip = args.trip

    global TEST_PATH, GRAPH_PATH, ORIGIN_PATH, OUTPUT_PATH
    TEST_PATH = ROOT / f"preprocessing/output/06_windows/{trip}_windows.npz"
    GRAPH_PATH = ROOT / f"preprocessing/output/osm_cache/{trip.lower()}_roads.graphml"
    ORIGIN_PATH = ROOT / f"preprocessing/output/osm_cache/{trip.lower()}_origin.json"
    OUTPUT_PATH = ROOT / f"preprocessing/output/diagnostics/{trip}_blackout_paths_osm.html"

    for required in (TEST_PATH, MODEL_PATH, GRAPH_PATH, ORIGIN_PATH):
        if not required.exists():
            raise FileNotFoundError(
                f"Missing {required}. Prepare the trip and fetch its OSM road cache first."
            )

    data = np.load(TEST_PATH)
    X = data["X"]
    # The prepared Vta1a test split uses `Y`; regular trip window files use
    # `y_local`. Support both formats so --trip works across datasets.
    heading_start = data["heading_start"]
    v_start = data["v_start"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predict_window = make_predict_window(X, device)
    gyro_rate = X[:, :, 3:6] @ GYRO_W

    print("Loading OSM road network...")
    matcher_true = RoadMatcher(
        graph_path=GRAPH_PATH, origin_path=ORIGIN_PATH, snap_gate_m=20.0
    )
    matcher_gyro = RoadMatcher(
        graph_path=GRAPH_PATH, origin_path=ORIGIN_PATH, snap_gate_m=20.0
    )

    tracked_path, _ = integrate_path(
        predict_window, gyro_rate, heading_start,
        use_true_heading=True, v0_true=v_start
    )
    gyro_path, _ = integrate_path(
        predict_window, gyro_rate, heading_start, v0_true=v_start,
        low_speed_thresh=3.0, bootstrap_window_count=24
    )
    map_path, map_fraction = integrate_path(
        predict_window, gyro_rate, heading_start,
        matcher=matcher_true, use_true_heading=True, v0_true=v_start
    )
    gyro_map_path, gyro_map_fraction = integrate_path(
        predict_window, gyro_rate, heading_start, matcher=matcher_gyro,
        v0_true=v_start, low_speed_thresh=3.0, bootstrap_window_count=24
    )

    # Use raw aligned GPS for the truth line. Summing overlapping window labels
    # is suitable for model targets but is not an exact GPS trajectory.
    true_coords = load_true_gps_coords(trip, len(X))
    tracked_coords = relative_xy_to_latlon(
        np.vstack([np.zeros(2), tracked_path])
    )
    gyro_coords = relative_xy_to_latlon(np.vstack([np.zeros(2), gyro_path]))
    map_coords = relative_xy_to_latlon(np.vstack([np.zeros(2), map_path]))
    gyro_map_coords = relative_xy_to_latlon(
        np.vstack([np.zeros(2), gyro_map_path])
    )

    center = true_coords[len(true_coords) // 2]
    osm = folium.Map(location=center, zoom_start=13, tiles="OpenStreetMap")

    layers = [
        ("True path", true_coords, "black", 5, None),
        ("Tracked/reference (GT heading)", tracked_coords, "blue", 4, None),
        ("Gyro only", gyro_coords, "red", 4, None),
        ("Map matching (GT heading)", map_coords, "purple", 3, "8 5"),
        ("Gyro + map matching", gyro_map_coords, "green", 4, None),
    ]
    for name, coords, color, weight, dash in layers:
        group = folium.FeatureGroup(name=name, show=True)
        add_path(group, coords, name, color, weight=weight, dash_array=dash)
        group.add_to(osm)

    folium.Marker(true_coords[0], popup="Start", icon=folium.Icon(color="green")).add_to(osm)
    folium.Marker(true_coords[-1], popup="True path end", icon=folium.Icon(color="red")).add_to(osm)
    folium.LayerControl(collapsed=False).add_to(osm)

    legend = (
        '<div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999; '
        'background: white; padding: 10px; border: 2px solid grey; font-size: 13px;">'
        '<b>Blackout path comparison</b><br>'
        '<span style="color:black">━━</span> True path<br>'
        '<span style="color:blue">━━</span> Tracked/reference (GT heading)<br>'
        '<span style="color:red">━━</span> Gyro only<br>'
        '<span style="color:purple">╌╌</span> Map matching (GT heading)<br>'
        '<span style="color:green">━━</span> Gyro + map matching</div>'
    )
    osm.get_root().html.add_child(folium.Element(legend))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    osm.save(OUTPUT_PATH)

    print(f"Saved interactive map: {OUTPUT_PATH}")
    print("True path source: aligned GPS trajectory")
    print(f"Map matching coverage: {100 * map_fraction:.1f}%")
    print(f"Gyro + map matching coverage: {100 * gyro_map_fraction:.1f}%")


if __name__ == "__main__":
    main()
