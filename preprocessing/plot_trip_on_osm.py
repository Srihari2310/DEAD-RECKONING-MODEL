"""
preprocessing/plot_trip_on_osm.py

Plots any trip's real GPS trajectory (from V-{trip}_labels.csv, or
V-{trip}_aligned.csv if labels aren't present) directly on top of an
OpenStreetMap background, using folium. Apples-to-apples visual check
against the real route -- no rotation/scale ambiguity, since it's real
lat/lon on a real map.

Requires: pip install folium

Usage (run from project root):
    python preprocessing/plot_trip_on_osm.py <trip_name>

Example:
    python preprocessing/plot_trip_on_osm.py Y1

Output:
    preprocessing/output/diagnostics/{trip_name}_trajectory_osm.html
    (open this file in a browser -- interactive, pannable/zoomable map)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import folium
import osmnx as ox
from pyproj import Transformer

OUTPUT_DIR = "preprocessing/output/diagnostics"
LABELS_DIR = "preprocessing/output/05_labels"
ALIGNED_DIR = "preprocessing/output/01_aligned"
DIAGNOSTICS_DIR = "preprocessing/output/diagnostics"
OSM_CACHE_DIR = "preprocessing/output/osm_cache"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def find_col(columns, keywords):
    def tokenize(name):
        cleaned = name.replace("(", " ").replace(")", " ").replace(",", " ")
        return [t.strip().lower() for t in cleaned.split() if t.strip()]
    keywords_lower = [k.lower() for k in keywords]
    for col in columns:
        tokens = tokenize(col)
        if all(kw in tokens for kw in keywords_lower):
            return col
    return None


def load_trip_latlon(trip_name):
    labels_path = os.path.join(LABELS_DIR, f"V-{trip_name}_labels.csv")
    aligned_path = os.path.join(ALIGNED_DIR, f"V-{trip_name}_aligned.csv")

    path = labels_path if os.path.exists(labels_path) else aligned_path
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {labels_path} or {aligned_path}. "
            f"Run Step 1 (and ideally Step 5) for trip '{trip_name}' first."
        )

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    lat_col = find_col(df.columns, ["latitude"])
    lon_col = find_col(df.columns, ["longitude"])
    if lat_col is None or lon_col is None:
        raise ValueError(f"Could not find lat/lon columns in {path}: {list(df.columns)}")

    lat = df[lat_col].astype(float).tolist()
    lon = df[lon_col].astype(float).tolist()
    print(f"Loaded {len(lat)} GPS points from {path}")
    return lat, lon


def load_overlay_latlon(trip_name, overlay):
    if overlay != "map_matched":
        raise ValueError(f"Unsupported overlay '{overlay}'. Expected 'map_matched'.")

    trajectory_path = os.path.join(
        DIAGNOSTICS_DIR, f"matched_trajectory_{trip_name}.npy"
    )
    origin_path = os.path.join(OSM_CACHE_DIR, f"{trip_name.lower()}_origin.json")
    graph_path = os.path.join(OSM_CACHE_DIR, f"{trip_name.lower()}_roads.graphml")

    if not os.path.exists(trajectory_path):
        raise FileNotFoundError(
            f"Could not find {trajectory_path}. Run evaluation/eval_map_matching.py first."
        )
    if not os.path.exists(origin_path) or not os.path.exists(graph_path):
        raise FileNotFoundError(
            f"Could not find map-matching origin/graph for trip '{trip_name}'."
        )

    matched_xy = np.load(trajectory_path)
    with open(origin_path) as f:
        origin = json.load(f)

    graph = ox.load_graphml(graph_path)
    graph_projected = ox.project_graph(graph)
    projected_crs = graph_projected.graph["crs"]

    to_projected = Transformer.from_crs(
        "EPSG:4326", projected_crs, always_xy=True
    )
    to_wgs84 = Transformer.from_crs(
        projected_crs, "EPSG:4326", always_xy=True
    )
    origin_easting, origin_northing = to_projected.transform(
        origin["origin_lon"], origin["origin_lat"]
    )

    easting = matched_xy[:, 0] + origin_easting
    northing = matched_xy[:, 1] + origin_northing
    lon, lat = to_wgs84.transform(easting, northing)
    print(f"Loaded {len(matched_xy)} map-matched points from {trajectory_path}")
    return list(lat), list(lon)


def plot_on_osm(lat, lon, out_path, trip_name, overlay=None):
    center = [lat[len(lat) // 2], lon[len(lon) // 2]]

    m = folium.Map(location=center, zoom_start=12, tiles="OpenStreetMap")

    coords = list(zip(lat, lon))
    folium.PolyLine(coords, color="red", weight=3, opacity=0.8).add_to(m)

    if overlay == "map_matched":
        overlay_lat, overlay_lon = load_overlay_latlon(trip_name, overlay)
        folium.PolyLine(
            list(zip(overlay_lat, overlay_lon)),
            color="green", weight=3, opacity=0.8,
            tooltip="Map-matched (no GPS)",
        ).add_to(m)

    folium.Marker(
        [lat[0], lon[0]], popup=f"{trip_name} Start",
        icon=folium.Icon(color="green"),
    ).add_to(m)
    folium.Marker(
        [lat[-1], lon[-1]], popup=f"{trip_name} End",
        icon=folium.Icon(color="red"),
    ).add_to(m)

    m.save(out_path)
    print(f"Saved interactive map: {out_path}")
    print("Open this file in a browser to view/pan/zoom.")


def main():
    if len(sys.argv) not in (2, 4) or (len(sys.argv) == 4 and sys.argv[2] != "--overlay"):
        print("Usage: python preprocessing/plot_trip_on_osm.py <trip_name> [--overlay map_matched]")
        print("Example: python preprocessing/plot_trip_on_osm.py Y1")
        sys.exit(1)

    trip_name = sys.argv[1]
    overlay = sys.argv[3] if len(sys.argv) == 4 else None
    lat, lon = load_trip_latlon(trip_name)
    suffix = f"_{overlay}" if overlay else ""
    out_path = os.path.join(OUTPUT_DIR, f"{trip_name}_trajectory_osm{suffix}.html")
    plot_on_osm(lat, lon, out_path, trip_name, overlay=overlay)


if __name__ == "__main__":
    main()
