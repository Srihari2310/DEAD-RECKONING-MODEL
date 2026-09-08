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
import pandas as pd
import folium

OUTPUT_DIR = "preprocessing/output/diagnostics"
LABELS_DIR = "preprocessing/output/05_labels"
ALIGNED_DIR = "preprocessing/output/01_aligned"
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


def plot_on_osm(lat, lon, out_path, trip_name):
    center = [lat[len(lat) // 2], lon[len(lon) // 2]]

    m = folium.Map(location=center, zoom_start=12, tiles="OpenStreetMap")

    coords = list(zip(lat, lon))
    folium.PolyLine(coords, color="red", weight=3, opacity=0.8).add_to(m)

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
    if len(sys.argv) != 2:
        print("Usage: python preprocessing/plot_trip_on_osm.py <trip_name>")
        print("Example: python preprocessing/plot_trip_on_osm.py Y1")
        sys.exit(1)

    trip_name = sys.argv[1]
    lat, lon = load_trip_latlon(trip_name)
    out_path = os.path.join(OUTPUT_DIR, f"{trip_name}_trajectory_osm.html")
    plot_on_osm(lat, lon, out_path, trip_name)


if __name__ == "__main__":
    main()
