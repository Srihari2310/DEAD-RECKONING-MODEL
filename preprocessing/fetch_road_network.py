"""
One-time fetch of the OSM road network covering Vta1a's route (plus buffer).
Also saves the trip's first GPS point (lat/lon) so map_matching.py can align
the projected graph into the SAME local ENU frame as x_cum/y_cum
(origin = first row of the aligned V-file, east-positive X, north-positive Y).
"""

import argparse
import osmnx as ox
import pandas as pd
import json
from pathlib import Path

CACHE_DIR = Path("preprocessing/output/osm_cache")
BUFFER_DEG = 0.01


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trip", default="Vta1a", help="aligned trip name, e.g. Vfa01")
    args = parser.parse_args()
    trip = args.trip
    graph_path = CACHE_DIR / f"{trip.lower()}_roads.graphml"
    origin_path = CACHE_DIR / f"{trip.lower()}_origin.json"
    v_aligned_path = Path(f"preprocessing/output/01_aligned/V-{trip}_aligned.csv")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(v_aligned_path)
    lat_col = "Latitude (degrees)"
    lon_col = "Longitude (degrees)"

    # The trip's origin -- MUST match what derive_labels() implicitly uses:
    # the very first row of the aligned V-file. x_cum/y_cum are computed by
    # cumsum(dx), so sample 0's true position is (0,0) even though it isn't
    # explicitly stored -- the first row IS the origin.
    origin_lat = float(df[lat_col].iloc[0])
    origin_lon = float(df[lon_col].iloc[0])

    north = df[lat_col].max() + BUFFER_DEG
    south = df[lat_col].min() - BUFFER_DEG
    east = df[lon_col].max() + BUFFER_DEG
    west = df[lon_col].min() - BUFFER_DEG

    print(f"Origin (trip start): lat={origin_lat:.6f} lon={origin_lon:.6f}")
    print(f"Bounding box: N={north:.5f} S={south:.5f} E={east:.5f} W={west:.5f}")

    print("Fetching road network from OSM (drive network)...")
    G = ox.graph_from_bbox((west, south, east, north), network_type="all")    
    print(f"Fetched graph: {len(G.nodes)} nodes, {len(G.edges)} edges")

    ox.save_graphml(G, graph_path)

    with open(origin_path, "w") as f:
        json.dump({"origin_lat": origin_lat, "origin_lon": origin_lon}, f)

    print(f"Saved graph: {graph_path}")
    print(f"Saved origin: {origin_path}")


if __name__ == "__main__":
    main()
