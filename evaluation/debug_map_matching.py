# preprocessing/debug_map_matching.py
import json
import numpy as np
import osmnx as ox
import pyproj

GRAPH_PATH = "preprocessing/output/osm_cache/vta1a_roads.graphml"
ORIGIN_PATH = "preprocessing/output/osm_cache/vta1a_origin.json"
V_ALIGNED_PATH = "preprocessing/output/01_aligned/V-Vta1a_aligned.csv"
BUFFER_DEG = 0.01

with open(ORIGIN_PATH) as f:
    origin = json.load(f)
origin_lat, origin_lon = origin["origin_lat"], origin["origin_lon"]

G = ox.load_graphml(GRAPH_PATH)
G_proj = ox.project_graph(G)
crs = G_proj.graph["crs"]
print("Projected CRS:", crs)

transformer = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
origin_easting, origin_northing = transformer.transform(origin_lon, origin_lat)
print(f"Origin projected: easting={origin_easting:.2f} northing={origin_northing:.2f}")

# Find the single nearest graph node to the origin point, print distance
min_dist = float("inf")
nearest_node = None
for n, data in G_proj.nodes(data=True):
    dx = data["x"] - origin_easting
    dy = data["y"] - origin_northing
    d = (dx**2 + dy**2) ** 0.5
    if d < min_dist:
        min_dist = d
        nearest_node = n

print(f"Nearest graph node to trip origin: {min_dist:.2f} m away (node {nearest_node})")

# Diagnostic: re-fetch a 2 km radius around the trip origin with a looser
# network filter. A much smaller distance would indicate that network_type
#="drive" was too strict, rather than a coordinate-frame problem.
print("Fetching diagnostic all-network graph within 2 km of trip origin...")
G2 = ox.graph_from_point((origin_lat, origin_lon), dist=2000, network_type="all")
G2_proj = ox.project_graph(G2, to_crs=crs)

min_dist2 = float("inf")
for n, data in G2_proj.nodes(data=True):
    dx = data["x"] - origin_easting
    dy = data["y"] - origin_northing
    d = (dx**2 + dy**2) ** 0.5
    if d < min_dist2:
        min_dist2 = d

print(f"Nearest node with network_type='all', 2km radius: {min_dist2:.2f} m")
