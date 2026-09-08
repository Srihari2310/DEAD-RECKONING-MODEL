import sys
import os
import glob
import pandas as pd
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def find_v_file(trip_name):
    v_root = os.path.join(BASE_DIR, "V-Dataset")
    matches = glob.glob(os.path.join(v_root, "**", f"V-{trip_name}.*"), recursive=True) + \
              glob.glob(os.path.join(v_root, f"V-{trip_name}.*"))
    if not matches:
        raise FileNotFoundError(f"No V-file found for trip '{trip_name}' under {v_root}")
    return matches[0]

def check_jumps(trip_name, top_n=10):
    path = find_v_file(trip_name)
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    lat_col = [c for c in df.columns if "latitude" in c.lower()][0]
    lon_col = [c for c in df.columns if "longitude" in c.lower()][0]

    dlat = df[lat_col].diff() * 111000
    dlon = df[lon_col].diff() * 111000 * np.cos(np.radians(df[lat_col]))
    dist = np.sqrt(dlat**2 + dlon**2)

    print(f"[{trip_name}] Top {top_n} largest single-row GPS jumps (meters):")
    print(dist.nlargest(top_n))

    idx = dist.idxmax()
    print(f"\n[{trip_name}] Rows around the biggest jump (index {idx}):")
    print(df.loc[max(0, idx-3):idx+3, [lat_col, lon_col]])

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python preprocessing/check_gps_jumps.py <trip_name>")
        sys.exit(1)
    check_jumps(sys.argv[1])