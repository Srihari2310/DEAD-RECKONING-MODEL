"""
preprocessing/step5_run.py

Thin runner for Step 5 (displacement label derivation) on any trip.
Saves a trajectory plot to preprocessing/output/{trip}_trajectory.png
for visual comparison against the trip's real route thumbnail (.jpg).

Usage:
    python preprocessing/step5_run.py Vta1a
    python preprocessing/step5_run.py M
"""

import sys
import os
from common import derive_labels

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python step5_run.py <trip_name>")
        sys.exit(1)

    trip_name = sys.argv[1]
    plot_path = os.path.join("preprocessing", "output", f"{trip_name}_trajectory.png")
    result = derive_labels(trip_name, plot_path=plot_path)
    print("\nDone. Compare the saved plot against the trip's real route thumbnail (.jpg).")
