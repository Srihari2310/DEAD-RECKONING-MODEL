"""
preprocessing/step2_run.py

Thin runner for Step 2 (calibration: static leveling + dynamic yaw
alignment) on any trip that has already been through Step 1.

Usage:
    python preprocessing/step2_run.py Vta1a
    python preprocessing/step2_run.py M
"""

import sys
from common import calibrate_trip

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python step2_run.py <trip_name>")
        sys.exit(1)

    trip_name = sys.argv[1]
    result = calibrate_trip(trip_name)
    print("\nDone.")
