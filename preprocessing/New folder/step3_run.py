"""
preprocessing/step3_run.py

Thin runner for Step 3 (noise filtering) on any trip that has already
been through Steps 1-2.

Usage:
    python preprocessing/step3_run.py Vta1a
    python preprocessing/step3_run.py M
"""

import sys
from common import filter_trip

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python step3_run.py <trip_name>")
        sys.exit(1)

    trip_name = sys.argv[1]
    result = filter_trip(trip_name)
    print("\nDone.")
