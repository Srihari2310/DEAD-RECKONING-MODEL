"""
preprocessing/step4_run.py

Thin runner for Step 4 (event detection: idling / harsh braking /
sharp turns) on any trip that has already been through Steps 1-3.

Usage:
    python preprocessing/step4_run.py Vta1a
    python preprocessing/step4_run.py M
"""

import sys
from common import detect_events

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python step4_run.py <trip_name>")
        sys.exit(1)

    trip_name = sys.argv[1]
    result = detect_events(trip_name)
    print("\nDone.")
