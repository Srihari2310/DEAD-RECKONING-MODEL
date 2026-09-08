"""
preprocessing/common.py

Backward-compatible re-export layer. The actual implementations now live
in separate files (utils.py, diagnostics.py, alignment.py, calibration.py,
filtering.py, events.py, labels.py, windowing.py) for readability, but
every existing import you've been using still works unchanged:

    from preprocessing.common import load_and_verify_trip, calibrate_trip, \
        filter_trip, detect_events, derive_labels, window_trip

No other code needs to change because of this split.
"""

from .utils import _find_trip_file, _load_csv_with_fallback, find_col
from .diagnostics import check_gps_freezing
from .alignment import detect_lag, align_shift, load_and_verify_trip
from .calibration import calibrate_trip, _rotation_between_vectors, _rotation_about_z
from .filtering import filter_trip
from .events import detect_events
from .labels import derive_labels
from .windowing import window_trip

__all__ = [
    "find_col",
    "check_gps_freezing",
    "detect_lag",
    "align_shift",
    "load_and_verify_trip",
    "calibrate_trip",
    "filter_trip",
    "detect_events",
    "derive_labels",
    "window_trip",
]
