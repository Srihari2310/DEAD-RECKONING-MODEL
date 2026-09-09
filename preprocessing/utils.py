"""
preprocessing/utils.py

Low-level, trip-agnostic helpers used by every pipeline step:
file discovery, encoding-fallback CSV loading, and column matching.
"""

import os
import glob
import re
from pathlib import Path
import pandas as pd


IOVNBD_ROOT = Path(r"D:\IO-VNBD\Synchronised V abd S datasets\Categorised IOVNB Dataset")


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def normalize_trip(name):
    """Normalize zero-padded trip names, e.g. Vtb01 -> Vtb1."""
    match = re.match(r"([A-Za-z]+)0*(\d+)([a-z]?)$", name)
    return f"{match.group(1)}{match.group(2)}{match.group(3)}" if match else name


def find_iovnbd_trip_files(trip_name, root=IOVNBD_ROOT):
    """Find S/V CSVs inside the matching IO-VNBD per-trip folder."""
    target_trip_name = normalize_trip(re.sub(r"^[Vv]-", "", trip_name))
    root = Path(root)
    if not root.exists():
        return None, None

    for series_dir in root.iterdir():
        if not series_dir.is_dir():
            continue
        for trip_dir in series_dir.iterdir():
            if not trip_dir.is_dir():
                continue
            folder_trip_name = re.sub(r"^[Vv]-", "", trip_dir.name)
            if normalize_trip(folder_trip_name).lower() != target_trip_name.lower():
                continue

            csv_files = list(trip_dir.glob("*.csv"))

            def choose(prefix):
                preferred = [f"{prefix}-{trip_name}.csv", f"{prefix}{trip_name}.csv"]
                for name in preferred:
                    for candidate in csv_files:
                        if candidate.name.lower() == name.lower():
                            return candidate
                return next(
                    (p for p in csv_files if p.name.lower().startswith(prefix.lower() + "-")),
                    None,
                )

            return choose("S"), choose("V")

    return None, None

def _find_trip_file(root_dir, trip_name, prefix_letter):
    """
    Search root_dir recursively for a file matching this trip.
    Handles naming variations seen so far:
        S-M.csv       (hyphen, trip M)
        SVta1a.csv    (no hyphen, trip Vta1a)
    Tries both '<prefix>-<trip>' and '<prefix><trip>' patterns,
    case-insensitive, .csv or .xlsx.
    """
    patterns = [
        f"{prefix_letter}-{trip_name}.csv",
        f"{prefix_letter}-{trip_name}.xlsx",
        f"{prefix_letter}{trip_name}.csv",
        f"{prefix_letter}{trip_name}.xlsx",
    ]
    for pattern in patterns:
        matches = glob.glob(os.path.join(root_dir, "**", pattern), recursive=True)
        if matches:
            return matches[0]
        # case-insensitive fallback
        all_files = glob.glob(os.path.join(root_dir, "**", "*.*"), recursive=True)
        for f in all_files:
            if os.path.basename(f).lower() == pattern.lower():
                return f
    return None


def _load_csv_with_fallback(path):
    """Try utf-8, then cp1252, then latin-1. Return (df, encoding_used)."""
    encodings_to_try = ["utf-8", "cp1252", "latin-1"]
    last_err = None
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(path, encoding=enc)
            return df, enc
        except UnicodeDecodeError as e:
            last_err = e
            continue
    raise UnicodeDecodeError(
        f"Could not decode {path} with any of {encodings_to_try}: {last_err}"
    )


# ---------------------------------------------------------------------------
# Column detection (bug-fixed: full-token match, not substring)
# ---------------------------------------------------------------------------

def find_col(columns, keywords):
    """
    Find the column whose name contains ALL given keywords as whole tokens.
    Fixes the bug where 'y' matched inside 'gravity' — keywords are matched
    against tokens split on spaces, parentheses, and common punctuation.
    """
    def tokenize(name):
        cleaned = name.replace("(", " ").replace(")", " ").replace(",", " ")
        return [t.strip().lower() for t in cleaned.split() if t.strip()]

    keywords_lower = [k.lower() for k in keywords]
    for col in columns:
        tokens = tokenize(col)
        if all(kw in tokens for kw in keywords_lower):
            return col
    return None
