"""
preprocessing/utils.py

Low-level, trip-agnostic helpers used by every pipeline step:
file discovery, encoding-fallback CSV loading, and column matching.
"""

import os
import glob
import pandas as pd


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

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
