"""
Batch-run the full preprocessing pipeline (run_pipeline.py) across many trips.
Generic — pass any trip names, not tied to any one series (Vtb, Vta, Vw, etc).

Usage:
    python preprocessing/batch_process.py Vtb02 Vtb03 Vtb04 Vtb05 Vtb06 Vtb07 Vtb08 Vtb09 Vtb10 Vtb11
    python preprocessing/batch_process.py --file trips_to_run.txt
"""
import sys
import argparse
import subprocess
import json
import re
from pathlib import Path
from datetime import datetime

try:
    from preprocessing.utils import normalize_trip
except ModuleNotFoundError:  # direct execution: python preprocessing/batch_process.py ...
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from preprocessing.utils import normalize_trip

LOG_PATH = Path("preprocessing/output/diagnostics/batch_log.json")
# Same trustworthiness bar used to catch Y1: weak corr + boundary-pinned lag = don't trust
MIN_CORR = 0.5
MIN_MAX_LAG_MARGIN = 0.9  # if |detected_lag| > 90% of max_lag search window, treat as boundary-pinned


def discover_trips(root="D:/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset", series=None):
    """Scan dataset folder and return trip names."""
    root = Path(root)
    trips = []
    for series_dir in root.iterdir():
        if not series_dir.is_dir():
            continue
        if series and not any(series_dir.name.startswith(s) for s in series):
            continue
        for trip_dir in series_dir.iterdir():
            if trip_dir.is_dir():
                trips.append(normalize_trip(trip_dir.name))
    return sorted(set(trips))


def append_to_data_paths(trip_names, train_script="training/train_model.py"):
    """Append new window paths without duplicating entries already in DATA_PATHS."""
    path = Path(train_script)
    text = path.read_text()
    match = re.search(r"DATA_PATHS\s*=\s*\[(.*?)\]", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Could not locate DATA_PATHS list in {train_script}")

    block = match.group(1)
    existing_paths = set(re.findall(r"['\"]([^'\"]+_windows\.npz)['\"]", block))
    print(f"Auto-append DATA_PATHS entries found: {len(existing_paths)}")

    new_paths = []
    for trip in trip_names:
        candidate = f"preprocessing/output/06_windows/{trip}_windows.npz"
        already_present = candidate in existing_paths
        print(f"  {trip}: status=OK candidate={candidate} regex_match={already_present}")
        if not already_present:
            new_paths.append(candidate)

    if new_paths:
        insertion = "".join(f'    "{candidate}",\n' for candidate in new_paths)
        text = text[:match.start(1)] + insertion + text[match.start(1):]
        path.write_text(text)

    print(f"Appended {len(new_paths)} trips to {train_script}")


def run_trip(trip_name: str) -> dict:
    """Run run_pipeline.py for one trip, capture stdout, parse key metrics."""
    result = {"trip": trip_name, "status": None, "corr": None, "lag_rows": None,
              "yaw_offset": None, "x_shape": None, "note": ""}

    proc = subprocess.run(
        [sys.executable, "preprocessing/run_pipeline.py", trip_name],
        capture_output=True, text=True
    )
    stdout = proc.stdout
    stderr = proc.stderr

    if proc.returncode != 0:
        result["status"] = "CRASHED"
        result["note"] = stderr.strip().splitlines()[-1] if stderr.strip() else "unknown error"
        return result

    # Parse the printed lines run_pipeline.py already emits (format matches v9-v18 sessions)
    for line in stdout.splitlines():
        line_low = line.lower()
        if "detected lag" in line_low:
            try:
                # e.g. "[Vtb02] Detected lag: 9 rows (0.9s), corr=0.951"
                lag_part = line.split("lag:")[1]
                rows = int(lag_part.split("rows")[0].strip())
                corr = float(line.split("corr=")[1].strip())
                result["lag_rows"] = rows
                result["corr"] = corr
            except Exception:
                pass
        if "yaw offset" in line_low:
            try:
                result["yaw_offset"] = float(line.split(":")[1].replace("deg", "").strip())
            except Exception:
                pass
        if "x shape" in line_low:
            result["x_shape"] = line.split(":")[1].strip()

    # Trust check — same logic that flagged Y1
    if result["corr"] is None:
        result["status"] = "UNKNOWN"
        result["note"] = "Could not parse lag/corr from output — inspect manually"
    elif result["corr"] < MIN_CORR:
        result["status"] = "UNTRUSTED"
        result["note"] = f"Weak correlation ({result['corr']}) — same red flag as Y1, do not add to training blindly"
    else:
        result["status"] = "OK"

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trips", nargs="*", help="Trip names to process")
    parser.add_argument("--file", help="Text file with one trip name per line")
    parser.add_argument("--series", help="Comma-separated series prefixes, e.g. Vtb or Vtb,Vw")
    parser.add_argument(
        "--root",
        default="D:/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset",
        help="Dataset root used by --series discovery",
    )
    parser.add_argument("--exclude", default="", help="Comma-separated trip names to skip")
    parser.add_argument(
        "--auto-append",
        action="store_true",
        help="Append OK trips to train_model.py DATA_PATHS",
    )
    args = parser.parse_args()

    trip_list = list(args.trips)
    if args.file:
        trip_list += [l.strip() for l in Path(args.file).read_text().splitlines() if l.strip()]
    if args.series:
        trip_list += discover_trips(args.root, [s.strip() for s in args.series.split(",") if s.strip()])

    exclude = set(t.strip() for t in args.exclude.split(",") if t.strip())
    trip_list = sorted(set(t for t in trip_list if t not in exclude))

    if not trip_list:
        print("No trips given. Pass trip names, --file, or --series")
        return

    print(f"Batch processing {len(trip_list)} trips: {trip_list}\n")

    results = []
    for trip in trip_list:
        print(f"--- {trip} ---")
        r = run_trip(trip)
        results.append(r)
        print(f"  status={r['status']}  corr={r['corr']}  lag_rows={r['lag_rows']}  "
              f"yaw={r['yaw_offset']}  x_shape={r['x_shape']}")
        if r["note"]:
            print(f"  note: {r['note']}")
        print()

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_entry = {"timestamp": datetime.now().isoformat(), "results": results}
    existing = json.loads(LOG_PATH.read_text()) if LOG_PATH.exists() else []
    existing.append(log_entry)
    LOG_PATH.write_text(json.dumps(existing, indent=2))

    # Summary
    ok = [r["trip"] for r in results if r["status"] == "OK"]
    untrusted = [r["trip"] for r in results if r["status"] == "UNTRUSTED"]
    crashed = [r["trip"] for r in results if r["status"] == "CRASHED"]
    unknown = [r["trip"] for r in results if r["status"] == "UNKNOWN"]

    print("=" * 50)
    print(f"OK ({len(ok)}): {ok}")
    print(f"UNTRUSTED — do not train on these ({len(untrusted)}): {untrusted}")
    print(f"CRASHED ({len(crashed)}): {crashed}")
    print(f"UNKNOWN / inspect manually ({len(unknown)}): {unknown}")
    print(f"\nFull log: {LOG_PATH}")
    print("\nNext: visually spot-check OK trips with plot_trip_on_osm.py before adding to DATA_PATHS.")
    if ok:
        print("\nWindow paths to add to train_model.py's DATA_PATHS:")
        for t in ok:
            print(f'    "preprocessing/output/06_windows/{t}_windows.npz",')
        if args.auto_append:
            append_to_data_paths(ok)


if __name__ == "__main__":
    main()
