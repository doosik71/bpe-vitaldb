"""
VitalDB dataset downloader
Downloads .vital files from the Seoul National University Hospital open dataset.
Targets cases that contain PPG (PLETH) and invasive arterial blood pressure (ART) signals.

Usage:
    uv run python scripts/download-vitaldb.py [OPTIONS]

Options:
    --vitaldb-dir   Destination directory (default: data/vitaldb)
    --max-cases     Maximum number of cases to download (default: all)
    --start-case    First case ID to download (default: 1)
    --end-case      Last case ID to download (default: 6388)
    --workers       Parallel download workers (default: 4)
    --no-resume     Re-download files that already exist
    --filter-tracks Only download cases that have all target tracks (deprecated trks API)
"""

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

# VitalDB public dataset endpoint
API_URL = "https://api.vitaldb.net"
DATASET_VERSION = "1.0.1"
TOTAL_CASES = 6388

# Tracks required for PPG-based blood pressure estimation:
#   SNUADC/PLETH     - photoplethysmography waveform (500 Hz)
#   SNUADC/ART       - invasive radial arterial blood pressure waveform (500 Hz)
#   Solar8000/ART_*  - numeric SBP / DBP / MBP derived from the arterial line
TARGET_TRACKS = [
    "SNUADC/PLETH",
    "SNUADC/ART",
    "Solar8000/ART_SBP",
    "Solar8000/ART_DBP",
    "Solar8000/ART_MBP",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download VitalDB PPG and blood-pressure data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--vitaldb-dir",
        type=Path,
        default=Path("data/vitaldb"),
        help="Destination directory (default: data/vitaldb)",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Maximum number of cases to download (default: all)",
    )
    parser.add_argument(
        "--start-case",
        type=int,
        default=1,
        help="First case ID (default: 1)",
    )
    parser.add_argument(
        "--end-case",
        type=int,
        default=TOTAL_CASES,
        help=f"Last case ID (default: {TOTAL_CASES})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel download workers (default: 4)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-download files that already exist locally",
    )
    parser.add_argument(
        "--filter-tracks",
        action="store_true",
        help="Only download cases that have all target tracks (uses deprecated trks API)",
    )
    return parser.parse_args()


def fetch_cases_with_tracks(track_names: list[str]) -> list[int]:
    """Return case IDs that contain every track in track_names (uses the trks index)."""
    import warnings

    import pandas as pd

    log.info("Fetching track index from https://api.vitaldb.net/trks ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        df = pd.read_csv(f"{API_URL}/trks")

    sets = [
        set(df.loc[df["tname"].str.endswith(t.split("/")[-1]), "caseid"])
        for t in track_names
    ]
    caseids = sorted(set.intersection(*sets))
    log.info(f"Cases with PPG + blood-pressure tracks: {len(caseids)}")
    return caseids


def download_vital_file(caseid: int, output_dir: Path, resume: bool) -> tuple[int, bool, str]:
    """Download a single case .vital file.

    Returns:
        (caseid, success, message)
    """
    dest = output_dir / f"{caseid}.vital"

    if resume and dest.exists() and dest.stat().st_size > 0:
        return caseid, True, "skip"

    url = f"{API_URL}/{DATASET_VERSION}/{caseid}.vital"
    try:
        resp = requests.get(url, timeout=120, stream=True)
        if resp.status_code == 404:
            return caseid, False, "404 Not Found"
        resp.raise_for_status()

        tmp = dest.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
        tmp.rename(dest)
        return caseid, True, f"{dest.stat().st_size / 1024:.0f} KB"
    except requests.exceptions.Timeout:
        return caseid, False, "timeout"
    except Exception as e:
        return caseid, False, str(e)


def main():
    args = parse_args()

    output_dir: Path = args.vitaldb_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Output directory : {output_dir.resolve()}")
    log.info(f"Target tracks    : {', '.join(TARGET_TRACKS)}")

    # Build list of case IDs to download
    if args.filter_tracks:
        caseids = fetch_cases_with_tracks(TARGET_TRACKS)
        caseids = [c for c in caseids if args.start_case <= c <= args.end_case]
    else:
        caseids = list(range(args.start_case, args.end_case + 1))

    if args.max_cases:
        caseids = caseids[: args.max_cases]

    log.info(f"Cases to download: {len(caseids)} (workers: {args.workers})")

    resume = not args.no_resume
    skipped = 0
    success = 0
    failed = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_vital_file, cid, output_dir, resume): cid
            for cid in caseids
        }
        with tqdm(total=len(caseids),
                  unit="case",
                  desc="Downloading",
                  ascii=" -+=",
                  dynamic_ncols=True) as pbar:
            for future in as_completed(futures):
                caseid, ok, msg = future.result()
                if msg == "skip":
                    skipped += 1
                    pbar.set_postfix(skip=skipped, ok=success, fail=len(failed))
                elif ok:
                    success += 1
                    pbar.set_postfix(skip=skipped, ok=success, fail=len(failed))
                else:
                    failed.append((caseid, msg))
                    log.warning(f"Case {caseid} failed: {msg}")
                pbar.update(1)

    log.info("=" * 60)
    log.info(f"Done - success: {success}, skipped: {skipped}, failed: {len(failed)}")
    if failed:
        log.warning("Failed cases:")
        for cid, reason in failed[:20]:
            log.warning(f"  case {cid}: {reason}")
        if len(failed) > 20:
            log.warning(f"  ... and {len(failed) - 20} more")

    if failed:
        fail_log = output_dir / "failed_cases.txt"
        fail_log.write_text(
            "\n".join(f"{cid}\t{reason}" for cid, reason in failed), encoding="utf-8"
        )
        log.info(f"Failed case list saved: {fail_log}")


if __name__ == "__main__":
    main()
