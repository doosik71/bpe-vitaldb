"""
Build data/vitaldb/index.csv by scanning track headers of all .vital files.

Skips cases already present in the CSV — safe to resume after interruption.
Each entry is flushed immediately, so progress is never lost on Ctrl+C.

Usage:
    uv run python scripts/build-vitaldb-index.py [--data-dir data/vitaldb]
"""

import argparse
import csv
from pathlib import Path

from tqdm import tqdm
from vitaldb.utils import VitalFile


def list_vital_files(data_dir: Path) -> list[Path]:
    return sorted(
        data_dir.glob("*.vital"),
        key=lambda p: int(p.stem) if p.stem.isdigit() else 0,
    )


def load_cached_ids(index_path: Path) -> set[int]:
    """Return set of caseids already recorded in index.csv."""
    cached: set[int] = set()
    if not index_path.exists():
        return cached
    try:
        with index_path.open(newline="") as f:
            for row in csv.reader(f):
                if row and row[0].strip().lstrip("-").isdigit():
                    cached.add(int(row[0]))
    except Exception:
        pass
    return cached


def scan_track_info(path: Path) -> tuple[int, int]:
    """Read track headers; return (ppg_len, abp_len) — 0 means absent."""
    try:
        vf = VitalFile(str(path), header_only=True)
        names = set(vf.get_track_names())

        def _nrecs(name: str) -> int:
            if name not in names:
                return 0
            trk = vf.trks.get(name)
            if trk is None:
                return 0
            n = len(trk.recs) if hasattr(trk, "recs") else 0
            return n if n > 0 else 1

        return _nrecs("SNUADC/PLETH"), _nrecs("SNUADC/ART")
    except Exception:
        return 0, 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/vitaldb"),
        help="Directory containing .vital files (default: data/vitaldb)",
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    index_path = data_dir / "index.csv"

    files = list_vital_files(data_dir)
    if not files:
        print(f"No .vital files found in {data_dir.resolve()}")
        return

    cached = load_cached_ids(index_path)
    to_scan = [f for f in files if not (f.stem.isdigit() and int(f.stem) in cached)]

    print(f"Total: {len(files)}  Cached: {len(cached)}  To scan: {len(to_scan)}")
    if not to_scan:
        print("All cases already indexed.")
        return

    n_done = 0
    try:
        with index_path.open("a", newline="") as out_f:
            writer = csv.writer(out_f)
            for path in tqdm(to_scan, unit="case"):
                caseid = int(path.stem) if path.stem.isdigit() else 0
                ppg_len, abp_len = scan_track_info(path)
                writer.writerow([caseid, ppg_len, abp_len])
                out_f.flush()
                n_done += 1
    except KeyboardInterrupt:
        print(f"\nInterrupted after {n_done}/{len(to_scan)} cases — resume is safe.")
        return

    print(f"Done. {n_done} new entries written to {index_path}")


if __name__ == "__main__":
    main()
