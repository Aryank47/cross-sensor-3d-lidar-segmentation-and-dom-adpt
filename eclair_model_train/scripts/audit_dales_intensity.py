"""Read all intensity records in uncompressed LAS 1.2 / point format 1 files.

No laspy, GPU, checkpoint access, network activity, or output-file writes.
Print JSON lines to stdout. Unsupported or unreadable files make the audit incomplete.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


def inspect_file(path: Path, chunk_points: int = 2_000_000) -> dict:
    before = path.stat()
    with path.open("rb") as handle:
        header = handle.read(227)
    if len(header) < 227 or header[:4] != b"LASF":
        raise ValueError("Not a complete LAS header")
    version = (header[24], header[25])
    raw_format = header[104]
    if version != (1, 2) or raw_format != 1:
        raise ValueError(
            f"Expected uncompressed LAS 1.2 point-format 1; got {version}, {raw_format}"
        )
    offset = struct.unpack_from("<I", header, 96)[0]
    record_bytes = struct.unpack_from("<H", header, 105)[0]
    points = struct.unpack_from("<I", header, 107)[0]
    if (
        offset < 227
        or record_bytes < 28
        or offset + points * record_bytes > before.st_size
    ):
        raise ValueError("Invalid or truncated point records")
    count_nonzero, minimum, maximum = 0, None, None
    if points:
        dtype = np.dtype(
            {
                "names": ["intensity"],
                "formats": ["<u2"],
                "offsets": [12],
                "itemsize": record_bytes,
            }
        )
        records = np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(points,))
        values = records["intensity"]
        for start in range(0, points, chunk_points):
            part = values[start : start + chunk_points]
            lo, hi = int(part.min()), int(part.max())
            minimum = lo if minimum is None else min(minimum, lo)
            maximum = hi if maximum is None else max(maximum, hi)
            count_nonzero += int(np.count_nonzero(part))
        del part, values, records
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("File changed during audit")
    return {
        "path": str(path.resolve()),
        "bytes": before.st_size,
        "mtime_ns": before.st_mtime_ns,
        "version": "1.2",
        "point_format": 1,
        "points": points,
        "intensity_min": minimum,
        "intensity_max": maximum,
        "nonzero_intensity_points": count_nonzero,
        "full_scan": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    args = parser.parse_args()
    seen, checked, total_points, total_nonzero, errors = set(), 0, 0, 0, 0
    for root in args.roots:
        if not root.exists():
            print(
                json.dumps({"error": "Path does not exist", "path": str(root)}),
                flush=True,
            )
            errors += 1
            continue
        paths = (
            [root]
            if root.is_file()
            else sorted(
                p
                for p in root.rglob("*")
                if p.is_file() and p.suffix.lower() in {".las", ".laz"}
            )
        )
        if not paths:
            print(
                json.dumps({"error": "No LAS/LAZ files found", "path": str(root)}),
                flush=True,
            )
            errors += 1
        for path in paths:
            try:
                stat = path.stat()
                identity = (stat.st_dev, stat.st_ino)
                if identity in seen:
                    continue
                seen.add(identity)
                row = inspect_file(path)
                checked += 1
                total_points += row["points"]
                total_nonzero += row["nonzero_intensity_points"]
                print(json.dumps(row), flush=True)
            except Exception as exc:
                errors += 1
                print(json.dumps({"path": str(path), "error": str(exc)}), flush=True)
    complete = errors == 0 and checked > 0
    print(
        json.dumps(
            {
                "summary": True,
                "files_fully_checked": checked,
                "points_fully_checked": total_points,
                "nonzero_intensity_points": total_nonzero,
                "errors": errors,
                "complete_for_supplied_roots": complete,
                "all_intensity_zero": complete
                and total_points > 0
                and total_nonzero == 0,
            }
        ),
        flush=True,
    )
    raise SystemExit(0 if complete else 2)


if __name__ == "__main__":
    main()
