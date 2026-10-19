"""CPU-only checks for raw LAS intensity audit completeness and decoding."""

import importlib.util
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

DELIVERY = Path(__file__).resolve().parent
SCRIPT = DELIVERY / "audit_dales_intensity.py"
if not SCRIPT.exists():
    SCRIPT = DELIVERY.parent / "scripts" / "audit_dales_intensity.py"
SPEC = importlib.util.spec_from_file_location("intensity_audit", SCRIPT)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def write_las(path, values, point_format=1):
    header = bytearray(227)
    header[:4] = b"LASF"
    header[24:26] = bytes((1, 2))
    struct.pack_into("<I", header, 96, 227)
    header[104] = point_format
    struct.pack_into("<H", header, 105, 28)
    struct.pack_into("<I", header, 107, len(values))
    records = bytearray(28 * len(values))
    for index, value in enumerate(values):
        struct.pack_into("<H", records, index * 28 + 12, value)
    path.write_bytes(header + records)


class IntensityAuditTest(unittest.TestCase):
    def cli(self, *paths):
        result = subprocess.run(
            [sys.executable, "-B", str(SCRIPT), *(str(p) for p in paths)],
            capture_output=True,
            text=True,
            check=False,
        )
        rows = [json.loads(line) for line in result.stdout.splitlines()]
        return result.returncode, rows[-1]

    def test_raw_uint16_values_across_chunk_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.las"
            write_las(path, [0, 7, 65535, 0, 3])
            row = AUDIT.inspect_file(path, chunk_points=2)
            self.assertEqual((row["points"], row["nonzero_intensity_points"]), (5, 3))
            self.assertEqual((row["intensity_min"], row["intensity_max"]), (0, 65535))

    def test_uppercase_file_discovery_and_duplicate_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.LAS"
            write_las(path, [0, 0])
            code, summary = self.cli(directory, path)
            self.assertEqual(code, 0)
            self.assertEqual(summary["files_fully_checked"], 1)
            self.assertTrue(summary["all_intensity_zero"])

    def test_compressed_file_is_an_incomplete_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            write_las(Path(directory) / "sample.laz", [0], point_format=129)
            code, summary = self.cli(directory)
            self.assertEqual(code, 2)
            self.assertFalse(summary["complete_for_supplied_roots"])
            self.assertFalse(summary["all_intensity_zero"])

    def test_truncated_records_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.las"
            write_las(path, [0, 0])
            path.write_bytes(path.read_bytes()[:-1])
            code, summary = self.cli(path)
            self.assertEqual(code, 2)
            self.assertEqual(summary["errors"], 1)
            self.assertFalse(summary["all_intensity_zero"])

    def test_empty_file_does_not_confirm_zero_intensity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.las"
            write_las(path, [])
            code, summary = self.cli(path)
            self.assertEqual(code, 0)
            self.assertEqual(summary["points_fully_checked"], 0)
            self.assertFalse(summary["all_intensity_zero"])

    def test_missing_root_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            code, summary = self.cli(Path(directory) / "missing")
            self.assertEqual(code, 2)
            self.assertFalse(summary["complete_for_supplied_roots"])


if __name__ == "__main__":
    unittest.main()
