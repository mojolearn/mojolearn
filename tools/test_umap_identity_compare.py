#!/usr/bin/env python3
"""Negative controls for the UMAP bit-capture comparator."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from umap_identity_compare import COUNTS, PROFILE


class IdentityComparatorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.left = Path(self.tmp.name) / "left.log"
        self.right = Path(self.tmp.name) / "right.log"
        self.lines = [PROFILE]
        self.lines += [f"UMAP_BITS {stage} {i} 0"
                       for stage, count in COUNTS.items() for i in range(count)]
        self.lines += ["UMAP identity fixture PASS"]
        self.left.write_text("\n".join(self.lines) + "\n")

    def compare(self, lines, expected):
        self.right.write_text("\n".join(lines) + "\n")
        result = subprocess.run(
            [sys.executable, str(Path(__file__).with_name("umap_identity_compare.py")),
             str(self.left), str(self.right)], capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)

    def test_complete_equal(self):
        self.compare(self.lines, 0)

    def test_one_bit_mismatch(self):
        lines = self.lines.copy()
        lines[1] = "UMAP_BITS input 0 1"
        self.compare(lines, 1)

    def test_signed_zero_is_not_equal(self):
        lines = self.lines.copy()
        lines[1] = "UMAP_BITS input 0 2147483648"
        self.compare(lines, 1)

    def test_missing_cell(self):
        self.compare(self.lines[:1] + self.lines[2:], 2)

    def test_duplicate_cell(self):
        self.compare(self.lines + [self.lines[1]], 2)

    def test_incomplete_run(self):
        self.compare(self.lines[:-1], 2)

    def test_nonfinite(self):
        for value in (0x7F800000, 0xFF800000, 0x7FC00001):
            with self.subTest(bits=value):
                lines = self.lines.copy()
                lines[1] = f"UMAP_BITS input 0 {value}"
                self.compare(lines, 2)

    def test_unknown_stage(self):
        self.compare(self.lines + ["UMAP_BITS unknown 0 0"], 2)


if __name__ == "__main__":
    unittest.main()
