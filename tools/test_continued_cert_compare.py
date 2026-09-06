#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tiny retained-record sabotage tests; no native execution or measurements."""
from pathlib import Path
import tempfile
import unittest

from continued_cert_compare import ARMS, KNN_FIXTURES, knn_records


class KnnRecordTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cells = [f"ADVERSARIAL_CELL {d} {p} {m} {i} 0 {i % 10}"
                      for d, p, m in sorted(KNN_FIXTURES) for i in range(170)]
        for arm in ARMS:
            self.write(arm)

    def write(self, arm, cells=None, flags=None):
        a, b = flags if flags is not None else ARMS[arm]
        lines = [f"ADVERSARIAL_FLAGS IDENTICAL {a} {b}", *(self.cells if cells is None else cells),
                 "KNN ADVERSARIAL PASS cases 32 selected_pairs 5440"]
        (self.root / f"knn-{arm}.log").write_text("\n".join(lines) + "\n")

    def test_complete_four_arm_records(self):
        rows, digests = knn_records(self.root)
        self.assertEqual(len(rows), 5440)
        self.assertEqual(len(set(digests.values())), 1)

    def test_wrong_effective_flags_fail(self):
        self.write("both", flags=(0, 0))
        with self.assertRaises(AssertionError):
            knn_records(self.root)

    def test_one_bit_difference_fails(self):
        cells = self.cells.copy()
        cells[0] = cells[0].rsplit(" 0 ", 1)[0] + " 1 0"
        self.write("transpose", cells=cells)
        with self.assertRaises(AssertionError):
            knn_records(self.root)

    def test_duplicate_or_wrong_fixture_fails_even_at_correct_count(self):
        for replacement in (self.cells[1], self.cells[0].replace("CELL 1 ", "CELL 99 ", 1)):
            with self.subTest(replacement=replacement):
                cells = self.cells.copy()
                cells[0] = replacement
                for arm in ARMS:
                    self.write(arm, cells=cells)
                with self.assertRaises(AssertionError):
                    knn_records(self.root)


if __name__ == "__main__":
    unittest.main()
