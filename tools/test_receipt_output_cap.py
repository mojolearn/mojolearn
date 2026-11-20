"""scored_receipt: outputs above MOJOLEARN_RECEIPT_OUTPUT_MAX_BYTES are hashed, not retained."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_board_state import canonical_hash, scored_receipt  # noqa: E402


class ReceiptOutputCap(unittest.TestCase):
    def run_one(self, n_floats):
        with tempfile.TemporaryDirectory() as d:
            env = {"MOJOLEARN_MEASUREMENT_RECEIPTS": str(Path(d) / "r.json"),
                   "MOJOLEARN_RECEIPT_OUTPUT_MAX_BYTES": str(1 << 20)}
            outputs = {"y": np.arange(n_floats, dtype=np.float64)}
            with mock.patch.dict(os.environ, env):
                r = scored_receipt(outputs)
            files = list((Path(d) / "scored").rglob("outputs.npz"))
            return r, files, canonical_hash(outputs)

    def test_over_limit_omitted_hash_kept(self):
        r, files, h = self.run_one(200_000)          # 1.6 MB
        self.assertEqual(files, [])
        self.assertTrue(r["retained_output"]["omitted"])
        self.assertEqual(r["retained_output"]["bytes"], 1_600_000)
        self.assertEqual(r["retained_output"]["limit"], 1 << 20)
        self.assertEqual(r["output_sha256"], h)

    def test_under_limit_retained(self):
        r, files, h = self.run_one(1_000)
        self.assertEqual(len(files), 1)
        self.assertNotIn("omitted", r["retained_output"])
        self.assertEqual(r["output_sha256"], h)


if __name__ == "__main__":
    unittest.main()
