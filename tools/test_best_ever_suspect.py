"""board_best_ever SUSPECT mark and paper_compare BEST-EVER-SUSPECT (no network)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import board_best_ever as BE  # noqa: E402
import paper_compare as PC  # noqa: E402

P = {"n_neighbors": 5}


def ent(ms, params=P):
    return {"ms": ms, "params": params, "hash": "h", "device": "NVIDIA L40S", "date": "2026-01-01"}


class Suspect(unittest.TestCase):
    def test_far_below_median_is_suspect(self):
        tl = [ent(0.164), ent(40.0), ent(42.0), ent(45.0)]
        self.assertIn("below 5%", BE.suspect_reason(tl[0], tl))

    def test_plausible_best(self):
        tl = [ent(30.0), ent(40.0), ent(42.0)]
        self.assertIsNone(BE.suspect_reason(tl[0], tl))

    def test_missing_params_is_suspect(self):
        tl = [ent(30.0, None), ent(40.0)]
        self.assertEqual(BE.suspect_reason(tl[0], tl), "record lacks params")

    def test_compare_flags_suspect_not_regression(self):
        tl = [ent(0.164), ent(40.0), ent(42.0)]
        best_rows = {"k": {"best": dict(tl[0], suspect=BE.suspect_reason(tl[0], tl)), "best_by_device": {},
                           "timeline": tl}}
        row = {"key": "k", "new_ms": 41.0, "flags": []}
        PC.attach_best_ever(row, dict(ent(41.0), device="NVIDIA L40S"), best_rows)
        self.assertIn("BEST-EVER-SUSPECT", row["flags"])
        self.assertNotIn("BEST-EVER-REGRESSION", row["flags"])
        self.assertEqual(row["best_ever_ms"], 0.164)      # kept, never dropped

    def test_compare_plausible_regression(self):
        tl = [ent(30.0), ent(40.0)]
        best_rows = {"k": {"best": dict(tl[0], suspect=None), "best_by_device": {}, "timeline": tl}}
        row = {"key": "k", "new_ms": 41.0, "flags": []}
        PC.attach_best_ever(row, ent(41.0), best_rows)
        self.assertIn("BEST-EVER-REGRESSION", row["flags"])


if __name__ == "__main__":
    unittest.main()
