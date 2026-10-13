"""The per-lane summary of tools/changed_lanes_check.sh (Andrew, 2026-10-13):
one grep-sized line per changed lane from a `verify --json` report.

    python3 -m pytest -q tools/test_changed_lanes_summary.py
"""
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import changed_lanes_summary  # noqa: E402


class Summary(unittest.TestCase):
    def test_one_line_per_lane_with_the_divergent_hashes(self):
        report = dict(
            lane_accounting=dict(lanes={
                "ols": dict(state="VERIFIED"),
                "knn": dict(state="DIVERGENT"),
                "pca": dict(state="OWED", reason="no committed record carries a reference for a part of it")}),
            cells=[dict(lane="knn", fixture="base", part="train", state="DIVERGENT",
                        value="aaaaaaaaaaaaaaaaaaaa", reference="bbbbbbbbbbbbbbbbbbbb"),
                   dict(lane="knn", fixture="base", part="infer", state="DIVERGENT", value="c", reference="d")])
        lines = changed_lanes_summary.summarize(report, "cuda", ["ols", "knn", "pca", "gone"])
        self.assertEqual(len(lines), 5)
        self.assertEqual(lines[0], "CHANGED-LANE column=cuda lane=ols state=VERIFIED")
        self.assertIn("state=DIVERGENT cell=base/train got=aaaaaaaaaaaa ref=bbbbbbbbbbbb +1 more", lines[1])
        self.assertIn("state=NO-REFERENCE", lines[2])
        self.assertIn("state=NOT-IN-REPORT", lines[3])
        self.assertEqual(lines[4], "CHANGED-LANES column=cuda lanes=4 VERIFIED=1 DIVERGENT=1 NO-REFERENCE=1 OTHER=1")

    def test_an_unreadable_report_is_never_a_pass(self):
        with tempfile.TemporaryDirectory() as d:
            bad = pathlib.Path(d) / "r.json"
            bad.write_text("not json")
            self.assertEqual(changed_lanes_summary.main([str(bad), "--column", "cpu", "--lanes", "ols"]), 2)


if __name__ == "__main__":
    unittest.main()
