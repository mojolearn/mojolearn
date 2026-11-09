#!/usr/bin/env python3
"""tools/test_measure_owed.py -- tools/measure_owed.py plan and report on in-memory fixtures (no box, no board run),
the committed owed file's shape, and tools/paper_compare.py's prior column."""
import json
import os
import sys
import tempfile
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import measure_owed as MO  # noqa: E402
import paper_compare as PC  # noqa: E402


def fake_planner():
    lanes = {"classical": ["kmeans"], "algos": ["gmm", "rfe"]}
    m = types.SimpleNamespace(FAMILIES=("classical", "algos"), DATASETS=("taxi", "istella"), SMOKE_ROWS=2000)
    m.family_lanes = lambda f: lanes[f]
    m.modes_for = lambda v, modes: ["identical"]
    m.gpu_opponents_only = lambda races, v: races

    def plan_races(vendor, modes, fams, lns, ds, rows, shape):
        tail = "rows=%s" % ("full" if rows is None else rows)
        return [{"id": "%s/%s/%s/%s" % (f, l, d, tail), "family": f, "lane": l, "dataset": d}
                for f in fams for l in lanes[f] if l in lns for d in ds]
    m.plan_races = plan_races
    return m


OWED = {"schema": MO.SCHEMA, "b_arms": {"b1": {"define": "X=1"}}, "rows": [
    {"id": "r-board", "kind": "board", "lane": ["kmeans", "rfe"], "dataset": "taxi", "rows": "full"},
    {"id": "r-ab", "kind": "ab", "lane": "gmm", "dataset": "*", "rows": "full", "b_arms": ["b1", "b2"]},
    {"id": "r-nb", "kind": "board", "lane": "kmeans", "dataset": "taxi", "rows": 300000},
    {"id": "r-gone", "kind": "board", "lane": "no-such-lane", "dataset": "*", "rows": "full"},
    {"id": "r-drv", "kind": "driver", "driver": "mlp_fit", "arms": ["candidate", "prior"]}]}
PAPER = {"race_ids": ["classical/kmeans/taxi/rows=full"],
         "bench_board_args": {"families": ["classical"], "lanes": ["kmeans"], "datasets": ["taxi"]}}


class Plan(unittest.TestCase):
    def test_units_skip_paper_races_and_record_missing_b(self):
        units, smoke, drivers, missing, rowmap, bargs = MO.plan(OWED, PAPER, fake_planner(), "nvidia", ["b1"])
        self.assertEqual(units[("candidate", "full")], ["algos/gmm/istella/rows=full", "algos/gmm/taxi/rows=full",
                                                        "algos/rfe/taxi/rows=full"])
        self.assertEqual(units[("prior", "full")], ["algos/rfe/taxi/rows=full"])
        self.assertEqual(units[("b1", "full")], ["algos/gmm/istella/rows=full", "algos/gmm/taxi/rows=full"])
        self.assertEqual(units[("candidate", "300000")], ["classical/kmeans/taxi/rows=300000"])
        self.assertNotIn("classical/kmeans/taxi/rows=2000", smoke)       # the paper smoke covers it
        self.assertIn("algos/rfe/taxi/rows=2000", smoke)
        self.assertIn("r-ab b2", missing)
        self.assertIn("r-gone no-such-lane", missing)
        self.assertEqual([d["id"] for d in drivers], ["r-drv"])
        self.assertIn("algos", bargs[bargs.index("--families") + 1])


class Report(unittest.TestCase):
    def test_prior_ratio_and_b_flags(self):
        cell = lambda ms: {"arm": "ours", "status": "ok", "median_ms": ms}  # noqa: E731
        with tempfile.TemporaryDirectory() as d:
            plan_dir, res = os.path.join(d, "plan"), os.path.join(d, "res")
            os.makedirs(plan_dir)
            json.dump({"r-board": ["algos/rfe/taxi/rows=full"], "r-ab": ["algos/gmm/taxi/rows=full"]},
                      open(os.path.join(plan_dir, "rowmap.json"), "w"))
            json.dump({}, open(os.path.join(plan_dir, "not-planned.json"), "w"))
            for sub, rid, ms in (("board", "algos/rfe/taxi/rows=full", 10.0), ("board", "algos/gmm/taxi/rows=full", 10.0),
                                 ("board-prior-full", "algos/rfe/taxi/rows=full", 5.0),
                                 ("board-b1-full", "algos/gmm/taxi/rows=full", 5.0)):
                os.makedirs(os.path.join(res, sub), exist_ok=True)
                p = os.path.join(res, sub, "board.json")
                doc = json.load(open(p)) if os.path.exists(p) else {"races": {}}
                doc["races"][rid] = {"cells": [cell(ms)]}
                json.dump(doc, open(p, "w"))
            doc = dict(OWED, rows=[r for r in OWED["rows"] if r["id"] in ("r-board", "r-ab")])
            rep = MO.report(doc, plan_dir, res, None, "nvidia")
            by = {r["row"]: r for r in rep["rows"]}
            self.assertIn("REGRESSION", by["r-board"]["flags"])
            self.assertAlmostEqual(by["r-board"]["ratio_vs_prior"], 2.0)
            self.assertIn("b1:B-FASTER", by["r-ab"]["flags"])
            self.assertIn("b2:B-NOT-RUN", by["r-ab"]["flags"])


class PriorColumn(unittest.TestCase):
    def test_recovered_and_regression_vs_prior(self):
        row = {"key": "k", "new_ms": 10.0, "flags": ["SAME"]}
        PC.attach_prior(row, {"k": {"new_ms": 50.0, "flags": ["REGRESSION"]}})
        self.assertIn("RECOVERED", row["flags"])
        self.assertIn("FASTER-VS-PRIOR", row["flags"])
        row = {"key": "k", "new_ms": 60.0, "flags": ["REGRESSION"]}
        PC.attach_prior(row, {"k": {"new_ms": 50.0, "flags": ["REGRESSION"]}})
        self.assertIn("STILL-REGRESSED", row["flags"])
        self.assertIn("REGRESSION-VS-PRIOR", row["flags"])

    def test_candidate_label(self):
        board = {"box": {"mojolearn": {"wheel": {"source": "candidate", "candidate": {
            "source_commit": "a" * 40, "version": "0.8.67", "vendor": "nvidia",
            "files": [{"distribution": "mojolearn-core", "file": "c.whl", "sha256": "0" * 64}]}}}}}
        cand = PC.board_candidate([board])
        self.assertEqual(PC.new_label(cand, ["0.8.67"]), "candidate aaaaaaaaaaaa (0.8.67)")


class CommittedOwed(unittest.TestCase):
    def test_owed_file_shape(self):
        doc = json.load(open(os.path.join(PC.OUT_DIR, "owed-pre0867.json")))
        self.assertEqual(doc["schema"], MO.SCHEMA)
        ids = [r["id"] for r in doc["rows"]]
        self.assertEqual(len(ids), len(set(ids)))
        for r in doc["rows"]:
            self.assertIn(r["kind"], ("board", "ab", "driver"))
            self.assertTrue(r.get("decides"))
            for b in r.get("b_arms") or []:
                self.assertIn(b, doc["b_arms"])


if __name__ == "__main__":
    unittest.main()
