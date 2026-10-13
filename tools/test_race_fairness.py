# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""tools/race_fairness.py: the race-fairness rule (AGENTS.md "Race fairness").

    python3 -m unittest discover -s tools -p test_race_fairness.py -v
"""
import importlib.util
import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("race_fairness", os.path.join(HERE, "race_fairness.py"))
RF = importlib.util.module_from_spec(spec)
spec.loader.exec_module(RF)


def race(lane, opps, params_check="MATCHED", ours_ms=5.0, ours_q=None):
    cells = [{"arm": "ours", "mode": "identical", "status": "ok", "median_ms": ours_ms, "quality": ours_q or {}}]
    for arm, ms, extra in opps:
        cells.append(dict({"arm": arm, "mode": "opponent", "device": "gpu", "status": "ok", "median_ms": ms}, **extra))
    return {"lane": lane, "params_check": params_check, "cells": cells, "infer_cells": []}


def board(**races):
    return {"config": {"modes": ["identical"]}, "races": races}


class RaceFairness(unittest.TestCase):
    def test_harness_arm_is_excluded_and_the_race_judged_on_the_rest(self):
        b = board(r=race("pca", [("torch-gpu", 1.0, {"library": "torch"}), ("cuml-gpu", 10.0, {"library": "cuml"})]))
        RF.apply(b, None, "2026-10-13")
        rr = b["races"]["r"]
        self.assertEqual(rr["opponents_excluded"]["arms"], ["torch-gpu"])
        self.assertEqual(rr["audit"]["identical/fit"]["opponent"], "cuml-gpu")

    def test_every_cell_from_a_slow_host_is_excluded_any_library(self):
        slow = {"library": "cuml", "copied_from": {"box": "cc560ebdaf91"}}
        b = board(n=race("rnn-clf", [("torch-eager-fp32", 9.0, dict(slow, library="torch"))]),
                  l=race("svd", [("cupy-gpu", 9.0, slow)]))
        RF.apply(b, None, "2026-10-13")
        self.assertEqual(b["races"]["n"]["opponents_excluded"]["arms"], ["torch-eager-fp32"])
        self.assertEqual(b["races"]["l"]["opponents_excluded"]["arms"], ["cupy-gpu"])
        self.assertNotIn("audit", b["races"]["n"])           # no valid opponent left: no row

    def test_a_slow_host_is_found_from_the_board(self):
        races = {}
        for i in range(6):
            r = race("lane%d" % i, [("torch-a", 20.0, {"copied_from": {"box": "hostS"}, "shape": "X 10x2"})])
            races["s%d" % i] = r
            races["f%d" % i] = race("lane%d" % i, [("torch-a", 10.0, {"copied_from": {"box": "hostF"}, "shape": "X 10x2"})])
        self.assertIn("hostS", RF.slow_hosts({"races": races}))
        self.assertNotIn("hostF", RF.slow_hosts({"races": races}))

    def test_iterations_differ_hides_and_ties_show(self):
        b = board(i=race("kmeans", [("cuml-gpu", 9.0, {"quality": {"n_iter": 21}})], ours_q={"n_iter": 33}),
                  t=race("ridge", [("cuml-gpu", 5.1, {})]))
        RF.apply(b, None, "2026-10-13")
        self.assertEqual(b["races"]["i"]["rerace"], "x")
        self.assertEqual(b["races"]["t"]["audit"]["identical/fit"]["verdict"], "TIE")
        self.assertNotIn("rerace", b["races"]["t"])

    def test_a_hand_verdict_survives_while_unchanged_and_rule_entries_are_recomputed(self):
        b = board(r=race("ridge", [("cuml-gpu", 10.0, {})], params_check="not checked"))
        rr = b["races"]["r"]
        rr["audit"] = {"identical/fit": {"verdict": "UNFAIR", "rerace": "x", "audit": "hand",
                                         "pairing": RF.pairing(rr, "identical", "fit")}}
        RF.apply(b, None, "2026-10-13")
        self.assertEqual((rr["audit"]["identical/fit"]["verdict"], rr["rerace"]), ("UNFAIR", "x"))
        rr["audit"]["identical/fit"] = {"verdict": "TIE", "rerace": "x", "rule": "old", "pairing": "stale"}
        RF.apply(b, None, "2026-10-13")
        self.assertEqual(rr["audit"]["identical/fit"]["verdict"], "UNCHECKED")
        self.assertNotIn("rerace", rr)


if __name__ == "__main__":
    unittest.main()
