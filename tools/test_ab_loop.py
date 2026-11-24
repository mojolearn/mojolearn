#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn
"""Tests of A/B LOOP MODE (tools/ab_loop.py, lane ab-loop-mode 2026-11-24): the table, the driver plan, the cell
record, the verdict, and the harness plumbing that carries them (bench_board loop_env, measure_owed base sharing,
bench_board_probe phase_split, targeted_rows summary, measure_candidate loop_full_arms). Pure Python, no mojolearn.

    python3 tools/test_ab_loop.py
"""
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "ops"))
import ab_loop as AL  # noqa: E402

PLAN = AL.race_plan("algos", "sgd-clf")


class Model(object):
    def __init__(self, loop=None, params=None):
        if loop is not None:
            self.fit_loop_ = loop
        self._p = params or {}

    def get_params(self):
        return dict(self._p)


def cell(ms, loop_ms=None, it=None, k=5, mode="capped", h="h1", quality=None):
    c = {"arm": "ours", "library": "mojolearn", "median_ms": ms, "status": "ok", "hash": h,
         "quality": quality or {"acc": 0.9}}
    if loop_ms is not None:
        c["loop_mode"] = {"mode": mode, "loop_ms": loop_ms, "iterations": it, "k": k,
                          "why": None if mode == "capped" else "lane not valid"}
    return c


class Table(unittest.TestCase):
    def test_every_row_is_well_formed(self):
        caps = AL.load_caps()
        self.assertTrue(caps)
        for (fam, lane), r in caps.items():
            if r["valid"]:
                self.assertTrue(r["param"] and r["cap"] >= 1, (fam, lane))
            self.assertIsInstance(r["pin"], dict)
            self.assertTrue(r["reason"], (fam, lane))

    def test_family_default_and_lane_row(self):
        self.assertFalse(AL.lookup("trees", "gbdt-lossguide")["valid"])
        self.assertTrue(AL.lookup("trees", "gbdt-symmetric")["valid"])
        self.assertEqual(AL.lookup("algos", "no-such-lane")["valid"], AL.lookup("algos", "*")["valid"])
        self.assertFalse(AL.lookup("nofamily", "x")["valid"])

    def test_every_family_has_a_default(self):
        for fam in ("trees", "classical", "classical2", "algos", "neural"):
            self.assertIn((fam, "*"), AL.load_caps())

    def test_conditions(self):
        self.assertEqual(AL.width_of("istella"), "wide")
        self.assertEqual(AL.width_of("taxi"), "narrow")
        self.assertEqual(AL.condition_of("trees/rf/istella/rows=full"), "istella")
        self.assertEqual(AL.condition_of("neural/mlp-fit/higgs/shape=small"), "small")

    def test_duplicate_row_refused(self):
        with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False) as fh:
            fh.write("family\tlane\tvalid\tcap_param\tcap\tpin\tunit\treason\n"
                     "a\tb\tyes\tn\t3\t{}\tx\ty\na\tb\tno\t-\t0\t{}\t-\tz\n")
        try:
            with self.assertRaises(ValueError):
                AL.load_caps(fh.name)
        finally:
            os.remove(fh.name)


class DriverSide(unittest.TestCase):
    def test_plan_modes(self):
        self.assertEqual(PLAN["mode"], "capped")
        self.assertEqual(AL.race_plan("algos", "sgd-clf", cap=3)["k"], 3)
        w = AL.race_plan("classical", "dbscan")
        self.assertEqual(w["mode"], "whole")
        self.assertIsNone(w["k"])

    def test_driver_plan_env(self):
        self.assertIsNone(AL.driver_plan(environ={}))
        env = {AL.ENV: json.dumps(PLAN)}
        self.assertEqual(AL.driver_plan("algos", "sgd-clf", environ=env)["k"], PLAN["k"])
        with self.assertRaises(ValueError):
            AL.driver_plan("algos", "sgd-reg", environ=env)

    def test_capped_params_copy_and_whole_untouched(self):
        p = {"alpha": 1.0, "max_iter": 1000}
        q = AL.capped_params(PLAN, p)
        self.assertEqual(q["max_iter"], PLAN["k"])
        self.assertEqual(p["max_iter"], 1000)
        for k, v in PLAN["pin"].items():
            self.assertEqual(q[k], v)
        self.assertEqual(AL.capped_params(AL.race_plan("classical", "dbscan"), p), p)
        self.assertEqual(AL.capped_params(None, p), p)

    def test_readback(self):
        eff = AL.capped_params(PLAN, {})
        self.assertTrue(AL.readback(PLAN, eff)["ok"])
        bad = dict(eff, max_iter=1000)
        rb = AL.readback(PLAN, bad)
        self.assertFalse(rb["ok"])
        self.assertEqual(rb["mismatch"], ["max_iter"])

    def test_read_fit_loop(self):
        self.assertEqual(AL.read_fit_loop(Model((5, 12.5))), {"status": "recorded", "iterations": 5, "loop_ms": 12.5})
        self.assertEqual(AL.read_fit_loop(Model())["status"], "unavailable")
        self.assertEqual(AL.read_fit_loop(Model((5, float("nan"))))["status"], "unavailable")

    def test_cell_record_capped(self):
        eff = AL.capped_params(PLAN, {})
        loops = [AL.read_fit_loop(Model((5, 10.0))), AL.read_fit_loop(Model((5, 12.0)))]
        rec = AL.cell_record(PLAN, eff, loops)
        self.assertEqual(rec["mode"], "capped")
        self.assertEqual(rec["iterations"], 5)
        self.assertEqual(rec["loop_ms"], 11.0)

    def test_cell_record_falls_back_to_whole_with_why(self):
        eff = AL.capped_params(PLAN, {})
        self.assertEqual(AL.cell_record(PLAN, eff, [AL.read_fit_loop(Model())])["mode"], "whole")
        self.assertIn("read-back", AL.cell_record(PLAN, dict(eff, max_iter=7),
                                                  [AL.read_fit_loop(Model((5, 1.0)))])["why"])
        mixed = AL.cell_record(PLAN, eff, [AL.read_fit_loop(Model((5, 1.0))), AL.read_fit_loop(Model((4, 1.0)))])
        self.assertEqual(mixed["mode"], "whole")
        w = AL.cell_record(AL.race_plan("classical", "dbscan"), {}, [])
        self.assertEqual(w["mode"], "whole")
        self.assertIn("not valid", w["why"])


class Verdict(unittest.TestCase):
    def test_pair_loop_basis_and_fixed_cost_flag(self):
        p = AL.pair(cell(100.0, 50.0, 5), cell(110.0, 40.0, 5))
        self.assertEqual(p["basis"], "loop")
        self.assertAlmostEqual(p["ratio"], 0.8)
        self.assertTrue(p["fixed_up"])        # fixed 50 -> 70 ms, beyond 5 % of the base's 100 ms
        self.assertTrue(p["same_hash"])

    def test_pair_whole_basis_when_iterations_differ_or_not_capped(self):
        p = AL.pair(cell(100.0, 50.0, 5), cell(90.0, 40.0, 4))
        self.assertEqual(p["basis"], "whole")
        self.assertAlmostEqual(p["ratio"], 0.9)
        self.assertIn("iterations differ", p["why"])
        q = AL.pair(cell(100.0), cell(80.0))
        self.assertEqual(q["basis"], "whole")

    def _v(self, specs, planned=None):
        cp = [("algos/sgd-clf/%s/rows=full" % ds, cell(100.0, 50.0, 5), cell(100.0, 50.0 * r, 5, h=h))
              for ds, r, h in specs]
        return AL.unit_verdict(cp, planned or [x[0] for x in cp])[0]

    def test_promote(self):
        v = self._v([("taxi", 0.8, "h1"), ("istella", 0.85, "h1")])
        self.assertEqual(v["verdict"], "PROMOTE")
        self.assertFalse(v["full_run"])

    def test_promote_with_different_hash_owes_one_full_run(self):
        v = self._v([("taxi", 0.8, "h2"), ("istella", 0.85, "h1")])
        self.assertEqual(v["verdict"], "PROMOTE")
        self.assertTrue(v["full_run"])
        self.assertEqual(v["full_run_races"], ["algos/sgd-clf/taxi/rows=full"])
        self.assertIn("FULL-RUN-OWED", AL.verdict_line("arm", "full", v))

    def test_split(self):
        v = self._v([("taxi", 0.5, "h1"), ("istella", 1.2, "h1")])
        self.assertEqual(v["verdict"], "SPLIT")
        self.assertFalse(v["full_run"])

    def test_not_faster(self):
        self.assertEqual(self._v([("taxi", 0.99, "h1"), ("istella", 1.01, "h1")])["verdict"], "NOT-FASTER")
        self.assertEqual(self._v([("taxi", 1.2, "h1"), ("istella", 1.3, "h1")])["verdict"], "NOT-FASTER")

    def test_incomplete_without_both_widths(self):
        v = self._v([("taxi", 0.8, "h1")], planned=["algos/sgd-clf/taxi/rows=full", "algos/sgd-clf/istella/rows=full"])
        self.assertEqual(v["verdict"], "INCOMPLETE")
        self.assertIn("wide", v["why"])

    def test_one_width_planned_is_named(self):
        v = self._v([("taxi", 0.8, "h1")])
        self.assertEqual(v["verdict"], "PROMOTE")
        self.assertTrue(v["one_width"])
        self.assertIn("only one width", v["why"])

    def test_neural_sizes(self):
        cp = [("neural/mlp-fit/higgs/shape=%s" % s, cell(100.0, 50.0, 2, k=2), cell(100.0, 40.0, 2, k=2))
              for s in ("full", "small")]
        self.assertEqual(AL.unit_verdict(cp, [x[0] for x in cp])[0]["verdict"], "PROMOTE")
        v = AL.unit_verdict(cp[:1], [x[0] for x in cp])[0]
        self.assertEqual(v["verdict"], "INCOMPLETE")


class Plumbing(unittest.TestCase):
    def test_phase_split_reads_the_loop(self):
        import bench_board_probe as PR
        ps = PR.phase_split(cell(100.0, 60.0, 5))
        self.assertEqual(ps["iterations"], 5)
        self.assertAlmostEqual(ps["loop_per_iteration_ms"], 12.0)
        self.assertAlmostEqual(ps["fixed_ms"], 40.0)
        self.assertIsNone(PR.phase_split(cell(100.0))["loop_ms"])

    def test_share_base_only_same_loop_mode(self):
        import measure_owed as MO
        rid = "algos/sgd-clf/taxi/rows=full"
        whole = {"races": {rid: {"status": "done", "finished": "2026-11-24T10:00:00Z"}}}
        capped = {"races": {rid: {"status": "done", "finished": "2026-11-24T10:00:00Z",
                                  "loop_mode": {"mode": "capped", "k": PLAN["k"], "param": PLAN["param"]}}}}
        now = MO._iso_s("2026-11-24T10:05:00Z")
        _, shared, todo = MO.share_base(None, [rid], [("w", whole)], now, 1200, loop_cap="table")
        self.assertEqual((shared, todo), ({}, [rid]))
        _, shared, todo = MO.share_base(None, [rid], [("c", capped)], now, 1200, loop_cap="table")
        self.assertIn(rid, shared)
        _, shared, todo = MO.share_base(None, [rid], [("w", whole)], now, 1200)
        self.assertIn(rid, shared)               # loop mode off: unchanged behaviour
        _, shared, todo = MO.share_base(None, [rid], [("c", capped)], now, 1200, loop_cap="3")
        self.assertEqual(todo, [rid])            # another cap

    def test_bench_board_loop_env(self):
        import bench_board as BB
        self.assertEqual(BB.loop_env({"loop_cap": None}, {"family": "algos", "lane": "sgd-clf"}), {})
        env = BB.loop_env({"loop_cap": "table"}, {"family": "algos", "lane": "sgd-clf"})
        self.assertEqual(json.loads(env[AL.ENV])["mode"], "capped")
        env = BB.loop_env({"loop_cap": 2}, {"family": "algos", "lane": "sgd-clf"})
        self.assertEqual(json.loads(env[AL.ENV])["k"], 2)
        self.assertIsNone(BB.loop_cap_arg("off"))
        self.assertEqual(BB.loop_cap_arg("7"), 7)

    def test_summary_prints_the_loop_verdict_and_writes_json(self):
        import targeted_rows as TR
        ids = ["algos/sgd-clf/taxi/rows=full", "algos/sgd-clf/istella/rows=full"]
        with tempfile.TemporaryDirectory() as res:
            for name, ratio, h in (("board-tg-ab-b1-base-full", 1.0, "h1"), ("board-tg-ab-b1-full", 0.8, "h2")):
                os.makedirs(os.path.join(res, name))
                races = {rid: {"status": "done", "cells": [cell(100.0, 50.0 * ratio, 5, h=h)]} for rid in ids}
                with open(os.path.join(res, name, "board.json"), "w") as fh:
                    json.dump({"races": races}, fh)
            out = os.path.join(res, "ab-loop-verdicts.json")
            text = TR.summary({}, [], res, ids, [("b1", "full", ids, set())], loop_out=out)
            self.assertIn("AB-LOOP b1 rows=full PROMOTE", text)
            self.assertIn("AB-LOOP-FULL-RUN-OWED PC_AB_LOOP_FULL_ARMS=b1", text)
            self.assertIn("loop/it base", text)
            doc = json.load(open(out))
            self.assertTrue(doc["units"][0]["full_run"])

    def test_measure_candidate_full_arms(self):
        import measure_candidate as MC
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump({"units": [{"arm": "b", "full_run": True}, {"arm": "a", "full_run": False},
                                 {"arm": "c", "full_run": True}]}, fh)
        try:
            self.assertEqual(MC.loop_full_arms(fh.name), "b,c")
        finally:
            os.remove(fh.name)


if __name__ == "__main__":
    unittest.main()
