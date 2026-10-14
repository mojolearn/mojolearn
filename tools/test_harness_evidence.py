# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Lane harness-evidence (2026-10-14): scorer classification, both clocks, call lines, the scorer preflight.

    python3 -m unittest discover -s tools -p test_harness_evidence.py -v
"""
import importlib.util
import json
import os
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    spec = importlib.util.spec_from_file_location("he_" + name, os.path.join(HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BQ = _load("board_quality")
BC = _load("board_clock_audit")
CALLS = _load("race_call_line")
RSP = _load("race_scorer_preflight")


class ScorerClassification(unittest.TestCase):
    def test_scorer_failures(self):
        for q in ({"error": "Own host reference failed (FixturesMissing)"},
                  {"error": "KeyError('yq')"},
                  {"scorer": "failed", "scorer_error": "no fixtures"},
                  {"scorer": "not-run"},
                  {"quality_error": "ValueError('x')"}):
            self.assertIsNotNone(BQ.scorer_failure(q), q)
            self.assertFalse(BQ.scored_failure(q), q)

    def test_scored_failures(self):
        for q in ({"error": "Integer host reference mismatch: y"},
                  {"error": "Nonfinite host reference output: y"},
                  {"error": "ValueError('nonfinite layer output or torch reference')"},
                  {"error": "boom", "scorer": "scored"},
                  {"finite": False}, {"host_reference_passed": False}):
            self.assertIsNone(BQ.scorer_failure(q), q)
            self.assertTrue(BQ.scored_failure(q), q)

    def test_clean_quality(self):
        self.assertIsNone(BQ.scorer_failure({"rmse": 1.0}))
        self.assertFalse(BQ.scored_failure({"rmse": 1.0}))

    def test_compare_reports_not_scored_without_a_failure(self):
        out = BQ.compare({"arm": "ours", "quality": {"error": "Own host reference failed (x)"}},
                         {"arm": "lib", "quality": {"rmse": 1.0}})
        self.assertEqual(out["failures"], [])
        self.assertEqual(out["verdict"], "INSUFFICIENT")
        self.assertTrue(out["not_scored"][0].startswith("quality not scored (scorer failed): ours"))
        out = BQ.compare({"arm": "ours", "status": "not-scored", "scorer_error": "x", "quality": {}},
                         {"arm": "lib", "quality": {}})
        self.assertEqual(len(out["not_scored"]), 1)


def cell(**kw):
    c = {"status": "ok", "median_ms": 100.0, "device": "gpu", "family": "algos", "comparability": {"span": {}}}
    span = kw.pop("span", {})
    c.update(kw)
    c["comparability"]["span"].update(span)
    return c


class TwoClocks(unittest.TestCase):
    def test_opponent_whole_from_its_timed_pre_clock_copy(self):
        k = BC.cell_clock(cell(library="torch", span={"input_home": "device", "upload_ms_untimed": 20.0}))
        self.assertEqual((k["kernel_ms"], k["whole_ms"], k["copy_ms"]), (100.0, 120.0, 20.0))

    def test_opponent_whole_from_a_separate_upload(self):
        k = BC.cell_clock(cell(library="cupy", span={"input_home": "device", "upload_ms_separate": 5.0}))
        self.assertEqual((k["kernel_ms"], k["whole_ms"], k["derivable"]), (100.0, 105.0, "both"))

    def test_inference_cells_use_their_own_predict_copy(self):
        ours = BC.cell_clock(cell(library="mojolearn", phase="infer",
                                  span={"input_home": "host", "upload_ms_separate": 50.0, "predict_upload_ms": 10.0}))
        self.assertEqual((ours["whole_ms"], ours["kernel_ms"], ours["copy_source"]), (100.0, 90.0, "predict_upload_ms"))
        opp = BC.cell_clock(cell(library="cuml", phase="infer",
                                 span={"input_home": "device", "upload_ms_untimed": 300.0, "predict_upload_ms": 7.0}))
        self.assertEqual((opp["kernel_ms"], opp["whole_ms"]), (100.0, 107.0))
        # without its own copy an algos inference cell still has one clock only (the fit upload is not it)
        opp = BC.cell_clock(cell(library="cuml", phase="infer", span={"input_home": "device", "upload_ms_untimed": 300.0}))
        self.assertEqual(opp["derivable"], "kernel")

    def test_mixed_pair_becomes_derivable_on_both_clocks(self):
        ours = cell(library="mojolearn", mode="identical", phase="infer",
                    span={"input_home": "host", "predict_upload_ms": 10.0})
        torch = cell(library="torch", mode="opponent", phase="infer", median_ms=45.0,
                     span={"input_home": "device", "predict_upload_ms": 5.0})
        r = BC.pair_ratio(BC.cell_clock(ours), BC.cell_clock(torch), torch)
        self.assertEqual((r["kind"], r["clock"]), ("preferred", "kernel"))
        self.assertAlmostEqual(r["value"], 90.0 / 45.0)
        cuml = cell(library="cuml", mode="opponent", phase="infer", median_ms=45.0,
                    span={"input_home": "device", "predict_upload_ms": 5.0})
        r = BC.pair_ratio(BC.cell_clock(ours), BC.cell_clock(cuml), cuml)
        self.assertEqual((r["kind"], r["clock"]), ("preferred", "whole"))
        self.assertAlmostEqual(r["value"], 100.0 / 50.0)

    def test_not_scored_cell_keeps_its_clocks(self):
        k = BC.cell_clock(cell(library="mojolearn", status="not-scored", span={"input_home": "host"}))
        self.assertEqual(k["whole_ms"], 100.0)

    def test_never_invents_a_copy(self):
        k = BC.cell_clock(cell(library="torch", span={"input_home": "device"}))
        self.assertEqual((k["whole_ms"], k["copy_ms"]), (None, None))


class CallLines(unittest.TestCase):
    def test_from_params(self):
        c = {"params": {"implementation": {"module": "cuml.cluster.kmeans", "name": "KMeans"},
                        "raw_params": {"n_clusters": 8, "max_iter": 300, "tol": 1e-4, "verbose": None}}}
        line, src = CALLS.call_line_of(c)
        self.assertEqual(line, "cuml.KMeans(max_iter=300, n_clusters=8, tol=0.0001)")
        self.assertEqual(src, "params")

    def test_invocation_preferred_and_arrays_as_shape_dtype(self):
        c = {"params": {"implementation": {"module": "x", "name": "Y"}},
             "measurement": {"invocation": {
                 "implementation": {"module": "mojolearn._x_sequence_optim", "name": "Adafactor"},
                 "effective_arguments": {"lr": 0.001, "input_arrays": [
                     {"dtype": "float32", "name": "p0", "sha256": "ab" * 32, "shape": [16777216]}]}}}}
        line, src = CALLS.call_line_of(c)
        self.assertEqual(src, "measurement.invocation")
        self.assertEqual(line, "mojolearn.Adafactor(lr=0.001, p0=float32[16777216])")
        self.assertNotIn("abab", line)

    def test_long_argument_lists_are_cut(self):
        args = {"k%02d" % i: i for i in range(100)}
        line = CALLS.call_line({"module": "sklearn.linear_model", "name": "Ridge"}, args)
        self.assertTrue(line.endswith(", ...)"))
        self.assertLessEqual(len(line), CALLS.MAX_CHARS + 8)

    def test_set_call_fit_and_infer(self):
        p = {"implementation": {"module": "torch.optim.adam", "name": "Adam"}, "raw_params": {"lr": 0.01}}
        fit = {"params": p}
        CALLS.set_call(fit)
        self.assertEqual(fit["call"], "torch.Adam(lr=0.01)")
        inf = {"params": p, "phase": "infer", "call": "predict(Xq)"}
        CALLS.set_call(inf)
        self.assertEqual(inf["call"], "predict(Xq)")
        self.assertEqual(CALLS.arm_call_text(inf), "torch.Adam(lr=0.01); predict(Xq)")

    def test_no_implementation_no_line(self):
        self.assertEqual(CALLS.call_line_of({"params": {"params": {"a": 1}}}), (None, None))
        b = {"races": {"r": {"cells": [{"params": {}}], "infer_cells": []}}}
        self.assertEqual(CALLS.annotate_board(b), 0)


class ScorerPreflight(unittest.TestCase):
    def test_failed_check_raises_and_is_recorded(self):
        pf = RSP.Preflight("gcn", "istella")
        pf.check("ok", lambda: 1)
        with self.assertRaises(RSP.ScorerUnavailable):
            pf.check("neural fixtures", lambda: (_ for _ in ()).throw(RuntimeError("fixtures missing")))
        self.assertEqual(pf.receipt["status"], "refused")
        self.assertEqual([c["status"] for c in pf.receipt["checks"]], ["ok", "failed"])

    def test_require_arrays(self):
        import numpy as np
        self.assertEqual(RSP.require_arrays({"yq": np.zeros(3)}, ("yq",)), {"yq": [3]})
        with self.assertRaises(ValueError):
            RSP.require_arrays({"yq": np.zeros(0)}, ("yq", "Xq"))

    def test_self_test(self):
        ok = RSP.self_test(lambda outs: {a: {"accuracy": 1.0} for a in outs}, {"p": {}}, ("accuracy",))
        self.assertEqual(ok, {"p": {"accuracy": 1.0}})
        with self.assertRaises(ValueError):
            RSP.self_test(lambda outs: {a: {"error": "KeyError"} for a in outs}, {"p": {}}, ("accuracy",))
        with self.assertRaises(ValueError):
            RSP.self_test(lambda outs: {a: {} for a in outs}, {"p": {}}, ("accuracy",))

    def test_refuse_marks_every_arm_and_times_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "r.json")
            result = {"lane": "gcn", "dataset": "istella", "arms": {}}
            rc = RSP.refuse(result, ["ours", "torch-eager-fp32"], "own host reference: FixturesMissing", path, "T")
            self.assertEqual(rc, 1)
            saved = json.load(open(path))
        for arm in ("ours", "torch-eager-fp32"):
            a = saved["arms"][arm]
            self.assertEqual((a["status"], a["ms"]), ("REFUSED", []))
            self.assertTrue(a["error"].startswith("quality scorer unavailable: "))
        self.assertEqual(saved["quality"]["scorer"], "failed")


if __name__ == "__main__":
    unittest.main()
