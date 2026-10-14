# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""tools/race_fairness.py: the race-fairness rule (AGENTS.md "Race fairness").

    python3 -m unittest discover -s tools -p test_race_fairness.py -v
"""
import importlib.util
import os
import unittest
import copy

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("race_fairness", os.path.join(HERE, "race_fairness.py"))
RF = importlib.util.module_from_spec(spec)
spec.loader.exec_module(RF)


def race(lane, opps, params_check="MATCHED", ours_ms=5.0, ours_q=None):
    cells = [{"arm": "ours", "mode": "identical", "status": "ok", "median_ms": ours_ms, "quality": ours_q or {},
              "params": {"params": {"alpha": 1.0}}}]
    for arm, ms, extra in opps:
        cells.append(dict({"arm": arm, "mode": "opponent", "device": "gpu", "status": "ok", "median_ms": ms,
                           "params": {"params": {"alpha": 1.0}}}, **extra))
    return {"lane": lane, "params_check": params_check, "cells": cells, "infer_cells": []}


def board(**races):
    return {"config": {"modes": ["identical"]}, "races": races}


class RaceFairness(unittest.TestCase):
    def test_derived_identity_annotation_preserves_supported_audit_but_settings_do_not(self):
        old=board(r=race("ridge",[("lib",10,{})],params_check="not checked"))
        rr=old["races"]["r"]
        rr["cells"][0]["main_board"]={"sha":"frozen","identity":{"status":"n/a"}}
        rr["audit"]={"identical/fit":dict(verdict="FAIR",opponent="lib",audit="human",
            evidence="retained source and worker review",reason="matched",pairing=RF.pairing(rr,"identical","fit"))}
        RF.apply(old,None,"2026-10-13")
        new=copy.deepcopy(old)
        new["races"]["r"]["cells"][0]["main_board"]["identity"]={"status":"MATCH"}
        RF.apply(new,old,"2026-10-14")
        self.assertEqual(new["races"]["r"]["audit"]["identical/fit"]["verdict"],"FAIR")
        changed=copy.deepcopy(new)
        changed["races"]["r"]["cells"][0]["params"]["params"]["alpha"]=2
        RF.apply(changed,old,"2026-10-14")
        self.assertEqual(changed["races"]["r"]["rerace"],"x")

    def test_new_iterative_measurements_need_executed_counts(self):
        rr=race("linearsvr",[("lib",10,{})])
        for c in rr["cells"]:
            c.update(rounds=1,params={"params":{"max_iter":1000}},
                     measurement={"completeness":"complete","source":{"status":"verified"},
                                  "artifacts":{"status":"verified"}})
        bad,missing,_=RF._pair_problems(rr,*rr["cells"])
        self.assertFalse(bad)
        self.assertTrue(any("iterations not recorded" in s for s in missing))
        for c in rr["cells"]:
            c["iteration_receipts"]=[dict(status="recorded",values=[17])]
        self.assertEqual(RF._pair_problems(rr,*rr["cells"])[:2],([],[]))
        rr["cells"][1]["iteration_receipts"][0]["values"]=[18]
        self.assertTrue(any("iteration receipts differ" in s for s in RF._pair_problems(rr,*rr["cells"])[0]))

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
        self.assertEqual(b["races"]["n"]["rerace"], "x")

    def test_a_slow_host_is_found_from_the_board(self):
        races = {}
        for i in range(6):
            r = race("lane%d" % i, [("torch-a", 20.0, {"copied_from": {"box": "hostS"}, "shape": "X 10x2"})])
            races["s%d" % i] = r
            races["f%d" % i] = race("lane%d" % i, [("torch-a", 10.0, {"copied_from": {"box": "hostF"}, "shape": "X 10x2"})])
        self.assertNotIn("hostS", RF.slow_hosts({"races": races}))
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
        rr["audit"] = {"identical/fit": {"verdict": "UNFAIR", "rerace": "x", "audit": "hand", "opponent": "cuml-gpu",
                                         "pairing": RF.pairing(rr, "identical", "fit")}}
        RF.apply(b, None, "2026-10-13")
        self.assertEqual((rr["audit"]["identical/fit"]["verdict"], rr["rerace"]), ("UNFAIR", "x"))
        rr["audit"]["identical/fit"] = {"verdict": "TIE", "rerace": "x", "rule": "old", "pairing": "stale"}
        RF.apply(b, None, "2026-10-13")
        self.assertEqual(rr["audit"]["identical/fit"]["verdict"], "INSUFFICIENT")
        self.assertEqual(rr["rerace"], "x")

    def test_one_bad_arm_does_not_hide_a_supported_opponent(self):
        b = board(r=race("solver", [("bad", 1, {"quality": {"n_iter": 20}}),
                                    ("good", 10, {"quality": {"n_iter": 10}})], ours_q={"n_iter": 10}))
        RF.apply(b)
        e = b["races"]["r"]["audit"]["identical/fit"]
        self.assertEqual(e["admitted_opponents"], ["good"])
        self.assertEqual(e["opponent"], "good")
        self.assertEqual(e["opponents"]["bad"]["verdict"], "UNFAIR")
        self.assertEqual(e["rerace"], "")

    def test_no_evidence_never_qualifies_even_with_matched_label(self):
        b = board(r=race("solver", [("lib", 10, {})]))
        for c in b["races"]["r"]["cells"]:
            c.pop("params")
        RF.apply(b)
        self.assertEqual(b["races"]["r"]["rerace"], "x")

    def test_supported_historical_audit_survives_without_new_measurement(self):
        b = board(r=race("solver", [("lib", 10, {})], params_check="not checked"))
        rr = b["races"]["r"]
        rr["audit"] = {"identical/fit": dict(verdict="FAIR", opponent="lib", audit="human",
            evidence="retained run log and source review", reason="same workload", pairing=RF.legacy_pairing(rr,"identical","fit"))}
        times = copy.deepcopy(rr["cells"])
        RF.apply(b, upgrade_legacy=True)
        self.assertNotIn("rerace", rr)
        RF.apply(b)
        self.assertNotIn("rerace", rr)
        self.assertEqual(rr["cells"], times)
        rr["cells"][1]["version"] = "changed"
        RF.apply(b)
        self.assertEqual(rr["rerace"], "x")
        self.assertTrue(rr["audit_history"])

    def test_fingerprint_covers_settings_versions_source_hardware_quality_clock(self):
        rr = race("solver", [("lib", 10, {})])
        expected = RF.pairing(rr, "identical", "fit")
        for field, value in [("version", "2"), ("hardware", "other"), ("quality", {"n_iter": 99}),
                             ("params", {"params": {"tol": .2}}), ("measurement", {"id":"new"}),
                             ("comparability", {"span": {"input_home": "device"}})]:
            changed = copy.deepcopy(rr)
            changed["cells"][1][field] = value
            self.assertNotEqual(RF.pairing(changed,"identical","fit"), expected, field)

    def test_missing_retention_cannot_be_approved_by_matched_params(self):
        b = board(r=race("solver", [("lib", 10, {})]))
        b["races"]["r"]["cells"][0]["measurement"] = {"schema":"mojolearn.measurement.v1", "completeness":"complete",
                                                                "source":{"status":"local"}, "artifacts":{"status":"local"}}
        RF.apply(b)
        self.assertEqual(b["races"]["r"]["rerace"], "x")

    def test_unresolved_row_cannot_switch_opponents_to_escape_audit(self):
        b = board(r=race("solver", [("lib",10,{}),("other",20,{})]))
        rr=b["races"]["r"]
        rr["audit"]={"identical/fit":dict(verdict="SUSPECT",opponent="lib",audit="human",evidence="output invalid",
                                           pairing=RF.pairing(rr,"identical","fit"))}
        RF.apply(b)
        self.assertEqual(rr["rerace"],"x")

    def test_inference_batches_must_match_before_comparing(self):
        rr=race("solver",[("lib",10,{})])
        rr["infer_cells"]=copy.deepcopy(rr["cells"])
        rr["infer_cells"][0].update(batch="bulk",batch_rows=1000)
        rr["infer_cells"][1].update(batch="single",batch_rows=1)
        b=board(r=rr);RF.apply(b)
        self.assertEqual(rr["audit"]["identical/infer"]["rerace"],"x")
        self.assertEqual(rr["audit"]["identical/fit"]["rerace"],"")


if __name__ == "__main__":
    unittest.main()
