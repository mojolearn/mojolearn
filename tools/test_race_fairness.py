# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""tools/race_fairness.py: the race-fairness rule (AGENTS.md "Race fairness").

    python3 -m unittest discover -s tools -p test_race_fairness.py -v
"""
import importlib.util
import json
import os
import unittest
import copy

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("race_fairness", os.path.join(HERE, "race_fairness.py"))
RF = importlib.util.module_from_spec(spec)
spec.loader.exec_module(RF)


def race(lane, opps, params_check="MATCHED", ours_ms=5.0, ours_q=None):
    cells = [{"arm": "ours", "mode": "identical", "status": "ok", "median_ms": ours_ms, "quality": {"rmse": 1.0} if ours_q is None else ours_q,
              "comparability": {"span": {"input_home": "host"}},
              "params": {"params": {"alpha": 1.0}}}]
    for arm, ms, extra in opps:
        cells.append(dict({"arm": arm, "mode": "opponent", "device": "gpu", "status": "ok", "median_ms": ms,
                           "params": {"params": {"alpha": 1.0}}, "quality": {"rmse": 1.0},
                           "comparability": {"span": {"input_home": "host"}}}, **extra))
    return {"lane": lane, "params_check": params_check, "cells": cells, "infer_cells": []}


def board(**races):
    return {"config": {"modes": ["identical"]}, "races": races}


class RaceFairness(unittest.TestCase):
    def test_alternate_clock_does_not_replace_required_comparison(self):
        for lib in ("cupy", "gpytorch"):
            rr = race("solver", [(lib + "-gpu", 30, {"library": lib, "family": "algos",
                "comparability": {"span": {"input_home": "device"}}})], ours_ms=100)
            rr["cells"][0]["comparability"]["span"]["upload_ms_separate"] = 40
            RF.apply(board(r=rr))   # ours whole vs arm kernel: shown, labelled (it only understates us)
            self.assertEqual(rr["audit"]["identical/fit"]["verdict"], "MATCHED")
            self.assertIn("MIXED", rr["audit"]["identical/fit"]["opponents"][lib + "-gpu"]["timing_comparison"]["label"])
            rr["cells"][1]["comparability"]["span"]["upload_ms_untimed"] = 10
            RF.apply(board(r=rr))
            self.assertEqual(rr["audit"]["identical/fit"]["verdict"], "MATCHED")
        rr = race("solver", [("torch-gpu", 30, {"library": "torch", "family": "classical",
            "comparability": {"span": {"input_home": "device", "upload_ms_untimed": 10}}})])
        RF.apply(board(r=rr))
        self.assertEqual(rr["audit"]["identical/fit"]["verdict"], "INSUFFICIENT")
        rr["cells"][1]["family"] = "neural"
        RF.apply(board(r=rr))
        self.assertEqual(rr["audit"]["identical/fit"]["verdict"], "MATCHED")
        self.assertEqual(rr["audit"]["identical/fit"]["opponents"]["torch-gpu"]["timing_comparison"]["clock"], "whole")

    def test_best_opponent_uses_comparable_clocks_and_retains_operands(self):
        rr = race("solver", [("torch-gpu", 30, {"library": "torch", "family": "classical",
            "comparability": {"span": {"input_home": "device", "upload_ms_untimed": 80}}}),
            ("lib", 40, {})], ours_ms=100)
        rr["cells"][0]["library"] = "mojolearn"
        rr["cells"][0]["comparability"]["span"]["upload_ms_separate"] = 40
        RF.apply(board(r=rr))
        a = rr["audit"]["identical/fit"]
        self.assertEqual(a["opponent"], "lib")
        self.assertEqual(a["ours_over_theirs"], 2.5)
        pair = a["opponents"]["torch-gpu"]["timing_comparisons"][0]
        self.assertEqual((pair["ours_ms"], pair["opponent_ms"]), (60., 30.))

    def test_old_count_only_audit_is_superseded_but_other_findings_are_not(self):
        for reason, admitted in (("iterations differ: ours 24, sklearn-cpu 30", True),
                                 ("iterations differ and dataset differs", False)):
            rr = race("gmm", [("sklearn-cpu", 10, {})])
            rr["audit"] = {"identical/fit": dict(verdict="UNFAIR", reason=reason,
                opponent="sklearn-cpu", audit="human", evidence="recorded counts",
                pairing=RF.pairing(rr, "identical", "fit"))}
            RF.apply(board(r=rr))
            self.assertEqual(bool(rr["audit"]["identical/fit"]["admitted_opponents"]), admitted)
            self.assertTrue(rr["audit_history"])

    def test_quality_and_settings_cannot_approve_mixed_clocks(self):
        rr = race("solver", [("lib", 10, {})])
        rr["cells"][0]["comparability"]["span"]["input_home"] = "device"   # ours kernel vs arm whole: favours us
        RF.apply(board(r=rr))
        self.assertEqual(rr["rerace"], "x")
        self.assertIn("timing boundary", rr["audit"]["identical/fit"]["opponents"]["lib"]["reason"])

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

    def test_rebuild_bookkeeping_keeps_supported_audit_and_history(self):
        old = board(r=race("ridge", [("lib", 10, {})], params_check="not checked"))
        rr = old["races"]["r"]
        rr["cells"][0].update(main_board={"sha": "frozen", "observation": {"job": "a1"}},
            settings={"rounds": 1, "source": "main board: one scored run of ours (previous)"},
            source="main@frozen amd/a1; identity vs the nvidia columns: n/a")
        rr["cells"][1]["copied_from"] = dict(board="saved", race="ridge/full", box="worker",
            measured="2026-10-13", path="/old/checkout/board.json")
        rr["audit"] = {"identical/fit": dict(verdict="FAIR", opponent="lib", audit="human",
            evidence="retained worker and settings review", reason="same workload",
            pairing=RF.pairing(rr, "identical", "fit"))}
        RF.apply(old)
        old["races"]["r"]["audit_history"] = [{"reason": "original retained finding"}]
        new = copy.deepcopy(old)
        nr = new["races"]["r"]
        nr.pop("audit_history")
        nr["cells"][0]["settings"]["source"] = "main board: one scored run of ours (algos)"
        nr["cells"][0]["main_board"]["observation"].update(paired_box=None, paired_race=None, measured_cell=None)
        nr["cells"][0]["source"] = "main@frozen amd/a1; identity vs nvidia-l40s: MATCH"
        nr["cells"][1]["copied_from"]["path"] = "/new/checkout/board.json"
        saved = copy.deepcopy(nr["cells"])
        RF.apply(new, old)
        self.assertEqual(nr["audit"]["identical/fit"]["verdict"], "FAIR")
        self.assertIn({"reason": "original retained finding"}, nr["audit_history"])
        self.assertEqual(nr["cells"], saved)
        for kind in ("settings", "worker", "quality", "clock", "source", "paired_evidence"):
            changed = copy.deepcopy(new)
            cells = changed["races"]["r"]["cells"]
            if kind == "settings": cells[0]["settings"]["rounds"] = 2
            elif kind == "worker": cells[1]["copied_from"]["box"] = "other-worker"
            elif kind == "quality": cells[1]["quality"]["rmse"] = 8
            elif kind == "clock": cells[1]["comparability"]["span"]["input_home"] = "device"
            elif kind == "source": cells[0]["source"] = "main@different amd/a1; identity vs nvidia-l40s: MATCH"
            else: cells[0]["main_board"]["observation"]["measured_cell"] = {"measurement": {"id": "new"}}
            RF.apply(changed, old)
            self.assertEqual(changed["races"]["r"]["rerace"], "x", kind)

    def test_added_opponent_does_not_discard_unchanged_supported_arm(self):
        old = board(r=race("ridge", [("lib", 10, {})], params_check="not checked"))
        rr = old["races"]["r"]
        rr["audit"] = {"identical/fit": dict(verdict="FAIR", opponent="lib", audit="human",
            evidence="retained worker review", pairing=RF.pairing(rr, "identical", "fit"))}
        RF.apply(old)
        new = copy.deepcopy(old)
        new["races"]["r"]["cells"].append(dict(rr["cells"][1], arm="new-library"))
        RF.apply(new, old)
        self.assertEqual(new["races"]["r"]["audit"]["identical/fit"]["admitted_opponents"], ["lib"])
        changed = copy.deepcopy(new)
        changed["races"]["r"]["cells"][1]["version"] = "new-version"
        RF.apply(changed, old)
        self.assertEqual(changed["races"]["r"]["rerace"], "x")

    def test_actual_iterations_are_evidence_not_an_admission_gate(self):
        rr=race("linearsvr",[("lib",10,{})])
        for c in rr["cells"]:
            c.update(rounds=1,params={"params":{"max_iter":1000}},
                     measurement={"completeness":"complete","source":{"status":"verified"},
                                  "artifacts":{"status":"verified"}})
        self.assertEqual(RF._pair_problems(rr,*rr["cells"])[:2],([],[]))
        for i,c in enumerate(rr["cells"]):
            c["iteration_receipts"]=[dict(status="recorded",values=[17+i])]
        self.assertEqual(RF._pair_problems(rr,*rr["cells"])[:2],([],[]))
        RF.apply(board(r=rr))
        self.assertEqual(rr["audit"]["identical/fit"]["admitted_opponents"],["lib"])

    def test_harness_arm_is_excluded_and_the_race_judged_on_the_rest(self):
        b = board(r=race("pca", [("torch-gpu", 1.0, {"library": "torch"}), ("cuml-gpu", 10.0, {"library": "cuml"})]))
        RF.apply(b, None, "2026-10-13")
        rr = b["races"]["r"]
        self.assertEqual(rr["opponents_excluded"]["arms"], ["torch-gpu"])
        self.assertEqual(rr["audit"]["identical/fit"]["opponent"], "cuml-gpu")

    def test_a_cell_from_another_host_is_not_excluded(self):
        other = {"library": "cuml", "copied_from": {"box": "cc560ebdaf91"}}
        b = board(l=race("svd", [("cupy-gpu", 9.0, other)]))
        RF.apply(b, None, "2026-10-14")
        self.assertNotIn("opponents_excluded", b["races"]["l"])

    def test_old_slow_host_exclusions_are_dropped_and_others_kept(self):
        rr = race("pca", [("torch-gpu", 1.0, {"library": "torch"}), ("cuml-gpu", 10.0, {"library": "cuml"})])
        rr["opponents_excluded"] = {"arms": ["cuml-gpu"], "date": "2026-10-13", "reason": "",
                                    "why": {"cuml-gpu": "measured on cc560ebdaf91, a slow box: old rule"}}
        rr["opponents_excluded"]["basis"] = RF.exclusion_basis(rr)
        b = board(r=rr)
        RF.apply(b, None, "2026-10-14")
        self.assertEqual(b["races"]["r"]["opponents_excluded"]["arms"], ["torch-gpu"])

    def test_iterations_differ_with_near_quality_and_ties_show(self):
        b = board(i=race("kmeans", [("cuml-gpu", 9.0, {"quality": {"n_iter": 21, "inertia": 101}})], ours_q={"n_iter": 33, "inertia": 100}),
                  t=race("ridge", [("cuml-gpu", 5.1, {})]))
        RF.apply(b, None, "2026-10-13")
        self.assertNotIn("rerace", b["races"]["i"])
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
        b = board(r=race("solver", [("bad", 1, {"params": {"params": {"alpha": 2.0}}}),
                                    ("good", 10, {"quality": {"n_iter": 10, "rmse": 1.0}})], ours_q={"n_iter": 10, "rmse": 1.0}))
        RF.apply(b)
        e = b["races"]["r"]["audit"]["identical/fit"]
        self.assertEqual(e["admitted_opponents"], ["good"])
        self.assertEqual(e["opponent"], "good")
        self.assertEqual(e["opponents"]["bad"]["verdict"], "UNFAIR")
        self.assertEqual(e["rerace"], "")

    def test_ours_whole_against_arm_kernel_is_shown_but_not_the_reverse(self):
        rr = race("cnn", [("torch-eager-fp32", 10, {"library": "torch", "comparability": {"span": {"input_home": "device"}}})])
        RF.apply(board(r=rr))
        e = rr["audit"]["identical/fit"]
        self.assertEqual(e["admitted_opponents"], ["torch-eager-fp32"], e["opponents"])
        self.assertIn("MIXED ours whole / arm kernel", e["opponents"]["torch-eager-fp32"]["timing_comparison"]["label"])

    def test_quality_never_hides_a_race_and_is_always_recorded(self):
        for theirs in (0.5, 2.0):   # we are better, we are worse: both shown, both recorded
            b = board(r=race("solver", [("lib", 10, {"quality": {"rmse": theirs}})], ours_q={"rmse": 1.0}))
            RF.apply(b)
            e = b["races"]["r"]["audit"]["identical/fit"]
            self.assertEqual(e["admitted_opponents"], ["lib"])
            self.assertEqual(e["opponents"]["lib"]["quality_comparison"][0]["checks"][0]["opponent"], theirs)

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

    def test_new_classical_measurements_require_retained_scored_outputs(self):
        b = board(r=race("solver", [("lib", 10, {})]))
        rr = b["races"]["r"]
        rr["family"] = "classical"
        for cell in rr["cells"]:
            cell["measurement"] = dict(completeness="complete", source={"status": "verified"},
                artifacts={"status": "verified"}, retained_outputs={"status": "verified"})
        RF.apply(b)
        self.assertNotEqual(rr.get("rerace"), "x")
        rr["cells"][0]["measurement"].pop("retained_outputs")
        RF.apply(b)
        self.assertEqual(rr["rerace"], "x")
        self.assertIn("scored classical output bytes", str(rr["audit"]))

    def test_inference_batches_must_match_before_comparing(self):
        rr=race("solver",[("lib",10,{})])
        rr["infer_cells"]=copy.deepcopy(rr["cells"])
        rr["infer_cells"][0].update(batch="bulk",batch_rows=1000)
        rr["infer_cells"][1].update(batch="single",batch_rows=1)
        b=board(r=rr);RF.apply(b)
        self.assertEqual(rr["audit"]["identical/infer"]["rerace"],"x")
        self.assertEqual(rr["audit"]["identical/fit"]["rerace"],"")


    # ---- lane harness-evidence (2026-10-14): "ok" only if scored; not-scored is shown, never gated ----

    def test_scorer_crash_is_not_scored_not_a_quality_failure(self):
        crash = {"error": "Own host reference failed (bba_neural_fixtures.FixturesMissing: neural fixtures "
                          "missing); see host.log", "scorer": "failed"}
        rr = race("gcn", [("lib", 10, {})], ours_q=crash)
        RF.apply(board(r=rr))
        e = rr["audit"]["identical/fit"]
        self.assertEqual(e["admitted_opponents"], ["lib"])          # never an exclusion
        q = e["opponents"]["lib"]["quality_comparison"][0]
        self.assertEqual(q["failures"], [])
        self.assertTrue(any("quality not scored (scorer failed): ours" in n for n in q["not_scored"]))
        self.assertIn("quality not scored (scorer failed)", e["opponents"]["lib"]["quality_note"])
        self.assertNotIn("recorded quality failure", json.dumps(e))

    def test_legacy_scorer_error_on_an_ok_cell_is_not_scored(self):
        # an older cell: status ok, the scorer's exception in `error` with no `scorer` marker
        rr = race("gcn", [("lib", 10, {})], ours_q={"error": "Own host reference failed (FixturesMissing)"})
        self.assertTrue(RF.timed_not_scored(rr["cells"][0]))
        RF.apply(board(r=rr))
        q = rr["audit"]["identical/fit"]["opponents"]["lib"]["quality_comparison"][0]
        self.assertEqual(q["failures"], [])
        self.assertTrue(q["not_scored"])

    def test_not_scored_status_keeps_its_time_and_its_ratio(self):
        rr = race("gcn", [("lib", 10, {})], ours_ms=5.0)
        rr["cells"][0].update(status="not-scored", quality={}, scorer_error="scorer did not run")
        RF.apply(board(r=rr))
        e = rr["audit"]["identical/fit"]
        self.assertEqual(e["admitted_opponents"], ["lib"])
        self.assertEqual(e["ours_over_theirs"], 0.5)
        self.assertNotIn("our measurement did not complete", e["opponents"]["lib"]["reason"])

    def test_not_scored_opponent_is_judged_not_dropped(self):
        rr = race("gcn", [("lib", 10, {"status": "not-scored", "quality": {}, "scorer_error": "no reference"})])
        RF.apply(board(r=rr))
        e = rr["audit"]["identical/fit"]
        self.assertIn("lib", e["opponents"])
        self.assertIn("quality not scored (scorer failed): lib", e["opponents"]["lib"].get("quality_note", ""))

    def test_scored_output_failure_stays_a_recorded_failure(self):
        rr = race("gcn", [("lib", 10, {})], ours_q={"error": "Integer host reference mismatch: y", "scorer": "scored"})
        RF.apply(board(r=rr))
        q = rr["audit"]["identical/fit"]["opponents"]["lib"]["quality_comparison"][0]
        self.assertEqual(q["failures"], ["recorded quality failure: ours"])
        self.assertNotIn("not_scored", q)

    def test_call_line_is_not_an_observation(self):
        rr = race("solver", [("lib", 10, {})])
        before = RF.pairing(rr, "identical", "fit")
        for c in rr["cells"]:
            c.update(call="lib.Solver(alpha=1)", call_source="params")
        self.assertEqual(RF.pairing(rr, "identical", "fit"), before)
        inf = dict(rr["cells"][1], phase="infer", call="predict(Xq)")
        rr["infer_cells"] = [inf]
        p = RF.pairing(rr, "identical", "infer")
        inf["constructor_call"] = "lib.Solver(alpha=1)"
        self.assertEqual(RF.pairing(rr, "identical", "infer"), p)
        inf["call"] = "transform(Xq)"         # what an inference clock ran stays evidence
        self.assertNotEqual(RF.pairing(rr, "identical", "infer"), p)


if __name__ == "__main__":
    unittest.main()
