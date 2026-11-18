#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn
"""tools/test_targeted_rows.py -- tools/ops/targeted_rows.py (the targeted measurement, Andrew 2026-11-17) on
in-memory fixtures (no box, no git archive): the selection reasons and their order, the estimate and the split under
the cap, the interleave chunks, the box plan against the real planner, the lane -> estimator module -> bindings map,
the changed-lane reason with a stubbed closure diff, the SUMMARY, and the job and stage wiring. Each test is
independent (the suite runs its files in parallel)."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "ops"))
sys.path.insert(0, HERE)
import targeted_rows as TR  # noqa: E402
import measure_candidate as MC  # noqa: E402

PLANNER = os.path.join(HERE, "bench_board.py")
JOB = os.path.join(HERE, "ops", "paper_compare_job.sh")
A, B, C, D = ("classical/knn/istella/rows=full", "classical/dbscan/istella/rows=full",
              "trees/rf/istella/rows=full", "classical/pca/taxi/rows=full")
INTERNAL = "algos/internal-call-overhead-lr-at/synthetic/rows=full"


def compare_doc():
    return {"paper_version": "0.8.55", "vendor": "nvidia", "new_label": "candidate x", "verdict_basis": "same box",
            "rows": [{"race_id": A, "key": A + " fit ours", "ratio": 1.5, "flags": ["REGRESSION"], "new_ms": 15,
                      "old_ms": 10},
                     {"race_id": B, "key": B + " fit ours", "ratio": 1.0, "flags": ["SAME"], "new_ms": 10,
                      "old_ms": 10},
                     # no verdict (no on-box baseline): its ratio never selects
                     {"race_id": D, "key": D + " fit ours", "ratio": 3.0, "flags": ["PARAMS-DIFFER"]}],
            "extra_rows": [{"race_id": C, "key": C + " fit ours", "flags": ["REGRESSION"], "new_ms": 30,
                            "priors": {"0.8.55": {"ms": 20, "ratio": 1.2}}},
                           {"race_id": "algos/lr-step/synthetic/rows=full", "key": "x", "flags": [],
                            "priors": {"0.8.55": {"ratio": 400.0}}}],
            "internal_rows": []}


def board(secs):
    return {"races": {rid: {"status": "done", "wall_s": s, "family": rid.split("/")[0]} for rid, s in secs.items()}}


class Select(unittest.TestCase):
    def test_reasons_order_and_estimate(self):
        plan = [A, B, C, D, INTERNAL]
        rows, counts, gone, guessed = TR.select(compare_doc(), plan, 1.10, board({A: 60, B: 30, C: 120}),
                                                board({A: 50, C: 100}), {B: "changed", D: "changed?"})
        ids = [r["id"] for r in rows]
        self.assertEqual(ids, [A, C, B, D, INTERNAL])      # paper, slower (worst first), changed, changed?, internal
        self.assertEqual(rows[0]["reasons"], ["paper", "slower"])
        self.assertEqual(counts, {"paper": 1, "slower": 2, "changed": 1, "changed?": 1, "fix": 0, "internal": 1,
                                  "control": 0})
        self.assertEqual(gone, ["algos/lr-step/synthetic/rows=full"])   # no longer planned: never selected
        self.assertEqual(guessed, 2)                       # D and INTERNAL: no stored candidate time
        self.assertEqual(rows[0]["input_ratio"], 1.5)
        self.assertEqual(rows[2]["prior_s"], 0.0)          # B: no stored 0.8.55 time

    def test_rows_file_round_trip_and_bare_ids(self):
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "rows.tsv")
            rows = [{"id": A, "reasons": ["paper"], "new_s": 60.0, "prior_s": 50.0, "input_ratio": 1.5}]
            Path(f).write_text(TR.rows_text(rows, ["hdr"]) + B + "\n")
            got = TR.read_rows(f)
            self.assertEqual(got[0], rows[0])
            self.assertEqual(got[1], {"id": B, "reasons": [], "new_s": None, "prior_s": None, "input_ratio": None})


class CapAndChunks(unittest.TestCase):
    def rows(self, n, s):
        return [{"id": "f/l%d/d/rows=full" % i, "reasons": ["slower"], "new_s": s, "prior_s": s, "input_ratio": 2.0}
                for i in range(n)]

    def test_split_parts_fit_under_the_cap(self):
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "rows.tsv")
            Path(f).write_text(TR.rows_text(self.rows(40, 600)))   # 40 x 20 min of both arms: about 800 min
            est, split = TR.check_cap(f, 230)
            self.assertGreater(est["total_min"], 230)
            self.assertGreaterEqual(len(split), 4)
            got = []
            for path, e in split:
                self.assertLessEqual(e["total_min"], 230)
                got += [r["id"] for r in TR.read_rows(path)]
            self.assertEqual(got, [r["id"] for r in TR.read_rows(f)])     # every row once, in order
            self.assertEqual(TR.check_cap(split[0][0], 230)[1], [])

    def test_chunks_interleave_size(self):
        by = {r["id"]: r for r in self.rows(6, 300)}
        ids = list(by)
        self.assertEqual(TR.chunk_ids(ids, by, 600), [ids[0:2], ids[2:4], ids[4:6]])
        self.assertEqual(TR.chunk_ids(ids, by, 0), [[i] for i in ids])      # strict interleave: one race a chunk
        self.assertEqual(TR.chunk_ids(ids, {}, 600), [ids[0:4], ids[4:6]])   # no estimate: chunk_s / 4 each


class BoxPlan(unittest.TestCase):
    def test_plan_keeps_rows_order_and_filters_recipes(self):
        planner = TR.PC.load_planner(PLANNER)
        full = TR.PC.plan_all(planner, "nvidia")[0]
        x, y = full[5], full[1]
        with tempfile.TemporaryDirectory() as d:
            rows = os.path.join(d, "rows.tsv")
            Path(rows).write_text("%s\tslower\t10\t10\t2\n%s\tchanged\t5\t5\t-\nnope/gone/x/rows=full\tpaper\n" % (x, y))
            rec = os.path.join(d, "recipes.json")
            Path(rec).write_text(json.dumps({"recipes": [{"race_id": y, "key": "k"}, {"race_id": "other"}],
                                             "race_ids": [y, "other"], "source": {}}))
            out = os.path.join(d, "plan")
            with mock.patch("sys.stdout"):
                TR.main(["plan", "--rows", rows, "--vendor", "nvidia", "--planner", PLANNER, "--recipes", rec,
                         "--out-dir", out])
            self.assertEqual(Path(out, "race-ids-full.txt").read_text().split(), [x, y])
            self.assertEqual(len(Path(out, "race-ids-smoke.txt").read_text().split()), 2)
            self.assertEqual(list(json.loads(Path(out, "not-planned.json").read_text())), ["nope/gone/x/rows=full"])
            r = json.loads(Path(out, "recipes.json").read_text())
            self.assertEqual((r["race_ids"], len(r["recipes"])), ([y], 1))
            self.assertEqual(len(json.loads(Path(out, "races.json").read_text())["recipes"]), 2)
            self.assertEqual(json.loads(Path(out, "owed-paper.json").read_text())["race_ids"], [])


class LaneMap(unittest.TestCase):
    PKG = Path(HERE).parent / "python" / "mojolearn"

    def test_call_maps_to_module_and_bindings(self):
        lazy = TR.lazy_names(self.PKG)
        race = {"cells": [{"arm": "ours", "call": "mojolearn.DBSCAN(algorithm='brute', eps=3)"}]}
        f, why = TR.race_module(race, lazy, self.PKG)
        self.assertEqual((f.name, why), ("density.py", None))
        self.assertIn("_mojolearn_estimators", TR.module_bindings(f))
        self.assertEqual(TR.script_of("_mojolearn_x_trees"), "build_x_trees.sh")
        self.assertEqual(TR.script_of("_mojolearn"), "build.sh")

    def test_uncertain_without_a_call(self):
        f, why = TR.race_module({"cells": [{"arm": "ours"}]}, TR.lazy_names(self.PKG), self.PKG)
        self.assertIsNone(f)
        self.assertIn("no public estimator", why)

    def test_changed_reason_from_the_closure_digests(self):
        races = {A: {"cells": [{"arm": "ours", "call": "mojolearn.KNeighborsRegressor()"}]},
                 B: {"cells": [{"arm": "ours", "call": "mojolearn.DBSCAN()"}]},
                 C: {"cells": [{"arm": "ours"}]}}
        lazy = TR.lazy_names(self.PKG)
        fa, _ = TR.race_module(races[A], lazy, self.PKG)
        scripts_a = {TR.script_of(b) for b in TR.module_bindings(fa)}
        closures = {s: {"sources": ["core/x.mojo"]} for s in ("build_estimators.sh", "build_estimators_host.sh")}
        closures.update({s: {"sources": []} for s in scripts_a})
        changed = {s: s == "build_estimators.sh" for s in closures}
        with mock.patch.object(TR, "closure_change", return_value=(changed, ["core/x.mojo"], closures)):
            out, notes = TR.changed_races([A, B, C, D], {"races": races}, "HEAD")
        self.assertEqual(out.get(B), "changed")            # DBSCAN -> density.py -> _mojolearn_estimators changed
        self.assertEqual(out.get(C), "changed?")           # no call recorded: included
        self.assertEqual(out.get(D), "changed?")           # never ran in the stored board: included
        if "build_estimators.sh" not in scripts_a:
            self.assertNotIn(A, out)
        self.assertIn("closures changed 1 of", notes[0])


class Summary(unittest.TestCase):
    def test_summary_counts_over_fixed_done_and_ab(self):
        doc = compare_doc()
        doc["rows"][0].update(ratio=1.3, flags=["REGRESSION"])
        doc["extra_rows"][0]["priors"]["0.8.55"]["ratio"] = 0.95
        doc["extra_rows"][0]["flags"] = ["SAME"]
        rows = [{"id": A, "reasons": ["paper"], "input_ratio": 1.5}, {"id": C, "reasons": ["slower"],
                                                                        "input_ratio": 1.2}]
        with tempfile.TemporaryDirectory() as res:
            for sub, ids in (("board-tg", [A, C]), ("board-tg-prior-0.8.55", [A])):
                os.makedirs(os.path.join(res, sub))
                Path(res, sub, "board.json").write_text(json.dumps(board({i: 1 for i in ids})))
            cell = lambda ms: {"arm": "ours", "status": "ok", "median_ms": ms}  # noqa: E731
            for sub, ms in (("board-tg-ab-em-check-1-base-full", 10.0), ("board-tg-ab-em-check-1-full", 8.0)):
                os.makedirs(os.path.join(res, sub))
                Path(res, sub, "board.json").write_text(json.dumps({"races": {B: {"cells": [cell(ms)]}}}))
            text = TR.summary(doc, rows, res, [A, C], [("em-check-1", "full", [B], None)])
        self.assertIn("timed=2 FASTER=0 SAME=1 REGRESSION=1 BOX-STATE=0", text)
        self.assertIn("1.30x " + A + " fit ours", text)
        self.assertIn("fixed (over 1.10 in the input compare, now at or under 1.00): 1", text)
        self.assertIn(C + " (1.20x -> 0.95x)", text)
        self.assertIn("done/total candidate 2/2", text)
        self.assertIn("done/total 0.8.55 1/2", text)
        self.assertIn("AB em-check-1 rows=full " + B, text)
        self.assertIn("ratio 0.80", text)

    def test_ab_against_a_stored_base(self):
        # STORED BASE (Andrew, 2026-11-17): no base board; the arm against the stored time, its host printed, and a
        # run stopped at the cap reads NOT-FASTER-THAN-BASE
        stored = {"round_seconds": {B: 10, C: 10}, "cells": {
            B + " fit ours": {"ms": 10000.0, "host_id": "pod-old/GPU-1", "version": "wheel abc", "cap": True},
            C + " fit ours": {"ms": 10000.0, "host_id": "pod-old/GPU-1", "version": "wheel abc", "cap": True}}}
        with tempfile.TemporaryDirectory() as res:
            os.makedirs(os.path.join(res, "board-tg-ab-x-full"))
            Path(res, "board-tg-ab-x-full", "board.json").write_text(json.dumps({
                "box": {"host_probe": {"schema": "mojolearn-host-probe/1", "host_id": "pod-a/GPU-0"}},
                "races": {B: {"cells": [{"arm": "ours", "status": "ok", "median_ms": 5000.0}]},
                          C: {"cells": [{"arm": "ours", "status": "REFUSED(timeout: null)", "median_ms": None}]}}}))
            text = TR.summary(compare_doc(), [], res, [], [("x", "full", [B, C], stored)])
        self.assertIn("base STORED, not timed in this job: host pod-old/GPU-1 (wheel abc); the arm ran on pod-a/GPU-0 "
                      "(ANOTHER HOST", text)
        self.assertIn("AB x rows=full %s fit ours: base 10000 (stored) -> arm 5000 ms ratio 0.50 FASTER"
                      % B, text)
        self.assertIn(C + " fit ours: base", text)
        self.assertIn("NOT-FASTER-THAN-BASE", text)


class OwnOnly(unittest.TestCase):
    """--own-only (Andrew, 2026-11-17: "we only run A/B and what has changed"): the rows a merged fix covers, from the
    lanes' per-row result files, plus the internal row; controls default 0; one-arm estimate."""
    PLAN = [A, B, C, D, INTERNAL, "algos/gcn/taxi/rows=full", "algos/gcn/istella/rows=full",
            "algos/graphsage/taxi/rows=full", "trees/gbdt-symmetric/istella/rows=full"]
    MERGED = {"aaaaaaaaaa", "15cca53af1"}

    def files(self, d):
        Path(d, "fb-x-RESULT.tsv").write_text(
            "row\tratio\tclass\tcause\tfix\n"
            "classical/knn/istella/rows=full fit ours\t1.5\ta\t6869626579 caused it\taaaaaaaaaa (door)\n"
            "dbscan/istella fit\t1.3\tb\twarmup\tharness (fixed by aaaaaaaaaa)\n"
            "trees/rf/istella fit\t1.2\t?\t-\tnot found (the cap aaaaaaaaaa may cover it)\n"
            "classical/pca/taxi/rows=full fit ours\t1.2\ta\tx\tbbbbbbbbbb\n")
        Path(d, "fb2-y-RESULT.tsv").write_text(
            "row\tratio\tdelta\twork\tfix\n"
            "algos/gcn/taxi/rows=full fit ours\t1.1\t+1\tx\t15cca53af1 + needs rerun (harness already fixed)\n")
        Path(d, "fb4-ten-RESULT.md").write_text(
            "# t\n\n| row | delta | explanation | status |\n|---|---|---|---|\n"
            "| gcn / graphsage fit (4 rows) | +0.5 ms | aaaaaaaaaa cited | 15cca53af1 by arithmetic |\n"
            "| gbdt-symmetric istella infer | 66 -> 108 | Not a regression: same wheel 65 ms back to back | - |\n"
            "| samba-train-step | +3 ms | real, see aaaaaaaaaa | Open. |\n")

    def select(self, d):
        entries = TR.result_entries(sorted(str(p) for g in TR.RESULT_GLOBS for p in Path(d).glob(g)))
        return TR.own_only_select(self.PLAN, entries, lambda s: s in self.MERGED)

    def test_fix_rows_exclusions_and_veto(self):
        with tempfile.TemporaryDirectory() as d:
            self.files(d)
            fixes, not_sel, unmapped = self.select(d)
        self.assertEqual(fixes[A], ["aaaaaaaaaa"])                    # the fix column, never the cause column
        self.assertNotIn(B, fixes)                                    # class b: harness, whatever sha it names
        self.assertNotIn(C, fixes)                                    # "not found (... may cover it)": a guess
        self.assertNotIn(D, fixes)                                    # bbbbbbbbbb is not merged
        self.assertEqual(fixes["algos/gcn/taxi/rows=full"], ["15cca53af1"])     # sha + "harness" words: assigned
        self.assertEqual(fixes["algos/graphsage/taxi/rows=full"], ["15cca53af1"])  # prose row, every dataset
        self.assertIn("algos/gcn/istella/rows=full", fixes)
        self.assertNotIn("trees/gbdt-symmetric/istella/rows=full", fixes)    # back to back: not a regression
        why = {e["row"]: w for e, w in not_sel}
        self.assertTrue(why["dbscan/istella fit"].startswith("harness"))
        self.assertTrue(why["trees/rf/istella fit"].startswith("no cause found"))
        self.assertIn("not on", why["classical/pca/taxi/rows=full fit ours"])
        self.assertIn("vetoes any fix", why["gbdt-symmetric istella infer"])
        self.assertEqual([e["row"] for e, _ in unmapped], ["samba-train-step"])   # no such lane planned here

    def test_rows_controls_and_one_arm_estimate(self):
        with tempfile.TemporaryDirectory() as d:
            self.files(d)
            fixes, _, _ = self.select(d)
        rows, counts, _ = TR.own_only_rows(self.PLAN, fixes, compare_doc(), board({A: 60, B: 30}))
        self.assertEqual(counts["control"], 0)                        # default: no controls
        self.assertEqual(rows[0]["id"], A)                            # the worst input ratio first among the fixes
        self.assertEqual(rows[-1]["id"], INTERNAL)
        self.assertIn("fix=aaaaaaaaaa", rows[0]["reasons"])
        self.assertTrue(all(r["prior_s"] is None for r in rows))
        rows2, counts2, _ = TR.own_only_rows(self.PLAN, {}, compare_doc(), board({}), control=1, seed=1)
        self.assertEqual(counts2["control"], 1)
        self.assertEqual([r["id"] for r in rows2 if "control" in r["reasons"]], [B])   # the one SAME race
        two = TR.estimate([{"id": A, "reasons": [], "new_s": 60.0, "prior_s": 50.0}])
        one = TR.estimate([{"id": A, "reasons": [], "new_s": 60.0, "prior_s": 50.0}], one_arm=True)
        self.assertEqual(one["prior_s"], 0.0)
        self.assertLess(one["total_s"], two["total_s"] - 50)
        self.assertIn("0.8.55=stored (one arm)", TR.estimate_line(one))

    def test_stored_summary(self):
        doc = compare_doc()
        doc.update(baseline_stored=True, verdict_basis="STORED BASELINE, not timed in this job: 0.8.55 stored (x)",
                   baseline_warnings=["WARNING: GPU driver a (stored 0.8.55) vs b (this job)"])
        doc["rows"][0]["flags"] = ["BOX-STATE"]
        doc["rows"][0]["upload_probe"] = {"new_gbps": 10.0, "new_ms": 88.0, "stored_gbps": 14.0, "stored_ms": 63.0,
                                          "diff": -0.29, "source": "race"}
        rows = [{"id": A, "reasons": ["fix"], "input_ratio": 1.5}, {"id": B, "reasons": ["control"],
                                                                     "input_ratio": 1.0}]
        with tempfile.TemporaryDirectory() as res:
            text = TR.summary(doc, rows, res, [A, B], [], None)
        lines = text.splitlines()
        self.assertTrue(lines[0].startswith("STORED BASELINE"))
        self.assertTrue(lines[1].startswith("WARNING"))
        self.assertIn("FASTER=0 SAME=1 REGRESSION=0 BOX-STATE=1", text)
        self.assertIn("still over 1.10 (0)", text)
        self.assertIn("control rows that got slower (over 1.10): 0 of 1", text)
        self.assertIn("1.50x " + A + " fit ours; probe 10.00 GB/s (88.00 ms) vs stored 14.00 GB/s", text)
        self.assertIn("0.8.55: stored, not timed in this job", text)
        self.assertNotIn("done/total 0.8.55", text)

    def test_box_rows_tsvs_parse_from_the_header(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "fb-site-RESULT.tsv").write_text("site\tstatus\nx.mojo:1\tfixed aaaaaaaaaa\n")
            self.assertEqual(TR.result_entries([str(Path(d, "fb-site-RESULT.tsv"))]), [])   # per-site table


class Wiring(unittest.TestCase):
    def test_job_script_targeted_block(self):
        text = Path(JOB).read_text()
        self.assertEqual(subprocess.run(["bash", "-n", JOB]).returncode, 0)
        self.assertIn('PARTS="$PARTS,%s,"' % ",".join(MC.TG_PARTS[1:]), text)
        self.assertIn("targeted_rows.py plan", text)
        self.assertIn('smoke_together tg- "$TG/smoke-ids-all.txt" "$RES/smoke-tg" --drop-race-scratch', text)
        self.assertIn('disk_gate tg-board "$left"', text)
        self.assertIn('tg_board candidate "$c" "$RES/board-tg"', text)          # interleaved: candidate, then prior
        self.assertIn('tg_board "$arm" "$c.$arm" "$RES/board-tg-$arm"', text)
        self.assertIn('"$RES/board-tg-ab-$arm-base-$rows"', text)               # A/B: base and arm back to back
        self.assertIn("--drop-race-scratch", text)
        self.assertIn("targeted_rows.py summary", text)

    def test_stage_parts_and_dataset_keys(self):
        self.assertTrue(MC.runs_targeted(MC.parse_parts("targeted")))
        self.assertFalse(MC.runs_targeted(MC.parse_parts("board-all")))
        planner = TR.PC.load_planner(PLANNER)
        full = TR.PC.plan_all(planner, "nvidia")[0]
        algos = next(i for i in full if i.startswith("algos/") and "/istella/" in i)
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "rows.tsv")
            Path(f).write_text(algos + "\n")
            keys = MC.dataset_keys("nvidia", None, None, PLANNER, parts=["targeted"], rows_file=f)
        self.assertTrue(keys)
        self.assertLess(len(keys), len(MC.dataset_keys("nvidia", None, None, PLANNER, parts=["board-all"])))

    def test_job_stored_baseline_and_env_arms(self):
        text = Path(JOB).read_text()
        self.assertIn('BASELINE_BOARD=${PC_BASELINE_BOARD:-}', text)
        self.assertIn('[ -n "$BASELINE_BOARD" ] && break', text)                 # no prior arm at all
        self.assertIn('PB=(--stored-baseline-board "$BASELINE_BOARD")', text)
        self.assertIn('[ -n "${ARM_ENV[$arm]:-}" ] && continue', text)          # an env arm installs nothing
        self.assertIn('env ${envs[@]+"${envs[@]}"} $PY tools/bench_board.py', text)

    def test_job_env_arm_line(self):
        block = Path(JOB).read_text()
        block = block[block.index("# >>> b-arms"):block.index("# <<< b-arms")]
        with tempfile.TemporaryDirectory() as td:
            f = os.path.join(td, "b.txt")
            Path(f).write_text("mcd-chol env:MOJOLEARN_CPU_THREADS=28,MOJOLEARN_XPREP_R3_OFF=view %s\n"
                               "bad-env env:lower=1 %s\n" % ("a" * 40, "a" * 40))
            script = ("declare -A ARM_ARGS ARM_PY ARM_MAN; ARM_ARGS[candidate]='--mojolearn-version 1.0'; "
                      "ARM_PY[candidate]=/v/py; ARM_MAN[candidate]=/m.json\nsay() { echo \"SAY $*\"; }\n"
                      "add_arm() { echo \"ARM $*\"; }\n" + block +
                      '\necho "B_CSV=$B_CSV ENV=${ARM_ENV[mcd-chol]} PY=${ARM_PY[mcd-chol]}"\n')
            env = dict(os.environ, OWED="1", V="nvidia", PC_B_ARMS=f, PARTS=",targeted,")
            p = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("ARM mcd-chol --mojolearn-version 1.0", p.stdout)
        self.assertIn("B_CSV=mcd-chol ENV=MOJOLEARN_CPU_THREADS=28,MOJOLEARN_XPREP_R3_OFF=view PY=/v/py", p.stdout)
        self.assertIn("SAY bad env arm bad-env", p.stdout)

    def test_stage_env_arm_parse(self):
        self.assertEqual(MC.parse_b_arm("mcd-chol=env:MOJOLEARN_CPU_THREADS=28"),
                         ("mcd-chol", "env:MOJOLEARN_CPU_THREADS=28", None))
        name, d, sha = MC.parse_b_arm("x=/tmp/w@" + "a" * 40)
        self.assertEqual((name, sha), ("x", "a" * 40))
        with self.assertRaises(SystemExit):
            MC.parse_b_arm("x=env:lower=1")
        src = Path(MC.__file__).read_text()
        self.assertIn('env["PC_BASELINE_BOARD"]', src)
        self.assertIn("one_arm=stored", src)

    def test_stage_source_refuses_over_the_cap(self):
        src = Path(MC.__file__).read_text()
        self.assertIn("TR.check_cap(a.rows, a.cap_min, a.chunk_s, one_arm=stored)", src)
        self.assertIn('env["PC_ROWS"]', src)


if __name__ == "__main__":
    unittest.main()
