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
import shutil
import subprocess
import sys
import tempfile
import time
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

    def test_estimate_from_measured_overheads(self):
        # lane fx-pod-job (2026-11-18): the A/B job was estimated at 224 min and took 101. The prep is counted once a
        # job, a process start is the measured 16 s, and no per-race gap is added on top of the races' own wall
        rows = self.rows(10, 60)
        est = TR.estimate(rows, 600, one_arm=True)
        self.assertFalse(hasattr(TR, "RACE_GAP_S"))
        want = TR.FIXED_S + TR.PREP_S + TR.PROC_S * est["chunks"]
        self.assertAlmostEqual(est["overhead_s"], want)
        more = TR.estimate(self.rows(20, 60), 600, one_arm=True)
        self.assertAlmostEqual(more["overhead_s"] - est["overhead_s"], TR.PROC_S * (more["chunks"] - est["chunks"]))
        self.assertIn("s of prep once", TR.estimate_line(est))

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

    def test_ab_capped_at_this_jobs_base(self):
        # IN-JOB BASE CAP (lane fx-same-pod-prior, 2026-11-19; the STORED BASE from another host is gone): base and
        # arm boards of this job; an arm run stopped at the cap reads NOT-FASTER-THAN-BASE, only for a capped race
        with tempfile.TemporaryDirectory() as res:
            probe = {"host_probe": {"schema": "mojolearn-host-probe/1", "host_id": "pod-a/GPU-0"}}
            for sub, races in (("board-tg-ab-x-base-full", {B: 10000.0, C: 10000.0, D: 10000.0}),
                               ("board-tg-ab-x-full", {B: 5000.0, C: None, D: None})):
                os.makedirs(os.path.join(res, sub))
                Path(res, sub, "board.json").write_text(json.dumps({"box": probe, "races": {
                    rid: {"cells": [{"arm": "ours", "status": "ok" if ms else "REFUSED(timeout: null)",
                                     "median_ms": ms}]} for rid, ms in races.items()}}))
            text = TR.summary(compare_doc(), [], res, [], [("x", "full", [B, C, D], {B, C})])
        self.assertNotIn("STORED", text)
        self.assertIn("AB x rows=full %s fit ours: base 10000 -> arm 5000 ms ratio 0.50 FASTER" % B, text)
        self.assertIn("%s fit ours: base 10000 -> arm - ms ratio - NOT-FASTER-THAN-BASE" % C, text)
        line_d = [ln for ln in text.splitlines() if ln.startswith("AB x rows=full %s" % D)][0]
        self.assertNotIn("NOT-FASTER-THAN-BASE", line_d)          # not in the cap list: no verdict

    def test_read_units_reads_the_cap_list(self):
        with tempfile.TemporaryDirectory() as d:
            ids = os.path.join(d, "ids-x-full.txt")
            Path(ids).write_text(B + "\n" + C + "\n")
            Path(d, "ids-x-full.cap.txt").write_text(B + "\n")
            Path(d, "units.tsv").write_text("candidate\tfull\t%s\nx\tfull\t%s\n" % (ids, ids))
            self.assertEqual(TR.read_units(os.path.join(d, "units.tsv")), [("x", "full", [B, C], {B})])
            # a unit whose ids file is gone stays, with no race (lane fx-tools-round3: never dropped silently)
            Path(d, "units.tsv").write_text("x\tfull\t%s\ny\tfull\t%s\n" % (ids, os.path.join(d, "gone.txt")))
            self.assertEqual(TR.read_units(os.path.join(d, "units.tsv"))[1], ("y", "full", [], set()))


class PriorWheel(unittest.TestCase):
    """--prior-wheel (lane fx-same-pod-prior; Andrew, 2026-11-19: a speed verdict compares two wheels timed on the SAME
    host in the SAME job; 0.8.55 is never installed or timed): one of our own earlier built wheels is one more arm of
    the targeted job, installed as a B set, timed after the candidate per chunk, and the compare's verdict baseline."""

    def rows(self, d):
        f = os.path.join(d, "rows.tsv")
        Path(f).write_text("%s\tr\t600\t500\n%s\tr\t300\t200\n" % (A, B))
        return f

    def test_estimate_counts_each_prior_wheel_as_an_arm(self):
        rows = [{"id": A, "new_s": 600.0, "prior_s": 500.0}, {"id": B, "new_s": 300.0, "prior_s": 200.0}]
        one = TR.estimate(rows, 600, one_arm=True)
        two = TR.estimate(rows, 600, one_arm=True, prior_wheels=1)
        self.assertEqual(two["wheels_s"], 900.0)
        self.assertAlmostEqual(two["total_s"] - one["total_s"],
                               900.0 + TR.SMOKE_S_PER_RACE * 2 + TR.PROC_S * one["chunks"])
        self.assertIn("prior-wheels=15 (1)", TR.estimate_line(two))
        self.assertIn("stored (screen)", TR.estimate_line(two))
        self.assertNotIn("prior-wheels", TR.estimate_line(one))

    def test_job_reads_a_prior_wheels_file(self):
        text = Path(JOB).read_text()
        block = text[text.index("# >>> b-arms"):text.index("# <<< b-arms")] + \
            text[text.index("# >>> prior-wheels"):text.index("# <<< prior-wheels")]
        with tempfile.TemporaryDirectory() as td:
            d = os.path.join(td, "p-prev", "nvidia")
            os.makedirs(d)
            for dist in ("core", "nvidia"):
                Path(d, "mojolearn_%s-0.8.67.dev0+d95f805cceba-py3-none-any.whl" % dist).write_text("")
            f = os.path.join(td, "prior-wheels-nvidia.txt")
            Path(f).write_text("# staged by measure_candidate\nprev %s %s\n" % (d, "d" * 40))
            script = ("set -u\nsay() { echo \"SAY $*\"; }\nadd_arm() { echo \"ARM $*\"; }\nPRIOR_ARMS=()\n" + block
                      + '\necho "PRIOR_ARMS=${PRIOR_ARMS[*]} NAMES=${PRIOR_WHEEL_NAMES[*]} '
                        'VP=$(IFS=,; echo "${PRIOR_ARMS[*]#prior-}")"\n')
            env = dict(os.environ, OWED="1", V="nvidia", PC_PRIOR_WHEELS=f, PARTS=",targeted,")
            p = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn("ARM prior-prev --mojolearn-version 0.8.67.dev0+d95f805cceba --candidate-source-commit "
                          + "d" * 40 + " --candidate-wheel " + d, p.stdout)
            self.assertIn("PRIOR_ARMS=prior-prev NAMES=prev VP=prev", p.stdout)
            self.assertIn("SAY prior wheels (the verdict baseline: prev): prev", p.stdout)
            # a set without its two wheels stops the job before anything is timed
            Path(f).write_text("prev %s %s\n" % (td, "d" * 40))
            p = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(p.returncode, 2)
            self.assertIn("must hold one core and one nvidia wheel", p.stdout + p.stderr)
            # the paper job does not take one
            p = subprocess.run(["bash", "-c", script], env=dict(env, PARTS=",plan,board,"), capture_output=True,
                               text=True, timeout=60)
            self.assertEqual(p.returncode, 2)
            self.assertIn("PC_PRIOR_WHEELS is for the targeted job", p.stdout)

    def test_job_wires_the_prior_wheel_through_install_board_and_compare(self):
        text = Path(JOB).read_text()
        # tg-install makes every non-B arm's venv: the candidate and the prior wheels
        self.assertIn('[[ " ${B_NAMES[*]:-} " == *" $arm "* ]] && continue', text)
        # the timed boards: the candidate then each prior arm on the same chunk
        i = text.index('tg_board candidate "$c" "$RES/board-tg" "" "${SJT[@]}"')
        self.assertLess(i, text.index('tg_board "$arm" "$c.$arm" "$RES/board-tg-$arm" "" "${psj[@]}"', i))
        # the compare: the prior wheel's board, the stored board as a screen, the verdict from the pair
        self.assertIn('[ -n "$BASELINE_BOARD" ] && PB+=(--stored-baseline-board "$BASELINE_BOARD")', text)
        self.assertIn('PB+=(--verdict-prior "${PRIOR_WHEEL_NAMES[0]}")', text)
        self.assertIn('--prior-wheels "$(IFS=,; echo "${PRIOR_WHEEL_NAMES[*]:-}")"', text)
        self.assertIn('--prior-wheels "${#PRIOR_WHEEL_NAMES[@]}"', text)

    def test_timed_check_names_an_untimed_prior_wheel(self):
        with tempfile.TemporaryDirectory() as res:
            os.makedirs(os.path.join(res, "board-tg"))
            Path(res, "board-tg", "board.json").write_text(json.dumps({"races": {A: {"status": "done"}}}))
            bad = TR.untimed(res, [A], None, None, ["prev"])
            self.assertEqual([arm for arm, _ in bad], ["prior-prev"])
            os.makedirs(os.path.join(res, "board-tg-prior-prev"))
            Path(res, "board-tg-prior-prev", "board.json").write_text(json.dumps({"races": {A: {"status": "done"}}}))
            self.assertEqual(TR.untimed(res, [A], None, None, ["prev"]), [])

    def test_summary_reads_the_compares_baseline(self):
        doc = dict(compare_doc(), baseline_version="prev", screen_stored=True,
                   verdict_basis="same box h: our wheel prev timed in this job beside the candidate")
        with tempfile.TemporaryDirectory() as res:
            text = TR.summary(doc, [{"id": A, "reasons": ["r"]}], res, [A], [], ["prev"])
        self.assertIn("verdict: same box h: our wheel prev", text)
        self.assertIn("the stored 0.8.55 board is a screen column only", text)
        self.assertIn("done/total prev 0/1", text)

    def test_stage_dry_run_accepts_a_prior_wheel(self):
        # the completion check of the brief: `stage nvidia ... --rows <file> --prior-wheel prev=DIR@SHA --dry-run`
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as d:
            cver, pver = "0.8.67.dev0+aaaaaaaaaaaa", "0.8.67.dev0+d95f805cceba"
            wd, pd = os.path.join(d, "cand"), os.path.join(d, "prev")
            for wdir, ver in ((wd, cver), (pd, pver)):
                os.makedirs(wdir)
                for dist in ("core", "nvidia"):
                    Path(wdir, "mojolearn_%s-%s-py3-none-any.whl" % (dist, ver)).write_text("w")
            Path(pd, "candidate.json").write_text(json.dumps({"version": pver}))
            tree = os.path.join(d, "tree")
            os.makedirs(os.path.join(tree, "bench", "results", "paper_compare"))
            os.makedirs(os.path.join(tree, "bench", "results", "dataset_store"))
            Path(tree, "bench", "results", "paper_compare", "recipes-nvidia.json").write_text("{}")
            Path(tree, "bench", "results", "dataset_store", "manifest.tsv").write_text("")
            owed = os.path.join(d, "owed.json")
            Path(owed).write_text(json.dumps({"b_arms": {}, "rows": []}))
            argv = ["stage", "nvidia", "--version", cver, "--source-commit", "a" * 40, "--wheel-dir", wd,
                    "--dry-run", "--state-dir", d, "--tree", tree, "--owed-file", owed, "--rows", self.rows(d),
                    "--prior-wheel", "prev=%s@%s" % (pd, "d" * 40)]
            seen = []

            def manifest(version, vendor, wheels, commit):
                seen.append((version, commit))
                return {"files": [{"file": Path(w).name, "sha256": "0" * 64} for w in wheels]}
            buf = io.StringIO()
            with mock.patch.object(MC.C, "build_manifest", manifest), \
                    mock.patch.object(MC, "dataset_keys", lambda *a, **k: ["k1"]), \
                    contextlib.redirect_stdout(buf):
                self.assertEqual(MC.main(argv + ["--baseline-board", "none"]), 0)
            text = buf.getvalue()
            self.assertEqual(seen, [(cver, "a" * 40), (pver, "d" * 40)])      # validated like a B set
            self.assertIn("prior wheel prev: %s dddddddddddd" % pver, text)
            self.assertIn("/root/pc-in/aaaaaaaaaaaa/p-prev/nvidia/mojolearn_core-%s-py3-none-any.whl" % pver, text)
            self.assertIn("--env PC_PRIOR_WHEELS=/root/pc-in/aaaaaaaaaaaa/prior-wheels-nvidia.txt", text)
            self.assertIn("prior-wheels=15 (1)", text)                        # the estimate counts it
            self.assertIn("TG-DISK need=", text)
            self.assertIn("3 venv(s) in tg-install", text)                   # candidate, 0.8.55 (none), prev
            self.assertEqual(Path(d, "prior-wheels-aaaaaaaaaaaa-nvidia.txt").read_text(),
                             "prev /root/pc-in/aaaaaaaaaaaa/p-prev/nvidia %s\n" % ("d" * 40))
            # beside the stored board (the default): 0.8.55 is not timed, the stored board rides as a screen
            board = os.path.join(d, "stored.json")
            Path(board).write_text("{}")
            buf = io.StringIO()
            with mock.patch.object(MC.C, "build_manifest", manifest), \
                    mock.patch.object(MC, "dataset_keys", lambda *a, **k: ["k1"]), \
                    contextlib.redirect_stdout(buf):
                self.assertEqual(MC.main(argv + ["--baseline-board", board]), 0)
            text = buf.getvalue()
            self.assertIn("2 venv(s) in tg-install", text)
            self.assertIn("PC_BASELINE_BOARD=", text)
            self.assertNotIn("PC_PRIOR_VERSIONS", text)
            self.assertIn("PC_PRIOR_WHEELS=", text)
            self.assertIn("stored (screen)", text)

    def test_stage_refuses_a_prior_wheel_out_of_place(self):
        with self.assertRaises(SystemExit) as e:
            MC.check_prior_wheels(["prev"], [], ["0.8.55"], None)
        self.assertIn("targeted job only", str(e.exception))
        with self.assertRaises(SystemExit) as e:
            MC.check_prior_wheels(["prev", "prev"], [], [], "rows.tsv")
        self.assertIn("given twice", str(e.exception))
        with self.assertRaises(SystemExit) as e:
            MC.check_prior_wheels(["b1"], ["b1"], [], "rows.tsv")
        self.assertIn("also a --b-arm", str(e.exception))
        with self.assertRaises(SystemExit):
            MC.parse_prior_wheel("prev=/x@abc")
        self.assertEqual(MC.parse_prior_wheel("prev-d95f805cce=~/w@" + "d" * 40)[0], "prev-d95f805cce")


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
        self.assertIn('PB+=(--stored-baseline-board "$BASELINE_BOARD")', text)
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

    def test_job_env_arm_value_from_the_box(self):
        # lane fb15-ab-verdicts (2026-11-18): =28 on a 14-core cgroup measured nothing; @cgroup_cores is the box's
        block = Path(JOB).read_text()
        block = block[block.index("# >>> b-arms"):block.index("# <<< b-arms")]
        with tempfile.TemporaryDirectory() as td:
            f = os.path.join(td, "b.txt")
            Path(f).write_text("cpu-threads-cgroup env:MOJOLEARN_CPU_THREADS=@cgroup_cores %s\n" % ("a" * 40))
            script = ("declare -A ARM_ARGS ARM_PY ARM_MAN; ARM_ARGS[candidate]='--mojolearn-version 1.0'; "
                      "ARM_PY[candidate]=/v/py; ARM_MAN[candidate]=/m.json\nsay() { echo \"SAY $*\"; }\n"
                      "alloc_cores() { echo 14; }\nadd_arm() { echo \"ARM $*\"; }\n" + block +
                      '\necho "ENV=${ARM_ENV[cpu-threads-cgroup]}"\n')
            env = dict(os.environ, OWED="1", V="nvidia", PC_B_ARMS=f, PARTS=",targeted,")
            p = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("ENV=MOJOLEARN_CPU_THREADS=14", p.stdout)
        self.assertEqual(MC.parse_b_arm("cpu-threads-cgroup=env:MOJOLEARN_CPU_THREADS=@cgroup_cores"),
                         ("cpu-threads-cgroup", "env:MOJOLEARN_CPU_THREADS=@cgroup_cores", None))

    @unittest.skipUnless(shutil.which("setsid"), "setsid (util-linux) is on the boxes, not on macOS")
    def test_job_cap_guard_stops_the_board_and_refuses_the_next(self):
        # lane fb15-ab-verdicts (2026-11-18): ab1 was killed at the cap and left no SUMMARY file
        block = Path(JOB).read_text()
        block = block[block.index("# >>> cap-guard"):block.index("# <<< cap-guard")]
        with tempfile.TemporaryDirectory() as td:
            script = ("say() { echo \"SAY $*\"; }\nP_T0=$(date +%s)\nJ=" + td + "\n" + block +
                      "\nt0=$(date +%s); run_timed " + td + "/b1.log 0 sleep 60; r1=$?; t1=$(date +%s)"
                      "\nrun_timed " + td + "/b2.log 0 true; r2=$?"
                      "\necho \"R1=$r1 R2=$r2 TOOK=$(( t1 - t0 ))\"; kill $CAP_GUARD 2> /dev/null; exit 0\n")
            env = dict(os.environ, GQ_DEADLINE=str(int(time.time()) + 3), PC_CAP_GUARD_S="1")
            p = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=90)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("SAY CAP GUARD", p.stdout)
        self.assertIn("R2=124", p.stdout)
        took = int(p.stdout.split("TOOK=")[1].split()[0])
        self.assertLess(took, 30, p.stdout)

    def test_job_b_arm_venvs_by_window(self):
        # lane fx-pod-job (2026-11-18): every arm's venv at once (1.3 GB each) did not fit a 42-arm job on the 80 GB
        # pod; a window of arms is installed before the first's unit and each venv goes after its arm's last unit
        block = Path(JOB).read_text()
        block = block[block.index("# >>> arm-window"):block.index("# <<< arm-window")]
        with tempfile.TemporaryDirectory() as td:
            fake = os.path.join(td, "fakepy")
            Path(fake).write_text('#!/bin/bash\nwhile [ $# -gt 0 ]; do [ "$1" = --venv ] && v=$2; shift; done\n'
                                  'mkdir -p "$v/bin" && : > "$v/bin/python" && chmod +x "$v/bin/python"\n'
                                  'echo "INSTALLED $(basename "$v")" >> "' + os.path.join(td, "installed") + '"\n')
            os.chmod(fake, 0o755)
            cache = os.path.join(td, "cache")
            os.makedirs(os.path.join(td, "logs"))
            head = ("set -u\nsay() { echo \"SAY $*\"; }\nalloc_cores() { echo 3; }\nPY=" + fake + "\nCACHE_ROOT=" + cache +
                    "\nRES=" + td + "\nTG=" + td + "\nmkdir -p $CACHE_ROOT\nCOMMON_TG=()\n"
                    # a 1 MB venv and no reserve: the window is the 3 cores on any disk (a CI runner's too)
                    "PC_VENV_MB=1\nPC_DISK_RESERVE_GB=0\n"
                    "declare -A ARM_ARGS ARM_PY ARM_MAN ARM_ENV\n"
                    "for a in a b c d e; do ARM_ARGS[$a]=''; ARM_PY[$a]=$CACHE_ROOT/venv-$a/bin/python; "
                    "ARM_MAN[$a]=" + td + "/m-$a.json; done\nARM_ENV[d]=X=1\n")
            script = (head + block + "\nWIN_ARMS=(a b c d e)\ninstall_window a\nls $CACHE_ROOT | tr '\\n' ' '; echo"
                      "\ndrop_arm_venv a\ninstall_window e\nls $CACHE_ROOT | tr '\\n' ' '; echo"
                      "\nEV_LEFT_MB=$(( 1 << 40 )); echo W=$(window_size)\n")
            p = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)
            # install_arms sends each install's output to its log, and the window's installs run at once: the
            # fake records each install in one file, in completion order
            installed = Path(os.path.join(td, "installed")).read_text().splitlines()
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(sorted(installed[:3]),
                         ["INSTALLED venv-a", "INSTALLED venv-b", "INSTALLED venv-c"])   # a window of 3 cores
        self.assertEqual(installed[3:], ["INSTALLED venv-e"])                            # e alone in the next one
        self.assertNotIn("INSTALLED venv-d", installed)                                  # an env arm installs nothing
        self.assertIn("venv-a venv-b venv-c ", p.stdout)
        self.assertIn("venv-b venv-c venv-e ", p.stdout)                                 # a deleted, e made
        self.assertIn("W=1", p.stdout)                                                   # no disk left: one at a time
        job = Path(JOB).read_text()
        tg = job[job.index("if part tg-install; then"):job.index("    if part tg-prep; then")]
        self.assertIn('[[ " ${B_NAMES[*]:-} " == *" $arm "* ]] && continue', tg)        # no B arm in tg-install
        owed = job[job.index("# the owed A/B arms: each unit chunk by chunk"):job.index("    if part tg-compare; then")]
        self.assertIn('install_window "$arm"', owed)
        self.assertIn('drop_arm_venv "$arm"', owed)
        # installs happen only between timed boards: never inside run_timed / tg_board
        self.assertLess(owed.index('install_window "$arm"'), owed.index("tg_board candidate"))

    def test_job_tg_prep_phase(self):
        # lane fx-pod-job (2026-11-18): the first timed process prepared graph-istella and graph-taxi one after the
        # other (486 s of its 695 s) while the GPU waited
        job = Path(JOB).read_text()
        self.assertIn("tg-install,tg-prep,tg-smoke,tg-board,tg-compare", job)
        prep = job[job.index("    if part tg-prep; then"):job.index("    if part tg-smoke; then")]
        self.assertIn("--preflight-only --prep-only", prep)
        self.assertIn('--cache "$CACHE_ROOT/c1"', prep)
        self.assertNotIn("fixtures-venv", prep.replace("No --fixtures-venv", ""))    # no shared fixture writer
        self.assertIn("targeted_rows.py prep-groups", prep)
        self.assertIn("tg-prep", MC.TG_PARTS)
        planner = TR.PC.load_planner(PLANNER)
        self.assertEqual(tuple(planner.DATASETS), TR.PREP_DATASETS)
        g = TR.prep_groups(["algos/a/taxi/rows=full", "algos/b/istella/rows=full", "algos/c/synthetic/rows=full",
                            "classical2/ivf/istella/rows=full", "algos/a/taxi/rows=full"])
        self.assertEqual(g, {"algos-taxi": ["algos/a/taxi/rows=full", "algos/c/synthetic/rows=full",
                                            "algos/a/taxi/rows=full"],
                             "algos-istella": ["algos/b/istella/rows=full"],
                             "classical2-istella": ["classical2/ivf/istella/rows=full"]})
        with tempfile.TemporaryDirectory() as td:
            a = os.path.join(td, "a.txt")
            Path(a).write_text("algos/a/taxi/rows=full\nalgos/b/istella/rows=full\n")
            self.assertEqual(TR.main(["prep-groups", "--ids", a, "--ids", os.path.join(td, "none.txt"),
                                      "--out-dir", os.path.join(td, "g")]), 0)
            self.assertEqual(sorted(os.listdir(os.path.join(td, "g"))), ["group-algos-istella.txt",
                                                                         "group-algos-taxi.txt"])

    def test_stage_disk_need_follows_the_arms(self):
        # lane fx-pod-job (2026-11-18): the need was a fixed 40 GB; a 42-arm job was cut to 14 by hand. With one
        # B-arm venv window the 42-arm A/B (about 5 races a unit, about 1.3 units an arm) fits an 80 GB pod
        units = 55
        need = MC.targeted_disk_gb(1, 1, 42 * 2 * 60 * 2 ** 20, 2 * 5 * units, 1 + 2 * units)
        self.assertLess(need, 70)
        self.assertGreater(MC.targeted_disk_gb(1, 1, 0, 2 * 5 * units, 1 + 2 * units),
                           MC.targeted_disk_gb(1, 1, 0, 2 * 5 * 10, 1 + 2 * 10))          # more arms, more disk
        self.assertGreaterEqual(MC.targeted_disk_gb(1, 0, 0, 0, 1), MC.RESERVE_GB + MC.SMOKE_SCRATCH_GB)
        free, got, _ = MC.disk_verdict("FREE_KB %d\n" % (70 * 1048576), "t", "abcdefabcdef", need_gb=need)
        self.assertEqual(got, need)
        src = Path(MC.__file__).read_text()
        self.assertIn("check_disk(a, runner, plan, tag, commit[:12], need_gb)", src)

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
        self.assertIn("TR.check_cap(a.rows, a.cap_min, a.chunk_s, one_arm=stored, prior_wheels=len(pw_specs))", src)
        self.assertIn('env["PC_ROWS"]', src)


def _board(path, ids, status="done"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Path(path).write_text(json.dumps({"schema": "mojolearn-bench-board/1", "races": {i: {"status": status} for i in ids}}))


class PodJob2(unittest.TestCase):
    """lane fx-pod-job2 (2026-11-18, nvc-ab-20261118T201200Z-d21ba0): 12 of 14 B arms were never timed (every
    worker race failed its smoke on a refused set_mempolicy, tools/test_numa_place.py), tg-prep sent the smoke ids to
    the full-row plan, no candidate board was written and the job said only 'no race to time'. A fake box dir with two
    arms: both smoked, both timed, the board written, a non-zero exit when one arm cannot be timed."""

    LISTED = "classical/knn/taxi/rows=full"
    RA, RB = "neural/lm-forward/bytes/shape=full", "trees/rf/istella/rows=full"

    def fake_box(self, td):
        res = os.path.join(td, "res")
        tg, tgo = os.path.join(res, "plan-tg"), os.path.join(res, "plan-tg-owed")
        for d in (tg, tgo, os.path.join(res, "logs")):
            os.makedirs(d, exist_ok=True)
        Path(tg, "race-ids-full.txt").write_text(self.LISTED + "\n")
        Path(tg, "race-ids-pass.txt").write_text(self.LISTED + "\n")
        units = []
        for arm, rid in (("arm-a", self.RA), ("arm-b", self.RB)):
            f = os.path.join(tgo, "ids-%s-full.txt" % arm)
            Path(f).write_text(rid + "\n")
            Path(f + ".pass").write_text(rid + "\n")          # its candidate smoke passed
            Path(f + ".pass.base").write_text(rid + "\n")
            units.append("%s\tfull\t%s" % (arm, f))
        units.insert(0, "candidate\tfull\t%s" % os.path.join(tgo, "ids-candidate-full.txt"))
        Path(tgo, "units.tsv").write_text("\n".join(units) + "\n")
        _board(os.path.join(res, "board-tg", "board.json"), [self.LISTED])
        for arm, rid in (("arm-a", self.RA), ("arm-b", self.RB)):
            _board(os.path.join(res, "board-tg-ab-%s-full" % arm, "board.json"), [rid])
            _board(os.path.join(res, "board-tg-ab-%s-base-full" % arm, "board.json"), [rid])
        return res, tg, tgo

    def check(self, res, tg, tgo, *extra):
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = TR.main(["timed-check", "--results", res, "--ids", os.path.join(tg, "race-ids-full.txt"),
                          "--units", os.path.join(tgo, "units.tsv")] + list(extra))
        return rc, buf.getvalue()

    def test_two_arms_timed_then_each_way_one_is_not(self):
        with tempfile.TemporaryDirectory() as td:
            res, tg, tgo = self.fake_box(td)
            rc, out = self.check(res, tg, tgo)
            self.assertEqual(rc, 0, out)
            self.assertIn("TG-TIMED-CHECK untimed=0", out)
            # arm-a: no race passed its smoke (the d21ba0 job's 12 arms), with the first failure named
            Path(tgo, "ids-arm-a-full.txt.pass").write_text("")
            Path(res, "logs", "pass-ids-tg-ab-arm-a.log").write_text(
                "PAPER-PASS-IDS-FAILED %s: smoke failed: -: preflight FAILED: worker_exited\n" % self.RA)
            # arm-b: its venv never came (no board)
            shutil.rmtree(os.path.join(res, "board-tg-ab-arm-b-full"))
            Path(res, "logs", "install-tg-arm-b.log").write_text("pip: no space left on device\n")
            rc, out = self.check(res, tg, tgo)
            self.assertEqual(rc, 3, out)
            self.assertIn("UNTIMED arm-a: full rows: none of its 1 race(s) passed the smoke: %s" % self.RA, out)
            self.assertIn("worker_exited", out)
            self.assertIn("UNTIMED arm-b: full rows: no board board-tg-ab-arm-b-full", out)
            self.assertIn("no space left", out)
            self.assertNotIn("UNTIMED candidate", out)
            # a split submit checks only its arms
            rc, out = self.check(res, tg, tgo, "--arms", "arm-a")
            self.assertEqual(rc, 3)
            self.assertNotIn("arm-b", out)

    def test_untimed_base_and_candidate(self):
        with tempfile.TemporaryDirectory() as td:
            res, tg, tgo = self.fake_box(td)
            _board(os.path.join(res, "board-tg-ab-arm-a-base-full", "board.json"), [self.RA], status="failed")
            Path(tg, "race-ids-pass.txt").write_text("")
            shutil.rmtree(os.path.join(res, "board-tg"))
            Path(res, "logs", "pass-ids-tg.log").write_text("PAPER-PASS-IDS-FAILED %s: smoke failed\n" % self.LISTED)
            rc, out = self.check(res, tg, tgo)
            self.assertEqual(rc, 3)
            self.assertIn("UNTIMED candidate: none of its 1 listed race(s) passed the smoke", out)
            self.assertIn("UNTIMED arm-a: full rows: its base (the candidate) has no race done", out)
            self.assertIn("failed 1", out)

    def test_empty_board_names_the_host_and_is_kept_or_replaced(self):
        with tempfile.TemporaryDirectory() as td:
            probe = os.path.join(td, "host-probe.json")
            Path(probe).write_text(json.dumps({"host_id": "h1/GPU-x", "cpu_model": "c"}))
            out = os.path.join(td, "board-tg")
            self.assertEqual(TR.main(["empty-board", "--out", out, "--host-probe", probe, "--why", "w"]), 0)
            doc = json.loads(Path(out, "board.json").read_text())
            self.assertEqual(doc["races"], {})
            self.assertEqual(doc["placeholder"], "w")
            self.assertEqual(TR.PC.board_host_ids(doc), {"h1/GPU-x"})        # the compare names this job's host
            _board(os.path.join(out, "board.json"), [self.LISTED])
            self.assertEqual(TR.main(["empty-board", "--out", out, "--host-probe", probe]), 0)
            self.assertIn(self.LISTED, json.loads(Path(out, "board.json").read_text())["races"])   # a real board stays

    def test_prep_groups_take_full_ids_only(self):
        with tempfile.TemporaryDirectory() as td:
            a = os.path.join(td, "a.txt")
            Path(a).write_text("algos/cnn-clf/synthetic/rows=2000\nneural/lm-forward/bytes/shape=small\n"
                               "algos/cnn-clf/synthetic/rows=full\nneural/lm-forward/bytes/shape=full\n")
            self.assertEqual(TR.main(["prep-groups", "--ids", a, "--out-dir", os.path.join(td, "g")]), 0)
            got = sorted(x for f in os.listdir(os.path.join(td, "g"))
                         for x in Path(td, "g", f).read_text().split())
            self.assertEqual(got, ["algos/cnn-clf/synthetic/rows=full", "neural/lm-forward/bytes/shape=full"])

    def test_job_wiring(self):
        job = Path(JOB).read_text()
        prep = job[job.index("    if part tg-prep; then"):job.index("    if part tg-smoke; then")]
        self.assertNotIn("smoke-ids.txt", prep)                 # the smoke ids never reach the full-row prep
        self.assertIn('[ "$prow" = full ] && PIDS+=(--ids "$pids_f")', prep)
        board = job[job.index("    if part tg-board; then"):job.index("    if part tg-compare; then")]
        self.assertIn("targeted_rows.py timed-check", board)
        self.assertLess(board.index("targeted_rows.py timed-check"), board.rindex("phase_end $rc"))
        self.assertNotIn("no race to time (none passed its smoke)", job)
        cmp_ = job[job.index("    if part tg-compare; then"):]
        self.assertLess(cmp_.index("targeted_rows.py empty-board"), cmp_.index("paper_compare.py compare"))
        tgb = job[job.index("    tg_board() {"):job.index("    if part tg-board; then")]
        self.assertLess(tgb.index('"placeholder"'), tgb.index("run_timed"))

    def test_tg_drivers_phase(self):
        # lane fx-verdicts2 added tg-drivers with no test: a B set only a driver row names is installed here, every
        # row runs on the candidate and on each B set present, the sets made here are dropped, the worst rc is kept
        job = Path(JOB).read_text()
        drv = job[job.index("# >>> owed-drivers"):job.index("# <<< owed-drivers")]
        tgd = job[job.index("    # >>> tg-drivers"):job.index("    # <<< tg-drivers")]
        with tempfile.TemporaryDirectory() as td:
            tgo = os.path.join(td, "tgo")
            os.makedirs(tgo)
            Path(tgo, "drivers.json").write_text(json.dumps([
                {"id": "r1", "b_arms": ["b1"]}, {"id": "r2", "b_arms": []}]))
            calls = os.path.join(td, "calls")
            fake = os.path.join(td, "fakepy")
            # the arm interpreter: records (arm, row); row r2 on b1 fails with rc 2
            Path(fake).write_text('#!/bin/bash\nwhile [ $# -gt 0 ]; do case $1 in --row) r=$2;; --arm) a=$2;; esac; '
                                  'shift; done\necho "$a $r" >> "' + calls + '"\necho OWED-DRIVER\n'
                                  '[ "$a $r" = "b1 r2" ] && exit 2; exit 0\n')
            os.chmod(fake, 0o755)
            script = ("set -u\nRES=" + td + "\nTGO=" + tgo + "\nPY=" + sys.executable + "\nOWED=1\nOWED_FILE=x\n"
                      "mkdir -p $RES/logs\nLAST_PRIOR=\nB_NAMES=(b1 b2)\ndeclare -A ARM_PY ARM_XENV\n"
                      "ARM_PY[candidate]=" + fake + "\nARM_PY[b2]=" + fake + "\n"
                      "venv_arm() { echo \"$1\"; }\npart() { [ \"$1\" = tg-drivers ]; }\nphase() { echo \"PHASE $1\"; }\n"
                      "phase_end() { echo \"PHASE_END $1\"; }\n"
                      "arm_ready() { [ -n \"${ARM_PY[$1]:-}\" ]; }\n"
                      "install_arms() { echo \"INSTALL $*\"; for a in \"$@\"; do ARM_PY[$a]=" + fake + "; done; }\n"
                      "drop_arm_venv() { echo \"DROP $1\"; unset 'ARM_PY[$1]'; }\n" + drv + "\n" + tgd + "\n")
            p = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)
            got = sorted(Path(calls).read_text().split("\n")) if os.path.exists(calls) else []
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("INSTALL b1", p.stdout)                   # the driver-only B set, installed here
        self.assertNotIn("INSTALL b2", p.stdout)                # already present (a board unit's venv)
        self.assertIn("DROP b1", p.stdout)
        self.assertNotIn("DROP b2", p.stdout)
        self.assertEqual([x for x in got if x], ["b1 r1", "b1 r2", "b2 r1", "b2 r2", "candidate r1", "candidate r2"])
        self.assertIn("PHASE_END 2", p.stdout)                  # a failed driver case fails the phase
        self.assertIn("driver cases: 6", p.stdout)

class AbReading(unittest.TestCase):
    """lane fx-tools-round3 (2026-11-20, A/B abf2): tf32-all had 12 of 13 cells faster and none slower and read
    'median 0.93 SAME'; gemm-leaf512's arm lost neural/gemm to the profile admission and its line read 'arm - ms
    ratio -'. The direction reading and a named reason for every untimed cell and lost row."""

    def test_direction_counts_and_one_way(self):
        tf32_all = [0.96, 0.85, 0.97, 0.95, 0.96, 0.93, 0.97, 0.90, 0.93, 0.90, 0.82, 1.00, 0.91]
        text = TR.direction(tf32_all)
        self.assertIn("faster 12 flat 1 slower 0 min 0.82 max 1.00", text)
        self.assertIn("ONE-WAY faster", text)
        # one cell beyond 1.05 the other way: no ONE-WAY
        self.assertNotIn("ONE-WAY", TR.direction(tf32_all[:-1] + [1.06]))
        # under 80%: no ONE-WAY; under five cells: no ONE-WAY
        self.assertNotIn("ONE-WAY", TR.direction([0.9] * 7 + [1.0] * 3))
        self.assertNotIn("ONE-WAY", TR.direction([0.9] * 4))
        self.assertIn("ONE-WAY slower", TR.direction([1.02] * 5 + [0.97]))
        self.assertEqual(TR.direction([]), "")

    def boards(self, res, arm, base_races, arm_races, arm_extra=None):
        cell = lambda ms: {"arm": "ours", "status": "ok", "median_ms": ms}  # noqa: E731
        for sub, races, extra in (("board-tg-ab-%s-base-full" % arm, base_races, None),
                                  ("board-tg-ab-%s-full" % arm, arm_races, arm_extra)):
            doc = {"races": {rid: {"status": "done", "cells": [cell(ms)]} for rid, ms in races.items()}}
            doc.update(extra or {})
            os.makedirs(os.path.join(res, sub))
            Path(res, sub, "board.json").write_text(json.dumps(doc))

    def test_summary_names_a_lost_row_and_reads_direction(self):
        why = {"failed_preflight_recipes": {D: {"status": "FAILED", "problems": [
            "probe refused mojolearn.linalg: binary numerical profile v2-leaf512 (version 4) does not match"]}}}
        with tempfile.TemporaryDirectory() as res:
            self.boards(res, "x", {A: 10.0, B: 10.0, C: 10.0, D: 10.0}, {A: 9.0, B: 9.0, C: 9.5}, why)
            text = TR.summary(compare_doc(), [], res, [], [("x", "full", [A, B, C, D], set())])
        table = [ln for ln in text.splitlines() if ln.startswith("  x ")][0]
        self.assertIn("median arm/base 0.90 SAME; faster 3 flat 0 slower 0 min 0.90 max 0.95", table)
        self.assertIn("untimed cells 1", table)
        self.assertIn("rows differ 1", table)
        self.assertIn("AB-ROWS-DIFFER arm=x rows=full %s missing on the arm: preflight FAILED: probe refused" % D, text)
        line_d = [ln for ln in text.splitlines() if ln.startswith("AB x rows=full %s fit" % D)][0]
        self.assertIn("base 10.00 -> arm - ms ratio - NOT-TIMED (arm: preflight FAILED: probe refused", line_d)
        self.assertIn("version 4", line_d)
        # the AB-ROWS-DIFFER line sits before this arm's own cell lines
        lines = text.splitlines()
        self.assertLess(lines.index([ln for ln in lines if ln.startswith("AB-ROWS-DIFFER")][0]),
                        lines.index([ln for ln in lines if ln.startswith("AB x rows=full %s" % A)][0]))

    def test_untimed_cell_names_its_status(self):
        with tempfile.TemporaryDirectory() as res:
            self.boards(res, "y", {A: 10.0}, {})
            Path(res, "board-tg-ab-y-full", "board.json").write_text(json.dumps({"races": {A: {
                "status": "failed", "failure": "PARAMS REFUSED: n_iter differs",
                "cells": [{"arm": "ours", "status": "crash(rc=1)", "median_ms": None}]}}}))
            text = TR.summary(compare_doc(), [], res, [], [("y", "full", [A], set())])
        line = [ln for ln in text.splitlines() if ln.startswith("AB y rows=full %s fit" % A)][0]
        self.assertIn("NOT-TIMED (arm: race failed: PARAMS REFUSED: n_iter differs; cell crash(rc=1))", line)

    def test_ab_gaps_cli_and_partial_in_timed_check(self):
        import io
        from contextlib import redirect_stdout
        with tempfile.TemporaryDirectory() as td:
            res = os.path.join(td, "res")
            self.boards(res, "z", {A: 10.0, B: 10.0}, {A: 9.0})
            ids = os.path.join(td, "ids-z-full.txt")
            Path(ids).write_text(A + "\n" + B + "\n")
            Path(ids + ".pass").write_text(A + "\n" + B + "\n")
            units = os.path.join(td, "units.tsv")
            Path(units).write_text("z\tfull\t%s\n" % ids)
            Path(td, "listed.txt").write_text("")
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = TR.main(["ab-gaps", "--arm", "z", "--rows", "full", "--ids", ids + ".pass",
                              "--base", os.path.join(res, "board-tg-ab-z-base-full", "board.json"),
                              "--board", os.path.join(res, "board-tg-ab-z-full", "board.json")])
            self.assertEqual(rc, 0)
            self.assertIn("AB-ROWS-DIFFER arm=z rows=full %s missing on the arm: race not in the board" % B,
                          buf.getvalue())
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = TR.main(["timed-check", "--results", res, "--ids", os.path.join(td, "listed.txt"),
                              "--units", units])
            self.assertEqual(rc, 0, buf.getvalue())                    # a reading: the exit is unchanged
            self.assertIn("PARTIAL z: AB-ROWS-DIFFER arm=z rows=full %s missing on the arm" % B, buf.getvalue())
            # matching rows: the SAME line, no PARTIAL
            Path(ids + ".pass").write_text(A + "\n")
            buf = io.StringIO()
            with redirect_stdout(buf):
                TR.main(["ab-gaps", "--arm", "z", "--rows", "full", "--ids", ids + ".pass",
                         "--base", os.path.join(res, "board-tg-ab-z-base-full", "board.json"),
                         "--board", os.path.join(res, "board-tg-ab-z-full", "board.json")])
            self.assertIn("AB-ROWS-SAME arm=z rows=full (1 race(s))", buf.getvalue())

    def test_job_names_lost_rows_per_unit(self):
        text = Path(JOB).read_text()
        self.assertIn("tools/ops/targeted_rows.py ab-gaps --arm \"$arm\"", text)
        self.assertIn("grep -m 20 '^PAPER-PASS-IDS-FAILED' \"$RES/logs/pass-ids-tg-ab-$arm.log\"", text)


class PhaseSplit(unittest.TestCase):
    """bench_board_probe.phase_split and the A/B summary's per-iteration column (lane ab-harness-speed, 2026-11-24)."""

    def test_iterations_from_n_iter_and_trees_else_named_missing(self):
        import bench_board_probe as PR
        c = {"median_ms": 100.0, "iteration_receipts": [{"status": "recorded", "values": [10, 20]}]}
        p = PR.phase_split(c)
        self.assertEqual((p["iterations"], p["per_iteration_ms"]), (20, 5.0))
        self.assertIsNone(p["loop_ms"])
        self.assertEqual(PR.phase_split({"median_ms": 50.0, "fit": "trees=500 leaves=9 depth=8"})["iterations"], 500)
        miss = PR.phase_split({"median_ms": 5.0, "iteration_receipts": [{"status": "unavailable", "reason": "r"}]})
        self.assertEqual((miss["iterations"], miss["iterations_source"], miss["per_iteration_ms"]), (None, "r", None))

    def test_per_iteration_verdict_marked_when_it_differs(self):
        b = {"median_ms": 100.0, "iteration_receipts": [{"status": "recorded", "values": [10]}]}
        a = {"median_ms": 80.0, "iteration_receipts": [{"status": "recorded", "values": [5]}]}
        t = TR.per_iteration_text(b, a, 0.8)
        self.assertIn("ratio 1.60 SLOWER PHASE-VERDICTS-DIFFER", t)
        same = TR.per_iteration_text(b, dict(a, iteration_receipts=b["iteration_receipts"]), 0.8)
        self.assertNotIn("DIFFER", same)
        self.assertEqual(TR.per_iteration_text(b, {"median_ms": 1.0}, 1.0), "")


if __name__ == "__main__":
    unittest.main()
