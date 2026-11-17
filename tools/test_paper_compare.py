#!/usr/bin/env python3
"""tools/test_paper_compare.py -- tests tools/paper_compare.py (recipe extraction, planned ids, smoke pass ids,
the compare and its flags) on small in-memory fixtures, and the argument checks of tools/ops/paper_compare_job.sh.
No board driver, no box, no mojolearn import.

  python3 tools/test_paper_compare.py
"""
import copy
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import paper_compare as PC  # noqa: E402

GPU = "NVIDIA L40S"


def cell(arm, ms, quality=None, status="ok", batch=None, params=None, kernel=None, lane="kmeans"):
    c = {"arm": arm, "median_ms": ms, "times_ms": [ms] if ms is not None else None, "status": status,
         "quality": quality or {}, "batch": batch, "device_name": GPU, "lane": lane, "library_version": "0.8.55",
         "params": {"implementation": {"module": "anonlib.cluster", "name": "KMeans"}, "library": "anonlib",
                    "params": params if params is not None else {"n_clusters": 8}},
         "call": "anonlib.KMeans(n_clusters=8)", "hash": "h-" + arm}
    if kernel is not None:
        c["comparability"] = {"span": {"fit_minus_upload_ms": kernel, "upload_ms_separate": ms - kernel}}
    return c


def opp(arm, opp_ms, ours_ms, clock="whole", batch=None, ours_arm=None, verdict="MATCHED"):
    label = {"whole": "whole/whole", "kernel": "kernel/kernel", "mixed": "MIXED ours whole / arm kernel"}[clock]
    tc = {"clock": clock, "label": label, "opponent_ms": opp_ms, "batch": batch}
    if clock != "mixed":
        tc["ours_ms"] = ours_ms
        tc["value"] = ours_ms / opp_ms
    d = {"verdict": verdict, "timing_comparison": tc, "timing_comparisons": [tc]}
    if ours_arm:
        d["ours_arm"] = ours_arm
    return arm, d


def race(rid, family, lane, dataset, our_arms, cells, infer=None, fit_opps=(), infer_opps=()):
    audit = {"identical/fit": {"opponents": dict(fit_opps)}}
    if infer_opps:
        audit["identical/infer"] = {"opponents": dict(infer_opps)}
    return {"id": rid, "family": family, "lane": lane, "dataset": dataset, "rows": None, "our_arms": our_arms,
            "cells": cells, "infer_cells": infer or [], "audit": audit, "params_check": "MATCHED"}


def paper_board():
    km = "classical/kmeans/taxi/rows=full"
    ols = "classical/ols/taxi/rows=full"
    gb = "trees/gbdt-depthwise/taxi/rows=full"
    rf = "trees/rf/istella/rows=full"
    gm = "neural/gemm/gaussian/shape=full"
    races = {
        km: race(km, "classical", "kmeans", "taxi", {"ours": "identical"},
                 [cell("ours", 100.0, {"inertia": 10.0, "n_iter": 9}, kernel=90.0), cell("cuml-gpu", 200.0)],
                 infer=[cell("ours", 5.0, batch="Xq"), cell("cuml-gpu", 10.0, batch="Xq")],
                 fit_opps=[opp("cuml-gpu", 200.0, 100.0)],
                 infer_opps=[opp("cuml-gpu", 10.0, 5.0, batch="Xq")]),
        ols: race(ols, "classical", "ols", "taxi", {"ours": "identical"},
                  [cell("ours", 50.0, {"r2": 0.5}, kernel=40.0, lane="ols"), cell("torch-gpu", 60.0, lane="ols")],
                  fit_opps=[opp("torch-gpu", 50.0, 40.0, clock="kernel")]),
        gb: race(gb, "trees", "gbdt-depthwise", "taxi", {"ours": "identical", "ours-prebinned": "identical"},
                 [cell("ours", 3714.0, {"auc": 0.632, "logloss": 0.528}, lane="gbdt-depthwise"),
                  cell("ours-prebinned", 3807.0, {"auc": 0.631}, lane="gbdt-depthwise"),
                  cell("catboost-gpu", 6204.0), cell("xgboost-gpu", 3038.0)],
                 fit_opps=[opp("catboost-gpu", 6204.0, 3714.0), opp("xgboost-gpu", 3038.0, 3807.0,
                                                                     ours_arm="ours-prebinned")]),
        rf: race(rf, "trees", "rf", "istella", {"ours": "identical"},
                 [cell("ours", None, status="REFUSED(timeout: null)", lane="rf")]),
        gm: race(gm, "neural", "gemm", "gaussian", {"ours": "identical"},
                 [cell("ours", 46.0, {"max_rel_err_vs_fp64": 2e-7}, lane="gemm"), cell("torch-eager-fp32", 30.0)],
                 fit_opps=[opp("torch-eager-fp32", 30.0, 46.0)]),
    }
    return {"created": "2026-10-30T00:00:00Z", "box": {"gpu": {"name": GPU}}, "races": races}


def new_cell(arm, ms, quality=None, status="ok", batch=None, params=None, kernel=None, lane="kmeans"):
    c = cell(arm, ms, quality, status, batch, params, kernel, lane)
    c["params"]["library"] = "mojolearn"
    c["library_version"] = "0.8.66"
    return c


def new_board(gpu=GPU):
    km = "classical/kmeans/taxi/rows=full"
    races = {
        km: {"cells": [new_cell("ours", 130.0, {"inertia": 10.0}, params={"n_clusters": 9})],
             "infer_cells": [new_cell("ours", 4.0, batch="Xq")]},
        "classical/ols/taxi/rows=full": {"cells": [new_cell("ours", 52.0, {"r2": 0.5}, kernel=60.0, lane="ols")]},
        "trees/gbdt-depthwise/taxi/rows=full": {
            "cells": [new_cell("ours", 3700.0, {"auc": 0.60, "logloss": 0.528}, lane="gbdt-depthwise")]},
        "trees/rf/istella/rows=full": {"cells": [new_cell("ours", 500.0, lane="rf")]},
    }
    return {"box": {"gpu": {"name": gpu}}, "races": races}


HOST = "pod-a/GPU-0000"


def with_host(board, host=HOST):
    """BOARD as the job writes it on a probed box (bench_board --host-probe: box.host_probe)."""
    b = copy.deepcopy(board)
    # the system state (2026-11-17): a verdict needs one GPU driver, kernel and boot id on both arms
    b.setdefault("box", {})["host_probe"] = {"schema": "mojolearn-host-probe/1", "host_id": host,
                                             "gpu": {"driver": "570.1"}, "kernel": "6.8.0-147-generic",
                                             "boot_id": "boot-1"}
    return b


def baseline_board(host=HOST):
    """0.8.55 re-timed on the same box in the same job (board-prior-0.8.55): the paper's our cells, same times."""
    races = {}
    for rid, r in paper_board()["races"].items():
        races[rid] = {"cells": [copy.deepcopy(c) for c in r["cells"] if c["arm"].startswith("ours")],
                      "infer_cells": [copy.deepcopy(c) for c in r["infer_cells"] if c["arm"].startswith("ours")]}
    return with_host({"box": {"gpu": {"name": GPU}}, "races": races}, host)


class Recipes(unittest.TestCase):
    def setUp(self):
        self.doc = PC.extract_recipes(paper_board(), "nvidia")

    def by_key(self):
        return {r["key"]: r for r in self.doc["recipes"]}

    def test_counts_and_args(self):
        c = self.doc["counts"]
        self.assertEqual(c["races"], 5)
        self.assertEqual(c["recipes"], 7)        # kmeans fit+infer, ols, gbdt x2 arms, rf, gemm
        self.assertEqual((c["fit"], c["infer"]), (6, 1))
        b = self.doc["bench_board_args"]
        self.assertEqual(b["families"], ["trees", "classical", "neural"])
        self.assertEqual(sorted(b["datasets"]), ["istella", "taxi"])   # the neural lane's own data is not a dataset
        self.assertTrue(b["gpu_opponents_only"])
        self.assertEqual(self.doc["source"]["gpu"], GPU)

    def test_only_our_arms_and_deanonymized(self):
        keys = self.by_key()
        self.assertNotIn("classical/kmeans/taxi/rows=full fit cuml-gpu", keys)
        r = keys["classical/kmeans/taxi/rows=full fit ours"]
        self.assertEqual(r["implementation"]["module"], "mojolearn.cluster")
        self.assertTrue(r["call"].startswith("mojolearn.KMeans"))
        self.assertNotIn("anonlib", json.dumps(self.doc))
        self.assertEqual(r["v0855"]["kernel_ms"], 90.0)
        self.assertEqual(r["v0855"]["median_ms"], 100.0)

    def test_opponents_paired_by_arm_and_batch(self):
        keys = self.by_key()
        gb = keys["trees/gbdt-depthwise/taxi/rows=full fit ours"]
        self.assertEqual([o["arm"] for o in gb["opponents"]], ["catboost-gpu"])
        pb = keys["trees/gbdt-depthwise/taxi/rows=full fit ours-prebinned"]
        self.assertEqual([o["arm"] for o in pb["opponents"]], ["xgboost-gpu"])
        inf = keys["classical/kmeans/taxi/rows=full infer ours Xq"]
        self.assertEqual(inf["opponents"][0]["opponent_ms"], 10.0)
        self.assertEqual(inf["opponents"][0]["opponent_status"], "ok")
        self.assertEqual(keys["classical/ols/taxi/rows=full fit ours"]["opponents"][0]["clock"], "kernel")

    def test_refused_kept_without_time(self):
        rf = self.by_key()["trees/rf/istella/rows=full fit ours"]
        self.assertIsNone(rf["v0855"]["median_ms"])
        self.assertTrue(rf["v0855"]["status"].startswith("REFUSED"))

    def test_cli_writes_file(self):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "board.json.gz")
            with gzip.open(src, "wt") as fh:
                json.dump(paper_board(), fh)
            out = os.path.join(td, "recipes.json")
            self.assertEqual(PC.main(["recipes", "--vendor", "nvidia", "--board", src, "--out", out]), 0)
            doc = json.load(open(out))
            self.assertEqual(doc["schema"], PC.SCHEMA)
            self.assertEqual(len(doc["source"]["sha256"]), 64)


class Compare(unittest.TestCase):
    def setUp(self):
        self.doc = PC.extract_recipes(paper_board(), "nvidia")

    def rows(self, board, why=None):
        res = PC.compare(self.doc, [with_host(board)], why or {}, prior_boards=[("0.8.55", baseline_board())])
        return res, {r["key"]: r for r in res["rows"]}

    def test_flags(self):
        res, rows = self.rows(new_board(), {"neural/gemm/gaussian/shape=full": "smoke failed: ours: no time"})
        km = rows["classical/kmeans/taxi/rows=full fit ours"]
        self.assertIn("REGRESSION", km["flags"])
        self.assertAlmostEqual(km["ratio"], 1.3)
        self.assertIn("PARAMS-DIFF", km["flags"])
        self.assertEqual(km["params_diff"], ["n_clusters"])
        self.assertEqual(km["quality_verdict"], "PASS")
        self.assertAlmostEqual(km["best_opponent"]["ratio_new"], 0.65)     # 130 / the stored 200 ms
        self.assertFalse({"NOW-WINS", "NOW-LOSES"} & set(km["flags"]))
        inf = rows["classical/kmeans/taxi/rows=full infer ours Xq"]
        self.assertIn("FASTER", inf["flags"])
        ols = rows["classical/ols/taxi/rows=full fit ours"]
        self.assertIn("SAME", ols["flags"])
        # kernel clock: 40/50 -> 60/50 is shown; the stored opponent ran on the paper's host, so no NOW-LOSES flag
        self.assertNotIn("NOW-LOSES", ols["flags"])
        self.assertAlmostEqual(ols["best_opponent"]["ratio_new"], 1.2)
        gb = rows["trees/gbdt-depthwise/taxi/rows=full fit ours"]
        self.assertIn("QUALITY-DROP", gb["flags"])
        self.assertTrue(any(f.startswith("auc") for f in gb["quality_failures"]))
        pb = rows["trees/gbdt-depthwise/taxi/rows=full fit ours-prebinned"]
        self.assertEqual(pb["flags"], ["NOT-RUN"])
        rf = rows["trees/rf/istella/rows=full fit ours"]
        self.assertIn("FIXED", rf["flags"])
        gm = rows["neural/gemm/gaussian/shape=full fit ours"]
        self.assertEqual(gm["flags"], ["NOT-RUN"])
        self.assertIn("smoke failed", gm["why"])
        self.assertFalse(res["hw_diff"])
        self.assertEqual(res["new_versions"], ["0.8.66"])

    def test_quality_missing_is_flagged_with_the_cells_reason_and_shown(self):
        # QUALITY-MISSING (2026-11-13, candidate ba08dac006 L40S): mamba*-forward lost rel_fro_vs_torch_eager_fp32
        # (the torch reference did not run in the ours-only venv) and the compare showed new_quality={}
        board = new_board()
        board["races"]["neural/gemm/gaussian/shape=full"] = {
            "cells": [new_cell("ours", 45.0, {"max_abs_diff_vs_host": 0.0, "host_reference_passed": True,
                                              "quality_unavailable": "torch eager fp32 reference did not run: "
                                                                     "No module named 'torch'"}, lane="gemm")]}
        res, rows = self.rows(board)
        gm = rows["neural/gemm/gaussian/shape=full fit ours"]
        self.assertIn("QUALITY-MISSING", gm["flags"])
        self.assertIn("SAME", gm["flags"])                     # shown beside the time, never a gate
        self.assertEqual(gm["quality_missing"], ["max_rel_err_vs_fp64"])
        self.assertIn("No module named 'torch'", gm["quality_unavailable"])
        # every recorded number is shown, not only the judged ones (was {} for the host-reference lanes)
        self.assertEqual(gm["new_quality"]["max_abs_diff_vs_host"], 0.0)
        self.assertNotIn("host_reference_passed", gm["new_quality"])
        self.assertIn("quality_unavailable", gm["new_quality"])
        self.assertIn("QUALITY-MISSING (1)", PC.render_summary(res))
        md = PC.render_md(res)
        self.assertIn("quality missing: max_rel_err_vs_fp64", md)
        # a row with the same judged keys is not flagged; an untimed row is BROKEN, not QUALITY-MISSING
        self.assertNotIn("QUALITY-MISSING", rows["trees/gbdt-depthwise/taxi/rows=full fit ours"]["flags"])
        board["races"]["neural/gemm/gaussian/shape=full"]["cells"][0].update(median_ms=None, status="FAILED(rc=1)")
        res, rows = self.rows(board)
        self.assertNotIn("QUALITY-MISSING", rows["neural/gemm/gaussian/shape=full fit ours"]["flags"])

    def test_quality_missing_without_a_reason_says_so(self):
        board = new_board()
        board["races"]["neural/gemm/gaussian/shape=full"] = {"cells": [new_cell("ours", 45.0, {}, lane="gemm")]}
        _, rows = self.rows(board)
        gm = rows["neural/gemm/gaussian/shape=full fit ours"]
        self.assertEqual(gm["quality_missing"], ["max_rel_err_vs_fp64"])
        self.assertIn("recorded no reason", gm["quality_unavailable"])

    def test_broken_now_wins_and_hw_diff(self):
        board = new_board(gpu="NVIDIA GeForce RTX 4090")
        board["races"]["neural/gemm/gaussian/shape=full"] = {
            "cells": [new_cell("ours", 25.0, {"max_rel_err_vs_fp64": 2e-7}, lane="gemm")]}
        board["races"]["trees/gbdt-depthwise/taxi/rows=full"]["cells"].append(
            new_cell("ours-prebinned", None, status="FAILED(rc=1)", lane="gbdt-depthwise"))
        res, rows = self.rows(board)
        self.assertTrue(res["hw_diff"])
        gm = rows["neural/gemm/gaussian/shape=full fit ours"]
        self.assertIn("FASTER", gm["flags"])
        self.assertNotIn("NOW-WINS", gm["flags"])           # the stored opponent is another host's: shown only
        self.assertLess(gm["best_opponent"]["ratio_new"], 1.0)
        self.assertIn("HW-DIFF", gm["flags"])
        pb = rows["trees/gbdt-depthwise/taxi/rows=full fit ours-prebinned"]
        self.assertIn("BROKEN", pb["flags"])

    def test_render_and_cli(self):
        with tempfile.TemporaryDirectory() as td:
            rec = os.path.join(td, "recipes.json")
            json.dump(self.doc, open(rec, "w"))
            nb = os.path.join(td, "board.json")
            json.dump(with_host(new_board()), open(nb, "w"))
            bb = os.path.join(td, "board-prior-0.8.55.json")
            json.dump(baseline_board(), open(bb, "w"))
            md = os.path.join(td, "COMPARE-nvidia.md")
            self.assertEqual(PC.main(["compare", "--recipes", rec, "--board", nb, "--prior-board", "0.8.55=" + bb,
                                      "--out-md", md]), 0)
            text = open(md).read()
            self.assertIn("| classical/kmeans/taxi/rows=full fit ours | 100 | 130 | 1.30 |", text)
            summary = open(os.path.join(td, "SUMMARY-nvidia.txt")).read()
            self.assertIn("REGRESSION=1", summary)
            self.assertIn("FIXED (1)", summary)
            self.assertLessEqual(len(summary.splitlines()), 40)
            self.assertEqual(json.load(open(os.path.join(td, "compare-nvidia.json")))["schema"], PC.COMPARE_SCHEMA)


FAKE_PLANNER = '''
DATASETS = ("taxi", "istella")
SMOKE_ROWS = 2000
def modes_for(vendor, requested=None):
    return ["identical"]
def family_lanes(f):
    return {"classical": ("kmeans", "ols"), "trees": ("gbdt-depthwise",), "neural": ("gemm",)}.get(f, ())
def plan_races(vendor, modes, families, lanes, datasets, rows, shape):
    out = []
    for f in families:
        for l in family_lanes(f):
            if lanes and l not in lanes:
                continue
            if f == "neural":
                out.append({"id": "neural/%s/gaussian/shape=%s" % (l, shape), "our_arms": {"ours": "identical"}})
                continue
            for d in datasets:
                out.append({"id": "%s/%s/%s/rows=%s" % (f, l, d, "full" if rows is None else rows),
                            "our_arms": {"ours": "identical"}})
    return out
def gpu_opponents_only(races, vendor):
    return races
'''


class ParamsDiff(unittest.TestCase):
    def test_prebinned_rows_clause_is_spelling_only(self):
        old = {"input_binning": "pre-binned input on both arms (GreedyLogSum, 254 borders); bins sha256 ab",
               "max_depth": 8}
        new = {"input_binning": "pre-binned input on both arms (GreedyLogSum, 254 borders); 2043304 training rows; "
                                "bins sha256 ab", "max_depth": 8}
        self.assertEqual(PC.params_diff(old, new), [])
        moved = dict(new, input_binning=new["input_binning"].replace("sha256 ab", "sha256 cd"))
        self.assertEqual(PC.params_diff(old, moved), ["input_binning"])
        self.assertEqual(PC.params_diff(old, dict(new, max_depth=6)), ["max_depth"])

    def test_float32_read_back_is_the_same_setting(self):
        # FAIRNESS FIX (2026-11-16): sequential-fit / sequential-predict, paper record 0.001, every arm now passes and
        # reads back float32(1e-3) (tools/bench_board_neural.py FIT_ADAM, 2026-10-27)
        old = {"learning_rate": 0.001, "betas": [0.9, 0.999], "epochs": 1, "optimizer": "adam"}
        new = dict(old, learning_rate=0.0010000000474974513)
        self.assertEqual(PC.params_diff(old, new), [])
        self.assertEqual(PC.params_diff(old, dict(new, betas=[0.8999999761581421, 0.999])), [])
        self.assertEqual(PC.params_diff(old, dict(new, learning_rate=0.002)), ["learning_rate"])
        self.assertEqual(PC.params_diff(old, dict(new, learning_rate=0.0010001)), ["learning_rate"])
        self.assertEqual(PC.params_diff(old, dict(new, epochs=2)), ["epochs"])
        self.assertEqual(PC.params_diff(old, dict(new, optimizer="sgd")), ["optimizer"])


class Ids(unittest.TestCase):
    def test_plan_ids_and_pass_ids(self):
        doc = PC.extract_recipes(paper_board(), "nvidia")
        with tempfile.TemporaryDirectory() as td:
            planner = os.path.join(td, "fake_board.py")
            open(planner, "w").write(FAKE_PLANNER)
            rec = os.path.join(td, "recipes.json")
            json.dump(doc, open(rec, "w"))
            out = os.path.join(td, "plan")
            self.assertEqual(PC.main(["ids", "--recipes", rec, "--planner", planner, "--out-dir", out]), 0)
            full = open(os.path.join(out, "race-ids-full.txt")).read().split()
            smoke = open(os.path.join(out, "race-ids-smoke.txt")).read().split()
            self.assertEqual(full, ["classical/kmeans/taxi/rows=full", "classical/ols/taxi/rows=full",
                                    "neural/gemm/gaussian/shape=full", "trees/gbdt-depthwise/taxi/rows=full"])
            self.assertIn("classical/kmeans/taxi/rows=2000", smoke)
            self.assertIn("neural/gemm/gaussian/shape=small", smoke)
            missing = json.load(open(os.path.join(out, "not-planned.json")))
            self.assertIn("no longer a board lane", missing["trees/rf/istella/rows=full"])
            self.assertIn("ours-prebinned", missing["trees/gbdt-depthwise/taxi/rows=full arms"])
            args = open(os.path.join(out, "board-args.txt")).read().split("\n")
            self.assertIn("--gpu-opponents-only", args)
            self.assertEqual(args[args.index("--modes") + 1], "identical")
            smoke_doc = {"races": {
                "classical/kmeans/taxi": {"id": "classical/kmeans/taxi/rows=2000", "pass": True},
                "neural/gemm/gaussian": {"id": "neural/gemm/gaussian/shape=small", "pass": True},
                "classical/ols/taxi": {"id": "classical/ols/taxi/rows=2000", "pass": False,
                                       "failures": [{"arm": "ours", "reason": "no time"}]}}}
            sj = os.path.join(td, "smoke.json")
            json.dump(smoke_doc, open(sj, "w"))
            ids = os.path.join(out, "race-ids-full.txt")
            dest = os.path.join(out, "race-ids-pass.txt")
            self.assertEqual(PC.main(["pass-ids", "--full-ids", ids, "--smoke-json", sj, "--out", dest]), 0)
            self.assertEqual(open(dest).read().split(), ["classical/kmeans/taxi/rows=full",
                                                         "neural/gemm/gaussian/shape=full"])
            failed = json.load(open(os.path.join(out, "smoke-failed.json")))
            self.assertIn("ours: no time", failed["classical/ols/taxi/rows=full"])
            self.assertEqual(failed["trees/gbdt-depthwise/taxi/rows=full"], "no smoke result")

    def test_smoke_key_matches_the_board(self):
        # the pass-ids match on bench_board.smoke_key (family/lane/dataset), whatever the smoke id's tail
        for rid in ("classical/kmeans/taxi/rows=full", "neural/gemm/gaussian/shape=full"):
            self.assertEqual(PC.race_smoke_key(PC.smoke_id(rid, 2000)), PC.race_smoke_key(rid))


class CommittedRecipes(unittest.TestCase):
    def test_committed_files_load(self):
        for vendor in PC.VENDORS:
            path = os.path.join(PC.OUT_DIR, "recipes-%s.json" % vendor)
            if not os.path.isfile(path):
                self.skipTest("%s not in this checkout" % path)
            doc = json.load(open(path))
            self.assertEqual((doc["schema"], doc["vendor"], doc["paper_version"]), (PC.SCHEMA, vendor, "0.8.55"))
            self.assertEqual(doc["counts"]["recipes"], len(doc["recipes"]))
            self.assertEqual(len({r["key"] for r in doc["recipes"]}), len(doc["recipes"]))
            for r in doc["recipes"]:
                self.assertIn(r["race_id"], doc["race_ids"])
                self.assertTrue(r["arm"].startswith("ours"))


class JobScript(unittest.TestCase):
    JOB = os.path.join(HERE, "ops", "paper_compare_job.sh")
    LIB = os.path.join(HERE, "ops", "box_job_lib.sh")

    def run_job(self, *args, env=None):
        e = {k: v for k, v in os.environ.items() if k not in ("MOJOLEARN_BOX_PERSIST", "MOJOLEARN_ALLOW_UNPERSISTED",
                                                              "MOJOLEARN_RESULTS_DIR")}
        e.update(env or {})
        return subprocess.run(["bash", self.JOB] + list(args), env=e, capture_output=True, text=True, timeout=60)

    def test_syntax(self):
        self.assertEqual(subprocess.run(["bash", "-n", self.JOB]).returncode, 0)

    def test_refusals_before_any_work(self):
        self.assertEqual(self.run_job().returncode, 2)
        self.assertEqual(self.run_job("apple").returncode, 2)
        self.assertEqual(self.run_job("nvidia", "not-a-version").returncode, 2)
        p = self.run_job("nvidia", "0.8.67")
        self.assertEqual(p.returncode, 2)
        self.assertIn("box_persist_run.sh", p.stderr)
        # published mode has no default version (Andrew, 2026-11-16: the only prior is 0.8.55; 0.8.66 is never timed)
        p = self.run_job("nvidia", env={"PC_WHEEL_DIR": ""})
        self.assertEqual(p.returncode, 2)
        self.assertIn("published mode needs the version under test", p.stderr)
        with open(self.JOB) as fh:
            self.assertNotIn("VER=0.8.66", fh.read())

    def test_datasets_from_data_keys_and_smoke_scratch_pruned(self):
        # 2026-11-12/13: a hand-written dataset list missed higgs; smoke scratch and staging tarballs filled the pod
        with open(self.JOB) as fh:
            text = fh.read()
        self.assertIn("measure_owed.py data-keys", text)
        self.assertNotIn("for f in taxi/taxi_speed.npz", text)
        self.assertIn('rm -rf "${d}work" "${d}evidence"', text)
        self.assertIn("prune_persist_staging", text)

    def functions(self, *names):
        """The named shell functions of the job script and its shared library (tools/ops/box_job_lib.sh), as text to
        source (each runs on its own, no job)."""
        text = ""
        for path in (self.JOB, self.LIB):
            with open(path) as fh:
                text += "\n" + fh.read()
        out = []
        for name in names:
            start = text.index("\n%s() {" % name) + 1
            first = text[start:text.index("\n", start)]
            if first.rstrip().endswith("}"):          # a one-line function
                out.append(first + "\n")
                continue
            end = text.index("\n}\n", start) + 3
            out.append(text[start:end])
        return "".join(out)

    def sh(self, body, env=None):
        e = dict(os.environ, **(env or {}))
        return subprocess.run(["bash", "-c", body], env=e, capture_output=True, text=True, timeout=60)

    def test_candidate_and_prior_smokes_overlap(self):
        # PARALLEL PLAN (2026-11-16): the untimed smokes of the candidate and of every prior run at once in one shard
        # set (the 0.8.55 smoke used to wait for the candidate's 3.6 h timed board); each arm keeps its own out dirs,
        # the prior gets a done file, and the shard caches other than c1 are pruned
        f = self.functions("nlines", "smoke_shards", "smoke_launch", "smoke_together")
        with tempfile.TemporaryDirectory() as td:
            fake = os.path.join(td, "fake.sh")
            with open(fake, "w") as fh:
                fh.write('out=""; while [ $# -gt 0 ]; do [ "$1" = --out ] && out=$2; shift; done\n'
                         'echo "start $(basename "$out") $(date +%s%N)" >> "' + td + '/events"; sleep 1\n'
                         'echo "end $(basename "$out") $(date +%s%N)" >> "' + td + '/events"\n')
            for name, n in (("cand.txt", 6), ("prior.txt", 3)):
                with open(os.path.join(td, name), "w") as fh:
                    fh.write("".join("r%d\n" % i for i in range(n)))
            for d in ("res/plan", "res/logs", "cache/c1", "cache/c2", "cache/cp1-1"):
                os.makedirs(os.path.join(td, d))
            open(os.path.join(td, "man"), "w").close()
            body = ("say() { echo \"SAY $*\"; }\ndisk_gate() { :; }\nrun_as() { echo --arm \"$1\"; }\n" + f +
                    "declare -A PRIOR_SMOKE_IDS ARM_MAN\nSMOKE_PIDS=(); SMOKE_PID_ARM=(); CARGS=(--x)\n"
                    "RES=%s/res; CACHE_ROOT=%s/cache; PY='bash %s'; V=amd; PC_SMOKE_SHARDS=3\n"
                    "PRIOR_ARMS=(prior-0.8.55); ARM_MAN[prior-0.8.55]=%s/man; PRIOR_SMOKE_IDS[prior-0.8.55]=%s/prior.txt\n"
                    "smoke_together '' %s/cand.txt $RES/smoke; echo rc=$?\n" % (td, td, fake, td, td, td))
            p = self.sh(body)
            self.assertIn("rc=0", p.stdout, p.stderr)
            ev = [ln.split() for ln in open(os.path.join(td, "events"))]
            starts = {o: int(t) for k, o, t in ev if k == "start"}
            ends = {o: int(t) for k, o, t in ev if k == "end"}
            self.assertEqual(sorted(starts), ["smoke-prior-0.8.55-s1", "smoke-s1", "smoke-s2"])
            # every shard starts before any shard ends: the candidate and the prior run at once
            self.assertLess(max(starts.values()), min(ends.values()))
            self.assertTrue(os.path.isfile(os.path.join(td, "res/plan/prior-0.8.55-smoke.done")))
            self.assertEqual(sorted(os.listdir(os.path.join(td, "cache"))), ["c1"])
            # a second call (the prior part of a split submit) does not smoke the prior again
            p = self.sh(body.replace("smoke_together '' %s/cand.txt" % td, "smoke_together '' ''"))
            self.assertIn("done in an earlier submit", p.stdout)

    def test_busy_cores_never_negative(self):
        # 2026-11-14, L40S pod nvc1: `busy=-18.3 total=128` from /proc/stat deltas (iowait goes backwards; the
        # container's /proc/stat is the host's). The pod's cgroup CPU time over wall time, clamped to the allocation.
        f = self.functions("busy_line")
        line = lambda prev, cur, n=32: self.sh(f + 'busy_line %d board "%s" "%s"' % (n, prev, cur)).stdout
        # cgroup: 60 s of wall, 1200 s of CPU = 20 busy cores
        self.assertRegex(line("1000000 5000000 cgroup", "61000000 1205000000 cgroup"),
                         r"busy=20\.0 total=32 phase=board source=cgroup$")
        # more CPU than the allocation (a burst over the quota period): clamped, and marked so
        self.assertIn("busy=32.0 total=32", line("0 0 cgroup", "60000000 3000000000 cgroup"))
        self.assertIn("clamped", line("0 0 cgroup", "60000000 3000000000 cgroup"))
        # a counter that went backwards is no sample: NA, never a negative number
        out = line("0 9000000 cgroup", "60000000 1000 cgroup")
        self.assertIn("busy=NA", out)
        self.assertNotIn("busy=-", out)
        # /proc/stat fallback: busy ticks over all ticks, scaled by the cpus listed, clamped to the allocation
        self.assertIn("busy=16.0 total=32", line("1000 200 64 procstat", "2000 450 64 procstat"))
        self.assertIn("busy=32.0 total=32", line("1000 200 128 procstat", "2000 700 128 procstat"))
        self.assertIn("busy=NA", line("1000 900 128 procstat", "2000 100 128 procstat"))
        with open(self.LIB) as fh:
            text = fh.read()
        self.assertIn("cpu.stat", text)
        self.assertIn("busy_line", text[text.index("cores_sampler() {"):])

    def test_disk_gate_refuses_and_prints_the_numbers(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        os.makedirs(os.path.join(d, "job"))
        f = self.functions("say", "nlines", "disk_gate")
        base = 'RES=%s; J=$RES/job; EXIT_DISK=28; WORST=0; ' % d
        ok = self.sh(base + f + "disk_gate board 10 1 1; echo PASSED", env={"PC_DISK_RESERVE_GB": "0", "PC_LOADED_MB": "1"})
        self.assertEqual(ok.returncode, 0, ok.stdout + ok.stderr)
        self.assertIn("DISK board: 10 races x 1 MB + 1 boards x 1 MB + reserve 0 GB = 11 MB needed", ok.stdout)
        self.assertIn("PASSED", ok.stdout)
        no = self.sh(base + f + "disk_gate board 1000000000 1 40; echo NOT-REACHED")
        self.assertEqual(no.returncode, 28)
        self.assertIn("DISK board refused", no.stdout)
        self.assertNotIn("NOT-REACHED", no.stdout)
        with open(os.path.join(d, "job", "phases.tsv")) as fh:
            self.assertTrue(fh.read().startswith("board-disk-refused\t28\t"))

    def test_enospc_stops_the_job_with_exit_28(self):
        # a fail-soft phase that hit ENOSPC must not run the next phase into the same wall (owed prior 0009 and
        # all 653 smoke races of screen 0010 failed one after the other, 2026-11-14)
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        os.makedirs(os.path.join(d, "job"))
        os.makedirs(os.path.join(d, "logs"))
        f = self.functions("say", "phase", "phase_end", "nlines", "free_gb", "disk_full_in_phase")
        base = 'RES=%s; J=$RES/job; EXIT_DISK=28; WORST=0; ' % d
        # an old ENOSPC line in the job's own log and a log untouched since before the phase are not this phase's
        with open(os.path.join(d, "job", "job.log"), "w") as fh:
            fh.write("DISK FULL in phase board: ENOSPC\n")
        old = os.path.join(d, "logs", "old.log")
        with open(old, "w") as fh:
            fh.write("OSError: [Errno 28] No space left on device\n")
        os.utime(old, (1, 1))
        clean = self.sh(base + f + "phase owed; phase_end 1; echo NEXT-PHASE")
        self.assertEqual(clean.returncode, 0, clean.stdout + clean.stderr)
        self.assertIn("NEXT-PHASE", clean.stdout)
        full = self.sh(base + f + "phase smoke; echo 'OSError: [Errno 28] No space left on device' > $RES/logs/smoke-s1.log; "
                       "phase_end 1; echo NEXT-PHASE")
        self.assertEqual(full.returncode, 28, full.stdout + full.stderr)
        self.assertIn("DISK FULL in phase smoke: ENOSPC in", full.stdout)
        self.assertIn("smoke-s1.log", full.stdout)
        self.assertNotIn("NEXT-PHASE", full.stdout)
        with open(os.path.join(d, "job", "phases.tsv")) as fh:
            self.assertIn("smoke\t28\t", fh.read())

    def job_dir(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        for sub in ("job", "logs", "board-all"):
            os.makedirs(os.path.join(d, sub))
        return d

    def test_a_cut_phase_is_partial_with_done_of_total_and_the_next_submit_resumes(self):
        """fb-harness (2026-11-17): the 240-minute cap cut every full-board arm (650 races take 3.5 to 4.5 h) and the
        next submit resumed it silently. The cut phase gets a `partial <done>/<total> <board>` line; the disk gate of
        the resumed board counts only the races left."""
        d = self.job_dir()
        ids = os.path.join(d, "ids.txt")
        with open(ids, "w") as fh:
            fh.write("r/a\nr/b\nr/c\n")
        with open(os.path.join(d, "board-all", "board.json"), "w") as fh:
            json.dump({"races": {"r/a": {"status": "done"}, "r/b": {"status": "done"}, "r/c": {"status": "failed"}}},
                      fh)
        f = self.functions("say", "nlines", "free_gb", "disk_full_in_phase", "races_left", "board_progress", "phase",
                           "phase_end", "phase_partial")
        base = 'cd %s; PY=%s; RES=%s; J=$RES/job; EXIT_DISK=28; WORST=0; PHASE_OPEN=0; ' % (
            os.path.dirname(HERE), sys.executable, d)
        p = self.sh(base + f + 'phase all-board; CUR_IDS=%s; CUR_BOARD=$RES/board-all; phase_partial 124; '
                    'echo LEFT=$(races_left %s $RES/board-all/board.json) NONE=$(races_left %s $RES/nowhere.json); '
                    'phase all-board; phase_end 0; phase_partial 0' % (ids, ids, ids))
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("phase all-board ended partial (rc=124): 2/3 board-all", p.stdout)
        self.assertIn("LEFT=1 NONE=3", p.stdout)
        self.assertIn("resuming (an earlier submit ended inside it: partial 2/3 board-all)", p.stdout)
        with open(os.path.join(d, "job", "phases.tsv")) as fh:
            lines = [x.split("\t") for x in fh.read().splitlines()]
        self.assertEqual([(x[0], x[1], x[4]) for x in lines],
                         [("all-board", "124", "partial 2/3 board-all"), ("all-board", "0", "done")])

    def test_a_reboot_between_submits_is_a_restart_line(self):
        """fb-harness (2026-11-17): the d95f805cce pod restarted between its arms (driver 595.91.07 -> 595.112.07,
        kernel 6.8.0-147 -> 149). The probe phase compares the boot id with the last submit's."""
        d = self.job_dir()
        fake = os.path.join(d, "fakepy")
        with open(fake, "w") as fh:
            fh.write('#!/bin/bash\nwhile [ $# -gt 0 ]; do [ "$1" = --out ] && out=$2; shift; done\n'
                     'printf \'{\\n "boot_id": "%s",\\n "gpu": {\\n  "driver": "%s"\\n },\\n "host_id": "h/GPU-1",\\n'
                     ' "kernel": "%s"\\n}\\n\' "$FAKE_BOOT" "$FAKE_DRIVER" "$FAKE_KERNEL" > "$out"\n')
        os.chmod(fake, 0o755)
        f = self.functions("say", "free_gb", "disk_full_in_phase", "phase", "phase_end", "probe_field",
                           "host_probe_phase")
        base = 'PY=%s; RES=%s; J=$RES/job; V=nvidia; EXIT_DISK=28; WORST=0; PHASE_OPEN=0; ' % (fake, d)
        one = self.sh(base + f + "host_probe_phase",
                      env={"FAKE_BOOT": "aaaa1111", "FAKE_DRIVER": "595.91.07", "FAKE_KERNEL": "6.8.0-147-generic"})
        self.assertEqual(one.returncode, 0, one.stdout + one.stderr)
        self.assertNotIn("RESTART", one.stdout)
        two = self.sh(base + f + "host_probe_phase",
                      env={"FAKE_BOOT": "bbbb2222", "FAKE_DRIVER": "595.112.07", "FAKE_KERNEL": "6.8.0-149-generic"})
        self.assertEqual(two.returncode, 0, two.stdout + two.stderr)
        self.assertIn("RESTART: the box rebooted since the last submit (boot id aaaa1111 -> bbbb2222", two.stdout)
        with open(os.path.join(d, "job", "phases.tsv")) as fh:
            text = fh.read()
        self.assertIn("restart boot aaaa1111 -> bbbb2222; driver 595.91.07 -> 595.112.07; "
                      "kernel 6.8.0-147-generic -> 6.8.0-149-generic", text)
        self.assertTrue(os.path.exists(os.path.join(d, "job", "host-probe-boot-aaaa1111.json")))

    def test_timed_boards_drop_their_scratch_and_gate_on_races_left(self):
        with open(self.JOB) as fh:
            text = fh.read()
        self.assertEqual(text.count("--drop-race-scratch --out"), 4)    # all-board, prior_board_run, board_run, tg_board
        self.assertNotIn('disk_gate all-board "$(nlines', text)
        self.assertNotIn('disk_gate board "$(nlines', text)
        self.assertIn('phase_partial "$code"', text)


# ---------------------------------------------------------------------------------------------------------------
# --best-ever (tools/board_best_ever.py files) and the full-board screen (ids-all, board-all)
# ---------------------------------------------------------------------------------------------------------------

PCA = "classical/pca/taxi/rows=full"
SVC = "classical/svc/taxi/rows=full"


def be_new_board(gpu=GPU):
    """new_board() as a real board writes it (our cells carry mode identical, the box its host probe), plus two
    races outside the recipes."""
    b = with_host(new_board(gpu))
    for race in b["races"].values():
        for c in race.get("cells", []) + race.get("infer_cells", []):
            c["mode"] = "identical"
    for rid, ms in ((PCA, 20.0), (SVC, 7.0)):
        c = new_cell("ours", ms, lane=rid.split("/")[1])
        c["mode"] = "identical"
        b["races"][rid] = {"family": "classical", "lane": rid.split("/")[1], "dataset": "taxi", "cells": [c]}
    return b


SOFTWARE = ["GPU-0000", "570.1", "6.8.0-147-generic", "boot-1"]     # with_host's system state


def be_entry(ms, params=None, hash_="h-ours", device=GPU, release="0.8.25", date="2026-09-30", host=HOST,
             software=SOFTWARE):
    """A best-ever entry; HOST, SOFTWARE: the box and the system state it was timed on (the new board's by default)."""
    return {"ms": ms, "params": params, "hash": hash_, "device": device, "release": release, "date": date,
            "commit": "c0ffee", "params_source": "cell" if params is not None else None, "host_id": host,
            "software": software}


def be_doc(vendor="nvidia", **rows):
    out = {}
    for key, best in rows.items():
        k = key.replace("__", " ")
        out[k] = {"key": k, "best": best, "best_by_device": {best["device"]: best}, "latest": best,
                  "flags": [], "timeline": [best]}
    return {"schema": "mojolearn-board-best-ever/1", "vendor": vendor, "mined_at_head": "abc", "rows": out}


INTERNAL = "algos/internal-call-overhead-lr-at/synthetic/rows=full"
OLD_LR = "algos/lr-step/synthetic/rows=full"


def internal_race(rid, ms):
    c = new_cell("ours", ms, lane=rid.split("/")[1])
    c["mode"] = "identical"
    return {"family": "algos", "lane": rid.split("/")[1], "dataset": "synthetic", "cells": [c]}


class InternalCallOverhead(unittest.TestCase):
    """Andrew 2026-11-17: the lr_at row is call overhead, an internal check, not a race: its own section, new against
    our own prior on this box only; never a recipe, an extra (best-ever) row; an old lr-* schedule row never shows."""

    def run_compare(self, best=True):
        doc = PC.extract_recipes(paper_board(), "nvidia")
        new = be_new_board()
        new["races"][INTERNAL] = internal_race(INTERNAL, 150.0)
        new["races"][OLD_LR] = internal_race(OLD_LR, 90.0)
        prior = baseline_board()
        prior["races"][INTERNAL] = internal_race(INTERNAL, 100.0)
        key = INTERNAL + " fit ours"
        best_doc = be_doc(**{key.replace(" ", "__"): be_entry(1.0)}) if best else None
        return PC.compare(doc, [new], {}, prior_boards=[("0.8.55", prior)], best_ever=best_doc)

    def test_internal_row_has_its_own_section_against_our_prior_only(self):
        for best in (True, False):
            res = self.run_compare(best)
            internal = {r["key"]: r for r in res["internal_rows"]}
            self.assertEqual(sorted(internal), [INTERNAL + " fit ours", OLD_LR + " fit ours"])
            r = internal[INTERNAL + " fit ours"]
            self.assertAlmostEqual(r["priors"]["0.8.55"]["ratio"], 1.5)
            self.assertIn("REGRESSION", r["flags"])
            self.assertNotIn("best_ever_ms", r)                        # never a best-ever column
            self.assertFalse(any(x["key"].startswith(("algos/internal-", "algos/lr-")) for x in res["extra_rows"]))
            self.assertFalse(any(x["key"].startswith("algos/") for x in res["rows"]))
            self.assertEqual(res["internal_flag_counts"].get("REGRESSION"), 1)
            self.assertIn("Internal checks, call overhead, internal check: 2", PC.render_md(res))
            summary = PC.render_summary(res)
            self.assertIn("internal checks (call overhead, internal check; not a race", summary)
            self.assertIn(INTERNAL + " fit ours new 150", summary)


class OpticsHasNoInferCell(unittest.TestCase):
    def test_a_cluster_lane_without_predict_plans_no_infer_cell(self):
        """fb-harness (2026-11-17): OPTICS has no predict; its infer cell timed an empty call (0.012 ms)."""
        import bench_board_algos as A
        self.assertFalse(A.has_infer("optics"))
        self.assertEqual([l for l, s in A.LANES.items() if s.get("task") == "cluster" and not A.has_infer(l)],
                         ["optics"])


class BestEver(unittest.TestCase):
    def setUp(self):
        self.doc = PC.extract_recipes(paper_board(), "nvidia")
        k = {"km": "classical/kmeans/taxi/rows=full fit ours", "ols": "classical/ols/taxi/rows=full fit ours",
             "gb": "trees/gbdt-depthwise/taxi/rows=full fit ours", "rf": "trees/rf/istella/rows=full fit ours",
             "pca": PCA + " fit ours"}
        self.keys = k
        rows = {
            k["km"]: be_entry(110.0, {"n_clusters": 9}),                  # 130 / 110 = 1.18, same params and hash
            k["ols"]: be_entry(40.0, {"n_clusters": 8}, hash_="h-old"),  # another hash only: bits change, no flag
            k["gb"]: be_entry(3000.0, None),                              # no params recorded: UNCHECKED
            k["rf"]: be_entry(600.0, {"n_clusters": 8}),                  # faster than ever: no flag
            k["pca"]: be_entry(10.0, {"n_clusters": 8}, device="NVIDIA GeForce RTX 4090"),
        }
        self.best = be_doc(**{key.replace(" ", "__"): v for key, v in rows.items()})

    def run_compare(self, best=True):
        res = PC.compare(self.doc, [be_new_board()], {}, prior_boards=[("0.8.55", baseline_board())],
                         best_ever=self.best if best else None)
        return res, {r["key"]: r for r in res["rows"]}, {r["key"]: r for r in res["extra_rows"]}

    def test_flags_on_recipe_rows(self):
        res, rows, _ = self.run_compare()
        km = rows[self.keys["km"]]
        self.assertIn("BEST-EVER-REGRESSION", km["flags"])
        self.assertAlmostEqual(km["ratio_vs_best_ever"], 130.0 / 110.0)
        self.assertEqual((km["best_ever_ms"], km["best_ever_release"], km["best_ever_date"]),
                         (110.0, "0.8.25", "2026-09-30"))
        self.assertIn("REGRESSION", km["flags"])            # the 0.8.55 flags are unchanged
        self.assertIn("PARAMS-DIFF", km["flags"])
        ols = rows[self.keys["ols"]]
        # PARAMS-DIFFER FIX (2026-11-17): an output hash of an older version alone is not a parameter difference
        # (bits may change between versions); it is shown and the time flag stands (52 / 40)
        self.assertNotIn("PARAMS-DIFFER", ols["flags"])
        self.assertTrue(ols["best_ever_hash_differs"])
        self.assertIn("BEST-EVER-REGRESSION", ols["flags"])
        self.assertIn("vs best ever: hash", PC.render_md(res))
        gb = rows[self.keys["gb"]]
        self.assertIn("BEST-EVER-REGRESSION-UNCHECKED", gb["flags"])
        rf = rows[self.keys["rf"]]
        self.assertFalse({"BEST-EVER-REGRESSION", "PARAMS-DIFFER", "BEST-EVER-REGRESSION-UNCHECKED"} & set(rf["flags"]))
        self.assertAlmostEqual(rf["ratio_vs_best_ever"], 500.0 / 600.0)
        gm = rows["neural/gemm/gaussian/shape=full fit ours"]
        self.assertEqual(gm["flags"], ["NOT-RUN"])
        self.assertIsNone(gm["best_ever_ms"])
        self.assertEqual(res["best_ever_label"], "best ever (NV)")
        self.assertEqual(res["flag_counts"]["BEST-EVER-REGRESSION"], 2)

    def test_params_differ_only_on_a_shared_parameter(self):
        self.best["rows"][self.keys["ols"]]["best"]["params"] = {"n_clusters": 7}
        _, rows, _ = self.run_compare()
        ols = rows[self.keys["ols"]]
        self.assertIn("PARAMS-DIFFER", ols["flags"])
        self.assertEqual(ols["best_ever_params_differ"], ["n_clusters"])
        self.assertNotIn("BEST-EVER-REGRESSION", ols["flags"])

    def test_a_best_ever_of_another_system_state_never_flags(self):
        # the same host id, but another boot (or a record mined before 2026-11-17, which has no system state)
        for sw in (["GPU-0000", "570.1", "6.8.0-147-generic", "boot-0"], None):
            self.best["rows"][self.keys["km"]]["best"]["software"] = sw
            _, rows, _ = self.run_compare()
            km = rows[self.keys["km"]]
            self.assertAlmostEqual(km["ratio_vs_best_ever"], 130.0 / 110.0)       # shown
            self.assertNotIn("BEST-EVER-REGRESSION", km["flags"])

    def test_extra_rows_outside_the_recipes(self):
        res, _, extra = self.run_compare()
        self.assertEqual(sorted(extra), [PCA + " fit ours", SVC + " fit ours"])
        pca = extra[PCA + " fit ours"]
        self.assertIn("BEST-EVER-REGRESSION", pca["flags"])
        self.assertIn("BEST-EVER-HW-DIFF", pca["flags"])     # its best ran on another GPU
        self.assertAlmostEqual(pca["ratio_vs_best_ever"], 2.0)
        svc = extra[SVC + " fit ours"]
        self.assertEqual(svc["flags"], [])
        self.assertIsNone(svc["best_ever_ms"])
        self.assertEqual(res["extra_flag_counts"]["BEST-EVER-REGRESSION"], 1)
        md = PC.render_md(res)
        self.assertIn("best ever (NV) ms", md)
        self.assertIn("Rows outside the paper's recipes: 2", md)
        self.assertIn("no earlier measurement", md)
        self.assertIn("worse than best ever", PC.render_summary(res))

    def test_without_best_ever_nothing_changes(self):
        res, rows, extra = self.run_compare(best=False)
        base = PC.compare(self.doc, [be_new_board()], {}, prior_boards=[("0.8.55", baseline_board())])
        self.assertEqual(extra, {})
        self.assertEqual(res["flag_counts"], base["flag_counts"])
        self.assertNotIn("best_ever_ms", rows[self.keys["km"]])
        self.assertNotIn("best ever", PC.render_md(res))

    def test_vendor_mismatch_refused_and_cli(self):
        with self.assertRaises(SystemExit):
            PC.compare(self.doc, [be_new_board()], {}, best_ever=dict(self.best, vendor="amd"))
        with tempfile.TemporaryDirectory() as td:
            rec, brd, be = (os.path.join(td, n) for n in ("recipes.json", "board.json", "best.json"))
            json.dump(self.doc, open(rec, "w"))
            json.dump(be_new_board(), open(brd, "w"))
            json.dump(self.best, open(be, "w"))
            md = os.path.join(td, "C.md")
            self.assertEqual(PC.main(["compare", "--recipes", rec, "--board", brd, "--best-ever", be,
                                      "--out-md", md]), 0)
            out = json.load(open(os.path.join(td, "compare-nvidia.json")))
            self.assertEqual(len(out["extra_rows"]), 2)


class IdsAll(unittest.TestCase):
    def test_every_planned_race(self):
        with tempfile.TemporaryDirectory() as td:
            planner = os.path.join(td, "fake_board.py")
            open(planner, "w").write(FAKE_PLANNER + 'FAMILIES = ("trees", "classical", "neural")\n')
            out = os.path.join(td, "plan-all")
            self.assertEqual(PC.main(["ids-all", "--vendor", "nvidia", "--planner", planner, "--out-dir", out]), 0)
            full = open(os.path.join(out, "race-ids-full.txt")).read().split()
            self.assertEqual(len(full), 7)          # kmeans, ols and gbdt-depthwise on two datasets, gemm
            self.assertIn("classical/kmeans/istella/rows=full", full)
            self.assertIn("neural/gemm/gaussian/shape=full", full)
            smoke = open(os.path.join(out, "race-ids-smoke.txt")).read().split()
            self.assertEqual(len(smoke), 7)
            args = open(os.path.join(out, "board-args.txt")).read().split("\n")
            self.assertNotIn("--gpu-opponents-only", args)
            self.assertNotIn("--with-opponents", args)       # our arm only; stored opponent cells join
            self.assertEqual(args[args.index("--families") + 1], "trees,classical,neural")
            races = json.load(open(os.path.join(out, "races.json")))["recipes"]
            self.assertEqual(len(races), 7)
            self.assertEqual(json.load(open(os.path.join(out, "not-planned.json"))), {})

    def test_job_script_board_all(self):
        with open(JobScript.JOB) as fh:
            text = fh.read()
        self.assertEqual(subprocess.run(["bash", "-n", JobScript.JOB]).returncode, 0)
        self.assertIn("paper_compare.py ids-all", text)
        self.assertIn("--best-ever", text)
        self.assertIn('smoke_shards "$total"', text)                # one shard set for every arm's smoke
        self.assertIn('smoke_together all- "$PA/race-ids-smoke.txt" "$RES/smoke-all" --drop-race-scratch', text)
        self.assertIn('smoke_together "" "$SMOKE_IDS" "$RES/smoke"', text)
        self.assertNotIn("prior_smoke ", text)
        self.assertEqual(text.count("BYMEM"), 0)                     # one shard formula, in smoke_shards
        self.assertIn('--out "$RES/board-all"', text)
        self.assertNotIn("--with-opponents", text)


if __name__ == "__main__":
    unittest.main()
