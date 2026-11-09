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
        res = PC.compare(self.doc, [board], why or {})
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
        self.assertIn("NOW-LOSES", ols["flags"])          # kernel clock: 40/50 -> 60/50
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
        self.assertIn("NOW-WINS", gm["flags"])
        self.assertIn("HW-DIFF", gm["flags"])
        pb = rows["trees/gbdt-depthwise/taxi/rows=full fit ours-prebinned"]
        self.assertIn("BROKEN", pb["flags"])

    def test_render_and_cli(self):
        with tempfile.TemporaryDirectory() as td:
            rec = os.path.join(td, "recipes.json")
            json.dump(self.doc, open(rec, "w"))
            nb = os.path.join(td, "board.json")
            json.dump(new_board(), open(nb, "w"))
            md = os.path.join(td, "COMPARE-nvidia.md")
            self.assertEqual(PC.main(["compare", "--recipes", rec, "--board", nb, "--out-md", md]), 0)
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
        p = self.run_job("nvidia", "0.8.66")
        self.assertEqual(p.returncode, 2)
        self.assertIn("box_persist_run.sh", p.stderr)


if __name__ == "__main__":
    unittest.main()
