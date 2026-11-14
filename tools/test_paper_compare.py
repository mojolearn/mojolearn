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

    def test_datasets_from_data_keys_and_smoke_scratch_pruned(self):
        # 2026-11-12/13: a hand-written dataset list missed higgs; smoke scratch and staging tarballs filled the pod
        with open(self.JOB) as fh:
            text = fh.read()
        self.assertIn("measure_owed.py data-keys", text)
        self.assertNotIn("for f in taxi/taxi_speed.npz", text)
        self.assertIn('rm -rf "${d}work" "${d}evidence"', text)
        self.assertIn("prune_persist_staging", text)

    def functions(self, *names):
        """The named shell functions of the job script, as text to source (each runs on its own, no job)."""
        with open(self.JOB) as fh:
            text = fh.read()
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
        with open(self.JOB) as fh:
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


# ---------------------------------------------------------------------------------------------------------------
# --best-ever (tools/board_best_ever.py files) and the full-board screen (ids-all, board-all)
# ---------------------------------------------------------------------------------------------------------------

PCA = "classical/pca/taxi/rows=full"
SVC = "classical/svc/taxi/rows=full"


def be_new_board(gpu=GPU):
    """new_board() as a real board writes it (our cells carry mode identical), plus two races outside the recipes."""
    b = new_board(gpu)
    for race in b["races"].values():
        for c in race.get("cells", []) + race.get("infer_cells", []):
            c["mode"] = "identical"
    for rid, ms in ((PCA, 20.0), (SVC, 7.0)):
        c = new_cell("ours", ms, lane=rid.split("/")[1])
        c["mode"] = "identical"
        b["races"][rid] = {"family": "classical", "lane": rid.split("/")[1], "dataset": "taxi", "cells": [c]}
    return b


def be_entry(ms, params=None, hash_="h-ours", device=GPU, release="0.8.25", date="2026-09-30"):
    return {"ms": ms, "params": params, "hash": hash_, "device": device, "release": release, "date": date,
            "commit": "c0ffee", "params_source": "cell" if params is not None else None}


def be_doc(vendor="nvidia", **rows):
    out = {}
    for key, best in rows.items():
        k = key.replace("__", " ")
        out[k] = {"key": k, "best": best, "best_by_device": {best["device"]: best}, "latest": best,
                  "flags": [], "timeline": [best]}
    return {"schema": "mojolearn-board-best-ever/1", "vendor": vendor, "mined_at_head": "abc", "rows": out}


class BestEver(unittest.TestCase):
    def setUp(self):
        self.doc = PC.extract_recipes(paper_board(), "nvidia")
        k = {"km": "classical/kmeans/taxi/rows=full fit ours", "ols": "classical/ols/taxi/rows=full fit ours",
             "gb": "trees/gbdt-depthwise/taxi/rows=full fit ours", "rf": "trees/rf/istella/rows=full fit ours",
             "pca": PCA + " fit ours"}
        self.keys = k
        rows = {
            k["km"]: be_entry(110.0, {"n_clusters": 9}),                  # 130 / 110 = 1.18, same params and hash
            k["ols"]: be_entry(40.0, {"n_clusters": 8}, hash_="h-old"),  # another hash: PARAMS-DIFFER
            k["gb"]: be_entry(3000.0, None),                              # no params recorded: UNCHECKED
            k["rf"]: be_entry(600.0, {"n_clusters": 8}),                  # faster than ever: no flag
            k["pca"]: be_entry(10.0, {"n_clusters": 8}, device="NVIDIA GeForce RTX 4090"),
        }
        self.best = be_doc(**{key.replace(" ", "__"): v for key, v in rows.items()})

    def run_compare(self, best=True):
        res = PC.compare(self.doc, [be_new_board()], {}, best_ever=self.best if best else None)
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
        self.assertIn("PARAMS-DIFFER", ols["flags"])
        self.assertTrue(ols["best_ever_hash_differs"])
        self.assertNotIn("BEST-EVER-REGRESSION", ols["flags"])
        gb = rows[self.keys["gb"]]
        self.assertIn("BEST-EVER-REGRESSION-UNCHECKED", gb["flags"])
        rf = rows[self.keys["rf"]]
        self.assertFalse({"BEST-EVER-REGRESSION", "PARAMS-DIFFER", "BEST-EVER-REGRESSION-UNCHECKED"} & set(rf["flags"]))
        self.assertAlmostEqual(rf["ratio_vs_best_ever"], 500.0 / 600.0)
        gm = rows["neural/gemm/gaussian/shape=full fit ours"]
        self.assertEqual(gm["flags"], ["NOT-RUN"])
        self.assertIsNone(gm["best_ever_ms"])
        self.assertEqual(res["best_ever_label"], "best ever (NV)")
        self.assertEqual(res["flag_counts"]["BEST-EVER-REGRESSION"], 1)

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
        base = PC.compare(self.doc, [be_new_board()], {})
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
        self.assertIn('smoke_shards "$NRACE"', text)
        self.assertEqual(text.count("BYMEM"), 0)                     # one shard formula, in smoke_shards
        self.assertIn('--out "$RES/board-all"', text)
        self.assertNotIn("--with-opponents", text)


if __name__ == "__main__":
    unittest.main()
