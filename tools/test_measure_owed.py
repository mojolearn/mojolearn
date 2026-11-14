#!/usr/bin/env python3
"""tools/test_measure_owed.py -- tools/measure_owed.py plan and report on in-memory fixtures (no box, no board run),
the committed owed file's shape, and tools/paper_compare.py's prior column."""
import json
import os
import sys
import tempfile
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import measure_owed as MO  # noqa: E402
import paper_compare as PC  # noqa: E402


def fake_planner():
    lanes = {"classical": ["kmeans"], "algos": ["gmm", "rfe"]}
    m = types.SimpleNamespace(FAMILIES=("classical", "algos"), DATASETS=("taxi", "istella"), SMOKE_ROWS=2000)
    m.family_lanes = lambda f: lanes[f]
    m.modes_for = lambda v, modes: ["identical"]
    m.gpu_opponents_only = lambda races, v: races

    def plan_races(vendor, modes, fams, lns, ds, rows, shape):
        tail = "rows=%s" % ("full" if rows is None else rows)
        return [{"id": "%s/%s/%s/%s" % (f, l, d, tail), "family": f, "lane": l, "dataset": d}
                for f in fams for l in lanes[f] if l in lns for d in ds]
    m.plan_races = plan_races
    return m


OWED = {"schema": MO.SCHEMA, "b_arms": {"b1": {"define": "X=1"}}, "rows": [
    {"id": "r-board", "kind": "board", "lane": ["kmeans", "rfe"], "dataset": "taxi", "rows": "full"},
    {"id": "r-ab", "kind": "ab", "lane": "gmm", "dataset": "*", "rows": "full", "b_arms": ["b1", "b2"]},
    {"id": "r-nb", "kind": "board", "lane": "kmeans", "dataset": "taxi", "rows": 300000},
    {"id": "r-gone", "kind": "board", "lane": "no-such-lane", "dataset": "*", "rows": "full"},
    {"id": "r-drv", "kind": "driver", "driver": "mlp_fit", "arms": ["candidate", "prior"]}]}
PAPER = {"race_ids": ["classical/kmeans/taxi/rows=full"],
         "bench_board_args": {"families": ["classical"], "lanes": ["kmeans"], "datasets": ["taxi"]}}


class Plan(unittest.TestCase):
    def test_units_skip_paper_races_and_record_missing_b(self):
        units, smoke, drivers, missing, rowmap, bargs = MO.plan(OWED, PAPER, fake_planner(), "nvidia", ["b1"])
        self.assertEqual(units[("candidate", "full")], ["algos/gmm/istella/rows=full", "algos/gmm/taxi/rows=full",
                                                        "algos/rfe/taxi/rows=full"])
        self.assertEqual(units[("prior", "full")], ["algos/rfe/taxi/rows=full"])
        self.assertEqual(units[("b1", "full")], ["algos/gmm/istella/rows=full", "algos/gmm/taxi/rows=full"])
        self.assertEqual(units[("candidate", "300000")], ["classical/kmeans/taxi/rows=300000"])
        self.assertNotIn("classical/kmeans/taxi/rows=2000", smoke)       # the paper smoke covers it
        self.assertIn("algos/rfe/taxi/rows=2000", smoke)
        self.assertIn("r-ab b2", missing)
        self.assertIn("r-gone no-such-lane", missing)
        self.assertEqual([d["id"] for d in drivers], ["r-drv"])
        self.assertIn("algos", bargs[bargs.index("--families") + 1])


class OwedBArms(unittest.TestCase):
    """2026-11-13, job 0003 of ba08dac006 (L40S): PC_B_ARMS named five installed B sets, but the job ran without the
    plan part and reused a plan made without them (OWED-PLAN b_arms=-), so PC_OWED_ARMS=candidate,<b names> ran no B
    arm. The job now reads the B file the same way, and re-makes a plan whose plan.json names other B sets."""
    JOB = os.path.join(HERE, "ops", "paper_compare_job.sh")
    NAMES = ["arima-async-all", "autograd-optim", "byte-lm-generic", "byte-lm-program-step", "chol-ops-fit"]

    def b_block(self):
        with open(self.JOB) as fh:
            text = fh.read()
        return text[text.index("# >>> b-arms"):text.index("# <<< b-arms")]

    def test_job_reads_a_fake_b_arms_file(self):
        import subprocess
        with tempfile.TemporaryDirectory() as td:
            lines = ["# staged by measure_candidate"]
            for i, name in enumerate(self.NAMES):
                d = os.path.join(td, "b-" + name, "nvidia")
                os.makedirs(d)
                for dist in ("core", "nvidia"):
                    open(os.path.join(d, "mojolearn_%s-0.8.67.dev0+b%d-py3-none-any.whl" % (dist, i)), "w").close()
                lines.append("%s %s %s" % (name, d, ("%x" % (i + 10)) * 40))
            lines.append("BadName /x " + "a" * 40)
            bfile = os.path.join(td, "b-arms-nvidia.txt")
            with open(bfile, "w") as fh:
                fh.write("\n".join(lines) + "\n")
            script = ("say() { echo \"SAY $*\"; }\nadd_arm() { echo \"ARM $1 $2 $3\"; }\n" + self.b_block()
                      + "\necho \"B_CSV=$B_CSV\"\n")
            env = dict(os.environ, OWED="1", V="nvidia", PC_B_ARMS=bfile)
            p = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn("B_CSV=" + ",".join(self.NAMES), p.stdout)
            self.assertIn("ARM arima-async-all --mojolearn-version 0.8.67.dev0+b0", p.stdout)
            self.assertIn("SAY bad PC_B_ARMS line: BadName", p.stdout)
            env["PC_B_ARMS"] = os.path.join(td, "missing.txt")
            p = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=60)
            self.assertIn("no such file on the box", p.stdout)
            self.assertIn("B_CSV=\n", p.stdout)

    def test_plan_records_its_b_arms_and_a_stale_plan_is_named(self):
        with tempfile.TemporaryDirectory() as td:
            owed, paper = os.path.join(td, "owed.json"), os.path.join(td, "paper.json")
            with open(owed, "w") as fh:
                json.dump(OWED, fh)
            with open(paper, "w") as fh:
                json.dump(PAPER, fh)
            out = os.path.join(td, "plan-owed")
            real = MO.load_planner
            MO.load_planner = lambda path: fake_planner()
            try:
                self.assertEqual(MO.main(["plan", "--owed", owed, "--paper-recipes", paper, "--planner", "x",
                                          "--vendor", "nvidia", "--out-dir", out]), 0)
                # the plan job 0001 made: no B sets
                why = MO.plan_stale(out, "b1", "nvidia")
                self.assertIn("b_arms=-", why)
                self.assertEqual(MO.main(["plan-check", "--plan-dir", out, "--b-arms", "b1", "--vendor", "nvidia"]), 1)
                self.assertIsNone(MO.plan_stale(out, "", "nvidia"))
                # the re-made plan has a B unit, which PC_OWED_ARMS=candidate,b1 then runs
                MO.main(["plan", "--owed", owed, "--paper-recipes", paper, "--planner", "x", "--vendor", "nvidia",
                         "--b-arms", "b1", "--out-dir", out])
            finally:
                MO.load_planner = real
            self.assertIsNone(MO.plan_stale(out, "b1", "nvidia"))
            self.assertEqual(MO.main(["plan-check", "--plan-dir", out, "--b-arms", "b1", "--vendor", "nvidia"]), 0)
            self.assertIn("vendor nvidia", MO.plan_stale(out, "b1", "amd"))
            with open(os.path.join(out, "units.tsv")) as fh:
                self.assertIn("b1", [ln.split("\t")[0] for ln in fh])
            # a plan from before plan.json: current for a job without B sets, stale for one with them
            os.remove(os.path.join(out, MO.PLAN_META))
            self.assertIsNone(MO.plan_stale(out, "", "nvidia"))
            self.assertIn("no plan.json", MO.plan_stale(out, "b1", "nvidia"))
            self.assertIn("units.tsv missing", MO.plan_stale(os.path.join(td, "none"), "", "nvidia"))

    def test_job_rechecks_the_plan_without_the_plan_part(self):
        with open(self.JOB) as fh:
            text = fh.read()
        self.assertIn('measure_owed.py plan-check --plan-dir "$RES/plan-owed" --b-arms "$B_CSV"', text)
        self.assertLess(text.index("plan-check"), text.index('FULL_ALL="$RES/plan/race-ids-full-all.txt"'))
        self.assertIn("PC_OWED_ARMS names $want: no owed unit", text)


class DataKeys(unittest.TestCase):
    def test_keys_come_from_the_boards_data_needs(self):
        # the stage step and the job read the dataset list from the board's data_needs over the paper and owed races
        # (2026-11-12: a hand-written list left out higgs), never from a literal
        m = fake_planner()
        seen = []

        def data_needs(races):
            seen.extend(races)
            return sorted({r["dataset"] for r in races} | {"higgs"}), []
        m.data_needs = data_needs
        m.R2_KEYS = {"taxi": "gbm-bench/taxi/taxi_speed.npz", "istella": "gbm-bench/istella/istella_speed.npz",
                     "higgs": "gbm-bench/higgs/higgs_speed.npz"}
        m.DATA_FILES = {"taxi": "taxi/taxi_speed.npz", "istella": "istella/istella_speed.npz",
                        "higgs": "higgs/higgs_speed.npz"}
        m.ALGOS = types.SimpleNamespace(r2_keys=lambda lane, ds: ["corpus/enwik8/input.txt"] if lane == "gmm" else [],
                                        corpus_path=lambda k: "/c/" + k)
        paper = dict(PAPER, recipes=[{"family": "classical", "lane": "kmeans", "dataset": "taxi"}])
        keys = MO.data_keys(OWED, paper, m, "nvidia", "/d")
        self.assertIn(("gbm-bench/higgs/higgs_speed.npz", "/d/higgs/higgs_speed.npz"), keys)
        self.assertIn(("gbm-bench/istella/istella_speed.npz", "/d/istella/istella_speed.npz"), keys)
        self.assertIn(("corpus/enwik8/input.txt", "/c/corpus/enwik8/input.txt"), keys)
        self.assertIn({"family": "classical", "lane": "kmeans", "dataset": "taxi"}, seen)
        self.assertTrue(any(r["lane"] == "gmm" for r in seen), "the owed rows' races count too")


class Report(unittest.TestCase):
    def test_prior_ratio_and_b_flags(self):
        cell = lambda ms: {"arm": "ours", "status": "ok", "median_ms": ms}  # noqa: E731
        with tempfile.TemporaryDirectory() as d:
            plan_dir, res = os.path.join(d, "plan"), os.path.join(d, "res")
            os.makedirs(plan_dir)
            json.dump({"r-board": ["algos/rfe/taxi/rows=full"], "r-ab": ["algos/gmm/taxi/rows=full"]},
                      open(os.path.join(plan_dir, "rowmap.json"), "w"))
            json.dump({}, open(os.path.join(plan_dir, "not-planned.json"), "w"))
            for sub, rid, ms in (("board", "algos/rfe/taxi/rows=full", 10.0), ("board", "algos/gmm/taxi/rows=full", 10.0),
                                 ("board-prior-full", "algos/rfe/taxi/rows=full", 5.0),
                                 ("board-b1-full", "algos/gmm/taxi/rows=full", 5.0)):
                os.makedirs(os.path.join(res, sub), exist_ok=True)
                p = os.path.join(res, sub, "board.json")
                doc = json.load(open(p)) if os.path.exists(p) else {"races": {}}
                doc["races"][rid] = {"cells": [cell(ms)]}
                json.dump(doc, open(p, "w"))
            doc = dict(OWED, rows=[r for r in OWED["rows"] if r["id"] in ("r-board", "r-ab")])
            rep = MO.report(doc, plan_dir, res, None, "nvidia")
            by = {r["row"]: r for r in rep["rows"]}
            self.assertIn("REGRESSION", by["r-board"]["flags"])
            self.assertAlmostEqual(by["r-board"]["ratio_vs_prior"], 2.0)
            self.assertIn("b1:B-FASTER", by["r-ab"]["flags"])
            self.assertIn("b2:B-NOT-RUN", by["r-ab"]["flags"])


class PriorColumn(unittest.TestCase):
    def test_recovered_and_regression_vs_prior(self):
        row = {"key": "k", "new_ms": 10.0, "flags": ["SAME"]}
        PC.attach_prior(row, {"k": {"new_ms": 50.0, "flags": ["REGRESSION"]}})
        self.assertIn("RECOVERED", row["flags"])
        self.assertIn("FASTER-VS-PRIOR", row["flags"])
        row = {"key": "k", "new_ms": 60.0, "flags": ["REGRESSION"]}
        PC.attach_prior(row, {"k": {"new_ms": 50.0, "flags": ["REGRESSION"]}})
        self.assertIn("STILL-REGRESSED", row["flags"])
        self.assertIn("REGRESSION-VS-PRIOR", row["flags"])

    def test_candidate_label(self):
        board = {"box": {"mojolearn": {"wheel": {"source": "candidate", "candidate": {
            "source_commit": "a" * 40, "version": "0.8.67", "vendor": "nvidia",
            "files": [{"distribution": "mojolearn-core", "file": "c.whl", "sha256": "0" * 64}]}}}}}
        cand = PC.board_candidate([board])
        self.assertEqual(PC.new_label(cand, ["0.8.67"]), "candidate aaaaaaaaaaaa (0.8.67)")


class CommittedOwed(unittest.TestCase):
    def test_owed_file_shape(self):
        doc = json.load(open(os.path.join(PC.OUT_DIR, "owed-pre0867.json")))
        self.assertEqual(doc["schema"], MO.SCHEMA)
        ids = [r["id"] for r in doc["rows"]]
        self.assertEqual(len(ids), len(set(ids)))
        for r in doc["rows"]:
            self.assertIn(r["kind"], ("board", "ab", "driver"))
            self.assertTrue(r.get("decides"))
            for b in r.get("b_arms") or []:
                self.assertIn(b, doc["b_arms"])


if __name__ == "__main__":
    unittest.main()
