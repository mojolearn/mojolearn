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
        # with a last-release prior arm (an explicit PC_PRIOR_VERSION other than the paper baseline)
        units, smoke, drivers, missing, rowmap, bargs = MO.plan(OWED, PAPER, fake_planner(), "nvidia", ["b1"], True)
        self.assertEqual(units[("candidate", "full")], ["algos/gmm/istella/rows=full", "algos/gmm/taxi/rows=full",
                                                        "algos/rfe/taxi/rows=full"])
        self.assertEqual(units[("prior", "full")], ["algos/rfe/taxi/rows=full"])
        self.assertEqual(units[("b1", "full")], ["algos/gmm/istella/rows=full", "algos/gmm/taxi/rows=full"])
        self.assertEqual(units[("candidate", "300000")], ["classical/kmeans/taxi/rows=300000"])
        self.assertNotIn("classical/kmeans/taxi/rows=2000", smoke["candidate"])       # the paper smoke covers it
        self.assertIn("algos/rfe/taxi/rows=2000", smoke["candidate"])
        # the prior is smoked only on its own owed rows (the board row's rfe), never on the A/B row's gmm
        self.assertEqual(smoke["prior"], ["algos/rfe/taxi/rows=2000"])
        self.assertIn("r-ab b2", missing)
        self.assertIn("r-gone no-such-lane", missing)
        self.assertEqual([d["id"] for d in drivers], ["r-drv"])
        self.assertIn("algos", bargs[bargs.index("--families") + 1])

    def test_only_rows_a_report_reads(self):
        # lane paper-plan-trim, 2026-11-16 (L40S job of d95f805cce, PC_PRIOR_VERSION=0.8.55, no B set): the candidate
        # timed 197 races for a compare that read 61, and 0.8.55 was planned on the last release's 159. By default the
        # job times one prior, the paper baseline: no board row (it compares with the last release), no A/B row
        # without a staged B set, no driver row with one arm.
        units, smoke, drivers, missing, rowmap, bargs = MO.plan(OWED, PAPER, fake_planner(), "nvidia", [])
        self.assertEqual(units, {})
        self.assertEqual(smoke, {"candidate": [], "prior": []})
        self.assertEqual(drivers, [])
        self.assertEqual(rowmap, {})
        self.assertIn("last release", missing["r-board prior"])
        self.assertIn("not staged", missing["r-ab b1"])
        self.assertIn("this job runs only candidate", missing["r-drv"])
        self.assertEqual(bargs[bargs.index("--families") + 1], "classical")     # the paper's own arguments only
        # one staged B set: only its A/B row, candidate and B, on the same races
        units, smoke, drivers, missing, rowmap, bargs = MO.plan(OWED, PAPER, fake_planner(), "nvidia", ["b1"])
        self.assertEqual(sorted(units), [("b1", "full"), ("candidate", "full")])
        self.assertEqual(units[("candidate", "full")], units[("b1", "full")])
        self.assertEqual(sorted(rowmap), ["r-ab"])
        self.assertEqual(smoke["prior"], [])

    def test_driver_rows_need_two_present_arms(self):
        row = {"id": "d", "kind": "driver", "arms": ["candidate:ndarray", "candidate:dataframe", "prior:ndarray"]}
        self.assertEqual(MO.driver_arms(row, []), ["candidate:ndarray", "candidate:dataframe"])
        self.assertEqual(MO.driver_arms(dict(row, arms=["candidate", "prior", "bx"]), ["bx"]), ["candidate", "bx"])
        self.assertEqual(MO.driver_arms(dict(row, arms=["candidate", "prior"]), [], True), ["candidate", "prior"])


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
                # a plan made without the last-release prior arm is stale for a job that times one
                self.assertIn("prior=no", MO.plan_stale(out, "", "nvidia", True))
                self.assertEqual(MO.main(["plan-check", "--plan-dir", out, "--vendor", "nvidia", "--prior"]), 1)
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
            # a plan from before plan.json (made with the prior arm, without B sets): current only for such a job
            os.remove(os.path.join(out, MO.PLAN_META))
            self.assertIsNone(MO.plan_stale(out, "", "nvidia", True))
            self.assertIn("prior=no", MO.plan_stale(out, "", "nvidia"))
            self.assertIn("no plan.json", MO.plan_stale(out, "b1", "nvidia"))
            self.assertIn("units.tsv missing", MO.plan_stale(os.path.join(td, "none"), "", "nvidia"))

    def test_job_rechecks_the_plan_without_the_plan_part(self):
        with open(self.JOB) as fh:
            text = fh.read()
        self.assertIn('measure_owed.py plan-check --plan-dir "$RES/plan-owed" --b-arms "$B_CSV" --vendor "$V"', text)
        self.assertIn('${OWED_PRIOR[@]+"${OWED_PRIOR[@]}"} 2>&1)', text)
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
        keys = MO.data_keys(OWED, paper, m, "nvidia", "/d", ["b1"])
        self.assertIn(("gbm-bench/higgs/higgs_speed.npz", "/d/higgs/higgs_speed.npz"), keys)
        self.assertIn(("gbm-bench/istella/istella_speed.npz", "/d/istella/istella_speed.npz"), keys)
        self.assertIn(("corpus/enwik8/input.txt", "/c/corpus/enwik8/input.txt"), keys)
        self.assertIn({"family": "classical", "lane": "kmeans", "dataset": "taxi"}, seen)
        self.assertTrue(any(r["lane"] == "gmm" for r in seen), "the planned owed rows' races count too")
        # the rows the plan does not time (no B set staged, no last release) stage no dataset
        del seen[:]
        MO.data_keys(OWED, paper, m, "nvidia", "/d")
        self.assertEqual([r["lane"] for r in seen], ["kmeans"])


def write_boards(res, specs, hosts=None):
    """SPECS: (board dir, race id, ms); HOSTS: {board dir: host id} (default one host for all: the same job)."""
    cell = lambda ms: {"arm": "ours", "status": "ok", "median_ms": ms}  # noqa: E731
    for sub, rid, ms in specs:
        os.makedirs(os.path.join(res, sub), exist_ok=True)
        p = os.path.join(res, sub, "board.json")
        doc = json.load(open(p)) if os.path.exists(p) else {"races": {}}
        host = (hosts or {}).get(sub, "pod-a/GPU-0")
        if host:
            doc["box"] = {"host_probe": {"schema": "mojolearn-host-probe/1", "host_id": host}}
        doc["races"][rid] = {"cells": [cell(ms)], "host": {"hostname": "x", "host_id": host}}
        json.dump(doc, open(p, "w"))


class Report(unittest.TestCase):
    SPECS = (("board", "algos/rfe/taxi/rows=full", 10.0), ("board", "algos/gmm/taxi/rows=full", 10.0),
             ("board-prior-0.8.66", "algos/rfe/taxi/rows=full", 5.0),
             ("board-b1-full", "algos/gmm/taxi/rows=full", 5.0))

    def plan(self, d):
        plan_dir, res = os.path.join(d, "plan"), os.path.join(d, "res")
        os.makedirs(plan_dir)
        json.dump({"r-board": ["algos/rfe/taxi/rows=full"], "r-ab": ["algos/gmm/taxi/rows=full"]},
                  open(os.path.join(plan_dir, "rowmap.json"), "w"))
        json.dump({}, open(os.path.join(plan_dir, "not-planned.json"), "w"))
        return plan_dir, res, dict(OWED, rows=[r for r in OWED["rows"] if r["id"] in ("r-board", "r-ab")])

    def test_prior_ratio_and_b_flags(self):
        with tempfile.TemporaryDirectory() as d:
            plan_dir, res, doc = self.plan(d)
            write_boards(res, self.SPECS)
            # the prior's full rows come from the prior release's own on-box board (board-prior-<version>)
            rep = MO.report(doc, plan_dir, res, "nvidia", "0.8.66")
            by = {r["row"]: r for r in rep["rows"]}
            self.assertIn("REGRESSION", by["r-board"]["flags"])
            self.assertAlmostEqual(by["r-board"]["ratio_vs_prior"], 2.0)
            self.assertIn("b1:B-FASTER", by["r-ab"]["flags"])
            self.assertIn("b2:B-NOT-RUN", by["r-ab"]["flags"])
            self.assertEqual(rep["host_id"], "pod-a/GPU-0")
            self.assertEqual(MO.board_dir(res, "prior", "full", "0.8.66"), os.path.join(res, "board-prior-0.8.66"))
            self.assertEqual(MO.board_dir(res, "prior", "300000", "0.8.66"), os.path.join(res, "board-prior-300000"))

    def test_prior_of_another_host_refuses_and_no_probe_is_not_compared(self):
        with tempfile.TemporaryDirectory() as d:
            plan_dir, res, doc = self.plan(d)
            write_boards(res, self.SPECS, {"board-prior-0.8.66": "pod-b/GPU-9"})
            with self.assertRaises(SystemExit) as cm:
                MO.report(doc, plan_dir, res, "nvidia", "0.8.66")
            self.assertIn("REFUSED", str(cm.exception))
        with tempfile.TemporaryDirectory() as d:
            plan_dir, res, doc = self.plan(d)
            write_boards(res, self.SPECS, {"board-prior-0.8.66": None, "board-b1-full": None})
            rep = MO.report(doc, plan_dir, res, "nvidia", "0.8.66")
            by = {r["row"]: r for r in rep["rows"]}
            self.assertEqual(by["r-board"]["flags"], ["NOT-COMPARED"])     # the ratio is shown, never a verdict
            self.assertAlmostEqual(by["r-board"]["ratio_vs_prior"], 2.0)
            self.assertIn("b1:B-NOT-COMPARED", by["r-ab"]["flags"])


class AtomicResults(unittest.TestCase):
    """2026-11-14, L40S pod nvc1: the owed report died on `JSONDecodeError: line 1 column 87`, a results line cut
    short while the disk was full. Results are written whole or not at all, and a cut driver line is reported."""

    def test_write_atomic_keeps_the_old_file_on_a_failed_write(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "sub", "compare.json")
            PC.write_atomic(p, PC.json_text({"a": 1}))
            self.assertEqual(json.load(open(p)), {"a": 1})

            class Full(Exception):
                pass

            real = os.fsync

            def no_space(fd):
                raise Full("[Errno 28] No space left on device")
            os.fsync = no_space
            try:
                with self.assertRaises(Full):
                    PC.write_atomic(p, PC.json_text({"a": 2, "long": "x" * 10000}))
            finally:
                os.fsync = real
            self.assertEqual(json.load(open(p)), {"a": 1}, "the old file is untouched")
            self.assertEqual(sorted(os.listdir(os.path.dirname(p))), ["compare.json"], "no temp file is left")

    def test_report_reads_a_cut_driver_line_as_unreadable(self):
        with tempfile.TemporaryDirectory() as d:
            plan_dir, res = os.path.join(d, "plan"), os.path.join(d, "res")
            os.makedirs(plan_dir)
            os.makedirs(os.path.join(res, "drivers"))
            PC.write_atomic(os.path.join(plan_dir, "rowmap.json"), PC.json_text({}))
            PC.write_atomic(os.path.join(plan_dir, "not-planned.json"), PC.json_text({}))
            good = json.dumps({"row": "r-drv", "arm": "candidate", "case": "c1", "ms": 1.5, "status": "ok"})
            with open(os.path.join(res, "drivers", "drivers.jsonl"), "w") as fh:
                fh.write(good + "\n" + good[:40] + "\n")
            doc = dict(OWED, rows=[r for r in OWED["rows"] if r["id"] == "r-drv"])
            rep = MO.report(doc, plan_dir, res, "nvidia", "0.8.66")
            self.assertEqual(rep["drivers"][0]["ms"], 1.5)
            self.assertEqual(rep["drivers"][1]["status"], "UNREADABLE")
            self.assertEqual(rep["drivers"][1]["line"], 2)
            self.assertIn("UNREADABLE", MO.render_summary(rep))
            self.assertIn("UNREADABLE", MO.render_md(rep))

    def test_report_files_are_written_atomically(self):
        import inspect
        for fn in (MO.cmd_report, MO.cmd_plan):
            src = inspect.getsource(fn)
            self.assertIn("write_atomic", src)
            self.assertNotIn('open(path, "w")', src)


class PriorColumn(unittest.TestCase):
    """The prior column is the prior release re-timed on this box (board-prior-<version>), judged against the on-box
    baseline (old_ms); a stored compare is a reference column only."""

    def row(self, new_ms, old_ms, flags):
        return {"key": "r fit ours", "race_id": "r", "phase": "fit", "arm": "ours", "batch": None,
                "new_ms": new_ms, "old_ms": old_ms, "flags": list(flags)}

    def races(self, ms):
        return {"r": {"cells": [{"arm": "ours", "status": "ok", "median_ms": ms}]}}

    def test_recovered_and_regression_vs_prior(self):
        row = PC.attach_prior_board(self.row(10.0, 10.0, ["SAME"]), "0.8.66", self.races(50.0), True)
        self.assertIn("RECOVERED", row["flags"])
        self.assertIn("FASTER-VS-PRIOR", row["flags"])
        row = PC.attach_prior_board(self.row(60.0, 10.0, ["REGRESSION"]), "0.8.66", self.races(50.0), True)
        self.assertIn("STILL-REGRESSED", row["flags"])
        self.assertIn("REGRESSION-VS-PRIOR", row["flags"])
        self.assertAlmostEqual(row["priors"]["0.8.66"]["ratio"], 1.2)

    def test_unverified_prior_and_reference_never_flag(self):
        row = PC.attach_prior_board(self.row(60.0, 10.0, ["REGRESSION"]), "0.8.66", self.races(50.0), False)
        self.assertEqual(row["flags"], ["REGRESSION"])
        self.assertAlmostEqual(row["priors"]["0.8.66"]["ratio"], 1.2)
        row = PC.attach_reference(self.row(60.0, 10.0, []), {"r fit ours": {"new_ms": 5.0, "flags": ["REGRESSION"]}})
        self.assertEqual((row["ref_prior_ms"], row["flags"]), (5.0, []))

    def test_prior_without_a_cell_is_not_run(self):
        row = PC.attach_prior_board(self.row(60.0, 10.0, ["REGRESSION"]), "0.8.66", {}, True)
        self.assertEqual(row["prior_status"], "NOT-RUN on this box")
        self.assertIsNone(row["ratio_vs_prior"])
        self.assertEqual(row["flags"], ["REGRESSION"])

    def test_candidate_label(self):
        board = {"box": {"mojolearn": {"wheel": {"source": "candidate", "candidate": {
            "source_commit": "a" * 40, "version": "0.8.67", "vendor": "nvidia",
            "files": [{"distribution": "mojolearn-core", "file": "c.whl", "sha256": "0" * 64}]}}}}}
        cand = PC.board_candidate([board])
        self.assertEqual(PC.new_label(cand, ["0.8.67"]), "candidate aaaaaaaaaaaa (0.8.67)")


class InJobBaseCap(unittest.TestCase):
    """Lane fx-same-pod-prior (2026-11-19; it replaces the 2026-11-17 STORED BASE, which capped a B arm at a base time
    stored from another host): an ab row with "cap_base" times its base (the candidate) and its B arm back to back in
    the same job like every ab row; each B race's rounds are capped at the time this job's base took for it
    (base-caps over the base board); a B run that reached the cap is NOT-FASTER-THAN-BASE."""
    HOST = "pod-a/GPU-0"
    ROW = {"id": "r-cb", "kind": "ab", "lane": "gmm", "dataset": "*", "rows": "full", "b_arms": ["b1"],
           "decides": "x", "cap_base": True}
    TAXI, ISTELLA = "algos/gmm/taxi/rows=full", "algos/gmm/istella/rows=full"

    def test_plan_times_base_and_arm_and_lists_the_capped_races(self):
        doc = dict(OWED, rows=[self.ROW])
        units, smoke, drivers, missing, rowmap, bargs = MO.plan(doc, PAPER, fake_planner(), "nvidia", ["b1"])
        self.assertIn(("candidate", "full"), units)                  # the base is timed in this job
        self.assertEqual(units[("b1", "full")], sorted([self.TAXI, self.ISTELLA]))
        self.assertEqual(MO.unit_caps(doc, rowmap, units), {("b1", "full"): sorted([self.TAXI, self.ISTELLA])})
        plain = dict(OWED, rows=[dict(self.ROW, cap_base=False)])
        u2, *_r, rm2, _b = MO.plan(plain, PAPER, fake_planner(), "nvidia", ["b1"])
        self.assertEqual(MO.unit_caps(plain, rm2, u2), {})

    def test_cmd_plan_writes_the_cap_list_and_drops_stale_sidecars(self):
        with tempfile.TemporaryDirectory() as d:
            owed, paper, planner = (os.path.join(d, n) for n in ("owed.json", "paper.json", "planner.py"))
            json.dump(dict(OWED, rows=[self.ROW]), open(owed, "w"))
            json.dump(PAPER, open(paper, "w"))
            out = os.path.join(d, "plan")
            os.makedirs(out)
            for stale in ("ids-b1-full.stored.txt", "ids-b1-full.base.json"):
                open(os.path.join(out, stale), "w").write("stale\n")
            fake = fake_planner()
            orig = MO.load_planner
            MO.load_planner = lambda path: fake
            try:
                MO.main(["plan", "--owed", owed, "--paper-recipes", paper, "--planner", planner,
                         "--vendor", "nvidia", "--b-arms", "b1", "--out-dir", out])
            finally:
                MO.load_planner = orig
            self.assertFalse(os.path.exists(os.path.join(out, "ids-b1-full.stored.txt")))
            self.assertFalse(os.path.exists(os.path.join(out, "ids-b1-full.base.json")))
            self.assertEqual(open(os.path.join(out, "ids-b1-full.cap.txt")).read().split(),
                             sorted([self.TAXI, self.ISTELLA]))

    def test_base_caps_come_from_this_jobs_base_board(self):
        with tempfile.TemporaryDirectory() as d:
            board = os.path.join(d, "board.json")
            json.dump({"races": {
                self.TAXI: {"cells": [{"arm": "ours", "status": "ok", "median_ms": 1170000.2},
                                      {"arm": "cuml", "status": "ok", "median_ms": 5.0}]},
                self.ISTELLA: {"cells": [{"arm": "ours", "status": "REFUSED(error: x)", "median_ms": None}]}}},
                open(board, "w"))
            ids, cap_ids, out = (os.path.join(d, n) for n in ("chunk.txt", "cap.txt", "caps.json"))
            open(ids, "w").write("%s\n%s\n" % (self.TAXI, self.ISTELLA))
            open(cap_ids, "w").write("%s\n%s\n" % (self.TAXI, self.ISTELLA))
            self.assertEqual(MO.main(["base-caps", "--board", board, "--ids", ids, "--cap-ids", cap_ids,
                                      "--out", out]), 0)
            caps = json.load(open(out))["round_seconds"]
            # this job's base time, whole seconds up; no cap where the base has no time (the family ceiling applies)
            self.assertEqual(caps, {self.TAXI: 1171})
            import importlib.util
            spec = importlib.util.spec_from_file_location("bb_for_owed", os.path.join(HERE, "bench_board.py"))
            bb = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(bb)
            ctx = {"race_round_seconds": bb.race_round_seconds(out), "round_seconds": 0}
            self.assertEqual(bb.round_seconds_for(ctx, {"id": self.TAXI}, 1800), 1171)
            self.assertEqual(bb.round_seconds_for(ctx, {"id": self.ISTELLA}, 1800), 1800)
            # a race not in the unit's cap list is never capped
            open(cap_ids, "w").write("")
            MO.main(["base-caps", "--board", board, "--ids", ids, "--cap-ids", cap_ids, "--out", out])
            self.assertEqual(json.load(open(out))["round_seconds"], {})

    def report(self, d, b_cell, row=None):
        plan_dir, res = os.path.join(d, "plan"), os.path.join(d, "res")
        os.makedirs(plan_dir)
        json.dump({"r-cb": [self.TAXI]}, open(os.path.join(plan_dir, "rowmap.json"), "w"))
        json.dump({}, open(os.path.join(plan_dir, "not-planned.json"), "w"))
        probe = {"host_probe": {"schema": "mojolearn-host-probe/1", "host_id": self.HOST}}
        for sub, cell in (("board", {"arm": "ours", "status": "ok", "median_ms": 1170000.0}),
                          ("board-b1-full", b_cell)):
            os.makedirs(os.path.join(res, sub))
            json.dump({"box": probe, "races": {self.TAXI: {"cells": [cell]}}},
                      open(os.path.join(res, sub, "board.json"), "w"))
        return MO.report(dict(OWED, rows=[row or self.ROW]), plan_dir, res, "nvidia")

    def test_report_against_this_jobs_base(self):
        with tempfile.TemporaryDirectory() as d:
            (r,) = self.report(d, {"arm": "ours", "status": "ok", "median_ms": 585000.0})["rows"]
            self.assertEqual(r["candidate_ms"], 1170000.0)
            self.assertAlmostEqual(r["b_ratio"]["b1"], 0.5)
            self.assertIn("b1:B-FASTER", r["flags"])
            self.assertNotIn("BASE-STORED", r["flags"])

    def test_a_run_stopped_at_the_cap_is_not_faster_than_base(self):
        with tempfile.TemporaryDirectory() as d:
            rep = self.report(d, {"arm": "ours", "status": "REFUSED(timeout: null)", "median_ms": None})
            (r,) = rep["rows"]
            self.assertIn("b1:B-NOT-FASTER-THAN-BASE", r["flags"])
            self.assertIn("not faster than base (1)", MO.render_summary(rep))
        with tempfile.TemporaryDirectory() as d:     # another failure is not a verdict
            (r,) = self.report(d, {"arm": "ours", "status": "REFUSED(error: x)", "median_ms": None})["rows"]
            self.assertIn("b1:B-NOT-RUN", r["flags"])
        with tempfile.TemporaryDirectory() as d:     # a timeout of an uncapped row is no verdict either
            (r,) = self.report(d, {"arm": "ours", "status": "REFUSED(timeout: null)", "median_ms": None},
                               dict(self.ROW, cap_base=False))["rows"]
            self.assertIn("b1:B-NOT-RUN", r["flags"])

    def test_committed_dbscan_rows(self):
        doc = json.load(open(os.path.join(PC.OUT_DIR, "owed-pre0867.json")))
        by = {r["id"]: r for r in doc["rows"]}
        for rid in ("dbscan-brute-edge-free-taxi", "dbscan-brute-edge-free-istella"):
            self.assertTrue(by[rid].get("cap_base"))
            self.assertNotIn("stored_base", by[rid])           # no base time from another host
        self.assertEqual(by["dbscan-brute-edge-free-taxi-250k"]["rows"], 250000)
        self.assertNotIn("stored_base", by["dbscan-brute-edge-free-taxi-250k"])    # base and arm back to back
        self.assertNotIn("dbscan-brute-edge-free-taxi-750k", by)
        self.assertFalse([r["id"] for r in doc["rows"] if "stored_base" in r])

    def test_job_caps_each_chunk_from_its_base_board(self):
        with open(os.path.join(HERE, "ops", "paper_compare_job.sh")) as fh:
            text = fh.read()
        self.assertNotIn(".stored.txt", text)
        self.assertNotIn('.base.json"', text)
        i = text.index('tg_board candidate "$c.base" "$RES/board-tg-ab-$arm-base-$rows"')
        j = text.index("measure_owed.py base-caps", i)
        k = text.index('tg_board "$arm" "$c" "$RES/board-tg-ab-$arm-$rows"', j)
        self.assertLess(i, j)
        self.assertLess(j, k)


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

    def test_every_arm_is_classified(self):
        # lane fb8-owed-arms, 2026-11-17: RUN / DECIDED / PAUSED / NO-ROW; lane fb15-ab-verdicts, 2026-11-18: HELD (A/B
        # done, the flip waits on a decision) and INVALID (the run measured nothing; a RUN arm replaces it); a non-RUN
        # arm cites its evidence; an env arm names VAR=VALUE and no define; every RUN arm has a row
        doc = json.load(open(os.path.join(PC.OUT_DIR, "owed-pre0867.json")))
        used = {b for r in doc["rows"] for b in r.get("b_arms") or []}
        for name, arm in doc["b_arms"].items():
            self.assertIn(arm.get("owed"), ("RUN", "DECIDED", "PAUSED", "NO-ROW", "HELD", "INVALID"), name)
            if arm["owed"] != "RUN":
                self.assertTrue(arm.get("evidence"), name)
            if arm.get("env"):
                self.assertRegex(arm["env"], r"^[A-Z_][A-Z0-9_]*=\S+$")
                self.assertFalse(arm.get("define"), name)
            else:
                self.assertTrue(arm.get("define"), name)
            if arm.get("combined"):
                # lane ab-prep, 2026-11-23: a combined B set is built once and staged under each member's name, so no
                # row names it; its members are RUN arms with rows and its define holds every member's define
                members = arm["combined"]
                self.assertGreater(len(members), 1, name)
                self.assertNotIn(name, used, name)
                have = set(arm["define"].split(","))
                for m in members:
                    self.assertEqual(doc["b_arms"][m]["owed"], "RUN", f"{name}: {m}")
                    self.assertLessEqual(set(doc["b_arms"][m]["define"].split(",")), have, f"{name}: {m}")
            elif arm["owed"] == "RUN":
                self.assertIn(name, used, name)

    def test_every_run_arm_define_is_a_build_argument(self):
        # lane fx-fast-set-build, 2026-11-20: tools/release.py candidate --owed-arm hands an arm's `define` to the
        # build as it is written. Seven fast arms read "NAME (with MOJOLEARN_MODE_FAST_ROWS)" and their builds were
        # refused ("--defines: not NAME or NAME=VALUE"). A RUN arm's define is what normalize_defines takes, and a
        # relaxation of `fast` names MOJOLEARN_MODE_FAST_ROWS with it (checks/mode_policy.mojo `_row`)
        from release_github_build import normalize_defines
        doc = json.load(open(os.path.join(PC.OUT_DIR, "owed-pre0867.json")))
        # lane fx-verdicts6, 2026-11-20: the matrix-units route is the default of the fast set; its old path
        # (MOJOLEARN_GEMM_FAST_MATRIX_UNITS_OFF) is the one family name left, and means something only in a fast set
        rows = ("MOJOLEARN_MODE_FAST_", "MOJOLEARN_GEMM_FAST_MATRIX_UNITS")
        for name, arm in doc["b_arms"].items():
            if arm["owed"] != "RUN" or arm.get("env"):
                continue
            try:
                names = [e.split("=", 1)[0] for e in normalize_defines(arm["define"]).split(",")]
            except SystemExit as exc:
                self.fail(f"{name}: {exc}")
            self.assertTrue(names[0], name)
            if any(n.startswith(rows) for n in names):
                self.assertIn("MOJOLEARN_MODE_FAST_ROWS", names, name)


if __name__ == "__main__":
    unittest.main()
