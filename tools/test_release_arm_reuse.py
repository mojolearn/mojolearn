# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn
"""A B arm reuses its base build (lane arm-build-reuse, 2026-11-16; tools/release_reuse.py base_build, unit_plan,
skip_units; tools/release_github_build.py build_plan --skip-units; tools/release_candidate.py run_arms):

1. the closure rule: a row whose closure names the define (a guard module that only refuses it too) builds, every
   other row keeps the base's bytes;
2. a shard unit builds when one member builds or a member has no plan row; otherwise it is REUSE and skipped;
3. the base is the finished A set of the same commit and targets with no defines (pending only for a dry run);
4. the workflow planner drops the skipped entries from the shard matrix, keeps the assemble swap of the whole set,
   and refuses an entry it does not build;
5. several arms run at once, one process per distinct define set.

Stdlib only; no network, no GitHub, no build.

    python3 -m unittest tools.test_release_arm_reuse
"""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import unittest.mock as mock

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
sys.argv = [sys.argv[0]]
import release_candidate as RC  # noqa: E402
import release_github_build as G  # noqa: E402
import release_reuse as R  # noqa: E402

C = "c" * 40
L = R.LINUX


def row(name, tier="identical", vendor="cuda", arch="sm_89", decision="REUSE", script=None):
    b = R.Binding(L, "host" if tier == "host" else vendor, "" if tier == "host" else arch, tier, name)
    return dict(b.row(), decision=decision, reason="identity unchanged" if decision == "REUSE" else "changed",
                identity_digest="d" * 64, reuse_from=None)


class ClosureRule(unittest.TestCase):
    """1: apply_defines moves exactly the rows whose closure names the define."""

    def test_readers_build_others_keep_the_base(self):
        names = ["_mojolearn_kde", "_mojolearn_gbdt", "_mojolearn_svm"]
        rows = [row(n) for n in names]
        current = [(R.Binding(L, "cuda", "sm_89", "identical", n), None) for n in names]
        facts = {"closures": {
            "build_kde.sh": {"sources": ["kde/impl/kernel_density.mojo", "core/x.mojo"]},
            "build_gbdt.sh": {"sources": ["gbdt/a.mojo"]},
            # a guard module that only refuses the define in fast mode still names it: it reads it
            "build_svm.sh": {"sources": ["core/experiment_guards_kernel.mojo"]}}}
        readers = {"kde/impl/kernel_density.mojo", "core/experiment_guards_kernel.mojo", "docs/EXPERIMENTS.md"}
        moved = R.apply_defines(rows, current, facts, readers, ["MOJOLEARN_X"])
        self.assertEqual([r["name"] for r in moved], ["_mojolearn_kde", "_mojolearn_svm"])
        self.assertEqual([r["decision"] for r in rows], ["BUILD", "REUSE", "BUILD"])
        self.assertIn("core/experiment_guards_kernel.mojo", rows[2]["reason"])

    def test_an_unreadable_closure_never_reuses_by_accident(self):
        rows = [row("_mojolearn_kde", decision="BUILD")]
        current = [(R.Binding(L, "cuda", "sm_89", "identical", "_mojolearn_kde"), None)]
        R.apply_defines(rows, current, {"closures": {}}, {"x.mojo"}, ["MOJOLEARN_X"])
        self.assertEqual(rows[0]["decision"], "BUILD")


class TimedSets(unittest.TestCase):
    """--timed-on (lane fx-build-tools, 2026-11-18): an arm timed on sm_89 builds the define's device rows in
    sm_89 only; the sm_80 rows ship the base's bytes and say so; host rows always build."""

    def rows(self):
        specs = [("_mojolearn_kde", "identical", "sm_89"), ("_mojolearn_kde", "identical", "sm_80"),
                 ("_mojolearn_kde_host", "host", "")]
        rows = [row(n, tier=t, arch=a) for n, t, a in specs]
        current = [(R.Binding(L, "host" if t == "host" else "cuda", a, t, n), None) for n, t, a in specs]
        facts = {"closures": {"build_kde.sh": {"sources": ["kde/k.mojo"]},
                              "build_kde_host.sh": {"sources": ["kde/k.mojo"]}}}
        return rows, current, facts

    def test_default_builds_every_set(self):
        rows, current, facts = self.rows()
        R.apply_defines(rows, current, facts, {"kde/k.mojo"}, ["MOJOLEARN_X"])
        self.assertEqual([r["decision"] for r in rows], ["BUILD", "BUILD", "BUILD"])

    def test_untimed_set_keeps_the_base(self):
        rows, current, facts = self.rows()
        moved = R.apply_defines(rows, current, facts, {"kde/k.mojo"}, ["MOJOLEARN_X"], timed=R.timed_sets("sm_89"))
        self.assertEqual([r["decision"] for r in rows], ["BUILD", "REUSE", "BUILD"])
        self.assertEqual(len(moved), 2)
        self.assertTrue(rows[1]["reason"].startswith("untimed set:"))
        rec = RC.untimed_record(dict(timed_sets=["sm_89"], rows=rows), ["sm_89", "sm_80"])
        self.assertEqual(rec, dict(timed_on=["sm_89"], untimed_from_base={"cuda/sm_80": ["_mojolearn_kde"]}))

    def test_timed_on_is_checked(self):
        self.assertEqual(RC.timed_on("", ["sm_89", "sm_80"], "MOJOLEARN_X"), [])
        self.assertEqual(RC.timed_on("sm_89", ["sm_89", "sm_80"], "MOJOLEARN_X"), ["sm_89"])
        self.assertEqual(RC.timed_on("sm_80,sm_89", ["sm_89", "sm_80"], "MOJOLEARN_X"), [])   # every set: default
        with self.assertRaises(SystemExit):
            RC.timed_on("sm_89", ["sm_89", "sm_80"], "")          # an A set
        with self.assertRaises(SystemExit):
            RC.timed_on("gfx942", ["sm_89", "sm_80"], "MOJOLEARN_X")
        self.assertIsNone(R.timed_sets(""))
        self.assertEqual(R.timed_sets("sm_89"), {("cuda", "sm_89")})


class Units(unittest.TestCase):
    """2: rows lifted to the shard units."""

    def setUp(self):
        self.sets = mock.patch.object(R, "LINUX_SETS", (("cuda", "sm_89"),))
        self.sets.start()

    def tearDown(self):
        self.sets.stop()

    def plan(self, rows):
        return dict(rows=rows, legs=["cuda-sm_89"])

    def test_one_reading_member_builds_its_merged_group(self):
        rows = [row("_mojolearn", decision="BUILD"), row("_mojolearn_metrics"), row("_mojolearn_gbdt"),
                row("_mojolearn_core_host", tier="host"), row("_mojolearn_forest_host", tier="host", decision="BUILD")]
        groups = {"identical": {"_mojolearn_native": ["_mojolearn", "_mojolearn_metrics"],
                                "_mojolearn_native_trees": ["_mojolearn_gbdt"]},
                  "host": {"_mojolearn_native_host": ["_mojolearn_core_host"],
                           "_mojolearn_native_forest_host": ["_mojolearn_forest_host"]}}
        units = {u["unit"]: u for u in R.unit_plan(self.plan(rows), groups)["cuda/sm_89"]}
        self.assertEqual(units["identical:merged_build.sh:_mojolearn_native"]["decision"], "BUILD")
        self.assertEqual(units["identical:merged_build.sh:_mojolearn_native_trees"]["decision"], "REUSE")
        self.assertEqual(units["host:merged_build.sh:_mojolearn_native_host"]["decision"], "REUSE")
        self.assertEqual(units["host:merged_build.sh:_mojolearn_native_forest_host"]["decision"], "BUILD")
        plan = dict(self.plan(rows), units=R.unit_plan(self.plan(rows), groups))
        self.assertEqual(sorted(R.skip_units(plan, "cuda", "sm_89")),
                         ["host:merged_build.sh:_mojolearn_native_host",
                          "identical:merged_build.sh:_mojolearn_native_trees"])

    def test_a_member_without_a_row_builds(self):
        groups = {"identical": {"_mojolearn_native": ["_mojolearn", "_mojolearn_new"]}}
        u = R.unit_plan(self.plan([row("_mojolearn")]), groups)["cuda/sm_89"][0]
        self.assertEqual(u["decision"], "BUILD")
        self.assertIn("_mojolearn_new", u["reason"])

    def test_per_binding_layout(self):
        units = R.unit_plan(self.plan([row("_mojolearn_gbdt", decision="BUILD"), row("_mojolearn_core_host", tier="host")]),
                            None)["cuda/sm_89"]
        self.assertEqual({u["unit"]: u["decision"] for u in units},
                         {"identical:build_gbdt.sh": "BUILD", "identical:build_core_host.sh": "REUSE"})

    def test_a_plan_without_units_skips_nothing(self):
        self.assertEqual(R.skip_units(dict(rows=[]), "cuda", "sm_89"), [])

    def test_a_full_build_is_named(self):
        rows = [row("_mojolearn", decision="BUILD")]
        plan = dict(self.plan(rows), units=R.unit_plan(self.plan(rows), {"identical": {"g": ["_mojolearn"]}}),
                    previous=dict(version="v", base=True), candidate_defines="MOJOLEARN_X")
        self.assertTrue(any("FULL BUILD" in line for line in R.render_units(plan)))
        self.assertIn("FULL BUILD", RC.arm_summary("x", plan))

    def test_pending_base_keeps_the_ptx_rows(self):
        rows = [row("_mojolearn", arch="sm_80")]
        got = R._settle_ptx(rows, "REUSE", dict(version="v", pending=True), None, None)
        self.assertTrue(got["pending"])
        self.assertEqual(rows[0]["decision"], "REUSE")


class Base(unittest.TestCase):
    """3: which build is a B arm's base."""

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.work = Path(self.td.name) / "0.8.67.dev0+cccccccccccc"
        self.work.mkdir()

    def tearDown(self):
        self.td.cleanup()

    def write(self, defines="", targets=("sm_89", "sm_80"), finished=True):
        (self.work / "state.json").write_text(json.dumps(dict(commit=C, candidate=dict(
            version=self.work.name, defines=defines, targets=list(targets)))))
        if finished:
            d = self.work / C[:12]
            (d / "linux" / "final").mkdir(parents=True)
            (d / "reuse").mkdir()
            (d / "linux" / "final" / "candidate.json").write_text(json.dumps(dict(
                source_commit=C, defines="", wheels=dict(linux=dict(file="core.whl", sha256="a" * 64),
                                                         nvidia=dict(file="nv.whl", sha256="b" * 64)))))
            (d / "reuse" / "plan.json").write_text(json.dumps(dict(schema=R.PLAN_SCHEMA, commit=C, rows=[])))

    def test_a_finished_a_set(self):
        self.write()
        b = R.base_build(self.work, C, ["sm_89", "sm_80"])
        self.assertEqual((b["channel"], b["linux"]["wheel"], b["nvidia"]["sha256"]), ("base", "core.whl", "b" * 64))
        self.assertTrue(b["identities"].endswith("reuse/plan.json"))
        self.assertIsNone(b["amd"])
        self.assertFalse(b.get("pending"))

    def test_not_a_base(self):
        self.write(defines="MOJOLEARN_X")
        self.assertIsNone(R.base_build(self.work, C, ["sm_89", "sm_80"]))           # a B set is no base
        self.assertIsNone(R.base_build(self.work, "d" * 40, ["sm_89", "sm_80"]))     # another commit

    def test_other_targets(self):
        self.write(targets=("sm_89", "sm_80"))
        self.assertIsNone(R.base_build(self.work, C, ["sm_89", "sm_80", "gfx942"]))

    def test_still_building(self):
        self.write(finished=False)
        self.assertIsNone(R.base_build(self.work, C, ["sm_89", "sm_80"]))           # a real run never takes it
        b = R.base_build(self.work, C, ["sm_89", "sm_80"], dry=True)
        self.assertTrue(b["pending"])
        self.assertIsNone(b["identities"])

    def test_none_at_all(self):
        self.assertIsNone(R.base_build(Path(self.td.name) / "missing", C, ["sm_89"]))


class WorkflowPlan(unittest.TestCase):
    """4: build_plan --skip-units."""

    def plan(self, skip):
        builds = [("identical:merged_build.sh:a", False), ("identical:merged_build.sh:b", False),
                  ("host:merged_build.sh:h", True)]
        ps = dict(builds=builds, per_binding=[], merged=True, members={e: 1 for e, _ in builds},
                  shares={e: 0.5 for e, _ in builds}, weights={e: 1 for e, _ in builds})
        with mock.patch.object(G, "plan_set", return_value=ps), mock.patch.object(G, "check_counts"):
            return G.build_plan("src", ["sm_89"], 6, ["identical"], skip_units=skip)

    def test_skipped_entries_get_no_shard_job(self):
        full = self.plan("")
        doc = self.plan("identical:merged_build.sh:b host:merged_build.sh:h")
        self.assertEqual([r["only"] for r in doc["matrix"]["include"]], ["identical:merged_build.sh:a"])
        self.assertEqual(doc["skipped_units"], ["host:merged_build.sh:h", "identical:merged_build.sh:b"])
        # a skipped unit the cache misses compiles in the assemble job, with the whole set's swap
        self.assertEqual(doc["assemble_swap_gib"], full["assemble_swap_gib"])
        self.assertEqual(full["skipped_units"], [])

    def test_an_unknown_entry_is_refused(self):
        with self.assertRaises(SystemExit):
            self.plan("identical:merged_build.sh:zzz")

    def test_run_refuses_skip_units_without_defines(self):
        with self.assertRaises(SystemExit):
            G.main(["run", "--commit", C, "--arch", "sm_89", "--out", "/nonexistent-out",
                    "--skip-units", "identical:merged_build.sh:a"])


class Arms(unittest.TestCase):
    """5: several arms at once."""

    def test_one_process_per_distinct_define_set(self):
        with tempfile.TemporaryDirectory() as td:
            owed = Path(td) / "owed.json"
            owed.write_text(json.dumps(dict(b_arms={"a-one": dict(define="MOJOLEARN_A"),
                                                    "a-same": dict(define="MOJOLEARN_A"),
                                                    "b-two": dict(define="MOJOLEARN_B")})))
            argv = ["--commit", C, "--owed-file", str(owed), "--work-root", td, "--dry-run",
                    "--owed-arm", "a-one", "--owed-arm", "a-same", "--owed-arm=b-two"]
            calls = []

            def fake(cmd, **kw):
                calls.append(cmd)
                kw["stdout"].write(f"ARM {cmd[-1]}: units BUILD 1 / REUSE 2\n")
                return 0
            with mock.patch.object(RC.subprocess, "call", side_effect=fake):
                self.assertEqual(RC.main(argv), 0)
            self.assertEqual(sorted(c[-1] for c in calls), ["a-one", "b-two"])
            for c in calls:
                self.assertEqual(c.count("--owed-arm"), 1)
                self.assertNotIn("--owed-arm=b-two", c[:-2])

    def test_the_dbscan_arm_is_owed(self):
        # the edge-free route was promoted on 2026-11-20 (lane fx-verdicts5): the owed arm is its old path
        self.assertEqual(RC.resolve_owed("", "dbscan-brute-edge-free-off", "")[1], "MOJOLEARN_DBSCAN_FAST_BRUTE_EDGE_FREE_OFF")


if __name__ == "__main__":
    unittest.main()
