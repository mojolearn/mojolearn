#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn
"""tools/test_measure_candidate.py -- tools/ops/measure_candidate.py stage's checks on in-memory fixtures (no box):
the B arm names against the owed file, the free-disk verdict and its prune list, where the candidate's source
tree (the dataset pins) is found, and the L40S-only GPU check of `stage nvidia`. Each test is independent (the suite runs its files in parallel)."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "ops"))
import measure_candidate as MC  # noqa: E402

GB = 1048576   # KiB per GiB (du -sk / df -Pk)


class BArms(unittest.TestCase):
    def owed(self, d):
        f = Path(d) / "owed.json"
        f.write_text(json.dumps({"b_arms": {"em-check-1": {}, "svd-one-block": {}}, "rows": []}))
        return f

    def test_unknown_name_refused_with_the_known_names(self):
        # 2026-11-13: six B sets staged as gbdtoff/svd1/... were all reported "not planned"
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit) as cm:
                MC.check_b_arms(["em-check-1", "svd1"], self.owed(d))
            self.assertIn("svd1", str(cm.exception))
            self.assertIn("svd-one-block", str(cm.exception))

    def test_known_names_pass_and_duplicates_refused(self):
        with tempfile.TemporaryDirectory() as d:
            MC.check_b_arms(["em-check-1", "svd-one-block"], self.owed(d))
            with self.assertRaises(SystemExit):
                MC.check_b_arms(["em-check-1", "em-check-1"], self.owed(d))

    def test_not_owed_and_wrong_form_refused(self):
        # lane fb8-owed-arms, 2026-11-17: a DECIDED / PAUSED / NO-ROW arm is never staged (nothing is timed twice);
        # an env arm is staged as NAME=env:VAR=VALUE, a built arm as NAME=DIR@SHA
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "owed.json"
            f.write_text(json.dumps({"b_arms": {
                "run1": {"define": "X", "owed": "RUN"}, "dec1": {"define": "Y", "owed": "DECIDED", "evidence": "L40S"},
                "env1": {"define": "", "env": "V=1", "owed": "RUN"}}, "rows": []}))
            MC.check_b_arms(["run1", "env1"], f, {"run1": None, "env1": "V=1"})
            with self.assertRaises(SystemExit) as cm:
                MC.check_b_arms(["dec1"], f)
            self.assertIn("DECIDED", str(cm.exception))
            with self.assertRaises(SystemExit):
                MC.check_b_arms(["env1"], f, {"env1": None})          # an env arm staged as a wheel set
            with self.assertRaises(SystemExit):
                MC.check_b_arms(["env1"], f, {"env1": "V=2"})         # another value than the file's
            with self.assertRaises(SystemExit):
                MC.check_b_arms(["run1"], f, {"run1": "X=1"})         # a built arm staged as env


class RunEnv(unittest.TestCase):
    """An arm's run variables come from its owed entry (lane fx-tools-round3, 2026-11-20: gemm-leaf512 was staged
    twice without MOJOLEARN_EXPERIMENT_GEMM_PROFILE, and its arm lost neural/gemm to the profile admission)."""
    PROFILE = "MOJOLEARN_EXPERIMENT_GEMM_PROFILE=mojolearn.identical.gemm.fp32.v2-leaf512"

    def owed(self, d, arms, rows=()):
        f = Path(d) / "owed.json"
        f.write_text(json.dumps({"b_arms": arms, "rows": list(rows)}))
        return f

    def test_built_arm_returns_its_run_env_and_others_none(self):
        with tempfile.TemporaryDirectory() as d:
            f = self.owed(d, {"leaf": {"define": "L", "owed": "RUN", "run_env": self.PROFILE},
                              "plain": {"define": "P", "owed": "RUN"},
                              "env1": {"define": "", "env": "V=@cgroup_cores", "owed": "RUN"}})
            got = MC.check_b_arms(["leaf", "plain", "env1"], f, {"leaf": None, "plain": None, "env1": "V=@cgroup_cores"})
            self.assertEqual(got, {"leaf": self.PROFILE, "plain": None, "env1": "V=@cgroup_cores"})
            # without the staged forms (older callers) the file's form decides
            self.assertEqual(MC.check_b_arms(["leaf", "env1"], f)["leaf"], self.PROFILE)

    def test_named_but_unset_env_refused(self):
        # the leaf512 entry: the status names the variable, no run_env carries it
        with tempfile.TemporaryDirectory() as d:
            f = self.owed(d, {"leaf": {"define": "L", "owed": "RUN",
                                       "status": "stage the arm with env:%s" % self.PROFILE}})
            with self.assertRaises(SystemExit) as cm:
                MC.check_b_arms(["leaf"], f, {"leaf": None})
            self.assertIn("MOJOLEARN_EXPERIMENT_GEMM_PROFILE", str(cm.exception))
            self.assertIn("run_env", str(cm.exception))
            # a row naming the arm counts too
            f = self.owed(d, {"leaf": {"define": "L", "owed": "RUN"}},
                          [{"id": "r", "b_arms": ["leaf"], "decides": "run with %s" % self.PROFILE}])
            with self.assertRaises(SystemExit):
                MC.check_b_arms(["leaf"], f, {"leaf": None})
            # set: admitted
            f = self.owed(d, {"leaf": {"define": "L", "owed": "RUN", "run_env": self.PROFILE,
                                       "status": "needs %s" % self.PROFILE}})
            self.assertEqual(MC.check_b_arms(["leaf"], f, {"leaf": None}), {"leaf": self.PROFILE})

    def test_env_the_job_cannot_supply_refused(self):
        with tempfile.TemporaryDirectory() as d:
            for bad in ("lower=1", "A=1 B=2", "A=@host_cores", "A=1,"):
                f = self.owed(d, {"x": {"define": "X", "owed": "RUN", "run_env": bad}})
                with self.assertRaises(SystemExit, msg=bad):
                    MC.check_b_arms(["x"], f, {"x": None})
            # run_env on an env arm: the env arm's variables are `env`
            f = self.owed(d, {"e": {"define": "", "env": "V=1", "run_env": "W=2", "owed": "RUN"}})
            with self.assertRaises(SystemExit):
                MC.check_b_arms(["e"], f, {"e": "V=1"})

    def test_stage_puts_the_run_env_on_that_arms_line_only(self):
        import contextlib
        import io
        from unittest import mock
        with tempfile.TemporaryDirectory() as d:
            cver = "0.8.67.dev0+aaaaaaaaaaaa"
            dirs = {}
            for name, ver in (("cand", cver), ("leaf", "0.8.67.dev0+bbbbbbbbbbbb"), ("plain", "0.8.67.dev0+cccccccccccc")):
                dirs[name] = os.path.join(d, name)
                os.makedirs(dirs[name])
                for dist in ("core", "nvidia"):
                    Path(dirs[name], "mojolearn_%s-%s-py3-none-any.whl" % (dist, ver)).write_text("w")
                if name != "cand":
                    Path(dirs[name], "candidate.json").write_text(json.dumps({"version": ver}))
            tree = os.path.join(d, "tree")
            os.makedirs(os.path.join(tree, "bench", "results", "paper_compare"))
            os.makedirs(os.path.join(tree, "bench", "results", "dataset_store"))
            Path(tree, "bench", "results", "paper_compare", "recipes-nvidia.json").write_text("{}")
            Path(tree, "bench", "results", "dataset_store", "manifest.tsv").write_text("")
            owed = self.owed(d, {"leaf": {"define": "L", "owed": "RUN", "run_env": self.PROFILE},
                                 "plain": {"define": "P", "owed": "RUN"}})
            argv = ["stage", "nvidia", "--version", cver, "--source-commit", "a" * 40, "--wheel-dir", dirs["cand"],
                    "--dry-run", "--state-dir", d, "--tree", tree, "--owed-file", str(owed),
                    "--b-arm", "leaf=%s@%s" % (dirs["leaf"], "b" * 40),
                    "--b-arm", "plain=%s@%s" % (dirs["plain"], "c" * 40)]
            manifest = lambda version, vendor, wheels, commit: {  # noqa: E731
                "files": [{"file": Path(w).name, "sha256": "0" * 64} for w in wheels]}
            buf = io.StringIO()
            with mock.patch.object(MC.C, "build_manifest", manifest), \
                    mock.patch.object(MC, "dataset_keys", lambda *a, **k: ["k1"]), \
                    contextlib.redirect_stdout(buf):
                self.assertEqual(MC.main(argv), 0)
            self.assertIn("B arm leaf: runs with %s" % self.PROFILE, buf.getvalue())
            lines = Path(d, "b-arms-aaaaaaaaaaaa-nvidia.txt").read_text().splitlines()
            self.assertEqual(lines, ["leaf /root/pc-in/aaaaaaaaaaaa/b-leaf/nvidia %s env:%s" % ("b" * 40, self.PROFILE),
                                     "plain /root/pc-in/aaaaaaaaaaaa/b-plain/nvidia %s" % ("c" * 40)])


class Built(unittest.TestCase):
    def test_stage_refuses_a_vendor_the_targets_did_not_build(self):
        with tempfile.TemporaryDirectory() as d:
            MC.check_built(d, "amd")    # no candidate.json: a release or an older candidate built every set
            (Path(d) / "candidate.json").write_text(json.dumps(
                {"version": "v", "targets": ["sm_89", "sm_80"], "wheels_built": ["nvidia"]}))
            MC.check_built(d, "nvidia")
            with self.assertRaises(SystemExit) as cm:
                MC.check_built(d, "amd")
            self.assertIn("packed no amd wheel", str(cm.exception))


class Disk(unittest.TestCase):
    OUT = "\n".join([
        "FREE_KB %d" % (26 * GB),
        "RES %d\t/root/mojolearn-measure/box-out/pc-old-nvidia/" % (30 * GB),
        "RES %d\t/root/mojolearn-measure/box-out/pc-abc-nvidia/" % (2 * GB),
        "STAGING %d\t/root/box-persist/state/pc-old-nvidia/upload.tgz" % (18 * GB),
        "CACHE %d\t/root/pc-in/0123456789ab/" % GB,
        "CACHE %d\t/root/pc-in/abcdefabcdef/" % GB,
        "CACHE %d\t/root/pc-cache/c1/" % GB,
        "CACHE %d\t/root/pc-cache/c2/" % GB,
        "CACHE %d\t/root/pc-cache/c12/" % GB])

    def test_need_is_largest_results_plus_headroom(self):
        free, need, prune = MC.disk_verdict(self.OUT, "pc-abc-nvidia", "abcdefabcdef")
        self.assertAlmostEqual(free, 26.0)
        self.assertAlmostEqual(need, 50.0)                     # 30 GB earlier results + 20
        paths = [p for _, p, _ in prune]
        self.assertEqual(paths[0], "/root/mojolearn-measure/box-out/pc-old-nvidia")     # largest first
        self.assertIn("/root/box-persist/state/pc-old-nvidia/upload.tgz", paths)
        self.assertIn("/root/pc-in/0123456789ab", paths)
        self.assertIn("/root/pc-cache/c2", paths)
        self.assertIn("/root/pc-cache/c12", paths)
        for keep in ("/root/mojolearn-measure/box-out/pc-abc-nvidia", "/root/pc-in/abcdefabcdef", "/root/pc-cache/c1"):
            self.assertNotIn(keep, paths)

    def test_floor_without_earlier_results(self):
        free, need, prune = MC.disk_verdict("FREE_KB %d\n" % (70 * GB), "t", "abcdefabcdef")
        self.assertEqual((round(free), need, prune), (70, float(MC.MIN_FREE_GB), []))

    def test_unreadable_df_refused(self):
        with self.assertRaises(SystemExit):
            MC.disk_verdict("df: /root: No such file\n", "t", "abcdefabcdef")

    def test_script_reports_every_kind(self):
        s = MC.disk_script("/root/mojolearn-measure/box-out")
        for word in ("FREE_KB", "RES", "STAGING", "CACHE", "/root/box-persist/state/", "df -Pk /root"):
            self.assertIn(word, s)


class Source(unittest.TestCase):
    def test_source_tree_beside_the_wheel_dir(self):
        # tools/release.py candidate: <work dir>/<commit12>/linux/final and <work dir>/source/<commit12>
        with tempfile.TemporaryDirectory() as d:
            w = Path(d) / "0.8.67.dev0+abc" / "abcdefabcdef" / "linux" / "final"
            w.mkdir(parents=True)
            self.assertEqual(MC.source_tree(w, "abcdefabcdef" + "0" * 28),
                             (Path(d) / "0.8.67.dev0+abc" / "source" / "abcdefabcdef").resolve())
            self.assertEqual(MC.source_tree(w, "x", override=d), Path(d).resolve())

    def test_results_file_falls_back_to_the_source_tree(self):
        # the lean tooling worktree has no bench/results/: the pins come from the candidate's source tree
        with tempfile.TemporaryDirectory() as d:
            lean, src = Path(d) / "lean", Path(d) / "src"
            m = src / "bench" / "results" / "dataset_store" / "manifest.tsv"
            m.parent.mkdir(parents=True)
            m.write_text("k\t1\tx\n")
            lean.mkdir()
            self.assertEqual(MC.results_file("dataset_store/manifest.tsv", lean, src), m)
            with self.assertRaises(SystemExit):
                MC.results_file("paper_compare/none.json", lean, src)

    def test_stage_plan_targets(self):
        nv = MC.StagePlan.__new__(MC.StagePlan)
        nv.vendor, nv.lane, nv.wrapper = "nvidia", "measure", "tools/nvidia_central.sh"
        self.assertEqual(nv.ssh_target(), ["bash", "tools/nvidia_central.sh", "ssh-target", "measure"])
        self.assertEqual(nv.results_base(), "/root/mojolearn-measure/box-out")
        amd = MC.StagePlan.__new__(MC.StagePlan)
        amd.vendor, amd.lane, amd.wrapper = "amd", "measure", "tools/do_amd_steward.sh"
        self.assertEqual(amd.results_base(), "/root/mojolearn/box-out")


class GpuCheck(unittest.TestCase):
    """`stage nvidia` measures on an L40S only (2026-11-13: an out-of-stock L40S got a 2x RTX 6000 Ada rented)."""

    class Runner:
        def __init__(self, out):
            self.out, self.calls = out, []

        def must(self, argv, what, timeout=None):
            self.calls.append(argv)
            return type("Got", (), {"stdout": self.out})()

    def plan(self):
        nv = MC.StagePlan.__new__(MC.StagePlan)
        nv.vendor, nv.lane, nv.wrapper = "nvidia", "measure", "tools/nvidia_central.sh"
        return nv

    def test_non_l40s_name_refused(self):
        r = self.Runner("NVIDIA RTX 6000 Ada Generation\nNVIDIA RTX 6000 Ada Generation\n")
        with self.assertRaises(SystemExit) as cm:
            MC.check_gpu(r, self.plan(), allow_hw_diff=False)
        self.assertIn("RTX 6000 Ada", str(cm.exception))
        self.assertIn("--allow-hw-diff", str(cm.exception))
        self.assertEqual(r.calls[0][-1], MC.GPU_SCRIPT)

    def test_unreadable_names_refused(self):
        with self.assertRaises(SystemExit):
            MC.check_gpu(self.Runner(""), self.plan(), allow_hw_diff=False)

    def test_l40s_passes_and_allow_goes_on(self):
        MC.check_gpu(self.Runner("NVIDIA L40S\n"), self.plan(), allow_hw_diff=False)
        MC.check_gpu(self.Runner("NVIDIA GeForce RTX 4090\n"), self.plan(), allow_hw_diff=True)

    def test_verdict_needs_every_gpu_to_be_l40s(self):
        self.assertEqual(MC.gpu_verdict("NVIDIA L40S\nNVIDIA L40S\n"), (["NVIDIA L40S", "NVIDIA L40S"], True))
        self.assertFalse(MC.gpu_verdict("NVIDIA L40S\nNVIDIA L4\n")[1])

    def test_job_script_refuses_another_nvidia_gpu(self):
        job = (Path(HERE) / "ops" / "paper_compare_job.sh").read_text()
        self.assertIn('[ "$V" = nvidia ] && [ "${PC_ALLOW_HW_DIFF:-0}" != 1 ]', job)
        self.assertIn("TOKEN=${PC_GPU_TOKEN:-L40S}", job)


class QueueCheck(unittest.TestCase):
    """`stage nvidia` never syncs a tree a job is reading (2026-11-13: a restage during job 0001 made bench_board stop
    with "source changed during measurement")."""

    QUEUE = ("== nvc1: pod abc, 1 GPU (NVIDIA L40S), $0.86/hr\n"
             "ID    LANE          GPUS STATE       SUBMITTED            STARTED              SCRIPT\n"
             "0003  measure       1    queued      2026-11-13T10:00:00Z -                    /root/x/job.sh\n"
             "0004  other         1    running@0   2026-11-13T10:00:00Z 2026-11-13T10:01:00Z /root/y/job.sh\n"
             "-- last finished:\n"
             "0001  measure       1    done 0      2026-11-13T09:00:00Z 2026-11-13T09:30:00Z /root/z/job.sh\n"
             "slot 0: BUSY other\n")

    class Runner:
        def __init__(self, out):
            self.out, self.calls = out, []

        def must(self, argv, what, timeout=None):
            self.calls.append(argv)
            return type("Got", (), {"stdout": self.out})()

    def plan(self, lane="measure"):
        nv = MC.StagePlan.__new__(MC.StagePlan)
        nv.vendor, nv.lane, nv.wrapper = "nvidia", lane, "tools/nvidia_central.sh"
        return nv

    def test_lane_jobs_reads_live_rows_only(self):
        self.assertEqual(MC.lane_jobs(self.QUEUE, "measure"), [("nvc1", "0003", "queued", "/root/x/job.sh")])
        self.assertEqual(MC.lane_jobs(self.QUEUE, "other"), [("nvc1", "0004", "running", "/root/y/job.sh")])
        self.assertEqual(MC.lane_jobs(self.QUEUE, "idle"), [])
        self.assertEqual(MC.lane_jobs("no shared NVIDIA pod is up\n", "measure"), [])

    def test_live_job_refuses_the_sync(self):
        r = self.Runner(self.QUEUE)
        with self.assertRaises(SystemExit) as cm:
            MC.check_queue(r, self.plan(), allow_running=False)
        self.assertIn("nvc1-0003 queued", str(cm.exception))
        self.assertIn("--allow-running", str(cm.exception))
        self.assertEqual(r.calls[0], ["bash", "tools/nvidia_central.sh", "queue"])

    def test_finished_job_or_override_goes_on(self):
        MC.check_queue(self.Runner(self.QUEUE), self.plan("idle"), allow_running=False)
        MC.check_queue(self.Runner(self.QUEUE), self.plan(), allow_running=True)

    def test_check_runs_before_the_sync(self):
        src = (Path(HERE) / "ops" / "measure_candidate.py").read_text()
        self.assertLess(src.index("check_queue(runner, plan, a.allow_running)"),
                        src.index('runner.must(plan.sync(), "the lane sync"'))
        self.assertIn('s.add_argument("--allow-running"', src)



class Priors(unittest.TestCase):
    def test_one_prior_by_default(self):
        # Andrew, 2026-11-16: a measurement job times ONE prior, 0.8.55; 0.8.66 is never timed
        self.assertEqual(MC.DEFAULT_PRIOR_VERSIONS, "0.8.55")
        self.assertEqual(MC.DEFAULT_PRIOR_VERSIONS, MC.PC.PAPER_VERSION)
        self.assertIn('s.add_argument("--prior-versions", default=DEFAULT_PRIOR_VERSIONS', Path(MC.__file__).read_text())

    def test_stage_keys_follow_the_owed_plan(self):
        # the staged datasets are the owed rows the job plans (B sets staged, a last release or not)
        src = Path(MC.__file__).read_text()
        self.assertIn("[n for n, _, _ in specs], prior_versions[-1] != PC.PAPER_VERSION and not stored, parts, a.rows)", src)


class Parts(unittest.TestCase):
    """--parts (2026-11-16: two PC_PARTS=board-all submits died on corpus/enwik8/input.txt and corpus/pile_github/
    input.txt, which stage had not staged: it listed the paper job's datasets only)."""
    PLANNER = os.path.join(HERE, "bench_board.py")

    def test_board_all_stages_the_full_boards_datasets(self):
        keys = MC.dataset_keys("nvidia", None, None, self.PLANNER, parts=MC.parse_parts("board-all"))
        self.assertIn("corpus/enwik8/input.txt", keys)
        self.assertIn("corpus/pile_github/input.txt", keys)
        # the same keys for one all-* part given alone (a split submit)
        self.assertEqual(keys, MC.dataset_keys("nvidia", None, None, self.PLANNER, parts=["all-smoke"]))

    def test_unknown_part_refused_and_default_is_the_paper_job(self):
        with self.assertRaises(SystemExit):
            MC.parse_parts("board-all,boards")
        self.assertEqual(MC.parse_parts(""), [])
        self.assertFalse(MC.runs_board_all(MC.parse_parts("plan,install,smoke,board,prior,compare")))
        self.assertTrue(MC.runs_board_all(MC.parse_parts("board-all")))

    def test_parts_match_the_job_and_ride_the_submit_line(self):
        job = Path(HERE, "ops", "paper_compare_job.sh").read_text()
        self.assertIn('PC_PARTS:-%s}' % ",".join(MC.PAPER_PARTS), job)
        self.assertIn(",".join(MC.ALL_PARTS[1:]), job)
        line = MC.submit_line("nvidia", "measure", "t", {"PC_PARTS": "board-all"})
        self.assertIn("--env PC_PARTS=board-all", line)


class AbEstimate(unittest.TestCase):
    """stored_times (newest wall), ab_units (shared base on one clock), split_boxes and _box_argv (lane
    ab-harness-speed, 2026-11-24)."""

    def test_stored_times_takes_the_newest_wall(self):
        with tempfile.TemporaryDirectory() as t:
            old, new = Path(t) / "old.json", Path(t) / "new.json"
            old.write_text(json.dumps({"races": {"f/a": {"status": "done", "wall_s": 1748.0,
                                                          "finished": "2026-11-18T10:00:00Z"},
                                                  "f/b": {"status": "done", "wall_s": 5.0}}}))
            new.write_text(json.dumps({"races": {"f/a": {"status": "done", "wall_s": 416.0,
                                                          "finished": "2026-11-24T10:00:00Z"}}}))
            self.assertEqual(MC.stored_times([new, old]), {"f/a": 416.0, "f/b": 5.0})

    def test_a_base_shared_within_the_bound_is_counted_once(self):
        units = {("x", "full"): ["f/a"], ("y", "full"): ["f/a"]}
        secs = {"f/a": 60.0}
        shared = MC.ab_units(units, secs, smoke=False)
        alone = MC.ab_units(units, secs, smoke=False, max_age_s=0)
        p = MC.TR.PROC_S
        self.assertAlmostEqual(shared[0][3] * 60, 2 * p + 120)
        self.assertAlmostEqual(shared[1][3] * 60, p + 60)            # y: its arm only, the base is x's
        self.assertAlmostEqual(alone[1][3] * 60, 2 * p + 120)

    def test_smoke_counted_once_per_race_and_only_when_the_submit_smokes(self):
        units = {("x", "full"): ["f/a"], ("y", "full"): ["f/a"]}
        on = sum(m for *_, m in MC.ab_units(units, {"f/a": 60.0}))
        off = sum(m for *_, m in MC.ab_units(units, {"f/a": 60.0}, smoke=False))
        self.assertAlmostEqual((on - off) * 60, MC.TR.SMOKE_S_PER_RACE)

    def test_split_boxes_keeps_sharing_arms_together_and_balances(self):
        units = {("a", "full"): ["f/1"], ("b", "full"): ["f/1"], ("c", "full"): ["f/2"], ("d", "full"): ["f/3"]}
        secs = {"f/1": 600.0, "f/2": 600.0, "f/3": 600.0}
        boxes = MC.split_boxes(units, secs, 2)
        self.assertEqual(len(boxes), 2)
        self.assertIn(["a", "b"], boxes)
        self.assertEqual(sorted(sum(boxes, [])), ["a", "b", "c", "d"])

    def test_split_boxes_breaks_a_group_larger_than_its_share(self):
        units = {(n, "full"): ["f/1"] for n in "abcd"}
        boxes = MC.split_boxes(units, {"f/1": 600.0}, 2)
        self.assertEqual(sorted(len(b) for b in boxes), [2, 2])

    def test_box_argv_names_each_boxs_arms_lane_tag_and_owed_only(self):
        argv = ["stage", "nvidia", "--boxes", "3", "--lane=measure", "--tag", "ab9", "--rows", "r.tsv"]
        one = MC._box_argv(argv, 1, ["a", "b"], "measure", "ab9")
        two = MC._box_argv(argv, 2, ["c"], "measure", "ab9")
        self.assertEqual(one, ["stage", "nvidia", "--rows", "r.tsv", "--boxes", "1", "--owed-arms", "a,b",
                               "--lane", "measure", "--tag", "ab9"])
        self.assertEqual(two[-7:], ["--owed-arms", "c", "--lane", "measure-b2", "--tag", "ab9-b2", "--owed-only"])


if __name__ == "__main__":
    unittest.main()
