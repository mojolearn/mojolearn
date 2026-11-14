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


if __name__ == "__main__":
    unittest.main()
