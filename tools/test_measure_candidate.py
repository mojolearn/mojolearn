#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn
"""tools/test_measure_candidate.py -- tools/ops/measure_candidate.py stage's checks on in-memory fixtures (no box):
the B arm names against the owed file, the free-disk verdict and its prune list, and where the candidate's source
tree (the dataset pins) is found. Each test is independent (the suite runs its files in parallel)."""
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


if __name__ == "__main__":
    unittest.main()
