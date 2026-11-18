# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn
"""tools/release_candidate.py's disk handling: the free-space refusal (before linux-assemble and before several
arms), the prune list of a finished set and what it keeps, and the retried `git ls-remote`. Temp dirs and fakes
only: no build, no network.

    python3 -m unittest tools.tests.test_release_candidate_disk
"""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import release_candidate as rc  # noqa: E402

GIB = rc.GIB


def put(path, size=10):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return path


class SpaceTest(unittest.TestCase):

    def test_enough_space_passes_and_too_little_refuses_with_the_numbers(self):
        need = 2 * rc.ASSEMBLE_PER_LINUX_SET_BYTES
        self.assertIsNone(rc.space_refusal("/x", need, "linux-assemble", free=need + rc.RESERVE_BYTES))
        why = rc.space_refusal("/x", need, "linux-assemble of 2 Linux set(s)", free=need)
        self.assertIn("linux-assemble of 2 Linux set(s)", why)
        self.assertIn("GB free at /x", why)
        self.assertIn("GB needed", why)

    def test_eight_arms_on_5_6_gb_are_refused_one_fits(self):
        # 2026-11-16: eight arms of sm_89+sm_80 with 5.6 GB free; seven died at linux-assemble
        free = int(5.6 * GIB)
        self.assertIsNotNone(rc.space_refusal("/x", 8 * rc.arm_need(2), "ARMS", free=free))
        self.assertIsNone(rc.space_refusal("/x", 1 * rc.arm_need(2), "ARMS", free=free + 2 * GIB))

    def test_free_bytes_reads_the_nearest_existing_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertGreater(rc.free_bytes(Path(tmp) / "not" / "yet"), 0)


class PruneTest(unittest.TestCase):

    def layout(self, work, commit="a" * 40):
        d = work / commit[:12]
        keep = [put(d / "legs/cuda-sm_80.provenance.json"), put(d / "legs/cuda-sm_80.log"),
                put(d / "legs/cuda-sm_80.exit"),
                put(d / "legs/cuda-sm_80/cuda-sm_80/release-build/build/build-provenance.json"),
                put(d / "legs/cuda-sm_80/cuda-sm_80/release-build/build/sets/cuda/sm_80/PTX_BASELINE.json"),
                put(d / "legs/cuda-sm_80/cuda-sm_80/release-build/build/sets/cuda/sm_80/manifest.json"),
                put(d / "host-composition/host-composition.json"), put(d / "reuse/plan.json"),
                put(d / "linux/final/candidate.json"), put(d / "linux/final/mojolearn_core-1.whl"),
                put(d / "linux/final/split-audit.json"), put(d / "linux/SIZES-1-linux.json"),
                put(d / "linux/audit/report.json"), put(d / "linux-pack.log"), put(work / "state.json"),
                put(d / "reuse/ptx-published/release-build/build/PTX_PROOF.json")]
        drop = [put(d / "legs/cuda-sm_80/cuda-sm_80/release-build/build/sets/cuda.tar.gz", 1000),
                put(d / "legs/cuda-sm_80/cuda-sm_80/release-build/build/sets/cuda/sm_80/identical/_m.so", 500),
                put(d / "legs/cuda-sm_89/cuda-sm_89/release-build/build/cuda-tools-12.5.82/bin/ptxas", 300),
                put(d / "host-composition/sets/cuda/sm_89/host/_h.so", 200),
                put(d / "reuse/sets/cuda/sm_80/identical/_m.so", 200),
                put(d / "linux/audit/repaired/mojolearn_core-1.whl", 100),
                put(d / "linux/mojolearn_core-1.whl", 100),
                put(d / "reuse/previous-split/mojolearn_core-0.whl", 400),
                put(d / "reuse/ptx-published/release-build/build/sets/cuda/sm_80/identical/_p.so", 140)]
        return d, keep, drop

    def test_prune_plan_lists_payload_and_never_a_record_or_a_final_wheel(self):
        with tempfile.TemporaryDirectory() as tmp:
            d, keep, drop = self.layout(Path(tmp))
            planned = {p for p, _ in rc.prune_plan(d)}
            self.assertEqual(planned, set(drop))
            self.assertFalse(planned & set(keep))

    def test_prune_set_deletes_the_plan_and_the_source_worktree_and_keeps_the_rest(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            d, keep, drop = self.layout(work)
            src = put(work / "source" / ("a" * 12) / "core" / "x.mojo", 50).parents[1]
            lines = []
            with mock.patch.object(rc, "sh", return_value=""):
                freed = rc.prune_set(work, "a" * 40, say=lines.append)
            self.assertEqual(freed, 1000 + 500 + 300 + 200 + 200 + 100 + 100 + 400 + 140 + 50)
            self.assertTrue(all(p.is_file() for p in keep))
            self.assertFalse(any(p.exists() for p in drop))
            self.assertFalse(src.exists())
            self.assertFalse((d / "host-composition/sets").exists())   # emptied trees go too
            self.assertIn("--keep-intermediates", lines[0])

    def test_keep_intermediates_is_a_flag(self):
        a = rc.parse(["--commit", "a" * 40, "--keep-intermediates"])
        self.assertTrue(a.keep_intermediates)
        self.assertFalse(rc.parse(["--commit", "a" * 40]).keep_intermediates)


class LsRemoteTest(unittest.TestCase):

    def test_a_timeout_is_retried_with_backoff_then_succeeds(self):
        ok = subprocess.CompletedProcess([], 0, stdout="abc\trefs/tags/v1\n", stderr="")
        calls, pauses = [], []

        def run(*a, **k):
            calls.append(k.get("timeout"))
            if len(calls) < 3:
                raise subprocess.TimeoutExpired("git", 120)
            return ok
        with mock.patch.object(rc.subprocess, "run", side_effect=run):
            out = rc.ls_remote("--tags", "origin", sleep=pauses.append)
        self.assertEqual(out, "abc\trefs/tags/v1")
        self.assertEqual(len(calls), 3)
        self.assertEqual(pauses, [10, 30])

    def test_three_failures_raise(self):
        bad = subprocess.CompletedProcess([], 128, stdout="", stderr="fatal: unable to access")
        with mock.patch.object(rc.subprocess, "run", return_value=bad):
            with self.assertRaises(SystemExit) as caught:
                rc.ls_remote("--tags", "origin", sleep=lambda s: None)
        self.assertIn("3 times", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
