"""A failed arm build's ARM line carries its first error (lane fb9-arm-builds, 2026-11-17).

CPU only: release_candidate.first_build_error on synthetic arm and leg logs.

    python3 -m unittest tools/tests/test_release_candidate_first_error.py
"""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import release_candidate as rc  # noqa: E402


class FirstBuildError(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.legs = self.root / "set" / "legs"
        self.arm = self.root / "arm.log"
        self.arm.write_text(f"[core-linux]   cuda-sm_89: launched pid 1, log {self.legs}/cuda-sm_89.log\n"
                            "   core-linux: FAILED at linux-assemble: cuda-sm_89 did not build\n"
                            "ARM x: units BUILD 1 / REUSE 2 on legs cuda-sm_89\n")

    def binding_log(self, rel, lines, ok=False):
        p = self.legs / "cuda-sm_89" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("start\n" + "\n".join(lines) + ("\nend OK\n" if ok else "\nend FAILED\n"))

    def test_the_compiler_diagnostic_comes_first(self):
        self.binding_log("GHA/shards/0/build_logs/identical_merged__mojolearn_native.log",
                         ["warning: something", "Error: wrapper gave up",
                          "/root/mojolearn/checks/nv_fma_rn.mojo:41:9: error: use of unknown declaration 'x'"])
        self.binding_log("GHA/shards/1/build_logs/identical_merged__mojolearn_native_trees.log",
                         ["/root/mojolearn/a.mojo:1:1: error: not this one"], ok=True)
        self.assertEqual(rc.first_build_error(self.arm),
                         "checks/nv_fma_rn.mojo:41:9: error: use of unknown declaration 'x'")

    def test_without_a_diagnostic_the_first_failure_line(self):
        self.binding_log("x/release-build/build/build_logs/identical_merged__mojolearn_native.log",
                         ["GeneratedSourceError: generated source: preparation failed for core/t.mojo; see "
                          "/root/mojolearn/build/g.json"])
        self.assertIn("identical_merged__mojolearn_native: Error: generated source: preparation failed for core/t.mojo",
                      rc.first_build_error(self.arm))

    def test_linux_wait_reads_a_leg_directly(self):
        # tools/release.py linux-wait prints this as soon as the leg fails (lane fx-build-tools, 2026-11-18)
        self.binding_log("GHA/shards/0/build_logs/host_merged__mojolearn_native_host.log",
                         ["/root/mojolearn/core/h.mojo:7:3: error: bad"])
        self.assertEqual(rc.legs_first_error([self.legs / "cuda-sm_89", self.root / "missing"]),
                         "core/h.mojo:7:3: error: bad")
        self.assertIsNone(rc.legs_first_error([self.root / "missing"]))

    def test_without_leg_logs_the_arms_failed_line(self):
        self.assertEqual(rc.first_build_error(self.arm),
                         "core-linux: FAILED at linux-assemble: cuda-sm_89 did not build")


if __name__ == "__main__":
    unittest.main()
