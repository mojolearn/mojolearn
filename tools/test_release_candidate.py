# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn
"""tools/release.py candidate (tools/release_candidate.py) and the pieces it adds to the release's own code: the local
version, the define spelling, the define-aware reuse plan, the packer's candidate _version.py, and the step selection
(the five build/pack steps, never a freeze step). No network, no git remote, no build."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS.parent / "packaging" / "linux"))
import release  # noqa: E402
import release_candidate as RC  # noqa: E402
import release_github_build as G  # noqa: E402
import release_reuse  # noqa: E402

SHA = "0123456789abcdef0123456789abcdef01234567"


class Version(unittest.TestCase):
    def test_local_version(self):
        self.assertEqual(RC.candidate_version("0.8.67", SHA, ""), "0.8.67.dev0+0123456789ab")
        self.assertEqual(RC.candidate_version("0.8.67", SHA, "d1a2b3c4d"), "0.8.67.dev0+0123456789ab.d1a2b3c4d")
        self.assertEqual(RC.next_version("0.8.66"), "0.8.67")
        with self.assertRaises(SystemExit):
            RC.next_version("0.8.66rc1")

    def test_label_from_defines(self):
        self.assertEqual(RC.default_label(""), "")
        a = RC.default_label("MOJOLEARN_EM_LOOP_CHECK_EVERY=1")
        self.assertRegex(a, r"^d[0-9a-f]{8}$")
        self.assertNotEqual(a, RC.default_label("MOJOLEARN_SGD_LOOP_CHECK_EVERY=1"))

    def test_packer_accepts_only_local_versions(self):
        import pack_wheel as P
        self.assertTrue(P.LOCAL_VERSION_RE.fullmatch("0.8.67.dev0+0123456789ab"))
        self.assertTrue(P.LOCAL_VERSION_RE.fullmatch("0.8.67.dev0+0123456789ab.em1"))
        for bad in ("0.8.67", "0.8.67.dev0", "0.8.67.dev0+0123456789ab.em-1", "0.8.67.dev0+0123"):
            self.assertIsNone(P.LOCAL_VERSION_RE.fullmatch(bad), bad)
        text = '# header\n__version__ = "0.8.66"\n'
        self.assertEqual(P.candidate_version_py(text, "0.8.66", "0.8.67.dev0+0123456789ab"),
                         b'# header\n__version__ = "0.8.67.dev0+0123456789ab"\n')
        with self.assertRaises(SystemExit):
            P.candidate_version_py(text, "0.8.65", "0.8.67.dev0+0123456789ab")


class Defines(unittest.TestCase):
    def test_spellings(self):
        self.assertEqual(G.normalize_defines(""), "")
        self.assertEqual(G.normalize_defines("-D B=2 -D A=1"), "A=1,B=2")
        self.assertEqual(G.normalize_defines("B=2,A"), "A,B=2")
        for bad in ("-D", "A=1 B=2", "-DA=1", "A=1,A=2", "1A=2"):
            with self.assertRaises(SystemExit, msg=bad):
                G.normalize_defines(bad)

    def test_plan_builds_only_the_readers(self):
        b1 = SimpleNamespace(script="build_mixture.sh")
        b2 = SimpleNamespace(script="build_ols.sh")
        rows = [dict(target=release_reuse.LINUX, decision="REUSE", reason="unchanged", reuse_from={}),
                dict(target=release_reuse.LINUX, decision="REUSE", reason="unchanged", reuse_from={})]
        facts = dict(closures={"build_mixture.sh": dict(sources=["mixture/em.mojo"]),
                               "build_ols.sh": dict(sources=["linear/ols.mojo"])})
        moved = release_reuse.apply_defines(rows, [(b1, None), (b2, None)], facts, {"mixture/em.mojo"},
                                            ["MOJOLEARN_EM_LOOP_CHECK_EVERY"])
        self.assertEqual(moved, [rows[0]])
        self.assertEqual(rows[0]["decision"], "BUILD")
        self.assertIsNone(rows[0]["reuse_from"])
        self.assertEqual(rows[1]["decision"], "REUSE")


class Steps(unittest.TestCase):
    def test_candidate_steps_are_release_steps_and_never_freeze(self):
        names = [s for s, _, _, _ in release.STEP_TABLE]
        self.assertEqual([s for s in names if s in RC.STEPS], list(RC.STEPS))
        never = {"freeze-version", "freeze-changelog", "freeze-docs-facts", "freeze-commit", "identity-at-freeze",
                 "rehearsal", "readiness", "cross-compile"} | set(release.PUBLISH_STEPS)
        self.assertFalse(never & set(RC.STEPS))
        # every need of a selected step is selected or skipped, never one that a candidate must run first
        needs = {s: n for s, _, n, _ in release.STEP_TABLE}
        for step in RC.STEPS:
            for need in needs[step]:
                self.assertTrue(need in RC.STEPS or need in never, (step, need))

    def test_parser_takes_the_candidate_argv(self):
        args = release.release_parser().parse_args(
            ["0.8.67.dev0+0123456789ab", "--only", ",".join(RC.STEPS), "--state-dir", "/x", "--source-checkout", "/y",
             "--build-backend", "github", "--identity-on-wheels"])
        self.assertEqual(args.only.split(","), list(RC.STEPS))
        self.assertTrue(args.identity_on_wheels)
        self.assertFalse(args.reuse_unchanged)       # ledger_on() is False: no ledger read or write

    def test_github_leg_carries_defines_and_ref(self):
        ctx = SimpleNamespace(rel=Path("/w/c"), commit=SHA, args=SimpleNamespace(
            build_backend="github", candidate_defines="A=1", candidate_ref="integ/x"))
        legs = release.github_legs(ctx)
        self.assertTrue(legs)
        for leg in legs:
            self.assertIn("--defines", leg.command)
            self.assertEqual(leg.command[leg.command.index("--defines") + 1], "A=1")
            self.assertEqual(leg.command[leg.command.index("--ref") + 1], "integ/x")
        plain = release.github_legs(SimpleNamespace(rel=Path("/w/c"), commit=SHA, args=SimpleNamespace()))
        self.assertTrue(all("--defines" not in l.command and "--ref" not in l.command for l in plain))


class OursOnlyVenv(unittest.TestCase):
    def test_pins_come_from_the_lock(self):
        import bench_board
        reqs = bench_board.ours_only_requirements(bench_board.opponent_pins())
        self.assertTrue(reqs[0].startswith("scikit-learn=="), reqs)
        for name in bench_board.OURS_ONLY_LOCKED:
            pin = [r for r in reqs if r.startswith(name + "==")]
            self.assertEqual(len(pin), 1, reqs)
            self.assertEqual(pin[0], "%s==%s" % (name, bench_board.locked_version(name)))


if __name__ == "__main__":
    unittest.main()
