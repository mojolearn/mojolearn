# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn
"""bincache.scope_build_defines: a candidate B set's MOJOLEARN_BUILD_DEFINES reaches only the builds whose closure
names the define (tools/release.py candidate --defines)."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bincache  # noqa: E402


class ScopeDefines(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        (self.repo / "bindings").mkdir()
        (self.repo / "em").mkdir()
        (self.repo / "em" / "loop.mojo").write_text('comptime E = env_get_int["MOJOLEARN_EM_LOOP_CHECK_EVERY", 4]()\n')
        (self.repo / "em" / "other.mojo").write_text("fn f(): pass  # MOJOLEARN_EM_LOOP_CHECK_EVERYX\n")
        (self.repo / "bindings" / "build_defines.sh").write_text("# MOJOLEARN_SGD_LOOP_CHECK_EVERY spelled here\n")
        (self.repo / "bindings" / "build_em.sh").write_text("#!/bin/sh\n")

    def tearDown(self):
        self.tmp.cleanup()

    def scope(self, closure, defines):
        sc = dict(closure={c: "0" * 64 for c in closure},
                  inputs=dict(files=["bindings/build_defines.sh", "bindings/build_em.sh"]))
        env = {"MOJOLEARN_BUILD_DEFINES": defines, "PATH": "/bin"}
        with mock.patch.object(bincache, "source_closure", return_value=sc):
            return bincache.scope_build_defines(self.repo, self.repo / "bindings" / "build_em.sh", [], env)

    def test_reader_keeps_only_its_define(self):
        env, kept = self.scope(["em/loop.mojo"], "MOJOLEARN_EM_LOOP_CHECK_EVERY=1,MOJOLEARN_SGD_LOOP_CHECK_EVERY=1")
        self.assertEqual(kept, ["MOJOLEARN_EM_LOOP_CHECK_EVERY=1"])
        self.assertEqual(env["MOJOLEARN_BUILD_DEFINES"], "MOJOLEARN_EM_LOOP_CHECK_EVERY=1")

    def test_non_reader_builds_without_the_variable(self):
        # a longer identifier and build_defines.sh's own text are not reads
        env, kept = self.scope(["em/other.mojo"], "MOJOLEARN_EM_LOOP_CHECK_EVERY=1,MOJOLEARN_SGD_LOOP_CHECK_EVERY=1")
        self.assertEqual(kept, [])
        self.assertNotIn("MOJOLEARN_BUILD_DEFINES", env)
        self.assertEqual(env["PATH"], "/bin")

    def test_unreadable_closure_keeps_every_define(self):
        env = {"MOJOLEARN_BUILD_DEFINES": "A=1,B"}
        with mock.patch.object(bincache, "source_closure", side_effect=bincache.UnkeyableBuild("x")):
            out, kept = bincache.scope_build_defines(self.repo, "bindings/build_em.sh", [], env)
        self.assertEqual(out, env)
        self.assertEqual(kept, ["A=1", "B"])

    def test_unset_is_untouched(self):
        env = {"PATH": "/bin"}
        self.assertEqual(bincache.scope_build_defines(self.repo, "x", [], env), (env, None))


if __name__ == "__main__":
    unittest.main()
