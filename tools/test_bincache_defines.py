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


REPO = Path(__file__).resolve().parents[1]
OS = dict(id="ubuntu", version_id="22.04", glibc="glibc 2.35", machine="x86_64", ld="GNU ld 2.38", cc="cc 11.4")
HOST_ARGS = ["host", "--group", "_mojolearn_native_x_host", "--rootdir", "/tmp/mojolearn-merged-roots/host"]
SVD = "MOJOLEARN_XD_FAST_SVD_ONE_BLOCK"
#: What the release legs set around the host compile (legs/*/GHA/bincache/keys/*.json, candidate
#: 4c2340386abc.svd1); the leg-level packaging switches differ by leg and must not reach the key.
LEG_ENV = {"cuda-sm_89": {"MOJOLEARN_CUDA_CODE_FORMAT": "native"},
           "cuda-sm_80": {"MOJOLEARN_CUDA_CODE_FORMAT": "ptx"},
           "hip-gfx942": {"MOJOLEARN_BUILD_TIERS": "identical"}}


class HostUnitReadsADefine(unittest.TestCase):
    """The svd1 B set (2026-11-13): the merged CPU host group _mojolearn_native_x_host compiles
    x_decomp, whose closure names MOJOLEARN_XD_FAST_SVD_ONE_BLOCK. Its key must keep the define and
    be ONE key on every leg (the host unit does not depend on the leg)."""

    def env(self, leg, defines=SVD):
        env = {"MOJOLEARN_NUMERIC_MODE": "identical", "MOJOLEARN_TARGET_COLUMN": "cpu",
               "MOJOLEARN_PACKAGE_BYTE_LM": "1", "MOJOLEARN_BUILD_PIXI_ENV": "default",
               "MOJOLEARN_COMPILE_JOBS": "2", "MOJOLEARN_LINUX_CPU": "x86-64-v3", "PATH": "/bin"}
        env.update(LEG_ENV[leg])
        if defines:
            env["MOJOLEARN_BUILD_DEFINES"] = defines
        return env

    def key(self, env):
        with mock.patch.object(bincache, "build_platform", return_value="linux"):
            scoped, _ = bincache.scope_build_defines(REPO, REPO / "bindings/merged_build.sh", HOST_ARGS, env)
            fields = bincache.key_fields(REPO, REPO / "bindings/merged_build.sh", HOST_ARGS, scoped, "img",
                                         dev_arch="none", os_info=dict(OS), plat="linux")
        return fields, bincache.key_of(fields)

    def test_the_host_closure_names_the_define_and_the_key_keeps_it(self):
        with mock.patch.object(bincache, "build_platform", return_value="linux"):
            sc = bincache.source_closure(REPO, REPO / "bindings/merged_build.sh", HOST_ARGS, self.env("cuda-sm_89"),
                                         plat="linux")
            env, kept = bincache.scope_build_defines(REPO, REPO / "bindings/merged_build.sh", HOST_ARGS,
                                                     self.env("cuda-sm_89", SVD + ",MOJOLEARN_ZZ_NOT_READ_ANYWHERE"))
        readers = [rel for rel in sc["closure"] if (REPO / rel).is_file() and rel.endswith(".mojo")
                   and bincache.define_named(SVD, (REPO / rel).read_text(errors="replace"))]
        self.assertTrue(readers, "no file of the x_host closure names %s" % SVD)
        self.assertEqual(kept, [SVD])
        self.assertEqual(env["MOJOLEARN_BUILD_DEFINES"], SVD)
        fields, _ = self.key(self.env("cuda-sm_89"))
        self.assertEqual(fields["build_env"].get("MOJOLEARN_BUILD_DEFINES"), SVD)

    def test_one_host_key_on_every_leg_and_a_new_key_for_the_define(self):
        keys = {leg: self.key(self.env(leg))[1] for leg in LEG_ENV}
        self.assertEqual(len(set(keys.values())), 1, keys)
        plain = {leg: self.key(self.env(leg, defines=""))[1] for leg in LEG_ENV}
        self.assertEqual(len(set(plain.values())), 1, plain)
        self.assertNotEqual(set(keys.values()), set(plain.values()))


class LinuxCpuFeaturesArePinned(unittest.TestCase):
    """Root cause of the leg-dependent host bytes (2026-11-13): `mojo build` defaults --target-features
    to the BUILD MACHINE's features even with --target-cpu pinned, so one key compiled to one byte
    variant per runner CPU. Every Linux x86-64 --target-cpu pin carries the same explicit feature list."""

    def test_every_pin_names_the_features(self):
        import release_linux_group_composition as rlgc
        want = "--target-features ${MOJOLEARN_LINUX_CPU_FEATURES:-%s}" % rlgc.LINUX_CPU_FEATURES
        files = sorted((REPO / "bindings").glob("*.sh")) + [REPO / "packaging/adapters/build.sh"]
        pins = 0
        for f in files:
            for n, line in enumerate(f.read_text().splitlines(), 1):
                if line.lstrip().startswith("#") or "--target-cpu" not in line or "MOJOLEARN_LINUX_CPU" not in line:
                    continue
                pins += 1
                with self.subTest(f"{f.relative_to(REPO)}:{n}"):
                    self.assertIn(want, line.replace('"', ""))
        self.assertGreater(pins, 30)


if __name__ == "__main__":
    unittest.main()
