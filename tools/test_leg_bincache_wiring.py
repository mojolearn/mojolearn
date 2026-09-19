#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The GPU leg bodies really do build through the R2 binding cache.

tools/test_bincache.py tests tools/bincache.py. This tests THE WIRING: it lifts
the `BINCACHE=` selection block and the `build()` / `build_host()` definitions
VERBATIM out of each leg body and runs them against a fake repository whose
"compiler" is a shell script that appends to a marker file. So "resolved from
the cache" is not read off a log line -- it is the marker NOT growing.

This exists because the wiring was absent for four days and nothing said so.
tools/gemm_remote_leg.sh staged a URL map before the payload and promoted an
empty inbox after it, while every body called `sh bindings/build_X.sh` directly
and compiled all 23 families on every rental (1010 s on 2026-09-14, 1204 s on
2026-09-19, three times in one day at the same commit). A staged cache that no
body reads looks exactly like a working one from the runner's side.

file:// URLs stand in for R2's presigned ones -- bincache.py's own transport
supports them and tools/test_bincache.py uses the same substitution -- and the
map is placed through MOJOLEARN_BINCACHE_MAP rather than at the box's
/root/.mojolearn_bincache/urls.tsv. Everything else is the shipped code: the
key, the archive, the manifest check, the placement and every refusal.

Stdlib only; no network, no Mojo, no GPU, no rental.

    python3 -m unittest -v test_leg_bincache_wiring
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import bincache as bc  # noqa: E402

# Every leg body that must build through the cache. Add a row when a new GPU
# leg body is written; a body missing from here is a body that recompiles.
LEGS = ("gpu_class_gaps_nvidia_leg.sh", "gpu_class_gaps_amd_leg.sh",
        "two_device_par_class_amd_leg.sh", "gap_column_leg.sh")

PARTITION = "sm_90a/runpod-img"

# The fake binding build. `if false; then ... fi` keeps a literal `mojo build`
# line for script_plan() to read the entry file out of, exactly as a real
# bindings/build_*.sh has one. COMPILER_RAN is the witness.
FAKE = '''#!/bin/sh
set -eu
here=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$here"
if false; then pixi run mojo build -j 2 --emit shared-lib -I . -I bindings bindings/_mojolearn_fake.mojo -o x; fi
echo compiled >> "$COMPILER_RAN"
mkdir -p python/mojolearn/identical
printf 'binding %s %s\\n' "${MOJOLEARN_GPU_ARCHS:-}" "${MOJOLEARN_EXTRA_DEFINES:-}" > python/mojolearn/identical/_mojolearn_fake.so
echo "built python/mojolearn/identical/_mojolearn_fake.so"
'''


def region(text, start, end):
    lines = text.splitlines()
    i = next(n for n, l in enumerate(lines) if re.match(start, l))
    j = next(n for n, l in enumerate(lines[i:], i) if re.match(end, l))
    return "\n".join(lines[i:j + 1]) + "\n"


class LegBincacheWiring(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="leg-bincache-")
        self.addCleanup(tmp.cleanup)
        self.w = Path(tmp.name)
        self.repo = self.w / "mojolearn"
        (self.repo / "bindings").mkdir(parents=True)
        (self.repo / "core").mkdir()
        (self.repo / "tools").mkdir()
        shutil.copy(TOOLS / "bincache.py", self.repo / "tools" / "bincache.py")
        (self.repo / "bindings" / "build_fake.sh").write_text(FAKE)
        (self.repo / "bindings" / "_mojolearn_fake.mojo").write_text(
            "from std.os import getenv\nfrom core import a\n")
        (self.repo / "core" / "a.mojo").write_text("fn f(): pass\n")
        (self.repo / "pixi.toml").write_text("[workspace]\n")
        (self.repo / "pixi.lock").write_text(
            "      - conda: https://conda.modular.com/max/%s/mojo-compiler-1.0.0-release.conda\n"
            % bc.toolchain(Path("/nonexistent"))["platform"])
        self.store = self.w / "r2"
        for pre in (bc.OBJECT_PREFIX, bc.SABOTAGE_PREFIX):
            (self.store / pre / PARTITION).mkdir(parents=True)
        (self.store / bc.INBOX_PREFIX / "LEG").mkdir(parents=True)
        self.mapdir = self.w / "mapdir"
        self.mapdir.mkdir()
        self.out = self.w / "bcout"
        self.marker = self.w / "compiler_ran"
        self.so = self.repo / "python" / "mojolearn" / "identical" / "_mojolearn_fake.so"
        # A `timeout` for the build() wrapper: macOS has none and the bodies
        # bound every phase with one. A shim, not a change to the body.
        self.bin = self.w / "bin"
        self.bin.mkdir()
        (self.bin / "timeout").write_text('#!/bin/sh\nshift\nexec "$@"\n')
        os.chmod(self.bin / "timeout", 0o755)

    # -------------------------------------------------------------- harness
    def write_map(self, gets=(), sgets=()):
        lines = ["#partition\t" + PARTITION, "#image\trunpod:img", "#leg\tLEG"]
        for verb, keys, pre in (("get", gets, bc.OBJECT_PREFIX),
                                ("sget", sgets, bc.SABOTAGE_PREFIX)):
            for k in keys:
                lines.append("%s\t%s\tfile://%s/%s/%s/%s.tar.gz"
                             % (verb, k, self.store, pre, PARTITION, k))
        for i in range(4):
            lines.append("put\t%03d\tfile://%s/%s/LEG/%03d.tar.gz"
                         % (i, self.store, bc.INBOX_PREFIX, i))
        p = self.mapdir / "urls.tsv"
        p.write_text("\n".join(lines) + "\n")
        shutil.rmtree(self.mapdir / "claimed", ignore_errors=True)
        return p

    def body(self, leg, fn="build"):
        """The leg body's own lines: how it picks the build command, and the
        function that runs it."""
        text = (TOOLS / leg).read_text()
        select = region(text, r'^BINCACHE="sh"$', r'^say "bincache=')
        build = region(text, r'^%s\(\) \{$' % fn, r'^\}$')
        self.assertIn("$BINCACHE", build, "%s: %s() does not use $BINCACHE" % (leg, fn))
        return ("cd '%s'\n" % self.repo
                + 'JOBS=8\ncap() { echo "$1"; }\nsay() { echo "GATE: $*"; }\n'
                  'run() { _n=$1; shift; "$@"; }\n'
                + select + build + "%s build_fake\n" % fn)

    def run_build(self, leg, env_extra=(), map_path=True, fn="build"):
        env = dict(os.environ)
        env.update(PATH="%s:%s" % (self.bin, env["PATH"]),
                   COMPILER_RAN=str(self.marker),
                   MOJOLEARN_BINCACHE_OUT=str(self.out),
                   MOJOLEARN_BINCACHE_MAP=str(self.mapdir / "urls.tsv"
                                              if map_path else self.w / "none.tsv"),
                   MOJOLEARN_GPU_ARCHS="sm_90a", MOJOLEARN_NUMERIC_MODE="identical")
        env.update(env_extra)
        before = self.marker.read_text() if self.marker.exists() else ""
        r = subprocess.run(["/bin/sh", "-c", self.body(leg, fn)], env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        after = self.marker.read_text() if self.marker.exists() else ""
        return r.returncode, r.stdout.decode(), after != before

    def last_upload(self):
        rows = (self.out / "uploads.tsv").read_text().strip().splitlines()
        return bc.check_upload_row(rows[-1], self.out / "keys")

    def promote(self):
        """What the Mac does after the leg: re-derive the key from the fields
        the box recorded, then copy the inbox object to its content address."""
        leg, slot, dest, key = self.last_upload()
        shutil.copy(self.store / bc.INBOX_PREFIX / leg / (slot + ".tar.gz"),
                    self.store / dest)
        return key, dest

    # ---------------------------------------------------------------- tests
    def test_a_cold_build_is_offered_and_a_second_box_takes_it(self):
        for leg in LEGS:
            with self.subTest(leg=leg):
                self.write_map()
                rc, log, compiled = self.run_build(leg)
                self.assertEqual(rc, 0, log)
                self.assertTrue(compiled, "a cold cache must compile:\n" + log)
                self.assertIn("miss+built-uploaded", log)
                built = self.so.read_bytes()
                key, dest = self.promote()
                self.assertTrue(dest.startswith(bc.OBJECT_PREFIX + "/"), dest)

                # A SECOND BOX: same key, nothing built yet.
                self.so.unlink()
                self.write_map(gets=[key])
                rc, log, compiled = self.run_build(leg)
                self.assertEqual(rc, 0, log)
                self.assertIn("BINCACHE hit", log)
                self.assertFalse(compiled, "a hit must not run the compiler:\n" + log)
                self.assertEqual(self.so.read_bytes(), built,
                                 "the placed binding differs from the built one")
                self.so.unlink()

    def test_a_changed_source_or_arch_misses_and_builds(self):
        self.write_map()
        self.assertEqual(self.run_build(LEGS[0])[0], 0)
        key, _ = self.promote()
        self.write_map(gets=[key])
        for label, mutate, env in (
                ("a source file in the closure",
                 lambda: (self.repo / "core" / "a.mojo").write_text("fn f(): pass\nfn h(): pass\n"), {}),
                ("the device architecture", lambda: None, {"MOJOLEARN_GPU_ARCHS": "sm_80"}),
                ("a build define", lambda: None,
                 {"MOJOLEARN_EXTRA_DEFINES": "-D MOJOLEARN_TRIAL=1"})):
            with self.subTest(changed=label):
                if self.so.exists():
                    self.so.unlink()
                mutate()
                rc, log, compiled = self.run_build(LEGS[0], env)
                self.assertEqual(rc, 0, log)
                self.assertTrue(compiled, "a moved key must compile:\n" + log)
                self.assertNotIn("BINCACHE hit", log)
                (self.repo / "core" / "a.mojo").write_text("fn f(): pass\n")

    def test_the_body_pins_the_numeric_mode_so_the_key_cannot_drift(self):
        # build() passes MOJOLEARN_NUMERIC_MODE=identical itself, so a stray
        # value in the leg's environment cannot change either the binary or
        # the key. This is why the caller's mode is absent from the list above.
        self.write_map()
        self.assertEqual(self.run_build(LEGS[0])[0], 0)
        key, _ = self.promote()
        self.so.unlink()
        self.write_map(gets=[key])
        rc, log, compiled = self.run_build(LEGS[0], {"MOJOLEARN_NUMERIC_MODE": "fast"})
        self.assertIn("BINCACHE hit", log)
        self.assertFalse(compiled, log)

    def test_a_miss_falls_through_and_a_failed_build_keeps_its_code(self):
        for label, map_path, env in (("no URL map at all", False, {}),
                                     ("MOJOLEARN_BINCACHE=0", True, {"MOJOLEARN_BINCACHE": "0"})):
            with self.subTest(case=label):
                self.write_map()
                if self.so.exists():
                    self.so.unlink()
                rc, log, compiled = self.run_build(LEGS[0], env, map_path=map_path)
                self.assertEqual(rc, 0, log)
                self.assertTrue(compiled, log)
                self.assertNotIn("BINCACHE", log, "a pass-through must not record:\n" + log)
                self.assertTrue(self.so.is_file())
        (self.repo / "bindings" / "build_fake.sh").write_text(
            FAKE.replace('echo "built', 'exit 3\necho "built'))
        self.so.unlink()
        rc, log, compiled = self.run_build(LEGS[0])
        self.assertEqual(rc, 3, "the build script's exit code must survive:\n" + log)
        self.assertIn("build-failed", log)

    def test_a_sabotage_build_is_never_served_as_a_production_binding(self):
        sab = {"MOJOLEARN_EXTRA_DEFINES": "-D MOJOLEARN_HOST_SABOTAGE=1"}
        self.write_map()
        rc, log, compiled = self.run_build(LEGS[0])
        self.assertEqual(rc, 0, log)
        prod_key, _ = self.promote()
        self.so.unlink()

        # 1. With a warm production cache, a sabotage build is refused outright:
        #    it neither reads nor writes the cache, and it compiles.
        self.write_map(gets=[prod_key])
        rc, log, compiled = self.run_build(LEGS[0], sab)
        self.assertEqual(rc, 0, log)
        self.assertIn("refused:sabotage:", log)
        self.assertNotIn("BINCACHE hit", log)
        self.assertTrue(compiled, log)
        self.so.unlink()

        # 2. With the explicit negative opt-in it uploads, but ONLY to the
        #    sabotage prefix, under a key that carries variant=sabotage.
        rc, log, compiled = self.run_build(
            LEGS[0], dict(sab, MOJOLEARN_BINCACHE_NEGATIVE="1"))
        self.assertEqual(rc, 0, log)
        self.assertIn("negative-miss+built-uploaded", log)
        leg_id, slot, dest, sab_key = self.last_upload()
        self.assertTrue(dest.startswith(bc.SABOTAGE_PREFIX + "/"), dest)
        self.assertNotEqual(sab_key, prod_key)
        self.assertEqual(
            json.loads((self.out / "keys" / (sab_key + ".json")).read_text()).get("variant"),
            "sabotage")

        # 3. THE ATTACK: that sabotage archive parked at the PRODUCTION content
        #    address, listed as a production `get`. The manifest is bound to
        #    the sabotage key, so the production build rejects it and compiles.
        shutil.copy(self.store / bc.INBOX_PREFIX / leg_id / (slot + ".tar.gz"),
                    self.store / bc.OBJECT_PREFIX / PARTITION / (prod_key + ".tar.gz"))
        self.so.unlink()
        rc, log, compiled = self.run_build(LEGS[0])
        self.assertEqual(rc, 0, log)
        self.assertIn("rejected:", log)
        self.assertNotIn("BINCACHE hit", log)
        self.assertTrue(compiled, "a rejected archive must fall through to a build:\n" + log)
        self.assertNotIn("SABOTAGE", self.so.read_text())

    def test_the_host_family_builds_go_through_the_cache_too(self):
        # gap_column_leg.sh builds the CPU host families with its own wrapper.
        self.write_map()
        rc, log, compiled = self.run_build("gap_column_leg.sh", fn="build_host")
        self.assertEqual(rc, 0, log)
        self.assertIn("miss+built-uploaded", log)
        key, _ = self.promote()
        self.so.unlink()
        self.write_map(gets=[key])
        rc, log, compiled = self.run_build("gap_column_leg.sh", fn="build_host")
        self.assertIn("BINCACHE hit", log)
        self.assertFalse(compiled, log)

    def test_no_body_still_calls_sh_on_a_binding_script(self):
        for leg in LEGS:
            with self.subTest(leg=leg):
                for n, line in enumerate((TOOLS / leg).read_text().splitlines(), 1):
                    if line.lstrip().startswith("#"):
                        continue
                    self.assertNotRegex(
                        line, r'(?<!\$BINCACHE )\bsh ["\']?bindings/',
                        "%s:%d compiles directly instead of through the cache" % (leg, n))


if __name__ == "__main__":
    unittest.main()
