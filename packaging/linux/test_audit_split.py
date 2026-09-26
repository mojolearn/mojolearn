#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""packaging/linux/audit.sh ON EACH SPLIT WHEEL, with docker stood in for.

    .pixi/envs/test/bin/python -m pytest -q packaging/linux/test_audit_split.py

The split set is packed by the real `pack_wheel.main` from the fake sets
test_split_wheels.py builds (inert bytes, so no real auditwheel could read
them). `docker` on PATH is a shim that records every call and plays
auditwheel: `show` prints a manylinux tag, `repair` copies the wheel into the
mounted output directory (or, when told to, grafts a `<dist>.libs/` directory
the way auditwheel does for a library it was not told to exclude), and twine
says PASSED. What is checked is audit.sh's own part: which libraries it
excludes for each wheel, where each wheel's logs go, and that a grafted
top-level `.libs/` fails the run.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_split_wheels as tsw  # noqa: E402

AUDIT = HERE / "audit.sh"

DOCKER_SHIM = r'''#!/usr/bin/env python3
import json, os, pathlib, shutil, sys, zipfile
args = sys.argv[1:]
log = pathlib.Path(os.environ["FAKE_DOCKER_LOG"])
with log.open("a") as fh:
    fh.write(json.dumps(args) + "\n")
mounts = {}
for i, a in enumerate(args):
    if a == "-v":
        host, container = args[i + 1].split(":")[:2]
        mounts[container] = host
cmd = args[-1]
def host(path):
    for container, h in sorted(mounts.items(), key=lambda kv: -len(kv[0])):
        if path == container or path.startswith(container + "/"):
            return h + path[len(container):]
    raise SystemExit("unmounted " + path)
if cmd.startswith("auditwheel show"):
    print("is consistent with the following platform tag: \"manylinux_2_35_x86_64\"")
elif cmd.startswith("auditwheel repair"):
    words = cmd.split()
    src = pathlib.Path(host(words[-1]))
    dest = pathlib.Path(host(words[words.index("-w") + 1])) / src.name
    shutil.copyfile(src, dest)
    if os.environ.get("FAKE_GRAFT"):
        with zipfile.ZipFile(dest, "a") as z:
            z.writestr(src.name.split("-")[0] + ".libs/libgrafted-1234.so", b"grafted")
    print("Fixed-up wheel written to", dest)
elif "twine check" in cmd:
    print("Checking wheels: PASSED")
'''


class AuditSplit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.set_dirs = tsw.make_sets(cls.root)
        args = [a for s in cls.set_dirs for a in ("--set", s)]
        cls.packed = cls.root / "packed"
        assert tsw.pw.main(args + ["--out", str(cls.packed)], _gates=False) == 0
        cls.manifests = [str(p) for p in sorted((cls.root / "sets").glob("*/*/manifest.json"))]
        # a staged library in every manifest, as build_sets.sh writes them
        for m in cls.manifests:
            doc = json.loads(Path(m).read_text())
            doc["staged_libs"] = [{"name": "libAsyncRTMojoBindings.so"}]
            doc["driver_libs_not_staged"] = ["libcuda.so.1"]
            Path(m).write_text(json.dumps(doc))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.work = Path(tempfile.mkdtemp(dir=self.root))
        self.dist = self.work / "dist"
        shutil.copytree(self.packed, self.dist)
        shims = self.work / "bin"
        shims.mkdir()
        (shims / "docker").write_text(DOCKER_SHIM)
        (shims / "docker").chmod(0o755)
        self.log = self.work / "docker.log"
        self.env = dict(os.environ, PATH=f"{shims}:{os.environ['PATH']}", FAKE_DOCKER_LOG=str(self.log))

    def wheel(self, prefix):
        return next(self.dist.glob(prefix + "-*.whl"))

    def audit(self, whl, manifests=None, **env):
        return subprocess.run(["bash", str(AUDIT), str(whl), *(manifests or self.manifests)],
                              capture_output=True, text=True, env=dict(self.env, **env), timeout=120)

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def excluded(self):
        repair = next(c[-1] for c in self.calls() if c[-1].startswith("auditwheel repair"))
        words = repair.split()
        return {words[i + 1] for i, w in enumerate(words) if w == "--exclude"}

    def test_each_wheel_of_the_set_is_audited_into_one_directory(self):
        for prefix in ("mojolearn", "mojolearn_nvidia", "mojolearn_amd"):
            r = self.audit(self.wheel(prefix))
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        audit = self.dist / "audit"
        self.assertEqual(sorted(p.name.split("-")[0] for p in (audit / "repaired").glob("*.whl")),
                         ["mojolearn", "mojolearn_amd", "mojolearn_nvidia"])
        for log in ("show.txt", "repair.txt", "twine.txt", "show-mojolearn_nvidia.txt", "repair-mojolearn_amd.txt",
                    "twine-mojolearn_amd.txt"):
            self.assertTrue((audit / log).is_file(), log)
        twine = [c[-1] for c in self.calls() if "twine check" in c[-1]]
        self.assertEqual(twine, [f"pip install -q twine >/dev/null 2>&1 && twine check /r/{d}-*.whl"
                                 for d in ("mojolearn", "mojolearn_nvidia", "mojolearn_amd")])

    def test_a_plugin_excludes_the_core_runtime_by_name(self):
        with zipfile.ZipFile(self.wheel("mojolearn")) as z:
            core_libs = {n.rsplit("/", 1)[-1] for n in z.namelist() if n.startswith("mojolearn/.libs/")}
        self.assertTrue(core_libs)
        # even with manifests that name nothing, the core's closure is excluded
        empty = self.work / "empty.json"
        empty.write_text(json.dumps(dict(staged_libs=[], driver_libs_not_staged=[])))
        r = self.audit(self.wheel("mojolearn_nvidia"), [str(empty)])
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(self.excluded(), core_libs)

    def test_the_core_excludes_what_the_manifests_name(self):
        r = self.audit(self.wheel("mojolearn"))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(self.excluded(), {"libAsyncRTMojoBindings.so", "libcuda.so.1"})

    def test_a_plugin_with_no_core_beside_it_is_refused(self):
        self.wheel("mojolearn").unlink()
        r = self.audit(self.wheel("mojolearn_amd"))
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn("no mojolearn-", r.stdout)
        self.assertFalse(self.log.exists())

    def test_a_grafted_top_level_libs_directory_fails(self):
        r = self.audit(self.wheel("mojolearn_nvidia"), FAKE_GRAFT="1")
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("top-level *.libs entries (want 0): 1", r.stdout)


if __name__ == "__main__":
    unittest.main()
