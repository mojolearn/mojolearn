#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""THE FRAMEWORK ADAPTERS IN THE LINUX WHEELS, packed from tiny fake sets and read as zips.

    .pixi/envs/test/bin/python -m pytest -q packaging/linux/test_adapter_wheels.py

The fake sets of test_split_wheels.py, each native one (cuda/sm_89,
hip/gfx942) given a `_adapters/` directory the way build_sets.sh leaves it
(three inert files and their adapters.json), packed by the real
`pack_wheel.main` with `_gates=False`. The adapters land in the vendor wheels
at the native set's depth, the core carries the `mojolearn.adapters` Python
package and the torch/max extras and no adapter file, the split audit passes,
and the packer refuses bytes that differ from their manifest, adapters on the
PTX slot, and (with --require-adapters) a native set without them.
"""
import email.parser
import email.policy
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import test_split_wheels as tsw  # noqa: E402

pw = tsw.pw
manifest = pw.adapter_manifest
FILES = {"libmojolearn_adapter.so": b"native adapter", "mojolearn_torch.so": b"torch registration",
         "mojolearn_ops.mojoc": b"max package"}


def write_adapters(adir, vendor, arch, data=None):
    """<set>/_adapters/ as build_sets.sh leaves it after manifest.py finalize."""
    d = Path(adir) / manifest.DIRECTORY
    d.mkdir(parents=True, exist_ok=True)
    for name, blob in (data or FILES).items():
        (d / name).write_bytes(blob + f" {vendor}/{arch}".encode())
    pins = manifest.pins()
    pin = pins["torch"]["build"][vendor]
    doc = dict(schema=manifest.SCHEMA, abi=manifest.ABI, vendor=vendor, arch=arch, platform="linux",
               source_commit="a" * 40, numeric_mode="identical",
               torch=dict(version=pin["version"], series=pins["torch"]["series"], backend=pin["backend"],
                          backend_version="x", cxx11_abi=1, wheel="torch.whl", wheel_sha256="b" * 64,
                          index_url=pin["index_url"], python_tag="cp312"),
               max=dict(version=pins["max"]["version"]), mojo=dict(version="Mojo 1.0.0"),
               requirements=dict(torch=pins["torch"]["requirement"], max=pins["max"]["requirement"]),
               commands={role: ["tool"] for role in manifest.FILES}, sources={},
               files=manifest.file_rows(d))
    (d / manifest.MANIFEST).write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return d


def members(whl):
    return tsw.members(whl)


def meta(whl):
    return tsw.meta(whl)


class AdapterWheels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.set_dirs = tsw.make_sets(cls.root, merged=True)
        for vendor, arch in tsw.NATIVE_SETS:
            write_adapters(cls.root / "sets" / vendor / arch, vendor, arch)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def pack(self, *extra, sets=None, env=None):
        out = Path(tempfile.mkdtemp(dir=self.root))
        args = [a for s in (sets or self.set_dirs) for a in ("--set", s)]
        with mock.patch.dict(os.environ, env or {}, clear=False):
            rc = pw.main(args + list(extra) + ["--out", str(out)], _gates=False)
        self.assertEqual(rc, 0)
        return {p.name.split("-")[0]: p for p in out.glob("*.whl")}, out

    def copy_sets(self):
        dest = Path(tempfile.mkdtemp(dir=self.root)) / "copy"
        shutil.copytree(self.root / "sets", dest / "sets")
        return sorted(str(dest / "sets" / v) for v in ("cuda", "hip")), dest / "sets"

    def test_vendor_wheels_carry_the_adapters_at_the_native_set_depth(self):
        wheels, out = self.pack("--require-adapters")
        for key, directory, arch in (("mojolearn_nvidia", "cuda_native", "sm_89"), ("mojolearn_amd", "hip_native", "gfx942")):
            m = members(wheels[key])
            prefix = f"mojolearn/{directory}/{arch}/{manifest.DIRECTORY}/"
            got = sorted(n[len(prefix):] for n in m if n.startswith(prefix))
            self.assertEqual(got, sorted([*FILES, manifest.MANIFEST]))
            doc = json.loads(m[prefix + manifest.MANIFEST])
            digests = {n[len(prefix):]: (__import__("hashlib").sha256(b).hexdigest(), len(b))
                       for n, b in m.items() if n.startswith(prefix) and not n.endswith(manifest.MANIFEST)}
            manifest.validate(doc, digests, vendor=directory.split("_")[0], arch=arch)
        ptx = members(wheels["mojolearn_nvidia_ptx"])
        self.assertFalse([n for n in ptx if f"/{manifest.DIRECTORY}/" in n], "the PTX slot carries no adapters")
        report = json.loads(next(out.glob("SPLIT-*.json")).read_text())
        self.assertEqual(report["problems"], [])

    def test_the_core_carries_the_python_package_and_the_extras_not_the_files(self):
        wheels, _ = self.pack()
        core = members(wheels["mojolearn_core"])
        for leaf in ("__init__.py", "_shipped.py", "torch.py", "max.py"):
            self.assertIn(f"mojolearn/adapters/{leaf}", core)
        self.assertFalse([n for n in core if f"/{manifest.DIRECTORY}/" in n])
        m = meta(wheels["mojolearn_core"])
        self.assertEqual(m.get_all("Provides-Extra"), ["numpy", "verify", "torch", "max"])
        pins = manifest.pins()
        self.assertIn(f'{pins["torch"]["requirement"]}; extra == "torch"', m.get_all("Requires-Dist"))
        self.assertIn(f'{pins["max"]["requirement"]}; extra == "max"', m.get_all("Requires-Dist"))

    def test_bytes_that_differ_from_their_manifest_are_refused(self):
        sets, base = self.copy_sets()
        (base / "cuda/sm_89/_adapters/mojolearn_torch.so").write_bytes(b"edited")
        with self.assertRaises(SystemExit) as ctx:
            pw.main([a for s in sets for a in ("--set", s)] + ["--out", str(base.parent / "o")], _gates=False)
        self.assertIn("differ from the manifest", str(ctx.exception))

    def test_adapters_on_the_ptx_slot_are_refused(self):
        sets, base = self.copy_sets()
        write_adapters(base / "cuda/sm_80", "cuda", "sm_80")
        with self.assertRaises(SystemExit) as ctx:
            pw.main([a for s in sets for a in ("--set", s)] + ["--out", str(base.parent / "o")], _gates=False)
        self.assertIn("PTX slot", str(ctx.exception))

    def test_require_adapters_refuses_a_native_set_without_them(self):
        sets, base = self.copy_sets()
        shutil.rmtree(base / "hip/gfx942/_adapters")
        with self.assertRaises(SystemExit) as ctx:
            pw.main([a for s in sets for a in ("--set", s)] + ["--require-adapters", "--out", str(base.parent / "o")],
                    _gates=False)
        self.assertIn("hip/gfx942", str(ctx.exception))
        # without the flag the set packs, without adapters
        out = base.parent / "o2"
        self.assertEqual(pw.main([a for s in sets for a in ("--set", s)] + ["--out", str(out)], _gates=False), 0)
        amd = members(next(out.glob("mojolearn_amd-*.whl")))
        self.assertFalse([n for n in amd if f"/{manifest.DIRECTORY}/" in n])

    def test_the_switch_off_packs_none(self):
        wheels, _ = self.pack(env={"MOJOLEARN_RELEASE_ADAPTERS": "0"})
        for key in ("mojolearn_nvidia", "mojolearn_amd"):
            self.assertFalse([n for n in members(wheels[key]) if f"/{manifest.DIRECTORY}/" in n])

    def test_adapter_records_have_the_build_provenance_shape(self):
        sets = [t for s in self.set_dirs for t in pw.load_set(s)]
        rec = pw.adapter_records(sets)
        self.assertEqual(sorted(rec), ["mojolearn/cuda/sm_89/_adapters", "mojolearn/hip/gfx942/_adapters"])
        row = rec["mojolearn/cuda/sm_89/_adapters"]
        self.assertEqual(sorted(row["files"]), sorted([*FILES, manifest.MANIFEST]))
        self.assertEqual(row["manifest_sha256"], row["files"][manifest.MANIFEST])


if __name__ == "__main__":
    unittest.main()
