#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE SPLIT LINUX WHEELS, PACKED FROM TINY FAKE SETS AND READ AS ZIPS.

    .pixi/envs/test/bin/python -m pytest -q packaging/linux/test_split_wheels.py

Three fake architecture sets (cuda sm_89, cuda sm_90a, hip gfx942) shaped the
way build_sets.sh leaves them -- every tier's bindings, read-back witnesses,
host bindings and a runtime closure, all inert bytes -- are packed twice by
the real `pack_wheel.main`: once as the combined wheel (`--profile generic`)
and once as the split (the default). Then the wheels are opened as zip files.

THE IDENTITY PROOF is `test_the_three_wheels_are_the_combined_wheel`: every
member of the combined wheel outside its .dist-info is in exactly one split
wheel, at the same archive path, with the same bytes, and the split wheels
carry nothing else. `_gates=False` skips portable_math/wheel.py and the API
audit, which read real ELF and the checkout's API surface; the first
rewrites only runtime directories (.libs), which the core carries whole and
no plugin carries, so it cannot touch a set's bytes.
"""
import email.parser
import email.policy
import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools"))
spec = importlib.util.spec_from_file_location("pack_wheel_split_under_test", HERE / "pack_wheel.py")
pw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pw)
import wheel_api_audit  # noqa: E402

SETS = (("cuda", "sm_89"), ("cuda", "sm_90a"), ("hip", "gfx942"))
TAG = "py3-none-manylinux_2_35_x86_64"


def make_sets(root, sets=SETS, libs_differ=False):
    """sets/<vendor>/<arch>/ trees in build_sets.sh's shape, inert bytes."""
    for vendor, arch in sets:
        adir = root / "sets" / vendor / arch
        rb, ab = [], []
        for tier in pw.TIERS:
            d = adir if tier == "fast" else adir / tier
            d.mkdir(parents=True, exist_ok=True)
            for name in pw.tier_names(tier):
                (d / f"{name}.so").write_bytes(f"{vendor}/{arch}/{tier}/{name}".encode())
                rb.append(f"{tier} {name} {vendor}")
                ab.append(f"{tier} {name} {arch}")
        (adir / "host").mkdir()
        for name in pw.HOST_NAMES:
            (adir / "host" / f"{name}.so").write_bytes(f"host {name}".encode())
            rb.append(f"host {name} cpu")
            ab.append(f"host {name} NONE-BY-DESIGN")
        (adir / "readback.txt").write_text("\n".join(rb) + "\n")
        (adir / "arch_readback.txt").write_text("\n".join(ab) + "\n")
        (adir / ".libs").mkdir()
        (adir / ".libs" / "libAsyncRTMojoBindings.so").write_bytes(
            b"runtime" + (f" {vendor}".encode() if libs_differ else b""))
        (adir / ".libs" / "libMojolearnMath.so").write_bytes(b"math")
        (adir / "manifest.json").write_text(json.dumps(
            dict(bytes_extensions=1, bytes_staged_libs=1, driver_libs_not_staged=[])))
    return sorted({str(root / "sets" / v) for v, _ in sets})


def members(whl):
    """{archive path: bytes} of one wheel."""
    with zipfile.ZipFile(whl) as z:
        return {n: z.read(n) for n in z.namelist()}


def dist_info(whl):
    with zipfile.ZipFile(whl) as z:
        dists = {n.split("/")[0] for n in z.namelist() if n.split("/")[0].endswith(".dist-info")}
    assert len(dists) == 1, dists
    return dists.pop()


def meta(whl, leaf="METADATA"):
    data = members(whl)[f"{dist_info(whl)}/{leaf}"]
    return email.parser.BytesParser(policy=email.policy.compat32).parsebytes(data)


class SplitWheels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.set_dirs = make_sets(cls.root)
        cls.version = pw.read_version()
        args = [a for s in cls.set_dirs for a in ("--set", s)]
        assert pw.main(args + ["--profile", "generic", "--out", str(cls.root / "single")], _gates=False) == 0
        assert pw.main(args + ["--out", str(cls.root / "split")], _gates=False) == 0
        cls.single = next((cls.root / "single").glob("*.whl"))
        cls.split = {p.name.split("-")[0]: p for p in (cls.root / "split").glob("*.whl")}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def pack(self, *extra, sets=None):
        out = Path(tempfile.mkdtemp(dir=self.root))
        args = [a for s in (sets or self.set_dirs) for a in ("--set", s)]
        pw.main(args + list(extra) + ["--out", str(out)], _gates=False)
        return sorted(out.glob("*.whl"))

    # ---- the partition ------------------------------------------------------
    def test_the_three_wheels_are_the_combined_wheel(self):
        """Identity: same members, same paths, same bytes; only .dist-info differs."""
        single = {n: b for n, b in members(self.single).items() if ".dist-info/" not in n}
        union = {}
        for whl in self.split.values():
            for n, b in members(whl).items():
                if ".dist-info/" in n:
                    continue
                self.assertNotIn(n, union, f"{n} is in two split wheels")
                union[n] = b
        self.assertEqual(sorted(union), sorted(single))
        differ = [n for n in single if hashlib.sha256(single[n]).digest() != hashlib.sha256(union[n]).digest()]
        self.assertEqual(differ, [])
        # and the sets really are in there, every tier of every architecture
        for vendor, arch in SETS:
            for tier in pw.TIERS:
                for name in pw.tier_names(tier):
                    rel = f"mojolearn/{vendor}/{arch}/" + ("" if tier == "fast" else tier + "/") + name + ".so"
                    self.assertEqual(union[rel], f"{vendor}/{arch}/{tier}/{name}".encode())

    def test_file_names_and_tags(self):
        self.assertEqual(sorted(p.name for p in self.split.values()), sorted([
            f"mojolearn-{self.version}-{TAG}.whl",
            f"mojolearn_nvidia-{self.version}-{TAG}.whl",
            f"mojolearn_amd-{self.version}-{TAG}.whl"]))
        for whl in self.split.values():
            wheel = meta(whl, "WHEEL")
            self.assertEqual(wheel.get_all("Tag"), [TAG])
            self.assertEqual(wheel.get("Root-Is-Purelib"), "false")
        # the combined profile still writes linux_x86_64 for audit.sh to retag
        self.assertTrue(self.single.name.endswith("-py3-none-linux_x86_64.whl"))

    def test_each_plugin_holds_exactly_its_sets_and_the_core_none(self):
        core = [n for n in members(self.split["mojolearn"]) if ".dist-info/" not in n]
        self.assertFalse([n for n in core if n.startswith(("mojolearn/cuda/", "mojolearn/hip/"))])
        self.assertIn("mojolearn/__init__.py", core)
        self.assertIn("mojolearn/gpu_plugins.py", core)
        self.assertTrue(any(n.startswith("mojolearn/host/") for n in core))
        self.assertTrue(any(n.startswith("mojolearn/.libs/") for n in core))
        for key, vendor in (("mojolearn_nvidia", "cuda"), ("mojolearn_amd", "hip")):
            payload = [n for n in members(self.split[key]) if ".dist-info/" not in n]
            self.assertTrue(payload)
            self.assertEqual([n for n in payload if not n.startswith(f"mojolearn/{vendor}/")], [])
            self.assertFalse([n for n in payload if n.endswith(".py") or "/.libs/" in n])
            want = {a for v, a in SETS if v == vendor}
            self.assertEqual({n.split("/")[2] for n in payload}, want)
        self.assertEqual(wheel_api_audit.split_audit(list(self.split.values()))["problems"], [])

    def test_metadata_plugins_pin_the_core_and_the_core_has_no_extras(self):
        v = self.version
        core = meta(self.split["mojolearn"])
        self.assertEqual(core.get("Name"), "mojolearn")
        # NO GPU EXTRAS: the core declares no extra and requires no plugin
        self.assertIsNone(core.get_all("Provides-Extra"))
        self.assertFalse([r for r in core.get_all("Requires-Dist") or [] if r.startswith("mojolearn")])
        for key, name in (("mojolearn_nvidia", "mojolearn-nvidia"), ("mojolearn_amd", "mojolearn-amd")):
            m = meta(self.split[key])
            self.assertEqual((m.get("Name"), m.get("Version")), (name, v))
            self.assertEqual(m.get_all("Requires-Dist"), [f"mojolearn=={v}"])
            self.assertEqual(m.get("Requires-Python"), core.get("Requires-Python"))
            self.assertEqual(m.get("License-Expression"), core.get("License-Expression"))
            self.assertIn(f"{dist_info(self.split[key])}/licenses/LICENSE", members(self.split[key]))
        # the split core's METADATA is the combined wheel's, byte for byte
        single = meta(self.single)
        self.assertIsNone(single.get_all("Provides-Extra"))
        self.assertEqual(members(self.split["mojolearn"])[f"{dist_info(self.split['mojolearn'])}/METADATA"],
                         members(self.single)[f"{dist_info(self.single)}/METADATA"])

    def test_markers_and_records(self):
        gp = pw.gpu_plugins
        core = self.split["mojolearn"]
        self.assertEqual(json.loads(members(core)[f"{dist_info(core)}/{gp.CORE_MARKER}"]),
                         gp.core_marker(self.version))
        nvidia = self.split["mojolearn_nvidia"]
        self.assertEqual(json.loads(members(nvidia)[f"{dist_info(nvidia)}/{gp.PLUGIN_MARKER}"]),
                         gp.plugin_marker("cuda", self.version, ["sm_89", "sm_90a"]))
        for whl in self.split.values():
            files = members(whl)
            record = files[f"{dist_info(whl)}/RECORD"].decode().splitlines()
            self.assertEqual({r.split(",")[0] for r in record}, set(files))

    # ---- subsets and refusals -------------------------------------------------
    def test_the_nvidia_legs_alone_pack_the_core_and_the_nvidia_plugin(self):
        cuda_only = [s for s in self.set_dirs if s.endswith("/cuda")]
        wheels = self.pack(sets=cuda_only)
        self.assertEqual([p.name.split("-")[0] for p in wheels], ["mojolearn", "mojolearn_nvidia"])
        self.assertEqual(wheel_api_audit.split_audit(wheels)["problems"], [])
        with self.assertRaises(SystemExit) as ctx:
            self.pack("--wheels", "core-linux,amd", sets=cuda_only)
        self.assertIn("no hip set", str(ctx.exception))
        only = self.pack("--wheels", "amd")
        self.assertEqual([p.name.split("-")[0] for p in only], ["mojolearn_amd"])

    def test_split_refuses_closures_that_differ(self):
        root = Path(tempfile.mkdtemp(dir=self.root))
        dirs = make_sets(root, libs_differ=True)
        out = root / "out"
        with self.assertRaises(SystemExit) as ctx:
            pw.main([a for s in dirs for a in ("--set", s)] + ["--out", str(out)], _gates=False)
        self.assertIn("ONE runtime closure", str(ctx.exception))
        self.assertFalse(list(out.glob("*.whl")) if out.exists() else [])

    def test_wheels_flag_needs_a_split_profile(self):
        with self.assertRaises(SystemExit):
            self.pack("--profile", "generic", "--wheels", "nvidia")

    def test_release_split_slots(self):
        self.assertEqual(pw.split_release_slots({"cuda"}), {("cuda", "sm_89"), ("cuda", "sm_90")})
        self.assertEqual(pw.split_release_slots({"hip"}), {("hip", "gfx942")})
        self.assertEqual(pw.split_release_slots({"cuda", "hip"}), pw.RELEASE_061_SETS)
        # one definition: the admission side's per-plugin share agrees
        import verify_linux_surface_qualification as surface
        for vendor in ("cuda", "hip"):
            slots = {"/".join(k) for k in pw.split_release_slots({vendor})}
            self.assertEqual(slots, set(surface.PLUGIN_ARCHES[vendor]))
            self.assertTrue(surface.plugin_arch_set_ok(vendor, slots))
        self.assertTrue(surface.plugin_arch_set_ok("cuda", {"cuda/sm_89", "cuda/sm_90a"}))
        self.assertFalse(surface.plugin_arch_set_ok("cuda", {"cuda/sm_89", "cuda/sm_90a", "cuda/sm_90"}))
        self.assertFalse(surface.plugin_arch_set_ok("hip", {"hip/gfx942", "cuda/sm_89"}))
        hip = pw.SetDir("hip", "gfx942", {}, {}, {}, {}, None)
        with self.assertRaises(SystemExit) as ctx:
            pw.release_inventory([hip], [], self.version, required=pw.split_release_slots({"cuda"}),
                                 profile=pw.RELEASE_SPLIT_PROFILE)
        self.assertIn("release-split requires exactly the sets cuda/sm_89, cuda/sm_90", str(ctx.exception))

    # ---- the audit sees a broken split ----------------------------------------
    def rewrite(self, whl, add=None, drop=(), replace=None):
        out = Path(tempfile.mkdtemp(dir=self.root)) / whl.name
        with zipfile.ZipFile(whl) as src, zipfile.ZipFile(out, "w") as dst:
            for n in src.namelist():
                if n in drop:
                    continue
                dst.writestr(n, (replace or {}).get(n, src.read(n)))
            for n, b in (add or {}).items():
                dst.writestr(n, b)
        return out

    def test_the_audit_refuses_a_broken_split(self):
        core, nvidia, amd = (self.split[k] for k in ("mojolearn", "mojolearn_nvidia", "mojolearn_amd"))
        ok = wheel_api_audit.split_audit([core, nvidia, amd])["problems"]
        self.assertEqual(ok, [])
        cases = {
            "core carries a set": [self.rewrite(core, add={"mojolearn/cuda/sm_89/x.so": b"x"}), nvidia, amd],
            "plugin carries python": [core, self.rewrite(nvidia, add={"mojolearn/cuda/x.py": b""}), amd],
            "plugin carries the other vendor": [core, self.rewrite(nvidia, add={"mojolearn/hip/gfx942/y.so": b""}), amd],
            "loose pin": [core, self.rewrite(nvidia, replace={
                f"{dist_info(nvidia)}/METADATA": members(nvidia)[f"{dist_info(nvidia)}/METADATA"].replace(
                    f"mojolearn=={self.version}".encode(), b"mojolearn>=0.1")}), amd],
            "a member in two wheels": [core, nvidia, self.rewrite(amd, add={"mojolearn/__init__.py": b""})],
            "core declares a GPU extra": [self.rewrite(core, replace={
                f"{dist_info(core)}/METADATA": members(core)[f"{dist_info(core)}/METADATA"].replace(
                    b"\nDynamic:", f'\nProvides-Extra: nvidia\nRequires-Dist: mojolearn-nvidia=={self.version}; '
                    f'extra == "nvidia"\nDynamic:'.encode(), 1)}), nvidia, amd],
            "core requires a plugin": [self.rewrite(core, replace={
                f"{dist_info(core)}/METADATA": members(core)[f"{dist_info(core)}/METADATA"].replace(
                    b"\nDynamic:", f"\nRequires-Dist: mojolearn-amd=={self.version}\nDynamic:".encode(), 1)}),
                nvidia, amd],
            "core lost its marker": [self.rewrite(core, drop={f"{dist_info(core)}/gpu_plugins.json"}), nvidia, amd],
        }
        why = {"core declares a GPU extra": "declares Provides-Extra ['nvidia']",
               "core requires a plugin": "requires a GPU plugin"}
        for label, wheels in cases.items():
            with self.subTest(label):
                problems = wheel_api_audit.split_audit(wheels)["problems"]
                self.assertTrue(problems, label)
                if label in why:
                    self.assertIn(why[label], "\n".join(problems))


if __name__ == "__main__":
    unittest.main()
