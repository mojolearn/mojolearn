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
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
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


def make_sets(root, sets=SETS, libs_differ=False, include_byte_lm=False):
    """sets/<vendor>/<arch>/ trees in build_sets.sh's shape, inert bytes."""
    for vendor, arch in sets:
        adir = root / "sets" / vendor / arch
        rb, ab = [], []
        for tier in pw.TIERS:
            d = adir if tier == "fast" else adir / tier
            d.mkdir(parents=True, exist_ok=True)
            for name in pw.tier_names(tier, include_byte_lm):
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

    def test_payload_only_keeps_binary_gate_without_a_core_api_audit(self):
        out = Path(tempfile.mkdtemp(dir=self.root))
        args = [a for s in self.set_dirs for a in ("--set", s)]
        real_run = subprocess.run
        binary_gate = mock.Mock()

        def run(command, **kwargs):
            if len(command) > 1 and command[1].endswith("packaging/portable_math/wheel.py"):
                return binary_gate(command, **kwargs)
            return real_run(command, **kwargs)

        with mock.patch("subprocess.run", side_effect=run), mock.patch.object(
                wheel_api_audit, "audit", side_effect=AssertionError("no core was requested")):
            self.assertEqual(pw.main(args + ["--wheels", "nvidia", "--out", str(out)]), 0)
        binary_gate.assert_called_once()
        self.assertIn("--audit-only", binary_gate.call_args.args[0])
        self.assertTrue(binary_gate.call_args.kwargs["check"])
        self.assertEqual(len(list(out.glob("*.whl"))), 1)
        self.assertEqual(json.loads(next(out.glob("SPLIT-*.json")).read_text())["problems"], [])
        self.assertFalse(list(out.glob("API-*.json")))
    def test_the_three_wheels_are_the_combined_wheel(self):
        """Identity: same members, same paths, same bytes; only .dist-info differs."""
        single = {pw.gpu_plugins.installed_member(n): b for n, b in members(self.single).items() if ".dist-info/" not in n}
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
                    rel = f"mojolearn/{pw.gpu_plugins.native_directory(vendor)}/{arch}/" + ("" if tier == "fast" else tier + "/") + name + ".so"
                    self.assertEqual(union[rel], f"{vendor}/{arch}/{tier}/{name}".encode())

    def test_qualification_requires_payloads_as_well_as_aggregates(self):
        import check_linux_release_qualification as qualification
        staged = Path(tempfile.mkdtemp(dir=self.root))
        wheels = []
        for name, wheel in self.split.items():
            row = next((r for r in pw.gpu_plugins.distribution_rows() if r['wheel_name'] == name), None)
            profile = row['profile'] if row else pw.gpu_plugins.CORE_PROFILE
            data = members(wheel)
            dist = dist_info(wheel)
            data.pop(dist + '/RECORD')
            data[dist + '/LINUX_PAYLOAD.json'] = json.dumps({
                'assembly_profile': 'release-split',
                'sets': {v + '/' + a: {} for v, a in SETS},
                'split': {'role': profile},
            }).encode()
            wheels.append(pw.write_wheel(staged / wheel.name, {}, data, dist))
        output = staged / 'combined'
        output.mkdir()
        combined, vendors = qualification.split_combined(wheels, output)
        self.assertEqual(vendors, ('cuda', 'hip'))
        self.assertEqual({n: b for n, b in members(combined).items() if '.dist-info/' not in n},
                         {pw.gpu_plugins.installed_member(n): b for n, b in members(self.single).items()
                          if '.dist-info/' not in n})
        incomplete = [w for w in wheels if not w.name.startswith('mojolearn_nvidia-')]
        with self.assertRaisesRegex(ValueError, 'every plugin'):
            qualification.split_combined(incomplete, output)

    def test_file_names_and_tags(self):
        self.assertEqual(sorted(p.name for p in self.split.values()), sorted([
            f"mojolearn-{self.version}-{TAG}.whl",
            f"mojolearn_nvidia-{self.version}-{TAG}.whl",
            f"mojolearn_amd-{self.version}-{TAG}.whl",
]))
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
        for row in pw.gpu_plugins.PAYLOADS.values():
            if not row["release_enabled"]:
                continue
            payload = [n for n in members(self.split[row["wheel_name"]]) if ".dist-info/" not in n]
            self.assertTrue(payload)
            self.assertTrue(all(n.startswith(f"mojolearn/{row['directory']}/") for n in payload))
            self.assertFalse([n for n in payload if n.endswith(".py") or "/.libs/" in n])
            self.assertTrue(pw.gpu_plugins.valid_arches(row["profile"], sorted({n.split("/")[2] for n in payload})))
        self.assertEqual(wheel_api_audit.split_audit(list(self.split.values()))["problems"], [])

    def test_upload_budget_is_vendor_specific_and_bounded(self):
        import os
        original = Path.stat
        for key, mib, accepted in (("mojolearn_nvidia", 105, True),
                                   ("mojolearn_nvidia", 250, True),
                                   ("mojolearn_nvidia", 251, False),
                                   ("mojolearn_amd", 101, False),
                                   ("mojolearn", 101, False)):
            target = self.split[key]
            def stat(path, *args, **kwargs):
                result = original(path, *args, **kwargs)
                if path == target:
                    values = list(result)
                    values[6] = mib * 1024**2
                    return os.stat_result(values)
                return result
            with self.subTest(project=key, mib=mib), mock.patch.object(Path, "stat", stat):
                problems = wheel_api_audit.split_audit([target])["problems"]
                self.assertEqual(not problems, accepted, problems)

    def test_metadata_core_requires_both_plugins_and_they_pin_it_back(self):
        v = self.version
        core = meta(self.split["mojolearn"])
        self.assertEqual(core.get("Name"), "mojolearn")
        # `pip install mojolearn` WORKS FOR EVERYONE: no extras, and the core
        # requires BOTH plugins at its own version exactly
        self.assertEqual(core.get_all("Provides-Extra"), ["numpy", "verify"])
        self.assertIn('numpy>=1.26.4; extra == "verify"', core.get_all("Requires-Dist"))
        self.assertNotIn('numpy>=1.26.4', core.get_all("Requires-Dist"))
        self.assertEqual([r for r in core.get_all("Requires-Dist") or [] if r.startswith("mojolearn")],
                         [f"mojolearn-nvidia=={v}", f"mojolearn-amd=={v}"])
        for key, name in (("mojolearn_nvidia", "mojolearn-nvidia"), ("mojolearn_amd", "mojolearn-amd")):
            m = meta(self.split[key])
            self.assertEqual((m.get("Name"), m.get("Version")), (name, v))
            self.assertEqual(m.get_all("Requires-Dist"), [f"mojolearn=={v}"])
            self.assertEqual(m.get("Requires-Python"), core.get("Requires-Python"))
            self.assertEqual(m.get("License-Expression"), core.get("License-Expression"))
            self.assertIn(f"{dist_info(self.split[key])}/licenses/LICENSE", members(self.split[key]))
        # the split core's METADATA is the combined wheel's plus EXACTLY the
        # two plugin requirements, nothing else added, removed or reordered
        single = meta(self.single)
        self.assertEqual(single.get_all("Provides-Extra"), ["numpy", "verify"])
        ours = members(self.split["mojolearn"])[f"{dist_info(self.split['mojolearn'])}/METADATA"].decode()
        theirs = members(self.single)[f"{dist_info(self.single)}/METADATA"].decode()
        self.assertNotIn("Requires-Dist: mojolearn-", theirs)
        ours_lines, theirs_lines = ours.split("\n"), theirs.split("\n")
        added = [f"Requires-Dist: mojolearn-nvidia=={v}", f"Requires-Dist: mojolearn-amd=={v}"]
        self.assertEqual([ln for ln in ours_lines if ln not in added], theirs_lines)
        self.assertEqual(len(ours_lines), len(theirs_lines) + 2)
        # and they sit with the other Requires-Dist lines, before Dynamic:
        at = ours_lines.index(added[0])
        self.assertEqual(ours_lines[at:at + 3], added + ["Dynamic: license-file"])

    def test_the_core_publishes_only_after_both_plugins_are_on_the_index(self):
        """wheel_api_audit.plugins_on_index, the workflow's last gate before a
        split core uploads: both plugins of its version must be on the index."""
        v = self.version
        core, nvidia, amd = (self.split[k] for k in ("mojolearn", "mojolearn_nvidia", "mojolearn_amd"))
        asked = []

        def index(served):
            def files(where, project, version):
                asked.append((where, project, version))
                return served.get(project, [])
            return files
        both = {"mojolearn-nvidia": [nvidia.name], "mojolearn-amd": [amd.name]}
        self.assertEqual(wheel_api_audit.plugins_on_index([core], "pypi", files=index(both), sleep=lambda s: None), [])
        self.assertIn(("pypi", "mojolearn-amd", v), asked)
        for label, served, missing in (
                ("neither", {}, ["mojolearn-nvidia", "mojolearn-amd"]),
                ("nvidia only", {"mojolearn-nvidia": [nvidia.name]}, ["mojolearn-amd"]),
                ("amd only", {"mojolearn-amd": [amd.name]}, ["mojolearn-nvidia"]),
                ("another version", {"mojolearn-nvidia": [nvidia.name.replace(v, "0.0.1")],
                                     "mojolearn-amd": [amd.name]}, ["mojolearn-nvidia"]),
                ("an sdist only", {"mojolearn-nvidia": [f"mojolearn_nvidia-{v}.tar.gz"],
                                   "mojolearn-amd": [amd.name]}, ["mojolearn-nvidia"])):
            with self.subTest(label):
                naps = []
                problems = wheel_api_audit.plugins_on_index([core], "testpypi", files=index(served),
                                                            attempts=3, sleep=naps.append)
                self.assertEqual([p.split(": ", 1)[1].split("==")[0] for p in problems], missing)
                self.assertIn("is not on testpypi", problems[0])
                self.assertEqual(len(naps), 2 * len(missing))   # retried, the index may be propagating
        # Vendor aggregates publish only after every payload is on the index.
        payload_index = {r["distribution"]: [self.split[r["wheel_name"]].name]
                         for r in pw.gpu_plugins.PAYLOADS.values() if r["release_enabled"]}
        self.assertEqual(wheel_api_audit.plugins_on_index([nvidia, amd], "pypi",
                          files=index(payload_index), sleep=lambda s: None), [])
        self.assertEqual(len(wheel_api_audit.plugins_on_index([nvidia, amd], "pypi",
                          files=index({}), sleep=lambda s: None)), 0)
        self.assertEqual(wheel_api_audit.plugins_on_index([self.single], "pypi",
                          files=index({}), sleep=lambda s: None), [])

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

    def test_payload_paths_cannot_be_removed_by_an_old_vendor_record(self):
        old_owned = {n for n in members(self.single) if n.startswith(("mojolearn/cuda/", "mojolearn/hip/"))}
        new_owned = {n for w in self.split.values() for n in members(w) if ".dist-info/" not in n}
        self.assertTrue(old_owned)
        self.assertFalse(old_owned & new_owned)
        for name in old_owned:
            mapped = pw.gpu_plugins.installed_member(name)
            self.assertEqual(len(name.split("/")), len(mapped.split("/")))
            self.assertIn(mapped, new_owned)

    def test_vendor_requires_all_its_native_architecture_slots(self):
        root = Path(tempfile.mkdtemp(dir=self.root))
        # 0.8.37: the Hopper slot is not required (gpu_plugins.py), so the slot that must be filled is sm_89
        dirs = make_sets(root, sets=(("cuda", "sm_90a"),))
        with self.assertRaisesRegex(SystemExit, "requires one set per registered architecture slot"):
            self.pack("--wheels", "nvidia", sets=dirs)
        with self.assertRaisesRegex(SystemExit, "name each"):
            self.pack("--wheels", "nvidia-sm89", sets=dirs)

    def test_experimental_ptx_payload_is_explicit_and_cannot_enter_release(self):
        self._check_experimental_baseline(include_byte_lm=False)

    def test_experimental_ptx_preserves_full_byte_lm_payload_and_readbacks(self):
        self._check_experimental_baseline(include_byte_lm=True)

    def test_native_generic_does_not_start_accepting_optional_byte_lm(self):
        root = Path(tempfile.mkdtemp(dir=self.root))
        dirs = make_sets(root, sets=(("cuda", "sm_89"),), include_byte_lm=True)
        with self.assertRaisesRegex(SystemExit, "undeclared or missing native payload"):
            self.pack("--wheels", "nvidia", sets=dirs)

    def _check_experimental_baseline(self, include_byte_lm):
        import subprocess
        root = Path(tempfile.mkdtemp(dir=self.root))
        dirs = make_sets(root, sets=(("cuda", "sm_80"),), include_byte_lm=include_byte_lm)
        adir = root / "sets" / "cuda" / "sm_80"
        source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        files = []
        for binary in sorted(adir.rglob("_mojolearn*.so")):
            rel = binary.relative_to(adir).as_posix()
            if rel.startswith("host/"):
                continue
            mode = rel.split("/", 1)[0] if "/" in rel else "fast"
            files.append(dict(file=rel, numeric_mode=mode, sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                              ptx_modules=[dict(target="sm_80", sha256="b" * 64)]))
        doc = dict(schema="mojolearn.ptx-baseline.v1", code_format="ptx-baseline", vendor="cuda",
                   target="sm_80", min_compute_capability=[8, 0], source_commit=source,
                   experimental=True, identical_qualified=False, qualification_required=True,
                   errors=[], files=files)
        (adir / pw.gpu_plugins.BASELINE_MANIFEST).write_text(json.dumps(doc))
        wheels = self.pack("--wheels", "nvidia-ptx80", sets=dirs)
        self.assertEqual(len(wheels), 1)
        payload = members(wheels[0])
        self.assertIn("mojolearn/cuda_ptx/sm_80/PTX_BASELINE.json", payload)
        self.assertTrue(all(n.startswith("mojolearn/cuda_ptx/sm_80/") for n in payload if ".dist-info/" not in n))
        byte_member = "mojolearn/cuda_ptx/sm_80/identical/_mojolearn_byte_lm.so"
        self.assertEqual(byte_member in payload, include_byte_lm)
        if include_byte_lm:
            self.assertEqual(payload[byte_member], (adir / "identical/_mojolearn_byte_lm.so").read_bytes())
            packed_doc = json.loads(payload["mojolearn/cuda_ptx/sm_80/PTX_BASELINE.json"])
            self.assertTrue(any(row["file"] == "identical/_mojolearn_byte_lm.so" for row in packed_doc["files"]))
            witness = adir / "readback.txt"
            original = witness.read_text()
            witness.write_text("\n".join(line for line in original.splitlines()
                                         if not line.startswith("identical _mojolearn_byte_lm ")) + "\n")
            with self.assertRaisesRegex(SystemExit, "incomplete release native readback"):
                self.pack("--wheels", "nvidia-ptx80", sets=dirs)
            witness.write_text(original)
        with self.assertRaisesRegex(SystemExit, "experimental payloads cannot"):
            self.pack("--profile", "release-split", "--wheels", "nvidia-ptx80", sets=dirs)
        for req in pw.gpu_plugins.core_requirements(self.version) + pw.gpu_plugins.payload_requirements("cuda", self.version):
            self.assertNotIn("ptx", req)
        doc["source_commit"] = "c" * 40
        (adir / pw.gpu_plugins.BASELINE_MANIFEST).write_text(json.dumps(doc))
        with self.assertRaisesRegex(SystemExit, "source commits differ"):
            self.pack("--wheels", "nvidia-ptx80", sets=dirs)

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
            "core lacks the amd plugin": [self.rewrite(core, replace={
                f"{dist_info(core)}/METADATA": members(core)[f"{dist_info(core)}/METADATA"].replace(
                    f"Requires-Dist: mojolearn-amd=={self.version}\n".encode(), b"", 1)}),
                nvidia, amd],
            "core pins a plugin loosely": [self.rewrite(core, replace={
                f"{dist_info(core)}/METADATA": members(core)[f"{dist_info(core)}/METADATA"].replace(
                    f"mojolearn-nvidia=={self.version}".encode(), b"mojolearn-nvidia>=0.1", 1)}),
                nvidia, amd],
            "core requires a plugin twice": [self.rewrite(core, replace={
                f"{dist_info(core)}/METADATA": members(core)[f"{dist_info(core)}/METADATA"].replace(
                    b"\nDynamic:", f"\nRequires-Dist: mojolearn-amd=={self.version}\nDynamic:".encode(), 1)}),
                nvidia, amd],
            "core lost its marker": [self.rewrite(core, drop={f"{dist_info(core)}/gpu_plugins.json"}), nvidia, amd],
        }
        why = {"core declares a GPU extra": "declares Provides-Extra ['numpy', 'nvidia', 'verify']",
               "core lacks the amd plugin": "dependency pins disagree",
               "core pins a plugin loosely": "dependency pins disagree",
               "core requires a plugin twice": "dependency pins disagree"}
        for label, wheels in cases.items():
            with self.subTest(label):
                problems = wheel_api_audit.split_audit(wheels)["problems"]
                self.assertTrue(problems, label)
                if label in why:
                    self.assertIn(why[label], "\n".join(problems))


if __name__ == "__main__":
    unittest.main()
