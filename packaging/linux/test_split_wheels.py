#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE SPLIT LINUX WHEELS, PACKED FROM TINY FAKE SETS AND READ AS ZIPS.

    .pixi/envs/test/bin/python -m pytest -q packaging/linux/test_split_wheels.py

Fake architecture sets (cuda sm_89, the cuda sm_80 PTX slot, hip gfx942)
shaped the way build_sets.sh leaves them -- every tier's bindings, read-back
witnesses, host bindings, a runtime closure and, for the PTX slot, its
PTX_BASELINE.json manifest, all inert bytes -- are packed by the real
`pack_wheel.main`: the native sets as the combined wheel (`--profile
generic`, which never carries the PTX slot) and all of them as the split
(the default). Then the wheels are opened as zip files.

BOTH BINDING LAYOUTS: `SplitWheels` packs per-binding sets (one `<name>.so`
per binding) and `SplitWheelsMerged` reruns every test on merged sets (one
`_mojolearn_native.so` / `_mojolearn_native_host.so` per directory plus the
index `_mojolearn_merged.json`, the bindings/merged_groups.json plan);
`MergedLayoutRefusals` covers what only the merged layout can get wrong.

THE IDENTITY PROOF is `test_the_three_wheels_are_the_combined_wheel`: every
member of the combined wheel outside its .dist-info is in exactly one split
wheel, at the same archive path, with the same bytes, and the split wheels
carry nothing else but the PTX slot (cuda_ptx/sm_80, in mojolearn-nvidia).
The split set (2026-10-20) is mojolearn_core (no vendor pin), the vendor
wheels (each pinning mojolearn-core), and the file-free Linux mojolearn
(pinning the core and both vendors). `_gates=False` skips portable_math/wheel.py and the API
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

#: Andrew 2026-10-10: PTX is a normal target; no flag. cuda/sm_80 is the PTX slot.
SETS = (("cuda", "sm_89"), ("cuda", "sm_80"), ("hip", "gfx942"))
NATIVE_SETS = tuple(k for k in SETS if k != ("cuda", "sm_80"))
TAG = "py3-none-manylinux_2_35_x86_64"


def source_head():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


MERGED_INDEX = pw.MERGED_INDEX


def write_ptx_manifest(adir, source=None, dirty=False, **override):
    """A v2 PTX_BASELINE.json (gpu_plugins.validate_baseline_manifest) over the
    set's GPU bindings, as packaging/linux/ptx_baseline.py writes it: a merged
    file's row names its `members`, and `indexes` hashes each merged index."""
    files, indexes = [], []
    for binary in sorted(Path(adir).rglob("_mojolearn*.so")):
        rel = binary.relative_to(adir).as_posix()
        if rel.startswith(("host/", ".libs/")):
            continue
        mode = rel.split("/", 1)[0] if "/" in rel else "fast"
        row = dict(file=rel, numeric_mode=mode, sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                   ptx_modules=[dict(target="sm_80", sha256="b" * 64)])
        index = binary.parent / MERGED_INDEX
        if index.is_file():
            row["members"] = json.loads(index.read_text())["files"][binary.name]
        files.append(row)
    for index in sorted(Path(adir).rglob(MERGED_INDEX)):
        rel = index.relative_to(adir).as_posix()
        if not rel.startswith("host/"):
            indexes.append(dict(file=rel, sha256=hashlib.sha256(index.read_bytes()).hexdigest()))
    doc = dict(schema=pw.gpu_plugins.PTX_MANIFEST_SCHEMA, code_format=pw.gpu_plugins.PTX_CODE_FORMAT,
               vendor="cuda", target="sm_80", min_compute_capability=[8, 0],
               source_commit=source or source_head(), source_dirty=dirty, mojo_version="test",
               errors=[], files=files, **({"indexes": indexes} if indexes else {}))
    doc.update(override)
    path = Path(adir) / pw.gpu_plugins.BASELINE_MANIFEST
    path.write_text(json.dumps(doc))
    return path


def write_index(directory, files):
    """A format-1 merged index {"<stem>.so": [members]} (bindings/merged_build.sh's)."""
    (Path(directory) / MERGED_INDEX).write_text(json.dumps(dict(format=1, files=files), indent=1, sort_keys=True))


def dir_files(vendor, arch, tier, include_byte_lm=False, merged=False):
    """{file name: bytes} one tier directory (fast, identical) or host/ holds:
    one inert file per binding, or the merged plan's files and the index."""
    if tier == "host":
        if not merged:
            return {f"{name}.so": f"host {name}".encode() for name in pw.HOST_NAMES}
        plan = pw.merged_layout.plan("host", include_byte_lm)
        out = {f: f"host {f[:-3]}".encode() for f in plan}
    else:
        if not merged:
            return {f"{name}.so": f"{vendor}/{arch}/{tier}/{name}".encode()
                    for name in pw.tier_names(tier, include_byte_lm)}
        plan = pw.merged_layout.plan(tier, include_byte_lm)
        out = {f: f"{vendor}/{arch}/{tier}/{f[:-3]}".encode() for f in plan}
    out[MERGED_INDEX] = json.dumps(dict(format=1, files=plan), indent=1, sort_keys=True).encode()
    return out


def make_sets(root, sets=SETS, libs_differ=False, include_byte_lm=False, merged=False):
    """sets/<vendor>/<arch>/ trees in build_sets.sh's shape, inert bytes; the
    cuda/sm_80 PTX slot also gets its manifest. `merged`: every directory in
    the merged layout (read-back rows still name each binding)."""
    for vendor, arch in sets:
        adir = root / "sets" / vendor / arch
        rb, ab = [], []
        for tier in pw.TIERS:
            d = adir if tier == "fast" else adir / tier
            d.mkdir(parents=True, exist_ok=True)
            for f, data in dir_files(vendor, arch, tier, include_byte_lm, merged).items():
                (d / f).write_bytes(data)
            for name in pw.tier_names(tier, include_byte_lm):
                rb.append(f"{tier} {name} {vendor}")
                ab.append(f"{tier} {name} {arch}")
        (adir / "host").mkdir()
        for f, data in dir_files(vendor, arch, "host", include_byte_lm, merged).items():
            (adir / "host" / f).write_bytes(data)
        for name in pw.HOST_NAMES:
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
        if (vendor, arch) == ("cuda", pw.gpu_plugins.PTX_ARCH):
            write_ptx_manifest(adir)
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
    #: the binding layout every set of this class is made in
    MERGED = False

    @classmethod
    def make_sets(cls, root, **kwargs):
        return make_sets(root, merged=cls.MERGED, **kwargs)

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.set_dirs = cls.make_sets(cls.root)
        cls.native_dirs = cls.make_sets(cls.root / "native", sets=NATIVE_SETS)
        cls.version = pw.read_version()
        args = [a for s in cls.set_dirs for a in ("--set", s)]
        native = [a for s in cls.native_dirs for a in ("--set", s)]
        assert pw.main(native + ["--profile", "generic", "--out", str(cls.root / "single")], _gates=False) == 0
        assert pw.main(args + ["--out", str(cls.root / "split")], _gates=False) == 0
        cls.single = next((cls.root / "single").glob("*.whl"))
        cls.split = {p.name.split("-")[0]: p for p in (cls.root / "split").glob("*.whl")}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def pack(self, *extra, sets=None):
        out = Path(tempfile.mkdtemp(dir=self.root))
        args = [a for s in (sets or self.set_dirs) for a in ("--set", s)]
        self.last_rc = pw.main(args + list(extra) + ["--out", str(out)], _gates=False)
        self.last_out = out
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
        union, ptx = {}, {}
        for whl in self.split.values():
            for n, b in members(whl).items():
                if ".dist-info/" in n:
                    continue
                self.assertNotIn(n, union, f"{n} is in two split wheels")
                self.assertNotIn(n, ptx, f"{n} is in two split wheels")
                if n.startswith(pw.gpu_plugins.BUNDLED_PTX_ROOT + "/"):
                    self.assertIn(whl, [self.split["mojolearn_nvidia_ptx"]])
                    ptx[n] = b
                else:
                    union[n] = b
        self.assertEqual(sorted(union), sorted(single))
        differ = [n for n in single if hashlib.sha256(single[n]).digest() != hashlib.sha256(union[n]).digest()]
        self.assertEqual(differ, [])
        union.update(ptx)
        # and the sets really are in there, every tier of every architecture
        # (its binaries, and its index when merged), the PTX slot at
        # cuda_ptx/sm_80 with its manifest, and host/ once in the core
        for vendor, arch in SETS:
            for tier in pw.TIERS:
                shipped = dir_files(vendor, arch, tier, merged=self.MERGED)
                for f, data in shipped.items():
                    rel = pw.gpu_plugins.installed_member(
                        f"mojolearn/{vendor}/{arch}/" + ("" if tier == "fast" else tier + "/") + f)
                    self.assertEqual(union[rel], data)
                tier_dir = pw.gpu_plugins.installed_member(
                    f"mojolearn/{vendor}/{arch}/" + ("" if tier == "fast" else tier + "/") + "x.so").rsplit("/", 1)[0]
                self.assertEqual(sorted(n.rsplit("/", 1)[1] for n in union if n.rsplit("/", 1)[0] == tier_dir
                                        and (n.endswith(".so") or n.endswith(MERGED_INDEX))), sorted(shipped))
        for f, data in dir_files("cuda", "sm_89", "host", merged=self.MERGED).items():
            self.assertEqual(union[f"mojolearn/host/{f}"], data)
        self.assertEqual(sorted(n for n in union if n.startswith("mojolearn/host/")),
                         sorted(f"mojolearn/host/{f}" for f in dir_files("cuda", "sm_89", "host", merged=self.MERGED)))
        self.assertIn(pw.gpu_plugins.BUNDLED_PTX_ROOT + "/" + pw.gpu_plugins.BASELINE_MANIFEST, ptx)
        self.assertTrue(pw.gpu_plugins.BUNDLED_PTX_ROOT.startswith("mojolearn/cuda_ptx/sm_80"))

    def test_qualification_requires_payloads_as_well_as_aggregates(self):
        import check_linux_release_qualification as qualification
        import hashlib
        staged = Path(tempfile.mkdtemp(dir=self.root))
        wheels = []
        # The PTX slot is checked against the inventory (wheel_api_audit): name its commit and bytes.
        prefix = pw.gpu_plugins.BUNDLED_PTX_ROOT + '/'
        nvidia = self.split['mojolearn_nvidia_ptx']
        ptx_members = members(nvidia)
        ptx_commit = json.loads(ptx_members[prefix + pw.gpu_plugins.BASELINE_MANIFEST])['source_commit']
        ptx_extensions = {n: hashlib.sha256(b).hexdigest() for n, b in ptx_members.items()
                          if n.startswith(prefix) and n.endswith('.so')}
        for name, wheel in self.split.items():
            row = next((r for r in pw.gpu_plugins.distribution_rows() if r['wheel_name'] == name), None)
            profile = (row['profile'] if row else pw.gpu_plugins.META_PROFILE
                       if name == pw.gpu_plugins.META_WHEEL_NAME else pw.gpu_plugins.CORE_PROFILE)
            data = members(wheel)
            dist = dist_info(wheel)
            data.pop(dist + '/RECORD')
            data[dist + '/LINUX_PAYLOAD.json'] = json.dumps({
                'assembly_profile': 'release-split',
                'sets': {v + '/' + a: {} for v, a in SETS},
                'source_commit': ptx_commit, 'extensions': ptx_extensions,
                'split': {'role': profile},
            }).encode()
            wheels.append(pw.write_wheel(staged / wheel.name, {}, data, dist))
        output = staged / 'combined'
        output.mkdir()
        combined, vendors = qualification.split_combined(wheels, output)
        self.assertEqual(vendors, ('cuda', 'hip'))
        # the rebuilt combined wheel keeps the combined layout's name, not mojolearn_core's
        self.assertEqual(combined.name, f"mojolearn-{self.version}-{TAG}.whl")
        self.assertIn(f"mojolearn-{self.version}.dist-info/LINUX_PAYLOAD.json", members(combined))
        expected = {n: b for w in self.split.values() for n, b in members(w).items() if '.dist-info/' not in n}
        self.assertEqual({n: b for n, b in members(combined).items() if '.dist-info/' not in n}, expected)
        # the file-free mojolearn adds no member: the set without it rebuilds the same payload
        without_meta = [w for w in wheels if not w.name.startswith(f'{pw.gpu_plugins.META_WHEEL_NAME}-')]
        self.assertEqual(len(without_meta), len(wheels) - 1)
        output2 = staged / 'combined-no-meta'
        output2.mkdir()
        combined2, _ = qualification.split_combined(without_meta, output2)
        self.assertEqual({n: b for n, b in members(combined2).items() if '.dist-info/' not in n}, expected)
        incomplete = [w for w in wheels if not w.name.startswith('mojolearn_nvidia-')]
        with self.assertRaisesRegex(ValueError, 'every native vendor package'):
            qualification.split_combined(incomplete, output)

    def test_file_names_and_tags(self):
        self.assertEqual(sorted(p.name for p in self.split.values()), sorted([
            f"mojolearn_core-{self.version}-{TAG}.whl",
            f"mojolearn-{self.version}-{TAG}.whl",
            f"mojolearn_nvidia-{self.version}-{TAG}.whl",
            f"mojolearn_nvidia_ptx-{self.version}-{TAG}.whl",
            f"mojolearn_amd-{self.version}-{TAG}.whl",
]))
        for whl in self.split.values():
            wheel = meta(whl, "WHEEL")
            self.assertEqual(wheel.get_all("Tag"), [TAG])
            self.assertEqual(wheel.get("Root-Is-Purelib"), "false")
        # the combined profile still writes linux_x86_64 for audit.sh to retag
        self.assertTrue(self.single.name.endswith("-py3-none-linux_x86_64.whl"))

    def test_each_plugin_holds_exactly_its_sets_and_the_core_none(self):
        self.assertEqual(dist_info(self.split["mojolearn_core"]), f"mojolearn_core-{self.version}.dist-info")
        core = [n for n in members(self.split["mojolearn_core"]) if ".dist-info/" not in n]
        # the Linux mojolearn wheel is metadata only
        self.assertEqual([n for n in members(self.split["mojolearn"]) if ".dist-info/" not in n], [])
        self.assertEqual(dist_info(self.split["mojolearn"]), f"mojolearn-{self.version}.dist-info")
        self.assertFalse([n for n in core if n.startswith(("mojolearn/cuda/", "mojolearn/hip/"))])
        self.assertIn("mojolearn/__init__.py", core)
        self.assertIn("mojolearn/gpu_plugins.py", core)
        self.assertTrue(any(n.startswith("mojolearn/host/") for n in core))
        self.assertTrue(any(n.startswith("mojolearn/.libs/") for n in core))
        for row in pw.gpu_plugins.PAYLOADS.values():
            payload = [n for n in members(self.split[row["wheel_name"]]) if ".dist-info/" not in n]
            self.assertTrue(payload)
            dirs = {row["directory"]} | ({row["ptx"]["directory"]} if row.get("ptx") else set())
            self.assertTrue(all(n.split("/")[1] in dirs for n in payload))
            self.assertFalse([n for n in payload if n.endswith(".py") or "/.libs/" in n])
            native = sorted({n.split("/")[2] for n in payload if n.split("/")[1] == row["directory"]})
            self.assertTrue(pw.gpu_plugins.valid_arches(row["profile"], native))
            if row.get("ptx"):
                self.assertFalse(any(n.startswith(pw.gpu_plugins.BUNDLED_PTX_ROOT + "/") for n in payload))
        self.assertEqual(wheel_api_audit.split_audit(list(self.split.values()))["problems"], [])

    def test_upload_budget_is_pypis_100_mib_for_every_project(self):
        import os
        original = Path.stat
        for key in ("mojolearn_nvidia", "mojolearn_amd", "mojolearn_core", "mojolearn"):
            self.assertEqual(pw.gpu_plugins.wheel_size_limit(key.replace("_", "-")), 100 * 1024**2)
        for key, mib, accepted in (("mojolearn_nvidia", 99, True),
                                   ("mojolearn_nvidia", 101, False),
                                   ("mojolearn_nvidia", 250, False),
                                   ("mojolearn_amd", 101, False),
                                   ("mojolearn_core", 101, False),
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

    def test_metadata_core_pins_no_vendor_vendors_pin_the_core_meta_pins_all(self):
        v = self.version
        core = meta(self.split["mojolearn_core"])
        self.assertEqual((core.get("Name"), core.get("Version")), ("mojolearn-core", v))
        # 2026-10-20: the core requires NO vendor package, so core alone,
        # core + one vendor and core + both are each a valid install
        self.assertEqual(core.get_all("Provides-Extra"), ["numpy", "verify"])
        self.assertIn('numpy>=1.26.4; extra == "verify"', core.get_all("Requires-Dist"))
        self.assertNotIn('numpy>=1.26.4', core.get_all("Requires-Dist"))
        self.assertEqual([r for r in core.get_all("Requires-Dist") or [] if r.startswith("mojolearn")], [])
        self.assertEqual(pw.gpu_plugins.core_requirements(v), [])
        for key, name in (("mojolearn_nvidia", "mojolearn-nvidia"), ("mojolearn_amd", "mojolearn-amd")):
            m = meta(self.split[key])
            self.assertEqual((m.get("Name"), m.get("Version")), (name, v))
            expected = [f"mojolearn-core=={v}"] + ([f"mojolearn-nvidia-ptx=={v}"] if name == "mojolearn-nvidia" else [])
            self.assertEqual(m.get_all("Requires-Dist"), expected)
            self.assertEqual(m.get("Requires-Python"), core.get("Requires-Python"))
            self.assertEqual(m.get("License-Expression"), core.get("License-Expression"))
            self.assertIn(f"{dist_info(self.split[key])}/licenses/LICENSE", members(self.split[key]))
        self.assertEqual(meta(self.split["mojolearn_nvidia_ptx"]).get_all("Requires-Dist"), [f"mojolearn-core=={v}"])
        # THE LINUX mojolearn: `pip install mojolearn` installs the core and both
        # vendors, exactly at its version; no extras
        agg = meta(self.split["mojolearn"])
        self.assertEqual((agg.get("Name"), agg.get("Version")), ("mojolearn", v))
        self.assertEqual(agg.get_all("Requires-Dist"),
                         [f"mojolearn-core=={v}", f"mojolearn-nvidia=={v}", f"mojolearn-amd=={v}"])
        self.assertEqual(agg.get_all("Requires-Dist"), pw.gpu_plugins.meta_requirements(v))
        self.assertIsNone(agg.get_all("Provides-Extra"))
        self.assertEqual(agg.get("Requires-Python"), core.get("Requires-Python"))
        self.assertEqual(agg.get("License-Expression"), core.get("License-Expression"))
        self.assertIn(f"{dist_info(self.split['mojolearn'])}/licenses/LICENSE", members(self.split["mojolearn"]))
        # the split core's METADATA is the combined wheel's under the name
        # mojolearn-core, with its own Summary and README note, and nothing
        # else added, removed or reordered (no vendor requirement)
        single = meta(self.single)
        self.assertEqual(single.get_all("Provides-Extra"), ["numpy", "verify"])
        ours = members(self.split["mojolearn_core"])[f"{dist_info(self.split['mojolearn_core'])}/METADATA"].decode()
        theirs = members(self.single)[f"{dist_info(self.single)}/METADATA"].decode()
        self.assertNotIn("Requires-Dist: mojolearn", theirs)
        self.assertNotIn("Requires-Dist: mojolearn", ours)
        ours_head, ours_body = ours.split("\n\n", 1)
        theirs_head, theirs_body = theirs.split("\n\n", 1)
        self.assertEqual(ours_body, pw.CORE_README_NOTE + theirs_body)
        renamed = ("Name: ", "Summary: ")
        self.assertEqual([ln for ln in ours_head.split("\n") if not ln.startswith(renamed)],
                         [ln for ln in theirs_head.split("\n") if not ln.startswith(renamed)])
        self.assertIn("Name: mojolearn-core", ours_head.split("\n"))
        self.assertIn("Name: mojolearn", theirs_head.split("\n"))

    def test_each_package_publishes_only_after_its_dependencies_are_on_the_index(self):
        """wheel_api_audit.plugins_on_index, the workflow's gate before each
        upload (2026-10-20 order: core, then PTX and AMD, then NVIDIA, then the
        Linux mojolearn): every mojolearn requirement of a wheel at its version
        must already be on the index."""
        v = self.version
        core, nvidia, ptx, amd, agg = (self.split[k] for k in (
            "mojolearn_core", "mojolearn_nvidia", "mojolearn_nvidia_ptx", "mojolearn_amd", "mojolearn"))
        asked = []

        def index(served):
            def files(where, project, version):
                asked.append((where, project, version))
                return served.get(project, [])
            return files
        def missing(wheels, served, **kw):
            problems = wheel_api_audit.plugins_on_index(wheels, "pypi", files=index(served),
                                                        sleep=kw.pop("sleep", lambda s: None), **kw)
            return [(p.split(": ", 1)[0], p.split(": ", 1)[1].split("==")[0]) for p in problems]
        # the core requires nothing: it publishes first, asking the index nothing
        self.assertEqual(missing([core], {}), [])
        self.assertEqual(asked, [])
        # each vendor package needs the core (and NVIDIA its PTX package)
        self.assertEqual(missing([ptx, amd, nvidia], {}), [
            (ptx.name, "mojolearn-core"), (amd.name, "mojolearn-core"),
            (nvidia.name, "mojolearn-core"), (nvidia.name, "mojolearn-nvidia-ptx")])
        with_core = {"mojolearn-core": [core.name]}
        self.assertEqual(missing([ptx, amd], with_core), [])
        self.assertEqual(missing([nvidia], with_core), [(nvidia.name, "mojolearn-nvidia-ptx")])
        self.assertEqual(missing([nvidia], dict(with_core, **{"mojolearn-nvidia-ptx": [ptx.name]})), [])
        # the Linux mojolearn publishes last: the core and both vendors first
        everything = {"mojolearn-core": [core.name], "mojolearn-nvidia": [nvidia.name],
                      "mojolearn-nvidia-ptx": [ptx.name], "mojolearn-amd": [amd.name]}
        asked.clear()
        self.assertEqual(missing([agg], everything), [])
        self.assertEqual([a[1] for a in asked], ["mojolearn-core", "mojolearn-nvidia", "mojolearn-amd"])
        self.assertIn(("pypi", "mojolearn-core", v), asked)
        vendors = {"mojolearn-nvidia": [nvidia.name], "mojolearn-amd": [amd.name]}
        for label, served, absent in (
                ("nothing", {}, ["mojolearn-core", "mojolearn-nvidia", "mojolearn-amd"]),
                ("core only", with_core, ["mojolearn-nvidia", "mojolearn-amd"]),
                ("vendors only", vendors, ["mojolearn-core"]),
                ("nvidia missing", dict(with_core, **{"mojolearn-amd": [amd.name]}), ["mojolearn-nvidia"]),
                ("another version", dict(everything, **{"mojolearn-nvidia": [nvidia.name.replace(v, "0.0.1")]}),
                 ["mojolearn-nvidia"]),
                ("an sdist only", dict(everything, **{"mojolearn-core": [f"mojolearn_core-{v}.tar.gz"]}),
                 ["mojolearn-core"]),
                ("the old core name", dict(everything, **{"mojolearn-core": [f"mojolearn-{v}-{TAG}.whl"]}),
                 ["mojolearn-core"])):
            with self.subTest(label):
                naps = []
                problems = wheel_api_audit.plugins_on_index([agg], "testpypi", files=index(served),
                                                            attempts=3, sleep=naps.append)
                self.assertEqual([p.split(": ", 1)[1].split("==")[0] for p in problems], absent)
                self.assertIn("is not on testpypi", problems[0])
                self.assertEqual(len(naps), 2 * len(absent))   # retried, the index may be propagating
        # the combined wheel (no marker) needs nothing
        self.assertEqual(missing([self.single], {}), [])

    def test_markers_and_records(self):
        gp = pw.gpu_plugins
        core = self.split["mojolearn_core"]
        self.assertEqual(json.loads(members(core)[f"{dist_info(core)}/{gp.CORE_MARKER}"]),
                         gp.core_marker(self.version))
        self.assertNotIn(f"{dist_info(core)}/{gp.PLUGIN_MARKER}", members(core))
        # the Linux mojolearn carries an aggregate marker naming the core and both vendors
        agg = self.split["mojolearn"]
        agg_marker = json.loads(members(agg)[f"{dist_info(agg)}/{gp.PLUGIN_MARKER}"])
        self.assertEqual(agg_marker, gp.meta_marker(self.version))
        self.assertEqual((agg_marker["role"], agg_marker["distribution"]), ("aggregate", "mojolearn"))
        self.assertEqual(agg_marker["requires"], gp.meta_requirements(self.version))
        self.assertNotIn(f"{dist_info(agg)}/{gp.CORE_MARKER}", members(agg))
        amd = self.split["mojolearn_amd"]
        self.assertEqual(json.loads(members(amd)[f"{dist_info(amd)}/{gp.PLUGIN_MARKER}"])["requires"],
                         f"mojolearn-core=={self.version}")
        nvidia = self.split["mojolearn_nvidia"]
        manifest = members(self.split['mojolearn_nvidia_ptx'])[gp.BUNDLED_PTX_ROOT + "/" + gp.BASELINE_MANIFEST]
        bundle = dict(manifest_sha256=hashlib.sha256(manifest).hexdigest())
        marker = json.loads(members(nvidia)[f"{dist_info(nvidia)}/{gp.PLUGIN_MARKER}"])
        self.assertEqual(marker, gp.plugin_marker("cuda", self.version, ["sm_89"], bundled_ptx=bundle))
        # THE MARKER BINDS THE PTX MANIFEST, and names the PTX slot; the native arches stay native
        self.assertEqual(marker["bundled_ptx"], bundle)
        self.assertEqual(marker["ptx"], dict(arch="sm_80", directory="cuda_ptx"))
        self.assertEqual(marker["arches"], ["sm_89"])
        # the digest is required only while the registry ships the slot (gpu_plugins ptx_shipped; off since
        # 2026-10-12 while the wheel is over PyPI's 100 MiB limit): without it the marker names the slot unbundled
        from unittest.mock import patch
        with patch.dict(gp.PLUGINS["cuda"], {"ptx_shipped": True}):
            with self.assertRaisesRegex(ValueError, "PTX manifest digest is required"):
                gp.plugin_marker("cuda", self.version, ["sm_89"])
        with patch.dict(gp.PLUGINS["cuda"], {"ptx_shipped": False}):
            bare = gp.plugin_marker("cuda", self.version, ["sm_89"])
            self.assertNotIn("bundled_ptx", bare)
            self.assertEqual(bare["ptx"], dict(arch="sm_80", directory="cuda_ptx"))
        with self.assertRaisesRegex(ValueError, "only the NVIDIA"):
            gp.plugin_marker("hip", self.version, ["gfx942"], bundled_ptx=bundle)
        for whl in self.split.values():
            files = members(whl)
            record = files[f"{dist_info(whl)}/RECORD"].decode().splitlines()
            self.assertEqual({r.split(",")[0] for r in record}, set(files))

    # ---- subsets and refusals -------------------------------------------------
    def test_the_nvidia_legs_alone_pack_the_core_and_the_nvidia_plugin(self):
        cuda_only = [s for s in self.set_dirs if s.endswith("/cuda")]
        wheels = self.pack(sets=cuda_only)
        # no Linux mojolearn without every vendor: it would require a package not packed
        self.assertEqual([p.name.split("-")[0] for p in wheels],
                         ["mojolearn_core", "mojolearn_nvidia", "mojolearn_nvidia_ptx"])
        self.assertEqual(wheel_api_audit.split_audit(wheels)["problems"], [])
        with self.assertRaises(SystemExit) as ctx:
            self.pack("--wheels", "core-linux,amd", sets=cuda_only)
        self.assertIn("no hip set", str(ctx.exception))
        with self.assertRaises(SystemExit) as ctx:
            self.pack("--wheels", pw.gpu_plugins.META_PROFILE, sets=cuda_only)
        self.assertIn("requires every vendor", str(ctx.exception))
        self.assertIn("no hip set", str(ctx.exception))
        only = self.pack("--wheels", "amd")
        self.assertEqual([p.name.split("-")[0] for p in only], ["mojolearn_amd"])
        only_meta = self.pack("--wheels", pw.gpu_plugins.META_PROFILE)
        self.assertEqual([p.name for p in only_meta], [f"mojolearn-{self.version}-{TAG}.whl"])

    def test_split_refuses_closures_that_differ(self):
        root = Path(tempfile.mkdtemp(dir=self.root))
        dirs = self.make_sets(root, libs_differ=True)
        out = root / "out"
        with self.assertRaises(SystemExit) as ctx:
            pw.main([a for s in dirs for a in ("--set", s)] + ["--out", str(out)], _gates=False)
        self.assertIn("ONE runtime closure", str(ctx.exception))
        self.assertFalse(list(out.glob("*.whl")) if out.exists() else [])

    def test_wheels_flag_needs_a_split_profile(self):
        with self.assertRaises(SystemExit):
            self.pack("--profile", "generic", "--wheels", "nvidia")

    def test_release_split_slots(self):
        # Andrew 2026-10-10: the PTX slot is a release set like sm_89 ... while the registry ships it
        # (ptx_shipped; off since 2026-10-12 while the wheel is over PyPI's 100 MiB limit)
        from unittest.mock import patch
        with patch.dict(pw.gpu_plugins.PLUGINS["cuda"], {"ptx_shipped": True}):
            self.assertEqual(pw.split_release_slots({"cuda"}), {("cuda", "sm_89"), ("cuda", "sm_80")})
        with patch.dict(pw.gpu_plugins.PLUGINS["cuda"], {"ptx_shipped": False}):
            self.assertEqual(pw.split_release_slots({"cuda"}), {("cuda", "sm_89")})
        self.assertEqual(pw.split_release_slots({"hip"}), {("hip", "gfx942")})
        self.assertEqual(pw.split_release_slots({"cuda", "hip"}), pw.RELEASE_SPLIT_SETS)
        shipped = {k for v in ("cuda", "hip") for k in pw.gpu_plugins.required_sets(v)}
        self.assertEqual(pw.RELEASE_SPLIT_SETS, shipped)
        self.assertTrue(shipped <= set(SETS))
        # one definition: the admission side's per-plugin share agrees
        import verify_linux_surface_qualification as surface
        for vendor in ("cuda", "hip"):
            slots = {"/".join(k) for k in pw.split_release_slots({vendor})}
            self.assertEqual(slots, set(surface.PLUGIN_ARCHES[vendor]))
            self.assertTrue(surface.plugin_arch_set_ok(vendor, slots))
        self.assertEqual(surface.plugin_arch_set_ok("cuda", {"cuda/sm_89"}),
                         ("cuda", "sm_80") not in pw.gpu_plugins.required_sets("cuda"))
        self.assertFalse(surface.plugin_arch_set_ok("cuda", {"cuda/sm_89", "cuda/sm_90a"}))
        self.assertFalse(surface.plugin_arch_set_ok("hip", {"hip/gfx942", "cuda/sm_89"}))
        # the rest of this test covers the shipped PTX slot (ptx_shipped; off since 2026-10-12, PyPI 100 MiB limit)
        with patch.dict(pw.gpu_plugins.PLUGINS["cuda"], {"ptx_shipped": True}):
            hip = pw.SetDir("hip", "gfx942", {}, {}, {}, {}, None)
            with self.assertRaises(SystemExit) as ctx:
                pw.release_inventory([hip], [], self.version, required=pw.split_release_slots({"cuda"}),
                                     profile=pw.RELEASE_SPLIT_PROFILE)
            self.assertIn("release-split requires the sets cuda/sm_80, cuda/sm_89", str(ctx.exception))
            # the PTX set is proved like any set: a release-split nvidia pack with the
            # sm_89 proof alone is short one proof
            sm89 = pw.SetDir("cuda", "sm_89", {"cuda/sm_89/a.so": Path(__file__)}, {}, {}, {}, None)
            sm80 = pw.SetDir("cuda", "sm_80", {"cuda/sm_80/a.so": Path(__file__)}, {}, {}, {}, None)
            with self.assertRaises(SystemExit) as ctx:
                pw.release_inventory([sm89, sm80], [Path("cuda-sm_89.json")], self.version,
                                     required=pw.split_release_slots({"cuda"}), profile=pw.RELEASE_SPLIT_PROFILE)
            self.assertIn("2 proof(s) needed, 1 given", str(ctx.exception))

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
        dirs = self.make_sets(root, sets=(("cuda", "sm_90a"), ("cuda", "sm_80")))
        with self.assertRaisesRegex(SystemExit, "requires one set per registered architecture slot"):
            self.pack("--wheels", "nvidia", sets=dirs)
        with self.assertRaisesRegex(SystemExit, "name each"):
            self.pack("--wheels", "nvidia-sm89", sets=dirs)
        with self.assertRaisesRegex(SystemExit, "name each"):
            self.pack("--wheels", "nvidia-ptx80", sets=dirs)

    def test_the_nvidia_wheel_requires_the_ptx_slot(self):
        from unittest.mock import patch
        root = Path(tempfile.mkdtemp(dir=self.root))
        dirs = self.make_sets(root, sets=(("cuda", "sm_89"),))
        # while the registry ships the slot (ptx_shipped; off since 2026-10-12, PyPI's 100 MiB limit)
        with patch.dict(pw.gpu_plugins.PLUGINS["cuda"], {"ptx_shipped": True}):
            with self.assertRaisesRegex(SystemExit, "the PTX slot cuda/sm_80 included"):
                self.pack("--wheels", "nvidia", sets=dirs)
            with self.assertRaisesRegex(SystemExit, "the PTX slot cuda/sm_80 included"):
                self.pack(sets=dirs)

    def test_the_ptx_slot_never_enters_the_combined_wheel(self):
        with self.assertRaisesRegex(SystemExit, "packs only into the split"):
            self.pack("--profile", "generic")

    def test_ptx_slot_packs_into_nvidia(self):
        self._check_ptx_slot(include_byte_lm=False)

    def test_ptx_slot_preserves_full_byte_lm_payload_and_readbacks(self):
        self._check_ptx_slot(include_byte_lm=True)

    def test_ptx_manifest_must_be_v2_clean_and_match_the_bytes(self):
        for label, override, why in (
                ("v1 schema", dict(schema="mojolearn.ptx-baseline.v1"), "invalid PTX set manifest"),
                ("old code format", dict(code_format="ptx-baseline"), "invalid PTX set manifest"),
                ("dirty source", dict(source_dirty=True), "clean source manifest"),
                ("wrong bytes", dict(files=[]), "does not match")):
            with self.subTest(label):
                root = Path(tempfile.mkdtemp(dir=self.root))
                dirs = self.make_sets(root, sets=(("cuda", "sm_89"), ("cuda", "sm_80")))
                write_ptx_manifest(root / "sets/cuda/sm_80", **override)
                with self.assertRaisesRegex(SystemExit, why):
                    self.pack("--wheels", "nvidia", sets=dirs)
        root = Path(tempfile.mkdtemp(dir=self.root))
        dirs = self.make_sets(root, sets=(("cuda", "sm_89"), ("cuda", "sm_80")))
        (root / "sets/cuda/sm_80" / pw.gpu_plugins.BASELINE_MANIFEST).unlink()
        with self.assertRaisesRegex(SystemExit, "requires its PTX_BASELINE.json"):
            self.pack("--wheels", "nvidia", sets=dirs)

    def test_an_oversize_wheel_fails_the_pack_with_its_payload_sizes(self):
        import contextlib
        import io
        import os
        original = Path.stat

        def stat(path, *args, **kwargs):
            result = original(path, *args, **kwargs)
            if Path(path).name.startswith("mojolearn_nvidia-") and str(path).endswith(".whl"):
                values = list(result)
                values[6] = 120 * 1024**2
                return os.stat_result(values)
            return result
        err = io.StringIO()
        # split_audit refuses an oversize wheel too (SystemExit); the size report
        # under test is the pack's own, so the audit is held clean here.
        with mock.patch.object(Path, "stat", stat), contextlib.redirect_stderr(err), \
                mock.patch.object(wheel_api_audit, "split_audit", return_value={"problems": []}), \
                contextlib.redirect_stdout(io.StringIO()):
            self.pack("--wheels", "nvidia")
        self.assertEqual(self.last_rc, 1)
        text = err.getvalue()
        self.assertIn("OVER PyPI's 100 MiB FILE LIMIT", text)
        self.assertIn("cuda_native/sm_89:", text)
        self.assertNotIn("cuda_ptx/sm_80:", text)
        report = json.loads(next(self.last_out.glob("SIZES-*.json")).read_text())
        row = next(iter(report["wheels"].values()))
        self.assertTrue(row["over_limit"])
        self.assertEqual(row["pypi_limit_bytes"], 100 * 1024**2)
        self.assertNotIn("cuda_ptx/sm_80", row["payloads"])
        self.assertIn("cuda_native/sm_89", row["payloads"])

    def test_native_generic_does_not_start_accepting_optional_byte_lm(self):
        root = Path(tempfile.mkdtemp(dir=self.root))
        dirs = self.make_sets(root, sets=(("cuda", "sm_89"),), include_byte_lm=True)
        with self.assertRaisesRegex(SystemExit, "undeclared or missing native payload"):
            self.pack("--wheels", "nvidia", sets=dirs)

    def _check_ptx_slot(self, include_byte_lm):
        root = Path(tempfile.mkdtemp(dir=self.root))
        self.make_sets(root, sets=(("cuda", "sm_89"),))
        dirs = self.make_sets(root, sets=(("cuda", "sm_80"),), include_byte_lm=include_byte_lm)
        adir = root / "sets" / "cuda" / "sm_80"
        wheels = self.pack("--wheels", "nvidia-ptx", sets=dirs)
        self.assertEqual(len(wheels), 1)
        self.assertTrue(wheels[0].name.startswith("mojolearn_nvidia_ptx-"))
        payload = members(wheels[0])
        root_ptx = pw.gpu_plugins.BUNDLED_PTX_ROOT + "/"
        self.assertIn(root_ptx + "PTX_BASELINE.json", payload)
        self.assertTrue(all(n.startswith((root_ptx, "mojolearn/cuda_native/sm_89/"))
                            for n in payload if ".dist-info/" not in n))
        for binary in [*adir.rglob("_mojolearn*.so"), *adir.rglob(MERGED_INDEX)]:
            rel = binary.relative_to(adir).as_posix()
            if not rel.startswith("host/"):
                self.assertEqual(payload[root_ptx + rel], binary.read_bytes())
        self.assertEqual(any(n.endswith("/" + MERGED_INDEX) for n in payload), self.MERGED)
        marker = json.loads(payload[f"{dist_info(wheels[0])}/{pw.gpu_plugins.PLUGIN_MARKER}"])
        self.assertEqual(marker["bundled_ptx"]["manifest_sha256"],
                         hashlib.sha256((adir / pw.gpu_plugins.BASELINE_MANIFEST).read_bytes()).hexdigest())
        self.assertEqual(wheel_api_audit.split_audit(wheels)["problems"], [])
        byte_member = root_ptx + "identical/_mojolearn_byte_lm.so"
        packed_doc = json.loads(payload[root_ptx + "PTX_BASELINE.json"])
        if self.MERGED:
            # the byte LM is a member of the identical index, once, and of no fast index
            self.assertNotIn(byte_member, payload)
            identical = json.loads(payload[root_ptx + "identical/" + MERGED_INDEX])["files"]
            fast = json.loads(payload[root_ptx + MERGED_INDEX])["files"]
            self.assertEqual(sum(m.count("_mojolearn_byte_lm") for m in identical.values()), int(include_byte_lm))
            self.assertFalse(any("_mojolearn_byte_lm" in m for m in fast.values()))
            self.assertEqual(any("_mojolearn_byte_lm" in row.get("members", []) for row in packed_doc["files"]),
                             include_byte_lm)
            self.assertEqual({row["file"] for row in packed_doc["indexes"]}, {MERGED_INDEX, "identical/" + MERGED_INDEX})
        else:
            self.assertEqual(byte_member in payload, include_byte_lm)
        if include_byte_lm:
            if not self.MERGED:
                self.assertEqual(payload[byte_member], (adir / "identical/_mojolearn_byte_lm.so").read_bytes())
                self.assertTrue(any(row["file"] == "identical/_mojolearn_byte_lm.so" for row in packed_doc["files"]))
            witness = adir / "readback.txt"
            original = witness.read_text()
            witness.write_text("\n".join(line for line in original.splitlines()
                                         if not line.startswith("identical _mojolearn_byte_lm ")) + "\n")
            with self.assertRaisesRegex(SystemExit, "incomplete release native readback"):
                self.pack("--wheels", "nvidia", sets=dirs)
            witness.write_text(original)
        self.assertEqual(pw.gpu_plugins.payload_requirements("cuda", self.version),
                         [f"mojolearn-nvidia-ptx=={self.version}"])
        write_ptx_manifest(adir, source="c" * 40)
        with self.assertRaisesRegex(SystemExit, "source commits differ"):
            self.pack("--wheels", "nvidia", sets=dirs)

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
        core, nvidia, amd, agg = (self.split[k] for k in ("mojolearn_core", "mojolearn_nvidia", "mojolearn_amd",
                                                          "mojolearn"))
        v = self.version
        ok = wheel_api_audit.split_audit([core, nvidia, amd])["problems"]
        self.assertEqual(ok, [])
        self.assertEqual(wheel_api_audit.split_audit([core, nvidia, amd, agg])["problems"], [])
        roles = {r["distribution"]: r["role"] for r in wheel_api_audit.split_audit([core, agg])["wheels"]}
        self.assertEqual(roles, {"mojolearn-core": pw.gpu_plugins.CORE_PROFILE,
                                 "mojolearn": pw.gpu_plugins.META_PROFILE})

        def edit_meta(whl, old, new):
            path = f"{dist_info(whl)}/METADATA"
            data = members(whl)[path]
            self.assertIn(old, data)
            return self.rewrite(whl, replace={path: data.replace(old, new, 1)})
        cases = {
            "core carries a set": [self.rewrite(core, add={"mojolearn/cuda/sm_89/x.so": b"x"}), nvidia, amd],
            "plugin carries python": [core, self.rewrite(nvidia, add={"mojolearn/cuda/x.py": b""}), amd],
            "plugin carries the other vendor": [core, self.rewrite(nvidia, add={"mojolearn/hip/gfx942/y.so": b""}), amd],
            "loose pin": [core, edit_meta(nvidia, f"mojolearn-core=={v}".encode(), b"mojolearn-core>=0.1"), amd],
            "vendor pins the old core name": [core, nvidia, edit_meta(
                amd, f"mojolearn-core=={v}".encode(), f"mojolearn=={v}".encode())],
            "a member in two wheels": [core, nvidia, self.rewrite(amd, add={"mojolearn/__init__.py": b""})],
            "core declares a GPU extra": [self.rewrite(core, replace={
                f"{dist_info(core)}/METADATA": members(core)[f"{dist_info(core)}/METADATA"].replace(
                    b"\nDynamic:", f'\nProvides-Extra: nvidia\nRequires-Dist: mojolearn-nvidia=={self.version}; '
                    f'extra == "nvidia"\nDynamic:'.encode(), 1)}), nvidia, amd],
            "core pins a vendor (the old graph)": [edit_meta(
                core, b"\nDynamic:", f"\nRequires-Dist: mojolearn-nvidia=={v}\nDynamic:".encode()), nvidia, amd],
            "core pins a vendor loosely": [edit_meta(
                core, b"\nDynamic:", b"\nRequires-Dist: mojolearn-amd>=0.1\nDynamic:"), nvidia, amd],
            "core lost its marker": [self.rewrite(core, drop={f"{dist_info(core)}/{pw.gpu_plugins.CORE_MARKER}"}),
                                     nvidia, amd],
            "meta carries a file": [core, nvidia, amd, self.rewrite(agg, add={"mojolearn/extra.py": b""})],
            "meta lacks amd": [core, nvidia, amd, edit_meta(agg, f"Requires-Dist: mojolearn-amd=={v}\n".encode(), b"")],
            "meta lacks the core": [core, nvidia, amd, edit_meta(
                agg, f"Requires-Dist: mojolearn-core=={v}\n".encode(), b"")],
            "meta pins loosely": [core, nvidia, amd, edit_meta(
                agg, f"mojolearn-nvidia=={v}".encode(), b"mojolearn-nvidia>=0.1")],
            "meta declares an extra": [core, nvidia, amd, edit_meta(
                agg, b"\nDynamic:", b"\nProvides-Extra: nvidia\nDynamic:")],
            "meta lost its marker": [core, nvidia, amd, self.rewrite(
                agg, drop={f"{dist_info(agg)}/{pw.gpu_plugins.PLUGIN_MARKER}"})],
        }
        why = {"core declares a GPU extra": "declares Provides-Extra ['numpy', 'nvidia', 'verify']",
               "core pins a vendor (the old graph)": "dependency pins disagree",
               "core pins a vendor loosely": "dependency pins disagree",
               "core lost its marker": "gpu_plugins.json is missing or disagrees",
               "vendor pins the old core name": "Requires-Dist differs from exact package dependencies",
               "meta carries a file": "the mojolearn meta wheel carries files",
               "meta lacks amd": "Requires-Dist differs from the core and both vendors",
               "meta lacks the core": "Requires-Dist differs from the core and both vendors",
               "meta pins loosely": "Requires-Dist differs from the core and both vendors",
               "meta declares an extra": "must not declare extras",
               "meta lost its marker": "is missing or disagrees with package ownership"}
        for label, wheels in cases.items():
            with self.subTest(label):
                problems = wheel_api_audit.split_audit(wheels)["problems"]
                self.assertTrue(problems, label)
                if label in why:
                    self.assertIn(why[label], "\n".join(problems))



class SplitWheelsMerged(SplitWheels):
    """Every SplitWheels test again, on sets in the merged layout."""
    MERGED = True


class MergedLayoutRefusals(unittest.TestCase):
    """What only a merged set can get wrong, each refused at pack time."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def pack(self, dirs, *extra):
        out = self.root / "out"
        return pw.main([a for d in dirs for a in ("--set", d)] + list(extra) + ["--out", str(out)], _gates=False)

    def native(self, merged=True, root=None):
        return make_sets(root or self.root, sets=NATIVE_SETS, merged=merged)

    def test_merged_generic_wheel_ships_files_and_indexes(self):
        dirs = self.native()
        self.assertEqual(self.pack(dirs, "--profile", "generic"), 0)
        names = set(members(next((self.root / "out").glob("*.whl"))))
        for vendor, arch in NATIVE_SETS:
            for sub in ("", "identical/"):
                self.assertIn(f"mojolearn/{vendor}/{arch}/{sub}_mojolearn_native.so", names)
                self.assertIn(f"mojolearn/{vendor}/{arch}/{sub}{MERGED_INDEX}", names)
                self.assertNotIn(f"mojolearn/{vendor}/{arch}/{sub}_mojolearn_gbdt.so", names)
        self.assertIn("mojolearn/host/_mojolearn_native_host.so", names)
        self.assertIn(f"mojolearn/host/{MERGED_INDEX}", names)
        self.assertFalse([n for n in names if n.startswith("mojolearn/host/") and n.endswith(".so")
                          and n != "mojolearn/host/_mojolearn_native_host.so"])

    def test_load_set_reads_the_layout_from_the_directory(self):
        dirs = self.native()
        for d in dirs:
            for s in pw.load_set(d):
                self.assertEqual(s.layout, "merged")
                self.assertEqual(sorted(s.indexes), sorted([MERGED_INDEX, "identical/" + MERGED_INDEX,
                                                            "host/" + MERGED_INDEX]))
                self.assertEqual({p.name for p in s.hosts.values()}, {"_mojolearn_native_host.so"})
                self.assertEqual(set(s.hosts), set(pw.HOST_NAMES))
        per = self.native(merged=False, root=self.root / "per")
        self.assertEqual({s.layout for d in per for s in pw.load_set(d)}, {"per-binding"})

    def test_sets_of_two_layouts_never_share_a_wheel(self):
        merged = make_sets(self.root, sets=(("cuda", "sm_89"),), merged=True)
        per = make_sets(self.root / "per", sets=(("hip", "gfx942"),), merged=False)
        with self.assertRaisesRegex(SystemExit, "mix the merged and per-binding layouts"):
            self.pack(merged + per, "--profile", "generic")

    def test_a_set_never_mixes_layouts_across_its_directories(self):
        dirs = self.native()
        ident = self.root / "sets/cuda/sm_89/identical"
        index = json.loads((ident / MERGED_INDEX).read_text())["files"]
        for f in list(index):
            (ident / f).unlink()
        (ident / MERGED_INDEX).unlink()
        for f, data in dir_files("cuda", "sm_89", "identical").items():
            (ident / f).write_bytes(data)
        with self.assertRaisesRegex(SystemExit, "mixes the merged and per-binding layouts"):
            self.pack(dirs, "--profile", "generic")

    def test_an_own_member_file_beside_the_index_is_refused(self):
        dirs = self.native()
        (self.root / "sets/cuda/sm_89/_mojolearn_gbdt.so").write_bytes(b"shadow")
        with self.assertRaisesRegex(SystemExit, "shadow the merged file"):
            self.pack(dirs, "--profile", "generic")

    def test_an_index_missing_a_member_is_refused(self):
        dirs = self.native()
        index = self.root / "sets/hip/gfx942/identical" / MERGED_INDEX
        doc = json.loads(index.read_text())
        doc["files"]["_mojolearn_native.so"].remove("_mojolearn_gbdt")
        index.write_text(json.dumps(doc))
        with self.assertRaisesRegex(SystemExit, "undeclared or missing native payload"):
            self.pack(dirs, "--profile", "generic")

    def test_an_index_off_the_manifest_plan_is_refused(self):
        dirs = self.native()
        fast = self.root / "sets/cuda/sm_89"
        doc = json.loads((fast / MERGED_INDEX).read_text())
        names = doc["files"].pop("_mojolearn_native.so")
        doc["files"] = {"_mojolearn_native.so": names[:1], "_mojolearn_native_b.so": names[1:]}
        (fast / MERGED_INDEX).write_text(json.dumps(doc))
        (fast / "_mojolearn_native_b.so").write_bytes(b"second group")
        with self.assertRaisesRegex(SystemExit, "merged_groups.json plan"):
            self.pack(dirs, "--profile", "generic")

    def test_host_index_must_be_byte_identical_across_sets(self):
        dirs = self.native()
        index = self.root / "sets/hip/gfx942/host" / MERGED_INDEX
        index.write_text(json.dumps(json.loads(index.read_text())))  # same content, other bytes
        with self.assertRaisesRegex(SystemExit, "merged host index"):
            self.pack(dirs, "--profile", "generic")

    def test_a_host_member_never_read_back_is_refused(self):
        dirs = self.native()
        adir = self.root / "sets/cuda/sm_89"
        dropped = sorted(pw.HOST_NAMES)[0]
        for witness in ("readback.txt", "arch_readback.txt"):
            path = adir / witness
            path.write_text("".join(ln + "\n" for ln in path.read_text().splitlines()
                                    if not ln.startswith(f"host {dropped} ")))
        with self.assertRaisesRegex(SystemExit, "undeclared or missing native payload"):
            self.pack(dirs, "--profile", "generic")


if __name__ == "__main__":
    unittest.main()
