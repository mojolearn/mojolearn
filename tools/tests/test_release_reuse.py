# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""tools/release_reuse.py and the packer's reuse.json path, CPU only, no
compiler, no network: the identity of a binding moves when a source in its
closure, its toolchain, its flags or its box image moves and not otherwise;
the decision is REUSE only on an exact identity match; the assembled set
records what was taken and from where; a byte that differs from the published
one is refused by the assembler and by the packer.

    python3 -m unittest tools.tests.test_release_reuse
"""
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "packaging" / "linux"))
import release_reuse as rr  # noqa: E402
import pack_wheel  # noqa: E402

PREV_COMMIT = "b" * 40
CUR_COMMIT = "c" * 40


def sha(data):
    return hashlib.sha256(data).hexdigest()


def tiny_repo(d):
    """A repository with two bindings (one host, one GPU), the builder scripts
    and pins the identities read, committed twice: the previous release and a
    Python-only patch on top."""
    d = pathlib.Path(d)
    (d / "bindings").mkdir()
    (d / "tools").mkdir()
    (d / "packaging" / "linux").mkdir(parents=True)
    (d / "packaging" / "macos").mkdir(parents=True)
    (d / "python" / "mojolearn").mkdir(parents=True)
    (d / "k.mojo").write_text("fn f() -> Int:\n    return 1\n")
    (d / "h.mojo").write_text("fn g() -> Int:\n    return 2\n")
    (d / "bindings" / "build_svm.sh").write_text("mojo build k.mojo -o python/mojolearn/identical/_mojolearn_svm.so\n")
    (d / "bindings" / "build_svm_host.sh").write_text("mojo build h.mojo -o python/mojolearn/host/_mojolearn_svm_host.so\n")
    for rel in rr.LINUX_BUILDERS + rr.MACOS_BUILDERS:
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_text("# builder " + rel + "\n")
    (d / "tools" / "gemm_remote_leg.sh").write_text(
        'LEG_IMAGE_NVIDIA="${MOJOLEARN_GEMM_LEG_IMAGE_NVIDIA:-runpod/pytorch:2.4.0-ubuntu22.04}"\n')
    (d / "tools" / "release_ubuntu22_build.sh").write_text("IMAGE=rocm/dev-ubuntu-22.04@sha256:abc\n")
    (d / "pixi.toml").write_text('[dependencies]\nmojo = "1.0.0"\n')
    (d / "pixi.lock").write_text(
        "- conda: https://conda.modular.com/max/linux-64/mojo-1.0.0-release.conda\n"
        "- conda: https://conda.modular.com/max/linux-64/max-26.5.0-release.conda\n"
        "- conda: https://conda.modular.com/max/osx-arm64/mojo-1.0.0-release.conda\n"
        "- conda: https://conda.modular.com/max/osx-arm64/max-26.5.0-release.conda\n")
    (d / "python" / "mojolearn" / "__init__.py").write_text("x = 1\n")
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@t")
    subprocess.run(["git", "init", "-q", str(d)], check=True, env=env)
    subprocess.run(["git", "-C", str(d), "add", "-A"], check=True, env=env)
    subprocess.run(["git", "-C", str(d), "commit", "-q", "-m", "previous release"], check=True, env=env)
    prev = subprocess.run(["git", "-C", str(d), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    (d / "python" / "mojolearn" / "__init__.py").write_text("x = 2\n")
    subprocess.run(["git", "-C", str(d), "commit", "-q", "-am", "python only"], check=True, env=env)
    cur = subprocess.run(["git", "-C", str(d), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    return d, prev, cur, env


def two_bindings():
    gpu = rr.Binding(rr.LINUX, "cuda", "sm_89", "identical", "_mojolearn_svm")
    host = rr.Binding(rr.LINUX, "host", "", "host", "_mojolearn_svm_host")
    mac = rr.Binding(rr.MACOS, "metal", "apple", "identical", "_mojolearn_svm")
    return gpu, host, mac


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.repo, self.prev, self.cur, self.env = tiny_repo(self._t.name)
        self.cache = pathlib.Path(self._t.name) / "cache"

    def tearDown(self):
        self._t.cleanup()

    def test_identity_is_a_function_of_the_commit_and_python_does_not_move_it(self):
        gpu, host, mac = two_bindings()
        a = rr.facts_for_commit(self.prev, self.cache, self.repo)
        b = rr.facts_for_commit(self.cur, self.cache, self.repo)
        for binding in (gpu, host):
            self.assertEqual(rr.compare(rr.identity(binding, b), rr.identity(binding, a)), ("REUSE", "identity unchanged"))
        # the cache serves the same facts
        self.assertEqual(rr.facts_for_commit(self.cur, self.cache, self.repo), b)
        # a macOS binding needs the Apple toolchain on both sides
        self.assertEqual(rr.compare(rr.identity(mac, b, dict(xcode="16")), rr.identity(mac, a, None))[0], "BUILD")
        self.assertIn("no record of the Apple toolchain",
                      rr.compare(rr.identity(mac, b, dict(xcode="16")), rr.identity(mac, a, None))[1])
        self.assertEqual(rr.compare(rr.identity(mac, b, dict(xcode="16")), rr.identity(mac, a, dict(xcode="16"))),
                         ("REUSE", "identity unchanged"))
        self.assertEqual(rr.compare(rr.identity(mac, b, dict(xcode="17")), rr.identity(mac, a, dict(xcode="16"))),
                         ("BUILD", "changed: host_toolchain"))

    def test_a_source_a_flag_a_toolchain_or_an_image_moves_the_identity(self):
        gpu, host, _ = two_bindings()
        a = rr.facts_for_commit(self.prev, self.cache, self.repo)
        (self.repo / "k.mojo").write_text("fn f() -> Int:\n    return 3\n")
        subprocess.run(["git", "-C", str(self.repo), "commit", "-q", "-am", "kernel"], check=True, env=self.env)
        b = rr.facts_for_commit("HEAD", self.cache, self.repo)
        self.assertEqual(rr.compare(rr.identity(gpu, b), rr.identity(gpu, a)), ("BUILD", "changed: closure"))
        self.assertEqual(rr.compare(rr.identity(host, b), rr.identity(host, a))[0], "REUSE")
        # a different flag set is a different binding even from one tree
        other = rr.Binding(rr.LINUX, "hip", "gfx942", "identical", "_mojolearn_svm")
        self.assertNotEqual(rr.digest_of(rr.identity(other, b)), rr.digest_of(rr.identity(gpu, b)))
        # the toolchain: every closure carries pixi.lock, and the packages are named too
        (self.repo / "pixi.lock").write_text(
            "- conda: https://conda.modular.com/max/linux-64/mojo-1.1.0-release.conda\n"
            "- conda: https://conda.modular.com/max/osx-arm64/mojo-1.1.0-release.conda\n")
        subprocess.run(["git", "-C", str(self.repo), "commit", "-q", "-am", "toolchain"], check=True, env=self.env)
        c = rr.facts_for_commit("HEAD", self.cache, self.repo)
        self.assertEqual(rr.compare(rr.identity(host, c), rr.identity(host, a)), ("BUILD", "changed: closure, toolchain"))
        # the box image
        (self.repo / "tools" / "release_ubuntu22_build.sh").write_text("IMAGE=rocm/dev-ubuntu-22.04@sha256:def\n")
        subprocess.run(["git", "-C", str(self.repo), "commit", "-q", "-am", "image"], check=True, env=self.env)
        d = rr.facts_for_commit("HEAD", self.cache, self.repo)
        self.assertEqual(rr.compare(rr.identity(host, d), rr.identity(host, c)), ("BUILD", "changed: image"))
        self.assertEqual(rr.compare(rr.identity(gpu, d), rr.identity(gpu, c)), ("REUSE", "identity unchanged"))
        # an unreadable pin builds
        (self.repo / "tools" / "release_ubuntu22_build.sh").write_text("# no pin\n")
        subprocess.run(["git", "-C", str(self.repo), "commit", "-q", "-am", "unpinned"], check=True, env=self.env)
        e = rr.facts_for_commit("HEAD", self.cache, self.repo)
        self.assertEqual(rr.compare(rr.identity(host, e), rr.identity(host, d)),
                         ("BUILD", "unreadable identity: box image pin unreadable"))
        # a binding with no script in the previous tree is new
        self.assertEqual(rr.compare(rr.identity(gpu, e), None)[0], "BUILD")
        self.assertIn("new binding", rr.compare(rr.identity(gpu, e), None)[1])

    def test_runtime_identity_follows_toolchain_and_portable_math(self):
        a = rr.facts_for_commit(self.prev, self.cache, self.repo)
        (self.repo / "packaging" / "portable_math").mkdir()
        (self.repo / "packaging" / "portable_math" / "m.c").write_text("int x;\n")
        subprocess.run(["git", "-C", str(self.repo), "add", "-A"], check=True, env=self.env)
        subprocess.run(["git", "-C", str(self.repo), "commit", "-q", "-m", "math"], check=True, env=self.env)
        b = rr.facts_for_commit("HEAD", self.cache, self.repo)
        self.assertNotEqual(rr.digest_of(rr.runtime_identity(a)), rr.digest_of(rr.runtime_identity(b)))
        self.assertEqual(rr.digest_of(rr.runtime_identity(b)), rr.digest_of(rr.runtime_identity(
            rr.facts_for_commit("HEAD", self.cache, self.repo))))


class DecisionTests(unittest.TestCase):
    """make_plan on the real binding lists, with the previous release's
    identities supplied from a record so no wheel or tree is needed."""

    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.repo, self.prev, self.cur, self.env = tiny_repo(self._t.name)
        self.cache = pathlib.Path(self._t.name) / "cache"
        # every real binding's script, so the plan has a full set of identities
        for target in (rr.LINUX, rr.MACOS):
            for b in rr.bindings(target):
                p = self.repo / "bindings" / b.script
                if not p.exists():
                    p.write_text(f"mojo build k.mojo -o python/mojolearn/x/{b.name}.so\n")
        subprocess.run(["git", "-C", str(self.repo), "add", "-A"], check=True, env=self.env)
        subprocess.run(["git", "-C", str(self.repo), "commit", "-q", "-m", "all scripts"], check=True, env=self.env)
        self.cur = subprocess.run(["git", "-C", str(self.repo), "rev-parse", "HEAD"], capture_output=True,
                                  text=True).stdout.strip()

    def tearDown(self):
        self._t.cleanup()

    def record(self, commit, host_toolchain, mutate=None):
        """binding-identities.json of a previous release at `commit`."""
        facts = rr.facts_for_commit(commit, self.cache, self.repo)
        rows = []
        for target in (rr.LINUX, rr.MACOS):
            for b in rr.bindings(target):
                ident = rr.identity(b, facts, host_toolchain)
                if mutate:
                    ident = mutate(b, ident)
                rows.append(dict(key=b.key, identity=ident))
        path = pathlib.Path(self._t.name) / "binding-identities.json"
        path.write_text(json.dumps(dict(rows=rows, runtime=dict(identity=rr.runtime_identity(facts)))))
        return dict(version="0.8.15", source_commit=commit, identities=str(path),
                    linux=dict(wheel="mojolearn-0.8.15-py3-none-manylinux_2_35_x86_64.whl", sha256="0" * 64),
                    macos=dict(wheel="mojolearn-0.8.15-py3-none-macosx_11_0_arm64.whl", sha256="1" * 64))

    def test_python_only_release_rents_nothing(self):
        prev = self.record(self.cur, dict(xcode="16"))
        plan = rr.make_plan(self.cur, self.cache, self.repo, host_toolchain=dict(xcode="16"), prev=prev)
        self.assertEqual(plan["legs"], [])
        self.assertEqual({r["decision"] for r in plan["rows"]}, {"REUSE"})
        self.assertEqual(plan["runtime"]["decision"], "REUSE")
        text = rr.render(plan, full=True)
        self.assertIn("legs to launch: none", text)
        self.assertIn("REUSE cuda/sm_90a", text.replace("  ", " ").replace("  ", " "))

    def test_one_changed_binding_launches_its_set_only_and_takes_the_rest(self):
        def mutate(b, ident):
            if b.name == "_mojolearn_gbdt" and (b.vendor, b.arch, b.tier) == ("hip", "gfx942", "identical"):
                ident = dict(ident, closure=dict(scope="closure", digest="9" * 64))
            return ident
        prev = self.record(self.cur, dict(xcode="16"), mutate)
        plan = rr.make_plan(self.cur, self.cache, self.repo, host_toolchain=dict(xcode="16"), prev=prev)
        self.assertEqual(plan["legs"], ["hip-gfx942"])
        builds = rr.plan_rows(plan, decision="BUILD")
        self.assertEqual([(r["vendor"], r["arch"], r["tier"], r["name"]) for r in builds],
                         [("hip", "gfx942", "identical", "_mojolearn_gbdt")])
        self.assertEqual(builds[0]["reason"], "changed: closure")
        self.assertEqual(len(rr.plan_rows(plan, rr.LINUX, "REUSE")), 118)

    def test_host_only_change_takes_the_cheapest_leg(self):
        def mutate(b, ident):
            return dict(ident, flags=dict(ident["flags"], compile_jobs="9")) if b.name == "_mojolearn_core_host" else ident
        prev = self.record(self.cur, dict(xcode="16"), mutate)
        plan = rr.make_plan(self.cur, self.cache, self.repo, host_toolchain=dict(xcode="16"), prev=prev)
        self.assertEqual(plan["legs"], ["cuda-sm_89"])
        self.assertIn("host binding", plan["leg_reasons"]["cuda-sm_89"])
        self.assertEqual(len(rr.plan_rows(plan, rr.LINUX, "BUILD")), 1)
        self.assertEqual(len(rr.plan_rows(plan, rr.MACOS, "BUILD")), 1)

    def test_no_record_or_unknown_apple_toolchain_builds(self):
        plan = rr.make_plan(self.cur, self.cache, self.repo, host_toolchain=dict(xcode="16"), prev=None)
        self.assertEqual({r["decision"] for r in plan["rows"]}, {"BUILD"})
        self.assertEqual(plan["legs"], ["cuda-sm_90a", "cuda-sm_89", "hip-gfx942"])
        prev = self.record(self.cur, None)
        plan = rr.make_plan(self.cur, self.cache, self.repo, host_toolchain=dict(xcode="16"), prev=prev)
        self.assertEqual({r["decision"] for r in rr.plan_rows(plan, rr.LINUX)}, {"REUSE"})
        self.assertEqual({r["decision"] for r in rr.plan_rows(plan, rr.MACOS)}, {"BUILD"})
        self.assertEqual(plan["legs"], [])


def elf_stub(needed=()):
    """A minimal ELF64 little-endian file with a dynamic section naming
    `needed`, enough for stage_libs.elf_dynamic (the assembler reads the
    driver libraries a reused set needs from the published bytes)."""
    import struct
    strtab = b"\0" + b"".join(n.encode() + b"\0" for n in needed)
    offsets, o = [], 1
    for n in needed:
        offsets.append(o)
        o += len(n) + 1
    dyn = b"".join(struct.pack("<qQ", 1, off) for off in offsets) + struct.pack("<qQ", 0, 0)
    ehdr_size, shentsize = 64, 64
    strtab_off = ehdr_size
    dyn_off = strtab_off + len(strtab)
    shoff = dyn_off + len(dyn)
    # sections: null, strtab (index 1), dynamic (index 2, link 1)
    sh = struct.pack("<IIQQQQIIQQ", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    sh += struct.pack("<IIQQQQIIQQ", 0, 3, 0, 0, strtab_off, len(strtab), 0, 0, 1, 0)
    sh += struct.pack("<IIQQQQIIQQ", 0, 6, 0, 0, dyn_off, len(dyn), 1, 0, 8, 16)
    ehdr = b"\x7fELF" + bytes([2, 1, 1, 0]) + b"\0" * 8
    ehdr += struct.pack("<HHIQQQIHHHHHH", 3, 62, 1, 0, 0, shoff, 0, 64, 56, 0, shentsize, 3, 1)
    assert len(ehdr) == 64
    return ehdr + strtab + dyn + sh


HOST_BYTES = elf_stub(["libc.so.6"])


def published_wheel_fixture(tmp):
    """A 0.8.15 Linux wheel of every real binding (ELF stubs) with its
    LINUX_PAYLOAD.json, and the record entry that names it."""
    payload = dict(schema="mojolearn.linux-payload.v1", source_commit=PREV_COMMIT, runtime_layout="shared",
                   extensions={}, host_native={}, runtime_sha256={})
    whl = tmp / "mojolearn-0.8.15-py3-none-manylinux_2_35_x86_64.whl"
    with zipfile.ZipFile(whl, "w") as z:
        for vendor, arch in rr.LINUX_SETS:
            for tier in pack_wheel.TIERS:
                for name in pack_wheel.tier_names(tier, True):
                    rel = f"{name}.so" if tier == "fast" else f"{tier}/{name}.so"
                    member = f"mojolearn/{vendor}/{arch}/{rel}"
                    data = elf_stub(["libcuda.so.1" if vendor == "cuda" else "libamdhip64.so.6", "libc.so.6"]) + member.encode()
                    z.writestr(member, data)
                    payload["extensions"][member] = sha(data)
        for name in pack_wheel.HOST_NAMES:
            member = f"mojolearn/host/{name}.so"
            data = HOST_BYTES + name.encode()
            z.writestr(member, data)
            payload["host_native"][name] = dict(archive_path=member, sha256=sha(data))
        z.writestr("mojolearn/.libs/libAsyncRTMojoBindings.so", b"runtime")
        payload["runtime_sha256"]["mojolearn/.libs/libAsyncRTMojoBindings.so"] = sha(b"runtime")
        z.writestr("mojolearn-0.8.15.dist-info/LINUX_PAYLOAD.json", json.dumps(payload))
    prev = dict(version="0.8.15", source_commit=PREV_COMMIT, linux=dict(wheel=whl.name, sha256=rr.sha256(whl)))
    return whl, prev


def linux_plan(prev, build=()):
    rows = []
    for b in rr.bindings(rr.LINUX):
        rows.append(dict(b.row(), decision="BUILD" if b.key in build else "REUSE", reason="test",
                         identity_digest="d" * 64))
    legs = sorted({f"{r['vendor']}-{r['arch']}" for r in rows if r["decision"] == "BUILD" and r["tier"] != "host"})
    return dict(schema=rr.PLAN_SCHEMA, commit=CUR_COMMIT, previous=prev, rows=rows, legs=legs,
                leg_reasons={}, runtime=dict(decision="REUSE"))


class AssemblyTests(unittest.TestCase):
    """A published wheel, the assembler's set directories and the packer's
    reading of them."""

    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.tmp = pathlib.Path(self._t.name)
        self.host_bytes = HOST_BYTES
        self.whl, self.prev = published_wheel_fixture(self.tmp)

    def tearDown(self):
        self._t.cleanup()

    def plan(self, build=()):
        return linux_plan(self.prev, build)

    def test_everything_reused_synthesizes_every_set_with_a_record(self):
        out = rr.assemble_linux(self.plan(), self.whl, {}, self.tmp / "sets", say=lambda *_: None)
        self.assertEqual(set(out), {"cuda/sm_90a", "cuda/sm_89", "hip/gfx942"})
        for key, sdir in out.items():
            doc = json.loads((sdir / "reuse.json").read_text())
            self.assertEqual(doc["schema"], rr.REUSE_SCHEMA)
            self.assertEqual(doc["set"], key)
            self.assertEqual(doc["from_release"]["wheel_sha256"], self.prev["linux"]["sha256"])
            self.assertEqual(len(doc["files"]), 29 + len(pack_wheel.HOST_NAMES) + 1)
            for rel, rec in doc["files"].items():
                self.assertEqual(rr.sha256(sdir / rel), rec["sha256"])
            manifest = json.loads((sdir / "manifest.json").read_text())
            self.assertEqual(manifest["driver_libs_not_staged"],
                             ["libcuda.so.1"] if key.startswith("cuda") else ["libamdhip64.so.6"])
            self.assertEqual([s["name"] for s in manifest["staged_libs"]], ["libAsyncRTMojoBindings.so"])
            self.assertFalse((sdir / "readback.txt").exists())
            # the packer reads it: every reused file verified, no witnesses needed
            loaded = pack_wheel.load_set(sdir.parent, include_byte_lm=True)
            s = [t for t in loaded if t.arch == sdir.name][0]
            self.assertEqual(len(s.reuse["files"]), len(doc["files"]))
            self.assertEqual(s.reuse["origin"]["version"], "0.8.15")

    def test_a_leg_set_keeps_its_bytes_for_build_rows_and_takes_the_rest(self):
        key = "linux-64/hip/gfx942/identical/_mojolearn_gbdt"
        plan = self.plan(build={key})
        leg = self.tmp / "leg" / "hip" / "gfx942"
        for r in rr.plan_rows(plan, rr.LINUX):
            if r["tier"] == "host" or (r["vendor"], r["arch"]) != ("hip", "gfx942"):
                continue
            p = leg / r["set_rel"]
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"LEG " + r["set_rel"].encode())
        for name in pack_wheel.HOST_NAMES:
            (leg / "host").mkdir(exist_ok=True)
            (leg / "host" / f"{name}.so").write_bytes(self.host_bytes + name.encode())
        (leg / ".libs").mkdir()
        (leg / ".libs" / "libAsyncRTMojoBindings.so").write_bytes(b"runtime")
        (leg / "manifest.json").write_text("{}")
        (leg / "readback.txt").write_text("identical _mojolearn_gbdt hip\n")
        (leg / "arch_readback.txt").write_text("identical _mojolearn_gbdt gfx942\n")
        lines = []
        out = rr.assemble_linux(plan, self.whl, {"hip-gfx942": leg}, self.tmp / "sets", say=lines.append)
        sdir = out["hip/gfx942"]
        self.assertEqual((sdir / "identical" / "_mojolearn_gbdt.so").read_bytes(), b"LEG identical/_mojolearn_gbdt.so")
        doc = json.loads((sdir / "reuse.json").read_text())
        self.assertNotIn("identical/_mojolearn_gbdt.so", doc["files"])
        rec = doc["files"]["identical/_mojolearn_svm.so"]
        self.assertEqual(rec["rebuilt_sha256"], sha(b"LEG identical/_mojolearn_svm.so"))
        self.assertEqual(rr.sha256(sdir / "identical" / "_mojolearn_svm.so"), rec["sha256"])
        self.assertTrue(any("NOTE hip/gfx942" in ln for ln in lines))
        self.assertTrue((sdir / "readback.txt").exists())

    def test_an_nvidia_rebuild_that_differs_is_refused(self):
        key = "linux-64/cuda/sm_89/identical/_mojolearn_gbdt"
        plan = self.plan(build={key})
        leg = self.tmp / "leg" / "cuda" / "sm_89"
        for r in rr.plan_rows(plan, rr.LINUX):
            if r["tier"] == "host" or (r["vendor"], r["arch"]) != ("cuda", "sm_89"):
                continue
            p = leg / r["set_rel"]
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"LEG " + r["set_rel"].encode())
        with self.assertRaises(SystemExit) as cm:
            rr.assemble_linux(plan, self.whl, {"cuda-sm_89": leg}, self.tmp / "sets", say=lambda *_: None)
        self.assertIn("NVIDIA rebuild must reproduce", str(cm.exception))

    def test_published_bytes_that_disagree_with_the_payload_are_refused(self):
        bad = self.tmp / "bad.whl"
        with zipfile.ZipFile(self.whl) as src, zipfile.ZipFile(bad, "w") as dst:
            for n in src.namelist():
                data = src.read(n)
                if n == "mojolearn/cuda/sm_89/identical/_mojolearn_svm.so":
                    data = data + b"x"
                dst.writestr(n, data)
        prev = dict(self.prev, linux=dict(wheel=bad.name, sha256=rr.sha256(bad)))
        plan = dict(self.plan(), previous=prev)
        with self.assertRaises(SystemExit) as cm:
            rr.assemble_linux(plan, bad, {}, self.tmp / "sets", say=lambda *_: None)
        self.assertIn("refusing to reuse", str(cm.exception))

    def test_the_packer_refuses_a_reused_file_whose_bytes_moved(self):
        out = rr.assemble_linux(self.plan(), self.whl, {}, self.tmp / "sets", say=lambda *_: None)
        sdir = out["cuda/sm_90a"]
        target = sdir / "identical" / "_mojolearn_svm.so"
        target.write_bytes(target.read_bytes() + b"\0")
        with self.assertRaises(SystemExit) as cm:
            pack_wheel.load_set(sdir.parent, include_byte_lm=True)
        self.assertIn("differs from the published 0.8.15 bytes", str(cm.exception))
        target.unlink()
        with self.assertRaises(SystemExit) as cm:
            pack_wheel.load_set(sdir.parent, include_byte_lm=True)
        self.assertIn("absent", str(cm.exception))

    def test_a_wheel_of_another_commit_is_refused(self):
        prev = dict(self.prev, source_commit="e" * 40)
        with self.assertRaises(SystemExit) as cm:
            rr.assemble_linux(dict(self.plan(), previous=prev), self.whl, {}, self.tmp / "sets", say=lambda *_: None)
        self.assertIn("names commit", str(cm.exception))

    def test_published_wheel_is_verified_against_the_record(self):
        ev = self.tmp / "ev"
        store = ev / "release" / "published"
        store.mkdir(parents=True)
        (store / self.whl.name).write_bytes(self.whl.read_bytes())
        self.assertEqual(rr.published_wheel(self.prev, "linux", ev, download=False), store / self.whl.name)
        (store / self.whl.name).write_bytes(self.whl.read_bytes() + b"x")
        with self.assertRaises(SystemExit) as cm:
            rr.published_wheel(self.prev, "linux", ev, download=False)
        self.assertIn("sha256 differs from the record", str(cm.exception))
        (store / self.whl.name).unlink()
        self.assertIsNone(rr.published_wheel(self.prev, "linux", ev, download=False))


class PayloadTests(unittest.TestCase):
    """release_inventory on assembled sets: no proof for a reused set, the
    origin of every binding recorded."""

    def test_reused_sets_need_no_proof_and_record_their_origin(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = pathlib.Path(d)
            whl, prev = published_wheel_fixture(tmp)
            rr.assemble_linux(linux_plan(prev), whl, {}, tmp / "sets", say=lambda *_: None)
            sets = [s for vendor in ("cuda", "hip") for s in pack_wheel.load_set(tmp / "sets" / vendor, True)]
            version = pack_wheel.read_version()
            inv = pack_wheel.release_inventory(sets, [], version, ROOT)
            self.assertEqual(inv["reuse"]["built"], 0)
            self.assertEqual(inv["reuse"]["reused"], 3 * 29 + len(pack_wheel.HOST_NAMES))
            self.assertEqual({v["origin"] for v in inv["sets"].values()}, {"reused"})
            self.assertEqual(inv["sets"]["hip/gfx942"]["from_release"]["version"], "0.8.15")
            self.assertEqual(inv["binding_origin"]["mojolearn/host/_mojolearn_core_host.so"]["origin"], "reused")
            self.assertEqual(inv["binding_origin"]["mojolearn/cuda/sm_89/_mojolearn_gbdt.so"]["from_release"]["version"], "0.8.15")
            self.assertEqual(inv["host_native"]["_mojolearn_core_host"]["origin"], "reused")
            self.assertTrue(inv["source_inventory"])
            self.assertEqual(len(inv["source_commit"]), 40)
            with self.assertRaises(SystemExit) as cm:
                pack_wheel.release_inventory(sets, [tmp / "nope.json"], version, ROOT)
            self.assertIn("one complete build proof per set a leg built", str(cm.exception))

    def test_a_mixed_set_needs_its_proof_and_records_both_digests(self):
        from check_linux_release_qualification import tracked_native_inventory, inventory_digest
        with tempfile.TemporaryDirectory() as d:
            tmp = pathlib.Path(d)
            whl, prev = published_wheel_fixture(tmp)
            plan = linux_plan(prev, build={"linux-64/hip/gfx942/identical/_mojolearn_gbdt"})
            leg = tmp / "leg" / "hip" / "gfx942"
            proof_ext = {}
            for r in rr.plan_rows(plan, rr.LINUX):
                if r["tier"] == "host" or (r["vendor"], r["arch"]) != ("hip", "gfx942"):
                    continue
                p = leg / r["set_rel"]
                p.parent.mkdir(parents=True, exist_ok=True)
                data = elf_stub(["libamdhip64.so.6"]) + b"LEG " + r["set_rel"].encode()
                p.write_bytes(data)
                proof_ext[r["archive_path"]] = sha(data)
            (leg / "host").mkdir()
            for name in pack_wheel.HOST_NAMES:
                (leg / "host" / f"{name}.so").write_bytes(HOST_BYTES + name.encode())
            (leg / ".libs").mkdir()
            (leg / ".libs" / "libAsyncRTMojoBindings.so").write_bytes(b"runtime")
            (leg / "manifest.json").write_text(json.dumps(dict(bytes_extensions=1, bytes_staged_libs=1,
                                                               driver_libs_not_staged=["libamdhip64.so.6"],
                                                               staged_libs=[dict(name="libAsyncRTMojoBindings.so")])))
            rows = [(tier, name) for tier in pack_wheel.TIERS for name in pack_wheel.tier_names(tier, True)]
            (leg / "readback.txt").write_text("".join(f"{t} {n} hip\n" for t, n in rows)
                                              + "".join(f"host {n} cpu\n" for n in pack_wheel.HOST_NAMES))
            (leg / "arch_readback.txt").write_text("".join(f"{t} {n} gfx942\n" for t, n in rows)
                                                   + "".join(f"host {n} NONE-BY-DESIGN\n" for n in pack_wheel.HOST_NAMES))
            inventory = tracked_native_inventory(ROOT)
            commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
            proof = tmp / "hip-gfx942.json"
            proof.write_text(json.dumps(dict(schema="mojolearn.linux.build-provenance.v1", complete=True, build_exit=0,
                                             action="build", source_commit=commit, source_inventory=inventory,
                                             source_sha256=inventory_digest(inventory), extensions=proof_ext)))
            rr.assemble_linux(plan, whl, {"hip-gfx942": leg}, tmp / "sets", say=lambda *_: None)
            sets = [s for vendor in ("cuda", "hip") for s in pack_wheel.load_set(tmp / "sets" / vendor, True)]
            inv = pack_wheel.release_inventory(sets, [proof], pack_wheel.read_version(), ROOT)
            self.assertEqual(inv["source_commit"], commit)
            self.assertEqual(inv["sets"]["hip/gfx942"]["origin"], "mixed")
            self.assertEqual(inv["sets"]["cuda/sm_89"]["origin"], "reused")
            gbdt = inv["binding_origin"]["mojolearn/hip/gfx942/identical/_mojolearn_gbdt.so"]
            svm = inv["binding_origin"]["mojolearn/hip/gfx942/identical/_mojolearn_svm.so"]
            self.assertEqual(gbdt["origin"], "built")
            self.assertEqual(svm["origin"], "reused")
            self.assertEqual(svm["rebuilt_sha256"], proof_ext["mojolearn/hip/gfx942/identical/_mojolearn_svm.so"])
            self.assertEqual(inv["extensions"]["mojolearn/hip/gfx942/identical/_mojolearn_gbdt.so"],
                             proof_ext["mojolearn/hip/gfx942/identical/_mojolearn_gbdt.so"])
            self.assertEqual(inv["reuse"]["built"], 1)
            # a proof for a set whose every binding is reused is refused
            with self.assertRaises(SystemExit) as cm:
                pack_wheel.release_inventory(sets, [proof, proof], pack_wheel.read_version(), ROOT)
            self.assertIn("one complete build proof per set a leg built", str(cm.exception))


if __name__ == "__main__":
    unittest.main()


class HostBuiltIntoReusedSets(unittest.TestCase):
    """A release whose only rebuilt binding is a host one: every GPU set is the
    published one, the cheapest leg (cuda-sm_89) built the host binding and
    read it back, and the sets without a read-back of their own carry that
    binding by byte-equality with the witnessed copy (the 0.8.17 shape)."""

    def _sets(self, tmp, tamper=False):
        whl, prev = published_wheel_fixture(tmp)
        plan = linux_plan(prev, build={"linux-64/host/-/host/_mojolearn_training_host"})
        leg = tmp / "leg" / "cuda" / "sm_89"
        proof_ext = {}
        for r in rr.plan_rows(plan, rr.LINUX):
            if r["tier"] == "host" or (r["vendor"], r["arch"]) != ("cuda", "sm_89"):
                continue
            p = leg / r["set_rel"]
            p.parent.mkdir(parents=True, exist_ok=True)
            # an NVIDIA rebuild reproduces the published bytes (the fixture wheel's)
            data = elf_stub(["libcuda.so.1", "libc.so.6"]) + r["archive_path"].encode()
            p.write_bytes(data)
            proof_ext[r["archive_path"]] = sha(data)
        (leg / "host").mkdir()
        for name in pack_wheel.HOST_NAMES:
            new = b"NEW " if name == "_mojolearn_training_host" else b""
            (leg / "host" / f"{name}.so").write_bytes(HOST_BYTES + new + name.encode())
        (leg / ".libs").mkdir()
        (leg / ".libs" / "libAsyncRTMojoBindings.so").write_bytes(b"runtime")
        (leg / "manifest.json").write_text(json.dumps(dict(bytes_extensions=1, bytes_staged_libs=1,
                                                           driver_libs_not_staged=["libcuda.so.1"],
                                                           staged_libs=[dict(name="libAsyncRTMojoBindings.so")])))
        rows = [(tier, name) for tier in pack_wheel.TIERS for name in pack_wheel.tier_names(tier, True)]
        (leg / "readback.txt").write_text("".join(f"{t} {n} cuda\n" for t, n in rows)
                                          + "".join(f"host {n} cpu\n" for n in pack_wheel.HOST_NAMES))
        (leg / "arch_readback.txt").write_text("".join(f"{t} {n} sm_89\n" for t, n in rows)
                                               + "".join(f"host {n} NONE-BY-DESIGN\n" for n in pack_wheel.HOST_NAMES))
        rr.assemble_linux(plan, whl, {"cuda-sm_89": leg}, tmp / "sets", say=lambda *_: None)
        if tamper:
            p = tmp / "sets" / "hip" / "gfx942" / "host" / "_mojolearn_training_host.so"
            p.write_bytes(p.read_bytes() + b"x")
        return [tmp / "sets" / "cuda", tmp / "sets" / "hip"], proof_ext

    def test_a_witnessed_host_build_packs_into_every_set(self):
        from check_linux_release_qualification import tracked_native_inventory, inventory_digest
        with tempfile.TemporaryDirectory() as d:
            tmp = pathlib.Path(d)
            set_paths, proof_ext = self._sets(tmp)
            witnesses = pack_wheel.host_witnesses(set_paths)
            self.assertEqual(sha(HOST_BYTES + b"NEW _mojolearn_training_host"), witnesses["_mojolearn_training_host"])
            sets = [s for sp in set_paths for s in pack_wheel.load_set(sp, True, host_witnesses_by_name=witnesses)]
            for s in sets:
                self.assertIn("_mojolearn_training_host", s.hosts)
                self.assertEqual(sha(s.hosts["_mojolearn_training_host"].read_bytes()), witnesses["_mojolearn_training_host"])
            self.assertEqual([s.arch for s in sets if (s.reuse or {}).get("files", {}).get("host/_mojolearn_training_host.so")], [])
            inventory = tracked_native_inventory(ROOT)
            commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
            proof = tmp / "cuda-sm_89.json"
            proof.write_text(json.dumps(dict(schema="mojolearn.linux.build-provenance.v1", complete=True, build_exit=0,
                                             action="build", source_commit=commit, source_inventory=inventory,
                                             source_sha256=inventory_digest(inventory), extensions=proof_ext)))
            inv = pack_wheel.release_inventory(sets, [proof], pack_wheel.read_version(), ROOT)
            host = inv["host_native"]["_mojolearn_training_host"]
            self.assertEqual(host["origin"], "built")
            self.assertEqual(host["sha256"], witnesses["_mojolearn_training_host"].hex())
            self.assertEqual(inv["host_native"]["_mojolearn_forest_host"]["origin"], "reused")
            # without a proof, or with two, the host build is not accounted for
            with self.assertRaises(SystemExit):
                pack_wheel.release_inventory(sets, [], pack_wheel.read_version(), ROOT)

    def test_a_host_build_without_a_witness_of_the_same_bytes_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = pathlib.Path(d)
            set_paths, _ = self._sets(tmp, tamper=True)
            witnesses = pack_wheel.host_witnesses(set_paths)
            pack_wheel.load_set(set_paths[0], True, host_witnesses_by_name=witnesses)
            with self.assertRaises(SystemExit) as err:
                pack_wheel.load_set(set_paths[1], True, host_witnesses_by_name=witnesses)
            self.assertIn("no set with a read-back holds the same bytes", str(err.exception))
            with self.assertRaises(SystemExit):
                pack_wheel.load_set(set_paths[1], True)
