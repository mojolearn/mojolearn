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
        # every Linux binding but the one built: the count follows the manifest
        # (the classical FAST tier added bindings on 2026-09-25), never a literal
        self.assertEqual(len(rr.plan_rows(plan, rr.LINUX, "REUSE")), len(rr.bindings(rr.LINUX)) - 1)

    def test_set_identity_moves_with_its_set_the_host_bindings_and_the_overlay_only(self):
        """What makes a completed leg reusable under a later freeze
        (tools/release.py): one digest per set over its bindings, every host
        binding and the runtime closure."""
        import copy
        prev = self.record(self.cur, dict(xcode="16"))
        plan = rr.make_plan(self.cur, self.cache, self.repo, host_toolchain=dict(xcode="16"), prev=prev)
        base = {s: rr.set_identity(plan, *s)["digest"] for s in rr.LINUX_SETS}
        self.assertEqual(len(set(base.values())), 3, "each set has its own identity")
        moved = copy.deepcopy(plan)
        for r in moved["rows"]:
            if (r["target"], r["vendor"], r["arch"], r["name"]) == (rr.LINUX, "hip", "gfx942", "_mojolearn_gbdt"):
                r["identity_digest"] = "9" * 64
        self.assertEqual(rr.set_identity(moved, "cuda", "sm_90a")["digest"], base[("cuda", "sm_90a")])
        self.assertNotEqual(rr.set_identity(moved, "hip", "gfx942")["digest"], base[("hip", "gfx942")])
        host = copy.deepcopy(plan)
        for r in host["rows"]:
            if r["target"] == rr.LINUX and r["tier"] == "host":
                r["identity_digest"] = "8" * 64
                break
        for s in rr.LINUX_SETS:
            self.assertNotEqual(rr.set_identity(host, *s)["digest"], base[s], "every leg builds the host bindings")
        # a builder the route overlay replaces is part of every identity
        over = rr.make_plan(self.cur, self.cache, self.repo, host_toolchain=dict(xcode="16"), prev=prev,
                            builders_override={"tools/release061_remote_build.sh": "7" * 64})
        self.assertEqual(over["builders_override"], {"tools/release061_remote_build.sh": "7" * 64})
        for s in rr.LINUX_SETS:
            self.assertNotEqual(rr.set_identity(over, *s)["digest"], base[s])
        self.assertEqual(over["legs"], ["cuda-sm_90a", "cuda-sm_89", "hip-gfx942"], "and it rebuilds")
        with self.assertRaises(SystemExit):
            rr.make_plan(self.cur, self.cache, self.repo, prev=prev, builders_override={"tools/unknown.sh": "1" * 64})
        # an unreadable binding identity makes a set unreusable
        broken = copy.deepcopy(plan)
        next(r for r in broken["rows"] if r["vendor"] == "cuda" and r["arch"] == "sm_89")["identity"] = None
        self.assertIsNone(rr.set_identity(broken, "cuda", "sm_89"))

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
            per_set = sum(len(pack_wheel.tier_names(t, True)) for t in pack_wheel.TIERS)
            self.assertEqual(len(doc["files"]), per_set + len(pack_wheel.HOST_NAMES) + 1)
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
        # a copy of another freeze under the store's name is passed over, not fatal:
        # the published one further down the candidates is taken
        (store / self.whl.name).write_bytes(self.whl.read_bytes() + b"x")
        other = ev / "release" / self.prev["version"] / "cafe" / "linux" / "final"
        other.mkdir(parents=True)
        (other / self.whl.name).write_bytes(self.whl.read_bytes())
        self.assertEqual(rr.published_wheel(self.prev, "linux", ev, download=False), other / self.whl.name)
        (other / self.whl.name).unlink()
        with self.assertRaises(SystemExit) as cm:
            rr.published_wheel(self.prev, "linux", ev, download=False)
        self.assertIn("passed over 1 of another freeze", str(cm.exception))
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
            per_set = sum(len(pack_wheel.tier_names(t, True)) for t in pack_wheel.TIERS)
            self.assertEqual(inv["reuse"]["reused"], 3 * per_set + len(pack_wheel.HOST_NAMES))
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

    def test_a_leg_of_an_earlier_freeze_packs_with_its_origin_and_no_proof(self):
        """tools/release.py took a completed hip/gfx942 leg of an earlier freeze
        (its set identity and build tooling equal this freeze's): its bytes are
        packed with their origin, the plan's REUSE rows still come from the
        published wheel, and a byte that moved is refused by name."""
        with tempfile.TemporaryDirectory() as d:
            tmp = pathlib.Path(d)
            whl, prev = published_wheel_fixture(tmp)
            build = {r["key"] for r in linux_plan(prev)["rows"]
                     if r["tier"] != "host" and (r["vendor"], r["arch"]) == ("hip", "gfx942")}
            plan = linux_plan(prev, build=build)
            old = tmp / "old" / "hip" / "gfx942"
            for r in rr.plan_rows(plan, rr.LINUX):
                if r["tier"] == "host" or (r["vendor"], r["arch"]) != ("hip", "gfx942"):
                    continue
                p = old / r["set_rel"]
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(elf_stub(["libamdhip64.so.6"]) + b"EARLIER LEG " + r["set_rel"].encode())
            (old / "host").mkdir()
            for name in pack_wheel.HOST_NAMES:
                (old / "host" / f"{name}.so").write_bytes(HOST_BYTES + name.encode())
            (old / ".libs").mkdir()
            (old / ".libs" / "libAsyncRTMojoBindings.so").write_bytes(b"runtime")
            (old / "manifest.json").write_text(json.dumps(dict(bytes_extensions=1, bytes_staged_libs=1,
                                                               driver_libs_not_staged=["libamdhip64.so.6"],
                                                               staged_libs=[dict(name="libAsyncRTMojoBindings.so")])))
            rows = [(tier, name) for tier in pack_wheel.TIERS for name in pack_wheel.tier_names(tier, True)]
            (old / "readback.txt").write_text("".join(f"{t} {n} hip\n" for t, n in rows)
                                              + "".join(f"host {n} cpu\n" for n in pack_wheel.HOST_NAMES))
            (old / "arch_readback.txt").write_text("".join(f"{t} {n} gfx942\n" for t, n in rows)
                                                   + "".join(f"host {n} NONE-BY-DESIGN\n" for n in pack_wheel.HOST_NAMES))
            origin = dict(source_commit="e" * 40, leg="hip-gfx942", proof_sha256="p" * 64, admitted_for=CUR_COMMIT,
                          set_identity_digest="d" * 64, tooling_digest="t" * 64)
            out = rr.assemble_linux(plan, whl, {}, tmp / "sets", say=lambda *_: None,
                                    leg_reuse={"hip-gfx942": dict(dir=old, origin=origin)})
            doc = json.loads((out["hip/gfx942"] / "reuse.json").read_text())
            self.assertEqual(doc["from_legs"]["hip-gfx942"], origin)
            gbdt = doc["files"]["identical/_mojolearn_gbdt.so"]
            self.assertEqual((gbdt["source"], gbdt["leg"]), ("leg", "hip-gfx942"))
            self.assertNotIn("source", doc["files"]["host/_mojolearn_core_host.so"], "a REUSE row is the published copy")
            sets = [s for vendor in ("cuda", "hip") for s in pack_wheel.load_set(tmp / "sets" / vendor, True)]
            inv = pack_wheel.release_inventory(sets, [], pack_wheel.read_version(), ROOT)
            o = inv["binding_origin"]["mojolearn/hip/gfx942/identical/_mojolearn_gbdt.so"]
            self.assertEqual((o["origin"], o["from_leg"]["source_commit"], o["from_release"]), ("reused", "e" * 40, None))
            self.assertEqual(inv["sets"]["hip/gfx942"]["from_legs"]["hip-gfx942"]["admitted_for"], CUR_COMMIT)
            self.assertIn("hip-gfx942", inv["reuse"]["from_legs"])
            self.assertEqual(inv["reuse"]["built"], 0)
            target = out["hip/gfx942"] / "identical" / "_mojolearn_gbdt.so"
            target.write_bytes(target.read_bytes() + b"\0")
            with self.assertRaises(SystemExit) as cm:
                pack_wheel.load_set(tmp / "sets" / "hip", True)
            self.assertIn("leg hip-gfx942 of eeeeeeeeeeee", str(cm.exception))
            # an origin that does not name what was taken is not a reuse record
            doc["from_legs"]["hip-gfx942"].pop("tooling_digest")
            (out["hip/gfx942"] / "reuse.json").write_text(json.dumps(doc))
            with self.assertRaises(SystemExit) as cm:
                pack_wheel.load_set(tmp / "sets" / "hip", True)
            self.assertIn("is not a", str(cm.exception))


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
            self.assertEqual(sha(HOST_BYTES + b"NEW _mojolearn_training_host"), witnesses["_mojolearn_training_host"].hex())
            sets = [s for sp in set_paths for s in pack_wheel.load_set(sp, True, host_witnesses_by_name=witnesses)]
            for s in sets:
                self.assertIn("_mojolearn_training_host", s.hosts)
                self.assertEqual(sha(s.hosts["_mojolearn_training_host"].read_bytes()), witnesses["_mojolearn_training_host"].hex())
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


class MacosAssembleTests(unittest.TestCase):
    """assemble_macos takes the REUSE rows of the macOS target from the published
    macOS wheel into dest/python/mojolearn/..., keyed by the package path the
    macOS build script writes (a plan row carries archive_path only)."""

    def test_reused_macos_bindings_land_under_python_and_the_plan_names_them(self):
        import base64
        with tempfile.TemporaryDirectory() as d:
            tmp = pathlib.Path(d)
            rows = [b.row() for b in rr.bindings(rr.MACOS)]
            self.assertTrue(rows and all("package_rel" not in r for r in rows))
            reuse = rows[:2]
            whl = tmp / "mojolearn-0.8.18-py3-none-macosx_11_0_arm64.whl"
            record_lines = []
            data = {}
            with zipfile.ZipFile(whl, "w") as z:
                for r in reuse:
                    payload = b"METAL " + r["archive_path"].encode()
                    z.writestr(r["archive_path"], payload)
                    digest = hashlib.sha256(payload).digest()
                    record_lines.append("%s,sha256=%s,%d" % (r["archive_path"], base64.urlsafe_b64encode(digest).rstrip(b"=").decode(), len(payload)))
                    data[r["archive_path"]] = payload
                z.writestr("mojolearn-0.8.18.dist-info/RECORD", "\n".join(record_lines) + "\n")
            plan = dict(schema=rr.PLAN_SCHEMA, commit=CUR_COMMIT,
                        previous=dict(version="0.8.18", source_commit=PREV_COMMIT),
                        rows=[dict(r, decision="REUSE", reason="test", identity_digest="d" * 64) for r in reuse]
                        + [dict(r, decision="BUILD", reason="test", identity_digest="e" * 64) for r in rows[2:]],
                        legs=[], leg_reasons={}, runtime=dict(decision="REUSE"))
            out = rr.assemble_macos(plan, whl, tmp / "macos", say=lambda *_: None)
            doc = json.loads(pathlib.Path(out).read_text())
            for r in reuse:
                rel = "python/" + r["archive_path"]
                self.assertIn(rel, doc["files"])
                self.assertEqual((tmp / "macos" / rel).read_bytes(), data[r["archive_path"]])
                self.assertEqual(doc["files"][rel]["archive_path"], r["archive_path"])
            self.assertEqual(len(doc["files"]), 2)


# ---------------------------------------------------------------- builder views
LINUX_SETS_SH = "packaging/linux/build_sets.sh"
MACOS_WHEEL_SH = "packaging/macos/build_release_wheel.sh"
SHELL_BUILDERS = (LINUX_SETS_SH, MACOS_WHEEL_SH, "tools/release061_remote_build.sh",
                  "tools/linux_surface_qualification.sh")
# bindings read through the views below: one per list kind
VIEWERS = ("_mojolearn_svm", "_mojolearn_gbdt", "_mojolearn_training", "_mojolearn", "_mojolearn_byte_lm",
           "_mojolearn_core_host")


def real(rel):
    return (ROOT / rel).read_text()


def view(rel, text, name):
    return rr.builder_view(rel, text.encode(), rr.binding_members(name))


def edit_line(text, startswith, fn):
    """`text` with the one line that starts with `startswith` rewritten."""
    lines = text.split("\n")
    hits = [i for i, line in enumerate(lines) if line.startswith(startswith)]
    assert len(hits) == 1, (startswith, hits)
    lines[hits[0]] = fn(lines[hits[0]])
    return "\n".join(lines)


def add_to_list(text, var, token):
    return edit_line(text, var + '="', lambda line: line[:-1] + " " + token + '"' if not line.endswith('}"')
                     else line[:-2] + " " + token + '}"')


class BuilderViewTests(unittest.TestCase):
    """The builder part of an identity: another binding joining a list, or a
    comment, does not move it; anything that can change this binding's bytes
    does. Runs on the REAL builder files."""

    def assertViews(self, rel, before, after, same, moved):
        for name in same:
            self.assertEqual(view(rel, before, name), view(rel, after, name), f"{rel}: {name} moved")
        for name in moved:
            self.assertNotEqual(view(rel, before, name), view(rel, after, name), f"{rel}: {name} did not move")

    # (a) another binding joining a list
    def test_an_unrelated_binding_joining_a_list_moves_no_other_binding(self):
        for rel in (LINUX_SETS_SH, MACOS_WHEEL_SH):
            text = real(rel)
            scripts_var = "SCRIPTS" if rel == LINUX_SETS_SH else "BUILD_SCRIPTS"
            for names_var, scripts in (("EXT_NAMES", scripts_var), ("FAST_CLASSICAL_NAMES", "FAST_CLASSICAL_SCRIPTS"),
                                       ("IDENTICAL_ONLY_NAMES", "IDENTICAL_ONLY_SCRIPTS")):
                after = add_to_list(add_to_list(text, names_var, "_mojolearn_newthing"), scripts, "build_newthing.sh")
                self.assertNotEqual(text, after)
                self.assertViews(rel, text, after, same=VIEWERS, moved=("_mojolearn_newthing",))
                # the runtime closure keeps every list whole: a new binding
                # can need one more MAX library in .libs
                self.assertNotEqual(rr.builder_view(rel, text.encode(), None), rr.builder_view(rel, after.encode(), None))
            # a binding leaving a list moves only that binding
            after = edit_line(text, 'FAST_CLASSICAL_NAMES="', lambda line: line.replace(" _mojolearn_ivf", ""))
            after = edit_line(after, 'FAST_CLASSICAL_SCRIPTS="', lambda line: line.replace(" build_ivf.sh", ""))
            self.assertViews(rel, text, after, same=VIEWERS, moved=("_mojolearn_ivf",))
            # reordering a list moves nothing
            after = edit_line(text, 'EXT_NAMES="', lambda line: 'EXT_NAMES="_mojolearn_trees _mojolearn_rf _mojolearn_gbdt"')
            self.assertViews(rel, text, after, same=VIEWERS + ("_mojolearn_rf",), moved=())

    def test_the_whole_identity_does_not_move_when_another_binding_joins(self):
        with tempfile.TemporaryDirectory() as d:
            repo, prev, cur, env = tiny_repo(d)
            for rel in (LINUX_SETS_SH, MACOS_WHEEL_SH):
                (repo / rel).write_text(real(rel))
            subprocess.run(["git", "-C", str(repo), "commit", "-q", "-am", "real builders"], check=True, env=env)
            cache = pathlib.Path(d) / "cache"
            a = rr.facts_for_commit("HEAD", cache, repo)
            for rel in (LINUX_SETS_SH, MACOS_WHEEL_SH):
                text = (repo / rel).read_text()
                text = add_to_list(text, "FAST_CLASSICAL_NAMES", "_mojolearn_newthing")
                text = add_to_list(text, "FAST_CLASSICAL_SCRIPTS", "build_newthing.sh")
                (repo / rel).write_text(text)
            subprocess.run(["git", "-C", str(repo), "commit", "-q", "-am", "a new binding"], check=True, env=env)
            b = rr.facts_for_commit("HEAD", cache, repo)
            self.assertNotEqual(a["builders"], b["builders"])  # the whole files did move
            gpu, host, mac = two_bindings()
            for binding in (gpu, host, mac):
                self.assertEqual(rr.compare(rr.identity(binding, b, dict(xcode="16")),
                                            rr.identity(binding, a, dict(xcode="16"))), ("REUSE", "identity unchanged"))
            self.assertNotEqual(rr.digest_of(rr.runtime_identity(a)), rr.digest_of(rr.runtime_identity(b)))
            # the same binding moved to another tier list rebuilds
            for rel in (LINUX_SETS_SH, MACOS_WHEEL_SH):
                text = (repo / rel).read_text()
                text = edit_line(text, 'FAST_CLASSICAL_NAMES="', lambda line: line.replace(" _mojolearn_svm", ""))
                text = add_to_list(text, "IDENTICAL_ONLY_NAMES", "_mojolearn_svm")
                (repo / rel).write_text(text)
            subprocess.run(["git", "-C", str(repo), "commit", "-q", "-am", "svm tier"], check=True, env=env)
            c = rr.facts_for_commit("HEAD", cache, repo)
            for binding in (gpu, mac):
                self.assertEqual(rr.compare(rr.identity(binding, c, dict(xcode="16")),
                                            rr.identity(binding, b, dict(xcode="16"))), ("BUILD", "changed: builders"))
            self.assertEqual(rr.compare(rr.identity(host, c), rr.identity(host, b))[0], "REUSE")

    # (b) anything that can change this binding's bytes
    def test_moving_this_binding_between_tier_lists_moves_it(self):
        # _mojolearn_solver is FAST classical on both targets (FAST svm is
        # Apple-only since 69a519c15, so the Linux list no longer holds svm)
        for rel in (LINUX_SETS_SH, MACOS_WHEEL_SH):
            text = real(rel)
            after = edit_line(text, 'FAST_CLASSICAL_NAMES="', lambda line: line.replace(" _mojolearn_solver", ""))
            after = edit_line(after, 'FAST_CLASSICAL_SCRIPTS="', lambda line: line.replace(" build_solver.sh", ""))
            after = add_to_list(add_to_list(after, "IDENTICAL_ONLY_NAMES", "_mojolearn_solver"),
                                "IDENTICAL_ONLY_SCRIPTS", "build_solver.sh")
            self.assertViews(rel, text, after, same=("_mojolearn_gbdt", "_mojolearn_training", "_mojolearn"),
                             moved=("_mojolearn_solver",))
            # the name alone or the script alone moving is enough
            only_name = edit_line(text, 'FAST_CLASSICAL_NAMES="', lambda line: line.replace(" _mojolearn_solver", ""))
            self.assertViews(rel, text, only_name, same=("_mojolearn_gbdt",), moved=("_mojolearn_solver",))

    def test_any_code_line_of_a_builder_moves_every_binding(self):
        text = real(LINUX_SETS_SH)
        edits = [
            # a flag on the compile line
            lambda t: t.replace("MOJOLEARN_COMPILE_JOBS=2 \\", "MOJOLEARN_COMPILE_JOBS=3 \\", 1),
            # the tier default
            lambda t: t.replace('TIERS="${MOJOLEARN_BUILD_TIERS:-fast deterministic identical}"',
                                'TIERS="${MOJOLEARN_BUILD_TIERS:-deterministic identical}"'),
            # the tier_names function body
            lambda t: t.replace('if [[ "$1" = identical || "$1" = fast ]]; then printf \' %s\' "$FAST_CLASSICAL_NAMES"; fi',
                                'if [[ "$1" = identical ]]; then printf \' %s\' "$FAST_CLASSICAL_NAMES"; fi'),
            # a line inside an embedded Python heredoc, and a '#' line there
            lambda t: t.replace("import importlib.machinery, importlib.util, sys",
                                "import importlib.machinery, importlib.util, sys, os", 1),
            lambda t: t.replace("import importlib.machinery, importlib.util, sys",
                                "# a heredoc line is data\nimport importlib.machinery, importlib.util, sys", 1),
            # a list variable renamed, or its environment override renamed
            lambda t: t.replace('EXT_NAMES="', 'EXTN_NAMES="', 1),
            lambda t: t.replace("${MOJOLEARN_BUILD_SCRIPTS:-", "${MOJOLEARN_SCRIPTS:-", 1),
            # a list that holds something other than bindings is kept whole
            lambda t: t.replace('EXT_NAMES="_mojolearn_gbdt _mojolearn_rf _mojolearn_trees"',
                                'EXT_NAMES="_mojolearn_gbdt _mojolearn_rf _mojolearn_trees $EXTRA"'),
            # indentation after a continuation line
            lambda t: t.replace("MOJOLEARN_COMPILE_JOBS=2 \\\n", "MOJOLEARN_COMPILE_JOBS=2 \\\n ", 1),
            # the shebang
            lambda t: ("#!/bin/sh" + t[len("#!/usr/bin/env bash"):]) if t.startswith("#!/usr/bin/env bash")
            else "#!/bin/sh\n" + t,
        ]
        for k, fn in enumerate(edits):
            after = fn(text)
            self.assertNotEqual(after, text, f"edit {k} did not apply")
            self.assertViews(LINUX_SETS_SH, text, after, same=(), moved=VIEWERS)
        for rel in rr.LINUX_BUILDERS + rr.MACOS_BUILDERS:
            text = real(rel)
            after = text + "\necho one more line\n" if rel.endswith(".sh") else text + "\nX = 1\n"
            self.assertViews(rel, text, after, same=(), moved=VIEWERS)

    def test_its_own_build_script_moves_it(self):
        with tempfile.TemporaryDirectory() as d:
            repo, prev, cur, env = tiny_repo(d)
            cache = pathlib.Path(d) / "cache"
            a = rr.facts_for_commit("HEAD", cache, repo)
            p = repo / "bindings" / "build_svm.sh"
            p.write_text(p.read_text().replace("mojo build", "mojo build -O2"))
            subprocess.run(["git", "-C", str(repo), "commit", "-q", "-am", "script"], check=True, env=env)
            b = rr.facts_for_commit("HEAD", cache, repo)
            gpu, host, _ = two_bindings()
            self.assertEqual(rr.compare(rr.identity(gpu, b), rr.identity(gpu, a)), ("BUILD", "changed: closure"))
            self.assertEqual(rr.compare(rr.identity(host, b), rr.identity(host, a))[0], "REUSE")

    # (c) comments
    def test_full_line_shell_comments_and_blank_lines_do_not_move_it(self):
        for rel in SHELL_BUILDERS:
            text = real(rel)
            lines = text.split("\n")
            first = [i for i, line in enumerate(lines[:120]) if line.startswith("# ") and i > 0][0]
            after = "\n".join(lines[:first] + ["# reworded", "", "   # indented comment"] + lines[first + 1:])
            self.assertViews(rel, text, after, same=VIEWERS, moved=())

    def test_shell_comments_that_are_not_unambiguous_move_it(self):
        base = "#!/usr/bin/env bash\nX=1\n"
        cases = [
            # a trailing comment: kept (only full-line comments come out)
            ("echo a  # one\n", "echo a  # two\n"),
            # a '#' line inside a multi-line string is data
            ("s='\n# one\n'\n", "s='\n# two\n'\n"),
            ('s="\n# one\n"\n', 's="\n# two\n"\n'),
            # inside a heredoc body
            ("cat <<EOF\n# one\nEOF\n", "cat <<EOF\n# two\nEOF\n"),
            ("cat <<-'EOF'\n\t# one\n\tEOF\n", "cat <<-'EOF'\n\t# two\n\tEOF\n"),
            # after a continuation line a comment line ends the command
            ("echo a \\\n# one\n", "echo a \\\n# two\n"),
            # after a construct the view does not model, everything is kept
            ("x=`date`\n# one\n", "x=`date`\n# two\n"),
            ("x=$'a\\'b'\n# one\n", "x=$'a\\'b'\n# two\n"),
            ('x="${a:-"b"}"\n# one\n', 'x="${a:-"b"}"\n# two\n'),
            ("y=$((1 << 2))\n# one\n", "y=$((1 << 2))\n# two\n"),
            ("y=$(case $a in b) echo;; esac)\n# one\n", "y=$(case $a in b) echo;; esac)\n# two\n"),
        ]
        for one, two in cases:
            self.assertNotEqual(view("x.sh", base + one, "_mojolearn_svm"), view("x.sh", base + two, "_mojolearn_svm"),
                                repr(one))
        # and these are modeled, so the comment after them comes out
        modeled = ['x="$(dirname "$0")"\n', "x='a # b'\n", 'x="a # b"\n', "echo a\\ #b\n",
                   "v=$(python3 - <<'PY'\nprint(1)\nPY\n)\n", "echo \"${#a}\" ${a#b}\n"]
        for code in modeled:
            self.assertEqual(view("x.sh", base + code + "# one\n", "_mojolearn_svm"),
                             view("x.sh", base + code + "# two\n", "_mojolearn_svm"), repr(code))

    def test_list_lines_are_normalized_only_at_the_top_level(self):
        base = "#!/usr/bin/env bash\n"
        for wrap in ("cat <<EOF\n%sEOF\n", "s='\n%s'\n", "f() { :\n  %s}\n"):
            a = base + wrap % 'EXT_NAMES="_mojolearn_gbdt"\n'
            b = base + wrap % 'EXT_NAMES="_mojolearn_gbdt _mojolearn_new"\n'
            self.assertNotEqual(view("x.sh", a, "_mojolearn_svm"), view("x.sh", b, "_mojolearn_svm"), wrap)
        a = base + 'EXT_NAMES="_mojolearn_gbdt"\n'
        b = base + 'EXT_NAMES="_mojolearn_gbdt _mojolearn_new"\n'
        self.assertEqual(view("x.sh", a, "_mojolearn_svm"), view("x.sh", b, "_mojolearn_svm"))

    def test_python_comments_do_not_move_it_but_code_and_strings_do(self):
        for rel in ("packaging/linux/stage_libs.py", "packaging/macos/stage_dylibs.py", "python/setup.py"):
            text = real(rel)
            self.assertIsNotNone(rr.python_view(text))
            lines = text.split("\n")
            first = [i for i, line in enumerate(lines) if line.startswith("# ") and i > 1][0]
            after = "\n".join(lines[:first] + ["# reworded", "    # indented"] + lines[first + 1:])
            self.assertViews(rel, text, after, same=VIEWERS, moved=())
        rel = "packaging/linux/stage_libs.py"
        a = "#!/usr/bin/env python3\nx = 1  # one\ns = '''\n# one\n'''\n"
        self.assertEqual(rr.builder_view(rel, a.encode(), None),
                         rr.builder_view(rel, a.replace("x = 1  # one", "x = 1  # two").encode(), None))
        for b in (a.replace("x = 1", "x = 2"), a.replace("'''\n# one", "'''\n# two"), a.replace("python3", "python3.12"),
                  a.replace("x = 1", "x =  1")):
            self.assertNotEqual(rr.builder_view(rel, a.encode(), None), rr.builder_view(rel, b.encode(), None), b)
        # not Python: hashed whole
        self.assertNotEqual(rr.builder_view(rel, b"def (:\n# one\n", None), rr.builder_view(rel, b"def (:\n# two\n", None))

    def test_bash_parses_the_view_as_it_parses_the_file(self):
        """The shell view with every list kept is the script bash reads: bash's
        own re-serialization of each real builder (wrapped in a function, never
        run) is byte-identical with and without the comments the view drops."""
        bash = "/bin/bash"
        if not os.path.exists(bash):
            self.skipTest("no /bin/bash")

        def canon(text):
            with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
                f.write("__v() {\n" + text + "\n}\ndeclare -f __v\n")
            try:
                r = subprocess.run([bash, f.name], capture_output=True, text=True)
            finally:
                os.unlink(f.name)
            self.assertEqual(r.returncode, 0, r.stderr)
            return r.stdout

        for rel in SHELL_BUILDERS:
            text = real(rel)
            dropped = rr.shell_view(text, None)
            self.assertLess(len(dropped), len(text), rel)
            self.assertEqual(canon(text), canon(dropped), rel)
            # and the check can fail: a code edit shows in bash's reading
            self.assertNotEqual(canon(text), canon(text + "\necho sentinel\n"), rel)


class IdentityUpgradeTests(unittest.TestCase):
    """0.8.18 and earlier recorded v1 identities (whole-file builder digests).
    They compare in today's schema only after reproducing exactly."""

    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.repo, self.prev, self.cur, self.env = tiny_repo(self._t.name)
        self.cache = pathlib.Path(self._t.name) / "cache"
        for target in (rr.LINUX, rr.MACOS):
            for b in rr.bindings(target):
                p = self.repo / "bindings" / b.script
                if not p.exists():
                    p.write_text(f"mojo build k.mojo -o python/mojolearn/x/{b.name}.so\n")
        (self.repo / LINUX_SETS_SH).write_text(real(LINUX_SETS_SH))
        subprocess.run(["git", "-C", str(self.repo), "add", "-A"], check=True, env=self.env)
        subprocess.run(["git", "-C", str(self.repo), "commit", "-q", "-m", "all scripts"], check=True, env=self.env)
        self.at = subprocess.run(["git", "-C", str(self.repo), "rev-parse", "HEAD"], capture_output=True,
                                 text=True).stdout.strip()

    def tearDown(self):
        self._t.cleanup()

    def v1_record(self, tamper=None):
        facts = rr.facts_for_commit(self.at, self.cache, self.repo)
        rows = []
        for target in (rr.LINUX, rr.MACOS):
            for b in rr.bindings(target):
                ident = rr.identity(b, facts, dict(xcode="16"), schema=rr.SCHEMA_V1)
                if tamper:
                    ident = tamper(b, ident)
                rows.append(dict(key=b.key, identity=ident))
        path = pathlib.Path(self._t.name) / "binding-identities.json"
        path.write_text(json.dumps(dict(commit=self.at, rows=rows,
                                        runtime=dict(identity=rr.runtime_identity(facts, rr.SCHEMA_V1)))))
        return dict(version="0.8.18", source_commit=self.at, identities=str(path),
                    linux=dict(wheel="l.whl", sha256="0" * 64), macos=dict(wheel="m.whl", sha256="1" * 64))

    def test_a_v1_record_that_reproduces_is_upgraded_and_one_that_does_not_builds(self):
        text = add_to_list(real(LINUX_SETS_SH), "EXT_NAMES", "_mojolearn_newthing")
        (self.repo / LINUX_SETS_SH).write_text(text)
        subprocess.run(["git", "-C", str(self.repo), "commit", "-q", "-am", "a list edit"], check=True, env=self.env)
        plan = rr.make_plan("HEAD", self.cache, self.repo, host_toolchain=dict(xcode="16"), prev=self.v1_record())
        self.assertIn("upgraded after reproducing", plan["previous_identities_from"])
        self.assertEqual({r["decision"] for r in plan["rows"]}, {"REUSE"})
        # the runtime keeps every list whole, so it builds, on the cheapest leg
        self.assertEqual(plan["runtime"]["decision"], "BUILD")
        self.assertEqual(plan["legs"], ["cuda-sm_89"])

        # one recorded identity that the commit does not reproduce
        def tamper(b, ident):
            if b.name == "_mojolearn_gbdt" and b.target == rr.LINUX and b.arch == "gfx942" and b.tier == "fast":
                return dict(ident, flags=dict(ident["flags"], compile_jobs="7"))
            return ident
        plan = rr.make_plan("HEAD", self.cache, self.repo, host_toolchain=dict(xcode="16"), prev=self.v1_record(tamper))
        builds = rr.plan_rows(plan, decision="BUILD")
        self.assertEqual([(r["arch"], r["tier"], r["name"]) for r in builds], [("gfx942", "fast", "_mojolearn_gbdt")])
        self.assertIn("does not reproduce", builds[0]["reason"])

    def test_identities_of_two_schemas_never_compare_equal(self):
        facts = rr.facts_for_commit(self.at, self.cache, self.repo)
        gpu, _, _ = two_bindings()
        v1, v2 = rr.identity(gpu, facts, schema=rr.SCHEMA_V1), rr.identity(gpu, facts)
        decision, reason = rr.compare(v2, v1)
        self.assertEqual(decision, "BUILD")
        self.assertIn(rr.SCHEMA_V1, reason)
        self.assertEqual(v2["builder_rule"], rr.BUILDER_RULE)
