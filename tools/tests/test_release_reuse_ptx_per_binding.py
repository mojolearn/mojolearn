"""The PTX set reuses per binding like every other set (lane ptx-reuse, 2026-11-01).

CPU only, no compiler, no network. Plan: an unchanged PTX binding is REUSE; a
changed closure, toolchain or flag still builds; a PTX set with no binding to
build is taken whole only with the retained original proof of the leg that
built the published bytes (record ptx-proof/ or the leg tree under the
evidence root), and otherwise builds with the missing input named. Assembly: a
merged published file is reused as a unit, only when every binding it serves
is REUSE, and an NVIDIA rebuild must reproduce it; a mixed PTX set's
PTX_BASELINE.json still names exactly its packed bytes; a published PTX set
taken from a record-only proof is rebound and admitted by ptx_reuse.

    python3 -m pytest tools/tests/test_release_reuse_ptx_per_binding.py
"""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "tests"))
sys.path.insert(0, str(ROOT / "packaging" / "linux"))
import release_reuse as rr  # noqa: E402
import pack_wheel  # noqa: E402
import ptx_reuse as pr  # noqa: E402
from test_release_reuse import elf_stub, tiny_repo  # noqa: E402
import test_release_reuse_ptx as ptx_fixtures  # noqa: E402

PTX_LEG = "%s-%s" % rr.PTX_SET
HOST_LEG = "%s-%s" % rr.HOST_LEG


def sha(data):
    return hashlib.sha256(data).hexdigest()


def tier_dir(tier):
    return "" if tier == "fast" else tier + "/"


class PlanTests(unittest.TestCase):
    """make_plan on the real binding lists against a record, as
    test_release_reuse.DecisionTests, with the PTX set's shipped bytes recorded."""

    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.addCleanup(self._t.cleanup)
        self.tmp = Path(self._t.name)
        (self.tmp / "repo").mkdir()
        self.repo, _, self.cur, env = tiny_repo(self.tmp / "repo")
        self.cache = self.tmp / "cache"
        for target in (rr.LINUX, rr.MACOS):
            for b in rr.bindings(target):
                p = self.repo / "bindings" / b.script
                if not p.exists():
                    p.write_text(f"mojo build k.mojo -o python/mojolearn/x/{b.name}.so\n")
        subprocess.run(["git", "-C", str(self.repo), "add", "-A"], check=True, env=env)
        subprocess.run(["git", "-C", str(self.repo), "commit", "-q", "-m", "all scripts"], check=True, env=env)
        self.cur = subprocess.run(["git", "-C", str(self.repo), "rev-parse", "HEAD"], capture_output=True,
                                  text=True).stdout.strip()
        self.ptx = [b for b in rr.bindings(rr.LINUX) if (b.vendor, b.arch) == rr.PTX_SET]
        self.assertTrue(self.ptx, "the release ships the PTX slot")
        # one merged file per tier carries every PTX binding of that tier
        self.groups = {f"mojolearn/cuda/sm_80/{tier_dir(t)}_mojolearn_native.so": sha(t.encode())
                       for t in {b.tier for b in self.ptx}}

    def record(self, mutate=None, record_dir=None):
        facts = rr.facts_for_commit(self.cur, self.cache, self.repo)
        rows = []
        for target in (rr.LINUX, rr.MACOS):
            for b in rr.bindings(target):
                ident = rr.identity(b, facts, dict(xcode="16"))
                if mutate:
                    ident = mutate(b, ident)
                row = dict(key=b.key, identity=ident)
                if (b.vendor, b.arch) == rr.PTX_SET:
                    canonical = f"mojolearn/cuda/sm_80/{tier_dir(b.tier)}_mojolearn_native.so"
                    row.update(shipped_archive_path=canonical.replace("/cuda/", "/cuda_ptx/"),
                               shipped_sha256=self.groups[canonical])
                rows.append(row)
        path = self.tmp / "binding-identities.json"
        path.write_text(json.dumps(dict(rows=rows, runtime=dict(identity=rr.runtime_identity(facts)))))
        prev = dict(version="0.8.15", source_commit=self.cur, identities=str(path),
                    linux=dict(wheel="mojolearn-0.8.15-py3-none-manylinux_2_35_x86_64.whl", sha256="0" * 64),
                    macos=dict(wheel="mojolearn-0.8.15-py3-none-macosx_11_0_arm64.whl", sha256="1" * 64))
        if record_dir is not None:
            prev["record_dir"] = str(record_dir)
        return prev

    def plan(self, prev, evidence_root=None):
        return rr.make_plan(self.cur, self.cache, self.repo, host_toolchain=dict(xcode="16"), prev=prev,
                            evidence_root=evidence_root)

    def original_identity(self):
        """The set identity the leg that built the published PTX bytes recorded."""
        return rr.set_identity(self.plan(self.record()), *rr.PTX_SET)

    def leg_tree(self, extensions=None, built_from=None):
        rb = self.tmp / "ptx-leg" / "release-build"
        sdir = rb / "build/sets/cuda/sm_80"
        sdir.mkdir(parents=True, exist_ok=True)
        (sdir / "PTX_BASELINE.json").write_text("{}\n")
        (sdir / "manifest.json").write_text("{}\n")
        (rb / "build/build-provenance.json").write_text(json.dumps(dict(
            schema="mojolearn.linux.build-provenance.v1", complete=True, build_exit=0,
            source_commit=built_from or self.cur, extensions=self.groups if extensions is None else extensions)))
        return rb

    def retained_record(self, **kw):
        rec = self.tmp / "record"
        rr.record_ptx_proof(rec, PTX_LEG, self.cur, self.leg_tree(**kw), self.original_identity(), "t" * 64)
        return rec

    @staticmethod
    def host_change(b, ident):
        return dict(ident, flags=dict(ident["flags"], compile_jobs="9")) if b.name == "_mojolearn_core_host" else ident

    def ptx_rows(self, plan):
        return [r for r in rr.plan_rows(plan, rr.LINUX) if r["tier"] != "host" and (r["vendor"], r["arch"]) == rr.PTX_SET]

    # ------------------------------------------------ one binding changes
    def test_a_changed_ptx_binding_builds_alone_and_the_rest_reuse(self):
        one = self.ptx[0]
        for field, value in (("closure", dict(scope="closure", digest="9" * 64)),
                             ("toolchain", dict(packages=["mojo-0.9.0-release"])),
                             ("flags", dict(numeric_mode="identical", gpu_archs="sm_80", compile_jobs="7"))):
            with self.subTest(field=field):
                def mutate(b, ident, field=field, value=value):
                    return dict(ident, **{field: value}) if b.key == one.key else ident
                plan = self.plan(self.record(mutate))
                rows = self.ptx_rows(plan)
                built = [r for r in rows if r["decision"] == "BUILD"]
                self.assertEqual([r["key"] for r in built], [one.key])
                self.assertTrue(built[0]["reason"].startswith("changed: " + field), built[0]["reason"])
                self.assertEqual({r["decision"] for r in rows if r["key"] != one.key}, {"REUSE"})
                # the PTX leg runs and proves its own set; no retained proof is asked for
                self.assertIn(PTX_LEG, plan["legs"])
                self.assertEqual(plan["leg_reasons"][PTX_LEG], "1 binding(s) to build")
                self.assertIsNone(plan["ptx_proof"])

    # ------------------------------------------------ no PTX binding changes
    def test_an_unchanged_ptx_set_is_taken_from_the_record_proof(self):
        rec = self.retained_record()
        plan = self.plan(self.record(self.host_change, rec))
        self.assertEqual({r["decision"] for r in self.ptx_rows(plan)}, {"REUSE"})
        self.assertNotIn(PTX_LEG, plan["legs"])
        self.assertEqual(plan["legs"], [HOST_LEG], "the host owner builds the changed host binding")
        self.assertTrue(plan["ptx_proof"]["source"].startswith("record "))
        self.assertEqual(plan["ptx_proof"]["built_from"], self.cur)
        self.assertIn("rebound from record", rr.render(plan))

    def test_an_unchanged_ptx_set_is_taken_from_the_leg_tree_under_the_evidence_root(self):
        ev = self.tmp / "evidence"
        legs = ev / "release" / "0.8.15" / self.cur[:12] / "legs"
        legs.mkdir(parents=True)
        rb = self.leg_tree()
        (legs / f"{PTX_LEG}.reused.json").write_text(json.dumps(dict(
            built_from=self.cur, release_build=str(rb), original_set_identity=self.original_identity(),
            tooling_digest="t" * 64)))
        plan = self.plan(self.record(self.host_change), evidence_root=ev)
        self.assertNotIn(PTX_LEG, plan["legs"])
        self.assertTrue(plan["ptx_proof"]["source"].startswith("leg tree "))
        self.assertEqual(plan["ptx_proof"]["release_build"], str(rb))

    def test_without_a_retained_proof_the_set_builds_and_says_what_is_missing(self):
        plan = self.plan(self.record(self.host_change, self.tmp / "empty-record"))
        rows = self.ptx_rows(plan)
        self.assertEqual({r["decision"] for r in rows}, {"BUILD"})
        reason = rows[0]["reason"]
        self.assertTrue(reason.startswith(rr.PTX_REASON))
        self.assertIn("no ptx-proof/ptx-proof.json in its record", reason)
        self.assertIn(PTX_LEG, plan["legs"])

    def test_a_proof_of_other_bytes_or_a_tampered_witness_is_refused(self):
        rec = self.retained_record(extensions={k: "f" * 64 for k in self.groups})
        rows = self.ptx_rows(self.plan(self.record(self.host_change, rec)))
        self.assertIn("names other PTX bytes than the release shipped", rows[0]["reason"])
        shutil.rmtree(rec)
        rec = self.retained_record()
        gz = rec / rr.PTX_PROOF_DIR / "PTX_BASELINE.json.gz"
        import gzip
        gz.write_bytes(gzip.compress(b'{"tampered": true}'))
        rows = self.ptx_rows(self.plan(self.record(self.host_change, rec)))
        self.assertEqual({r["decision"] for r in rows}, {"BUILD"})
        self.assertIn("is not the bytes ptx-proof.json names", rows[0]["reason"])

    def test_a_proof_of_another_commit_or_other_inputs_is_refused(self):
        # the record refuses to retain it ...
        with self.assertRaisesRegex(SystemExit, "incomplete or not of"):
            self.retained_record(built_from="e" * 40)
        # ... and a leg tree of it under the evidence root is refused by the plan
        ev = self.tmp / "evidence"
        legs = ev / "release" / "0.8.15" / self.cur[:12] / "legs"
        legs.mkdir(parents=True)
        (legs / f"{PTX_LEG}.reused.json").write_text(json.dumps(dict(
            built_from=self.cur, release_build=str(self.leg_tree(built_from="e" * 40)),
            original_set_identity=self.original_identity())))
        rows = self.ptx_rows(self.plan(self.record(self.host_change), evidence_root=ev))
        self.assertEqual({r["decision"] for r in rows}, {"BUILD"})
        self.assertIn("incomplete or not of", rows[0]["reason"])
        shutil.rmtree(self.tmp / "record", ignore_errors=True)
        shutil.rmtree(self.tmp / "ptx-leg")
        rec = self.tmp / "record"
        original = copy.deepcopy(self.original_identity())
        key = self.ptx[0].key
        original["bindings"][key] = "9" * 64
        original = {k: v for k, v in original.items() if k != "digest"}
        original["digest"] = pr.digest(original)
        rr.record_ptx_proof(rec, PTX_LEG, self.cur, self.leg_tree(), original, "t" * 64)
        rows = self.ptx_rows(self.plan(self.record(self.host_change, rec)))
        self.assertEqual({r["decision"] for r in rows}, {"BUILD"})
        self.assertIn("built other inputs for " + self.ptx[0].name, rows[0]["reason"])

    def test_a_runtime_change_or_no_host_owner_leg_builds_the_set(self):
        rec = self.retained_record()
        facts = rr.facts_for_commit(self.cur, self.cache, self.repo)
        prev = self.record(self.host_change, rec)
        doc = json.loads(Path(prev["identities"]).read_text())
        doc["runtime"]["identity"] = dict(rr.runtime_identity(facts), portable_math="0" * 64)
        Path(prev["identities"]).write_text(json.dumps(doc))
        rows = self.ptx_rows(self.plan(prev))
        self.assertEqual({r["decision"] for r in rows}, {"BUILD"})
        self.assertIn("runtime closure changes", rows[0]["reason"])
        # a Python-only release: no leg runs, so no host owner builds the PTX set's host bindings
        plan = self.plan(self.record(None, rec))
        self.assertEqual(plan["legs"], [PTX_LEG])
        self.assertIn("no %s leg runs" % HOST_LEG, self.ptx_rows(plan)[0]["reason"])


def merged_wheel(path, commit, groups, index_dir, index, runtime=b"runtime", index_bytes=None):
    """A published merged wheel: `groups` {archive path: bytes}, one index
    {"format": 1, "files": index} in archive directory `index_dir`."""
    payload = dict(source_commit=commit, extensions={}, merged_indexes={}, host_native={}, runtime_sha256={},
                   runtime_layout="shared", binding_layout="merged")
    with zipfile.ZipFile(path, "w") as z:
        for member, data in groups.items():
            z.writestr(member, data)
            payload["extensions"][member] = sha(data)
        data = index_bytes or json.dumps(dict(format=1, files=index)).encode()
        z.writestr(index_dir + "/_mojolearn_merged.json", data)
        payload["merged_indexes"][index_dir + "/_mojolearn_merged.json"] = dict(sha256=sha(data), files=index)
        z.writestr("mojolearn/.libs/runtime.so", runtime)
        payload["runtime_sha256"]["mojolearn/.libs/runtime.so"] = sha(runtime)
        z.writestr("mojolearn-0.8.15.dist-info/LINUX_PAYLOAD.json", json.dumps(payload))
    return path


class MergedAssemblyTests(unittest.TestCase):
    """A merged published file is the unit of reuse."""

    PREV = "b" * 40
    INDEX = {"_mojolearn_native.so": ["_mojolearn_a1", "_mojolearn_a2"], "_mojolearn_native_x.so": ["_mojolearn_b1"]}

    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.addCleanup(self._t.cleanup)
        self.tmp = Path(self._t.name)

    def fixture(self, vendor, arch, build=("_mojolearn_b1",)):
        base = f"mojolearn/{vendor}/{arch}/identical"
        self.published = {f"{base}/_mojolearn_native.so": self.A, f"{base}/_mojolearn_native_x.so": elf_stub([]) + b"PUBLISHED B"}
        whl = merged_wheel(self.tmp / "prev.whl", self.PREV, self.published, base, self.INDEX)
        prev = dict(version="0.8.15", source_commit=self.PREV, linux=dict(wheel=whl.name, sha256=rr.sha256(whl)))
        rows = [dict(target=rr.LINUX, vendor=vendor, arch=arch, tier="identical", name=n, key=f"identical/{n}",
                     set_rel=f"identical/{n}.so", archive_path=f"{base}/{n}.so",
                     decision="BUILD" if n in build else "REUSE", reason="changed: closure" if n in build else "x",
                     identity_digest=n[-2:] * 32)
                for n in ("_mojolearn_a1", "_mojolearn_a2", "_mojolearn_b1")]
        plan = dict(commit="c" * 40, previous=prev, rows=rows, runtime=dict(decision="REUSE"), legs=[])
        return whl, plan

    A = elf_stub([]) + b"PUBLISHED A"

    def leg(self, vendor, arch, a=None, index=None):
        leg = self.tmp / "leg" / vendor / arch
        (leg / "identical").mkdir(parents=True)
        (leg / "identical/_mojolearn_native.so").write_bytes(self.A if a is None else a)
        (leg / "identical/_mojolearn_native_x.so").write_bytes(b"LEG B")
        (leg / "identical/_mojolearn_merged.json").write_text(json.dumps(dict(format=1, files=index or self.INDEX)))
        (leg / ".libs").mkdir()
        (leg / ".libs/runtime.so").write_bytes(b"runtime")
        return leg

    def assemble(self, plan, whl, legs, vendor, arch, say=lambda *_: None):
        with patch.object(rr, "LINUX_SETS", ((vendor, arch),)):
            return rr.assemble_linux(plan, whl, legs, self.tmp / "sets", say=say)[f"{vendor}/{arch}"]

    def test_a_file_whose_every_binding_reuses_is_published_and_a_built_one_is_the_legs(self):
        whl, plan = self.fixture("cuda", "sm_89")
        lines = []
        sdir = self.assemble(plan, whl, {"cuda-sm_89": self.leg("cuda", "sm_89")}, "cuda", "sm_89", lines.append)
        self.assertEqual((sdir / "identical/_mojolearn_native.so").read_bytes(), self.A)
        self.assertEqual((sdir / "identical/_mojolearn_native_x.so").read_bytes(), b"LEG B")
        files = json.loads((sdir / "reuse.json").read_text())["files"]
        self.assertEqual(files["identical/_mojolearn_native.so"]["binding_identities"],
                         {"_mojolearn_a1": "a1" * 32, "_mojolearn_a2": "a2" * 32})
        self.assertEqual(files["identical/_mojolearn_native.so"]["rebuilt_sha256"], sha(self.A))
        self.assertNotIn("identical/_mojolearn_native_x.so", files)
        self.assertNotIn("identical/_mojolearn_merged.json", files, "the leg's index of the same map ships")

    def test_an_nvidia_rebuild_with_other_bytes_packs_the_legs_build(self):
        whl, plan = self.fixture("cuda", "sm_89")
        sdir = self.assemble(plan, whl, {"cuda-sm_89": self.leg("cuda", "sm_89", a=b"OTHER A")}, "cuda", "sm_89")
        self.assertEqual((sdir / "identical/_mojolearn_native.so").read_bytes(), b"OTHER A",
                         "a device file the leg rebuilt with other bytes packs the leg's build, never the published copy")
        files = json.loads((sdir / "reuse.json").read_text())["files"]
        self.assertNotIn("identical/_mojolearn_native.so", files)

    def test_a_leg_index_of_another_map_is_refused(self):
        whl, plan = self.fixture("cuda", "sm_89")
        other = {"_mojolearn_native.so": ["_mojolearn_a1", "_mojolearn_a2", "_mojolearn_b1"]}
        with self.assertRaisesRegex(SystemExit, "maps other files"):
            self.assemble(plan, whl, {"cuda-sm_89": self.leg("cuda", "sm_89", index=other)}, "cuda", "sm_89")

    def test_with_no_leg_every_file_and_the_index_come_from_the_wheel(self):
        whl, plan = self.fixture("hip", "gfx942", build=())
        sdir = self.assemble(plan, whl, {}, "hip", "gfx942")
        files = json.loads((sdir / "reuse.json").read_text())["files"]
        self.assertEqual({k for k in files if k.startswith("identical/")},
                         {"identical/_mojolearn_native.so", "identical/_mojolearn_native_x.so",
                          "identical/_mojolearn_merged.json"})
        for rel, rec in files.items():
            self.assertEqual(rr.sha256(sdir / rel), rec["sha256"])

    def test_a_file_serving_a_binding_to_build_needs_a_leg(self):
        whl, plan = self.fixture("hip", "gfx942")
        with self.assertRaisesRegex(SystemExit, "no leg ran"):
            self.assemble(plan, whl, {}, "hip", "gfx942")

    def test_a_mixed_ptx_set_ships_a_baseline_naming_exactly_its_packed_bytes(self):
        whl, plan = self.fixture("cuda", "sm_80")
        leg = self.leg("cuda", "sm_80")
        files = []
        for name, members in self.INDEX.items():
            data = (leg / "identical" / name).read_bytes()
            files.append(dict(file="identical/" + name, sha256=sha(data), numeric_mode="identical", members=members,
                              ptx_modules=[dict(target="sm_80", ptx_isa="8.1", approx={}, sha256="0" * 64)]))
        index = leg / "identical/_mojolearn_merged.json"
        (leg / pack_wheel.gpu_plugins.BASELINE_MANIFEST).write_text(json.dumps(dict(
            schema="mojolearn.ptx-set.v2", code_format="ptx", vendor="cuda", target="sm_80",
            min_compute_capability=[8, 0], source_commit="c" * 40, source_dirty=False, mojo_version="test",
            files=files, errors=[], indexes=[dict(file="identical/_mojolearn_merged.json", sha256=rr.sha256(index))])))
        sdir = self.assemble(plan, whl, {PTX_LEG: leg}, "cuda", "sm_80")
        doc = json.loads((sdir / pack_wheel.gpu_plugins.BASELINE_MANIFEST).read_text())
        packed = {p.relative_to(sdir).as_posix(): rr.sha256(p) for p in sdir.glob("identical/*.so")}
        pack_wheel.gpu_plugins.validate_baseline_manifest(
            doc, packed, indexes={"identical/_mojolearn_merged.json": rr.sha256(sdir / "identical/_mojolearn_merged.json")})
        self.assertEqual(json.loads((sdir / "reuse.json").read_text())["files"]["identical/_mojolearn_native.so"]["sha256"],
                         sha(self.A))


class PublishedPtxSetTests(unittest.TestCase):
    """The published PTX set taken whole from a RECORD-ONLY proof: the original
    witnesses from ptx-proof/, the bytes from the published wheel, the host
    bindings from the host owner, the manifest rebound and admitted."""

    # test_release_reuse_ptx's retained leg and host owner, on this case
    fixture = ptx_fixtures.PTXReuse.fixture
    host_owner_fixture = ptx_fixtures.PTXReuse.host_owner_fixture

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.host_from = self.host_owner_fixture()

    def wheel(self):
        src = self.source
        groups = {pr.PREFIX + p.relative_to(src).as_posix(): p.read_bytes() for p in src.glob("identical/*.so")}
        index = json.loads((src / "identical" / pr.INDEX).read_text())["files"]
        return merged_wheel(self.root / "prev.whl", ptx_fixtures.OLD, groups, pr.PREFIX + "identical", index,
                            runtime=(src / ".libs/runtime.so").read_bytes(),
                            index_bytes=(src / "identical" / pr.INDEX).read_bytes())

    def test_a_record_only_proof_is_materialized_rebound_and_admitted(self):
        whl = self.wheel()
        rec = self.root / "record"
        rr.record_ptx_proof(rec, PTX_LEG, ptx_fixtures.OLD, self.origin["release_build"],
                            self.origin["original_set_identity"], "c" * 64)
        shutil.rmtree(self.origin["release_build"])        # nothing of the leg tree survives
        proof, why = rr.ptx_proof_source(dict(version="0.8.15", source_commit=ptx_fixtures.OLD, record_dir=str(rec)))
        self.assertIsNone(why)
        self.assertEqual(proof["proof_sha256"], self.origin["proof_sha256"])
        rb = rr.materialize_ptx_leg(proof, whl, self.root / "materialized" / "release-build")
        sdir = rb / "build/sets/cuda/sm_80"
        self.assertEqual(rr.sha256(rb / rr.PTX_WITNESSES["proof"]), self.origin["proof_sha256"])
        origin = dict(self.origin, release_build=str(rb))
        with patch.object(rr, "LINUX_SETS", (rr.PTX_SET,)):
            staged = rr.assemble_linux(self.plan, None, {}, self.root / "assembled", say=lambda _: None,
                                       leg_reuse={PTX_LEG: dict(dir=sdir, origin=origin,
                                                                host_from=self.host_from)})["cuda/sm_80"]
        manifest = json.loads((staged / pr.MANIFEST).read_text())
        self.assertEqual(manifest["source_commit"], ptx_fixtures.NEW)
        reuse = pack_wheel.load_reuse(staged, "cuda", "sm_80")
        self.assertTrue(pr.validate_admission(manifest, staged, reuse["legs"]))
        self.assertEqual((staged / "host/_mojolearn_native_host.so").read_bytes(), b"NEW host bytes")

    def test_a_wheel_without_the_proofs_bytes_is_refused(self):
        whl = self.wheel()
        rec = self.root / "record"
        rr.record_ptx_proof(rec, PTX_LEG, ptx_fixtures.OLD, self.origin["release_build"],
                            self.origin["original_set_identity"], "c" * 64)
        proof, _ = rr.ptx_proof_source(dict(version="0.8.15", source_commit=ptx_fixtures.OLD, record_dir=str(rec)))
        proof = dict(proof, release_build=None)
        bad = self.root / "bad.whl"
        with zipfile.ZipFile(whl) as z, zipfile.ZipFile(bad, "w") as out:
            for n in z.namelist():
                out.writestr(n, b"other bytes" if n.endswith("identical/_mojolearn_native.so") else z.read(n))
        with self.assertRaises(SystemExit):
            rr.materialize_ptx_leg(proof, bad, self.root / "m2")


if __name__ == "__main__":
    unittest.main()
