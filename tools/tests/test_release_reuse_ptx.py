"""PTX metadata refreezes preserve proved binaries; absent/stale proof refuses."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "packaging/linux"))
import release_reuse as rr
import pack_wheel
import ptx_reuse as pr

OLD = "a" * 40
NEW = "b" * 40


def write(path, doc):
    path.write_text(json.dumps(doc, indent=2) + "\n")


class PTXReuse(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def fixture(self, merged=True):
        rb = self.root / "original-build"
        source = rb / "build/sets/cuda/sm_80"
        source.mkdir(parents=True)
        identities = dict(target=rr.LINUX, toolchain={"packages": {"mojo": "26.5"}},
                          builders={"build": "f" * 64}, image={"digest": "image"},
                          closure="source", flags={"target": "sm_80"})
        rows, files, ext, host, indexes = [], [], {}, {}, {}
        for tier, member in (("identical", "_mojolearn_gpu"), ("host", "_mojolearn_test_host")):
            directory = source / tier
            directory.mkdir()
            binary = ("_mojolearn_native_host.so" if tier == "host" else "_mojolearn_native.so") if merged else member + ".so"
            p = directory / binary
            p.write_bytes(("retained native bytes " + tier).encode())
            rel = p.relative_to(source).as_posix()
            (host if tier == "host" else ext)[pr.PREFIX + rel] = pr.file_hash(p)
            if merged:
                index = directory / pr.INDEX
                members = {binary: [member]}
                write(index, dict(format=1, files=members))
                indexes[pr.PREFIX + index.relative_to(source).as_posix()] = dict(sha256=pr.file_hash(index), files=members)
            if tier != "host":
                row = dict(file=rel, sha256=pr.file_hash(p), numeric_mode=tier,
                           ptx_modules=[dict(target="sm_80", ptx_isa="8.1", approx={}, sha256="f" * 64)])
                if merged:
                    row["members"] = [member]
                files.append(row)
            ident = copy.deepcopy(identities)
            rows.append(dict(target=rr.LINUX, vendor="cuda", arch="sm_80", tier=tier, name=member,
                             key=tier + "/" + member, set_rel=tier + "/" + member + ".so",
                             archive_path=pr.PREFIX + tier + "/" + member + ".so", decision="BUILD",
                             identity=ident, identity_digest=rr.digest_of(ident)))
        runtime = dict(identities)
        self.plan = dict(commit=NEW, previous=None, rows=rows,
                         runtime=dict(decision="BUILD", identity=runtime, identity_digest=rr.digest_of(runtime)))
        (source / ".libs").mkdir()
        lib = source / ".libs/runtime.so"
        lib.write_bytes(b"original runtime")
        write(source / "manifest.json", dict(staged_libs=[dict(name=lib.name, sha256=pr.file_hash(lib))]))
        manifest = dict(schema="mojolearn.ptx-set.v2", code_format="ptx", vendor="cuda", target="sm_80",
                        min_compute_capability=[8, 0], source_commit=OLD, source_dirty=False,
                        mojo_version="original compiler", files=files, errors=[])
        if merged:
            manifest["indexes"] = [dict(file=k.removeprefix(pr.PREFIX), sha256=v["sha256"])
                                   for k, v in indexes.items() if "/host/" not in k]
        write(source / pr.MANIFEST, manifest)
        proof = dict(schema="mojolearn.linux.build-provenance.v1", complete=True, build_exit=0,
                     action="build", source_commit=OLD, extensions=ext, host_extension=host,
                     merged_indexes=indexes, binding_layout="merged" if merged else "per-binding")
        self.proof_path = rb / "build/build-provenance.json"
        write(self.proof_path, proof)
        current = rr.set_identity(self.plan, "cuda", "sm_80")
        original = {k: v for k, v in current.items() if k != "digest"}
        original["source_commit"] = OLD  # Original, pre-repair provenance has the legacy salt.
        original["digest"] = pr.digest(original)
        self.origin = dict(source_commit=OLD, leg="cuda-sm_80", admitted_for=NEW,
                           proof_sha256=pr.file_hash(self.proof_path), set_identity_digest=current["digest"],
                           tooling_digest="c" * 64, release_build=str(rb), original_set_identity=original)
        self.source = source
        return source

    def assemble(self):
        with patch.object(rr, "LINUX_SETS", (rr.PTX_SET,)):
            return rr.assemble_linux(self.plan, None, {}, self.root / "assembled", say=lambda _: None,
                                     leg_reuse={"cuda-sm_80": dict(dir=self.source, origin=self.origin)})["cuda/sm_80"]

    def test_commit_only_refreeze_preserves_bytes_and_raw_build_provenance(self):
        self.fixture()
        original_manifest = (self.source / pr.MANIFEST).read_bytes()
        original_proof = self.proof_path.read_bytes()
        original_files = {p.relative_to(self.source): p.read_bytes() for p in self.source.rglob("*") if p.is_file()}
        staged = self.assemble()
        manifest = json.loads((staged / pr.MANIFEST).read_text())
        self.assertEqual(manifest["source_commit"], NEW)
        admission = manifest["release_admission"]
        self.assertEqual(admission["original_manifest"]["utf8"].encode(), original_manifest)
        self.assertEqual(admission["original_build_proof"]["utf8"].encode(), original_proof)
        self.assertEqual(json.loads(admission["original_manifest"]["utf8"])["source_commit"], OLD)
        for rel, raw in original_files.items():
            self.assertEqual((self.source / rel).read_bytes(), raw, "original evidence was modified")
            if rel.name != pr.MANIFEST:
                self.assertEqual((staged / rel).read_bytes(), raw)
        reuse = pack_wheel.load_reuse(staged, "cuda", "sm_80")
        self.assertNotIn(pr.MANIFEST, reuse["files"], "fresh metadata must not claim identical bytes")
        self.assertTrue(pr.validate_admission(manifest, staged, reuse["legs"]))

    def test_all_set_identities_ignore_only_release_commit(self):
        self.fixture()
        original = rr.set_identity(self.plan, *rr.PTX_SET)
        self.plan["commit"] = OLD
        self.assertEqual(rr.set_identity(self.plan, *rr.PTX_SET), original)
        self.assertEqual(pr.normalized_set_identity(self.origin["original_set_identity"]), original)
        for field in ("closure", "toolchain", "flags", "image", "builders"):
            with self.subTest(field=field):
                plan = copy.deepcopy(self.plan)
                plan["rows"][0]["identity"][field] = {"changed": True}
                plan["rows"][0]["identity_digest"] = rr.digest_of(plan["rows"][0]["identity"])
                self.assertNotEqual(rr.set_identity(plan, *rr.PTX_SET), original)
        self.plan["runtime"]["identity_digest"] = "e" * 64
        self.assertNotEqual(rr.set_identity(self.plan, *rr.PTX_SET), original)

    def test_corrupt_or_incomplete_saved_identity_is_not_normalized(self):
        self.fixture()
        for field in ("runtime", "bindings", "source_commit", "digest"):
            saved = copy.deepcopy(self.origin["original_set_identity"])
            saved.pop(field)
            self.assertIsNone(pr.normalized_set_identity(saved), field)
        saved = copy.deepcopy(self.origin["original_set_identity"])
        saved["runtime"] = "e" * 64
        self.assertIsNone(pr.normalized_set_identity(saved), "stale digest must fail before normalization")
        for changes in ({"unexpected": True}, {"set": "cuda/sm_89"}):
            saved = dict(self.origin["original_set_identity"], **changes)
            saved.pop("digest")
            saved["digest"] = pr.digest(saved)
            self.assertIsNone(pr.normalized_set_identity(saved))

    def test_source_toolchain_flag_or_runtime_mismatch_refuses(self):
        self.fixture()
        for field in ("closure", "toolchain", "flags"):
            with self.subTest(field=field):
                plan = copy.deepcopy(self.plan)
                self.plan["rows"][0]["identity"][field] = {"changed": True}
                self.plan["rows"][0]["identity_digest"] = rr.digest_of(self.plan["rows"][0]["identity"])
                with self.assertRaisesRegex(SystemExit, "set identity"):
                    self.assemble()
                self.plan = plan
        self.plan["runtime"]["identity_digest"] = "e" * 64
        with self.assertRaisesRegex(SystemExit, "set identity"):
            self.assemble()

    def test_missing_proof_refuses(self):
        self.fixture()
        self.proof_path.unlink()
        with self.assertRaisesRegex(SystemExit, "cannot admit completed PTX leg"):
            self.assemble()

    def test_corrupt_or_stale_proof_refuses(self):
        self.fixture()
        original = json.loads(self.proof_path.read_text())
        for change in ({"complete": False}, {"build_exit": 1}, {"source_commit": NEW}, {"action": "check"},
                       {"extensions": {}}, {"host_extension": {}}, {"merged_indexes": {}},
                       {"binding_layout": "per-binding"}):
            with self.subTest(change=change):
                write(self.proof_path, dict(original, **change))
                self.origin["proof_sha256"] = pr.file_hash(self.proof_path)
                with self.assertRaises(SystemExit):
                    self.assemble()
        write(self.proof_path, original)
        self.origin["proof_sha256"] = "0" * 64
        with self.assertRaisesRegex(SystemExit, "stale original build proof"):
            self.assemble()

    def test_changed_binary_index_host_or_runtime_refuses(self):
        self.fixture()
        for rel in ("identical/_mojolearn_native.so", "identical/" + pr.INDEX,
                    "host/_mojolearn_native_host.so", "host/" + pr.INDEX, ".libs/runtime.so"):
            with self.subTest(rel=rel):
                p = self.source / rel
                raw = p.read_bytes()
                p.write_bytes(raw + b" ")
                with self.assertRaises(SystemExit):
                    self.assemble()
                p.write_bytes(raw)

    def test_original_manifest_cannot_claim_another_source_or_dirty_build(self):
        self.fixture()
        path = self.source / pr.MANIFEST
        original = json.loads(path.read_text())
        for changes in ({"source_commit": NEW}, {"source_dirty": True}):
            write(path, dict(original, **changes))
            with self.assertRaisesRegex(SystemExit, "differs or is dirty"):
                self.assemble()

    def test_missing_original_identity_or_tooling_refuses(self):
        self.fixture()
        for key in ("original_set_identity", "tooling_digest", "proof_sha256"):
            value = self.origin.pop(key)
            with self.assertRaises(SystemExit):
                self.assemble()
            self.origin[key] = value

    def test_packer_rechecks_embedded_proof_after_assembly(self):
        self.fixture(merged=False)
        staged = self.assemble()
        # Small fixture inventories, while retaining the real loader/proof checks.
        with patch.object(pack_wheel, "tier_names", return_value=["_mojolearn_gpu"]), \
                patch.object(pack_wheel, "HOST_NAMES", ("_mojolearn_test_host",)), \
                patch.object(pack_wheel, "release_modes", return_value=("identical",)):
            self.assertEqual(len(pack_wheel.load_set(staged.parent)), 1)
            path = staged / pr.MANIFEST
            doc = json.loads(path.read_text())
            missing = dict(doc)
            del missing["release_admission"]
            write(path, missing)
            with self.assertRaisesRegex(SystemExit, "missing release admission"):
                pack_wheel.load_set(staged.parent)
            doc["release_admission"]["original_build_proof"]["utf8"] += " "
            write(path, doc)
            with self.assertRaisesRegex(SystemExit, "Raw witness SHA mismatch"):
                pack_wheel.load_set(staged.parent)

    def test_envelope_cannot_rewrite_compiler_or_arithmetic_evidence(self):
        self.fixture()
        staged = self.assemble()
        doc = json.loads((staged / pr.MANIFEST).read_text())
        doc["mojo_version"] = "new compiler that did not build this"
        with self.assertRaisesRegex(ValueError, "original arithmetic/build metadata was changed"):
            pr.validate_admission(doc, staged, {"cuda-sm_80": self.origin})


    # --------------------------------------------- HOST_OWNER_ADMISSION
    def host_owner_fixture(self):
        """The PTX leg retained; the host rows of the plan changed; the host
        owner (cuda-sm_89, built later) carries the new host bytes."""
        self.fixture()
        for row in self.plan["rows"]:
            if row["tier"] == "host":
                row["identity"] = dict(row["identity"], closure="changed host source")
                row["identity_digest"] = rr.digest_of(row["identity"])
        current = rr.set_identity(self.plan, *rr.PTX_SET)
        self.origin["set_identity_digest"] = current["digest"]
        orb = self.root / "owner-build"
        owner_set = orb / "build/sets/cuda/sm_89"
        (owner_set / "host").mkdir(parents=True)
        (owner_set / ".libs").mkdir()
        (owner_set / ".libs/runtime.so").write_bytes(b"original runtime")
        binary = owner_set / "host/_mojolearn_native_host.so"
        binary.write_bytes(b"NEW host bytes")
        members = {binary.name: ["_mojolearn_test_host"]}
        write(owner_set / "host" / pr.INDEX, dict(format=1, files=members))
        prefix = pr.HOST_OWNER_PREFIX
        owner_proof = dict(schema="mojolearn.linux.build-provenance.v1", complete=True, build_exit=0,
                           action="build", source_commit=NEW, extensions={},
                           host_extension={prefix + "host/" + binary.name: pr.file_hash(binary)},
                           merged_indexes={prefix + "host/" + pr.INDEX: dict(
                               sha256=pr.file_hash(owner_set / "host" / pr.INDEX), files=members)},
                           binding_layout="merged")
        write(orb / "build/build-provenance.json", owner_proof)
        self.owner_set = owner_set
        self.owner_origin = dict(source_commit=NEW, leg="cuda-sm_89", admitted_for=NEW,
                                 proof_sha256=pr.file_hash(orb / "build/build-provenance.json"),
                                 set_identity_digest="d" * 64, tooling_digest="c" * 64, release_build=str(orb))
        return dict(name="cuda-sm_89", dir=owner_set, origin=self.owner_origin)

    def assemble_owner(self, host_from):
        with patch.object(rr, "LINUX_SETS", (rr.PTX_SET,)):
            return rr.assemble_linux(self.plan, None, {}, self.root / "assembled", say=lambda _: None,
                                     leg_reuse={"cuda-sm_80": dict(dir=self.source, origin=self.origin,
                                                                   host_from=host_from)})["cuda/sm_80"]

    def test_host_owner_admission_ships_owner_host_and_keeps_ptx_proof(self):
        host_from = self.host_owner_fixture()
        self.assertIn("HOST_OWNER_ADMISSION", Path(pr.__file__).read_text())
        staged = self.assemble_owner(host_from)
        self.assertEqual((staged / "host/_mojolearn_native_host.so").read_bytes(), b"NEW host bytes")
        manifest = json.loads((staged / pr.MANIFEST).read_text())
        reuse = pack_wheel.load_reuse(staged, "cuda", "sm_80")
        self.assertEqual(set(reuse["legs"]), {"cuda-sm_80", "cuda-sm_89"})
        self.assertTrue(pr.validate_admission(manifest, staged, reuse["legs"]))
        # without the owner's origin among the legs it is refused
        with self.assertRaisesRegex(ValueError, "host owner differs"):
            pr.validate_admission(manifest, staged, {"cuda-sm_80": reuse["legs"]["cuda-sm_80"]})
        # a changed GPU byte is still refused by the PTX leg's own proof
        gpu = staged / "identical/_mojolearn_native.so"
        gpu.write_bytes(b"tampered")
        with self.assertRaises(ValueError):
            pr.validate_admission(manifest, staged, reuse["legs"])

    def test_host_owner_admission_refuses_a_changed_gpu_identity(self):
        host_from = self.host_owner_fixture()
        for row in self.plan["rows"]:
            if row["tier"] != "host":
                row["identity"] = dict(row["identity"], closure="changed GPU source")
                row["identity_digest"] = rr.digest_of(row["identity"])
        self.origin["set_identity_digest"] = rr.set_identity(self.plan, *rr.PTX_SET)["digest"]
        with self.assertRaisesRegex(SystemExit, "GPU binding identity"):
            self.assemble_owner(host_from)

    def test_host_owner_admission_refuses_host_bytes_not_in_the_owner_proof(self):
        host_from = self.host_owner_fixture()
        (self.owner_set / "host/_mojolearn_native_host.so").write_bytes(b"not the proved bytes")
        with self.assertRaisesRegex(SystemExit, "host owner's build proof"):
            self.assemble_owner(host_from)

    def test_host_owner_admission_refuses_another_runtime_closure(self):
        host_from = self.host_owner_fixture()
        (self.owner_set / ".libs/runtime.so").write_bytes(b"another runtime")
        with self.assertRaisesRegex(SystemExit, "runtime closure differs"):
            self.assemble_owner(host_from)

    def test_without_host_owner_a_changed_host_identity_still_refuses(self):
        self.host_owner_fixture()
        with self.assertRaises(SystemExit):
            self.assemble()


if __name__ == "__main__":
    unittest.main()
