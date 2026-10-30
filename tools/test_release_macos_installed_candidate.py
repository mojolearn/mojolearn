import hashlib
import json
import subprocess
from pathlib import Path
import tempfile
import unittest
import tarfile
from unittest.mock import patch

import release_macos_installed_candidate as candidate


class RetainedCandidateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "bindings").mkdir()
        self.manifest = {"groups": {"host": {"a": ["_mojolearn_a_host"]},
                                    "identical": {"a": ["_mojolearn_a"]}}}
        self.write_manifest()
        self.records = {"host/_mojolearn_a_host.so": {}, "identical/_mojolearn_a.so": {}}

    def write_manifest(self):
        (self.root / "bindings/development_groups.json").write_text(json.dumps(self.manifest))

    def corpus_repository(self):
        from verify_linux_surface_qualification import CORPUS_CASES
        repo = self.root / 'repo'
        repo.mkdir()
        def git(*args):
            return subprocess.check_output(['git', '-C', str(repo), *args], stderr=subprocess.DEVNULL).decode().strip()
        git('init', '-q')
        git('config', 'user.email', 'fixture@example.invalid')
        git('config', 'user.name', 'Fixture')
        (repo / '.gitattributes').write_text('/mamba/corpus/** export-ignore\n')
        for case in CORPUS_CASES:
            directory = repo / 'mamba/corpus' / case
            directory.mkdir(parents=True)
            (directory / 'x.f32').write_bytes(b'frozen operand')
            (directory / 'ref.f64').write_bytes(b'frozen reference')
        git('add', '.')
        git('-c', 'core.hooksPath=/dev/null', 'commit', '-qm', 'corpus fixture')
        return repo, git('rev-parse', 'HEAD'), CORPUS_CASES

    def test_compressed_source_preserves_unexpanded_nonbuild_bytes(self):
        repo, _, _ = self.corpus_repository()
        payloads = {"python/setup.py": b"build input", "packaging/build.py": b"packaging input",
                    "bindings/entry.mojo": b"native source", "README.md": b"package readme",
                    "bench/result.json": b"historical result", "docs/design.md": b"design",
                    "experiments/input.bin": b"experiment bytes"}
        for name, content in payloads.items():
            path = repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
        subprocess.run(["git", "-C", str(repo), "-c", "core.hooksPath=/dev/null",
                        "commit", "-qm", "packaging inputs"], check=True)
        commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"]).decode().strip()
        (repo / "bench/result.json").write_bytes(b"dirty result")
        archive, stage = self.root / "source.tar.gz", self.root / "source"
        stage.mkdir()
        report = candidate.stage_frozen_source(repo, commit, archive, stage)
        with tarfile.open(archive) as stream:
            for name, content in payloads.items():
                self.assertEqual(stream.extractfile(name).read(), content)
                if name.split("/")[0] in report["excluded_top_level_directories"]:
                    self.assertFalse((stage / name).exists())
                    self.assertIn(name, report["unexpanded_nonbuild_members"])
                else:
                    self.assertEqual((stage / name).read_bytes(), content)
        self.assertEqual(report["complete_archive"]["sha256"], hashlib.sha256(archive.read_bytes()).hexdigest())

    def test_export_ignored_operands_restored_from_frozen_blobs(self):
        repo, commit, cases = self.corpus_repository()
        # A dirty checkout must not replace the frozen expectation.
        (repo / 'mamba/corpus' / cases[0] / 'x.f32').write_bytes(b'changed checkout')
        stage = self.root / 'stage'
        report = candidate.stage_smoke_corpora(repo, commit, stage)
        self.assertEqual(len(report['files']), 6)
        self.assertEqual(report['source_commit'], commit)
        for name, row in report['files'].items():
            content = (stage / name).read_bytes()
            self.assertNotEqual(content, b'changed checkout')
            self.assertEqual(hashlib.sha256(content).hexdigest(), row['sha256'])

    def test_staged_corpus_mutation_refused(self):
        repo, commit, cases = self.corpus_repository()
        stage = self.root / 'stage'
        candidate.stage_smoke_corpora(repo, commit, stage)
        (stage / 'mamba/corpus' / cases[0] / 'x.f32').write_bytes(b'wrong')
        with self.assertRaisesRegex(ValueError, 'differs from frozen blob'):
            candidate.stage_smoke_corpora(repo, commit, stage)

    def test_absent_frozen_case_refused(self):
        repo, commit, cases = self.corpus_repository()
        subprocess.run(['git', '-C', str(repo), 'rm', '-r', 'mamba/corpus/' + cases[-1]],
                       check=True, stdout=subprocess.DEVNULL)
        subprocess.run(['git', '-C', str(repo), '-c', 'core.hooksPath=/dev/null', 'commit', '-qm', 'remove case'], check=True)
        missing = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip()
        with self.assertRaisesRegex(ValueError, 'missing frozen smoke corpus'):
            candidate.stage_smoke_corpora(repo, missing, self.root / 'stage')

    def test_truthful_single_member_indexes(self):
        result = candidate.native_indexes(self.root, self.records)
        self.assertEqual(result["host"], {"format": 1, "files": {"_mojolearn_a_host.so": ["_mojolearn_a_host"]}})
        self.assertNotIn("layout", result["host"])

    def test_missing_group_refused(self):
        del self.records["host/_mojolearn_a_host.so"]
        with self.assertRaisesRegex(ValueError, "coverage"):
            candidate.native_indexes(self.root, self.records)

    def test_extra_binary_refused(self):
        self.records["host/_mojolearn_old.so"] = {}
        with self.assertRaisesRegex(ValueError, "coverage"):
            candidate.native_indexes(self.root, self.records)

    def test_multi_member_source_not_relabelled(self):
        self.manifest["groups"]["host"]["a"].append("_mojolearn_b_host")
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "single-member"):
            candidate.native_indexes(self.root, self.records)

    def test_duplicate_member_refused(self):
        self.manifest["groups"]["host"]["b"] = ["_mojolearn_a_host"]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            candidate.native_indexes(self.root, self.records)

    def test_adapter_compiler_location_reuse_requires_same_complete_facts(self):
        old = {"available": True, "compiler": "/old/worktree/mojo",
               "compiler_sha256": "a" * 64, "version": "pinned",
               "precompiled_packages": {"stdlib": "original"}}
        current = dict(old, compiler="/current/worktree/mojo")
        candidate.validate_adapter_toolchain(old, current)
        for changed in [dict(current, compiler_sha256="b" * 64),
                        dict(current, precompiled_packages={"stdlib": "changed"}),
                        dict(current, version="other")]:
            with self.subTest(changed=changed):
                with self.assertRaisesRegex(ValueError, "toolchain changed"):
                    candidate.validate_adapter_toolchain(old, changed)

    def test_adapter_compiler_location_without_content_identity_refused(self):
        with self.assertRaisesRegex(ValueError, "toolchain changed"):
            candidate.validate_adapter_toolchain({"available": True}, {"available": True})

    def adapter_snapshot(self):
        adapter = self.root / "_adapters/libmojolearn_adapter.so"
        adapter.parent.mkdir()
        adapter.write_bytes(b"retained adapter bytes")
        return adapter, {"native_and_runtime_files": {}, "adapter_files": {
            "_adapters/libmojolearn_adapter.so": {
                "sha256": hashlib.sha256(adapter.read_bytes()).hexdigest()}}}

    def test_separately_recorded_adapter_inventory(self):
        _, receipt = self.adapter_snapshot()
        candidate.validate_native_inventory(self.root, receipt)
        self.assertEqual(receipt["native_and_runtime_files"], {})

    def test_unrecorded_adapter_image_refused(self):
        adapter, receipt = self.adapter_snapshot()
        (adapter.parent / "unexpected.so").write_bytes(b"unrecorded")
        with self.assertRaisesRegex(ValueError, "unrecorded native"):
            candidate.validate_native_inventory(self.root, receipt)

    def test_changed_recorded_adapter_refused(self):
        adapter, receipt = self.adapter_snapshot()
        adapter.write_bytes(b"different compiled bytes")
        with self.assertRaisesRegex(ValueError, "differs"):
            candidate.validate_native_inventory(self.root, receipt)

    def test_adapter_inventory_cannot_admit_other_native_owners(self):
        _, receipt = self.adapter_snapshot()
        receipt["adapter_files"]["host/extra.so"] = {"sha256": "0" * 64}
        with self.assertRaisesRegex(ValueError, "non-adapter path"):
            candidate.validate_native_inventory(self.root, receipt)

    def test_matching_retained_bytes(self):
        (self.root / "image.so").write_bytes(b"compiled bytes")
        candidate.check_files(self.root, {"image.so": {"sha256": hashlib.sha256(b"compiled bytes").hexdigest()}})

    def test_stale_bytes_refused(self):
        (self.root / "image.so").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "differs"):
            candidate.check_files(self.root, {"image.so": hashlib.sha256(b"original").hexdigest()})

    def test_missing_bytes_refused(self):
        with self.assertRaisesRegex(ValueError, "differs"):
            candidate.check_files(self.root, {"missing.so": "0" * 64})

    def test_path_escape_refused(self):
        for value in ("../outside.so", "/outside.so", "a/../outside.so", "./image.so"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "unsafe"):
                candidate.member_path(self.root, value)

    def test_symlink_escape_refused(self):
        (self.root / "escape").symlink_to(self.root.parent, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "escapes"):
            candidate.member_path(self.root, "escape/image.so")

    def test_packaging_refresh_is_explicit(self):
        with patch.object(candidate.common, 'git', return_value='packaging/portable_math/wheel.py\n'):
            candidate.validate_refresh(self.root, 'original', 'packaged')

    def test_selected_source_must_be_the_checked_physical_checkout(self):
        with patch.object(candidate.common, 'revision', return_value='current'):
            candidate.validate_source_root(self.root, 'current')
            with self.assertRaisesRegex(ValueError, 'physical source checkout'):
                candidate.validate_source_root(self.root, 'different-archive')

    def test_runtime_refresh_requires_new_native_evidence(self):
        for name in ('autograd/ops_api.mojo', 'python/mojolearn/_backend.py', 'tools/merged_bindings.py', 'pixi.lock'):
            with self.subTest(name=name), patch.object(candidate.common, 'git', return_value=name):
                with self.assertRaisesRegex(ValueError, 'exceeds packaging-only'):
                    candidate.validate_refresh(self.root, 'original', 'packaged')


if __name__ == "__main__":
    unittest.main()
