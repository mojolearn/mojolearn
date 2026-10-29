import hashlib
import json
from pathlib import Path
import tempfile
import unittest
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

    def test_runtime_refresh_requires_new_native_evidence(self):
        for name in ('autograd/ops_api.mojo', 'python/mojolearn/_backend.py', 'tools/merged_bindings.py', 'pixi.lock'):
            with self.subTest(name=name), patch.object(candidate.common, 'git', return_value=name):
                with self.assertRaisesRegex(ValueError, 'exceeds packaging-only'):
                    candidate.validate_refresh(self.root, 'original', 'packaged')


if __name__ == "__main__":
    unittest.main()
