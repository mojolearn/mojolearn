# SPDX-License-Identifier: Apache-2.0
"""Artifact admission negative controls; no compiler/product runtime involved."""
import base64
import csv
import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import types
import unittest
import zipfile

SPEC = importlib.util.spec_from_file_location("installed_native_audit", Path(__file__).with_name("qualify_native_runtime_installed.py"))
q = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(q)


def wheel(path, *, tamper=False, extra=False):
    files = {"mojolearn/__init__.py": b"source\n", "mojolearn/host/example.so": b"native-bytes"}
    rows = []
    for name, raw in files.items():
        sha = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip("=")
        rows.append((name, "sha256=" + sha, str(len(raw))))
    record = "mojolearn_core-0.dist-info/RECORD"
    out = io.StringIO();csv.writer(out).writerows(rows + [(record, "", "")])
    if tamper:files["mojolearn/host/example.so"] = b"tampered"
    if extra:files["mojolearn/unchecked.py"] = b"unrecorded"
    with zipfile.ZipFile(path, "w") as archive:
        for name, raw in files.items():archive.writestr(name, raw)
        archive.writestr(record, out.getvalue())


class InstalledNativeAudit(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_every_wheel_file_and_native_byte_is_checked(self):
        path = self.root / "test.whl";wheel(path)
        payload, records = q.wheel_inventory([path])
        self.assertIn("mojolearn/host/example.so", payload)
        self.assertEqual(records[0]["sha256"], q.digest(path))
        wheel(path, tamper=True)
        with self.assertRaisesRegex(ValueError, "RECORD mismatch"):
            q.wheel_inventory([path])
        wheel(path, extra=True)
        with self.assertRaisesRegex(ValueError, "every wheel file"):
            q.wheel_inventory([path])

    def test_duplicate_split_ownership_refuses(self):
        one, two = self.root / "one.whl", self.root / "two.whl"
        wheel(one);wheel(two)
        with self.assertRaisesRegex(ValueError, "two wheels own"):
            q.wheel_inventory([one, two])

    def modules(self):
        prefix = self.root / "venv";site = prefix / "lib/python3.14/site-packages"
        package = site / "mojolearn";package.mkdir(parents=True)
        init = package / "__init__.py";init.write_text("source\n")
        native = package / "host/example.so";native.parent.mkdir();native.write_bytes(b"native-bytes")
        base = types.ModuleType("mojolearn");base.__file__ = str(init)
        ext = types.ModuleType("_mojolearn_example");ext.__file__ = str(native)
        base.runtime = ext  # Also inspect cached extensions absent from sys.modules.
        payload = {str(p.relative_to(site)):dict(sha256=q.digest(p)) for p in [init, native]}
        tests = self.root / "frozen-tests";tests.mkdir()
        return prefix, payload, tests, {"mojolearn":base}

    def test_only_installed_product_paths_and_hashes_are_admitted(self):
        prefix, payload, tests, modules = self.modules()
        found = q.installed_module_inventory(modules, prefix, payload, tests)
        self.assertTrue(found["mojolearn/host/example.so"]["native"])
        Path(modules["mojolearn"].runtime.__file__).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "bytes differ"):
            q.installed_module_inventory(modules, prefix, payload, tests)

    def test_product_source_shadow_and_unshipped_native_module_refuse(self):
        prefix, payload, tests, modules = self.modules()
        source = self.root / "source.py";source.write_text("source\n")
        modules["mojolearn"].__file__ = str(source)
        with self.assertRaisesRegex(ValueError, "outside installed"):
            q.installed_module_inventory(modules, prefix, payload, tests)
        modules["mojolearn"].__file__ = str(prefix / "lib/python3.14/site-packages/mojolearn/__init__.py")
        del payload["mojolearn/host/example.so"]
        with self.assertRaisesRegex(ValueError, "absent from wheel"):
            q.installed_module_inventory(modules, prefix, payload, tests)

    def test_only_frozen_test_namespace_can_live_outside_install(self):
        prefix, payload, tests, modules = self.modules()
        helper = tests / "test_helper.py";helper.write_text("oracle\n")
        test = types.ModuleType("mojolearn.tests.test_helper");test.__file__ = str(helper)
        modules[test.__name__] = test
        q.installed_module_inventory(modules, prefix, payload, tests)
        test.__file__ = str(self.root / "unchecked-test.py")
        with self.assertRaisesRegex(ValueError, "test namespace escaped"):
            q.installed_module_inventory(modules, prefix, payload, tests)

    def test_original_failure_and_skip_are_kept_distinct(self):
        path = self.root / "tests.xml"
        path.write_text('<testsuites><testsuite><testcase name="ok"/><testcase name="bad"><failure>original numerical mismatch</failure></testcase><testcase name="missing"><skipped message="no physical GPU"/></testcase></testsuite></testsuites>')
        report = q.junit_summary(path)
        self.assertEqual((report["total"], report["passed"]), (3, 1))
        self.assertIn("original numerical mismatch", report["failures"][0]["reason"])
        self.assertEqual(report["skipped"][0]["reason"], "no physical GPU")


if __name__ == "__main__":
    unittest.main()
