"""No native execution: committed verification bytes and rejection paths."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import verification_repository as V


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL).decode().strip()


def seed_repo(root):
    pkg = root / "python/mojolearn"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "verify_reference").mkdir(exist_ok=True)
    col = root / V.RECORDS / "fixture/column.json"
    col.parent.mkdir(parents=True)
    col.write_text(json.dumps(dict(commit="a" * 40, vendor="cpu")))
    table = dict(records=[dict(dir="fixture", file="column.json", commit="a" * 40,
                              vendor="cpu", sha256=hashlib.sha256(col.read_bytes()).hexdigest())], cells={})
    (root / V.TABLE).write_text(json.dumps(table))
    (pkg / "host_surface.py").write_text("TRAINING_GPU_COLUMNS = []\nGBDT_CTR_MODELS_DIR = 'fixtures'\ndef training_gpu_column_record(): return 'fixture'\n")
    (pkg / "_verification_ctr_models.py").write_text("MODEL_SHA256 = {}\n")
    (pkg / "_verification_evidence_data.py").write_text("# historical fixture\n")
    git(root, "init", "-q")
    git(root, "config", "user.email", "fixture@example.invalid")
    git(root, "config", "user.name", "Fixture")
    git(root, "add", ".")
    git(root, "commit", "-qm", "fixture")
    return col


class RepositoryVerification(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "repo"
        self.column = seed_repo(self.root)

    def commit(self):
        git(self.root, "add", ".")
        git(self.root, "commit", "-qm", "change")

    def table(self, **changes):
        path = self.root / V.TABLE
        table = json.loads(path.read_text())
        table["records"][0].update(changes)
        path.write_text(json.dumps(table))
        self.commit()

    def test_committed_and_exported_payloads(self):
        self.assertIn("mojolearn/verify_reference/table.json", V.validate(self.root))
        exported = Path(self.temp.name) / "export"
        shutil.copytree(self.root, exported, ignore=shutil.ignore_patterns(".git"))
        V.validate(exported, git_root=self.root)

    def test_missing_record(self):
        self.column.unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            V.validate(self.root)

    def test_dirty_record(self):
        self.column.write_text("{}")
        with self.assertRaisesRegex(ValueError, "source commit"):
            V.validate(self.root)

    def test_checksum_and_provenance(self):
        self.table(sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "checksum"):
            V.validate(self.root)
        self.table(sha256=hashlib.sha256(self.column.read_bytes()).hexdigest(), commit="b" * 40)
        with self.assertRaisesRegex(ValueError, "provenance"):
            V.validate(self.root)

    def test_external_record_refused(self):
        self.table(dir="../../../../outside")
        with self.assertRaisesRegex(ValueError, "inside"):
            V.validate(self.root)

    def test_untracked_model_refused(self):
        p = self.root / "python/mojolearn/verify_reference/models/stale.npz"
        p.parent.mkdir()
        p.write_bytes(b"old fixture")
        with self.assertRaisesRegex(ValueError, "source commit"):
            V.validate(self.root)

    def test_generator_refuses_external_dirty_and_untracked_records(self):
        path = Path(__file__).resolve().parents[1] / "python/mojolearn/_verify_reference.py"
        spec = importlib.util.spec_from_file_location("vref", path)
        ref = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ref)
        ref.require_committed_records([self.column], self.root)
        extra = self.column.with_name("untracked.json")
        extra.write_bytes(self.column.read_bytes())
        with self.assertRaisesRegex(ValueError, "uncommitted"):
            ref.require_committed_records([extra], self.root)
        external = Path(self.temp.name) / "external.json"
        external.write_bytes(self.column.read_bytes())
        with self.assertRaisesRegex(ValueError, "committed under"):
            ref.require_committed_records([external], self.root)
        self.column.write_text("{}")
        with self.assertRaisesRegex(ValueError, "changed"):
            ref.require_committed_records([self.column], self.root)


if __name__ == "__main__":
    unittest.main()
