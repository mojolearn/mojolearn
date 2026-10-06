# SPDX-License-Identifier: Apache-2.0
"""Metadata-only regression checks; these never build or execute GPU fixtures."""
import importlib.util
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("i18_pair", Path(__file__).with_name("paired_identity.py"))
PAIR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PAIR)


class FrozenPairTests(unittest.TestCase):
    def init_repo(self, path):
        path.mkdir()
        subprocess.run(["git", "init", "-q", str(path)], check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Receipt test"], cwd=path, check=True)
        subprocess.run(["git", "config", "user.email", "receipt-test@example.invalid"], cwd=path, check=True)
        manifest = path / "experiments/performance_ideas/I18/manifest.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text("{}\n")
        self.commit(path)
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()

    def commit(self, path):
        subprocess.run(["git", "add", "."], cwd=path, check=True)
        subprocess.run(["git", "commit", "-qm", "receipt metadata fixture"], cwd=path, check=True)

    def test_source_change_and_untracked_files_are_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "src"
            source = self.init_repo(root)
            PAIR.require_frozen(source, root)
            (root / "untracked.txt").write_text("metadata\n")
            with self.assertRaises(RuntimeError):
                PAIR.require_frozen(source, root)
            self.commit(root)
            with self.assertRaises(RuntimeError):
                PAIR.require_frozen(source, root)

    def test_receipts_require_frozen_source_and_completed_matching_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary)
            receipt = dict(schema=1, id="I18", mode="identical", vendor="apple", stage="build",
                           arm="baseline", source_sha="expected-source", manifest_sha256="expected-manifest",
                           status="COMPLETED", returncode=0)
            path = evidence / "I18-apple-build-baseline-receipt.json"
            path.write_text(json.dumps(receipt))
            verified = PAIR.checked_stage_receipt(evidence, "apple", "build", "baseline", "expected-source", "expected-manifest")
            self.assertEqual(verified["source_sha"], "expected-source")
            for key, value in {"source_sha": "other-source", "manifest_sha256": "other-manifest",
                               "stage": "validate", "arm": "candidate", "status": "SOURCE_CHANGED",
                               "returncode": 125, "vendor": "amd"}.items():
                with self.subTest(key=key):
                    mismatched = receipt | {key: value}
                    path.write_text(json.dumps(mismatched))
                    with self.assertRaises(RuntimeError):
                        PAIR.checked_stage_receipt(evidence, "apple", "build", "baseline", "expected-source", "expected-manifest")

    def test_all_four_completed_stage_receipts_are_retained(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "src"
            source = self.init_repo(root)
            output = Path(temporary) / "evidence"
            original_run = subprocess.run
            original_require = PAIR.require_frozen
            manifest = root / "experiments/performance_ideas/I18/manifest.json"
            manifest_digest = hashlib.sha256(manifest.read_bytes()).hexdigest()

            def fake_runner(command, **kwargs):
                if command[0] == "git":
                    return original_run(command, **kwargs)
                stage = command[command.index("--stage") + 1]
                arm = command[command.index("--arm") + 1]
                evidence = Path(command[command.index("--evidence") + 1])
                evidence.mkdir()
                receipt = dict(schema=1, id="I18", mode="identical", vendor="apple",
                               stage=stage, arm=arm, source_sha=source, manifest_sha256=manifest_digest,
                               status="COMPLETED", returncode=0)
                (evidence / f"I18-apple-{stage}-{arm}-receipt.json").write_text(json.dumps(receipt))
                if stage == "validate":
                    lines = [f"{tag} {stream} 17" for tag in PAIR.TAGS for stream in (1, 2)]
                    lines.append("I18 PASS mechanism permutations cache_bound reset full_forest_digests; compare OFF arm")
                    (evidence / f"I18-apple-{stage}-{arm}.log").write_text("\n".join(lines))
                return subprocess.CompletedProcess(command, 0)

            with patch.object(PAIR, "ROOT", root), patch.object(PAIR, "require_frozen", side_effect=lambda expected: original_require(expected, root)), patch.object(PAIR.subprocess, "run", side_effect=fake_runner), patch("sys.argv", ["paired_identity.py", "--vendor", "apple", "--source-sha", source, "--output", str(output)]), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(PAIR.main(), 0)
            summary = json.loads((output / "device-witness.json").read_text())
            self.assertEqual(set(summary["stage_receipts"]), {"baseline/build", "baseline/validate", "candidate/build", "candidate/validate"})
            self.assertTrue(all(packet["source_sha"] == source for packet in summary["stage_receipts"].values()))

    def test_midstage_commit_cannot_create_a_passing_pair_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "src"
            source = self.init_repo(root)
            output = Path(temporary) / "evidence"
            original_run = subprocess.run
            original_require = PAIR.require_frozen
            calls = []

            def fake_runner(command, **kwargs):
                if command[0] == "git":
                    return original_run(command, **kwargs)
                calls.append(command)
                # Simulate a clean source change while the first build runs.
                (root / "new-source.txt").write_text("metadata\n")
                original_run(["git", "add", "."], cwd=root, check=True)
                original_run(["git", "commit", "-qm", "change during stage"], cwd=root, check=True)
                return subprocess.CompletedProcess(command, 0)

            with patch.object(PAIR, "ROOT", root), patch.object(PAIR, "require_frozen", side_effect=lambda expected: original_require(expected, root)), patch.object(PAIR.subprocess, "run", side_effect=fake_runner), patch("sys.argv", ["paired_identity.py", "--vendor", "apple", "--source-sha", source, "--output", str(output)]):
                with self.assertRaises(RuntimeError):
                    PAIR.main()
            self.assertEqual(len(calls), 1)
            self.assertFalse((output / "device-witness.json").exists())


if __name__ == "__main__":
    unittest.main()
