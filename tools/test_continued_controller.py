#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""No-cloud controls for the optional native-check controller gate."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class ContinuedControllerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'remote/continued').mkdir(parents=True)
        (self.root / 'remote/continued/commit.txt').write_text('a' * 40 + '\n')
        (self.root / 'remote/leg.txt').write_text('continued_exit=0\n')
        stub = self.root / 'python3'
        stub.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$OUT/called"\nexit "${VALIDATOR_RC:-0}"\n')
        stub.chmod(0o700)
        source = Path(__file__).with_name('gemm_remote_leg.sh').read_text()
        self.function = source.split('leg_continued_artifacts() {', 1)[1].split('\n}\n', 1)[0]

    def run_gate(self, **extra):
        env = dict(os.environ, OUT=str(self.root), COMMIT='a' * 40,
                   PATH=str(self.root) + os.pathsep + os.environ['PATH'])
        env.update(extra)
        return subprocess.run(['/bin/sh', '-c', 'gate() {' + self.function + '\n}\ngate'],
                              env=env, capture_output=True).returncode

    def test_success_requires_record_validator(self):
        self.assertEqual(self.run_gate(), 0)
        self.assertIn('continued_cert_compare.py', (self.root / 'called').read_text())

    def test_failed_or_missing_completion_is_rejected(self):
        for text in ('continued_exit=143\n', 'mamba_cert_exit=0\n'):
            (self.root / 'remote/leg.txt').write_text(text)
            self.assertNotEqual(self.run_gate(), 0)
            self.assertFalse((self.root / 'called').exists())

    def test_wrong_source_is_rejected(self):
        (self.root / 'remote/continued/commit.txt').write_text('b' * 40 + '\n')
        self.assertNotEqual(self.run_gate(), 0)
        self.assertFalse((self.root / 'called').exists())

    def test_record_validation_failure_propagates(self):
        self.assertNotEqual(self.run_gate(VALIDATOR_RC='1'), 0)


if __name__ == '__main__':
    unittest.main()
