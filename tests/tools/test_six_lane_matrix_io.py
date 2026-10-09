"""Storage/metadata tests only: no bindings, models, builds or input arrays."""
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from six_lane_matrix_io import (MATRIX_PATH, git_matrix_bytes, gzip_bytes,
                                matrix_bytes, matrix_digest, read_git_matrix,
                                read_matrix, write_matrix)


class MatrixStorage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.raw = self.root / 'matrix.json'
        self.gz = self.root / 'matrix.json.gz'
        self.data = b'{"cells": [], "label": "exact whitespace"}\n'

    def test_raw_gzip_and_legacy_path_have_identical_logical_digest(self):
        self.raw.write_bytes(self.data)
        self.gz.write_bytes(gzip_bytes(self.data))
        expected = hashlib.sha256(self.data).hexdigest()
        for path in (self.raw, self.gz):
            self.assertEqual(matrix_bytes(path), self.data)
            self.assertEqual(matrix_digest(path), expected)
        self.raw.unlink()
        self.assertEqual(matrix_bytes(self.raw), self.data)
        self.assertEqual(matrix_digest(self.raw), expected)
        self.assertEqual(read_matrix(self.raw), json.loads(self.data))

    def test_existing_raw_takes_precedence_and_corruption_is_not_hidden(self):
        self.gz.write_bytes(gzip_bytes(self.data))
        self.raw.write_text('not JSON')
        with self.assertRaises(json.JSONDecodeError):
            read_matrix(self.raw)

    def test_missing_and_corrupt_gzip_fail(self):
        with self.assertRaises(FileNotFoundError):
            read_matrix(self.raw)
        self.gz.write_bytes(b'broken gzip')
        with self.assertRaises(gzip.BadGzipFile):
            read_matrix(self.raw)

    def test_writer_is_deterministic_and_external_raw_stays_supported(self):
        value = {'cells': [], 'configurations': [{'id': 'fixture'}]}
        write_matrix(self.raw, value)
        write_matrix(self.gz, value)
        first = self.gz.read_bytes()
        write_matrix(self.gz, value)
        self.assertEqual(first, self.gz.read_bytes())
        self.assertEqual(matrix_bytes(self.gz), self.raw.read_bytes())
        self.assertEqual(read_matrix(self.gz), value)

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.root), *args],
                                       stderr=subprocess.PIPE).decode().strip()

    def test_frozen_raw_and_compressed_commits_resolve_without_rewriting(self):
        self.git('init', '-q')
        self.git('config', 'user.email', 'metadata-test@example.invalid')
        self.git('config', 'user.name', 'Metadata fixture')
        raw = self.root / MATRIX_PATH
        raw.parent.mkdir(parents=True)
        raw.write_bytes(self.data)
        self.git('add', '.')
        self.git('commit', '-qm', 'raw matrix')
        old = self.git('rev-parse', 'HEAD')
        raw.with_suffix('.json.gz').write_bytes(gzip_bytes(self.data))
        raw.unlink()
        self.git('add', '-A')
        self.git('commit', '-qm', 'compressed matrix')
        new = self.git('rev-parse', 'HEAD')
        for source in (old, new):
            self.assertEqual(git_matrix_bytes(self.root, source), self.data)
            self.assertEqual(read_git_matrix(self.root, source), json.loads(self.data))
        raw.write_bytes(b'broken JSON')
        self.git('add', '.')
        self.git('commit', '-qm', 'invalid raw does not fall back')
        with self.assertRaises(json.JSONDecodeError):
            read_git_matrix(self.root, 'HEAD')
        with self.assertRaises(subprocess.CalledProcessError):
            git_matrix_bytes(self.root, 'missing-freeze')


if __name__ == '__main__':
    unittest.main()
