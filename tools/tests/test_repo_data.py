#!/usr/bin/env python3
"""Tests for tools/repo_data.py, the one resolver for bulk data that left the repository.

Pure Python on scratch directories: a fake bucket is packed into a scratch evidence root, resolved,
tampered with, installed from its tar and refused when it cannot be fetched. No network, no R2, no
credentials (the dataset_store.sh call is replaced). The cases are independent and touch only their own
temporary directories, so unittest runs them in one process in a second or two; there is nothing to split.

  python3 tools/tests/test_repo_data.py
"""
import hashlib
import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import repo_data  # noqa: E402

BUCKET = 'experiments/fake_bucket/measurements'
KEY = 'repo-data/experiments/fake_bucket/measurements.tar'


class RepoDataTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.src = self.tmp / 'src'
        for rel, body in (('20261006/receipts/INDEX.tsv', b'path\tsha256\tvendor\n'),
                          ('20261006/board.json', b'{"a": 1}\n'),
                          ('20261006/receipts/nvidia/r/receipt.json', b'{"workload": {}}\n')):
            (self.src / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.src / rel).write_bytes(body)
        store_manifest = self.tmp / 'manifest.tsv'
        store_manifest.write_text('ann-bench/sift1m/sift1m.npz\t1\t' + '0' * 64 + '\n')
        self.patches = [
            mock.patch.dict(os.environ, MOJOLEARN_EVIDENCE_ROOT=str(self.tmp / 'ev'), MOJOLEARN_REPO_DATA_WORKERS='2'),
            mock.patch.object(repo_data, 'BUCKETS', {BUCKET: KEY}),
            mock.patch.object(repo_data, 'MANIFEST_DIR', self.tmp / 'pins'),
            mock.patch.object(repo_data, 'STORE_MANIFEST', store_manifest),
            mock.patch.object(repo_data, 'STAGING', self.tmp / 'cache'),
            mock.patch.object(repo_data, 'ROOT', self.tmp),
        ]
        for p in self.patches:
            p.start()
        repo_data.bucket_root.cache_clear()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        repo_data.bucket_root.cache_clear()

    def test_pack_copies_verifies_and_pins(self):
        tar = repo_data.pack(BUCKET, self.src)
        base = self.tmp / 'ev/repo-data' / BUCKET
        self.assertEqual((base / '20261006/board.json').read_bytes(), b'{"a": 1}\n')
        rows, _ = repo_data.read_manifest(BUCKET)
        self.assertEqual(len(rows), 3)
        size, sha = repo_data.pinned(KEY)
        self.assertEqual(size, tar.stat().st_size)
        self.assertEqual(sha, hashlib.sha256(tar.read_bytes()).hexdigest())
        # the other pins are kept
        self.assertIsNotNone(repo_data.pinned('ann-bench/sift1m/sift1m.npz'))
        # deterministic: packing the same bytes again gives the same tar
        first = tar.read_bytes()
        repo_data.pack(BUCKET, self.src)
        self.assertEqual(tar.read_bytes(), first)

    def test_resolve_once_per_run(self):
        repo_data.pack(BUCKET, self.src)
        got = repo_data.resolve(BUCKET + '/20261006/receipts')
        self.assertEqual(got, self.tmp / 'ev/repo-data' / BUCKET / '20261006/receipts')
        with mock.patch.object(repo_data, 'check_tree', side_effect=AssertionError('checked twice')):
            repo_data.resolve(BUCKET + '/20261006/board.json')

    def test_changed_copy_is_refused_and_left_alone(self):
        repo_data.pack(BUCKET, self.src)
        board = self.tmp / 'ev/repo-data' / BUCKET / '20261006/board.json'
        board.write_bytes(b'{"a": 2}\n')
        (board.parent / '..' / repo_data.STAMP).resolve().unlink()
        with self.assertRaises(SystemExit) as raised:
            repo_data.resolve(BUCKET)
        self.assertIn('does not match', str(raised.exception))
        self.assertIn('tools/repo_data.py fetch ' + BUCKET, str(raised.exception))
        self.assertEqual(board.read_bytes(), b'{"a": 2}\n')

    def test_size_change_is_refused_even_after_a_verified_run(self):
        repo_data.pack(BUCKET, self.src)
        (self.tmp / 'ev/repo-data' / BUCKET / '20261006/board.json').write_bytes(b'{}')
        with self.assertRaises(SystemExit):
            repo_data.resolve(BUCKET)

    def test_absent_copy_is_fetched_and_installed(self):
        tar = repo_data.pack(BUCKET, self.src)
        blob = tar.read_bytes()
        base = self.tmp / 'ev/repo-data' / BUCKET
        moved = self.tmp / 'aside'
        base.rename(moved)

        def pull(*args, check=True):
            self.assertEqual(args[:2], ('pull', KEY))
            Path(args[2]).write_bytes(blob)
            return mock.Mock(returncode=0)
        with mock.patch.object(repo_data, 'store', side_effect=pull):
            got = repo_data.resolve(BUCKET + '/20261006/board.json')
        self.assertEqual(got.read_bytes(), b'{"a": 1}\n')
        self.assertTrue((base / repo_data.STAMP).is_file())
        self.assertFalse((self.tmp / 'cache' / repo_data.tar_name(BUCKET)).exists())

    def test_unfetchable_copy_is_refused_in_words(self):
        repo_data.pack(BUCKET, self.src)
        (self.tmp / 'ev/repo-data' / BUCKET).rename(self.tmp / 'aside')
        with mock.patch.object(repo_data, 'store', return_value=mock.Mock(returncode=1)):
            with self.assertRaises(SystemExit) as raised:
                repo_data.resolve(BUCKET)
        text = str(raised.exception)
        self.assertIn('could not be fetched from R2 key ' + KEY, text)
        self.assertIn('dataset_store.sh stage', text)

    def test_tar_with_a_wrong_member_is_refused(self):
        tar = repo_data.pack(BUCKET, self.src)
        (self.tmp / 'ev/repo-data' / BUCKET).rename(self.tmp / 'aside')
        (self.src / '20261006/board.json').write_bytes(b'{"a": 9}\n')
        rows, _ = repo_data.read_manifest(BUCKET)
        repo_data.write_tar(self.src, rows, tar)
        with self.assertRaises(SystemExit) as raised:
            repo_data.install(BUCKET, tar)
        self.assertIn('does not match the manifest', str(raised.exception))
        self.assertFalse((self.tmp / 'ev/repo-data' / BUCKET).exists())

    def test_path_outside_buckets_is_refused(self):
        with self.assertRaises(SystemExit):
            repo_data.resolve('bench/results/bench_board/x.json')


class GridRunReceiptsTest(unittest.TestCase):
    def test_receipts_come_from_the_resolver(self):
        tmp = Path(tempfile.mkdtemp())
        receipts = tmp / 'receipts'
        (receipts / 'nvidia/r').mkdir(parents=True)
        body = b'{"workload": {"workload_id": "w", "artifact_provenance": {"A": [{"path": "x/_a.so"}]}}}'
        (receipts / 'nvidia/r/receipt.json').write_bytes(body)
        (receipts / 'INDEX.tsv').write_text('path\tsha256\tvendor\nnvidia/r/receipt.json\t'
                                            + hashlib.sha256(body).hexdigest() + '\tnvidia\n')
        grid_run = importlib.import_module('six_lane_grid_run')
        with mock.patch.object(repo_data, 'resolve', return_value=receipts) as resolve:
            rows = grid_run.oct6_receipt_rows()
        resolve.assert_called_once_with(grid_run.OCT6_RECEIPTS_REL)
        self.assertEqual(rows, [(receipts / 'nvidia/r/receipt.json', 'nvidia')])


if __name__ == '__main__':
    unittest.main()
