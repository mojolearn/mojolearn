"""Publication storage checks only; no product execution or numerical checks."""
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from six_lane_matrix_io import gzip_bytes, matrix_bytes, read_matrix
from six_lane_review_history import NAME, publish_review_history


class ReviewHistoryStorage(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.out = Path(directory.name)
        self.plain = self.out / NAME
        self.compressed = self.out / (NAME + '.gz')

    def test_migration_preserves_original_bytes_and_legacy_read_path(self):
        raw = b'{ "rows": [{"receipt_sha256":"old", "reason":"failed"}], "policy":"legacy" }\n'
        self.plain.write_bytes(raw)
        index = publish_review_history(self.out, [])
        self.assertFalse(self.plain.exists())
        snapshot = index['legacy_snapshots'][0]
        self.assertEqual(matrix_bytes(self.out / snapshot['path']), raw)
        self.assertEqual(snapshot['decoded_sha256'], hashlib.sha256(raw).hexdigest())
        self.assertEqual(read_matrix(self.plain)['rows'], json.loads(raw)['rows'])
        self.assertEqual(index['row_count'], 1)

    def test_append_versions_keeps_failures_and_deduplicates_exact_rows(self):
        failed = dict(receipt_sha256='same', reason='failed')
        passed = dict(receipt_sha256='same', reason='passed revised review')
        publish_review_history(self.out, [failed])
        index = publish_review_history(self.out, [passed, failed])
        self.assertEqual(read_matrix(self.compressed)['rows'], [failed, passed])
        self.assertEqual(index['row_count'], 2)

    def test_repeated_publication_is_byte_deterministic(self):
        self.plain.write_text(json.dumps(dict(rows=[dict(reason='original')])))
        first = publish_review_history(self.out, [])
        stored = self.compressed.read_bytes()
        second = publish_review_history(self.out, [])
        self.assertEqual(first, second)
        self.assertEqual(stored, self.compressed.read_bytes())
        self.assertEqual(first['decoded_sha256'], hashlib.sha256(gzip.decompress(stored)).hexdigest())
        self.assertEqual(first['stored_sha256'], hashlib.sha256(stored).hexdigest())

    def test_interrupted_migration_merges_both_forms(self):
        self.plain.write_text(json.dumps(dict(rows=[dict(reason='legacy')])) )
        self.compressed.write_bytes(gzip_bytes(json.dumps(dict(rows=[dict(reason='new')])).encode()))
        index = publish_review_history(self.out, [])
        self.assertEqual(index['row_count'], 2)
        self.assertEqual({r['reason'] for r in read_matrix(self.compressed)['rows']}, {'legacy', 'new'})

    def test_corrupt_history_fails_without_discarding_legacy(self):
        self.plain.write_text('{"rows":[]}')
        self.compressed.write_bytes(b'not gzip')
        with self.assertRaises(gzip.BadGzipFile):
            publish_review_history(self.out, [])
        self.assertTrue(self.plain.exists())
        self.assertEqual(self.compressed.read_bytes(), b'not gzip')


if __name__ == '__main__':
    unittest.main()
