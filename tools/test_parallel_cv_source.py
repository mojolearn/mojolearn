import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from parallel_cross_val_check import source_identity
from mojolearn._verify_parallel_cv import PROFILE_FILES


class SourceIdentityTests(unittest.TestCase):
    def test_archive_requires_full_commit_and_hashes_actual_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in PROFILE_FILES:
                path = root / 'python/mojolearn' / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('actual source')
            with patch.dict(os.environ, {'MOJOLEARN_COMMIT': 'a' * 40}):
                before = source_identity(root)
                (root / 'python/mojolearn/_parallel_worker.py').write_text('different source')
                after = source_identity(root)
            self.assertEqual(before['kind'], 'guarded-source-archive')
            self.assertNotEqual(before['source_sha256'], after['source_sha256'])
            for witness in ('', 'a' * 8, 'a' * 40 + ' (dirty)'):
                with patch.dict(os.environ, {'MOJOLEARN_COMMIT': witness}):
                    with self.assertRaisesRegex(RuntimeError, 'full source commit'):
                        source_identity(root)
