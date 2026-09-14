"""Release-date regression tests; no GPU or network required."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import citation_metadata as citation


class CitationMetadataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'python/mojolearn').mkdir(parents=True)
        (self.root / 'python/mojolearn/_version.py').write_text('__version__ = "0.8.5"\n')
        self.changelog = self.root / 'CHANGELOG.md'
        self.changelog.write_text('## 0.8.5 (published 2026-09-14)\n')
        self.path = self.root / 'CITATION.cff'
        self.path.write_text('authors:\n  - family-names: Hendel\n'
                             'version: "0.8.4"\ndate-released: "2026-09-11"\n'
                             'doi: 10.5281/zenodo.22068632\n'
                             'preferred-citation:\n  date-released: "2025-01-01"\n')

    def test_stale_metadata_rejected_then_generated_idempotently(self):
        self.assertEqual(citation.sync(self.root), 1)
        self.assertEqual(citation.sync(self.root, write=True), 0)
        text = self.path.read_text()
        self.assertIn('version: "0.8.5"\ndate-released: "2026-09-14"', text)
        self.assertIn('doi: 10.5281/zenodo.22068632', text)
        self.assertIn('  date-released: "2025-01-01"', text)
        self.assertEqual(citation.sync(self.root), 0)
        self.assertEqual(citation.sync(self.root, write=True), 0)
        self.assertEqual(text, self.path.read_text())

    def test_unreleased_does_not_claim_publication(self):
        self.changelog.write_text('## 0.8.5 (unreleased 2026-09-14)\n')
        self.assertEqual(citation.sync(self.root, write=True), 0)
        self.assertNotIn('\ndate-released:', self.path.read_text())
        self.changelog.write_text('## 0.8.5 (published 2026-09-15)\n')
        self.assertEqual(citation.sync(self.root, write=True), 0)
        self.assertIn('\ndate-released: "2026-09-15"', self.path.read_text())

    def test_invalid_missing_or_ambiguous_release_fails_without_writing(self):
        original = self.path.read_text()
        for heading in ('## 0.8.4 (published 2026-09-14)\n',
                        '## 0.8.5 (published 2026-02-30)\n',
                        '## 0.8.5 (published 2026-09-14)\n' * 2):
            with self.subTest(heading=heading):
                self.changelog.write_text(heading)
                self.assertEqual(citation.sync(self.root, write=True), 1)
                self.assertEqual(self.path.read_text(), original)


if __name__ == '__main__':
    unittest.main()
