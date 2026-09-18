"""An omitted new module or stale implementation must block a release wheel."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import zipfile

import wheel_api_audit as audit


class WheelPayloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name, content in {
            'python/mojolearn/__init__.py': '__all__ = []\n',
            'python/mojolearn/new_algorithm.py': 'def predict(): return 1\n',
            'python/mojolearn/new_package/__init__.py': '__all__ = []\n',
            'python/mojolearn/new_package/inference.py': 'def decode(): return 2\n',
            'python/mojolearn/tests/test_not_shipped.py': 'raise AssertionError\n',
            'python/mojolearn/verify_reference/table.json': '{"current": true}',
            'python/mojolearn/verify_reference/models/models.json': '{"models": []}',
            'python/mojolearn/verify_reference/models/kernel.base.npz': 'saved-model-bytes',
            'tools/identity_break.py': 'CURRENT_HARNESS = True\n',
            'tools/identity_trace_diff.py': 'CURRENT_COMPARATOR = True\n',
        }.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)

    def wheel(self, *, omit=(), replace=None):
        path = self.root / 'candidate.whl'
        with zipfile.ZipFile(path, 'w') as wheel:
            for name, source in {**audit.source_payload(self.root), **audit.reference_payload(self.root)}.items():
                if name not in omit:
                    wheel.writestr(name, (replace or {}).get(name, source.read_bytes()))
        return path

    def test_complete_payload_excludes_tests(self):
        wheel = self.wheel()
        self.assertEqual(audit.payload_gaps(wheel, self.root),
                         dict(missing_python_payload=[], changed_python_payload=[],
                              missing_reference_payload=[], changed_reference_payload=[]))
        with zipfile.ZipFile(wheel) as z:
            self.assertFalse(any('/tests/' in name for name in z.namelist()))

    def test_unlisted_new_subpackage_is_required(self):
        missing = 'mojolearn/new_package/inference.py'
        result = audit.payload_gaps(self.wheel(omit=[missing]), self.root)
        self.assertEqual(result['missing_python_payload'], [missing])

    def test_stale_implementation_and_verifier_detected(self):
        changed = ['mojolearn/new_algorithm.py', 'mojolearn/_identity_break.py']
        result = audit.payload_gaps(self.wheel(replace={n: b'OLD = True' for n in changed}), self.root)
        self.assertEqual(result['changed_python_payload'], sorted(changed))

    def test_generated_verifier_required_without_source_copy(self):
        missing = 'mojolearn/_identity_trace_diff.py'
        self.assertEqual(audit.payload_gaps(self.wheel(omit=[missing]), self.root)
                         ['missing_python_payload'], [missing])

    def test_missing_model_and_stale_reference_block_candidate(self):
        missing = 'mojolearn/verify_reference/models/kernel.base.npz'
        changed = 'mojolearn/verify_reference/table.json'
        result = audit.payload_gaps(self.wheel(omit=[missing], replace={changed: b'{}'}), self.root)
        self.assertEqual(result['missing_reference_payload'], [missing])
        self.assertEqual(result['changed_reference_payload'], [changed])

    def test_strict_cli_rejects_bad_candidate(self):
        for complete, expected in [(False, 1), (True, 0)]:
            with self.subTest(complete=complete), patch('sys.argv', ['audit', 'candidate.whl', '--require-complete']), \
                    patch.object(audit, 'audit', return_value={'wheels': [{'source_payload_complete': complete}]}), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(audit.main(), expected)


if __name__ == '__main__':
    unittest.main()
