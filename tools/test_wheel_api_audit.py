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
            'tools/identity_lanes/linear.py': 'CURRENT_LINEAR_FIXTURE = True\n',
            'python/mojolearn/verify_reference/ctr_models/ctr.npz': 'ctr-model-bytes',
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

    def test_generated_fragment_staleness_and_missing_ctr_model(self):
        fragment = 'mojolearn/_identity_lane_linear.py'
        model = 'mojolearn/verify_reference/ctr_models/ctr.npz'
        result = audit.payload_gaps(self.wheel(omit=[model], replace={fragment: b'OLD'}), self.root)
        self.assertEqual(result['changed_python_payload'], [fragment])
        self.assertEqual(result['missing_reference_payload'], [model])
        result = audit.payload_gaps(self.wheel(omit=[fragment]), self.root)
        self.assertEqual(result['missing_python_payload'], [fragment])

    def test_explicit_source_root_qualifies_frozen_payload_not_newer_tooling(self):
        import shutil
        tooling = self.root / 'newer-tooling'
        shutil.copytree(self.root / 'python', tooling / 'python')
        shutil.copytree(self.root / 'tools', tooling / 'tools')
        (tooling / 'tools/identity_break.py').write_text('NEWER_HARNESS = True\n')
        (tooling / 'python/mojolearn/verify_reference/table.json').write_text('{"newer": true}')
        wheel = self.wheel()
        with patch.object(audit.matrix, 'ROOT', str(tooling)):
            old_default = audit.audit([wheel])['wheels'][0]
            self.assertFalse(old_default['source_payload_complete'])
            self.assertEqual(old_default['changed_python_payload'], ['mojolearn/_identity_break.py'])
            self.assertEqual(old_default['changed_reference_payload'], ['mojolearn/verify_reference/table.json'])
            frozen = audit.audit([wheel], source_root=self.root)['wheels'][0]
            self.assertTrue(frozen['source_payload_complete'])
            self.assertEqual(audit.matrix.ROOT, str(tooling))
            # Explicit staging must still reject a stale candidate table.
            stale = self.wheel(replace={'mojolearn/verify_reference/table.json': b'{}'})
            self.assertFalse(audit.audit([stale], source_root=self.root)['wheels'][0]['source_payload_complete'])

    def test_explicit_source_root_restores_default_after_error(self):
        before = audit.matrix.ROOT
        with self.assertRaises(FileNotFoundError):
            audit.audit([self.root / 'missing.whl'], source_root=self.root)
        self.assertEqual(audit.matrix.ROOT, before)

    def test_strict_cli_rejects_bad_candidate(self):
        for complete, expected in [(False, 1), (True, 0)]:
            with self.subTest(complete=complete), patch('sys.argv', ['audit', 'candidate.whl', '--require-complete']), \
                    patch.object(audit, 'audit', return_value={'wheels': [{'source_payload_complete': complete}]}), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(audit.main(), expected)


if __name__ == '__main__':
    unittest.main()
