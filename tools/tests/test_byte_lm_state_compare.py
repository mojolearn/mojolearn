"""Authored stdlib artifact-boundary tests; ROOT ONLY, never a GPU/model test.

Not executed by the authoring subagent. Root may run this file under the
remote resource guard. Fixtures are at most two MiB; no model data is created.
"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    'byte_lm_state_compare', Path(__file__).resolve().parents[1] / 'byte_lm_state_compare.py')
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)


class ArtifactBoundaries(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.terminal = dict(guard='nvidia-root-serial-v1', reason=None,
                             returncode=0, cpu_affinity='2,3', thread_limit=2)
        (self.root / 'command.sh').write_bytes(b'exec retained-root-job\n')
        (self.root / 'result.json').write_bytes(b'{"complete":true}\n')
        self.write_log()
        self.receipt = dict(schema='mojolearn.root-job-receipt.v1', vendor='cuda',
                            exit_code=0, job_kind='capture', guard_terminal=self.terminal,
                            boundary='root-observed exit', command=self.descriptor('command.sh'),
                            guard_log=self.descriptor('guard.log'), result=self.descriptor('result.json'))
        self.flush()

    def descriptor(self, name):
        raw = (self.root / name).read_bytes()
        return dict(file=name, bytes=len(raw), sha256=compare.sha(raw))

    def write_log(self):
        (self.root / 'guard.log').write_bytes(b'job output\n' + compare.canonical(self.terminal))

    def flush(self):
        (self.root / 'receipt.json').write_bytes(compare.canonical(self.receipt))

    def admit(self):
        return compare.receipt(self.root / 'receipt.json',
                               compare.sha((self.root / 'result.json').read_bytes()), 'cuda')

    def test_complete_root_receipt(self):
        self.assertEqual(self.admit()['final'], self.terminal)

    def test_summary_bytes_must_be_bound(self):
        (self.root / 'result.json').write_bytes(b'{"complete":false}\n')
        with self.assertRaises(ValueError):
            self.admit()

    def test_exit_after_success_summary_is_required(self):
        self.terminal['returncode'] = -15
        self.write_log()
        self.receipt['guard_log'] = self.descriptor('guard.log')
        self.flush()
        with self.assertRaises(ValueError):
            self.admit()

    def test_guard_final_record_cannot_be_followed_by_other_output(self):
        with (self.root / 'guard.log').open('ab') as stream:
            stream.write(b'{"later":"failure"}\n')
        self.receipt['guard_log'] = self.descriptor('guard.log')
        self.flush()
        with self.assertRaises((ValueError, KeyError)):
            self.admit()

    def test_wrong_vendor_and_job_kind(self):
        for key, value in (('vendor', 'hip'), ('job_kind', 'oracle')):
            before = self.receipt[key]
            self.receipt[key] = value
            self.flush()
            with self.assertRaises(ValueError):
                self.admit()
            self.receipt[key] = before

    def test_parent_traversal_refused(self):
        self.receipt['result']['file'] = '../result.json'
        self.flush()
        with self.assertRaises(ValueError):
            self.admit()

    def test_nested_relative_artifact_allowed(self):
        (self.root / 'nested').mkdir()
        (self.root / 'nested/result.json').write_bytes((self.root / 'result.json').read_bytes())
        self.receipt['result'] = self.descriptor('nested/result.json')
        self.flush()
        self.assertEqual(self.admit()['result']['file'], 'nested/result.json')

    def test_symlink_and_oversized_files_refused(self):
        (self.root / 'alias.json').symlink_to(self.root / 'result.json')
        with self.assertRaises(ValueError):
            compare.read(self.root / 'alias.json')
        with self.assertRaises(ValueError):
            compare.read(self.root / 'result.json', limit=2)

    def test_duplicate_json_and_nonfinite_numbers_refused(self):
        for raw in (b'{"key":1,"key":2}', b'{"value":NaN}', b'{"value":Infinity}', b'{"value":1e999}'):
            with self.assertRaises(ValueError):
                compare.parse(raw)

    def test_same_vendor_cannot_swap_binary_between_legs(self):
        left = dict(source={}, runtime=dict(native_vendor='cuda', binding_sha256='a' * 64))
        right = dict(source={}, runtime=dict(native_vendor='cuda', binding_sha256='b' * 64))
        with self.assertRaisesRegex(ValueError, 'same-vendor binding hashes differ'):
            compare.compatible(left, right)

    def test_checkpoint_integrity_and_canonical_bytes(self):
        # The checkpoint parser only checks envelope/array integrity. Full
        # registry/config admission is separately required by load_capture.
        payload = {'completed_steps': 64}
        for key in compare.STATE_NAMES:
            count = 20 if key == 'flags' else compare.N
            payload[key] = dict(dtype='<i4' if key == 'flags' else '<f4',
                                shape=[count], hex='00' * (count * 4))
        envelope = dict(schema='mojolearn.small-byte-lm-json-checkpoint.v1', payload=payload,
                        payload_sha256=compare.sha(compare.canonical(payload, False)))
        path = self.root / 'checkpoint.json'
        path.write_bytes(compare.canonical(envelope))
        raw, metadata, arrays = compare.checkpoint(self.root, path.name, self.descriptor(path.name))
        self.assertEqual(metadata, {'completed_steps': 64})
        self.assertEqual(len(arrays['parameters']), compare.N * 4)
        path.write_bytes(json.dumps(envelope, indent=2).encode())
        with self.assertRaises(ValueError):
            compare.checkpoint(self.root, path.name, self.descriptor(path.name))


if __name__ == '__main__':
    unittest.main()
