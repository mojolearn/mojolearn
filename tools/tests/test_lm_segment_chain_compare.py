"""The chain compare of tools/lm_segment.py: an expected line written before
`hash_scheme` existed is the first scheme, so a replay of identical bits under
sha256.v1 agrees and the digests decide. CPU only, no model data."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.argv = [sys.argv[0]]
SPEC = importlib.util.spec_from_file_location(
    'lm_segment', Path(__file__).resolve().parents[1] / 'lm_segment.py')
seg = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(seg)

OLD_LINE = dict(schema=seg.CHAIN_SCHEMA, step=19, route='A', segment='2', label='amd-A-2',
                lr_f32_hex='3a0380db', losses_f32_hex=['410b6076', '4113a730'],
                state_sha256='9f' * 32, gradient_sha256='54' * 32, batch_index=[1152, 1216],
                seconds=138.7, hash_seconds=3.4, prev='5f' * 32)


def _replay(scheme, **over):
    row = {k: OLD_LINE[k] for k in ('schema', 'step', 'route', 'segment', 'lr_f32_hex',
                                    'losses_f32_hex', 'state_sha256', 'gradient_sha256', 'batch_index')}
    row.update(label='nvidia-A-3', seconds=1.0, hash_seconds=0.1, hash_scheme=scheme)
    row.update(over)
    return row


class ChainCompare(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        chain = Path(self.tmp.name) / 'expected.jsonl'
        chain.write_text(json.dumps(OLD_LINE) + '\n')
        self.expect = seg._chain_index(chain)
        self.out = Path(self.tmp.name) / 'chain.jsonl'

    def tearDown(self):
        self.tmp.cleanup()

    def _writer(self):
        return seg.ChainWriter(self.out, 18, self.expect, lambda *_: None)

    def test_a_line_without_the_field_reads_as_the_first_scheme(self):
        self.assertEqual(self.expect[19]['hash_scheme'], seg.SCHEME_V1)

    def test_identical_bits_under_v1_agree(self):
        w = self._writer()
        self.assertTrue(w.write(_replay(seg.SCHEME_V1)))
        self.assertEqual(w.close(19)[0], 'PASS')

    def test_the_newer_scheme_is_a_disagreement_on_its_own(self):
        w = self._writer()
        self.assertFalse(w.write(_replay(seg.SCHEME_V2)))
        verdict, disagreements, _ = w.close(19)
        self.assertEqual(verdict, 'FAIL')
        self.assertEqual(disagreements[0]['fields'], ['hash_scheme'])

    def test_a_moved_state_bit_is_caught_under_v1(self):
        w = self._writer()
        self.assertFalse(w.write(_replay(seg.SCHEME_V1, state_sha256='9e' + '9f' * 31)))
        self.assertEqual(w.close(19)[1][0]['fields'], ['state_sha256'])

    def test_the_recipe_names_the_scheme(self):
        self.assertEqual(seg.hash_scheme_of({}), seg.SCHEME_V1)
        self.assertEqual(seg.hash_scheme_of({'hash_scheme': seg.SCHEME_V2}), seg.SCHEME_V2)
        with self.assertRaises(SystemExit):
            seg.hash_scheme_of({'hash_scheme': 'md5'})


if __name__ == '__main__':
    unittest.main()
