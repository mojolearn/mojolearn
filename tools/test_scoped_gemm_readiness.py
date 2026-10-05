#!/usr/bin/env python3
"""Metadata-only regression checks: never import NumPy/native or run GPU work."""
import subprocess
import unittest
from unittest.mock import patch

from apple_fast_job_policy import policy_for
from scoped_gemm_preflight_spec import make_spec
from scoped_gemm_quality import verify_source

H = 'a' * 40
C = 'b' * 40


class ReadinessTest(unittest.TestCase):
    def test_spec_complete(self):
        spec = make_spec(H, C, 'scoped-r2-q-v1', 'all')
        self.assertEqual(spec['args'], [C, 'scoped-r2-q-v1', 'all'])
        self.assertEqual(len(spec['cases']), 28)
        self.assertTrue(all(c['requires'] == ['probe'] for c in spec['cases']))
        artifact = spec['artifacts'][0]
        self.assertEqual(artifact['compiled_source'], C)
        self.assertEqual(artifact['defines_A'], '-D MOJOLEARN_SCOPED_GEMM_AUDIT')
        self.assertEqual(artifact['defines_B'].count('-D '), 7)
        self.assertEqual(spec['prerequisites'], [])  # direct probe, no package install

    def test_policy_rejects_invalid_contract(self):
        for args in ([C, 'tag'], ['main', 'tag', 'all'], [C, '../tag', 'all'],
                     [C, 'tag', 'unknown'], [C, 'tag', 'all', 'timing']):
            with self.subTest(args=args), self.assertRaises(ValueError):
                policy_for(H, 'tools/scoped_gemm_quality.py', args)

    def test_source_accepts_tools_only(self):
        with patch('scoped_gemm_quality.subprocess.check_output',
                   side_effect=[H + '\n', 'tools/scoped_gemm_quality.py\n']), \
             patch('scoped_gemm_quality.subprocess.run') as run:
            self.assertEqual(verify_source(C), H)
            self.assertEqual(run.call_count, 2)  # ancestor and clean tracked source

    def test_source_rejects_native_or_config_drift(self):
        for file in ('bindings/_mojolearn_scoped_gemm_probe.mojo', 'pixi.lock',
                     'python/mojolearn/_buffer.py', 'tools/unreviewed.py'):
            with self.subTest(file=file), \
                 patch('scoped_gemm_quality.subprocess.check_output', side_effect=[H, file]), \
                 patch('scoped_gemm_quality.subprocess.run'), self.assertRaises(AssertionError):
                verify_source(C)

    def test_source_rejects_nonancestor(self):
        with patch('scoped_gemm_quality.subprocess.check_output', return_value=H), \
             patch('scoped_gemm_quality.subprocess.run',
                   side_effect=subprocess.CalledProcessError(1, ['git'])), \
             self.assertRaises(subprocess.CalledProcessError):
            verify_source(C)


if __name__ == '__main__':
    unittest.main()
