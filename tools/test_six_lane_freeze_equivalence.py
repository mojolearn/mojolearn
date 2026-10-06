"""Synthetic metadata rejection tests; no estimators, arrays, builds or devices."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import six_lane_freeze_equivalence as eq
from six_lane_compare_results import expected_scope
import test_six_lane_compare_results as fixtures


class EquivalenceMetadataTest(fixtures.ComparisonMetadataTest):
    def prepare(self):
        self.case['columns'] = {k: v for k, v in self.case['columns'].items() if k in eq.ROUTES}
        self.receipts['amd']['source_sha'] = '7' * 40
        for run in self.receipts['amd']['runs']:
            run['result']['source_sha'] = '7' * 40
        entries = {}
        for column, (vendor, define, _) in eq.ROUTES.items():
            receipt = self.receipts[column]
            refs = {}
            for arm in ('A', 'B'):
                artifact = receipt['workload']['artifact_provenance'][arm][0]
                artifact.update(compiler_sha256='8'*64, source_closure_sha256='9'*64, defines=[arm])
                record = dict(status='COMPILED', returncode=0, artifact_sha256=artifact['sha256'],
                              source_sha=artifact['numerical_source_sha'], compiler_sha256='8'*64,
                              source_closure_sha256='9'*64, defines=[arm], vendor=vendor, target=artifact['target'],
                              binding='bindings/fixture.mojo', source_files={'bindings/fixture.mojo': arm.lower()*64},
                              argv=['/compiler/mojo', 'build', '-j', '6', '--target-accelerator',
                                    'sm_90' if vendor == 'nvidia' else 'gfx942', '-D', define, '-D', arm,
                                    '-I', '/src', '/src/bindings/fixture.mojo', '-o', '/output.so'])
                refs[arm] = {artifact['path']: self.write_ref(column+'-'+arm+'-compile.json', record)}
            receipt_ref = self.write_ref(column+'.json', receipt)
            closure = dict(schema=eq.CLOSURE_SCHEMA, status='CAPTURED', method='observed_deployed_files',
                           complete=True, missing=[], source_sha=receipt['source_sha'],
                           receipt_sha256=receipt_ref['sha256'], capture_version='fixture-v1',
                           sections={section: {'fixture-'+section: 'a'*64} for section in eq.SECTIONS})
            entries[column] = dict(source_sha=receipt['source_sha'], receipt=receipt_ref,
                                   reviewed_changed_files=[], compile_receipts=refs,
                                   execution_closure=self.write_ref(column+'-closure.json', closure))
        self.att = dict(schema=eq.SCHEMA, status='REVIEWED', case_id=self.case['id'],
                        configuration_id=self.case['configuration_id'], repository='.',
                        anchor_source_sha=self.case['source_sha'],
                        scope_sha256=eq.digest({k:v for k,v in expected_scope(self.case).items() if k!='source_sha'}),
                        review=dict(reviewer='fixture reviewer', reviewed_at='fixture time', rationale='fixture only'),
                        columns=entries)
        self.case['freeze_equivalence'] = self.write_ref('attestation.json', self.att)

    def write_ref(self, name, value):
        raw = json.dumps(value).encode()
        (self.root/name).write_bytes(raw)
        return dict(path=name, sha256=hashlib.sha256(raw).hexdigest())

    def attest(self):
        self.case['freeze_equivalence'] = self.write_ref('attestation.json', self.att)
        with patch.object(eq, 'reviewed_source_diff', return_value=[]):
            return eq.validate(self.case['freeze_equivalence'], self.root, self.case, expected_scope(self.case),
                               'amd', self.att['columns']['amd']['receipt']['sha256'])

    def change_evidence(self, key, mutation):
        entry = self.att['columns']['amd']
        ref = entry[key] if key == 'execution_closure' else entry['compile_receipts']['A']['amd.so']
        value = json.loads((self.root/ref['path']).read_text()); mutation(value)
        new = self.write_ref(ref['path'], value); ref.update(new)

    def test_reviewed_equivalence_retains_original_source(self):
        self.prepare()
        proof = self.attest()
        self.assertEqual(proof['original_source_sha'], '7'*40)
        self.assertFalse(proof['accepted'])
        with patch.object(eq, 'reviewed_source_diff', return_value=[]):
            report = self.compare()['cases'][0]
        self.assertEqual(self.pair(report)['status'], 'MATCH')
        self.assertEqual(report['columns']['amd']['attempt']['source_sha'], '7'*40)

    def test_unreviewed_or_wrong_scope_rejected(self):
        for mutation in (lambda a:a.update(status='DRAFT'), lambda a:a.update(scope_sha256='0'*64),
                         lambda a:a.update(review={}), lambda a:a.update(case_id='other')):
            self.prepare();mutation(self.att)
            with self.assertRaises(ValueError): self.attest()

    def test_missing_or_reconstructed_execution_closure_rejected(self):
        for mutation in (lambda v:v.update(method='git_reconstruction'), lambda v:v.update(complete=False),
                         lambda v:v.update(missing=['runtime']), lambda v:v['sections'].pop('api'),
                         lambda v:v.update(receipt_sha256='0'*64), lambda v:v.update(capture_version='changed')):
            self.prepare(); self.change_evidence('execution_closure', mutation)
            with self.assertRaises(ValueError): self.attest()

    def test_numerical_flags_compiler_sources_and_backend_rejected(self):
        mutations = [lambda v:v['source_files'].update({'bindings/fixture.mojo':'c'*64}),
                     lambda v:v.update(compiler_sha256='0'*64),
                     lambda v:v['argv'].insert(2, '--fast-math'),
                     lambda v:v['argv'].__setitem__(v['argv'].index('gfx942'), 'gfx'),
                     lambda v:v.update(artifact_sha256='0'*64)]
        for mutation in mutations:
            self.prepare();self.change_evidence('compile',mutation)
            with self.assertRaises(ValueError): self.attest()

    def test_tampered_reference_rejected(self):
        self.prepare();(self.root/'attestation.json').write_text('{}')
        with self.assertRaises(ValueError):
            eq.validate(self.case['freeze_equivalence'],self.root,self.case,expected_scope(self.case),'amd','0'*64)

    def test_observed_output_bits_survive_unqualified_provenance(self):
        self.receipts['amd']['source_sha']='7'*40
        for run in self.receipts['amd']['runs']:
            run['result']['source_sha']='7'*40
            run['result']['model_state']={'status':'UNAVAILABLE'}
        report=self.compare()['cases'][0]
        self.assertEqual(self.pair(report)['status'],'INCOMPLETE')
        pair=next(p for p in report['observed_output_bits']['A'] if p['left']=='nvidia-native' and p['right']=='amd')
        self.assertEqual(pair['status'],'AGREE');self.assertFalse(pair['qualified_identity'])
        self.result()['outputs']['sha256']='f'*64;self.result()['output_sha256']='f'*64
        report=self.compare()['cases'][0]
        pair=next(p for p in report['observed_output_bits']['A'] if p['left']=='nvidia-native' and p['right']=='amd')
        self.assertEqual(pair['status'],'DIFFER')

    def test_source_review_rejects_numerical_change_and_undeclared_file(self):
        class Result:
            stdout=b'bindings/fixture.mojo\0'
        with patch.object(eq.subprocess,'run',return_value=Result()):
            with self.assertRaises(ValueError):eq.reviewed_source_diff('.', '1'*40,'2'*40,['bindings/fixture.mojo'])
            with self.assertRaises(ValueError):eq.reviewed_source_diff('.', '1'*40,'2'*40,[])
        Result.stdout=b'tools/six_lane_prepare_full_tsvd.py\0'
        with patch.object(eq.subprocess,'run',return_value=Result()):
            eq.reviewed_source_diff('.', '1'*40,'2'*40,['tools/six_lane_prepare_full_tsvd.py'])


if __name__=='__main__':unittest.main()
