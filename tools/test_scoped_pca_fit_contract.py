#!/usr/bin/env python3
"""Metadata-only tests. No NumPy, native module, model, GPU or compiler import."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from apple_fast_job_policy import policy_for
from scoped_pca_fit import admit_quality, source_contract, METRIC_NAMES, compare, record
from scoped_pca_fit_spec import make_spec

H='a'*40
C='b'*40
D='c'*64


class ContractTests(unittest.TestCase):
    def test_spec_artifacts_and_data(self):
        s=make_spec(H,C,'test-q','quality','/data/big-istella.npz',D)
        self.assertEqual(s['args'],[C,'test-q','quality','/data/big-istella.npz',D])
        self.assertEqual(s['artifacts'][0]['defines_B'].count('-D '),4)
        self.assertEqual(s['cases'][0]['requires'],['estimators','board-data'])
        self.assertEqual(s['prerequisites'][0]['sha256'],D)

    def test_timing_requires_receipt(self):
        with self.assertRaises(ValueError):
            policy_for(H,'tools/scoped_pca_fit.py',[C,'test-t','timing','/data',D])
        s=make_spec(H,C,'test-t','timing','/data',D,'/q/report.json',D)
        self.assertEqual(s['prerequisites'][1]['json_equals']['scored'],False)

    def test_policy_rejects_bad_source_tag_hash(self):
        for args in (['main','t','quality','/data',D],[C,'../t','quality','/data',D],
                     [C,'t','quality','/data','invalid']):
            with self.assertRaises(ValueError):policy_for(H,'tools/scoped_pca_fit.py',args)

    def test_source_rejects_native_drift(self):
        with patch('scoped_pca_fit.subprocess.check_output',side_effect=[H,'decomposition/estimator.mojo\n']), \
             patch('scoped_pca_fit.subprocess.run'),self.assertRaises(AssertionError):
            source_contract(C)

    def test_source_allows_only_reviewed_tools(self):
        with patch('scoped_pca_fit.subprocess.check_output',side_effect=[H,'tools/scoped_pca_fit.py\n']), \
             patch('scoped_pca_fit.subprocess.run'):
            self.assertEqual(source_contract(C),H)

    def test_foreign_scalar_comparisons_serialize_as_builtin_bool(self):
        class ForeignBool:
            def __init__(self, value): self.value=value
            def __bool__(self): return self.value
        class ForeignFloat(float):
            def __le__(self, other): return ForeignBool(super().__le__(other))
        foreign=ForeignFloat(0.0)
        with self.assertRaises(TypeError): json.dumps(foreign <= foreign)
        with patch('scoped_pca_fit.metrics', return_value={k:foreign for k in METRIC_NAMES}):
            ok,rows=compare(Path('/unused'))
        self.assertIs(ok,True)
        self.assertTrue(all(type(v['A']) is float and type(v['pass_no_worse']) is bool for v in rows.values()))
        json.dumps(rows,allow_nan=False)

    def test_serialization_failure_creates_no_partial_receipt(self):
        with patch.object(Path,'open') as opened,self.assertRaises(TypeError):
            record(Path('/unused'),dict(unsupported=object()))
        opened.assert_not_called()

    def test_recovery_metadata_pins_all_six_inputs(self):
        from scoped_pca_recover import FILES, spec
        self.assertEqual(len(FILES),6)
        self.assertTrue(all(len(v)==64 and all(c in '0123456789abcdef' for c in v) for v in FILES.values()))
        s=spec(H,'recovery-test')
        self.assertEqual(s['artifacts'],[])
        self.assertEqual(len(s['prerequisites']),6)
        self.assertEqual(policy_for(H,s['script'],s['args']),'reference')

    def test_recovery_retains_capture_identity_not_recovery_identity(self):
        from scoped_pca_recover import partial_identity, COMPILED, CAPTURE, CAPTURE_HELPER, CONTRACT, DATA, BINARIES, A_FLAGS, B_FLAGS
        identity=dict(source_sha=COMPILED,harness_source=CAPTURE,helper_sha=CAPTURE_HELPER,
                      contract=CONTRACT,bound=5e-6,error_regression_allowance=0,data_sha=DATA,status='HOLD',
                      manifest=dict(source_sha=COMPILED,binding='estimators',numeric_mode='fast',
                                    defines_A=A_FLAGS,defines_B=B_FLAGS,hashes=BINARIES))
        prefix=json.dumps(identity,indent=2)[:-2]+',\n  "metrics": {\n'
        self.assertEqual(partial_identity(prefix),identity)
        with self.assertRaises(AssertionError):partial_identity(prefix.replace('"HOLD"','"PASS"'))

    def receipt(self):
        return dict(status='PASS',scored=False,source_sha=C,
                    metrics={k:dict(A=0.0,B=0.0,pass_no_worse=True) for k in METRIC_NAMES},
                    packets={k:D for k in ('A.npz','B.npz','A.npz.json','B.npz.json','oracle.npz')},
                    A=dict(counts=[0,0,0,0,0,0,1,0,0]),B=dict(counts=[0,0,0,0,0,0,0,1,0]))

    def test_admission_rejects_regression_missing_packets_and_no_reach(self):
        good=self.receipt()
        bad=[]
        for field,value in [('status','HOLD'),('scored',True),('source_sha',H),('packets',{}),('metrics',{})]:
            r=copy.deepcopy(good);r[field]=value;bad.append(r)
        r=copy.deepcopy(good);r['B']['counts']=[0]*9;bad.append(r)
        r=copy.deepcopy(good);r['metrics']['mean_relative']['B']=1.0;bad.append(r)
        for r in bad:
            with patch.object(Path,'read_text',return_value=json.dumps(r)), \
                 patch('scoped_pca_fit.sha',return_value=D),self.assertRaises(AssertionError):
                admit_quality(Path('/quality/report.json'),dict(source_sha=C))
        with patch.object(Path,'read_text',return_value=json.dumps(good)), \
             patch('scoped_pca_fit.sha',return_value=D):
            self.assertEqual(admit_quality(Path('/quality/report.json'),dict(source_sha=C))['status'],'PASS')


if __name__=='__main__':unittest.main()
