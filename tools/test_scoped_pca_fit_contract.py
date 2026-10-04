#!/usr/bin/env python3
"""Metadata-only tests. No NumPy, native module, model, GPU or compiler import."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from apple_fast_job_policy import policy_for
from scoped_pca_fit import admit_quality, source_contract, METRIC_NAMES
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
