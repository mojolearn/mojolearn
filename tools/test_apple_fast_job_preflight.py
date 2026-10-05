#!/usr/bin/env python3
"""Local metadata fixtures only: no native import, hardware call, or GPU test."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from apple_fast_job_preflight import InfrastructureError,preflight
from apple_fast_job_policy import policy_for

SOURCE='1'*40
OTHER='2'*40
SCRIPT='tools/catalog_resident_timing.py'


class FakeMirror:
    def commit(self,source):
        if source not in (SOURCE,OTHER):raise InfrastructureError('missing mirror commit')
    def blob(self,source,path):
        self.commit(source)
        if path not in (SCRIPT,'tools/catalog_resident_quality.py'):
            raise InfrastructureError('missing pinned file')
        return b'# metadata fixture, never executed\n'


class PreflightFixtures(unittest.TestCase):
    def setUp(self):
        base=Path.home()/'mojolearn-evidence/apple-fast/preflight-fixtures'
        base.mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(prefix='metadata-',dir=base)
        self.home=Path(self.tmp.name)
        mq=self.home/'mq';(mq/'out').mkdir(parents=True)
        (mq/'queue.txt').write_text('CMD old-branch other-tag echo done\n')
        (mq/'results.txt').write_text('CMD old-branch completed-tag rc=0\n')
        pair=mq/'verified-arms'/SOURCE/'resident_gemm_probe';pair.mkdir(parents=True)
        hashes={}
        for arm in ('A','B'):
            data=('fixture-'+arm).encode();(pair/(arm+'.so')).write_bytes(data)
            hashes[arm]=hashlib.sha256(data).hexdigest()
        (pair/'manifest.json').write_text(json.dumps(dict(source_sha=SOURCE,binding='resident_gemm_probe',
            numeric_mode='fast',defines_A='',defines_B='-D PROBE',hashes=hashes)))
        single=mq/'verified-arms'/OTHER/'ibase';single.mkdir(parents=True)
        (single/'_mojolearn.so').write_bytes(b'single-fixture')
        (single/'manifest.json').write_text(json.dumps(dict(source_sha=OTHER,binding='ibase',
            numeric_mode='identical',defines='',artifact='_mojolearn.so',
            sha256=hashlib.sha256(b'single-fixture').hexdigest(),contract='shared-gemm-prerequisite-v1',
            required_exports=['all_finite_f32'])))
        prior=mq/'prior.json';prior.write_text('{"status":"PASS"}')
        self.spec=dict(version=1,tag='future-job',harness_source=SOURCE,script=SCRIPT,
            args=[SOURCE,'future-job',str(prior),hashlib.sha256(prior.read_bytes()).hexdigest()],
            artifacts=[dict(id='probe',kind='pair',compiled_source=SOURCE,binding='resident_gemm_probe',
                numeric_mode='fast',defines_A='',defines_B='-D PROBE'),
                dict(id='ibase',kind='single',compiled_source=OTHER,binding='ibase',numeric_mode='identical',
                defines='',artifact='_mojolearn.so',manifest_equals=dict(contract='shared-gemm-prerequisite-v1',
                required_exports=['all_finite_f32']))],
            prerequisites=[dict(id='quality',kind='file',path=str(prior),
                sha256=hashlib.sha256(prior.read_bytes()).hexdigest(),json_equals=dict(status='PASS'))],
            cases=[dict(name='fixture',requires=['probe','ibase','quality'],source_files=['tools/catalog_resident_quality.py'])])

    def tearDown(self):self.tmp.cleanup()

    def check(self,spec=None):
        return preflight(spec or self.spec,self.home,'Apple M3 Ultra',FakeMirror())

    def test_ready_is_metadata_only_and_read_only(self):
        before={p.relative_to(self.home):p.read_bytes() for p in self.home.rglob('*') if p.is_file()}
        got=self.check()
        self.assertEqual(got['status'],'READY')
        self.assertFalse(got['symbols_validated'])
        self.assertTrue(got['requires_final_helper_gate'])
        self.assertEqual(len(got['dependencies']),3)
        after={p.relative_to(self.home):p.read_bytes() for p in self.home.rglob('*') if p.is_file()}
        self.assertEqual(before,after)

    def test_missing_pair_or_single_binary(self):
        for binding,filename,source in [('resident_gemm_probe','B.so',SOURCE),('ibase','_mojolearn.so',OTHER)]:
            path=self.home/'mq/verified-arms'/source/binding/filename
            data=path.read_bytes();path.unlink()
            with self.assertRaisesRegex(InfrastructureError,'missing artifact'):self.check()
            path.write_bytes(data)

    def test_bad_hash_mode_and_defines(self):
        path=self.home/'mq/verified-arms'/SOURCE/'resident_gemm_probe/manifest.json'
        original=json.loads(path.read_text())
        for key,value in [('numeric_mode','identical'),('defines_B','-D WRONG'),('hashes',{'A':'0'*64,'B':'0'*64})]:
            data=copy.deepcopy(original);data[key]=value;path.write_text(json.dumps(data))
            with self.assertRaises(InfrastructureError):self.check()
        path.write_text(json.dumps(original))

    def test_duplicate_queue_result_or_output(self):
        for name in ('queue.txt','results.txt'):
            path=self.home/'mq'/name;old=path.read_text();path.write_text(old+'CMD branch future-job cmd\n')
            with self.assertRaisesRegex(InfrastructureError,'duplicate tag'):self.check()
            path.write_text(old)
        (self.home/'mq/out/future-job-quality').mkdir()
        with self.assertRaisesRegex(InfrastructureError,'already exists'):self.check()

    def test_missing_case_dependency_source_and_prerequisite(self):
        spec=copy.deepcopy(self.spec);spec['cases'][0]['requires'].append('missing')
        with self.assertRaisesRegex(InfrastructureError,'missing declared dependency'):self.check(spec)
        spec=copy.deepcopy(self.spec);spec['cases'][0]['source_files']=['tools/absent.py']
        with self.assertRaisesRegex(InfrastructureError,'missing pinned file'):self.check(spec)
        (self.home/'mq/prior.json').write_text('{"status":"FAIL"}')
        with self.assertRaisesRegex(InfrastructureError,'hash mismatch'):self.check()

    def test_policy_and_source_fail_closed(self):
        for key,value in [('script','tools/not_allowed.py'),('harness_source','short'),('args',[OTHER])]:
            spec=copy.deepcopy(self.spec);spec[key]=value
            with self.assertRaises(InfrastructureError):self.check(spec)
        with self.assertRaises(InfrastructureError):
            preflight(self.spec,self.home,'Apple M2 Pro',FakeMirror())
        self.assertEqual(policy_for(SOURCE,SCRIPT,[SOURCE,'tag','report','hash']),'verified-resident-timing')

    def test_missing_state_is_not_empty_queue(self):
        (self.home/'mq/results.txt').unlink()
        with self.assertRaisesRegex(InfrastructureError,'missing queue state'):self.check()


if __name__=='__main__':unittest.main()
