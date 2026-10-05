import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('board',Path(__file__).with_name('bench_board.py'))
B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B)

class Capabilities(unittest.TestCase):
    def test_fast_454_eligible_2_explicit_unsupported_identical_456(self):
        fast=B.plan_races('apple',['fast']); identical=B.plan_races('apple',['identical'])
        self.assertEqual(len(fast),456);self.assertEqual(len(identical),456)
        self.assertEqual(B.plan_summary(fast)['unsupported_races'],2)
        self.assertEqual(B.plan_summary(identical)['unsupported_races'],0)
        self.assertEqual({r['lane'] for r in fast if B.unsupported_our_arms(r)}, {'gemm-bf16','gemm-int8'})

    def test_unsupported_is_recorded_without_running_any_driver(self):
        for lane in ('gemm-bf16','gemm-int8'):
            race=B.plan_races('apple',['fast'],families=['neural'],lanes=[lane],neural_shape='small')[0]
            ctx=dict(vendor='apple',rounds=1,with_opponents=False,retime=False,store_path=None,box={})
            with patch.object(B,'stored_opponents',return_value={}),patch.object(B,'_run_race',side_effect=AssertionError('must not time unsupported API')):
                result=B.run_race(ctx,race)
            self.assertEqual(result['status'],'unsupported')
            self.assertIsNone(result['cells'][0]['median_ms'])
            self.assertEqual(B.smoke_verdict(race,result,'apple'),[])
            result['cells'][0]['median_ms']=12
            self.assertTrue(B.smoke_verdict(race,result,'apple'))

    def test_identical_or_other_fast_failure_is_not_exempted(self):
        for lane,mode in [('gemm-bf16','identical'),('gemm','fast')]:
            race=B.plan_races('apple',[mode],families=['neural'],lanes=[lane])[0]
            self.assertTrue(B.smoke_verdict(race,dict(status='failed',cells=[]),'apple'))
            self.assertTrue(B.smoke_verdict(race,dict(status='unsupported',cells=[],unsupported_arms={}), 'apple'))

    def test_fast_smoke_does_not_qualify_identical(self):
        fast=B.plan_races('apple',['fast'],families=['neural'],lanes=['gemm-bf16'])[0]
        identical=B.plan_races('apple',['identical'],families=['neural'],lanes=['gemm-bf16'])[0]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'smoke.json'
            path.write_text(json.dumps(dict(schema=B.SMOKE_SCHEMA,vendor='apple',files_sha256=B.smoke_files_sha256(),modes=['fast'],races={B.smoke_key(fast):{'pass':True}})))
            self.assertIsNone(B.smoke_gate(tmp,'apple',[fast],[str(path)]))
            self.assertIsNotNone(B.smoke_gate(tmp,'apple',[identical],[str(path)]))

if __name__=='__main__':unittest.main()
