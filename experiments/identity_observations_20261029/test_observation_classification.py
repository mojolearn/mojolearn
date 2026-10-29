import copy,json,pathlib,unittest,gzip
from observation_classification import classify
B=pathlib.Path(__file__).parent/'runs/apple/neural-inference'
def reports(lane):
 if B.exists():
  p=B/lane;return json.loads((p/'verify.json').read_text()),json.loads((p/'verifier-run.json').read_text())
 retained=pathlib.Path(__file__).resolve().parents[2]/'bench/results/identity_break/retained'
 paths=list(retained.glob('*/apple-'+lane+'.identical.source-report.json.gz'))
 assert len(paths)==1, paths
 p=paths[0];return json.loads(gzip.decompress(p.read_bytes())),json.loads(p.with_name(p.name.replace('.source-report.json.gz','.verifier-run.json')).read_text())
class DecisionTests(unittest.TestCase):
 def setUp(self):
  self.d,self.r=reports('hf-model-surface')
 def test_actual_historical_observation(self):self.assertTrue(classify(self.d,self.r)['observation_complete']);self.assertEqual(self.r['exit_code'],1)
 def test_actual_pass(self):
  d,r=reports('hf-causal-lm');self.assertTrue(classify(d,r)['observation_complete'])
 def test_current_reference_is_not_excused(self):
  self.d['reference_evidence']['columns'][0]['recorded_at_commit']=self.d['device']['commit'];self.assertFalse(classify(self.d,self.r)['observation_complete'])
 def test_local_failure_not_excused(self):
  self.d['cells'][0]['local_check']='failed';self.assertFalse(classify(self.d,self.r)['observation_complete'])
 def test_refusal_not_excused(self):
  self.d['cells'][0]['state']='REFUSED';self.assertFalse(classify(self.d,self.r)['observation_complete'])
 def test_crash_not_excused(self):
  self.d['execution']['crashed_cells']=['base'];self.assertFalse(classify(self.d,self.r)['observation_complete'])
 def test_missing_part_not_excused(self):
  self.d['cells'].pop();self.assertFalse(classify(self.d,self.r)['observation_complete'])
 def test_unknown_nonzero_not_excused(self):
  self.r['exit_code']=2;self.assertFalse(classify(self.d,self.r)['observation_complete'])
 def test_arbitrary_divergence_not_excused(self):
  next(r for r in self.d['cells'] if r['state']=='DIVERGENT')['detail']='different local result';self.assertFalse(classify(self.d,self.r)['observation_complete'])
if __name__=='__main__':unittest.main()
