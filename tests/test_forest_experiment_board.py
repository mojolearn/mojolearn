import importlib.util,pathlib,unittest
p=pathlib.Path(__file__).parents[1]/'tools/forest_experiment_board.py';s=importlib.util.spec_from_file_location('board',p);b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
def rows():
 return [dict(vendor=v,dataset=d,profile=p,family='rf',status='CAPTURED_PENDING_COMPARISON',source_sha='s',harness_sha='h',settings={'samples':1},hardware=v,scored_ms=10 if p=='rf-k4' else 9,state=dict(input_hash=d,inputs={'x':d},model_params_record={'trees':500},model_native_config={'depth':8},model_state_hash=d,prediction_hash=d,metrics=[{'metric':'auc','value':.9}])) for v in ['nvidia','amd'] for d in ['taxi','istella'] for p in ['rf-k4','rf-k1']]
class BoardTests(unittest.TestCase):
 def test_complete_pair(self):
  r=b.evaluate(rows())[0];self.assertEqual(r['status'],'CROSS_VENDOR_COMPARABLE_REQUIRES_DECISION');self.assertAlmostEqual(r['combined_ratio'],.9);self.assertFalse(r['promoted'])
 def test_incomplete_not_decided(self):self.assertEqual(b.evaluate(rows()[:-1])[0]['status'],'PENDING_REQUIRED_CELLS')
 def test_wrong_bits_refused(self):
  r=rows();r[-1]['state']['model_state_hash']='bad';self.assertEqual(b.evaluate(r)[0]['status'],'REFUSED_COMPARISON')
 def test_different_input_refused(self):
  r=rows();r[-1]['state']['input_hash']='bad';self.assertEqual(b.evaluate(r)[0]['status'],'REFUSED_COMPARISON')
 def test_crossbox_ab_refused(self):
  r=rows();r[-1]['hardware']='otherAMD';self.assertEqual(b.evaluate(r)[0]['status'],'REFUSED_COMPARISON')
 def test_failed_evidence_never_promoted(self):
  r=rows();r[-1]['status']='REFUSED_EVIDENCE';self.assertEqual(b.evaluate(r)[0]['status'],'PENDING_REQUIRED_CELLS')
if __name__=='__main__':unittest.main()
