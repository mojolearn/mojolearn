import tempfile,pathlib,unittest
from settings import worker_environment
class Settings(unittest.TestCase):
 def test_explicit_vendor_and_clean_flags(self):
  with tempfile.TemporaryDirectory() as d:
   for n in ['libgcc_s.so.1','libstdc++.so.6']:(pathlib.Path(d)/n).touch()
   for vendor,expected in [('amd','hip'),('nvidia','cuda')]:
    e=worker_environment(vendor,d,{'GBM_BENCH_DATA':'/data'},{'PATH':'/bin','MOJOLEARN_SPEED_EXPECTED_VENDOR':'metal','MOJOLEARN_MOJO_BUILD_FLAGS':'bad','LD_PRELOAD':'wrong'})
    self.assertEqual(e['MOJOLEARN_SPEED_EXPECTED_VENDOR'],expected);self.assertEqual(e['MOJOLEARN_VENDOR'],expected)
    self.assertNotIn('MOJOLEARN_MOJO_BUILD_FLAGS',e);self.assertEqual(e['MOJOLEARN_SPEED_ROUNDS'],'1');self.assertNotIn('wrong',e['LD_PRELOAD'])
 def test_runtime_missing_refuses(self):
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(ValueError):worker_environment('amd',d,{'GBM_BENCH_DATA':'/data'}, {})
 def test_data_cannot_override_vendor(self):
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(AssertionError):worker_environment('amd',d,{'GBM_BENCH_DATA':'/data','MOJOLEARN_VENDOR':'cuda'}, {})
if __name__=='__main__':unittest.main()

class Selection(unittest.TestCase):
 def test_exact_selected_cells(self):
  from settings import planned_cells
  d={'vendor':'amd','arms':['rf-k4','rf-k1','rf-k2','rf-k8','et-float','et-u16'],'selected_cells':[{'vendor':'amd','family':'rf','dataset':'taxi','profile':p} for p in ['rf-k4','rf-k2']]}
  self.assertEqual(planned_cells(d),{'rf-taxi-rf-k4','rf-taxi-rf-k2'})
 def test_missing_baseline_refuses(self):
  from settings import planned_cells
  with self.assertRaises(AssertionError):planned_cells({'vendor':'amd','arms':['rf-k2'],'selected_cells':[{'vendor':'amd','family':'rf','dataset':'taxi','profile':'rf-k2'}]})
 def test_wrong_vendor_refuses(self):
  from settings import planned_cells
  with self.assertRaises(AssertionError):planned_cells({'vendor':'amd','arms':['rf-k4'],'selected_cells':[{'vendor':'nvidia','family':'rf','dataset':'taxi','profile':'rf-k4'}]})
