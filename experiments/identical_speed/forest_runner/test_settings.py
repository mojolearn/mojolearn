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
