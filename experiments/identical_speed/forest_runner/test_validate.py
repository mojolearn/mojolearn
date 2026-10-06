import tempfile,pathlib,unittest
from validate import full_log
GOOD='''FSPEED-HEADER family=forest lane=rf arm=ours mode=IDENTICAL device=GPU rounds=1 size=shipped
FSPEED-SCALE stage=before rows=4110786 features=16 input_mib=251 large_candidate=true
FSPEED-WARMUP lane=rf arm=ours shape=taxi-4110786x16 ms=100.0
FSPEED lane=rf arm=ours shape=taxi-4110786x16 round=1 ms=99.0 hash=0123456789abcdef
FSPEED-SCALE stage=after rows=4110786 features=16 input_mib=251 large_candidate=true
'''
class Gate(unittest.TestCase):
 def check(self,s):
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d)/'log';p.write_text(s);return full_log(p,'rf','taxi')
 def test_accept_exact(self):self.assertEqual(self.check(GOOD)['scored_ms'],99)
 def test_reject_nonfinite(self):
  with self.assertRaises(AssertionError):self.check(GOOD.replace('ms=99.0','ms=nan'))
 def test_reject_reduced(self):
  with self.assertRaises(AssertionError):self.check(GOOD.replace('rows=4110786','rows=50000'))
 def test_reject_no_sample(self):
  with self.assertRaises(AssertionError):self.check('\n'.join(x for x in GOOD.splitlines() if not x.startswith('FSPEED ')))
 def test_reject_refusal(self):
  with self.assertRaises(AssertionError):self.check(GOOD+'FSPEED-REFUSED lane=rf arm=ours reason=quality error\n')
 def test_reject_opponent(self):
  with self.assertRaises(AssertionError):self.check(GOOD.replace('arm=ours','arm=sklearn'))
if __name__=='__main__':unittest.main()
