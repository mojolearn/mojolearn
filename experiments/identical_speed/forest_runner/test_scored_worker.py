import unittest
from scored_worker import require_family
class Family(unittest.TestCase):
 def test_same_file_aliases_are_valid(self):
  rows=[{'module':m,'file':'identical/_mojolearn_rf.so','sha256':'abc'} for m in ['mojolearn._mojolearn_rf','mojolearn._sets.identical._mojolearn_rf']]
  self.assertEqual(require_family({'loaded_files':rows},{'files':{'identical/_mojolearn_rf.so':'abc'}},'identical/_mojolearn_rf.so'),rows)
 def test_missing_family_refused(self):
  with self.assertRaises(ValueError):require_family({'loaded_files':[]},{'files':{'rf.so':'abc'}},'rf.so')
 def test_one_wrong_alias_refused(self):
  with self.assertRaises(ValueError):require_family({'loaded_files':[{'file':'rf.so','sha256':'abc'},{'file':'rf.so','sha256':'wrong'}]},{'files':{'rf.so':'abc'}},'rf.so')
if __name__=='__main__':unittest.main()
