"""Revision-copy refusal checks; no numerical execution."""
import pathlib,tempfile,unittest
from identical_wave_revision import copy_checked,digest,inventory
class RevisionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=pathlib.Path(self.temp.name);self.src=self.root/'src';self.dst=self.root/'dst'
        self.path=self.src/'python/mojolearn/identical/module.so';self.path.parent.mkdir(parents=True);self.path.write_bytes(b'fixture')
    def test_verified_copy_and_collision(self):
        records=inventory(self.src);copy_checked(self.src,self.dst,records)
        self.assertEqual(inventory(self.dst),records)
        with self.assertRaisesRegex(ValueError,'already exists'):copy_checked(self.src,self.dst,records)
    def test_tampered_donor(self):
        records=inventory(self.src);self.path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'changed'):copy_checked(self.src,self.dst,records)
    def test_symlink_refused(self):
        self.path.unlink();target=self.root/'target';target.write_bytes(b'fixture');self.path.symlink_to(target)
        with self.assertRaisesRegex(ValueError,'unsafe'):inventory(self.src)
    def test_destination_escape_refused(self):
        self.dst.mkdir();(self.dst/'python').symlink_to(self.root,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'escape'):copy_checked(self.src,self.dst,inventory(self.src))
if __name__=='__main__':unittest.main()
