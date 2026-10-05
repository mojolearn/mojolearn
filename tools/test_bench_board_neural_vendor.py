"""Neural board metadata only; no libraries, GPU work or benchmarks."""
import os
import types
import unittest
from unittest.mock import patch
import bench_board_neural as neural


class NeuralVendorReadback(unittest.TestCase):
    def library(self,vendor):return types.SimpleNamespace(__version__='source-test',vendor=lambda:vendor)
    def test_metal_library_is_apple_fast_column(self):
        with patch.dict(os.environ,MOJOLEARN_NUMERIC_MODE='fast'):
            for vendor in ('metal','apple'):
                info=neural._ours_info(self.library(vendor),'frozen.so','fast')
                self.assertEqual(info['vendor_used'],vendor)
                self.assertEqual(info['numeric_mode_used'],'fast')
    def test_fast_refuses_non_apple_and_unavailable_vendor(self):
        with patch.dict(os.environ,MOJOLEARN_NUMERIC_MODE='fast'):
            for vendor in ('cuda','hip','cpu','unknown'):
                with self.assertRaisesRegex(RuntimeError,'REFUSED'):
                    neural._ours_info(self.library(vendor),'frozen.so','fast')
    def test_numeric_mode_mismatch_still_refused(self):
        with patch.dict(os.environ,MOJOLEARN_NUMERIC_MODE='fast'):
            with self.assertRaisesRegex(RuntimeError,'ours is not FAST'):
                neural._ours_info(self.library('metal'),'frozen.so','identical')
    def test_identical_other_vendor_scope_unchanged(self):
        with patch.dict(os.environ,MOJOLEARN_NUMERIC_MODE='identical'):
            info=neural._ours_info(self.library('cuda'),'frozen.so','identical')
            self.assertEqual(info['vendor_used'],'cuda')

if __name__=='__main__':unittest.main()
