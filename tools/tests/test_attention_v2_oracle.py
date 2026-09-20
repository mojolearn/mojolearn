import unittest
import numpy as np
from tools.attention_v2_oracle import online_row, eager_row, workspace_bytes, materialized_bytes

class AttentionV2OracleTest(unittest.TestCase):
    def fixture(self):
        r=np.random.default_rng(2657)
        return r.standard_normal(97).astype(np.float32), r.standard_normal((97,17)).astype(np.float32)
    def test_repeatable_and_fixed_tile(self):
        s,v=self.fixture(); a=online_row(s,v); b=online_row(s,v)
        self.assertEqual(a[0].tobytes(),b[0].tobytes()); self.assertEqual(a[1:],b[1:])
        with self.assertRaisesRegex(ValueError,"fixed"): online_row(s,v,16)
    def test_intentional_v1_separation(self):
        s,v=self.fixture()
        self.assertNotEqual(online_row(s,v)[0].tobytes(), eager_row(s,v).tobytes())
        np.testing.assert_allclose(online_row(s,v)[0],eager_row(s,v),rtol=3e-6,atol=3e-6)
    def test_memory_is_linear_not_quadratic(self):
        lean=workspace_bytes(8,12,2048,64); full=materialized_bytes(8,12,2048,2048)
        self.assertEqual(lean, 8*12*2048*66*4)
        self.assertGreater(full, lean*50)

if __name__ == '__main__': unittest.main()
