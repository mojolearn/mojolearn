import hashlib
import unittest
import numpy as np

from tools.attention_v2_backward_oracle import attention_v2_backward


class AttentionV2BackwardOracleTest(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(3110)
        return (rng.standard_normal((37, 9), dtype=np.float32),
                rng.standard_normal((35, 9), dtype=np.float32),
                rng.standard_normal((35, 7), dtype=np.float32),
                rng.standard_normal((37, 7), dtype=np.float32))

    def test_repeatable_tail_tile_and_mask(self):
        q, k, v, dy = self.fixture()
        mask = np.tril(np.ones((37, 35), bool), 2)
        a = attention_v2_backward(q, k, v, dy, np.float32(1 / 3), mask)
        b = attention_v2_backward(q, k, v, dy, np.float32(1 / 3), mask)
        self.assertTrue(all(x.tobytes() == y.tobytes() for x, y in zip(a, b)))
        self.assertEqual(tuple(x.shape for x in a), (q.shape, k.shape, v.shape))
        expected = (
            "932cd778e83535b4675af51c8da7bcdac32e0b14850e8cdf46f3d556f7d1bd5b",
            "01ede2b25915794307237ede5b1feb5fbcff8ee2ac6ab121e77d3bd0a523bccd",
            "9ba8159489ec0eae519776664751032ab6c29bd58c1c823de71377ac8d6eee82",
        )
        self.assertEqual(tuple(hashlib.sha256(x.tobytes()).hexdigest() for x in a),
                         expected)

    def test_ascending_query_fold_has_teeth(self):
        q, k, v, dy = self.fixture()
        good = attention_v2_backward(q, k, v, dy, np.float32(0.375))
        bad = attention_v2_backward(q, k, v, dy, np.float32(0.375),
                                    reverse_query_fold=True)
        self.assertEqual(good[0].tobytes(), bad[0].tobytes())
        self.assertNotEqual(good[1].tobytes(), bad[1].tobytes())
        self.assertNotEqual(good[2].tobytes(), bad[2].tobytes())

    def test_small_float64_witness_is_close(self):
        q, k, v, dy = (x[:4].astype(np.float64) for x in self.fixture())
        got = attention_v2_backward(q, k, v, dy, np.float32(0.5))
        scores = q @ k.T * 0.5
        p = np.exp(scores - scores.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        ds = p * ((dy @ v.T) - (p * (dy @ v.T)).sum(1, keepdims=True)) * 0.5
        want = (ds @ k, ds.T @ q, p.T @ dy)
        for a, b in zip(got, want):
            np.testing.assert_allclose(a, b, rtol=3e-5, atol=3e-5)

    def test_refusals(self):
        q, k, v, dy = self.fixture()
        with self.assertRaisesRegex(ValueError, "mask"):
            attention_v2_backward(q, k, v, dy, mask=np.ones((2, 2), bool))
        mask = np.ones((37, 35), bool); mask[3] = False
        with self.assertRaisesRegex(ValueError, "visible key"):
            attention_v2_backward(q, k, v, dy, mask=mask)


if __name__ == "__main__":
    unittest.main()
