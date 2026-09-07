"""Root-only remote regression gate; authored without execution.

Opt in with MOJOLEARN_NEIGHBORS_EXTENDED_GATE=1 and
MOJOLEARN_EXPECT_VENDOR=cuda|hip. No GPU work occurs without both.
"""
import os
import sys
import unittest

if (os.environ.get("MOJOLEARN_NEIGHBORS_EXTENDED_GATE") != "1"
        or sys.platform != "linux"
        or os.environ.get("MOJOLEARN_EXPECT_VENDOR") not in ("cuda", "hip")):
    raise unittest.SkipTest("Explicit root-only Linux CUDA/HIP opt-in required")

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "MOJOLEARN_CPU_THREADS"):
    os.environ[_name] = "2"

import numpy as np
import mojolearn as ml


class NeighborsWeightedLargeK(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if ml.vendor() != os.environ["MOJOLEARN_EXPECT_VENDOR"]:
            raise RuntimeError("Native vendor does not match the explicit remote target")
        cls.mode = ml.numeric_mode()
        if cls.mode != os.environ.get("MOJOLEARN_NUMERIC_MODE"):
            raise RuntimeError("Native mode does not match the requested mode")

    def test_exact_matches_replace_the_whole_weight_row(self):
        x = np.array([[0], [0], [2]], dtype=np.float32)
        q = np.array([[0]], dtype=np.float32)
        clf = ml.KNeighborsClassifier(n_neighbors=3, weights="distance").fit(
            x, np.array([7, 3, 7], dtype=np.int32))
        np.testing.assert_array_equal(clf.predict(q), [3])
        np.testing.assert_array_equal(clf.predict_proba(q), [[.5, .5]])
        # Two outputs and signed targets: the distant value must contribute
        # exactly zero, rather than a small inverse-distance weight.
        reg = ml.KNeighborsRegressor(n_neighbors=3, weights="distance").fit(
            x, np.array([[2, -8], [6, 4], [100, 100]], dtype=np.float32))
        got = reg.predict(q)
        np.testing.assert_array_equal(got, [[4, -2]])
        if self.mode == "identical":
            # DEVIATION 2460: predictions are mojolearn.Array; bits via np.asarray
            np.testing.assert_array_equal(np.asarray(got).view(np.uint32),
                                          np.asarray(reg.predict(q)).view(np.uint32))

    def test_inverse_distance_changes_the_uniform_winner(self):
        x = np.array([[1], [2], [4]], dtype=np.float32)
        q = np.array([[0]], dtype=np.float32)
        y = np.array([2, 1, 1], dtype=np.int32)
        weighted = ml.KNeighborsClassifier(n_neighbors=3, weights="distance").fit(x, y)
        uniform = ml.KNeighborsClassifier(n_neighbors=3, weights="uniform").fit(x, y)
        np.testing.assert_array_equal(weighted.predict(q), [2])
        np.testing.assert_array_equal(uniform.predict(q), [1])
        np.testing.assert_allclose(weighted.predict_proba(q), [[3/7, 4/7]], rtol=2e-6, atol=1e-7)
        reg = ml.KNeighborsRegressor(n_neighbors=3, weights="distance").fit(
            x, np.array([2, 6, 10], dtype=np.float32))
        np.testing.assert_allclose(reg.predict(q), [30/7], rtol=2e-6, atol=1e-7)

    def test_strided_rank_boundaries_and_index_ties(self):
        if self.mode not in ("identical", "deterministic"):
            self.skipTest("The pinned selector is not the FAST selector")
        # Duplicate distances cross every thread-stride boundary. Values are
        # exact dyadics, so the independent FP64 Manhattan oracle rounds
        # exactly to FP32 without depending on native numerical helpers.
        x = (np.arange(1056, dtype=np.int32) // 3).astype(np.float32)[:, None] / 16
        q = np.array([[0], [4]], dtype=np.float32)
        distance64 = np.abs(q.astype(np.float64)[:, None, :] - x.astype(np.float64)[None, :, :]).sum(axis=2)
        expected = np.stack([np.lexsort((np.arange(len(x)), row)) for row in distance64])
        for k in (256, 257, 300, 513, 1024):
            with self.subTest(k=k):
                a = ml.NearestNeighbors(n_neighbors=k, metric="manhattan", query_tile=1).fit(x)
                b = ml.NearestNeighbors(n_neighbors=k, metric="manhattan", query_tile=2).fit(x)
                d, i = a.kneighbors(q)
                dr, ir = a.kneighbors(q)
                dt, it = b.kneighbors(q)
                np.testing.assert_array_equal(i, expected[:, :k])
                np.testing.assert_array_equal(d, np.take_along_axis(distance64, expected[:, :k], axis=1).astype(np.float32))
                np.testing.assert_array_equal(i, ir)
                np.testing.assert_array_equal(i, it)
                d, dr, dt = np.asarray(d), np.asarray(dr), np.asarray(dt)  # DEVIATION 2460
                np.testing.assert_array_equal(d.view(np.uint32), dr.view(np.uint32))
                np.testing.assert_array_equal(d.view(np.uint32), dt.view(np.uint32))
        with self.assertRaisesRegex(Exception, "1024"):
            ml.NearestNeighbors(n_neighbors=1025, metric="manhattan").fit(x).kneighbors(q)

    def test_unported_kd_tree_remains_explicit(self):
        with self.assertRaisesRegex(ValueError, "kd_tree"):
            ml.NearestNeighbors(algorithm="kd_tree").fit(np.ones((4, 2), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
