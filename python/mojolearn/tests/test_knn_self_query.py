"""`kneighbors(X=None)`, the all-kNN query (lane/neighbors-rest, 2026-09-15).

scikit-learn's `_base.py:868-889` rule: search at `k + 1`, drop each row's
own index, and drop the FIRST column instead for a row whose own index was
crowded out by duplicate points. The tests hold the result to a manual
construction from `kneighbors(X)` BIT FOR BIT, so a change in the rule
cannot pass by being close.

Run: `cd python && python3 -m mojolearn.tests.test_knn_self_query`.
"""
import unittest

import numpy as np

import mojolearn as ml


def _binding_works():
    try:
        x = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        ml.NearestNeighbors(n_neighbors=2).fit(x).kneighbors(x[:1])
        return True
    except Exception:
        return False


HAVE_BINDING = _binding_works()

RNG = np.random.default_rng(5)
X = RNG.standard_normal((48, 4)).astype(np.float32)


@unittest.skipUnless(HAVE_BINDING, "no k-NN binding on this box")
class SelfQuery(unittest.TestCase):
    def test_no_row_is_its_own_neighbour(self):
        est = ml.NearestNeighbors(n_neighbors=5).fit(X)
        dist, idx = est.kneighbors(None)
        idx = np.asarray(idx)
        self.assertEqual(idx.shape, (X.shape[0], 5))
        self.assertFalse((idx == np.arange(X.shape[0])[:, None]).any())
        self.assertTrue(np.all(np.diff(np.asarray(dist, dtype=np.float64)) >= -1e-7))

    def test_equals_the_manual_construction_bit_for_bit(self):
        est = ml.NearestNeighbors(n_neighbors=5).fit(X)
        got_d, got_i = est.kneighbors(None)
        # the same search at k + 1 on the fitted rows, with each row's own
        # index removed by hand
        full_d, full_i = est.kneighbors(X, n_neighbors=6)
        full_d = np.asarray(full_d)
        full_i = np.asarray(full_i)
        want_d = np.empty((X.shape[0], 5), dtype=np.float32)
        want_i = np.empty((X.shape[0], 5), dtype=np.int64)
        for r in range(X.shape[0]):
            keep = [j for j in range(6) if full_i[r, j] != r]
            if len(keep) == 6:
                keep = list(range(1, 6))
            keep = keep[:5]
            want_d[r] = full_d[r, keep]
            want_i[r] = full_i[r, keep]
        np.testing.assert_array_equal(
            np.asarray(got_d).view(np.uint32), want_d.view(np.uint32))
        np.testing.assert_array_equal(np.asarray(got_i), want_i)

    def test_duplicate_points_drop_the_first_column(self):
        # five identical rows and one far point: for each duplicate row the
        # k + 1 = 3 nearest are three OTHER duplicates in index order, so the
        # row's own index is absent and the first column goes.
        dup = np.zeros((6, 2), dtype=np.float32)
        dup[:5] = np.float32(1.0)
        dup[5] = np.float32(9.0)
        est = ml.NearestNeighbors(n_neighbors=2).fit(dup)
        dist, idx = est.kneighbors(None)
        idx = np.asarray(idx)
        self.assertEqual(idx.shape, (6, 2))
        self.assertFalse((idx == np.arange(6)[:, None]).any())
        np.testing.assert_allclose(np.asarray(dist, dtype=np.float64)[:5], 0.0, atol=0)

    def test_return_distance_false_returns_indices_only(self):
        est = ml.NearestNeighbors(n_neighbors=3).fit(X)
        idx = est.kneighbors(None, return_distance=False)
        self.assertEqual(np.asarray(idx).shape, (X.shape[0], 3))

    def test_k_must_leave_room_for_the_dropped_self_edge(self):
        small = X[:4]
        est = ml.NearestNeighbors(n_neighbors=4).fit(small)
        # k = 4 is fine with an explicit X (four points, four neighbours)
        est.kneighbors(small)
        with self.assertRaises(ValueError) as cm:
            est.kneighbors(None)
        self.assertIn("excludes each point from its own", str(cm.exception))

    def test_the_ball_cover_arm_answers_the_same_query(self):
        brute = ml.NearestNeighbors(n_neighbors=4).fit(X)
        rbc = ml.NearestNeighbors(n_neighbors=4, algorithm="rbc").fit(X)
        bd, bi = brute.kneighbors(None)
        rd, ri = rbc.kneighbors(None)
        np.testing.assert_array_equal(np.asarray(ri), np.asarray(bi))
        np.testing.assert_allclose(
            np.asarray(rd, dtype=np.float64), np.asarray(bd, dtype=np.float64),
            rtol=1e-6, atol=1e-6)

    def test_radius_neighbors_keeps_the_self_edge(self):
        # the documented difference: RadiusNeighbors(X=None) keeps it
        est = ml.RadiusNeighbors(radius=10.0).fit(X[:8])
        _, idx = est.radius_neighbors(None, sort_results=True)
        for r in range(8):
            self.assertIn(r, np.asarray(idx[r]).tolist())


if __name__ == "__main__":
    unittest.main()
