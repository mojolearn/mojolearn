"""The seven brute force metrics of lane/neighbors-rest (2026-09-15).

Two halves. The TABLE AND THE REFUSALS need no binding: they are the name
table, the refusal messages and the parameter rules of
`python/mojolearn/neighbors.py`. The VALUES need one, GPU or the CPU host
set, and are skipped where neither loads; they hold each metric to its
definition in float64 (`neighbors/impl/distance/detail/distance_ops.mojo`,
THE SEVEN METRICS) and to the exact facts a wrong op would break: hamming's
values are multiples of 1/n_features, an inner product search returns the
LARGEST products first, and a self query is at distance zero under every
metric that has one.

Run: `cd python && python3 -m mojolearn.tests.test_knn_extended_metrics`.
"""
import unittest

import numpy as np

import mojolearn as ml
from mojolearn import neighbors as nb


def _binding_works():
    """Whether a k-NN search runs here at all (a GPU binding, or the CPU
    host set on a box with no GPU). The value tests need one."""
    try:
        x = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        ml.NearestNeighbors(n_neighbors=2).fit(x).kneighbors(x[:1])
        return True
    except Exception:
        return False


HAVE_BINDING = _binding_works()

#: A positive fixture: jensenshannon refuses a negative entry and the log
#: arms want spread, so every value is in (0, 4].
RNG = np.random.default_rng(11)
INDEX = (RNG.random((64, 5), dtype=np.float32) * 4.0 + 0.05).astype(np.float32)
QUERIES = (RNG.random((7, 5), dtype=np.float32) * 4.0 + 0.05).astype(np.float32)


def _reference(metric, q, x):
    """The metric as mathematics in float64, one query row against every
    index row. A third spelling of `neighbors/checks/metric_oracle.mojo`'s
    float64 reference, in numpy."""
    q = np.asarray(q, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    k = x.shape[1]
    if metric == "canberra":
        num = np.abs(q - x)
        den = np.abs(q) + np.abs(x)
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0).sum(axis=1)
    if metric == "braycurtis":
        return np.abs(q - x).sum(axis=1) / np.abs(q + x).sum(axis=1)
    if metric == "correlation":
        a = q - q.mean()
        b = x - x.mean(axis=1, keepdims=True)
        return 1.0 - (a * b).sum(axis=1) / np.sqrt((a * a).sum() * (b * b).sum(axis=1))
    if metric == "jensenshannon":
        m = 0.5 * (q + x)
        s = np.where(q > 0, q * np.log(np.where(q > 0, q, 1.0) / m), 0.0).sum(axis=1)
        s += np.where(x > 0, x * np.log(np.where(x > 0, x, 1.0) / m), 0.0).sum(axis=1)
        return np.sqrt(np.maximum(0.5 * s, 0.0))
    if metric == "hamming":
        return (q != x).sum(axis=1) / k
    if metric == "russellrao":
        return (k - ((q != 0) & (x != 0)).sum(axis=1)) / k
    if metric == "inner_product":
        return (q * x).sum(axis=1)
    raise AssertionError(metric)


class MetricTable(unittest.TestCase):
    def test_every_new_name_resolves_to_its_cuvs_value(self):
        want = {
            "canberra": 8, "correlation": 10, "jensenshannon": 15,
            "inner_product": 6, "braycurtis": 14, "hamming": 16,
            "russellrao": 18,
        }
        for name, value in want.items():
            got, arg = nb._resolve_metric("NearestNeighbors", name, 2)
            self.assertEqual(got, value, name)
            self.assertEqual(arg, 2.0)

    def test_haversine_is_refused_by_name_with_its_reason(self):
        with self.assertRaises(ValueError) as cm:
            nb._resolve_metric("NearestNeighbors", "haversine", 2)
        self.assertIn("NOT IMPLEMENTED", str(cm.exception))

    def test_an_unknown_name_is_still_unknown(self):
        with self.assertRaises(ValueError) as cm:
            nb._resolve_metric("NearestNeighbors", "not_a_metric", 2)
        self.assertIn("unknown metric", str(cm.exception))

    def test_p_is_refused_as_inert_under_a_new_metric(self):
        with self.assertRaises(ValueError) as cm:
            nb._resolve_metric("NearestNeighbors", "canberra", 3)
        self.assertIn("p is read only by metric='minkowski'", str(cm.exception))

    def test_the_ball_cover_refuses_all_seven_with_the_inequality(self):
        for name in ("canberra", "correlation", "jensenshannon",
                     "inner_product", "braycurtis", "hamming", "russellrao"):
            with self.assertRaises(ValueError) as cm:
                nb._resolve_rbc_metric("NearestNeighbors", name, 2)
            self.assertIn("triangle inequality", str(cm.exception), name)
            self.assertIn("algorithm='brute'", str(cm.exception), name)

    def test_distance_weighting_over_a_similarity_is_refused(self):
        for cls in (ml.KNeighborsClassifier, ml.KNeighborsRegressor):
            est = cls(n_neighbors=3, metric="inner_product", weights="distance")
            with self.assertRaises(ValueError) as cm:
                est._check_refusals()
            self.assertIn("weights='distance'", str(cm.exception))
            # uniform is fine under the same metric
            cls(n_neighbors=3, metric="inner_product")._check_refusals()


@unittest.skipUnless(HAVE_BINDING, "no k-NN binding on this box")
class MetricValues(unittest.TestCase):
    def _search(self, metric, k=5):
        est = ml.NearestNeighbors(n_neighbors=k, metric=metric).fit(INDEX)
        dist, idx = est.kneighbors(QUERIES)
        return np.asarray(dist, dtype=np.float64), np.asarray(idx)

    def test_each_metric_matches_its_float64_definition(self):
        # The tolerance is the formula's cost, as `metric_check.mojo` prices
        # it: correlation's expansion cancels, jensen-shannon composes logs.
        tol = {"canberra": 1e-5, "braycurtis": 1e-5, "correlation": 1e-2,
               "jensenshannon": 1e-3, "hamming": 0.0, "russellrao": 0.0,
               "inner_product": 1e-5}
        for metric, rel in tol.items():
            dist, idx = self._search(metric)
            for r in range(QUERIES.shape[0]):
                ref = _reference(metric, QUERIES[r], INDEX)
                got = dist[r]
                want = ref[idx[r]]
                np.testing.assert_allclose(
                    got, want, rtol=max(rel, 1e-12), atol=1e-6,
                    err_msg=f"{metric} row {r}")

    def test_each_metric_returns_the_right_neighbour_set(self):
        for metric in ("canberra", "braycurtis", "correlation",
                       "jensenshannon", "hamming", "russellrao"):
            dist, idx = self._search(metric)
            for r in range(QUERIES.shape[0]):
                ref = _reference(metric, QUERIES[r], INDEX)
                kth = np.sort(ref)[4]
                # every returned neighbour is within the true k-th distance
                # (ties at the boundary may pick either member)
                self.assertTrue((ref[idx[r]] <= kth + 1e-5).all(), metric)
                self.assertTrue(np.all(np.diff(dist[r]) >= -1e-6), metric)

    def test_inner_product_returns_the_largest_products_first(self):
        dist, idx = self._search("inner_product")
        for r in range(QUERIES.shape[0]):
            ref = _reference("inner_product", QUERIES[r], INDEX)
            self.assertTrue(np.all(np.diff(dist[r]) <= 1e-6))
            self.assertAlmostEqual(float(dist[r][0]), float(ref.max()), places=4)

    def test_hamming_values_are_multiples_of_one_over_n_features(self):
        dist, _ = self._search("hamming")
        scaled = dist * INDEX.shape[1]
        np.testing.assert_allclose(scaled, np.round(scaled), atol=1e-6)

    def test_a_self_query_is_at_distance_zero(self):
        for metric in ("canberra", "braycurtis", "jensenshannon", "hamming"):
            est = ml.NearestNeighbors(n_neighbors=1, metric=metric).fit(INDEX)
            dist, idx = est.kneighbors(INDEX[:8])
            np.testing.assert_array_equal(np.asarray(idx).reshape(-1), np.arange(8))
            np.testing.assert_allclose(np.asarray(dist, dtype=np.float64), 0.0, atol=1e-6)

    def test_jensenshannon_refuses_a_negative_entry_by_name(self):
        bad = INDEX.copy()
        bad[3, 1] = -0.5
        est = ml.NearestNeighbors(n_neighbors=2, metric="jensenshannon").fit(bad)
        with self.assertRaises(Exception) as cm:
            est.kneighbors(QUERIES)
        self.assertIn("jensenshannon", str(cm.exception))

    def test_correlation_refuses_a_constant_row_by_name(self):
        bad = INDEX.copy()
        bad[5, :] = np.float32(2.0)
        est = ml.NearestNeighbors(n_neighbors=2, metric="correlation").fit(bad)
        with self.assertRaises(Exception) as cm:
            est.kneighbors(QUERIES)
        self.assertIn("correlation", str(cm.exception))

    def test_a_saved_model_reloads_and_searches_the_same(self):
        import tempfile, os
        for metric in ("canberra", "correlation", "inner_product"):
            est = ml.NearestNeighbors(n_neighbors=4, metric=metric).fit(INDEX)
            want = np.asarray(est.kneighbors(QUERIES)[0])
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, "m.npz")
                est.save(path)
                back = ml.NearestNeighbors.load(path)
                got = np.asarray(back.kneighbors(QUERIES)[0])
            np.testing.assert_array_equal(
                got.view(np.uint32), want.view(np.uint32), err_msg=metric)


if __name__ == "__main__":
    unittest.main()
