"""The recall oracle must retain the full stable-sort reference exactly."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

TOOLS = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bench_board_algos", TOOLS / "bench_board_algos.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class RecallSelectionTest(unittest.TestCase):
    def test_stable_prefix_including_boundary_ties_and_nonfinite(self):
        rng = np.random.default_rng(7)
        cases = [rng.normal(size=(19, 103)), rng.integers(-3, 4, (19, 103)).astype(float),
                 np.zeros((3, 103)), np.full((3, 103), np.inf),
                 np.array([[np.nan, np.inf, -np.inf, 0., -0., 1., np.nan]])]
        for d in cases:
            for k in (0, 1, 3, 10, d.shape[1], d.shape[1] + 1):
                with self.subTest(shape=d.shape, k=k):
                    np.testing.assert_array_equal(bench._stable_topk_indices(d, k),
                                                  np.argsort(d, axis=1, kind="stable")[:, :k])

    def test_recall_matches_old_oracle_with_filters_and_duplicates(self):
        rng = np.random.default_rng(8)
        X = rng.integers(-2, 3, (73, 4)).astype(np.float32)
        X[20:30] = X[0]
        Q = rng.integers(-2, 3, (259, 4)).astype(np.float32)
        ind = rng.integers(0, len(X), (len(Q), 10))
        for allowed in (None, np.arange(len(X)) % 3 == 0, np.arange(len(X)) < 3):
            hits = 0
            xx = X.astype(np.float64)
            for start in range(0, len(Q), 256):
                q = Q[start:start + 256].astype(np.float64)
                d = (xx * xx).sum(1)[None, :] - 2 * q @ xx.T
                if allowed is not None:
                    d[:, ~allowed] = np.inf
                truth = np.argsort(d, axis=1, kind="stable")[:, :10]
                for i, row in enumerate(truth):
                    hits += len(set(row) & set(ind[start + i]))
            expected = hits / float(len(Q) * 10)
            self.assertEqual(bench._recall({"index": X, "queries": Q}, ind, allowed=allowed), expected)


if __name__ == "__main__":
    unittest.main()
