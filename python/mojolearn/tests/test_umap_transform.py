# SPDX-License-Identifier: Apache-2.0
"""Main-run UMAP transform gate; existing fit-byte gates remain unchanged."""
import unittest

import numpy as np

from mojolearn import UMAP


class UMAPTransformTests(unittest.TestCase):
    def setUp(self):
        self.x = np.array([0, 1, 2.2, 4, 6.5, 10, 14.5, 20],
                          dtype=np.float32).reshape(-1, 1)

    def model(self, **kwargs):
        return UMAP(n_neighbors=3, n_epochs=12, random_state=19, **kwargs)

    def test_requires_fit_and_valid_queries(self):
        with self.assertRaisesRegex(ValueError, "successful fit"):
            self.model().transform(self.x)
        model = self.model().fit(self.x)
        for value in (np.nan, np.inf, -np.inf):
            with self.assertRaisesRegex(ValueError, "finite"):
                model.transform(np.array([[value]], dtype=np.float32))
        for x in (np.empty((0, 1)), np.zeros((2, 2)), np.zeros(2)):
            with self.assertRaises(ValueError):
                model.transform(x)

    def test_training_shortcut_is_copy_and_snapshots_are_isolated(self):
        original = self.x.copy()
        model = self.model().fit(self.x)
        fitted = model.embedding_.copy()
        first = model.transform(original)
        self.assertEqual(first.tobytes(), fitted.tobytes())
        self.assertFalse(np.shares_memory(first, model.embedding_))
        query = np.array([[0.5], [3], [17]], dtype=np.float32)
        before = model.transform(query)
        self.x[:] = 100
        model.embedding_[:] = -999
        first[:] = 999
        self.assertEqual(model.transform(original).tobytes(), fitted.tobytes())
        np.testing.assert_array_equal(model.transform(query), before)

    def test_unseen_repeat_and_readonly_conversion(self):
        model = self.model().fit(self.x)
        query = np.array([[0.5], [3], [17]], dtype=np.float64)
        query.setflags(write=False)
        saved_embedding = model.embedding_.copy()
        first = model.transform(query)
        second = model.transform(query)
        self.assertEqual(first.shape, (3, 2))
        self.assertEqual(first.dtype, np.float32)
        self.assertTrue(np.isfinite(first).all())
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(model.embedding_, saved_embedding)
        self.assertFalse(np.shares_memory(first, second))
        self.assertGreater(float(np.max(np.abs(first[0] - first[-1]))), 0)

    def test_three_dimensions_and_single_query(self):
        x = np.arange(12, dtype=np.float32).reshape(-1, 1)
        model = self.model(n_components=3).fit(x)
        result = model.transform(np.array([[2.5]], dtype=np.float32))
        self.assertEqual(result.shape, (1, 3))
        self.assertTrue(np.isfinite(result).all())

    def test_parameter_change_requires_refit_and_failed_fit_preserves_model(self):
        model = self.model().fit(self.x)
        query = np.array([[0.5], [3]], dtype=np.float32)
        baseline = model.transform(query)
        model.n_neighbors = 4
        with self.assertRaisesRegex(ValueError, "changed after fit"):
            model.transform(query)
        model.n_neighbors = 3
        fitted_mode = model._transform_mode
        model.numeric_mode = "fast" if fitted_mode != "fast" else "identical"
        with self.assertRaisesRegex(ValueError, "changed after fit"):
            model.transform(query)
        model.numeric_mode = fitted_mode
        with self.assertRaises(ValueError):
            model.fit(np.full_like(self.x, np.nan))
        np.testing.assert_array_equal(model.transform(query), baseline)
        model.fit(self.x + 0.25)
        self.assertTrue(np.isfinite(model.transform(query)).all())


if __name__ == "__main__":
    unittest.main()
