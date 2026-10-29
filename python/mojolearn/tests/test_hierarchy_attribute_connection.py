"""Installed hierarchy execution consumes its connected attribute-plan owner."""
import unittest
from unittest.mock import patch


class HierarchyAttributeConnection(unittest.TestCase):
    @staticmethod
    def data():
        import numpy as np
        return np.array([[0, 0], [0, 1], [1, 0], [1, 1],
                         [8, 8], [8, 9], [9, 8], [9, 9]], dtype=np.float32)

    def test_fit_publishes_fitted_fields(self):
        import mojolearn as ml
        model = ml.AgglomerativeClustering(n_clusters=2)
        with self.assertRaises(AttributeError):
            _ = model.labels_
        self.assertIs(model.fit(self.data()), model)
        self.assertEqual(model.labels_.shape, (8,))
        self.assertEqual(set(model.labels_.tolist()), {0, 1})
        self.assertEqual(model.children_.shape, (7, 2))
        self.assertEqual(model.n_leaves_, 8)

    def test_changed_context_is_refused_and_original_remains_usable(self):
        import mojolearn as ml
        from mojolearn import _native_public_attributes as attributes
        model = ml.AgglomerativeClustering(n_clusters=2)
        values = self.data()
        with patch.object(attributes, '_CONTEXT', {}):
            with self.assertRaisesRegex(TypeError, 'module connection export changed'):
                model.fit(values)
        model.fit(values)
        self.assertEqual(model.labels_.shape, (8,))


if __name__ == '__main__':
    unittest.main()
