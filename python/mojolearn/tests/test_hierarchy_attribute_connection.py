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


class NativeMetadataOwnerConnections(unittest.TestCase):
    def test_initialized_owner_exports_are_connected_without_framework_imports(self):
        import importlib
        import sys
        import mojolearn
        from mojolearn import _native_loaded_modules
        frameworks_before = {name for name in ('torch', 'onnx', 'tensorflow') if name in sys.modules}
        owners = {
            'mojolearn._array': '_ARRAY_CONTEXT',
            'mojolearn._expansion_trees': '_TREES_CONTEXT',
            'mojolearn._mode': '_MODE_METADATA_CONTEXT',
            'mojolearn._native_family_bridge': '_CONTEXT',
            'mojolearn.importers._program': '_IMPORT_CONTEXT',
        }
        for name, export in owners.items():
            module = importlib.import_module(name)
            records = [row for row in _native_loaded_modules.CONNECTIONS if row[0] == name]
            self.assertEqual(len(records), 1, name)
            self.assertIs(records[0][1], module)
            self.assertIs(dict(records[0][2])[export], vars(module)[export])
        self.assertEqual(frameworks_before, {name for name in ('torch', 'onnx', 'tensorflow') if name in sys.modules})

    def test_gp_fit_predict_uses_connected_mode_metadata(self):
        import numpy as np
        import mojolearn as ml
        x = np.array([[-2, -1], [-1, 1], [0, -2], [1, 2], [2, 0]], dtype=np.float32)
        y = np.array([-1, 0, 1, 2, 3], dtype=np.float32)
        model = ml.GaussianProcessRegressor(kernel=ml.RBF(1.0), alpha=2.0 ** -20)
        self.assertIs(model.fit(x, y), model)
        self.assertEqual(model.info_, 0)
        result = model.predict(x)
        self.assertEqual(result.shape, (5,))
        self.assertTrue(np.isfinite(result).all())


if __name__ == '__main__':
    unittest.main()
