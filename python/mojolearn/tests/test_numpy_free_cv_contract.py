"""Installed NumPy-free public cross-validation and backing-buffer admission."""
from array import array
import importlib.util
import sys
import unittest


class NumpyFreeCrossValidationContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec('numpy') is not None:
            raise unittest.SkipTest('run this dependency-free contract in a clean installed-wheel environment')
        import mojolearn
        cls.ml = mojolearn

    def check_folds(self, values):
        model = self.ml.RandomForestClassifier(n_estimators=2, max_depth=2, numeric_mode='identical')
        result = self.ml.cross_val_score(model, values, [i % 2 for i in range(16)], cv=2)
        self.assertIs(type(result), self.ml.Array)
        self.assertEqual(result.shape, (2,))
        self.assertTrue(all(0 <= score <= 1 for score in result.tolist()))
        self.assertFalse(any(name == 'numpy' or name.startswith('numpy.') for name in sys.modules))

    def test_native_array_snapshot_for_public_folds(self):
        values = self.ml.Array.from_list([[float(i), float(i % 3)] for i in range(16)], '<f4')
        self.check_folds(values)

    def test_memoryview_backing_for_public_folds(self):
        store = array('f', (value for i in range(16) for value in (float(i), float(i % 3))))
        values = memoryview(store).cast('B').cast('f', shape=(16, 2))
        self.check_folds(values)

    def test_released_view_keeps_its_admission_error(self):
        values = memoryview(array('f', range(32))).cast('B').cast('f', shape=(16, 2))
        values.release()
        model = self.ml.RandomForestClassifier(n_estimators=2, max_depth=2, numeric_mode='identical')
        with self.assertRaises(ValueError) as error:
            self.ml.cross_val_score(model, values, [i % 2 for i in range(16)], cv=2)
        self.assertIn('released memoryview', str(error.exception))


if __name__ == '__main__':
    unittest.main()
