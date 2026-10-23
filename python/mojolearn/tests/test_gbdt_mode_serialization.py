# SPDX-License-Identifier: Apache-2.0
"""GBDT archive numeric-mode checks through the real public fit, save, load
and predict (small inputs)."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from mojolearn import _backend, _serialize
from mojolearn.ensemble import GradientBoosting, OrderedRMSE, _MODEL_FORMAT

_X = np.random.default_rng(0).random((32, 3), dtype=np.float32)
_Y = (2.0 * _X[:, 0] + _X[:, 1]).astype(np.float32)


def _other(mode):
    return 'fast' if mode == 'identical' else 'identical'


def fitted(cls=GradientBoosting, mode='identical'):
    if cls is OrderedRMSE:
        obj = OrderedRMSE(n_estimators=2, max_depth=2, numeric_mode=mode)
        return obj.fit(_X, _Y, permutation=list(range(len(_Y))))
    obj = GradientBoosting(n_estimators=2, max_depth=2, bootstrap_type='No', random_strength=0.0,
                           numeric_mode=mode)
    return obj.fit(_X, _Y)


class GbdtModeSerializationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / 'model.npz'

    def _fitted(self, cls=GradientBoosting, mode='identical'):
        try:
            return fitted(cls, mode)
        except ImportError as error:
            self.skipTest(f"this install has no {mode} GBDT binding: {error}")

    def test_explicit_mode_survives_different_default_for_both_classes(self):
        for cls in (GradientBoosting, OrderedRMSE):
            for mode in ('identical', 'fast'):
                with self.subTest(cls=cls.__name__, mode=mode):
                    obj = self._fitted(cls, mode)
                    expected = np.asarray(obj.predict(_X)).view(np.uint32).copy()
                    obj.save(self.path)
                    with patch.object(_backend, '_DEFAULT_MODE', _other(mode)):
                        restored = cls.load(self.path)
                        self.assertEqual(restored.numeric_mode, mode)
                        np.testing.assert_array_equal(np.asarray(restored.predict(_X)).view(np.uint32), expected)

    def test_none_pins_effective_default_at_save(self):
        obj = self._fitted(OrderedRMSE, None)
        # The native save reads the process default the backend settled.
        with patch.object(_backend, '_DEFAULT_MODE', 'identical'):
            obj.save(self.path)
        arrays = _serialize.read_npz(self.path, _MODEL_FORMAT)
        self.assertEqual(_serialize.scalar_str(arrays, 'numeric_mode'), 'identical')
        self.assertIsNone(obj.numeric_mode)  # saving does not mutate the live policy
        with patch.object(_backend, '_DEFAULT_MODE', 'fast'):
            restored = OrderedRMSE.load(self.path)
        self.assertEqual(restored.numeric_mode, 'identical')

    def test_corrupt_mode_refused(self):
        self._fitted().save(self.path)
        original = _serialize.read_npz(self.path, _MODEL_FORMAT)
        for bad in (np.asarray('unknown'), np.asarray(''), np.asarray('IDENTICAL'),
                    np.asarray(['identical', 'fast']), np.asarray([], dtype='U1'),
                    np.asarray(1), np.asarray(b'identical')):
            with self.subTest(bad=bad):
                arrays = dict(original, numeric_mode=bad)
                _serialize.write_npz(self.path, arrays)
                with self.assertRaisesRegex(ValueError, 'numeric_mode'):
                    GradientBoosting.load(self.path)

    def test_legacy_file_keeps_process_default(self):
        self._fitted(OrderedRMSE).save(self.path)
        arrays = _serialize.read_npz(self.path, _MODEL_FORMAT)
        del arrays['numeric_mode']
        _serialize.write_npz(self.path, arrays)
        restored = OrderedRMSE.load(self.path)
        self.assertIsNone(restored.numeric_mode)
        self.assertEqual(np.asarray(restored.predict(_X)).shape, (len(_X),))

    def test_save_normalizes_effective_selection(self):
        obj = self._fitted(mode=' IDENTICAL ')
        obj.save(self.path)
        arrays = _serialize.read_npz(self.path, _MODEL_FORMAT)
        self.assertEqual(_serialize.scalar_str(arrays, 'numeric_mode'), 'identical')

    def test_invalid_live_mode_does_not_overwrite_archive(self):
        obj = self._fitted()
        obj.save(self.path)
        original = self.path.read_bytes()
        obj.numeric_mode = 'unsupported'
        with self.assertRaisesRegex(ValueError, 'numeric_mode'):
            obj.save(self.path)
        self.assertEqual(self.path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
