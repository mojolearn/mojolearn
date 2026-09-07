# SPDX-License-Identifier: Apache-2.0
"""Lightweight archive/mode-routing checks; no native extension calls."""
import ctypes
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from mojolearn import _backend, _serialize
from mojolearn.ensemble import GradientBoosting, OrderedRMSE, _MODEL_FORMAT


class _Binding:
    def __init__(self, mode):
        self.mode = mode

    def gbdt_model_dim(self, text):
        return 1

    def gbdt_predict(self, model, x, out, params):
        bits = {'fast': 0x3f800000, 'deterministic': 0x3f800001, 'identical': 0x3f800002}
        buffer = (ctypes.c_uint32 * params[0]).from_address(out)
        for index in range(params[0]):
            buffer[index] = bits[self.mode]
        return params[0]


def fitted(cls=GradientBoosting, mode='identical'):
    obj = cls(numeric_mode=mode)
    obj.model_ = 'test-model-text'
    obj.n_features_in_ = 1
    obj.approx_dim_ = 1
    obj.n_classes_ = None
    obj.best_iteration_ = None
    obj.stopped_early_ = False
    obj.bias_ = 0.0
    return obj


class GbdtModeSerializationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / 'model.npz'
        self.bound_modes = []
        observed = self.bound_modes

        def bind(obj, name=None):
            mode = getattr(obj, 'numeric_mode', None) or _backend.default_mode()
            observed.append(mode)
            return _Binding(mode)

        self.binding_patch = patch.object(GradientBoosting, '_bind', bind)
        self.binding_patch.start()
        self.addCleanup(self.binding_patch.stop)

    def test_explicit_mode_survives_different_default_for_both_classes(self):
        X = np.zeros((2, 1), dtype=np.float32)
        for cls in (GradientBoosting, OrderedRMSE):
            for mode in ('identical', 'deterministic', 'fast'):
                with self.subTest(cls=cls.__name__, mode=mode):
                    obj = fitted(cls, mode)
                    expected = np.asarray(obj.predict(X)).view(np.uint32).copy()  # DEVIATION 2460
                    obj.save(self.path)
                    self.bound_modes.clear()
                    with patch.object(_backend, 'default_mode', return_value='fast' if mode != 'fast' else 'identical'):
                        restored = cls.load(self.path)
                        self.assertEqual(self.bound_modes, [mode])
                        self.assertEqual(restored.numeric_mode, mode)
                        np.testing.assert_array_equal(np.asarray(restored.predict(X)).view(np.uint32), expected)

    def test_none_pins_effective_default_at_save(self):
        obj = fitted(OrderedRMSE, None)
        with patch.object(_backend, 'default_mode', return_value='identical'):
            obj.save(self.path)
        arrays = _serialize.read_npz(self.path, _MODEL_FORMAT)
        self.assertEqual(_serialize.scalar_str(arrays, 'numeric_mode'), 'identical')
        self.assertIsNone(obj.numeric_mode)  # saving does not mutate the live policy
        with patch.object(_backend, 'default_mode', return_value='fast'):
            restored = OrderedRMSE.load(self.path)
        self.assertEqual(restored.numeric_mode, 'identical')
        self.assertEqual(self.bound_modes, ['identical'])

    def test_corrupt_mode_refused_before_binding(self):
        fitted().save(self.path)
        original = _serialize.read_npz(self.path, _MODEL_FORMAT)
        for bad in (np.asarray('unknown'), np.asarray(''), np.asarray('IDENTICAL'),
                    np.asarray(['identical', 'fast']), np.asarray([], dtype='U1'),
                    np.asarray(1), np.asarray(b'identical')):
            with self.subTest(bad=bad):
                arrays = dict(original, numeric_mode=bad)
                _serialize.write_npz(self.path, arrays)
                self.bound_modes.clear()
                with self.assertRaisesRegex(ValueError, 'numeric_mode'):
                    GradientBoosting.load(self.path)
                self.assertEqual(self.bound_modes, [])

    def test_legacy_file_keeps_process_default(self):
        fitted(OrderedRMSE).save(self.path)
        arrays = _serialize.read_npz(self.path, _MODEL_FORMAT)
        del arrays['numeric_mode']
        _serialize.write_npz(self.path, arrays)
        with patch.object(_backend, 'default_mode', return_value='fast'):
            restored = OrderedRMSE.load(self.path)
        self.assertIsNone(restored.numeric_mode)
        self.assertEqual(self.bound_modes, ['fast'])

    def test_save_normalizes_effective_selection_without_loading_native(self):
        obj = fitted(mode=' IDENTICAL ')
        obj.save(self.path)
        self.assertEqual(self.bound_modes, [])
        arrays = _serialize.read_npz(self.path, _MODEL_FORMAT)
        self.assertEqual(_serialize.scalar_str(arrays, 'numeric_mode'), 'identical')

    def test_invalid_live_mode_does_not_overwrite_archive(self):
        obj = fitted()
        obj.save(self.path)
        original = self.path.read_bytes()
        obj.numeric_mode = 'unsupported'
        with self.assertRaisesRegex(ValueError, 'numeric_mode'):
            obj.save(self.path)
        self.assertEqual(self.path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
