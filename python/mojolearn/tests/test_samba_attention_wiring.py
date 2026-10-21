# SPDX-License-Identifier: Apache-2.0
"""Native ABI/layout checks and an independent full mixed-stack gradient oracle.

These controller tests use the selected installed native backend. They are not
cross-vendor qualification by themselves.
"""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from mojolearn import _native_neural_bridge as bridge
from mojolearn import _samba_impl as S
from mojolearn._transformer_impl import TransformerBlock
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests.test_samba_gradients_native import _model, _reference


def config(tied=True):
    return S.SambaConfig(11, 32, ('attention', 'mamba3', 'attention'),
                         n_heads=2, n_kv_heads=1, head_dim=16,
                         intermediate=48, tie_embeddings=tied)


def weights(cfg):
    return {name: np.full(shape, (j + 1) / 32, np.float32)
            for j, (name, shape) in enumerate(cfg.block_shapes('attention'))}


class TransformerBackwardWiring(unittest.TestCase):
    def test_native_abi_window_and_owned_outputs(self):
        cfg = config()
        for window in (0, 3):
            with self.subTest(window=window):
                w = weights(cfg)
                block = TransformerBlock(w, n_heads=2, n_kv_heads=1,
                                         head_dim=16, window=window,
                                         numeric_mode='identical', numeric_profile='fp32_v1')
                x = np.arange(192, dtype=np.float32).reshape(2, 3, 32)[:, ::-1]
                dy = -x[:, :, ::-1]
                snapshots = [a.copy() for a in (x, *w.values(), dy)]
                outputs = [np.empty_like(value) for value in (snapshots[0], *w.values())]
                addresses = [value.ctypes.data for value in snapshots]
                addresses += [value.ctypes.data for value in outputs]
                # Explicit established ABI is independent of the coordinator's
                # argument assembly and strided-buffer admission.
                block._extension().transformer_backward(
                    addresses, [2, 3, 32, 2, 1, 16, 48, window])
                first = audit_native_call(block.backward, x, dy).assert_native()
                second = audit_native_call(block.backward, x, dy).assert_native()
                self.assertEqual(tuple(first), ('x',) + block._W_NAMES)
                for name, expected in zip(first, outputs):
                    self.assertEqual(first[name].shape, expected.shape)
                    self.assertEqual(first[name].tobytes(), expected.tobytes(), name)
                    self.assertEqual(second[name].tobytes(), expected.tobytes(), name)
                    self.assertFalse(np.shares_memory(first[name], second[name]))
                for actual, saved in zip((x, *w.values(), dy), snapshots):
                    np.testing.assert_array_equal(actual, saved)

    def test_missing_native_backward_is_actionable(self):
        block = TransformerBlock(weights(config()), n_heads=2, n_kv_heads=1,
                                 head_dim=16, numeric_mode='identical', numeric_profile='fp32_v1')
        x = np.zeros((1, 2, 32), np.float32)
        selected = bridge.CONTEXT['training_bindings'][block._native_route + ':identical']
        stale = SimpleNamespace(**vars(selected['transformer']))
        delattr(stale, 'transformer_backward')
        with patch.dict(selected, transformer=stale):
            result = audit_native_call(block.backward, x, x)
        self.assertIsInstance(result.exception, RuntimeError)
        self.assertIn('lacks transformer_backward', str(result.exception))
        for key in ('python_frames', 'forbidden_native_calls', 'wrapper_work', 'unobserved_existing_threads'):
            self.assertFalse(result.report()[key], result.report())

    def test_mixed_stack_reverse_order_and_registry(self):
        for tied in (False, True):
            with self.subTest(tied=tied):
                model = _model(['attention', 'mamba3', 'attention'], tied)
                ids = np.array([[1, 4, 7], [3, 6, 9]], dtype='i4')
                targets = np.array([[4, 7, 10], [6, -100, 12]], dtype='i4')
                expected_loss, expected_gradients = _reference(model, ids, targets, None, None, 0)
                with patch.object(S.SambaStack, '_block', side_effect=AssertionError('Python block walk')):
                    loss, gradients = audit_native_call(model.loss_and_grads, ids, targets).assert_native()
                self.assertEqual(np.float32(loss).tobytes(), np.float32(expected_loss).tobytes())
                self.assertEqual(len(gradients), len(model.names))
                for name, value, expected in zip(model.names, gradients, expected_gradients):
                    self.assertEqual(value.shape, model.shapes[name])
                    self.assertEqual(value.tobytes(), expected, name)


if __name__ == '__main__':
    unittest.main()
