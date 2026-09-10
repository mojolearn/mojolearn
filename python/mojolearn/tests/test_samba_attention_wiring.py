# SPDX-License-Identifier: Apache-2.0
"""CPU-only wiring checks; mocked numerics make no GPU correctness claim.

Root: PYTHONPATH=python python -m unittest mojolearn.tests.test_samba_attention_wiring
"""
import ctypes
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from mojolearn import _samba_impl as S
from mojolearn._transformer_impl import TransformerBlock


def view(address, shape):
    return np.ctypeslib.as_array(
        (ctypes.c_float * int(np.prod(shape))).from_address(address)).reshape(shape)


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
                                         numeric_mode='identical')
                x = np.arange(192, dtype=np.float32).reshape(2, 3, 32)[:, ::-1]
                dy = -x[:, :, ::-1]
                snapshots = [a.copy() for a in (x, *w.values(), dy)]
                templates = [x, *w.values()]
                calls = []

                def native(addresses, params):
                    self.assertEqual(params, [2, 3, 32, 2, 1, 16, 48, window])
                    self.assertEqual(len(addresses), 21)
                    for address, expected in zip(addresses[:11], snapshots):
                        np.testing.assert_array_equal(view(address, expected.shape), expected)
                    for j, (address, template) in enumerate(zip(addresses[11:], templates)):
                        view(address, template.shape)[:] = j + len(calls) * 20
                    calls.append(params)

                with patch.object(block, '_extension', return_value=SimpleNamespace(transformer_backward=native)):
                    first = block.backward(x, dy)
                    second = block.backward(x, dy)
                self.assertEqual(tuple(first), ('x',) + block._W_NAMES)
                for j, name in enumerate(first):
                    np.testing.assert_array_equal(first[name], np.full(templates[j].shape, j, np.float32))
                    np.testing.assert_array_equal(second[name], np.full(templates[j].shape, j + 20, np.float32))
                    self.assertFalse(np.shares_memory(first[name], second[name]))
                for actual, saved in zip((x, *w.values(), dy), snapshots):
                    np.testing.assert_array_equal(actual, saved)

    def test_missing_native_backward_is_actionable(self):
        block = TransformerBlock(weights(config()), n_heads=2, n_kv_heads=1,
                                 head_dim=16, numeric_mode='identical')
        x = np.zeros((1, 2, 32), np.float32)
        with patch.object(block, '_extension', return_value=SimpleNamespace()):
            with self.assertRaisesRegex(RuntimeError, 'rebuild bindings/build_transformer.sh'):
                block.backward(x, x)

    def test_mixed_stack_reverse_order_and_registry(self):
        for tied in (False, True):
            with self.subTest(tied=tied):
                # Bypass optimizer construction; run the real loss_and_grads
                # routing and real attention block construction/backward.
                st = S.SambaStack.__new__(S.SambaStack)
                st.config = cfg = config(tied)
                st.numeric_mode = 'identical'
                st.names = [n for n, _ in cfg.registry()]
                st.arrays = {n: np.ones(shape, np.float32) for n, shape in cfg.registry()}
                shape = (2, 3, 32)
                ids = np.zeros((2, 3), np.int32)
                xs = [np.full(shape, i + 1, np.float32) for i in range(3)]
                acts = dict(ids=ids, key=None, xs=xs, h=xs[-1], hn=xs[-1],
                            logits=np.zeros((6, 11), np.float32))
                order = []
                original_block = st._block
                expected = {}

                def block_at(i):
                    def backward(x, dy):
                        order.append(i)
                        np.testing.assert_array_equal(x, xs[i])
                        np.testing.assert_array_equal(dy, np.full(shape, 100 if i == 2 else (i + 2) * 10, np.float32))
                        out = {'x': np.full(shape, (i + 1) * 10, np.float32)}
                        for j, (name, dims) in enumerate(cfg.block_shapes(cfg.layers[i])):
                            out[name] = np.full(dims, 1000 + i * 100 + j, np.float32)
                            expected['layers.%d.%s' % (i, name)] = out[name].copy()
                        return out
                    if cfg.layers[i] == 'mamba3':
                        return SimpleNamespace(backward=backward)
                    block = original_block(i)
                    def native(addresses, params):
                        out = backward(view(addresses[0], shape), view(addresses[10], shape))
                        for address, value in zip(addresses[11:], out.values()):
                            view(address, value.shape)[:] = value
                    block._extension = lambda: SimpleNamespace(transformer_backward=native)
                    return block

                def emb_backward(dy, flat_ids, vocab, mode):
                    np.testing.assert_array_equal(dy, np.full((6, 32), 10, np.float32))
                    return np.full((11, 32), 7, np.float32)

                head = np.full((11, 32), 5, np.float32)
                norm = np.full((32,), 9, np.float32)
                with patch.object(st, '_forward', return_value=acts), \
                     patch.object(st, '_block', side_effect=block_at), \
                     patch.object(S.T, 'cross_entropy', return_value=(2.5, np.zeros((6, 11), np.float32))), \
                     patch.object(S.T, 'linear_backward', return_value=(np.zeros((6, 32), np.float32), head)), \
                     patch.object(S.T, 'rms_norm_backward', return_value=(np.full(shape, 100, np.float32), norm)), \
                     patch.object(S.T, 'embedding_backward', side_effect=emb_backward), \
                     patch.object(S.T, 'accumulate_grads', side_effect=lambda parts, **kw: parts[0] + parts[1]) as accumulate:
                    loss, grads = st.loss_and_grads(ids, ids)
                self.assertEqual(order, [2, 1, 0])
                self.assertEqual(loss, 2.5)
                expected['embed.weight'] = np.full((11, 32), 12 if tied else 7, np.float32)
                expected['norm_f.weight'] = norm
                if not tied:
                    expected['lm_head.weight'] = head
                self.assertEqual(len(grads), len(st.names))
                for name, grad in zip(st.names, grads):
                    np.testing.assert_array_equal(grad, expected[name])
                self.assertEqual(accumulate.call_count, int(tied))


if __name__ == '__main__':
    unittest.main()
