"""Invalid state metadata must name its field and leave successful KV intact."""
import unittest

import numpy as np
from mojolearn import TransformerBlock
from test_transformer_surface import _weights


class TransformerStateRefusals(unittest.TestCase):
    def refused_without_publication(self, changes, name, error_type):
        rng = np.random.default_rng(20261029)
        block = TransformerBlock(_weights(rng, 8, 2, 1, 4, 16),
                                 n_heads=2, n_kv_heads=1)
        state = block.allocate_state(1, 4)
        x = np.full((1, 1, 8), 0.125, dtype=np.float32)
        y = block.forward(x, state)
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(state.cached_tokens, 1)
        before = (state.k_cache.tobytes(), state.v_cache.tobytes())
        original = {field: getattr(state, field) for field in changes}
        for field, value in changes.items():
            setattr(state, field, value)
        with self.assertRaisesRegex(error_type, name):
            block.step(x, state)
        self.assertEqual((state.k_cache.tobytes(), state.v_cache.tobytes()), before)
        for field, value in changes.items():
            self.assertEqual(getattr(state, field), value)
        if 'cached_tokens' not in changes:
            self.assertEqual(state.cached_tokens, 1)
        for field, value in original.items():
            setattr(state, field, value)
        # Admission failure must not poison a state with successful prior tokens.
        y = block.step(x, state)
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(state.cached_tokens, 2)

    def test_batch_size_refusal_preserves_prior_kv(self):
        self.refused_without_publication({'batch_size': 2}, 'batch_size', ValueError)

    def test_window_refusal_preserves_prior_kv(self):
        self.refused_without_publication({'window': 1}, 'window', ValueError)

    def test_negative_cached_tokens_preserves_prior_kv(self):
        self.refused_without_publication({'cached_tokens': -1}, 'cached_tokens', Exception)

    def test_first_dimension_refusal_keeps_original_precedence(self):
        self.refused_without_publication({'batch_size': 2, 'window': 1, 'cached_tokens': -1},
                                         'batch_size', ValueError)


if __name__ == '__main__':
    unittest.main()
