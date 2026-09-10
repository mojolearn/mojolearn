#!/usr/bin/env python3
"""Require fresh dispatch and compare complete results to explicit zero caches."""
import numpy as np
from mojolearn import TransformerBlock
from mojolearn.tests.test_transformer_surface import _weights


def error_text(call):
    try:
        call()
    except Exception as exc:
        return str(exc)
    raise AssertionError('expected refusal')


def main():
    total = 0
    for hd in (64, 128):
        dm, nh, nk, it = hd * 2, 2, 1, hd * 4
        rng = np.random.default_rng(9107 + hd)
        weights = _weights(rng, dm, nh, nk, hd, it)
        for window in (0, 5, 129):
            block = TransformerBlock(weights, n_heads=nh, n_kv_heads=nk,
                                     window=window)
            ext = block._extension()
            assert hasattr(ext, 'transformer_forward_fresh'), 'fresh entry absent'
            def no_host_cache(*args, **kwargs):
                raise AssertionError('fresh path allocated discarded host cache')
            allocate = block.allocate_state
            for length in (1, 7, 9, 65):
                x = rng.uniform(-0.5, 0.5, (2, length, dm)).astype(np.float32)
                state = allocate(2, length)
                expected = block.forward(x, state)
                assert state.cached_tokens == length
                block.allocate_state = no_host_cache
                try:
                    actual = block.forward(x)
                finally:
                    block.allocate_state = allocate
                assert np.array_equal(actual.view(np.uint32), expected.view(np.uint32)), (hd, window, length)
                total += actual.size
                print('TF_FRESH_CASE_PASS', hd, window, length, actual.size)
            x = np.zeros((2, 1, dm), dtype=np.float32)
            # Two poisoned caller-backed weights prove the original ordered
            # refusal, plus absence of a cache of previously validated weights.
            old_q = weights['q_proj.weight'].flat[0].copy()
            old_gate = weights['gate_proj.weight'].flat[0].copy()
            weights['q_proj.weight'].flat[0] = np.inf
            weights['gate_proj.weight'].flat[0] = np.nan
            try:
                fresh_error = error_text(lambda: block.forward(x))
                ordinary_error = error_text(lambda: block.forward(x, allocate(2, 1)))
                assert fresh_error == ordinary_error and 'q_proj.weight' in fresh_error
            finally:
                weights['q_proj.weight'].flat[0] = old_q
                weights['gate_proj.weight'].flat[0] = old_gate
            # The original device capacity constructor must still refuse before
            # any quadratic stage allocation, including sliding-window calls.
            too_long = np.zeros((1, 8193, dm), dtype=np.float32)
            fresh_error = error_text(lambda: block.forward(too_long))
            ordinary_error = error_text(lambda: block.forward(too_long, allocate(1, 8193)))
            assert fresh_error == ordinary_error and 'absolute-position ceiling' in fresh_error
    print('TF_FRESH_PASS', total, 'output cells; live dispatch, mutable refusals and capacity guards')


if __name__ == '__main__':
    main()
