# SPDX-License-Identifier: Apache-2.0
"""Public boundary of ops.Session.attention (lane nn-blocks-qkv, 2026-11-25): the result shapes of the accepted
forms, the refusals in their words, the mask kinds as the call reads them, and a captured program's save / load
round trip. The numerics (the float64 reference, the q, k and v gradients against central differences, grouped
heads, cross-attention, causal, fully masked rows, host versus device words) are the native check
checks/native_tests/nn_qkv.mojo; the cross-vendor digest is the verify lane ops-attention-qkv."""
import os
import tempfile

import numpy as np
import pytest

from mojolearn import ops


def _a(*shape, seed=0):
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def test_shapes_of_the_accepted_forms():
    with ops.Session() as s:
        q, k, v = s.tensor(_a(2, 4, 3, 8)), s.tensor(_a(2, 2, 5, 8, seed=1)), s.tensor(_a(2, 2, 5, 6, seed=2))
        assert s.shape(s.attention(q, k, v)) == (2, 4, 3, 6)            # grouped heads, cross-attention
        q3, k3 = s.tensor(_a(2, 3, 8)), s.tensor(_a(2, 5, 8, seed=1))
        assert s.shape(s.attention(q3, k3, k3, causal=True)) == (2, 3, 8)   # no heads axis


def test_refusals_name_the_problem():
    with ops.Session() as s:
        q, k = s.tensor(_a(1, 2, 4, 3)), s.tensor(_a(1, 2, 4, 3, seed=1))
        with pytest.raises(ValueError, match="causal=True cannot be combined with an explicit mask"):
            s.attention(q, k, k, mask=s.tensor(np.zeros((4, 4), np.float32)), causal=True)
        with pytest.raises(ValueError, match="q and k must have the same feature size"):
            s.attention(q, s.tensor(_a(1, 2, 4, 5)), k)
        with pytest.raises(ValueError, match="must be a multiple of the key/value heads"):
            k3 = s.tensor(_a(1, 3, 4, 3))
            s.attention(q, k3, k3)
        with pytest.raises(ValueError, match="does not broadcast to the scores"):
            s.attention(q, k, k, mask=s.tensor(np.zeros((3, 4), np.float32)))
        with pytest.raises(ValueError, match="must all be"):
            s.attention(s.tensor(_a(4, 3)), k, k)
        with pytest.raises(ValueError, match="floating-point"):
            s.attention(s.ids(np.zeros((1, 2, 4, 3), np.int32)), k, k)
        with pytest.raises(ValueError, match="scale must be a finite number"):
            s.attention(q, k, k, scale=float("inf"))


def test_mask_kinds_and_a_fully_masked_row():
    q, k, v = _a(1, 1, 2, 4), _a(1, 1, 3, 4, seed=1), _a(1, 1, 3, 2, seed=2)
    keep = np.array([[1, 0, 1], [0, 0, 0]], dtype=np.int8)
    with ops.Session() as s:
        hq, hk, hv = s.tensor(q), s.tensor(k), s.tensor(v)
        boolean = s.numpy(s.attention(hq, hk, hv, mask=s.ids(keep)))
        additive = s.numpy(s.attention(hq, hk, hv, mask=s.tensor(np.where(keep == 1, 0.0, -np.inf).astype(np.float32))))
    # the two documented spellings of the same mask agree, and the row with no key left is zeros, not NaN
    np.testing.assert_array_equal(boolean, additive)
    assert np.all(boolean[0, 0, 1] == 0.0)
    assert np.isfinite(boolean).all()


def test_captured_program_save_load_round_trip():
    q, k, v = _a(2, 2, 3, 4), _a(2, 1, 5, 4, seed=1), _a(2, 1, 5, 4, seed=2)
    keep = np.ones((2, 1, 1, 5), dtype=np.int8)
    keep[1, ..., 3:] = 0
    with ops.Session() as s:
        hq, hk, hv, hm = s.tensor(q), s.tensor(k), s.tensor(v), s.ids(keep)
        prog = s.capture([hq, hk, hv, hm], [s.attention(hq, hk, hv, mask=hm)])
        first = np.asarray(prog.run(q, k, v, keep))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "attention.mlp")
            prog.save(path)
            again = np.asarray(s.load_program(path).run(q, k, v, keep))
    np.testing.assert_array_equal(first, again)
