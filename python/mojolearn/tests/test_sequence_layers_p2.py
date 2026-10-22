# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Release gate 2026-10-22 (part 2): the sequence layers whose forward call
changed since 0.8.55 (MoEBlock, LayerNorm, layer_norm_forward/backward,
Embedding), called by name (`forward` and `__call__`) on the real binding
path, each against NumPy, and each refusing a float64 or a bad shape.
"""
import numpy as np
import pytest

import mojolearn as ml


def _silu(v):
    return v / (1 + np.exp(-v))


def _moe_reference(m, x, top_k):
    xd = x.astype(np.float64)
    logits = xd @ m.router.T.astype(np.float64)
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    ref = np.zeros_like(xd)
    for t in range(len(x)):
        idx = np.argsort(-p[t], kind="stable")[:top_k]
        w = p[t, idx] / p[t, idx].sum()
        for j, e in enumerate(idx):
            g, u = np.split(m.gate_up_proj[e].astype(np.float64) @ xd[t], 2)
            ref[t] += w[j] * (m.down_proj[e].astype(np.float64) @ (_silu(g) * u))
    return ref


def test_moe_block_forward_and_call_by_name():
    x = np.random.default_rng(0).standard_normal((10, 6)).astype(np.float32)
    m = ml.MoEBlock(6, 8, num_experts=4, top_k=2, random_state=3)
    ref = _moe_reference(m, x, 2)
    np.testing.assert_allclose(m.forward(x), ref, atol=2e-5)
    np.testing.assert_allclose(m.__call__(x), ref, atol=2e-5)


def test_moe_block_forward_and_call_refuse_float64():
    m = ml.MoEBlock(6, 8, num_experts=4, top_k=2, random_state=3)
    x = np.zeros((3, 6))
    with pytest.raises(TypeError):
        m.forward(x)
    with pytest.raises(TypeError):
        m.__call__(x)
    with pytest.raises(ValueError):
        ml.MoEBlock(6, 8, num_experts=2, top_k=3)


def _ln_reference(x, w, b, eps=1e-5):
    xd = x.astype(np.float64)
    mu = xd.mean(-1, keepdims=True)
    return (xd - mu) / np.sqrt(xd.var(-1, keepdims=True) + eps) * w + b


def test_layer_norm_module_forward_and_call_by_name():
    rng = np.random.default_rng(1)
    x = (rng.standard_normal((5, 7)) * 3 + 2).astype(np.float32)
    ln = ml.LayerNorm(7)
    ln.weight = rng.standard_normal(7).astype(np.float32)
    ln.bias = rng.standard_normal(7).astype(np.float32)
    ref = _ln_reference(x, ln.weight, ln.bias)
    np.testing.assert_allclose(ln.forward(x), ref, atol=2e-5)
    np.testing.assert_allclose(ln.__call__(x), ref, atol=2e-5)


def test_layer_norm_module_refuses_float64_and_wrong_trailing_shape():
    ln = ml.LayerNorm(4)
    with pytest.raises(TypeError):
        ln.forward(np.zeros((2, 4)))
    with pytest.raises(ValueError):
        ln.forward(np.zeros((2, 5), np.float32))
    with pytest.raises(ValueError):
        ln.__call__(np.zeros((2, 3), np.float32))


def test_layer_norm_functions_refuse_bad_inputs():
    x = np.ones((3, 4), np.float32)
    with pytest.raises(TypeError):
        ml.layer_norm_forward(x.astype(np.float64))
    with pytest.raises(ValueError):
        ml.layer_norm_forward(x, weight=np.ones(3, np.float32))
    with pytest.raises(ValueError):
        ml.layer_norm_forward(x, normalized_shape=0)
    with pytest.raises(TypeError):
        ml.layer_norm_backward(np.ones((3, 4)), x)
    with pytest.raises(ValueError):
        ml.layer_norm_backward(np.ones((3, 5), np.float32), x)


def test_embedding_forward_and_call_by_name():
    rng = np.random.default_rng(2)
    w = rng.uniform(-1, 1, (11, 3)).astype(np.float32)
    ids = rng.integers(0, 11, (2, 5)).astype(np.int32)
    e = ml.Embedding(11, 3, weight=w)
    np.testing.assert_array_equal(np.asarray(e.forward(ids)), w[ids])
    np.testing.assert_array_equal(np.asarray(e.__call__(ids.reshape(-1))), w[ids.reshape(-1)])


def test_embedding_forward_and_call_refuse_float_ids():
    w = np.zeros((4, 2), np.float32)
    e = ml.Embedding(4, 2, weight=w)
    with pytest.raises(TypeError):
        e.forward(np.array([0.0, 1.0], np.float32))
    with pytest.raises(TypeError):
        e.__call__(np.array([1.5], np.float64))
