# SPDX-License-Identifier: Apache-2.0
"""Public boundary of every Sequential layer method added 2026-11-22 (lane nn-surface, 2026-11-23): each is
admitted and named in `layers`, refuses a bad argument in clear words at add, and survives save_checkpoint /
from_checkpoint and export / load with the same predictions, bit for bit. The models are the bounded fixtures
of the verify public endpoints (`_verify_api_neural.SEQUENTIAL_LAYER_FIXTURES`), so the boundary test and the
release record exercise the same calls. The numerics (forward, gradients, host versus device words) are the
native checks checks/native_tests/neural_blocks.mojo, sequence_blocks.mojo and sequential_lengths.mojo."""
import numpy as np
import pytest

import mojolearn
from mojolearn._verify_api_neural import SEQUENTIAL_LAYER_FIXTURES

#: the name `layers` gives the last layer each method appends (containers end on their join)
LAST_KIND = {"lstm": "recurrent", "gru": "recurrent", "rnn": "recurrent", "batch_norm1d": "batch_norm",
             "batch_norm2d": "batch_norm", "residual_block": "residual", "parallel": "concat",
             "gated": "multiply"}


def _inputs(method, shape):
    size = int(np.prod(shape))
    if method in ("embedding", "lm_head"):
        return (np.arange(2 * size, dtype=np.float32) % 5).reshape((2,) + shape)
    return ((np.arange(2 * size, dtype=np.float32) % 7) - 3).reshape((2,) + shape) / 8


def _fitted(method):
    shape, build = SEQUENTIAL_LAYER_FIXTURES[method]
    S = mojolearn.Sequential
    m = build(S(input_shape=shape, numeric_mode="identical"), S)
    x = _inputs(method, shape)
    if method == "lm_head":
        y = ((np.arange(2 * int(np.prod(shape)), dtype=np.int32) + 1) % 5).reshape(2, -1)
        m.compile(loss="token_cross_entropy", optimizer="sgd", lr=0.125)
    else:
        y = np.array([0.25, -0.5], np.float32)
        m.flatten().linear(1).compile(optimizer="sgd", lr=0.125)
    m.fit(x, y, epochs=1, batch_size=2, shuffle=False, seed=7)
    return m, x


@pytest.mark.parametrize("method", sorted(SEQUENTIAL_LAYER_FIXTURES))
def test_layer_is_admitted_and_named(method):
    shape, build = SEQUENTIAL_LAYER_FIXTURES[method]
    S = mojolearn.Sequential
    m = build(S(input_shape=shape, numeric_mode="identical"), S)
    kinds = [layer[0] for layer in m.layers]
    assert kinds and all(kinds), kinds          # no layer is described by an empty name
    assert kinds[-1] == LAST_KIND.get(method, method), kinds


@pytest.mark.parametrize("method", sorted(SEQUENTIAL_LAYER_FIXTURES))
def test_checkpoint_and_export_round_trip(method, tmp_path):
    m, x = _fitted(method)
    want = np.asarray(m.predict(x))
    assert np.isfinite(want).all() or method == "ssd_scan", want   # ssd_scan: see docs/plans/RELEASE_NEW_SURFACE.md
    m.save_checkpoint(str(tmp_path / "m.ckpt"))
    back = mojolearn.Sequential.from_checkpoint(str(tmp_path / "m.ckpt"))
    assert [l[0] for l in back.layers] == [l[0] for l in m.layers]
    assert np.asarray(back.predict(x)).tobytes() == want.tobytes()
    m.export(str(tmp_path / "m.mlx"))
    loaded = mojolearn.Sequential.load(str(tmp_path / "m.mlx"))
    assert np.asarray(loaded.predict(x)).tobytes() == want.tobytes()
    assert np.asarray(back.params_).tobytes() == np.asarray(m.params_).tobytes()


@pytest.mark.parametrize("build,message", [
    (lambda S: S(input_shape=(4,)).silu(1), "takes no arguments"),
    (lambda S: S(input_shape=(4,)).softplus(1), "takes no arguments"),
    (lambda S: S(input_shape=(4,)).gelu(approximate="erf"), "approximate must be 'none' or 'tanh'"),
    (lambda S: S(input_shape=(4,)).rms_norm(eps=0.0), "eps must be positive"),
    (lambda S: S(input_shape=(2, 4)).rotary(1), "head_dim must be an integer >= 2"),
    (lambda S: S(input_shape=(2, 4)).rotary(4, theta=0.0), "theta must be positive"),
    (lambda S: S(input_shape=(4,)).multiply(5), "source must be -1 or the index of an earlier layer"),
    (lambda S: S(input_shape=(2, 4)).narrow(1, 0, 0), "length must be an integer >= 1"),
    (lambda S: S(input_shape=(3,)).embedding(0, 2), "num_embeddings must be an integer >= 1"),
    (lambda S: S(input_shape=(3,)).embedding(5, 0), "embedding_dim must be an integer >= 1"),
    (lambda S: S(input_shape=(4, 8)).attention(0), "heads must be an integer >= 1"),
    (lambda S: S(input_shape=(4, 8)).attention(2, window=-1), "window must be an integer >= 0"),
    (lambda S: S(input_shape=(4, 8)).attention(2, theta=0.0), "theta must be positive"),
    (lambda S: S(input_shape=(3, 2)).lstm(0), "hidden_size must be an integer >= 1"),
    (lambda S: S(input_shape=(3, 2)).rnn(0), "hidden_size must be an integer >= 1"),
    (lambda S: S(input_shape=(4, 321)).ssd_scan(0), "heads must be an integer >= 1"),
    (lambda S: S(input_shape=(4, 30)).trapezoid_scan(2, 0), "d_state must be an integer >= 1"),
    (lambda S: S(input_shape=(3, 4)).lm_head(1), "vocab must be an integer >= 2"),
    (lambda S: S(input_shape=(4,)).gated(S(), S().linear(4)), "gate must hold at least one layer"),
])
def test_layer_refusals_are_clear(build, message):
    with pytest.raises(Exception, match=message):
        build(mojolearn.Sequential)
