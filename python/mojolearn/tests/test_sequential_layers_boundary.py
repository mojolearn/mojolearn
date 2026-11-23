# SPDX-License-Identifier: Apache-2.0
"""Public boundary of every Sequential layer (lanes nn-surface and layer-tables, 2026-11-23): each is admitted and
named in `layers`, refuses a bad argument in clear words at add, and survives state_dict -> load_state_dict,
save_checkpoint -> from_checkpoint and export -> load with the same predictions, bit for bit. The tests are
generated from the code table (`_x_sequence_model._LAYER`): every layer code is covered through the bounded
fixtures of the verify public endpoints (`_verify_api_neural.SEQUENTIAL_LAYER_FIXTURES`), and a code with no
fixture fails `test_every_layer_code_has_a_constructor` (tools/block_tables_check.py refuses the build too), so a
new layer is covered without anyone remembering to add a test. The numerics (forward, gradients, host versus device
words) are the native checks checks/native_tests/neural_blocks.mojo, sequence_blocks.mojo and
sequential_lengths.mojo."""
import numpy as np
import pytest

import mojolearn
from mojolearn._verify_api_neural import SEQUENTIAL_LAYER_FIXTURES
from mojolearn._x_sequence_model import _LAYER, _SEQUENTIAL_CONTEXT

LAYER_NAMES = _SEQUENTIAL_CONTEXT["layer_names"]
#: the parameter-free core codes verify covers through its own endpoint branches (tools/block_tables_check.py
#: CORE_NO_FIXTURE): relu, sigmoid, tanh, softmax, dropout, residual, reshape, permute, flatten, max_pool2d,
#: avg_pool2d. Every other code, and every new one, needs a fixture.
CORE_NO_FIXTURE = frozenset((2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14))
#: containers (code 0) end on their join layer
CONTAINER_LAST = {"residual_block": "residual", "parallel": "concat", "gated": "multiply"}
#: every layer code of the table, each with the fixtures that build it
CODES = sorted({c for c in _LAYER.values() if c})
FIXTURES_OF = {c: sorted(n for n in SEQUENTIAL_LAYER_FIXTURES if _LAYER.get(n) == c) for c in CODES}


def _last_kind(method):
    return CONTAINER_LAST.get(method) or LAYER_NAMES[_LAYER[method]]


def test_every_layer_code_has_a_constructor():
    missing = [(c, LAYER_NAMES[c] if c < len(LAYER_NAMES) else "?") for c in CODES
               if not FIXTURES_OF[c] and c not in CORE_NO_FIXTURE]
    assert not missing, ("layer codes with no constructor in _verify_api_neural.SEQUENTIAL_LAYER_FIXTURES "
                         f"(add its minimal arguments there): {missing}")
    assert all(n in _LAYER for n in SEQUENTIAL_LAYER_FIXTURES), sorted(set(SEQUENTIAL_LAYER_FIXTURES) - set(_LAYER))


def _inputs(method, shape):
    size = int(np.prod(shape))
    if method in ("embedding", "lm_head"):
        return (np.arange(2 * size, dtype=np.float32) % 5).reshape((2,) + shape)
    return ((np.arange(2 * size, dtype=np.float32) % 7) - 3).reshape((2,) + shape) / 8


def _built(method):
    """(the fixture's model, compiled and not fitted, its x, its y)."""
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
    return m, x, y


def _fitted(method):
    m, x, y = _built(method)
    m.fit(x, y, epochs=1, batch_size=2, shuffle=False, seed=7)
    return m, x


@pytest.mark.parametrize("method", sorted(SEQUENTIAL_LAYER_FIXTURES))
def test_layer_is_admitted_and_named(method):
    shape, build = SEQUENTIAL_LAYER_FIXTURES[method]
    S = mojolearn.Sequential
    m = build(S(input_shape=shape, numeric_mode="identical"), S)
    kinds = [layer[0] for layer in m.layers]
    assert kinds and all(kinds), kinds          # no layer is described by an empty name
    assert kinds[-1] == _last_kind(method), kinds


@pytest.mark.parametrize("code", CODES, ids=[LAYER_NAMES[c] if c < len(LAYER_NAMES) else str(c) for c in CODES])
def test_every_layer_code_round_trips(code, tmp_path):
    """For each layer code: every fixture that builds it fits one step, then state_dict -> load_state_dict into a
    fresh model of the same layers, save_checkpoint -> from_checkpoint and export -> load give the same predictions
    and params_ bit for bit, and the files the same digest_. (A model given its parameters by load_state_dict was
    never fitted, so it holds no fitted loss and y shape: its digest_ is not the fitted model's, by design.)"""
    if not FIXTURES_OF[code]:
        if code in CORE_NO_FIXTURE:
            pytest.skip("parameter-free core layer, covered by its verify endpoint")
        pytest.fail(f"layer code {code} has no constructor in SEQUENTIAL_LAYER_FIXTURES")
    for method in FIXTURES_OF[code]:
        m, x = _fitted(method)
        want = np.asarray(m.predict(x))
        assert np.isfinite(want).all(), (method, want)
        params = np.asarray(m.params_).tobytes()
        state = m.state_dict()
        assert state, method
        fresh, _, _ = _built(method)
        fresh.load_state_dict(state)                  # strict: every tensor the model has is named and used
        assert np.asarray(fresh.params_).tobytes() == params, method
        assert np.asarray(fresh.predict(x)).tobytes() == want.tobytes(), method
        again = fresh.state_dict()
        assert sorted(again) == sorted(state), method
        assert all(np.asarray(again[k]).tobytes() == np.asarray(state[k]).tobytes() for k in state), method
        m.save_checkpoint(str(tmp_path / f"{method}.ckpt"))
        back = mojolearn.Sequential.from_checkpoint(str(tmp_path / f"{method}.ckpt"))
        assert [l[0] for l in back.layers] == [l[0] for l in m.layers], method
        assert np.asarray(back.params_).tobytes() == params, method
        assert np.asarray(back.predict(x)).tobytes() == want.tobytes(), method
        assert back.digest_ == m.digest_, method
        m.export(str(tmp_path / f"{method}.mlx"))
        loaded = mojolearn.Sequential.load(str(tmp_path / f"{method}.mlx"))
        assert np.asarray(loaded.params_).tobytes() == params, method
        assert np.asarray(loaded.predict(x)).tobytes() == want.tobytes(), method
        assert loaded.digest_ == m.digest_, method


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
