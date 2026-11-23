# SPDX-License-Identifier: Apache-2.0
"""Public boundary of the 1-D sequence blocks and the residual_block / parallel containers
(lane seq-blocks, 2026-11-22): the flat description each call writes and the refusals at add.
The numerics (forward, gradients, the composed baselines, host versus device words) are the
native check checks/native_tests/sequence_blocks.mojo."""
import pytest

import mojolearn


def _kinds(model):
    return [layer[0] for layer in model.layers]


def test_one_d_layers_describe_their_arguments():
    m = (mojolearn.Sequential(input_shape=(2, 16)).conv1d(4, 3, padding=1).max_pool1d(3, stride=1, padding=1)
         .avg_pool1d(2).adaptive_avg_pool1d(2).flatten().linear(2))
    assert _kinds(m) == ["conv1d", "max_pool1d", "avg_pool1d", "adaptive_avg_pool1d", "flatten", "linear"]
    assert m.layers[0][1] == (4, 3, 1, 1, 1, 1)
    assert m.layers[1][1] == (3, 1, 1, 0, 0, 0)
    assert m.layers[2][1] == (2, 2, 0, 0, 0, 0)
    assert m.layers[3][1] == (2, 0, 0, 0, 0, 0)


def test_residual_block_with_shortcut_is_flattened():
    body = mojolearn.Sequential().conv1d(8, 3, padding=1).relu().conv1d(8, 3, padding=1)
    shortcut = mojolearn.Sequential().conv1d(8, 1)
    m = mojolearn.Sequential(input_shape=(2, 16)).conv1d(4, 1).residual_block(body, shortcut).relu()
    assert _kinds(m) == ["conv1d", "conv1d", "relu", "conv1d", "branch", "conv1d", "residual", "relu"]
    assert m.layers[4][1][0] == 0      # the branch returns to the block input (layer 0)
    assert m.layers[6][1] == 3         # the residual adds the body's last layer
    assert len(body.layers) == 3 and len(shortcut.layers) == 1


def test_identity_residual_block_adds_the_block_input():
    body = mojolearn.Sequential().conv1d(4, 3, padding=1)
    m = mojolearn.Sequential(input_shape=(4, 16)).residual_block(body)
    assert _kinds(m) == ["conv1d", "residual"]
    assert m.layers[1][1] == -1        # the block input is the model input


def test_parallel_concatenates_branch_ends():
    a = mojolearn.Sequential().conv1d(4, 9, padding=4)
    b = mojolearn.Sequential().conv1d(4, 5, padding=2)
    c = mojolearn.Sequential().max_pool1d(3, stride=1, padding=1).conv1d(4, 1)
    m = mojolearn.Sequential(input_shape=(2, 16)).conv1d(3, 1).parallel(a, b, c)
    assert _kinds(m) == ["conv1d", "conv1d", "branch", "conv1d", "branch", "max_pool1d", "conv1d", "concat"]
    assert m.layers[7][1] == (1, 3, 6, -2, -2, -2)


def test_nested_references_are_renumbered():
    inner = mojolearn.Sequential().conv1d(4, 3, padding=1).residual(-1)
    m = mojolearn.Sequential(input_shape=(4, 16)).relu().parallel(inner, mojolearn.Sequential())
    # the inner residual's -1 (its own input) becomes the block input, layer 0
    assert m.layers[2] == ("residual", 0, 0, 0.0)
    assert m.layers[3][1] == (2, 0, -2, -2, -2, -2)


def test_container_refusals():
    m = mojolearn.Sequential(input_shape=(2, 16))
    with pytest.raises(Exception, match="Sequential"):
        m.residual_block("conv")
    with pytest.raises(Exception, match="at least one layer"):
        m.residual_block(mojolearn.Sequential())
    with pytest.raises(Exception, match="1 to 6 branches"):
        m.parallel(*[mojolearn.Sequential().relu() for _ in range(7)])
    with pytest.raises(Exception, match="half of kernel_size"):
        m.max_pool1d(2, padding=2)


# ---------------------------------------------------------------- lane seq-blocks-3: batch norm and lengths
# The numerics are the native checks checks/native_tests/sequence_blocks.mojo (batch norm) and
# sequential_lengths.mojo (lengths); these tests hold the Python boundary: what the calls accept, write and refuse.
import numpy as np


def _tracks(n=24, t=6, f=2, seed=3):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, t, f)).astype(np.float32)
    y = (x[:, 0, 0] > 0).astype(np.int64)
    lengths = (np.arange(n) % t + 1).astype(np.int64)
    return x, y, lengths


def _gru(t=6, f=2):
    return (mojolearn.Sequential(input_shape=(t, f)).gru(4).narrow(0, t - 1, 1).flatten().linear(2)
            .compile(loss="cross_entropy", optimizer="adam", lr=0.01))


def test_batch_norm_describes_its_arguments_and_state():
    m = (mojolearn.Sequential(input_shape=(3, 8)).conv1d(4, 3, padding=1).batch_norm1d(4, eps=1e-3, momentum=0.25)
         .relu().flatten().linear(2))
    assert _kinds(m) == ["conv1d", "batch_norm", "relu", "flatten", "linear"]
    args = m.layers[1][1]
    assert args[0] == 4 and args[2] == 1
    assert np.int32(args[1]).view(np.float32) == np.float32(0.25)   # the momentum's float32 bits
    assert m.layers[1][2] == pytest.approx(1e-3)
    m2 = mojolearn.Sequential(input_shape=(2, 3, 3)).batch_norm2d(2).flatten().linear(1)
    assert m2.layers[0][1][2] == 2


def test_batch_norm_running_statistics_are_saved_and_not_trained():
    x = np.random.default_rng(0).standard_normal((16, 2, 8)).astype(np.float32)
    y = np.random.default_rng(1).standard_normal((16, 1)).astype(np.float32)
    m = (mojolearn.Sequential(input_shape=(2, 8)).batch_norm1d(2, momentum=0.5).flatten().linear(1)
         .compile(loss="mse", optimizer="adamw", lr=0.1, weight_decay=0.9))
    m.fit(x, y, epochs=1, batch_size=16, shuffle=False)
    p = np.asarray(m.params_)
    mean = x.transpose(1, 0, 2).reshape(2, -1).mean(axis=1)
    var = x.transpose(1, 0, 2).reshape(2, -1).var(axis=1, ddof=1)
    np.testing.assert_allclose(p[4:6], 0.5 * mean, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(p[6:8], 0.5 + 0.5 * var, rtol=1e-5, atol=1e-6)
    state = m.state_dict()
    assert {"0.weight", "0.bias", "0.running_mean", "0.running_var"} <= set(state)
    np.testing.assert_array_equal(np.asarray(state["0.running_var"]).ravel(), p[6:8])


@pytest.mark.parametrize("kind,args,message", [
    ("batch_norm1d", (3,), "num_features"),
    ("batch_norm1d", (2,), None),
])
def test_batch_norm_refusals(kind, args, message):
    m = mojolearn.Sequential(input_shape=(2, 8))
    if message is None:
        with pytest.raises(Exception, match="momentum"):
            getattr(m, kind)(*args, momentum=1.5)
        with pytest.raises(Exception, match="eps"):
            getattr(m, kind)(*args, eps=0.0)
        return
    m = getattr(m, kind)(*args).flatten().linear(1).compile()
    with pytest.raises(Exception, match=message):
        m.fit(np.zeros((4, 2, 8), np.float32), np.zeros((4, 1), np.float32), epochs=1)


def test_lengths_are_admitted_on_fit_predict_and_predict_proba():
    x, y, lengths = _tracks()
    m = _gru().fit(x, y, epochs=1, batch_size=8, lengths=lengths)
    assert m.predict(x, lengths=lengths).shape == (24,)
    proba = m.predict_proba(x, lengths=lengths.astype(np.float64))
    assert proba.shape == (24, 2)
    # a list of ints is an array-like too
    assert m.predict(x[:3], lengths=[6, 2, 1]).shape == (3,)


@pytest.mark.parametrize("bad,message", [
    (np.zeros(24, np.int64), "whole number from 1"),
    (np.full(24, 7, np.int64), "whole number from 1"),
    (np.full(24, 2.5), "whole number from 1"),
    (np.ones(23, np.int64), "one length per sample"),
    (np.ones((24, 1), np.int64), "one length per sample"),
    (np.array(["a"] * 24), "numeric"),
])
def test_lengths_refusals(bad, message):
    x, y, _ = _tracks()
    with pytest.raises(Exception, match=message):
        _gru().fit(x, y, epochs=1, lengths=bad)


def test_lengths_need_a_recurrent_layer_and_no_validation_data():
    x, y, lengths = _tracks()
    flat = (mojolearn.Sequential(input_shape=(6, 2)).flatten().linear(2)
            .compile(loss="cross_entropy"))
    with pytest.raises(Exception, match="recurrent layers"):
        flat.fit(x, y, epochs=1, lengths=lengths)
    with pytest.raises(Exception, match="validation_data with lengths"):
        _gru().fit(x, y, epochs=1, lengths=lengths, validation_data=(x, y))


def test_padded_values_do_not_change_the_fit():
    x, y, lengths = _tracks()
    noisy = x.copy()
    for i, n in enumerate(lengths):  # glue: test fixture, the padded steps of each row
        noisy[i, n:] = np.inf
    a = _gru().fit(x * (np.arange(6)[None, :, None] < lengths[:, None, None]), y, epochs=2, batch_size=8,
                   lengths=lengths)
    b = _gru().fit(noisy, y, epochs=2, batch_size=8, lengths=lengths)
    np.testing.assert_array_equal(np.asarray(a.params_), np.asarray(b.params_))
    np.testing.assert_array_equal(a.predict_proba(x, lengths=lengths), b.predict_proba(noisy, lengths=lengths))
