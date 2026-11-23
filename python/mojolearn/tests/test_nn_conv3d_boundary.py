# SPDX-License-Identifier: Apache-2.0
"""Public boundary of Sequential's 3-D layers (lane nn-conv3d, 2026-11-23): conv3d, max_pool3d, avg_pool3d and
adaptive_avg_pool3d. What the calls accept and write, the refusals, and the parameters' round trips (state_dict,
load_state_dict, checkpoint). The numerics (forward, every gradient, the 3-D CNN fits, host versus device words)
are the native check checks/native_tests/nn_conv3d.mojo."""
import numpy as np
import pytest

import mojolearn


def _pack(d, h, w):
    return d | (h << 10) | (w << 20)


def _kinds(model):
    return [layer[0] for layer in model.layers]


def _volumes(n=12, shape=(1, 4, 4, 4), seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n,) + shape).astype(np.float32)
    y = rng.standard_normal((n, 1)).astype(np.float32)
    return x, y


def _model():
    return (mojolearn.Sequential(input_shape=(1, 4, 4, 4)).conv3d(3, 3, padding=1).relu().max_pool3d(2)
            .conv3d(4, (1, 2, 2), groups=1, bias=False).avg_pool3d((2, 1, 1)).adaptive_avg_pool3d(1)
            .flatten().linear(1).compile(loss="mse", optimizer="sgd", lr=0.01))


def test_layers_describe_their_arguments():
    m = (mojolearn.Sequential(input_shape=(2, 6, 6, 6)).conv3d(4, (3, 1, 3), stride=(1, 2, 1), padding=(1, 0, 1),
                                                                dilation=(2, 1, 1), groups=2, bias=False)
         .max_pool3d(2, padding=1, ceil_mode=True).avg_pool3d(3, stride=1, padding=1, count_include_pad=False,
                                                              divisor_override=7)
         .adaptive_avg_pool3d((1, 2, 1)).flatten().linear(1))
    assert _kinds(m) == ["conv3d", "max_pool3d", "avg_pool3d", "adaptive_avg_pool3d", "flatten", "linear"]
    assert m.layers[0][1] == (4, _pack(3, 1, 3), _pack(1, 2, 1), _pack(1, 0, 1), 0 + 2 * 2, _pack(2, 1, 1))
    assert m.layers[1][1] == (_pack(2, 2, 2), _pack(2, 2, 2), _pack(1, 1, 1), _pack(1, 1, 1), 1, 0)
    assert m.layers[2][1] == (_pack(3, 3, 3), _pack(1, 1, 1), _pack(1, 1, 1), 0, 0, 7)
    assert m.layers[3][1] == (_pack(1, 2, 1), 0, 0, 0, 0, 0)


def test_valid_padding_is_zero():
    m = mojolearn.Sequential(input_shape=(1, 4, 4, 4)).conv3d(2, 3, padding="valid")
    assert m.layers[0][1][3] == 0


@pytest.mark.parametrize("call,message", [
    (lambda m: m.conv3d(2, (3, 3)), "tuple of three"),
    (lambda m: m.conv3d(2, 3, padding="same"), "valid"),
    (lambda m: m.conv3d(0, 3), "out_channels"),
    (lambda m: m.conv3d(2, 3, stride=0), "stride"),
    (lambda m: m.conv3d(2, 2048), "1023"),
    (lambda m: m.max_pool3d(2, return_indices=True), "return_indices"),
    (lambda m: m.max_pool3d(2, padding=2), "half of kernel_size"),
    (lambda m: m.avg_pool3d((3, 3, 2), padding=(1, 1, 2)), "half of kernel_size"),
    (lambda m: m.avg_pool3d(2, divisor_override=0), "divisor_override"),
    (lambda m: m.adaptive_avg_pool3d(0), "output_size"),
])
def test_refusals_at_add(call, message):
    with pytest.raises(Exception, match=message):
        call(mojolearn.Sequential(input_shape=(1, 4, 4, 4)))


@pytest.mark.parametrize("shape,build,message", [
    ((2, 4, 4), lambda m: m.conv3d(2, 3), "depth"),                 # a (C, H, W) sample
    ((2, 4, 4, 4), lambda m: m.conv3d(3, 1, groups=2), "groups"),     # groups must divide out_channels
    ((2, 4, 4, 4), lambda m: m.conv3d(3, 5), "larger"),               # the kernel does not fit
])
def test_refusals_at_fit(shape, build, message):
    x = np.zeros((4,) + shape, np.float32)
    m = build(mojolearn.Sequential(input_shape=shape)).flatten().linear(1).compile()
    with pytest.raises(Exception, match=message):
        m.fit(x, np.zeros((4, 1), np.float32), epochs=1)


def test_adaptive_pool_refuses_uneven_windows():
    x = np.zeros((4, 1, 5, 4, 4), np.float32)
    m = mojolearn.Sequential(input_shape=(1, 5, 4, 4)).adaptive_avg_pool3d((2, 2, 2)).flatten().linear(1).compile()
    with pytest.raises(Exception, match="divide"):
        m.fit(x, np.zeros((4, 1), np.float32), epochs=1)


def test_fit_predict_state_dict_and_checkpoint_round_trip(tmp_path):
    x, y = _volumes()
    m = _model().fit(x, y, epochs=2, batch_size=4, shuffle=False, seed=0)
    pred = np.asarray(m.predict(x))
    assert pred.shape == (12, 1) and np.isfinite(pred).all()
    assert np.isfinite(np.asarray(m.params_)).all()
    state = m.state_dict()
    assert np.asarray(state["0.weight"]).shape == (3, 1, 3, 3, 3)
    assert np.asarray(state["0.bias"]).shape == (3,)
    assert np.asarray(state["3.weight"]).shape == (4, 3, 1, 2, 2)
    assert "3.bias" not in state
    fresh = _model().fit(x, y, epochs=1, batch_size=12, shuffle=False, seed=1)
    # load the trained weights into a fresh model: the same predictions, bit for bit
    fresh.load_state_dict(state)
    np.testing.assert_array_equal(np.asarray(fresh.predict(x)), pred)
    path = str(tmp_path / "conv3d.ckpt")
    m.save_checkpoint(path)
    loaded = mojolearn.Sequential.from_checkpoint(path)
    assert _kinds(loaded) == _kinds(m)
    np.testing.assert_array_equal(np.asarray(loaded.predict(x)), pred)
    np.testing.assert_array_equal(np.asarray(loaded.params_), np.asarray(m.params_))


def test_initial_weights_follow_pytorch_bounds():
    x, y = _volumes()
    m = (mojolearn.Sequential(input_shape=(1, 4, 4, 4)).conv3d(8, 3, padding=1).flatten().linear(1)
         .compile(loss="mse", optimizer="sgd", lr=1e-9))
    m.fit(x, y, epochs=1, batch_size=12, shuffle=False, seed=3)
    w = np.asarray(m.state_dict()["0.weight"])
    b = np.asarray(m.state_dict()["0.bias"])
    bound = 1.0 / np.sqrt(27.0)
    assert np.abs(w).max() <= bound * (1 + 1e-6) and np.abs(b).max() <= bound * (1 + 1e-6)
    assert np.abs(w).max() > 0.5 * bound
