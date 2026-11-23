# SPDX-License-Identifier: Apache-2.0
"""Public boundary of Sequential.group_norm, instance_norm1d and instance_norm2d (lane nn-norms, 2026-11-23): the
description each call writes, the refusals in words, the state the layers carry (state_dict names, running
statistics, checkpoints, load_state_dict). The numerics (forward and gradients against a float64 oracle, host
versus device words, the fits) are the native check checks/native_tests/nn_norms.mojo; the NumPy forward below is
an independent oracle of the fitted model's predictions."""
import numpy as np
import pytest

import mojolearn


def _kinds(model):
    return [layer[0] for layer in model.layers]


def _group_norm_ref(x, groups, weight, bias, eps):
    n, c = x.shape[:2]
    g = x.reshape(n, groups, -1).astype(np.float64)
    y = (g - g.mean(axis=2, keepdims=True)) / np.sqrt(g.var(axis=2, keepdims=True) + eps)
    y = y.reshape(x.shape)
    shape = (1, c) + (1,) * (x.ndim - 2)
    return y * weight.reshape(shape) + bias.reshape(shape)


def test_norms_describe_their_arguments():
    m = (mojolearn.Sequential(input_shape=(4, 8)).group_norm(2, 4, eps=1e-3, affine=False)
         .instance_norm1d(4, momentum=0.25, affine=True, track_running_stats=True).flatten().linear(2))
    assert _kinds(m) == ["group_norm", "instance_norm", "flatten", "linear"]
    assert m.layers[0][1] == (2, 4, 0, 0, 0, 0)
    assert m.layers[0][2] == pytest.approx(1e-3)
    args = m.layers[1][1]
    assert args[0] == 4 and args[2] == 1 and args[3] == 1 and args[4] == 1
    assert np.int32(args[1]).view(np.float32) == np.float32(0.25)   # the momentum's float32 bits
    m2 = mojolearn.Sequential(input_shape=(2, 3, 3)).instance_norm2d(2).flatten().linear(1)
    assert m2.layers[0][1][2:5] == (2, 0, 0)                        # PyTorch's defaults: no affine, untracked


@pytest.mark.parametrize("call,message", [
    (lambda m: m.group_norm(3, 4), "divisible"),
    (lambda m: m.group_norm(2, 4, eps=0.0), "eps"),
    (lambda m: m.instance_norm1d(4, momentum=1.5), "momentum"),
    (lambda m: m.instance_norm1d(4, momentum=None), "momentum=None"),
    (lambda m: m.instance_norm1d(4, eps=-1.0), "eps"),
])
def test_norm_refusals_at_add(call, message):
    with pytest.raises(Exception, match=message):
        call(mojolearn.Sequential(input_shape=(4, 8)))


@pytest.mark.parametrize("build,shape,message", [
    (lambda m: m.group_norm(2, 6), (4, 8), "num_channels"),
    (lambda m: m.instance_norm1d(3), (4, 8), "num_features"),
    (lambda m: m.instance_norm1d(4), (4, 1), "more than 1 spatial element"),
    (lambda m: m.instance_norm2d(4), (4, 8), r"\(channels, height, width\)"),
])
def test_norm_refusals_at_fit(build, shape, message):
    m = build(mojolearn.Sequential(input_shape=shape)).flatten().linear(1).compile()
    with pytest.raises(Exception, match=message):
        m.fit(np.zeros((4,) + shape, np.float32), np.zeros((4, 1), np.float32), epochs=1)


def test_group_norm_predictions_match_a_numpy_forward():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((16, 4, 6)).astype(np.float32)
    y = rng.standard_normal((16, 1)).astype(np.float32)
    m = (mojolearn.Sequential(input_shape=(4, 6)).group_norm(2, 4).flatten().linear(1)
         .compile(loss="mse", optimizer="adam", lr=0.05))
    m.fit(x, y, epochs=2, batch_size=8, shuffle=False)
    state = m.state_dict()
    assert set(state) == {"0.weight", "0.bias", "2.weight", "2.bias"}
    w, b = np.asarray(state["0.weight"]).ravel(), np.asarray(state["0.bias"]).ravel()
    assert not np.array_equal(w, np.ones(4, np.float32))            # the affine trained
    h = _group_norm_ref(x, 2, w, b, 1e-5).reshape(16, -1)
    lw, lb = np.asarray(state["2.weight"], np.float64), np.asarray(state["2.bias"], np.float64)
    want = h @ (lw.T if lw.shape[0] == 1 else lw) + lb
    np.testing.assert_allclose(np.asarray(m.predict(x), np.float64).reshape(want.shape), want, rtol=1e-4, atol=1e-5)


def test_untracked_instance_norm_without_affine_has_no_tensors():
    m = mojolearn.Sequential(input_shape=(2, 8)).instance_norm1d(2).flatten().linear(1).compile()
    m.fit(np.random.default_rng(2).standard_normal((8, 2, 8)).astype(np.float32), np.zeros((8, 1), np.float32),
          epochs=1)
    assert set(m.state_dict()) == {"2.weight", "2.bias"}


def test_tracked_instance_norm_running_statistics_are_saved_and_not_trained():
    x = np.random.default_rng(0).standard_normal((16, 2, 8)).astype(np.float32)
    y = np.random.default_rng(1).standard_normal((16, 1)).astype(np.float32)
    m = (mojolearn.Sequential(input_shape=(2, 8)).instance_norm1d(2, momentum=0.5, track_running_stats=True)
         .flatten().linear(1).compile(loss="mse", optimizer="adamw", lr=0.1, weight_decay=0.9))
    m.fit(x, y, epochs=1, batch_size=16, shuffle=False)
    state = m.state_dict()
    assert {"0.running_mean", "0.running_var"} <= set(state) and "0.weight" not in state
    mean = x.mean(axis=2).mean(axis=0)
    var = x.var(axis=2, ddof=1).mean(axis=0)
    np.testing.assert_allclose(np.asarray(state["0.running_mean"]).ravel(), 0.5 * mean, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(state["0.running_var"]).ravel(), 0.5 + 0.5 * var, rtol=1e-5, atol=1e-6)


def test_norm_models_round_trip_through_files_and_state_dicts(tmp_path):
    rng = np.random.default_rng(3)
    x = rng.standard_normal((12, 4, 3, 3)).astype(np.float32)
    y = rng.integers(0, 2, 12)
    def build():
        return (mojolearn.Sequential(input_shape=(4, 3, 3)).group_norm(2, 4)
                .instance_norm2d(4, affine=True, track_running_stats=True).flatten().linear(2)
                .compile(loss="cross_entropy", optimizer="adam", lr=0.01))
    m = build()
    m.fit(x, y, epochs=2, batch_size=4, seed=1)
    want = np.asarray(m.predict_proba(x))
    path = str(tmp_path / "norms.ckpt")
    m.save_checkpoint(path)
    back = mojolearn.Sequential.from_checkpoint(path)
    np.testing.assert_array_equal(np.asarray(back.params_), np.asarray(m.params_))
    np.testing.assert_array_equal(np.asarray(back.predict_proba(x)), want)
    assert back.digest_ == m.digest_
    other = build()
    other.fit(x, y, epochs=1, batch_size=4, seed=2)
    other.load_state_dict({k: np.asarray(v) for k, v in m.state_dict().items()})
    np.testing.assert_array_equal(np.asarray(other.predict_proba(x)), want)
