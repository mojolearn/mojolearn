# SPDX-License-Identifier: Apache-2.0
"""Public boundary of the activation layers and the identity layer of Sequential (lane nn-activations,
2026-11-23): what each call accepts, writes into the layer description and refuses, and that a fitted model's
parameters (PReLU's slopes) survive state_dict, load_state_dict and a checkpoint. The numerics (forward against a
float64 oracle, every gradient against finite differences, host versus device words, fits) are the native check
checks/native_tests/nn_activations.mojo."""
import numpy as np
import pytest

import mojolearn


def _kinds(model):
    return [layer[0] for layer in model.layers]


def _data(n=32, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 3, 4)).astype(np.float32)
    y = rng.standard_normal((n, 1)).astype(np.float32)
    return x, y


def _model():
    return (mojolearn.Sequential(input_shape=(3, 4)).linear(4).leaky_relu(0.2).elu(0.5).prelu(3, init=0.1)
            .mish().identity().hardswish().hardsigmoid().prelu().flatten().linear(1)
            .compile(loss="mse", optimizer="adam", lr=0.01))


def test_layers_describe_their_arguments():
    m = _model()
    assert _kinds(m) == ["linear", "leaky_relu", "elu", "prelu", "mish", "identity", "hardswish", "hardsigmoid",
                         "prelu", "flatten", "linear"]
    assert m.layers[1][2] == pytest.approx(0.2)
    assert m.layers[2][2] == pytest.approx(0.5)
    args = m.layers[3][1]
    assert args[0] == 3
    assert np.int32(args[1]).view(np.float32) == np.float32(0.1)   # the init's float32 bits
    assert m.layers[8][1][0] == 1
    assert np.int32(m.layers[8][1][1]).view(np.float32) == np.float32(0.25)


def test_pytorch_defaults():
    m = mojolearn.Sequential().linear(2).leaky_relu().elu().prelu()
    assert m.layers[1][2] == pytest.approx(0.01)
    assert m.layers[2][2] == pytest.approx(1.0)
    assert m.layers[3][1][0] == 1


@pytest.mark.parametrize("call,message", [
    (lambda m: m.leaky_relu(float("nan")), "negative_slope must be a finite number"),
    (lambda m: m.elu(float("inf")), "alpha must be a finite number"),
    (lambda m: m.prelu(0), "num_parameters must be an integer"),
    (lambda m: m.prelu(2.5), "num_parameters must be an integer"),
    (lambda m: m.prelu(init=-1.0), "init must be a finite number greater than -1"),
    (lambda m: m.add("identity", 1), "takes no arguments"),
    (lambda m: m.add("mish", alpha=1), "takes no arguments"),
    (lambda m: m.leaky_relu(0.1, 0.2), "too many positional arguments"),
])
def test_refusals_at_add(call, message):
    with pytest.raises(Exception, match=message):
        call(mojolearn.Sequential(input_shape=(3, 4)).linear(4))


def test_prelu_channels_must_match_the_first_axis():
    x, y = _data()
    m = mojolearn.Sequential(input_shape=(3, 4)).linear(4).prelu(4).flatten().linear(1).compile()
    with pytest.raises(Exception, match="num_parameters must be 1 or the input's channels"):
        m.fit(x, y, epochs=1)


def test_fit_trains_the_prelu_slopes_and_state_dict_names_them():
    x, y = _data()
    m = _model().fit(x, y, epochs=3, batch_size=8, seed=0)
    state = m.state_dict()
    assert np.asarray(state["3.weight"]).shape == (3,)
    assert np.asarray(state["8.weight"]).shape == (1,)
    assert not np.allclose(np.asarray(state["3.weight"]), 0.1)        # trained away from the init
    assert np.all(np.isfinite(m.params_)) and np.all(np.isfinite(m.predict(x)))
    fresh = _model().load_state_dict(state)
    np.testing.assert_array_equal(fresh.predict(x), m.predict(x))


def test_checkpoint_round_trip(tmp_path):
    x, y = _data()
    m = _model().fit(x, y, epochs=2, batch_size=8, seed=0)
    path = str(tmp_path / "act.ckpt")
    m.save_checkpoint(path)
    again = mojolearn.Sequential.from_checkpoint(path)
    assert again.layers == m.layers
    np.testing.assert_array_equal(np.asarray(again.params_), np.asarray(m.params_))
    np.testing.assert_array_equal(again.predict(x), m.predict(x))
    assert again.digest_ == m.digest_


def test_identity_changes_nothing():
    x, y = _data()
    with_id = (mojolearn.Sequential(input_shape=(3, 4)).linear(4).relu().identity().flatten().linear(1)
               .compile(loss="mse", optimizer="sgd", lr=0.01))
    without = (mojolearn.Sequential(input_shape=(3, 4)).linear(4).relu().flatten().linear(1)
               .compile(loss="mse", optimizer="sgd", lr=0.01))
    with_id.fit(x, y, epochs=2, batch_size=8, shuffle=False, seed=0)
    without.fit(x, y, epochs=2, batch_size=8, shuffle=False, seed=0)
    np.testing.assert_array_equal(np.asarray(with_id.params_), np.asarray(without.params_))
    np.testing.assert_array_equal(with_id.predict(x), without.predict(x))
