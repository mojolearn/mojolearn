# SPDX-License-Identifier: Apache-2.0
"""`device=None` (default), `device="gpu"` and `device="cpu"` on the multi-layer perceptrons and the recurrent
estimators, at the Python boundary. Same rule and same words as Ridge and LinearRegression
(test_linear_device_boundary.py).

The device is a constructor parameter, checked where it is given and read once, by the estimator's first
operation, into its one native route owner; no operation reads it again. On a GPU installation both sets of
compiled entries exist in one process and `device="cpu"` runs the CPU binaries; in `identical` mode it returns
the bits the GPU returns. What needs two devices is skipped by name on a CPU-only installation.
"""
import pickle

import numpy as np
import pytest

import mojolearn
from mojolearn import GRUClassifier, LSTMRegressor, MLPClassifier, MLPRegressor

TWO_DEVICES = mojolearn.vendor() != "cpu"
two_devices = pytest.mark.skipif(not TWO_DEVICES, reason="a CPU-only installation has one device")


def table(rows=96, cols=6, seed=3):
    rng = np.random.default_rng(seed)
    x = rng.random((rows, cols), dtype=np.float32)
    return x, (x @ np.arange(1, cols + 1, dtype=np.float32)).astype(np.float32)


def table_labels(rows=96, cols=6, seed=3):
    x, y = table(rows, cols, seed)
    return x, (y > np.median(y)).astype(np.int64)


def table_three(rows=96, cols=6, seed=3):
    x, y = table(rows, cols, seed)
    return x, np.digitize(y, np.quantile(y, [1 / 3, 2 / 3])).astype(np.int64)


def series(rows=48, steps=5, cols=3, seed=5):
    rng = np.random.default_rng(seed)
    x = rng.random((rows, steps, cols), dtype=np.float32)
    return x, x.sum(axis=(1, 2)).astype(np.float32)


def series_labels(rows=48, steps=5, cols=3, seed=5):
    x, y = series(rows, steps, cols, seed)
    return x, (y > np.median(y)).astype(np.int64)


MLP = dict(hidden_layer_sizes=(8,), max_iter=6, random_state=7)
RNN = dict(hidden_size=6, max_epochs=3, batch_size=16, random_state=7)

# (id, class, constructor keywords, data, the fitted attributes compared bit for bit, the methods compared bit for bit,
#  whether `score` runs on a CPU route beside a GPU set)
CASES = [
    ("mlp-regressor", MLPRegressor, MLP, table, ("coefs_", "intercepts_", "loss_curve_"), ("predict",), True),
    ("mlp-binary", MLPClassifier, MLP, table_labels, ("coefs_", "intercepts_", "loss_curve_"),
     ("predict", "predict_proba"), False),
    ("mlp-three-classes", MLPClassifier, MLP, table_three, ("coefs_", "intercepts_", "loss_curve_"),
     ("predict", "predict_proba"), False),
    ("lstm-regressor", LSTMRegressor, RNN, series, ("params_", "loss_curve_"), ("predict",), True),
    ("gru-classifier", GRUClassifier, RNN, series_labels, ("params_", "loss_curve_"),
     ("predict", "predict_proba"), False),
]
case = pytest.mark.parametrize("spec", CASES, ids=[spec[0] for spec in CASES])


def build(spec, **extra):
    return spec[1](**dict(spec[2], **extra))


@case
def test_device_is_a_constructor_parameter_with_default_none(spec):
    assert build(spec).get_params()["device"] is None
    assert build(spec, device="cpu" if not TWO_DEVICES else "gpu").get_params()["device"] is not None


@case
@pytest.mark.parametrize("value", ["GPU", "cuda", "", 0, True, b"cpu"])
def test_unknown_device_is_refused_in_words_at_construction_and_at_set_params(spec, value):
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None"):
        build(spec, device=value)
    model = build(spec)
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None"):
        model.set_params(device=value)
    assert model.get_params()["device"] is None                # a refused set_params leaves the estimator whole


@case
def test_default_device_is_this_installations_device(spec):
    x, y = spec[3]()
    model = build(spec).fit(x, y)
    assert model.vendor_used() == mojolearn.vendor()
    assert model.get_params()["device"] is None                # what was passed, not what it resolved to
    assert np.asarray(model.predict(x)).shape[0] == len(x)


@case
def test_explicit_gpu_runs_on_a_gpu_installation_and_is_refused_in_words_on_a_cpu_only_one(spec):
    x, y = spec[3]()
    if TWO_DEVICES:
        model = build(spec, device="gpu").fit(x, y)
        assert model.vendor_used() == mojolearn.vendor() and model.get_params()["device"] == "gpu"
        return
    with pytest.raises(ValueError, match="device='gpu' needs a GPU build of mojolearn.*Use device='cpu'"):
        build(spec, device="gpu")


@case
def test_cpu_device_runs_the_cpu_binary(spec):
    x, y = spec[3]()
    model = build(spec, device="cpu").fit(x, y)
    assert model.vendor_used() == "cpu"
    assert np.asarray(model.predict(x)).shape[0] == len(x)
    if spec[6] or not TWO_DEVICES:
        assert isinstance(model.score(x, y), float)


@two_devices
@case
def test_set_params_moves_an_estimator_to_the_other_device(spec):
    x, y = spec[3]()
    model = build(spec).fit(x, y)
    assert model.vendor_used() != "cpu"
    assert model.set_params(device="cpu") is model
    assert model.fit(x, y).vendor_used() == "cpu"
    assert model.set_params(device=None).fit(x, y).vendor_used() == mojolearn.vendor()


@two_devices
@case
def test_scores_of_the_metrics_family_are_refused_in_words_on_a_cpu_route(spec):
    if spec[6]:
        pytest.skip("this score runs on the route's own metrics entry")
    x, y = spec[3]()
    model = build(spec, device="cpu").fit(x, y)
    with pytest.raises(ValueError, match=r"device='cpu' is not available for \w+\.score"):
        model.score(x, y)
    assert isinstance(build(spec).fit(x, y).score(x, y), float)


@two_devices
def test_multi_output_r2_is_refused_in_words_on_a_cpu_route():
    x, y = table()
    both = np.stack([y, -y], axis=1)
    model = MLPRegressor(device="cpu", **MLP).fit(x, both)
    assert model.predict(x).shape == both.shape
    with pytest.raises(ValueError, match=r"device='cpu' is not available for MLPRegressor\.score"):
        model.score(x, both)


@case
def test_pickle_and_clone_carry_the_device(spec):
    x, y = spec[3]()
    device = "cpu" if TWO_DEVICES else None
    model = build(spec, device=device).fit(x, y)
    again = pickle.loads(pickle.dumps(model))
    assert again.get_params()["device"] == device
    assert np.asarray(again.predict(x)).tobytes() == np.asarray(model.predict(x)).tobytes()
    assert again.vendor_used() == model.vendor_used()
    fresh = type(model)(**model.get_params())
    assert fresh.get_params()["device"] == device


@two_devices
@case
def test_check_compares_the_same_estimator_on_both_devices_in_this_process(spec, capsys):
    if mojolearn.numeric_mode() != "identical":
        pytest.skip("bits are promised in identical mode only")
    x, y = spec[3]()
    model = build(spec, numeric_mode="identical").fit(x, y)
    result = model.check(x)
    capsys.readouterr()
    assert result["ran"] and result["transfer"] == "device"   # no save, no host_model, no child process
    assert result["sides"]["gpu"]["route"]["vendor_used"] != "cpu"
    assert result["sides"]["cpu"]["route"]["vendor_used"] == "cpu"
    assert result["verdict"]["match"]


@two_devices
@pytest.mark.parametrize("cls", ["CrostonClassic", "CrostonOptimized", "CrostonSBA"])
def test_croston_runs_on_its_device_and_returns_the_same_bits(cls):
    if mojolearn.numeric_mode() != "identical":
        pytest.skip("bits are promised in identical mode only")
    rng = np.random.default_rng(11)
    y = (rng.random((4, 64), dtype=np.float32) * (rng.random((4, 64)) > 0.6)).astype(np.float32)
    kind = getattr(mojolearn, cls)
    cpu = kind(numeric_mode="identical", device="cpu").fit(y)
    gpu = kind(numeric_mode="identical").fit(y)
    assert cpu.get_params()["device"] == "cpu" and gpu.get_params()["device"] is None
    assert np.asarray(cpu.mean_).tobytes() == np.asarray(gpu.mean_).tobytes()
    for _ in range(2):                                         # interleaved calls: each keeps its own route
        assert np.asarray(cpu.predict(3)).tobytes() == np.asarray(gpu.predict(3)).tobytes()
    assert kind(device="cpu").set_params(device=None).get_params()["device"] is None


# moved from test_x_sequence_mlp.py (vdv-pytest-b, 2026-11-24)
import mojolearn as ml


def _smlp_data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 5)).astype(np.float32)
    yr = (X[:, 0] * 2 - X[:, 1]).astype(np.float32)
    y3 = (X[:, 0] > 0).astype(int) + (X[:, 1] > 0.5)
    return X, yr, y3


def test_shapes_and_proba():
    X, yr, y3 = _smlp_data()
    c = ml.MLPClassifier(hidden_layer_sizes=(5,), max_iter=5, random_state=0).fit(X, y3)
    p = c.predict_proba(X)
    assert p.shape == (200, 3)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-5)
    b = ml.MLPClassifier(hidden_layer_sizes=(5,), max_iter=5, random_state=0).fit(X, y3 > 0)
    assert b.predict_proba(X).shape == (200, 2)
    r = ml.MLPRegressor(hidden_layer_sizes=(5,), max_iter=5, random_state=0).fit(X, yr)
    assert r.predict(X).shape == (200,) and r.n_iter_ == 5


# moved from python/mojolearn/tests/test_py_bugs.py
# The RNN's row order is int32: its length is no longer capped at 2^24; the 2^31 - 1 bound is refused in words.
import pytest


def test_rnn_schedule_bound_is_2_31_not_2_24():
    # the epoch orders are built on the executor by rnn_fit; the schedule
    # scalars come from the native fit (bindings/xseq_neural_native_door.mojo),
    # and the 2^31 - 1 bound on epochs * n (not the old float32 2^24 cap)
    # still holds: refused before any weight is trained
    np = pytest.importorskip("numpy")
    from mojolearn._x_sequence_rnn import RNNRegressor
    X = np.zeros((1_000_000, 1, 1), dtype=np.float32)
    y = np.zeros(1_000_000, dtype=np.float32)
    with pytest.raises(ValueError, match="max_epochs \\* n_samples must be below 2\\^31 - 1"):
        RNNRegressor(max_epochs=2200).fit(X, y)
