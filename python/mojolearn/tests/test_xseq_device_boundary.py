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


def words(value):
    # glue: a fitted attribute as bytes: one array, or a list of arrays (coefs_, intercepts_)
    if isinstance(value, (list, tuple)):
        return [np.asarray(item).tobytes() for item in value]
    return np.asarray(value).tobytes()


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
def test_cpu_and_gpu_estimators_keep_their_routes_and_their_bits_in_one_process(spec):
    if mojolearn.numeric_mode() != "identical":
        pytest.skip("bits are promised in identical mode only")
    x, y = spec[3]()
    cpu = build(spec, numeric_mode="identical", device="cpu").fit(x, y)
    gpu = build(spec, numeric_mode="identical").fit(x, y)
    assert gpu.vendor_used() == mojolearn.vendor() != "cpu" and cpu.vendor_used() == "cpu"
    for name in spec[4]:
        assert words(getattr(gpu, name)) == words(getattr(cpu, name)), name
    for _ in range(2):                                         # interleaved calls: each keeps its own route
        for method in spec[5]:
            assert np.asarray(getattr(gpu, method)(x)).tobytes() == np.asarray(getattr(cpu, method)(x)).tobytes(), method


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
