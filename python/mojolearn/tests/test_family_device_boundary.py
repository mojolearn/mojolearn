# SPDX-License-Identifier: Apache-2.0
"""`device=None` (default), `device="gpu"` and `device="cpu"` on the estimators of lane dw-classical, at the Python
boundary. Same rule and same words as Ridge and LinearRegression (test_linear_device_boundary.py), one table of
classes.

The device is a constructor parameter. None is this installation's device: the GPU where the installation has a GPU
set, else the CPU. `get_params` returns what was passed (None stays None). An explicit `device="gpu"` on a CPU-only
installation is refused in words that name `device="cpu"`; `device="cpu"` always runs. It is checked when it is given
and read once, by the estimator's first operation, into its native route owner; `get_params`, `set_params`, clone and
pickle carry it. On a GPU installation both entry tables exist in one process and `device="cpu"` runs the CPU
binaries (`vendor_used() == "cpu"`); in `identical` mode it returns the bits the GPU returns. These tests run on a
GPU installation and on a CPU-only one; what needs two devices is skipped by name on the CPU-only one.
"""
import pickle
import sys

import numpy as np
import pytest

import mojolearn
from mojolearn import QNRegressor
from mojolearn.kernel_methods import KernelRidge
from mojolearn.svm import SVC, SVR, LinearSVC, LinearSVR
from mojolearn import (
    BayesianRidge, ElasticNetCV, HuberRegressor, IsotonicRegression, LassoCV, LogisticRegressionCV, Perceptron,
    PoissonRegressor, QNRegressor, QuantileRegressor, RidgeCV, RidgeClassifier, SGDClassifier, SGDRegressor,
)

TWO_DEVICES = mojolearn.vendor() != "cpu"
two_devices = pytest.mark.skipif(not TWO_DEVICES, reason="a CPU-only installation has one device")


def regression(rows=200, cols=6, seed=3):
    rng = np.random.default_rng(seed)
    x = rng.random((rows, cols), dtype=np.float32)
    y = (x @ np.arange(1, cols + 1, dtype=np.float32) + np.float32(0.5)).astype(np.float32)
    return x, y


def classification(rows=200, cols=6, seed=5):
    rng = np.random.default_rng(seed)
    x = rng.random((rows, cols), dtype=np.float32)
    y = (x @ np.arange(1, cols + 1, dtype=np.float32) > np.float32(10.5)).astype(np.int64)
    return x, y
def classification(rows=200, cols=6, seed=3):
    x, y = regression(rows, cols, seed)
    return x, (y > np.median(y)).astype(np.int32)


def monotone(rows=200, cols=1, seed=3):
    # one feature whatever `cols` says (IsotonicRegression): a noisy increasing target
    rng = np.random.default_rng(seed)
    x = np.sort(rng.random(rows, dtype=np.float32))
    y = (x + np.float32(0.05) * rng.random(rows, dtype=np.float32)).astype(np.float32)
    return x, y


# RidgeCV's split path: explicit (train, test) index pairs over the 200 rows of `regression` (two halves, and a
# third split whose sides overlap neither; no splitter object, no callback).
RIDGE_SPLITS = [
    (np.arange(100, 200, dtype=np.int64), np.arange(0, 100, dtype=np.int64)),
    (np.arange(0, 100, dtype=np.int64), np.arange(100, 200, dtype=np.int64)),
    (np.arange(0, 150, dtype=np.int64), np.arange(150, 200, dtype=np.int64)),
]

# (id, class, constructor keywords, data, the fitted arrays compared bit for bit, the methods compared bit for bit)
CASES = [
    ("qn-squared", QNRegressor, dict(loss="squared_error", l2_strength=0.5, max_iter=60), regression,
     ("coef_", "_w"), ("predict",)),
    ("qn-absolute", QNRegressor, dict(loss="absolute_error", l2_strength=0.5, max_iter=60), regression,
     ("coef_", "_w"), ("predict",)),
    ("qn-l1", QNRegressor, dict(l1_strength=0.01, max_iter=60, fit_intercept=False), regression,
     ("coef_", "_w"), ("predict",)),
    ("linear-svr", LinearSVR, dict(max_iter=60), regression, ("coef_", "_w"), ("predict",)),
    ("linear-svc", LinearSVC, dict(max_iter=60), classification, ("coef_", "intercept_"), ("predict", "decision_function")),
    ("svr-rbf", SVR, dict(kernel="rbf", C=1.0), regression, ("dual_coef_", "support_vectors_"), ("predict",)),
    ("svc-rbf", SVC, dict(kernel="rbf", C=1.0), classification, ("dual_coef_", "support_vectors_"),
     ("predict", "decision_function")),
    ("kernel-ridge-rbf", KernelRidge, dict(kernel="rbf", alpha=1.0), regression, ("dual_coef_",), ("predict",)),
    # lane dw-xlinear: the linear expansion estimators (one row per group of the x_linear door: SGD, the GLM and
    # Huber and Bayes and quantile fits, the cross-validated ridge, the classifiers with class preparation, the
    # cross-validated logistic fit)
    ("xl-sgd-regressor", SGDRegressor, dict(max_iter=20, random_state=0), regression,
     ("coef_", "intercept_"), ("predict",)),
    ("xl-huber", HuberRegressor, dict(max_iter=50), regression, ("coef_", "intercept_"), ("predict",)),
    ("xl-poisson", PoissonRegressor, dict(max_iter=50), regression, ("coef_", "intercept_"), ("predict",)),
    ("xl-bayesian-ridge", BayesianRidge, dict(max_iter=50), regression, ("coef_", "intercept_"), ("predict",)),
    ("xl-quantile", QuantileRegressor, dict(), regression, ("coef_", "intercept_"), ("predict",)),
    ("xl-ridge-cv", RidgeCV, dict(), regression, ("coef_", "intercept_"), ("predict",)),
    ("xl-sgd-classifier", SGDClassifier, dict(max_iter=20, random_state=0), classification,
     ("coef_", "intercept_"), ("predict", "decision_function")),
    ("xl-ridge-classifier", RidgeClassifier, dict(), classification,
     ("coef_", "intercept_"), ("predict", "decision_function")),
    ("xl-logistic-cv", LogisticRegressionCV, dict(Cs=3, cv=3, max_iter=50), classification,
     ("coef_", "intercept_"), ("predict", "decision_function")),
    # lane dw-xlinear, the door split's last group: every path of the cross-validated fits (RidgeCV's k-fold and
    # split paths with their typed fold R^2, the coordinate-descent paths) and the class-weight step that the fit's
    # RUN makes (SGD: float parameters; RidgeClassifier and LogisticRegressionCV: row weights), and the isotonic fit
    ("xl-ridge-cv-kfold", RidgeCV, dict(alphas=(0.1, 1.0, 10.0), cv=5), regression,
     ("coef_", "intercept_"), ("predict",)),
    ("xl-ridge-cv-splits", RidgeCV, dict(alphas=(0.1, 1.0, 10.0), cv=RIDGE_SPLITS), regression,
     ("coef_", "intercept_"), ("predict",)),
    ("xl-lasso-cv", LassoCV, dict(cv=3, max_iter=200), regression, ("coef_", "intercept_"), ("predict",)),
    ("xl-elastic-net-cv", ElasticNetCV, dict(cv=3, l1_ratio=[0.5, 0.9], max_iter=200), regression,
     ("coef_", "intercept_"), ("predict",)),
    ("xl-sgd-classifier-balanced", SGDClassifier, dict(max_iter=20, random_state=0, class_weight="balanced"),
     classification, ("coef_", "intercept_"), ("predict", "decision_function")),
    ("xl-perceptron-class-weight", Perceptron, dict(max_iter=20, random_state=0, class_weight={0: 1.0, 1: 2.0}),
     classification, ("coef_", "intercept_"), ("predict", "decision_function")),
    ("xl-ridge-classifier-balanced", RidgeClassifier, dict(class_weight="balanced"), classification,
     ("coef_", "intercept_"), ("predict", "decision_function")),
    ("xl-logistic-cv-balanced", LogisticRegressionCV, dict(Cs=3, cv=3, max_iter=50, class_weight="balanced"),
     classification, ("coef_", "intercept_"), ("predict", "decision_function")),
    ("xl-isotonic", IsotonicRegression, dict(), monotone, (), ("predict",)),
    ("svgp", mojolearn.SVGP, dict(n_inducing=8, lengthscale=0.5, noise_variance=0.1), regression,
     ("Z_", "q_mu_", "q_sqrt_"), ("predict",)),
]
IDS = [case[0] for case in CASES]
case = pytest.mark.parametrize("spec", CASES, ids=IDS)


def build(spec, **extra):
    return spec[1](**dict(spec[2], **extra))


@case
def test_device_is_a_constructor_parameter_with_default_none(spec):
    assert build(spec).get_params()["device"] is None
    assert build(spec, device=None).get_params()["device"] is None
    assert build(spec, device="cpu").get_params()["device"] == "cpu"
    assert "device" in spec[1]._get_param_names()


@case
@pytest.mark.parametrize("bad", ["tpu", "GPU", " cpu", "", "None", 0, False, b"cpu"])
def test_unknown_device_is_refused_in_words_at_construction(spec, bad):
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None, got"):
        build(spec, device=bad)


@case
def test_default_device_is_this_installations_device(spec):
    x, y = spec[3]()
    model = build(spec).fit(x, y)
    assert model.vendor_used() == mojolearn.vendor()          # the GPU set where there is one, else the CPU
    assert model.get_params()["device"] is None               # what was passed, not what it resolved to
    assert np.asarray(model.predict(x)).shape[0] == len(x)


@case
def test_vendor_used_answers_before_the_first_fit(spec):
    assert build(spec).vendor_used() == mojolearn.vendor()
    assert build(spec, device="cpu").vendor_used() == "cpu"
    assert build(spec).numeric_mode_used() in ("identical", "fast")


@case
def test_explicit_gpu_runs_on_a_gpu_installation_and_is_refused_in_words_on_a_cpu_only_one(spec):
    x, y = spec[3]()
    if TWO_DEVICES:
        model = build(spec, device="gpu").fit(x, y)
        assert model.vendor_used() == mojolearn.vendor() and model.get_params()["device"] == "gpu"
        return
    with pytest.raises(ValueError, match="device='gpu' needs a GPU build of mojolearn.*Use device='cpu'"):
        build(spec, device="gpu")
    model = build(spec).fit(x, y)
    with pytest.raises(ValueError, match="Use device='cpu'"):
        model.set_params(device="gpu")
    # an instance that carries 'gpu' without its constructor's check is refused by its first operation
    carried = build(spec)
    carried.device = "gpu"
    with pytest.raises(ValueError, match="Use device='cpu'"):
        carried.fit(x, y)


@case
def test_set_params_refuses_an_unknown_device(spec):
    model = build(spec)
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None, got 'tpu'"):
        model.set_params(device="tpu")


@case
def test_cpu_device_runs_the_cpu_binary(spec):
    x, y = spec[3]()
    model = build(spec, device="cpu").fit(x, y)
    assert model.vendor_used() == "cpu"
    assert model.numeric_mode_used() == "identical" or not TWO_DEVICES
    assert np.asarray(model.predict(x)).shape[0] == len(x)
    assert isinstance(model.score(x, y), float)


@two_devices
@case
def test_default_device_runs_the_gpu_binary_beside_a_cpu_estimator_in_one_process(spec):
    x, y = spec[3]()
    cpu = build(spec, device="cpu").fit(x, y)
    gpu = build(spec).fit(x, y)
    assert gpu.vendor_used() == mojolearn.vendor() != "cpu"
    assert cpu.vendor_used() == "cpu"
    # each keeps its own route across interleaved calls
    for _ in range(2):
        assert np.asarray(gpu.predict(x)).shape == np.asarray(cpu.predict(x)).shape
        assert gpu.vendor_used() != cpu.vendor_used()


@two_devices
@case
@pytest.mark.parametrize("weighted", [False, True])
def test_identical_mode_cpu_device_returns_the_gpu_bits(spec, weighted):
    if mojolearn.numeric_mode() != "identical":
        pytest.skip("bits are promised in identical mode only")
    x, y = spec[3](rows=400, cols=8)
    weight = np.linspace(0.5, 2.0, len(x), dtype=np.float32) if weighted else None
    gpu = build(spec, numeric_mode="identical").fit(x, y, sample_weight=weight)
    cpu = build(spec, numeric_mode="identical", device="cpu").fit(x, y, sample_weight=weight)
    for name in spec[4]:
        assert np.asarray(getattr(gpu, name)).tobytes() == np.asarray(getattr(cpu, name)).tobytes(), name
    for method in spec[5]:
        assert np.asarray(getattr(gpu, method)(x)).tobytes() == np.asarray(getattr(cpu, method)(x)).tobytes(), method


@two_devices
@case
def test_cpu_device_never_runs_a_fast_binary(spec):
    x, y = spec[3]()
    model = build(spec, device="cpu", numeric_mode="fast")
    try:
        model.fit(x, y)
    except ValueError as refusal:
        # where `fast` names its own binaries (Apple): refused in words
        assert "device='cpu' runs in identical mode only" in str(refusal)
    except ImportError:
        # where no fast binaries are installed, the mode's own refusal stands
        pass
    else:
        # where the identical set answers `fast` for this family
        assert model.numeric_mode_used() == "identical" and model.vendor_used() == "cpu"


@case
def test_set_params_moves_the_estimator_to_the_other_device(spec):
    x, y = spec[3]()
    model = build(spec).fit(x, y)
    assert model.set_params(device="cpu") is model
    assert model.get_params()["device"] == "cpu"
    model.fit(x, y)
    assert model.vendor_used() == "cpu"
    model.set_params(device=None).fit(x, y)
    assert model.get_params()["device"] is None
    assert model.vendor_used() == mojolearn.vendor()


@case
@pytest.mark.parametrize("device", [None, "cpu", "gpu"])
def test_clone_and_pickle_carry_the_device(spec, device):
    if device == "gpu" and not TWO_DEVICES:
        pytest.skip("an explicit device='gpu' is refused on a CPU-only installation")
    x, y = spec[3]()
    model = build(spec, device=device).fit(x, y)
    expected = np.asarray(model.predict(x)).copy()
    used = model.vendor_used()

    rebuilt = spec[1](**model.get_params())
    assert rebuilt.get_params() == model.get_params()
    assert rebuilt.get_params()["device"] == device          # None stays None
    assert rebuilt.fit(x, y).vendor_used() == used

    try:
        from sklearn.base import clone
    except ImportError:
        clone = None
    if clone is not None:
        cloned = clone(model)
        assert cloned.get_params()["device"] == device
        assert cloned.fit(x, y).vendor_used() == used

    restored = pickle.loads(pickle.dumps(model))
    assert restored.get_params()["device"] == device
    # the unpickled estimator was built without its constructor: its first
    # operation reads the device from its restored parameters
    np.testing.assert_array_equal(np.asarray(restored.predict(x)), expected)
    assert restored.vendor_used() == used


@case
def test_a_changed_numeric_mode_resolves_the_route_again(spec):
    x, y = spec[3]()
    model = build(spec, device="cpu").fit(x, y)
    assert model.vendor_used() == "cpu"
    model.numeric_mode = "identical"
    assert model.numeric_mode_used() == "identical"
    assert model.vendor_used() == "cpu"
    np.asarray(model.predict(x))


@case
def test_check_runs_both_devices_in_this_process(spec, capsys):
    x, y = spec[3]()
    model = build(spec).fit(x, y)
    result = model.check(x)
    capsys.readouterr()
    if not TWO_DEVICES or mojolearn.numeric_mode() != "identical":
        assert not result["ran"]
        return
    assert result["ran"] and result["transfer"] == "device"
    gpu, cpu = result["sides"]["gpu"], result["sides"]["cpu"]
    assert gpu["route"] == dict(device=None, vendor_used=mojolearn.vendor())
    assert cpu["route"] == dict(device="cpu", vendor_used="cpu")
    assert result["verdict"]["match"]
    assert gpu["outputs"][0]["sha256"] == cpu["outputs"][0]["sha256"]
    # the given estimator is untouched
    assert model.get_params()["device"] is None and model.vendor_used() == mojolearn.vendor()


def _python_events(call):
    """Python-level events while `call` runs: every Python function call and
    every C function called from a Python frame."""
    counts = {"call": 0, "c_call": 0}

    def profile(frame, event, arg):
        if event in counts:
            counts[event] += 1

    sys.setprofile(profile)
    try:
        call()
    finally:
        sys.setprofile(None)
    return counts


@case
@pytest.mark.parametrize("device", [None, "cpu"])
def test_python_operations_per_call_do_not_grow_with_rows(spec, device):
    counts = {}
    for name, rows in (("small", 64), ("large", 8192)):
        x, y = spec[3](rows=rows)
        model = build(spec, device=device).fit(x, y)
        model.predict(x)                                     # the route is resolved; the next call is a hit
        counts[name] = (_python_events(lambda: model.fit(x, y)), _python_events(lambda: model.predict(x)))
    assert counts["small"] == counts["large"]
