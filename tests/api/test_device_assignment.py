# SPDX-License-Identifier: Apache-2.0
"""Python boundary tests of a direct `model.device = ...` on a live estimator.

`device` is read once into the estimator's route (providers/python/device_route_boundary_door.mojo). A direct
assignment after the first operation must reach the next operation: it runs on the new device, or it refuses a bad
value in words. One case per route owner kind: the linear entry route, LogisticRegression's own hit test, the
decomposition route, the scaler route, the tree binding route and the shared estimator binding route. The fits are
small and independent; pytest-xdist splits the tests across cores when it is installed.
"""
import numpy as np
import pytest

mojolearn = pytest.importorskip("mojolearn")

RNG = np.random.default_rng(0)
X = RNG.random((300, 5), dtype=np.float32)
Y = X @ np.arange(1, 6, dtype=np.float32)
C = (Y > np.median(Y)).astype(np.int32)

#: (class name, parameters, target, the operation that follows the assignment)
CASES = [
    ("Ridge", dict(alpha=1.0), "r", "predict"),
    ("LinearRegression", {}, "r", "predict"),
    ("LogisticRegression", {}, "c", "predict"),
    ("QNRegressor", {}, "r", "predict"),
    ("KMeans", dict(n_clusters=3, random_state=0), None, "predict"),
    ("PCA", dict(n_components=2), None, "transform"),
    ("TruncatedSVD", dict(n_components=2), None, "transform"),
    ("StandardScaler", {}, None, "transform"),
    ("MinMaxScaler", {}, None, "transform"),
    ("RandomForestRegressor", dict(n_estimators=5, random_state=0), "r", "predict"),
    ("GradientBoosting", dict(n_estimators=5), "r", "predict"),
    ("GaussianNB", {}, "c", "predict"),
]

pytestmark = pytest.mark.skipif(mojolearn.numeric_mode() != "identical", reason="device='cpu' runs in identical mode")


def _build(name, params, **more):
    cls = getattr(mojolearn, name, None)
    if cls is None:
        pytest.skip(name + " is not in this installation")
    return cls(**params, **more)


def _fit(est, target):
    return est.fit(X) if target is None else est.fit(X, Y if target == "r" else C)


def _cpu_twin(name, params, target):
    """The same estimator constructed with device='cpu'; skipped where the class does not run there."""
    try:
        return _fit(_build(name, params, device="cpu"), target)
    except (ValueError, ImportError) as error:
        if "device='cpu'" in str(error):
            pytest.skip(str(error))
        raise


@pytest.mark.parametrize("name,params,target,operation", CASES)
def test_assigned_cpu_runs_the_next_operation_on_the_cpu(name, params, target, operation):
    twin = _cpu_twin(name, params, target)
    expected = np.asarray(getattr(twin, operation)(X))
    model = _fit(_build(name, params), target)
    before = np.asarray(getattr(model, operation)(X))
    model.device = "cpu"
    after = np.asarray(getattr(model, operation)(X))
    assert model.vendor_used() == "cpu"
    assert model.get_params()["device"] == "cpu"
    # identical mode: the same bits on every device, so the fitted state carries over
    assert np.array_equal(before, after) and np.array_equal(after, expected)


@pytest.mark.parametrize("name,params,target,operation", CASES)
def test_assigned_none_returns_to_the_installation_device(name, params, target, operation):
    model = _cpu_twin(name, params, target)
    getattr(model, operation)(X)
    assert model.vendor_used() == "cpu"
    model.device = None
    getattr(model, operation)(X)
    assert model.vendor_used() == mojolearn.vendor()


@pytest.mark.parametrize("name,params,target,operation", CASES)
def test_a_bad_value_is_refused_in_words_at_the_next_operation(name, params, target, operation):
    model = _fit(_build(name, params), target)
    expected = np.asarray(getattr(model, operation)(X))
    for bad in ("cuda", "CPU", 0, b"cpu"):
        model.device = bad
        with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None"):
            getattr(model, operation)(X)
    # nothing was changed by the refusals: the estimator runs again once the value is one of the three
    model.device = None
    assert np.array_equal(np.asarray(getattr(model, operation)(X)), expected)


def test_explicit_gpu_is_refused_in_words_on_a_cpu_only_installation():
    model = _fit(_build("Ridge", dict(alpha=1.0)), "r")
    model.predict(X)
    model.device = "gpu"
    if mojolearn.vendor() == "cpu":
        with pytest.raises(ValueError, match="needs a GPU build"):
            model.predict(X)
    else:
        model.predict(X)
        assert model.vendor_used() == mojolearn.vendor()


@pytest.mark.parametrize("write", ["setattr", "vars", "dict_update"])
def test_every_form_of_the_write_reaches_the_route(write):
    _cpu_twin("Ridge", dict(alpha=1.0), "r")
    model = _fit(_build("Ridge", dict(alpha=1.0)), "r")
    model.predict(X)
    if write == "setattr":
        setattr(model, "device", "cpu")
    elif write == "vars":
        vars(model)["device"] = "cpu"
    else:
        vars(model).update(device="cpu")
    model.predict(X)
    assert model.vendor_used() == "cpu"


def test_deleting_the_field_runs_the_installation_device():
    model = _cpu_twin("Ridge", dict(alpha=1.0), "r")
    model.predict(X)
    del model.device
    model.predict(X)
    assert model.vendor_used() == mojolearn.vendor()


def test_an_assignment_on_one_estimator_leaves_the_others_on_their_device():
    _cpu_twin("Ridge", dict(alpha=1.0), "r")
    first = _fit(_build("Ridge", dict(alpha=1.0)), "r")
    second = _fit(_build("Ridge", dict(alpha=2.0)), "r")
    first.predict(X)
    second.predict(X)
    first.device = "cpu"
    second.predict(X)
    first.predict(X)
    assert second.vendor_used() == mojolearn.vendor() and first.vendor_used() == "cpu"
    # the same object assigned again keeps the route
    first.device = first.device
    first.predict(X)
    assert first.vendor_used() == "cpu"


def test_assignment_before_the_first_operation_is_the_constructor_value():
    _cpu_twin("Ridge", dict(alpha=1.0), "r")
    model = _build("Ridge", dict(alpha=1.0))
    model.device = "cpu"
    _fit(model, "r").predict(X)
    assert model.vendor_used() == "cpu"
