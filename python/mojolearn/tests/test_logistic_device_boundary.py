# SPDX-License-Identifier: Apache-2.0
"""`device=None` (default), `device="gpu"` and `device="cpu"` on LogisticRegression, at the Python boundary.

The same rule and the same words as Ridge and LinearRegression (test_linear_device_boundary.py). The device is a
constructor parameter. None is this installation's device: the GPU where the installation has a GPU set, else the CPU.
`get_params` returns what was passed (None stays None). An explicit `device="gpu"` on a CPU-only installation is
refused in words that name `device="cpu"`; `device="cpu"` always runs. It is checked when it is given and read once, by
the estimator's first operation, into its native route owner; `get_params`, `set_params`, clone and pickle carry it.
On a GPU installation both entry sets exist in one process and `device="cpu"` runs the CPU binaries
(`vendor_used() == "cpu"`); in `identical` mode it returns the bits the GPU returns: labels, scores, probabilities,
log probabilities and coefficients, for two classes and for more, with and without sample and class weights. These
tests run on a GPU installation and on a CPU-only one; what needs two devices is skipped by name on the CPU-only one.
"""
import pickle
import sys

import numpy as np
import pytest

import mojolearn
from mojolearn import LogisticRegression

TWO_DEVICES = mojolearn.vendor() != "cpu"
two_devices = pytest.mark.skipif(not TWO_DEVICES, reason="a CPU-only installation has one device")
QUERIES = ("predict", "decision_function", "predict_proba", "predict_log_proba")
PROBLEMS = ("binary", "multiclass")


def data(problem="binary", rows=240, cols=6, seed=5):
    """Rows whose label is a threshold (two classes) or a band (four classes) of a fixed linear score, so every
    class is present and no class is separable to the last row."""
    rng = np.random.default_rng(seed)
    x = rng.random((rows, cols), dtype=np.float32)
    score = x @ np.linspace(-1.0, 1.5, cols, dtype=np.float32) + rng.normal(0.0, 0.2, rows).astype(np.float32)
    if problem == "binary":
        y = (score > np.median(score)).astype(np.int64)
    else:
        y = np.digitize(score, np.quantile(score, [0.25, 0.5, 0.75])).astype(np.int64)
    return x, y


def weights(rows):
    return np.linspace(0.5, 2.0, rows, dtype=np.float32)


def class_weight_of(problem, kind):
    if kind == "balanced":
        return "balanced"
    return {0: 1.0, 1: 2.5} if problem == "binary" else {0: 1.0, 1: 2.5, 2: 0.5, 3: 1.5}


def answers(model, x):
    return {name: np.asarray(getattr(model, name)(x)) for name in QUERIES}


def test_device_is_a_constructor_parameter_with_default_none():
    assert LogisticRegression().get_params()["device"] is None
    assert LogisticRegression(device=None).get_params()["device"] is None
    assert LogisticRegression(device="cpu").get_params()["device"] == "cpu"
    names = list(LogisticRegression().get_params())
    assert "device" in names and names == sorted(names)
    # the ten parameters the estimator had before are unchanged
    assert set(names) - {"device"} == {"penalty", "tol", "C", "fit_intercept", "class_weight", "max_iter",
                                       "linesearch_max_iter", "l1_ratio", "solver", "numeric_mode"}


@pytest.mark.parametrize("bad", ["tpu", "GPU", " cpu", "", "None", 0, False, b"cpu"])
def test_unknown_device_is_refused_in_words_at_construction(bad):
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None, got"):
        LogisticRegression(device=bad)


@pytest.mark.parametrize("problem", PROBLEMS)
def test_default_device_is_this_installations_device(problem):
    x, y = data(problem)
    model = LogisticRegression().fit(x, y)
    assert model.vendor_used() == mojolearn.vendor()          # the GPU set where there is one, else the CPU
    assert model.get_params()["device"] is None               # what was passed, not what it resolved to
    assert np.asarray(model.predict(x)).shape == (len(x),)


def test_the_route_answers_before_fit():
    model = LogisticRegression(device="cpu")
    assert model.vendor_used() == "cpu"
    assert model.numeric_mode_used() in ("identical", "fast")
    assert LogisticRegression().vendor_used() == mojolearn.vendor()


def test_explicit_gpu_runs_on_a_gpu_installation_and_is_refused_in_words_on_a_cpu_only_one():
    x, y = data()
    if TWO_DEVICES:
        model = LogisticRegression(device="gpu").fit(x, y)
        assert model.vendor_used() == mojolearn.vendor() and model.get_params()["device"] == "gpu"
        return
    with pytest.raises(ValueError, match="device='gpu' needs a GPU build of mojolearn.*Use device='cpu'"):
        LogisticRegression(device="gpu")
    model = LogisticRegression().fit(x, y)
    with pytest.raises(ValueError, match="Use device='cpu'"):
        model.set_params(device="gpu")
    assert model.get_params()["device"] is None
    # an instance that carries 'gpu' without its constructor's check is refused by its first operation
    carried = LogisticRegression()
    carried.device = "gpu"
    with pytest.raises(ValueError, match="Use device='cpu'"):
        carried.fit(x, y)


def test_set_params_refuses_an_unknown_device_and_leaves_the_estimator_unchanged():
    x, y = data()
    model = LogisticRegression().fit(x, y)
    before = np.asarray(model.predict_proba(x)).copy()
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None, got 'tpu'"):
        model.set_params(device="tpu")
    assert model.get_params()["device"] is None
    np.testing.assert_array_equal(np.asarray(model.predict_proba(x)), before)   # still fitted, same route


@pytest.mark.parametrize("problem", PROBLEMS)
def test_cpu_device_runs_the_cpu_binary(problem):
    x, y = data(problem)
    classes = 2 if problem == "binary" else 4
    model = LogisticRegression(device="cpu").fit(x, y)
    assert model.vendor_used() == "cpu"
    assert model.numeric_mode_used() == "identical" or not TWO_DEVICES
    got = answers(model, x)
    assert got["predict"].shape == (len(x),)
    assert got["decision_function"].shape == ((len(x),) if classes == 2 else (len(x), classes))
    assert got["predict_proba"].shape == got["predict_log_proba"].shape == (len(x), classes)
    assert set(np.unique(got["predict"])) <= set(model.classes_)
    assert isinstance(model.score(x, y), float)


@two_devices
@pytest.mark.parametrize("problem", PROBLEMS)
def test_default_device_runs_the_gpu_binary_beside_a_cpu_estimator_in_one_process(problem):
    x, y = data(problem)
    cpu = LogisticRegression(device="cpu").fit(x, y)
    gpu = LogisticRegression().fit(x, y)
    assert gpu.vendor_used() == mojolearn.vendor() != "cpu"
    assert cpu.vendor_used() == "cpu"
    # each keeps its own route across interleaved calls
    for _ in range(2):
        assert np.asarray(gpu.predict_proba(x)).shape == np.asarray(cpu.predict_proba(x)).shape
        assert gpu.vendor_used() != cpu.vendor_used()


@two_devices
@pytest.mark.parametrize("problem", PROBLEMS)
@pytest.mark.parametrize("sample_weight", [False, True])
@pytest.mark.parametrize("class_weight", [None, "balanced", "dict"])
def test_identical_mode_cpu_device_returns_the_gpu_bits(problem, sample_weight, class_weight):
    if mojolearn.numeric_mode() != "identical":
        pytest.skip("bits are promised in identical mode only")
    x, y = data(problem, rows=400, cols=8)
    weight = weights(len(x)) if sample_weight else None
    params = dict(numeric_mode="identical", max_iter=60,
                  class_weight=None if class_weight is None else class_weight_of(problem, class_weight))
    gpu = LogisticRegression(**params).fit(x, y, weight)
    cpu = LogisticRegression(device="cpu", **params).fit(x, y, weight)
    assert gpu.vendor_used() != "cpu" and cpu.vendor_used() == "cpu"
    assert list(gpu.classes_) == list(cpu.classes_)
    assert int(np.asarray(gpu.n_iter_)[0]) == int(np.asarray(cpu.n_iter_)[0])
    assert np.asarray(gpu.coef_).tobytes() == np.asarray(cpu.coef_).tobytes()
    assert np.asarray(gpu.intercept_).tobytes() == np.asarray(cpu.intercept_).tobytes()
    on_gpu, on_cpu = answers(gpu, x), answers(cpu, x)
    for name in QUERIES:
        assert on_gpu[name].dtype == on_cpu[name].dtype and on_gpu[name].shape == on_cpu[name].shape, name
        assert on_gpu[name].tobytes() == on_cpu[name].tobytes(), name
    assert gpu.score(x, y) == cpu.score(x, y)


@two_devices
def test_cpu_device_never_runs_a_fast_binary():
    x, y = data()
    model = LogisticRegression(device="cpu", numeric_mode="fast")
    try:
        model.fit(x, y)
    except ValueError as refusal:
        # `fast` names its own binaries: the CPU has none, refused in words
        assert "device='cpu' runs in identical mode only" in str(refusal)
    except ImportError:
        # where no fast binaries are installed, the mode's own refusal stands
        pass
    else:
        # Off Apple with no fast set the identical set answers `fast` for this family (core/mode_answer.mojo,
        # MODE_FAMILIES_ANSWERED; lane gs-fix: the test predated it and failed on the L40S GPU install).
        assert model.vendor_used() == "cpu"


@pytest.mark.parametrize("problem", PROBLEMS)
def test_set_params_moves_the_estimator_to_the_other_device(problem):
    x, y = data(problem)
    model = LogisticRegression().fit(x, y)
    assert model.set_params(device="cpu") is model
    assert model.get_params()["device"] == "cpu"
    model.fit(x, y)
    assert model.vendor_used() == "cpu"
    model.set_params(device=None).fit(x, y)
    assert model.get_params()["device"] is None
    assert model.vendor_used() == mojolearn.vendor()


@pytest.mark.parametrize("device", [None, "cpu", "gpu"])
def test_clone_and_pickle_carry_the_device(device):
    if device == "gpu" and not TWO_DEVICES:
        pytest.skip("an explicit device='gpu' is refused on a CPU-only installation")
    x, y = data("multiclass")
    model = LogisticRegression(device=device, class_weight="balanced").fit(x, y)
    expected = np.asarray(model.predict_proba(x)).copy()
    used = model.vendor_used()

    rebuilt = LogisticRegression(**model.get_params())
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
    np.testing.assert_array_equal(np.asarray(restored.predict_proba(x)), expected)
    assert restored.vendor_used() == used


def test_an_instance_without_the_device_field_runs_this_installations_device():
    # a model pickled before the parameter existed carries no `device`
    x, y = data()
    model = LogisticRegression().fit(x, y)
    expected = np.asarray(model.predict(x)).copy()
    state = dict(object.__getattribute__(model, "__dict__"))
    state.pop("device")
    old = LogisticRegression.__new__(LogisticRegression)
    object.__getattribute__(old, "__dict__").update(state)
    assert old.get_params()["device"] is None
    np.testing.assert_array_equal(np.asarray(old.predict(x)), expected)
    assert old.vendor_used() == mojolearn.vendor()


def test_a_changed_numeric_mode_resolves_the_route_again():
    x, y = data()
    model = LogisticRegression(device="cpu").fit(x, y)
    assert model.vendor_used() == "cpu"
    model.numeric_mode = "identical"
    assert model.numeric_mode_used() == "identical"
    assert model.vendor_used() == "cpu"
    np.asarray(model.predict(x))
    model.numeric_mode = "no such mode"
    with pytest.raises(ValueError):
        model.predict(x)


@pytest.mark.parametrize("problem", PROBLEMS)
def test_a_saved_model_loads_on_the_default_device(problem, tmp_path):
    x, y = data(problem)
    model = LogisticRegression(device="cpu").fit(x, y)
    path = str(tmp_path / "model.npz")
    model.save(path)
    loaded = LogisticRegression.load(path)
    assert loaded.get_params()["device"] is None
    assert loaded.vendor_used() == mojolearn.vendor()
    np.testing.assert_array_equal(np.asarray(loaded.predict(x)), np.asarray(model.predict(x)))
    host = mojolearn.host_model(path)
    assert host.vendor_used() == "cpu"
    np.testing.assert_array_equal(np.asarray(host.predict(x)), np.asarray(model.predict(x)))


@pytest.mark.parametrize("problem", PROBLEMS)
@pytest.mark.parametrize("method", [None, "predict_proba", "decision_function", "predict_log_proba"])
def test_check_runs_both_devices_in_this_process(problem, method, capsys):
    x, y = data(problem)
    model = LogisticRegression().fit(x, y)
    result = model.check(x) if method is None else model.check(x, method=method)
    printed = capsys.readouterr().out
    if not TWO_DEVICES or mojolearn.numeric_mode() != "identical":
        assert not result["ran"]
        return
    assert result["ran"] and result["transfer"] == "device" and result["method"] == (method or "predict")
    gpu, cpu = result["sides"]["gpu"], result["sides"]["cpu"]
    assert gpu["route"] == dict(device=None, vendor_used=mojolearn.vendor())
    assert cpu["route"] == dict(device="cpu", vendor_used="cpu")
    assert result["verdict"]["match"]
    assert gpu["outputs"][0]["sha256"] == cpu["outputs"][0]["sha256"]
    assert "device=None (this installation's device) in this process" in printed
    assert "device='cpu' in this process" in printed
    # the given estimator is untouched
    assert model.get_params()["device"] is None and model.vendor_used() == mojolearn.vendor()


@two_devices
@pytest.mark.parametrize("problem", PROBLEMS)
def test_check_fit_fits_one_estimator_per_device_and_compares_them(problem):
    if mojolearn.numeric_mode() != "identical":
        pytest.skip("the check runs in identical mode only")
    import mojolearn._check_both as cb
    x, y = data(problem)
    est = LogisticRegression(class_weight="balanced", max_iter=60)
    result = cb.check_estimator(mojolearn, est, x, method="predict_proba", fit=(x, y, weights(len(x))),
                                fit_both=True)
    assert result["ran"] and result["form"] == "fit-both" and result["transfer"] == "device"
    assert result["sides"]["cpu"]["route"]["vendor_used"] == "cpu"
    assert result["verdict"]["match"]


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


@pytest.mark.parametrize("problem", PROBLEMS)
@pytest.mark.parametrize("device", [None, "cpu"])
def test_python_operations_per_call_do_not_grow_with_rows(problem, device):
    counts = {}
    for name, rows in (("small", 64), ("large", 8192)):
        x, y = data(problem, rows=rows)
        model = LogisticRegression(device=device, max_iter=20)
        model.fit(x, y)                    # the first operation resolves the route
        counts[name] = (_python_events(lambda: model.fit(x, y)),
                        _python_events(lambda: model.predict(x)),
                        _python_events(lambda: model.predict_proba(x)),
                        _python_events(lambda: model.score(x, y)))
    assert counts["small"] == counts["large"]
    # one public method is one Python frame and one call into the binding
    # (plus the measuring lambda's own frame)
    for per_operation in counts["small"]:
        assert per_operation["call"] <= 2
