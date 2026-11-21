# SPDX-License-Identifier: Apache-2.0
"""`device=None` (default), `device="gpu"` and `device="cpu"` on the estimators of the mixture, cluster, neighbors and
density families that take it, at the Python boundary (the template is test_linear_device_boundary.py).

The device is a constructor parameter. None is this installation's device: the GPU where the installation has a GPU
set, else the CPU. `get_params` returns what was passed (None stays None). `device="cpu"` always runs; on a GPU
installation it runs the CPU binaries of the same family in the same process (`vendor_used() == "cpu"`) and in
`identical` mode returns the bits the GPU returns. An unknown value, and an explicit `device="gpu"` on a CPU-only
installation, are refused in words by the estimator's first operation: these classes are constructed and
re-parameterized by the shared field-only constructor, which stores the value as given. The estimator reads the device
once, at its first operation, into its native route owner, and reads it again only when the stored value is replaced.
`get_params`, `set_params`, clone and pickle carry it.

Each row of CASES is one estimator class: how to build it, its fit arguments and the methods whose answers are
compared across devices. A family that gains `device=` adds its row.
"""
import pickle
import sys

import numpy as np
import pytest

import mojolearn

TWO_DEVICES = mojolearn.vendor() != "cpu"
two_devices = pytest.mark.skipif(not TWO_DEVICES, reason="a CPU-only installation has one device")


def blobs(rows=240, cols=5, centers=3, seed=7):
    rng = np.random.default_rng(seed)
    means = rng.normal(0.0, 6.0, size=(centers, cols)).astype(np.float32)
    x = means[np.arange(rows) % centers] + rng.normal(0.0, 1.0, size=(rows, cols)).astype(np.float32)
    return np.ascontiguousarray(x, dtype=np.float32)


class Case:
    def __init__(self, name, build, methods, saves=False):
        self.name, self.build, self.methods, self.saves = name, build, methods, saves

    def __repr__(self):
        return self.name


CASES = [
    Case("GaussianMixture", lambda **kw: mojolearn.GaussianMixture(n_components=3, random_state=0, **kw),
         ("predict", "predict_proba", "score_samples"), saves=True),
    # the extended solver (any option the default binding does not carry): the x_cluster image
    Case("GaussianMixture-diag",
         lambda **kw: mojolearn.GaussianMixture(n_components=3, covariance_type="diag", random_state=0, **kw),
         ("predict", "predict_proba", "score_samples")),
    Case("BayesianGaussianMixture",
         lambda **kw: mojolearn.BayesianGaussianMixture(n_components=3, random_state=0, **kw),
         ("predict", "predict_proba", "score_samples")),
]
ids = [case.name for case in CASES]


def answers(model, case, x):
    return [np.asarray(getattr(model, method)(x)) for method in case.methods]


@pytest.mark.parametrize("case", CASES, ids=ids)
def test_device_is_a_constructor_parameter_with_default_none(case):
    assert case.build().get_params()["device"] is None
    assert case.build(device=None).get_params()["device"] is None
    assert case.build(device="cpu").get_params()["device"] == "cpu"
    cls = type(case.build())
    assert "device" in cls._get_param_names()
    assert sorted(cls._get_param_names()) == list(cls._get_param_names())
    assert cls._mojolearn_device is True


@pytest.mark.parametrize("case", CASES, ids=ids)
@pytest.mark.parametrize("bad", ["tpu", "GPU", " cpu", "", "None", 0, False, b"cpu"])
def test_unknown_device_is_refused_in_words_by_the_first_operation(case, bad):
    x = blobs()
    model = case.build(device=bad)
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None, got"):
        model.fit(x)
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None, got"):
        model.vendor_used()
    # nothing was fitted by the refused call
    assert not mojolearn._mode.is_fitted(model)


@pytest.mark.parametrize("case", CASES, ids=ids)
def test_default_device_is_this_installations_device(case):
    x = blobs()
    model = case.build().fit(x)
    assert model.vendor_used() == mojolearn.vendor()          # the GPU set where there is one, else the CPU
    assert model.get_params()["device"] is None               # what was passed, not what it resolved to
    assert answers(model, case, x)[0].shape[0] == len(x)


@pytest.mark.parametrize("case", CASES, ids=ids)
def test_explicit_gpu_runs_on_a_gpu_installation_and_is_refused_in_words_on_a_cpu_only_one(case):
    x = blobs()
    model = case.build(device="gpu")
    if TWO_DEVICES:
        model.fit(x)
        assert model.vendor_used() == mojolearn.vendor() and model.get_params()["device"] == "gpu"
        return
    with pytest.raises(ValueError, match="device='gpu' needs a GPU build of mojolearn.*Use device='cpu'"):
        model.fit(x)
    with pytest.raises(ValueError, match="Use device='cpu'"):
        model.vendor_used()


@pytest.mark.parametrize("case", CASES, ids=ids)
def test_cpu_device_runs_the_cpu_binary(case):
    x = blobs()
    model = case.build(device="cpu").fit(x)
    assert model.vendor_used() == "cpu"
    assert answers(model, case, x)[0].shape[0] == len(x)


@two_devices
@pytest.mark.parametrize("case", CASES, ids=ids)
def test_default_device_runs_the_gpu_binary_beside_a_cpu_estimator_in_one_process(case):
    x = blobs()
    cpu = case.build(device="cpu").fit(x)
    gpu = case.build().fit(x)
    assert gpu.vendor_used() == mojolearn.vendor() != "cpu"
    assert cpu.vendor_used() == "cpu"
    # each keeps its own route across interleaved calls
    for _ in range(2):
        assert answers(gpu, case, x)[0].shape == answers(cpu, case, x)[0].shape
        assert gpu.vendor_used() != cpu.vendor_used()


@two_devices
@pytest.mark.parametrize("case", CASES, ids=ids)
def test_identical_mode_cpu_device_returns_the_gpu_bits(case):
    if mojolearn.numeric_mode() != "identical":
        pytest.skip("bits are promised in identical mode only")
    x = blobs(rows=400, cols=6)
    gpu = case.build(numeric_mode="identical").fit(x)
    cpu = case.build(numeric_mode="identical", device="cpu").fit(x)
    for got, expected in zip(answers(cpu, case, x), answers(gpu, case, x)):
        assert got.dtype == expected.dtype and got.tobytes() == expected.tobytes()
    # one fitted model, answered by each device: the CPU estimator handed the GPU's fitted attributes
    twin = case.build(numeric_mode="identical", device="cpu")
    state, target = vars(gpu), vars(twin)
    for key, value in state.items():
        if key not in twin.get_params():
            target[key] = value
    assert twin.vendor_used() == "cpu"
    for got, expected in zip(answers(twin, case, x), answers(gpu, case, x)):
        assert got.tobytes() == expected.tobytes()


@two_devices
@pytest.mark.parametrize("case", CASES, ids=ids)
def test_cpu_device_never_runs_a_fast_binary(case):
    x = blobs()
    model = case.build(device="cpu", numeric_mode="fast")
    with pytest.raises(Exception, match="device='cpu' runs in identical mode only|numeric_mode"):
        model.fit(x)


@two_devices
@pytest.mark.parametrize("case", CASES, ids=ids)
def test_set_params_and_assignment_move_the_estimator_to_the_other_device(case):
    x = blobs()
    model = case.build().fit(x)
    assert model.vendor_used() == mojolearn.vendor()
    model.set_params(device="cpu")
    assert model.get_params()["device"] == "cpu" and model.vendor_used() == "cpu"
    model.fit(x)
    assert model.vendor_used() == "cpu"
    # a directly assigned device is a replaced value: the next operation reads it
    model.device = None
    assert model.vendor_used() == mojolearn.vendor()
    model.device = "cpu"
    assert model.vendor_used() == "cpu"


@pytest.mark.parametrize("case", CASES, ids=ids)
@pytest.mark.parametrize("device", [None, "cpu", "gpu"])
def test_clone_and_pickle_carry_the_device(case, device):
    if device == "gpu" and not TWO_DEVICES:
        pytest.skip("an explicit device='gpu' is refused on a CPU-only installation")
    x = blobs()
    model = case.build(device=device).fit(x)
    expected = answers(model, case, x)
    used = model.vendor_used()

    rebuilt = type(model)(**model.get_params())
    assert rebuilt.get_params() == model.get_params()
    assert rebuilt.get_params()["device"] == device          # None stays None
    assert rebuilt.fit(x).vendor_used() == used

    try:
        from sklearn.base import clone
    except ImportError:
        clone = None
    if clone is not None:
        cloned = clone(model)
        assert cloned.get_params()["device"] == device
        assert cloned.fit(x).vendor_used() == used

    restored = pickle.loads(pickle.dumps(model))
    assert restored.get_params()["device"] == device
    # the unpickled estimator was built without its constructor: its first
    # operation reads the device from its restored parameters
    for got, want in zip(answers(restored, case, x), expected):
        np.testing.assert_array_equal(got, want)
    assert restored.vendor_used() == used


@pytest.mark.parametrize("case", CASES, ids=ids)
def test_an_instance_without_the_field_runs_this_installations_device(case):
    x = blobs()
    model = case.build().fit(x)
    del vars(model)["device"]
    assert model.vendor_used() == mojolearn.vendor()
    assert answers(model, case, x)[0].shape[0] == len(x)


@pytest.mark.parametrize("case", [case for case in CASES if case.saves], ids=[c.name for c in CASES if c.saves])
def test_a_saved_model_loads_on_the_default_device(case, tmp_path):
    x = blobs()
    model = case.build(device="cpu").fit(x)
    path = str(tmp_path / "model.npz")
    model.save(path)
    loaded = type(model).load(path)
    assert loaded.get_params()["device"] is None
    assert loaded.vendor_used() == mojolearn.vendor()
    assert answers(loaded, case, x)[0].shape[0] == len(x)


@pytest.mark.parametrize("case", CASES, ids=ids)
def test_check_runs_both_devices_in_this_process(case, capsys):
    x = blobs()
    model = case.build().fit(x)
    result = model.check(x, method=case.methods[0])
    printed = capsys.readouterr().out
    if not TWO_DEVICES or mojolearn.numeric_mode() != "identical":
        assert not result["ran"]
        return
    assert result["ran"] and result["transfer"] == "device"
    gpu, cpu = result["sides"]["gpu"], result["sides"]["cpu"]
    assert gpu["route"] == dict(device=None, vendor_used=mojolearn.vendor())
    assert cpu["route"] == dict(device="cpu", vendor_used="cpu")
    assert result["verdict"]["match"]
    assert gpu["outputs"][0]["sha256"] == cpu["outputs"][0]["sha256"]
    assert "device='cpu' in this process" in printed
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


@pytest.mark.parametrize("case", CASES, ids=ids)
@pytest.mark.parametrize("device", [None, "cpu"])
def test_python_operations_per_call_do_not_grow_with_rows(case, device):
    counts = {}
    for name, x in (("small", blobs(rows=96)), ("large", blobs(rows=8192))):
        model = case.build(device=device)
        model.fit(x)                    # the first operation resolves the route
        counts[name] = (_python_events(lambda: model.fit(x)),
                        _python_events(lambda: getattr(model, case.methods[0])(x)))
    assert counts["small"] == counts["large"]
    # one public method is one Python frame and one call into the binding
    # (plus the measuring lambda's own frame)
    for per_operation in counts["small"]:
        assert per_operation["call"] <= 2
