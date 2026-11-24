# SPDX-License-Identifier: Apache-2.0
"""`device=None` (default), `device="gpu"` and `device="cpu"` on the tree estimators: the fit-level rows.

Random forests, extra trees, decision trees and the isolation forest take `device=`. The parameter itself
(construction, refusals in words at construction and `set_params`, clone, pickle, a constructor run again) is covered
for every class by test_device_parameter.py. This file holds what is the family's own: on a GPU installation both
table sets exist in one process and `device="cpu"` runs the CPU binaries (`vendor_used() == "cpu"`); in `identical`
mode it returns the bits the GPU returns; `set_params(device=...)` is seen by the next fit; a saved forest does not
depend on the device. These tests run on a GPU installation and on a CPU-only one; what needs two devices is skipped
by name on the CPU-only one. This file replaces test_extratrees_gpu_only.py, whose rule (`device` must be 'gpu') is
retired.
"""
import pickle

import numpy as np
import pytest

import mojolearn
from mojolearn import (
    DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreesClassifier, ExtraTreesRegressor, IsolationForest,
    RandomForestClassifier, RandomForestRegressor,
)

CLASSIFIERS = [RandomForestClassifier, ExtraTreesClassifier, DecisionTreeClassifier]
REGRESSORS = [RandomForestRegressor, ExtraTreesRegressor, DecisionTreeRegressor]
SUPERVISED = CLASSIFIERS + REGRESSORS
CLASSES = SUPERVISED + [IsolationForest]
TWO_DEVICES = mojolearn.vendor() != "cpu"
two_devices = pytest.mark.skipif(not TWO_DEVICES, reason="a CPU-only installation has one device")
REFUSAL = "device must be 'gpu', 'cpu' or None, got"


def data(cls, rows=240, cols=6, seed=3):
    rng = np.random.default_rng(seed)
    x = rng.random((rows, cols), dtype=np.float32)
    if cls in CLASSIFIERS:
        return x, (x[:, 0] + x[:, 1] > 1.0).astype(np.int64)
    return x, (x @ np.arange(1, cols + 1, dtype=np.float32)).astype(np.float32)


def build(cls, **params):
    if cls is IsolationForest:
        return cls(n_estimators=8, **params)
    if cls in (DecisionTreeClassifier, DecisionTreeRegressor):
        return cls(max_depth=5, **params)
    return cls(n_estimators=8, max_depth=5, **params)


def fit(model, cls):
    x, y = data(cls)
    return (model.fit(x) if cls is IsolationForest else model.fit(x, y)), x


def answer(model, cls, x):
    if cls is IsolationForest:
        return np.asarray(model.score_samples(x))
    if cls in CLASSIFIERS:
        return np.asarray(model.predict_proba(x))
    return np.asarray(model.predict(x))


@pytest.mark.parametrize("cls", CLASSES)
def test_device_is_a_constructor_parameter_with_default_none(cls):
    assert cls().get_params()["device"] is None
    assert cls(device=None).get_params()["device"] is None
    assert cls(device="cpu").get_params()["device"] == "cpu"
    if TWO_DEVICES:                                           # 'gpu' is refused at construction on a CPU-only one
        assert cls(device="gpu").get_params()["device"] == "gpu"


@pytest.mark.parametrize("cls", CLASSES)
def test_default_device_is_this_installations_device(cls):
    model, x = fit(build(cls), cls)
    assert model.vendor_used() == mojolearn.vendor()          # the GPU set where there is one, else the CPU
    assert model.get_params()["device"] is None               # what was passed, not what it resolved to
    assert len(answer(model, cls, x)) == len(x)


@pytest.mark.parametrize("cls", CLASSES)
def test_explicit_gpu_runs_on_a_gpu_installation_and_is_refused_in_words_on_a_cpu_only_one(cls):
    if TWO_DEVICES:
        fitted, x = fit(build(cls, device="gpu"), cls)
        assert fitted.vendor_used() == mojolearn.vendor()
        return
    with pytest.raises(ValueError, match="device='gpu' needs a GPU build of mojolearn.*device='cpu'"):
        build(cls, device="gpu")


@pytest.mark.parametrize("cls", CLASSES)
def test_cpu_device_runs_everywhere(cls):
    model, x = fit(build(cls, device="cpu"), cls)
    assert model.vendor_used() == "cpu"
    assert len(answer(model, cls, x)) == len(x)


@pytest.mark.skipif(mojolearn.vendor() != "metal", reason="numeric_mode='fast' exists on Apple GPUs only")
@pytest.mark.parametrize("cls", CLASSES)
def test_cpu_device_refuses_fast_mode_in_words(cls):
    with pytest.raises(ValueError, match="device='cpu' runs in identical mode only"):
        fit(build(cls, device="cpu", numeric_mode="fast"), cls)


@pytest.mark.parametrize("cls", CLASSES)
def test_clone_and_pickle_carry_the_device(cls):
    model = build(cls, device="cpu")
    replica = cls(**model.get_params())
    assert replica.get_params()["device"] == "cpu"
    fitted, x = fit(replica, cls)
    restored = pickle.loads(pickle.dumps(fitted))
    assert restored.get_params()["device"] == "cpu"
    assert restored.vendor_used() == "cpu"
    assert answer(restored, cls, x).tobytes() == answer(fitted, cls, x).tobytes()


@pytest.mark.parametrize("cls", [RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor])
def test_saved_file_does_not_depend_on_the_device_and_a_loaded_model_carries_none(cls, tmp_path):
    a, x = fit(build(cls, numeric_mode="identical"), cls)
    b, _ = fit(build(cls, numeric_mode="identical", device="cpu"), cls)
    a.save(str(tmp_path / "a.mlf"))
    b.save(str(tmp_path / "b.mlf"))
    assert (tmp_path / "a.mlf").read_bytes() == (tmp_path / "b.mlf").read_bytes()
    loaded = cls.load(str(tmp_path / "b.mlf"))
    # A loaded forest is inference-only: get_params refuses (it has no constructor parameters); the loader
    # writes device=None, this installation's device.
    assert vars(loaded)["device"] is None and loaded.vendor_used() == mojolearn.vendor()
    assert answer(loaded, cls, x).tobytes() == answer(a, cls, x).tobytes()


@two_devices
@pytest.mark.parametrize("cls", SUPERVISED)
def test_check_fits_both_devices_in_this_process(cls):
    from mojolearn import _check_both
    model = build(cls, numeric_mode="identical")
    assert _check_both.takes_device(model)
    gpu = _check_both.on_device(model, "gpu", fitted=False)
    cpu = _check_both.on_device(model, "cpu", fitted=False)
    assert gpu is model and cpu is not model and cpu.get_params()["device"] == "cpu"
    x, y = data(cls)
    assert answer(gpu.fit(x, y), cls, x).tobytes() == answer(cpu.fit(x, y), cls, x).tobytes()
    assert gpu.vendor_used() != "cpu" and cpu.vendor_used() == "cpu"


@pytest.mark.parametrize("cls", CLASSES)
def test_unknown_device_is_refused_in_words_at_construction_and_set_params(cls):
    with pytest.raises(ValueError, match=REFUSAL):
        build(cls, device="tpu")
    model = build(cls)
    with pytest.raises(ValueError, match=REFUSAL):
        model.set_params(device="tpu")
    assert model.get_params()["device"] is None               # refused before the estimator changed


@two_devices
@pytest.mark.parametrize("cls", CLASSES)
def test_set_params_moves_the_next_fit_to_the_other_device(cls):
    model, x = fit(build(cls, numeric_mode="identical"), cls)
    on_gpu = answer(model, cls, x)
    assert model.vendor_used() != "cpu"
    model.set_params(device="cpu")                            # fitted state is dropped with the parameters
    fit(model, cls)
    assert model.vendor_used() == "cpu"
    assert answer(model, cls, x).tobytes() == on_gpu.tobytes()
    model.set_params(device=None)
    fit(model, cls)
    assert model.vendor_used() != "cpu"
    assert answer(model, cls, x).tobytes() == on_gpu.tobytes()
