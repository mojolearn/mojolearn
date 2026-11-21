# SPDX-License-Identifier: Apache-2.0
"""`device=` on the tree ensembles built from tree members: the fit-level rows.

The parameter itself (construction, refusals in words at construction and `set_params`, clone, pickle, a constructor
run again) is covered for every class by test_device_parameter.py. This file holds what is these families' own: the
ensemble's operations and its members run on the ensemble's device (`vendor_used()`), `identical` mode returns the
same bits on both devices of one process, a member read back carries the ensemble's device, and a member estimator
given with another explicit device is refused in words. What needs two devices is skipped by name on a CPU-only
installation.
"""
import pickle

import numpy as np
import pytest

import mojolearn
from mojolearn import (
    AdaBoostClassifier, AdaBoostRegressor, BaggingClassifier, BaggingRegressor, DecisionTreeClassifier,
    DecisionTreeRegressor,
)

CLASSIFIERS = [BaggingClassifier, AdaBoostClassifier]
REGRESSORS = [BaggingRegressor, AdaBoostRegressor]
CLASSES = CLASSIFIERS + REGRESSORS
MEMBER = {BaggingClassifier: DecisionTreeClassifier, AdaBoostClassifier: DecisionTreeClassifier,
          BaggingRegressor: DecisionTreeRegressor, AdaBoostRegressor: DecisionTreeRegressor}
TWO_DEVICES = mojolearn.vendor() != "cpu"
two_devices = pytest.mark.skipif(not TWO_DEVICES, reason="a CPU-only installation has one device")


def data(cls, rows=240, cols=6, seed=3):
    rng = np.random.default_rng(seed)
    x = rng.random((rows, cols), dtype=np.float32)
    if cls in CLASSIFIERS:
        return x, (x[:, 0] + x[:, 1] > 1.0).astype(np.int64)
    return x, (x @ np.arange(1, cols + 1, dtype=np.float32)).astype(np.float32)


def build(cls, **params):
    return cls(n_estimators=6, random_state=5, **params)


def fit(model, cls):
    x, y = data(cls)
    return model.fit(x, y), x


def answer(model, cls, x):
    if cls in CLASSIFIERS:
        return np.asarray(model.predict_proba(x))
    return np.asarray(model.predict(x))


@pytest.mark.parametrize("cls", CLASSES)
def test_default_device_is_this_installations_device(cls):
    model, x = fit(build(cls), cls)
    assert model.vendor_used() == mojolearn.vendor()
    assert model.get_params()["device"] is None
    assert len(answer(model, cls, x)) == len(x)


@pytest.mark.parametrize("cls", CLASSES)
def test_cpu_device_runs_the_ensemble_and_its_members_on_the_cpu(cls):
    model, x = fit(build(cls, device="cpu"), cls)
    assert model.vendor_used() == "cpu"
    assert len(answer(model, cls, x)) == len(x)
    members = model.estimators_
    assert len(members) > 0
    assert all(member.get_params()["device"] == "cpu" for member in members)
    assert members[0].vendor_used() == "cpu"
    assert len(np.asarray(members[0].predict(x))) == len(x)


@two_devices
@pytest.mark.parametrize("cls", CLASSES)
def test_two_devices_in_one_process_return_the_same_bits(cls):
    gpu, x = fit(build(cls, numeric_mode="identical"), cls)
    cpu, _ = fit(build(cls, numeric_mode="identical", device="cpu"), cls)
    assert gpu.vendor_used() != "cpu" and cpu.vendor_used() == "cpu"
    first = answer(gpu, cls, x)
    second = answer(cpu, cls, x)
    third = answer(gpu, cls, x)
    assert first.tobytes() == second.tobytes() == third.tobytes()


@two_devices
@pytest.mark.parametrize("cls", CLASSES)
def test_bags_with_per_member_fits_return_the_same_bits(cls):
    if cls not in (BaggingClassifier, BaggingRegressor):
        pytest.skip("the row and feature draws are Bagging parameters")
    extra = dict(max_samples=0.7, max_features=0.5, bootstrap_features=True, numeric_mode="identical")
    gpu, x = fit(build(cls, **extra), cls)
    cpu, _ = fit(build(cls, device="cpu", **extra), cls)
    assert answer(gpu, cls, x).tobytes() == answer(cpu, cls, x).tobytes()


@pytest.mark.parametrize("cls", CLASSES)
def test_a_member_estimator_with_another_explicit_device_is_refused_in_words(cls):
    x, y = data(cls)
    with pytest.raises(ValueError, match="estimator has device='cpu' but the ensemble has device=None"):
        build(cls, estimator=MEMBER[cls](max_depth=3, device="cpu")).fit(x, y)
    model = build(cls, estimator=MEMBER[cls](max_depth=3, device="cpu"), device="cpu").fit(x, y)
    assert model.vendor_used() == "cpu"


@pytest.mark.parametrize("cls", CLASSES)
def test_a_member_estimator_without_a_device_follows_the_ensemble(cls):
    x, y = data(cls)
    base = MEMBER[cls](max_depth=3)
    model = build(cls, estimator=base, device="cpu").fit(x, y)
    assert base.get_params()["device"] is None                # the given estimator is not changed
    assert all(member.get_params()["device"] == "cpu" for member in model.estimators_)


@pytest.mark.parametrize("cls", CLASSES)
def test_pickle_carries_the_device(cls):
    fitted, x = fit(build(cls, device="cpu"), cls)
    restored = pickle.loads(pickle.dumps(fitted))
    assert restored.get_params()["device"] == "cpu"
    assert restored.vendor_used() == "cpu"
    assert answer(restored, cls, x).tobytes() == answer(fitted, cls, x).tobytes()


@two_devices
@pytest.mark.parametrize("cls", CLASSES)
def test_set_params_moves_the_next_fit_to_the_other_device(cls):
    model, x = fit(build(cls, numeric_mode="identical"), cls)
    on_gpu = answer(model, cls, x)
    model.set_params(device="cpu")
    fit(model, cls)
    assert model.vendor_used() == "cpu"
    assert answer(model, cls, x).tobytes() == on_gpu.tobytes()


@two_devices
@pytest.mark.parametrize("cls", CLASSES)
def test_check_fits_both_devices_in_this_process(cls):
    from mojolearn import _check_both
    model = build(cls, numeric_mode="identical")
    assert _check_both.takes_device(model)
    gpu = _check_both.on_device(model, "gpu", fitted=False)
    cpu = _check_both.on_device(model, "cpu", fitted=False)
    assert gpu is model and cpu is not model and cpu.get_params()["device"] == "cpu"
    x, y = data(cls)
    assert answer(gpu.fit(x, y), cls, x).tobytes() == answer(cpu.fit(x, y), cls, x).tobytes()


# ---- RandomTreesEmbedding and DART (the same route; their trees store the ensemble's device)
from mojolearn import DARTClassifier, DARTRegressor, RandomTreesEmbedding  # noqa: E402

LATER = [DARTRegressor, DARTClassifier, RandomTreesEmbedding]


def later_fit(cls, **params):
    x, y = data(BaggingClassifier if cls is DARTClassifier else BaggingRegressor)
    if cls is RandomTreesEmbedding:
        model = cls(n_estimators=6, max_depth=3, random_state=5, **params)
        return model, np.asarray(model.fit_transform(x)), x
    model = cls(n_estimators=8, num_leaves=7, random_state=5, **params).fit(x, y)
    return model, np.asarray(model.predict(x)), x


@pytest.mark.parametrize("cls", LATER)
def test_later_families_cpu_device_runs_on_the_cpu(cls):
    model, out, x = later_fit(cls, device="cpu")
    assert model.vendor_used() == "cpu"
    assert len(out) == len(x)
    assert model.get_params()["device"] == "cpu"


@two_devices
@pytest.mark.parametrize("cls", LATER)
def test_later_families_two_devices_return_the_same_bits(cls):
    gpu, on_gpu, _ = later_fit(cls, numeric_mode="identical")
    cpu, on_cpu, _ = later_fit(cls, numeric_mode="identical", device="cpu")
    assert gpu.vendor_used() != "cpu" and cpu.vendor_used() == "cpu"
    assert on_gpu.tobytes() == on_cpu.tobytes()


@pytest.mark.parametrize("cls", [DARTRegressor, DARTClassifier])
def test_dart_refuses_an_unknown_device_at_construction_and_set_params(cls):
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None, got"):
        cls(device="tpu")
    model = cls()
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None, got"):
        model.set_params(device="tpu")
    assert model.get_params()["device"] is None
    assert cls(**cls(device="cpu").get_params()).get_params()["device"] == "cpu"


# ---- gradient boosting (the same route: one table set per estimator; the adapters' learner stores their device)
from mojolearn.ensemble import (  # noqa: E402
    ExperimentalTwoLevelFeatureFreq, GradientBoosting, GradientBoostingClassifier, GradientBoostingRegressor,
    OrderedRMSE,
)

GBDT = [GradientBoosting, GradientBoostingRegressor, GradientBoostingClassifier]


def gbdt_build(cls, **params):
    if cls is GradientBoosting:
        return cls(loss="RMSE", n_estimators=8, max_depth=3, learning_rate=0.1, **params)
    return cls(n_estimators=8, max_depth=3, learning_rate=0.1, **params)


def gbdt_fit(cls, **params):
    x, y = data(BaggingClassifier if cls is GradientBoostingClassifier else BaggingRegressor)
    model = gbdt_build(cls, **params).fit(x, y)
    if cls is GradientBoostingClassifier:
        return model, np.asarray(model.predict_proba(x)), x
    return model, np.asarray(model.predict(x)), x


@pytest.mark.parametrize("cls", GBDT)
def test_gradient_boosting_cpu_device_runs_on_the_cpu(cls):
    model, out, x = gbdt_fit(cls, device="cpu")
    assert model.vendor_used() == "cpu"
    assert len(out) == len(x)
    assert model.get_params()["device"] == "cpu"


@two_devices
@pytest.mark.parametrize("cls", GBDT)
def test_gradient_boosting_two_devices_return_the_same_bits(cls):
    gpu, on_gpu, _ = gbdt_fit(cls, numeric_mode="identical")
    cpu, on_cpu, _ = gbdt_fit(cls, numeric_mode="identical", device="cpu")
    assert gpu.vendor_used() != "cpu" and cpu.vendor_used() == "cpu"
    assert on_gpu.tobytes() == on_cpu.tobytes()


@pytest.mark.parametrize("cls", [GradientBoostingRegressor, GradientBoostingClassifier])
def test_the_adapters_learner_stores_the_adapters_device(cls):
    learner = gbdt_build(cls, device="cpu")._new_learner(None)
    assert learner.get_params()["device"] == "cpu"


@pytest.mark.parametrize("cls", GBDT + [OrderedRMSE, ExperimentalTwoLevelFeatureFreq])
def test_gradient_boosting_refuses_an_unknown_device_at_construction_and_set_params(cls):
    first = {"sources": (0, 1)} if cls is ExperimentalTwoLevelFeatureFreq else {}
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None, got"):
        cls(device="tpu", **first)
    model = cls(**first)
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None, got"):
        model.set_params(device="tpu")
    assert model.get_params()["device"] is None
    assert cls(**cls(device="cpu", **first).get_params()).get_params()["device"] == "cpu"


@pytest.mark.parametrize("cls", GBDT)
def test_gradient_boosting_pickle_carries_the_device_and_no_resident_copy(cls):
    model, out, x = gbdt_fit(cls, device="cpu")
    again = pickle.loads(pickle.dumps(model))
    assert again.get_params()["device"] == "cpu" and again.vendor_used() == "cpu"
    back = again.predict_proba(x) if cls is GradientBoostingClassifier else again.predict(x)
    assert np.asarray(back).tobytes() == out.tobytes()


@two_devices
@pytest.mark.parametrize("cls", GBDT)
def test_gradient_boosting_set_params_moves_the_next_fit_to_the_other_device(cls):
    model, _, x = gbdt_fit(cls)
    assert model.vendor_used() != "cpu"
    model.set_params(device="cpu")
    y = data(BaggingClassifier if cls is GradientBoostingClassifier else BaggingRegressor)[1]
    model.fit(x, y)
    assert model.vendor_used() == "cpu"


# TreeExplainer takes no device of its own: it is prepared in the run context of the model it explains.
def test_tree_explainer_follows_a_cpu_models_device():
    from mojolearn import RandomForestRegressor, TreeExplainer

    x, y = data(BaggingRegressor)
    model = RandomForestRegressor(n_estimators=4, max_depth=3, random_state=0, device="cpu").fit(x, y)
    explainer = TreeExplainer(model, x[:32])
    values = np.asarray(explainer.shap_values(x[:8]))
    assert values.shape == (8, x.shape[1])
    assert np.isfinite(values).all()
    explainer.close()
