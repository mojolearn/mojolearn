# SPDX-License-Identifier: Apache-2.0
"""`device=` in model selection and the multi-GPU drivers, at the Python boundary.

These owners hold member estimators. A member keeps its own `device`: a clone, a search candidate and a worker copy
carry it unchanged and the member's own fit, predict and score run on the member's device. What the owner decides
itself is where its fold rows live: a member that says `device="cpu"` on a GPU installation takes host-gathered
folds. `GridSearchCV` and `RandomizedSearchCV` take `device=None | "gpu" | "cpu"` for their own fold rows, checked
when given in the same words as every estimator. A multi-GPU driver has no host form and refuses a `device="cpu"`
member in words. These tests run on a GPU installation and on a CPU-only one; what needs two devices is skipped by
name on the CPU-only one.
"""
import pickle

import numpy as np
import pytest

import mojolearn
from mojolearn import Ridge
from mojolearn.model_selection import (
    GridSearchCV, KFold, RandomizedSearchCV, cross_val_predict, cross_val_score, cross_validate, validation_curve,
)

TWO_DEVICES = mojolearn.vendor() != "cpu"
two_devices = pytest.mark.skipif(not TWO_DEVICES, reason="a CPU-only installation has one device")
SEARCHES = [
    lambda **kw: GridSearchCV(Ridge(), {"alpha": [0.1, 1.0, 10.0]}, cv=3, **kw),
    lambda **kw: RandomizedSearchCV(Ridge(), {"alpha": [0.01, 0.1, 1.0, 10.0]}, n_iter=3, cv=3, random_state=0, **kw),
]
IDS = ["grid", "randomized"]


def data(rows=240, cols=6, seed=5):
    rng = np.random.default_rng(seed)
    x = rng.random((rows, cols), dtype=np.float32)
    y = (x @ np.arange(1, cols + 1, dtype=np.float32) + np.float32(0.25)).astype(np.float32)
    return x, y


def same_bits(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


# ------------------------------------------------------------------ the search's own device

@pytest.mark.parametrize("make", SEARCHES, ids=IDS)
def test_search_device_is_a_constructor_parameter_with_default_none(make):
    assert make().get_params()["device"] is None
    assert make(device=None).get_params()["device"] is None
    assert make(device="cpu").get_params()["device"] == "cpu"
    assert list(make().get_params())[-1] == "device"          # the last constructor parameter


@pytest.mark.parametrize("make", SEARCHES, ids=IDS)
@pytest.mark.parametrize("bad", ["tpu", "GPU", " cpu", "", "None", 0, False, b"cpu"])
def test_search_refuses_an_unknown_device_in_words_at_construction(make, bad):
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None, got"):
        make(device=bad)


@pytest.mark.parametrize("make", SEARCHES, ids=IDS)
def test_search_explicit_gpu_is_refused_in_words_on_a_cpu_only_installation(make):
    if TWO_DEVICES:
        assert make(device="gpu").get_params()["device"] == "gpu"
        return
    with pytest.raises(ValueError, match="device='gpu' needs a GPU build of mojolearn.*Use device='cpu'"):
        make(device="gpu")
    search = make()
    with pytest.raises(ValueError, match="Use device='cpu'"):
        search.set_params(device="gpu")
    assert search.get_params()["device"] is None


@pytest.mark.parametrize("make", SEARCHES, ids=IDS)
def test_search_set_params_refuses_an_unknown_device_and_leaves_the_search_unchanged(make):
    search = make(device="cpu")
    before = search.get_params()
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None, got"):
        search.set_params(device="metal", refit=False)
    after = search.get_params()
    assert after["device"] == "cpu" and after["refit"] is before["refit"]
    assert search.set_params(device=None).get_params()["device"] is None


@pytest.mark.parametrize("make", SEARCHES, ids=IDS)
@pytest.mark.parametrize("device", [None, "cpu"])
def test_search_pickle_carries_the_device(make, device):
    search = make(device=device)
    assert pickle.loads(pickle.dumps(search)).get_params()["device"] == device


@pytest.mark.parametrize("make", SEARCHES, ids=IDS)
def test_search_on_the_cpu_selects_the_same_candidate_with_the_same_scores(make):
    # the search's own device moves its fold rows, never a bit of a score
    x, y = data()
    default = make().fit(x, y)
    on_cpu = make(device="cpu").fit(x, y)
    assert on_cpu.best_params_ == default.best_params_
    assert same_bits(on_cpu.cv_results_["mean_test_score"], default.cv_results_["mean_test_score"])
    assert same_bits(on_cpu.predict(x), default.predict(x))
    # the member's device is the member's: the search's device does not move it
    assert on_cpu.best_estimator_.get_params()["device"] is None
    assert on_cpu.best_estimator_.vendor_used() == mojolearn.vendor()


# ------------------------------------------------------------------ the member's device passes through

@pytest.mark.parametrize("device", [None, "cpu"])
def test_search_passes_the_members_device_through_unchanged(device):
    x, y = data()
    search = GridSearchCV(Ridge(device=device), {"alpha": [0.1, 1.0]}, cv=3).fit(x, y)
    best = search.best_estimator_
    assert best.get_params()["device"] == device
    assert best.vendor_used() == ("cpu" if device == "cpu" else mojolearn.vendor())
    assert search.estimator.get_params()["device"] == device   # the base estimator is untouched


@two_devices
def test_a_cpu_member_cross_validates_to_the_gpu_members_bits():
    x, y = data()
    cv = KFold(4)
    gpu, cpu = Ridge(alpha=0.5), Ridge(alpha=0.5, device="cpu")
    assert same_bits(cross_val_score(cpu, x, y, cv=cv), cross_val_score(gpu, x, y, cv=cv))
    assert same_bits(cross_val_predict(cpu, x, y, cv=cv), cross_val_predict(gpu, x, y, cv=cv))
    a, b = cross_validate(cpu, x, y, cv=cv), cross_validate(gpu, x, y, cv=cv)
    assert same_bits(a["test_score"], b["test_score"])
    # the estimators handed in are untouched and unfitted
    assert cpu.get_params()["device"] == "cpu" and gpu.get_params()["device"] is None


@two_devices
def test_a_search_over_the_members_device_fits_each_candidate_on_its_own_device():
    x, y = data()
    search = GridSearchCV(Ridge(alpha=0.5), {"device": ["gpu", "cpu"]}, cv=3).fit(x, y)
    scores = np.asarray(search.cv_results_["mean_test_score"])
    assert scores.shape == (2,) and scores[0].tobytes() == scores[1].tobytes()    # identical mode: the same bits
    assert search.best_estimator_.get_params()["device"] in ("gpu", "cpu")
    assert search.best_estimator_.vendor_used() == (
        "cpu" if search.best_estimator_.get_params()["device"] == "cpu" else mojolearn.vendor())


@two_devices
def test_a_validation_curve_over_the_members_device_matches_across_devices():
    x, y = data()
    train, test = validation_curve(Ridge(alpha=0.5), x, y, param_name="device", param_range=["gpu", "cpu"], cv=3)
    train, test = np.asarray(train), np.asarray(test)
    assert train[0].tobytes() == train[1].tobytes() and test[0].tobytes() == test[1].tobytes()


def test_cross_validation_refuses_a_member_with_an_unknown_device_in_words():
    # a member that carries an unknown value without its constructor's check
    x, y = data()
    carried = Ridge()
    carried.device = "tpu"
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None, got"):
        cross_val_score(carried, x, y, cv=3)


# ------------------------------------------------------------------ check

@two_devices
def test_search_check_is_the_best_estimators_check_on_both_devices_in_this_process(capsys):
    x, y = data()
    search = GridSearchCV(Ridge(), {"alpha": [0.1, 1.0]}, cv=3).fit(x, y)
    result = search.check(x)
    printed = capsys.readouterr().out
    assert result["ran"] and result["verdict"]["verdict"] == "MATCH"
    assert result["estimator"] == "Ridge" and result.get("transfer") == "device"
    assert result["sides"]["cpu"]["route"]["vendor_used"] == "cpu"
    assert result["sides"]["gpu"]["route"]["vendor_used"] == mojolearn.vendor()
    assert "MATCH" in printed


def test_search_check_without_a_fitted_best_estimator_says_why(capsys):
    x, y = data()
    search = GridSearchCV(Ridge(), {"alpha": [0.1, 1.0]}, cv=3, refit=False).fit(x, y)
    result = search.check(x)
    assert result["ran"] is False and "refit=False" in result["reason"]
    assert "NOT RUN" in capsys.readouterr().out


# ------------------------------------------------------------------ the multi-GPU drivers

@two_devices
def test_a_multi_gpu_driver_refuses_a_cpu_member_in_words():
    from mojolearn.parallel_classical import fit_gram_estimator
    x, y = data()
    member = Ridge(device="cpu")
    with pytest.raises(ValueError, match=r"parallel fit_gram_estimator spreads its work across the GPUs listed in "
                                         r"devices=, and this estimator has device='cpu'"):
        fit_gram_estimator(member, x, y, devices=(0,))
    assert not hasattr(member, "coef_")                          # refused before anything was fitted


@two_devices
def test_a_multi_gpu_driver_takes_the_default_and_the_gpu_member_as_before():
    from mojolearn.parallel_classical import fit_gram_estimator
    x, y = data()
    own = Ridge().fit(x, y)
    for device in (None, "gpu"):
        fitted = fit_gram_estimator(Ridge(device=device), x, y, devices=(0,))
        assert same_bits(fitted.coef_, own.coef_)
        assert fitted.get_params()["device"] == device
