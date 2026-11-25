# SPDX-License-Identifier: Apache-2.0
"""`device=None` (default), `device="gpu"` and `device="cpu"` on the encoders, imputers, discretizer, spline, label
transformers and the score selector, at the Python boundary.

SimpleImputer, IterativeImputer, KBinsDiscretizer, OrdinalEncoder, OneHotEncoder, TargetEncoder, SplineTransformer,
SelectKBest, LabelEncoder, LabelBinarizer and MultiLabelBinarizer take the device as a constructor parameter, with the rule and the words of the
scalers (test_scaler_device_boundary.py). None is this installation's device. An unknown value, and an explicit
`device="gpu"` on a CPU-only installation, are refused in words where they are given; `device="cpu"` always runs. The
device is read once, by the estimator's first data operation; `get_params`, `set_params`, clone and pickle carry it.
On a GPU installation `device="cpu"` returns, in `identical` mode, the bits the GPU returns. What needs two devices is
skipped by name on a CPU-only installation.
"""
import pickle

import numpy as np
import pytest

import mojolearn
from mojolearn import (IterativeImputer, KBinsDiscretizer, LabelBinarizer, LabelEncoder, MultiLabelBinarizer,
                       OneHotEncoder, OrdinalEncoder, SelectKBest, SimpleImputer, SplineTransformer, TargetEncoder)

TWO_DEVICES = mojolearn.vendor() != "cpu"
two_devices = pytest.mark.skipif(not TWO_DEVICES, reason="a CPU-only installation has one device")


def _categories(rows=240, cols=4, seed=5):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 7, size=(rows, cols)).astype(np.float32)


def _continuous(rows=240, cols=4, seed=5):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((rows, cols)) * np.arange(1, cols + 1)).astype(np.float32)


def _with_nan():
    x = _continuous()
    x[::7, 1] = np.nan
    return x


def _labels(rows=240, seed=5):
    return np.random.default_rng(seed).integers(0, 5, size=rows).astype(np.int64)


def _label_rows():
    rng = np.random.default_rng(5)
    return [sorted(set(rng.integers(0, 6, size=3).tolist())) for _ in range(120)]


# (id, class, constructor keywords, fit arguments, has inverse_transform)
CASES = [
    ("imputer-mean", SimpleImputer, dict(strategy="mean"), lambda: (_with_nan(),), False),
    ("imputer-median", SimpleImputer, dict(strategy="median"), lambda: (_with_nan(),), False),
    ("kbins-ordinal", KBinsDiscretizer, dict(n_bins=4, encode="ordinal", strategy="uniform"),
     lambda: (_continuous(),), True),
    ("ordinal", OrdinalEncoder, dict(), lambda: (_categories(),), True),
    ("onehot", OneHotEncoder, dict(), lambda: (_categories(),), True),
    ("target", TargetEncoder, dict(target_type="continuous", random_state=0),
     lambda: (_categories(), _continuous(cols=1)[:, 0]), False),
    ("spline", SplineTransformer, dict(n_knots=4, degree=2), lambda: (_continuous(),), False),
    ("label-encoder", LabelEncoder, dict(), lambda: (_labels(),), True),
    ("label-binarizer", LabelBinarizer, dict(), lambda: (_labels(),), True),
    ("multilabel", MultiLabelBinarizer, dict(), lambda: (_label_rows(),), False),
    # the score selector: one row per score program (class statistics, Pearson, chi2 with its column statistics,
    # and the mutual-information scores, which the fit takes through the family entry in its own context)
    ("kbest-f-classif", SelectKBest, dict(score_func=mojolearn.f_classif, k=2),
     lambda: (_continuous(), _labels()), False),
    ("kbest-chi2", SelectKBest, dict(score_func=mojolearn.chi2, k=2), lambda: (_categories(), _labels()), False),
    ("kbest-f-regression", SelectKBest, dict(score_func=mojolearn.f_regression, k=2),
     lambda: (_continuous(), _continuous(cols=1, seed=9)[:, 0]), False),
    ("kbest-r-regression", SelectKBest, dict(score_func=mojolearn.r_regression, k=2),
     lambda: (_continuous(), _continuous(cols=1, seed=9)[:, 0]), False),
    ("kbest-mutual-info", SelectKBest, dict(score_func=mojolearn.mutual_info_classif, k=2),
     lambda: (_continuous(), _labels()), False),
    # the round imputer with its built-in regression; its initial SimpleImputer takes its device
    ("iterative-imputer", IterativeImputer, dict(max_iter=3, random_state=0), lambda: (_with_nan(),), False),
]
case = pytest.mark.parametrize("spec", CASES, ids=[row[0] for row in CASES])
# The rows whose CPU and GPU bits (transform, fit_transform, inverse_transform) are a verify lane's digest
# (x-prep-* in tools/identity_lanes/prep.py and m_transform.py) and whose vendor_used on both sides
# test_check_runs_both_devices_in_this_process reads are left out of the cross-device bits test.
DIGESTED = ("imputer-mean", "imputer-median", "kbins-ordinal", "ordinal", "onehot", "spline", "label-encoder",
            "label-binarizer", "iterative-imputer")
UNDIGESTED = [row for row in CASES if row[0] not in DIGESTED]
undigested_case = pytest.mark.parametrize("spec", UNDIGESTED, ids=[row[0] for row in UNDIGESTED])


def build(spec, **extra):
    return spec[1](**dict(spec[2], **extra))


def same_bits(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


@case
def test_device_is_a_constructor_parameter_with_default_none(spec):
    assert build(spec).get_params()["device"] is None
    assert build(spec, device=None).get_params()["device"] is None
    assert build(spec, device="cpu").get_params()["device"] == "cpu"
    assert spec[1]._parameters[-1] == "device"


@case
@pytest.mark.parametrize("bad", ["tpu", "GPU", " cpu", "", "None", 0, False, b"cpu"])
def test_unknown_device_is_refused_in_words_at_construction(spec, bad):
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None, got"):
        build(spec, device=bad)


@case
def test_explicit_gpu_runs_on_a_gpu_installation_and_is_refused_in_words_on_a_cpu_only_one(spec):
    if TWO_DEVICES:
        model = build(spec, device="gpu")
        model.fit(*spec[3]())
        assert model.get_params()["device"] == "gpu"
    else:
        with pytest.raises(ValueError, match="device='cpu'"):
            build(spec, device="gpu")


@case
def test_set_params_refuses_an_unknown_device_and_leaves_the_estimator_unchanged(spec):
    model = build(spec)
    before = model.get_params()
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None, got"):
        model.set_params(device="tpu")
    after = model.get_params()
    assert list(after) == list(before) and all(after[name] is before[name] for name in before)


@case
def test_cpu_device_fits_and_transforms(spec):
    args = spec[3]()
    model = build(spec, device="cpu").fit(*args)
    assert model.vendor_used() == "cpu"
    out = model.transform(args[0])
    assert np.asarray(out).shape[0] == len(args[0])
    if spec[4]:
        model.inverse_transform(out)


@two_devices
@undigested_case
def test_identical_mode_cpu_device_returns_the_gpu_bits(spec):
    args = spec[3]()
    gpu, cpu = build(spec), build(spec, device="cpu")
    # two estimators on two devices, interleaved in one process
    gpu.fit(*args)
    cpu.fit(*args)
    assert gpu.vendor_used() == mojolearn.vendor() != "cpu" and cpu.vendor_used() == "cpu"
    out_gpu, out_cpu = gpu.transform(args[0]), cpu.transform(args[0])
    assert same_bits(out_gpu, out_cpu)
    assert same_bits(build(spec).fit_transform(*args), build(spec, device="cpu").fit_transform(*args))
    if spec[4]:
        assert same_bits(gpu.inverse_transform(out_gpu), cpu.inverse_transform(out_cpu))


@two_devices
@case
def test_set_params_moves_the_estimator_to_the_other_device_with_the_same_bits(spec):
    args = spec[3]()
    model = build(spec).fit(*args)
    first = model.transform(args[0])
    model.set_params(device="cpu")
    assert model.get_params()["device"] == "cpu"
    assert same_bits(first, model.fit(*args).transform(args[0]))
    model.set_params(device=None)
    assert same_bits(first, model.fit(*args).transform(args[0]))


@case
@pytest.mark.parametrize("device", [None, "cpu"])
def test_clone_and_pickle_carry_the_device(spec, device):
    args = spec[3]()
    model = build(spec, device=device).fit(*args)
    out = model.transform(args[0])
    rebuilt = spec[1](**model.get_params())
    assert rebuilt.get_params()["device"] == device
    assert same_bits(out, rebuilt.fit(*args).transform(args[0]))
    restored = pickle.loads(pickle.dumps(model))
    assert restored.get_params()["device"] == device
    assert same_bits(out, restored.transform(args[0]))
    try:
        from sklearn.base import clone
    except ImportError:
        return
    cloned = clone(model)
    assert cloned.get_params()["device"] == device
    assert same_bits(out, cloned.fit(*args).transform(args[0]))


@case
def test_an_instance_without_the_parameter_runs_this_installations_device(spec):
    args = spec[3]()
    model = build(spec)
    del model.__dict__["device"]
    assert same_bits(model.fit(*args).transform(args[0]), build(spec).fit(*args).transform(args[0]))


def test_rfe_takes_device_and_hands_it_to_its_estimator():
    # 0e3056074e ("RFE: device="): RFE takes `device` as its last constructor
    # parameter, stored under its own name (the declared constructor writes it),
    # checked at construction with the estimators' words, and handed to the
    # estimator it fits; an estimator constructed with a different explicit
    # device is refused before anything runs (core/function_device.mojo
    # `function_device_member`).
    assert mojolearn.RFE._parameters[-1] == "device"
    assert mojolearn.RFE(mojolearn.Ridge()).get_params()["device"] is None
    assert mojolearn.RFE(mojolearn.Ridge(), device="cpu").get_params()["device"] == "cpu"
    with pytest.raises(ValueError, match="device must be 'gpu', 'cpu' or None, got"):
        mojolearn.RFE(mojolearn.Ridge(), device="tpu")
    x, y = _continuous(), _continuous(cols=1).ravel()
    model = mojolearn.RFE(mojolearn.Ridge(), n_features_to_select=2, device="cpu").fit(x, y)
    assert sum(model.support_) == 2  # support_ is a list of bools (the select door stores list(support))
    if TWO_DEVICES:
        with pytest.raises(ValueError, match="give both the same device"):
            mojolearn.RFE(mojolearn.Ridge(device="gpu"), n_features_to_select=2, device="cpu").fit(x, y)


@pytest.mark.parametrize("device", [None, "cpu"])
def test_iterative_imputer_hands_its_device_to_its_initial_imputer(device):
    x = _with_nan()
    model = IterativeImputer(max_iter=2, random_state=0, device=device).fit(x)
    initial = model.initial_imputer_
    assert initial.get_params()["device"] == device
    assert initial.vendor_used() == model.vendor_used()
    # the initial imputer on its own, then the owner again: each resolves its own route
    assert same_bits(initial.transform(x), SimpleImputer(device=device).fit(x).transform(x))
    assert same_bits(model.transform(x), IterativeImputer(max_iter=2, random_state=0, device=device).fit_transform(x))


@pytest.mark.parametrize("device", [None, "cpu"])
def test_select_k_best_fit_and_transform_run_on_one_device(device):
    # fit is the encode owner's call and transform the selector door's: both resolve the estimator's device
    x, y = _continuous(), _labels()
    model = SelectKBest(mojolearn.f_classif, k=2, device=device).fit(x, y)
    expected = "cpu" if device == "cpu" else mojolearn.vendor()
    assert model.vendor_used() == expected
    kept = model.transform(x)
    assert same_bits(kept, np.asarray(x)[:, np.asarray(model.get_support())])
    assert model.vendor_used() == expected
    assert same_bits(kept, SelectKBest(mojolearn.f_classif, k=2, device=device).fit_transform(x, y))


@two_devices
@case
def test_check_runs_both_devices_in_this_process(spec):
    if mojolearn.numeric_mode() != "identical" or len(spec[3]()) != 1 or spec[1] is MultiLabelBinarizer:
        pytest.skip("the check takes one array input in identical mode")
    args = spec[3]()
    model = build(spec).fit(*args)
    result = model.check(args[0])
    assert result["ran"] and result["transfer"] == "device"
    assert result["sides"]["cpu"]["route"] == dict(device="cpu", vendor_used="cpu")
    assert result["sides"]["gpu"]["route"] == dict(device=None, vendor_used=mojolearn.vendor())
    assert result["verdict"]["match"]
    assert model.get_params()["device"] is None
