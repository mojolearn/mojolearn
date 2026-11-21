# SPDX-License-Identifier: Apache-2.0
"""`device=` on the preprocessing estimators, at the Python boundary: the
fit-level rows (the construction, refusal, clone and pickle rows of every
class are test_device_parameter.py).

One estimator runs on the device its constructor names: None is this
installation's device, "cpu" runs the host image beside a GPU set, in the
same process, and returns the same bits in identical mode. What needs two
devices is skipped by name on a CPU-only installation.
"""
import numpy as np
import pytest

import mojolearn
from mojolearn import (
    BernoulliNB, Binarizer, CategoricalNB, ComplementNB, GaussianNB, LinearDiscriminantAnalysis, MultinomialNB,
    PolynomialFeatures, PowerTransformer, QuadraticDiscriminantAnalysis, QuantileTransformer, VarianceThreshold,
)

TWO_DEVICES = mojolearn.vendor() != "cpu"
two_devices = pytest.mark.skipif(not TWO_DEVICES, reason="a CPU-only installation has one device")


def data(rows=300, cols=6, seed=11):
    x = np.random.default_rng(seed).standard_normal((rows, cols)).astype(np.float32)
    x[:, 0] *= np.float32(0.01)                               # a low-variance column for the selector
    return x


#: (id, class, constructor parameters, fitted attribute names compared as bytes, has inverse_transform)
CASES = [
    ("binarizer", Binarizer, dict(threshold=0.1), (), False),
    ("polynomial", PolynomialFeatures, dict(degree=2), ("powers_",), False),
    ("polynomial-f", PolynomialFeatures, dict(degree=3, interaction_only=True, order="F"), ("powers_",), False),
    ("variance", VarianceThreshold, dict(threshold=0.05), ("variances_",), False),
    ("quantile", QuantileTransformer, dict(n_quantiles=64), ("quantiles_", "references_"), True),
    ("quantile-normal", QuantileTransformer, dict(n_quantiles=64, output_distribution="normal", subsample=200,
                                                  random_state=3), ("quantiles_", "references_"), True),
    ("power", PowerTransformer, dict(), ("lambdas_",), True),
]
case = pytest.mark.parametrize("spec", CASES, ids=[row[0] for row in CASES])


def build(spec, **extra):
    return spec[1](**dict(spec[2], **extra))


def outputs(spec, model, x):
    out = {name: np.asarray(getattr(model, name)).tobytes() for name in spec[3]}
    t = np.asarray(model.transform(x))
    out["transform"] = (t.shape, t.tobytes())
    if spec[4]:
        back = np.asarray(model.inverse_transform(t))
        out["inverse_transform"] = (back.shape, back.tobytes())
    return out


@case
def test_default_device_is_this_installations_device(spec):
    x = data()
    model = build(spec).fit(x)
    assert model.get_params()["device"] is None               # what was passed, not what it resolved to
    assert np.asarray(model.transform(x)).shape[0] == len(x)


@case
def test_explicit_cpu_always_runs(spec):
    x = data()
    model = build(spec, device="cpu")
    assert np.asarray(model.fit_transform(x)).shape[0] == len(x)
    assert model.get_params()["device"] == "cpu"


@case
def test_explicit_gpu_runs_on_a_gpu_installation_and_is_refused_in_words_on_a_cpu_only_one(spec):
    x = data()
    if TWO_DEVICES:
        assert np.asarray(build(spec, device="gpu").fit(x).transform(x)).shape[0] == len(x)
    else:
        with pytest.raises((ValueError, RuntimeError, ImportError), match="gpu"):
            build(spec, device="gpu").fit(x)


@two_devices
@case
def test_cpu_device_returns_the_gpu_bits(spec):
    x = data()
    gpu = build(spec, device="gpu").fit(x)
    cpu = build(spec, device="cpu").fit(x)
    left, right = outputs(spec, gpu, x), outputs(spec, cpu, x)
    assert left.keys() == right.keys()
    for name in left:
        assert left[name] == right[name], name


@two_devices
@case
def test_two_estimators_on_two_devices_interleave_in_one_process(spec):
    x = data()
    gpu, cpu = build(spec, device="gpu"), build(spec, device="cpu")
    for _ in range(3):
        gpu.fit(x), cpu.fit(x)
        assert outputs(spec, gpu, x) == outputs(spec, cpu, x)


@two_devices
@case
def test_a_changed_device_resolves_again(spec):
    x = data()
    model = build(spec).fit(x)
    first = outputs(spec, model, x)
    model.set_params(device="cpu")
    assert model.get_params()["device"] == "cpu"
    assert outputs(spec, model, x) == first                   # the fitted model now answers on the CPU image
    model.set_params(device=None)
    assert outputs(spec, model, x) == first


@two_devices
@case
def test_a_model_fitted_on_one_device_transforms_on_the_other(spec):
    from mojolearn import _check_both
    x = data()
    model = build(spec).fit(x)
    assert _check_both.takes_device(model)
    twin = _check_both.on_device(model, "cpu")
    assert type(twin) is spec[1] and twin is not model and twin.get_params()["device"] == "cpu"
    assert outputs(spec, twin, x) == outputs(spec, model, x)  # the same fitted arrays, in memory: no file


@two_devices
@case
def test_check_runs_the_same_estimator_on_both_devices_in_this_process(spec, capsys):
    x = data()
    model = build(spec).fit(x)
    result = model.check(x)
    capsys.readouterr()
    assert result["ran"] and result["transfer"] == "device"   # no save, no host_model, no child process
    assert result["verdict"]["match"]


# ---------------------------------------------------------------- classifiers
def labelled(rows=360, cols=6, classes=3, seed=7, counts=False):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, classes, size=rows).astype(np.int32)
    if counts:                                                # non-negative integer features (the discrete NB family)
        x = rng.poisson(2.0 + y[:, None], size=(rows, cols)).astype(np.float32)
    else:
        x = (rng.standard_normal((rows, cols)) + y[:, None]).astype(np.float32)
    return x, y


#: (id, class, constructor parameters, count features, fitted attribute names compared as bytes, methods)
CLASSIFIERS = [
    ("gnb", GaussianNB, dict(), False, ("theta_", "var_", "class_prior_"), ("predict", "predict_proba")),
    ("mnb", MultinomialNB, dict(), True, ("feature_log_prob_", "class_log_prior_"), ("predict", "predict_proba")),
    ("bnb", BernoulliNB, dict(binarize=2.0), True, ("feature_log_prob_", "class_log_prior_"),
     ("predict", "predict_proba")),
    ("cnb", ComplementNB, dict(), True, ("feature_log_prob_",), ("predict",)),
    ("cat", CategoricalNB, dict(), True, ("class_log_prior_",), ("predict", "predict_proba")),
    ("lda", LinearDiscriminantAnalysis, dict(), False, ("coef_", "intercept_", "means_"),
     ("predict", "predict_proba", "decision_function", "transform")),
    ("qda", QuadraticDiscriminantAnalysis, dict(), False, ("means_",), ("predict", "predict_proba")),
]
classifier = pytest.mark.parametrize("spec", CLASSIFIERS, ids=[row[0] for row in CLASSIFIERS])


def answers(spec, model, x, y):
    out = {name: np.asarray(getattr(model, name)).tobytes() for name in spec[4]}
    for name in spec[5]:
        value = np.asarray(getattr(model, name)(x))
        out[name] = (value.shape, value.tobytes())
    out["score"] = float(model.score(x, y))
    return out


@classifier
def test_classifier_default_and_explicit_cpu_run(spec):
    x, y = labelled(counts=spec[3])
    assert spec[1](**spec[2]).fit(x, y).get_params()["device"] is None
    model = spec[1](**dict(spec[2], device="cpu")).fit(x, y)
    assert model.get_params()["device"] == "cpu"
    assert np.asarray(model.predict(x)).shape[0] == len(x)


@two_devices
@classifier
def test_classifier_cpu_device_returns_the_gpu_bits(spec):
    x, y = labelled(counts=spec[3])
    gpu = spec[1](**dict(spec[2], device="gpu")).fit(x, y)
    cpu = spec[1](**dict(spec[2], device="cpu")).fit(x, y)
    left, right = answers(spec, gpu, x, y), answers(spec, cpu, x, y)
    assert left.keys() == right.keys()
    for name in left:
        assert left[name] == right[name], name


@two_devices
@classifier
def test_classifiers_on_two_devices_interleave_in_one_process(spec):
    x, y = labelled(counts=spec[3])
    gpu, cpu = spec[1](**dict(spec[2], device="gpu")), spec[1](**dict(spec[2], device="cpu"))
    for _ in range(3):
        gpu.fit(x, y), cpu.fit(x, y)
        assert answers(spec, gpu, x, y) == answers(spec, cpu, x, y)


@two_devices
@classifier
def test_classifier_changed_device_resolves_again(spec):
    x, y = labelled(counts=spec[3])
    model = spec[1](**spec[2]).fit(x, y)
    first = answers(spec, model, x, y)
    model.set_params(device="cpu")
    assert model.get_params()["device"] == "cpu"
    assert answers(spec, model, x, y) == first
    model.set_params(device="gpu")                            # checked against the installation, not the CPU route
    assert answers(spec, model, x, y) == first


@two_devices
def test_gaussian_nb_cpu_device_refuses_fast_in_words():
    x, y = labelled()
    with pytest.raises((ValueError, ImportError), match="identical"):
        GaussianNB(device="cpu", numeric_mode="fast").fit(x, y)


@two_devices
@classifier
def test_classifier_check_runs_both_devices_in_this_process(spec, capsys):
    x, y = labelled(counts=spec[3])
    model = spec[1](**spec[2]).fit(x, y)
    result = model.check(x)
    capsys.readouterr()
    assert result["ran"] and result["transfer"] == "device"   # no save, no host_model, no child process
    assert result["verdict"]["match"]
