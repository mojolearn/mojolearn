# SPDX-License-Identifier: Apache-2.0
"""KernelDensity `fit`, `score_samples` and `score` are one native call each
(bindings/neighbors_kde_native.mojo, `family_xn_kde_*`): the stored state,
the scores and every refusal equal the former Python path
(`_legacy_neighbors_kde`, frozen at b32103c00) bit for bit."""
import math

import numpy as np
import pytest

from mojolearn import Array, KernelDensity
from mojolearn import density
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests import _legacy_neighbors_kde as legacy

KERNELS = ("gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine")
METRICS = ("euclidean", "manhattan", "chebyshev", "sqeuclidean")


def _rows(n=37, d=5, seed=3, dtype="<f4"):
    return np.random.default_rng(seed).standard_normal((n, d)).astype(dtype)


def _layout(x, layout):
    if layout == "f32":
        return x
    if layout == "f64":
        return x.astype("<f8")
    if layout == "fortran":
        return np.asfortranarray(x)
    if layout == "strided":
        return np.repeat(x, 2, axis=1)[:, ::2]
    if layout == "array":
        return Array.from_buffer(np.ascontiguousarray(x))
    if layout == "list":
        return x.astype("<f8").tolist()
    if layout == "sparse":
        sparse = pytest.importorskip("scipy.sparse")
        y = x.copy()
        y[np.abs(y) < 0.5] = 0.0
        return sparse.csr_matrix(y)
    raise AssertionError(layout)


def _weights(kind, n):
    if kind is None:
        return None
    if kind == "scalar":
        return 2.5
    w = np.linspace(0.25, 3.0, n).astype("<f8")
    return w.tolist() if kind == "list" else w


def _bytes(value):
    return None if value is None else np.ascontiguousarray(np.asarray(value)).tobytes()


def _pair(params, x, w):
    native = KernelDensity(**params).fit(x, sample_weight=w)
    old = KernelDensity(**params)
    legacy.fit(old, x, None, w)
    return native, old


def _state(model):
    fields = vars(model)
    return (fields["input_copied_"], fields["n_features_in_"], fields["n_samples_fit_"],
            _bytes(fields["_x"]), _bytes(fields["_w"]))


@pytest.mark.parametrize("layout", ["f32", "f64", "fortran", "strided", "array", "list", "sparse"])
@pytest.mark.parametrize("weights", [None, "scalar", "list", "array"])
def test_fit_state_equals_former_path(layout, weights):
    x = _rows()
    data = _layout(x, layout)
    native, old = _pair({}, data, _weights(weights, x.shape[0]))
    assert _state(native) == _state(old)


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("weights", [None, "array"])
def test_scores_equal_former_path(kernel, metric, weights):
    x = _rows()
    q = _rows(n=11, seed=9)
    params = dict(kernel=kernel, metric=metric, bandwidth=0.75)
    native, old = _pair(params, x, _weights(weights, x.shape[0]))
    # twice each: the first call prepares the resident fit, the second reuses it
    for _ in range(2):
        expected = np.asarray(legacy.scores(old, q, False))
        actual = np.asarray(native.score_samples(q))
        assert actual.dtype == expected.dtype == np.dtype("<f4")
        assert actual.tobytes() == expected.tobytes()
        total, want = native.score(q), legacy.scores(old, q, True)
        assert type(total) is float and math.copysign(1.0, total) == math.copysign(1.0, want)
        assert total == want or (math.isnan(total) and math.isnan(want))


def test_resident_key_change_revalidates_like_former_path():
    x, q = _rows(), _rows(n=4, seed=5)
    native, old = _pair(dict(bandwidth=0.5), x, None)
    native.score_samples(q)
    legacy.scores(old, q, False)
    for model in (native, old):
        model.bandwidth = 1.25
    assert np.asarray(native.score_samples(q)).tobytes() == np.asarray(legacy.scores(old, q, False)).tobytes()
    for model in (native, old):
        model.bandwidth = "scott"
    errors = []
    for call in (lambda: native.score_samples(q), lambda: legacy.scores(old, q, False)):
        with pytest.raises(NotImplementedError) as info:
            call()
        errors.append(str(info.value))
    assert errors[0] == errors[1]


@pytest.mark.parametrize("change", [
    ("bandwidth", -1.0), ("bandwidth", float("nan")), ("algorithm", "kd_tree"), ("atol", 1e-3),
    ("rtol", 1), ("breadth_first", False), ("leaf_size", 30), ("kernel", 3),
    ("metric_params", {"p": 2.0, "q": 1.0}), ("metric_params", {"p": 3.0}),
])
def test_validate_refusals_equal_former_path(change):
    x, q = _rows(), _rows(n=3, seed=6)
    native, old = _pair({}, x, None)
    native.score_samples(q)
    legacy.scores(old, q, False)
    seen = []
    for model, call in ((native, lambda: native.score_samples(q)), (old, lambda: legacy.scores(old, q, False))):
        setattr(model, change[0], change[1])
        with pytest.raises(Exception) as info:
            call()
        seen.append((type(info.value), str(info.value)))
    assert seen[0] == seen[1]


def _refusal(call):
    with pytest.raises(Exception) as info:
        call()
    return type(info.value), str(info.value)


@pytest.mark.parametrize("case", ["nan", "negative-weight", "zero-weight", "short-weight", "weight-2d", "empty", "1d"])
def test_fit_refusals_equal_former_path(case):
    x = _rows()
    w = None
    if case == "nan":
        x = x.copy()
        x[2, 1] = np.nan
    elif case == "negative-weight":
        w = -np.ones(x.shape[0])
    elif case == "zero-weight":
        w = np.zeros(x.shape[0])
    elif case == "short-weight":
        w = np.ones(x.shape[0] - 1)
    elif case == "weight-2d":
        w = np.ones((x.shape[0], 1))
    elif case == "empty":
        x = np.zeros((0, 3), "<f4")
    elif case == "1d":
        x = x[:, 0]
    native = _refusal(lambda: KernelDensity().fit(x, sample_weight=w))
    old = _refusal(lambda: legacy.fit(KernelDensity(), x, None, w))
    assert native == old


def test_score_refusals_equal_former_path():
    q = _rows(n=3, d=4)
    assert _refusal(lambda: KernelDensity().score_samples(q)) == \
        _refusal(lambda: legacy.scores(KernelDensity(), q, False))
    native, old = _pair({}, _rows(), None)
    assert _refusal(lambda: native.score(q)) == _refusal(lambda: legacy.scores(old, q, True))


def test_fit_and_score_doors_run_no_python():
    x, q = _rows(), _rows(n=6, seed=8)
    model = KernelDensity(bandwidth=0.5)
    binding = model._bind("_mojolearn_estimators")
    door = density._NATIVE
    assert audit_native_call(door, model, "family_xn_kde_fit", (binding, x, None),
                             density._NATIVE_CONTEXT).assert_native() is model
    scores = audit_native_call(door, model, "family_xn_kde_score_samples", (binding, q),
                               density._NATIVE_CONTEXT).assert_native()
    total = audit_native_call(door, model, "family_xn_kde_score", (binding, q),
                              density._NATIVE_CONTEXT).assert_native()
    old = KernelDensity(bandwidth=0.5)
    legacy.fit(old, x)
    assert np.asarray(scores).tobytes() == np.asarray(legacy.scores(old, q, False)).tobytes()
    assert total == legacy.scores(old, q, True)
    assert audit_native_call(door, model, "family_xn_kde_release", (), density._NATIVE_CONTEXT).assert_native() is None
    assert vars(model).get("_resident") is None
