# SPDX-License-Identifier: Apache-2.0
"""LabelPropagation / LabelSpreading and the chi2 samplers: each public method
is one native call (bindings/neighbors_semi_native.mojo) whose outputs and
fitted state equal the former Python path's bit for bit
(`_legacy_neighbors_semi`, frozen at b32103c00), with the same refusals."""
import struct

import pytest

np = pytest.importorskip("numpy")

import mojolearn
from mojolearn.tests import _legacy_neighbors_semi as legacy


def _bits(value):
    """Exact comparison key of an output: arrays by dtype, shape and bytes."""
    if isinstance(value, (list, tuple)):
        return [_bits(v) for v in value]
    if isinstance(value, float):
        return struct.pack("<d", value)
    if hasattr(value, "toarray") and hasattr(value, "nnz"):
        csr = value.tocsr()
        return ("csr", csr.shape, _bits(csr.data), _bits(csr.indices), _bits(csr.indptr))
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        a = np.asarray(value)
        return (str(a.dtype), a.shape, a.tobytes())
    return value


def _state(est, names):
    d = vars(est)
    return {k: _bits(d[k]) for k in names if k in d}


def _data(n=40, d=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((n, d), dtype=np.float32)
    y = (X[:, 0] > 0.5).astype(np.int64)
    y[n // 2:] = -1
    return X, y


def _audit(call, *args):
    from mojolearn._verify_native_runtime import audit_native_call
    return audit_native_call(call, *args).assert_native()


_LP_STATE = ("X_", "classes_", "label_distributions_", "n_iter_", "transduction_", "n_features_in_",
             "_classical_raw_graph")


@pytest.mark.parametrize("cls", [mojolearn.LabelPropagation, mojolearn.LabelSpreading])
@pytest.mark.parametrize("kernel", ["rbf", "knn"])
@pytest.mark.parametrize("labels", ["int", "str"])
def test_label_propagation_methods_match_former_path(cls, kernel, labels):
    X, y = _data()
    if labels == "str":
        y = ["u" if v == -1 else ("a" if v else "b") for v in y.tolist()]
    params = dict(kernel=kernel, gamma=2.0, n_neighbors=5)
    native, old = cls(**params), cls(**params)
    assert native.fit(X, y) is native
    legacy.lp_fit(old, X, y)
    assert _state(native, _LP_STATE) == _state(old, _LP_STATE)
    Q = X[:7] + np.float32(0.01)
    assert _bits(native.predict_proba(Q)) == _bits(legacy.lp_predict_proba(old, Q))
    assert _bits(native.predict(Q)) == _bits(legacy.lp_predict(old, Q))
    yq = y[:7]
    w = np.linspace(0.5, 1.5, 7)
    assert _bits(native.score(Q, yq)) == _bits(legacy.lp_score(old, Q, yq))
    assert _bits(native.score(Q, yq, w)) == _bits(legacy.lp_score(old, Q, yq, w))


@pytest.mark.parametrize("max_iter", [0, 1, 3])
def test_label_propagation_iteration_counts_match(max_iter):
    X, y = _data(seed=3)
    for cls in (mojolearn.LabelPropagation, mojolearn.LabelSpreading):
        native, old = cls(kernel="knn", max_iter=max_iter), cls(kernel="knn", max_iter=max_iter)
        native.fit(X, y)
        legacy.lp_fit(old, X, y)
        assert _state(native, _LP_STATE) == _state(old, _LP_STATE)


def test_label_propagation_is_one_native_call():
    X, y = _data()
    est = mojolearn.LabelSpreading(kernel="knn")
    _audit(est.fit, X, y)
    _audit(est.predict_proba, X[:3])


@pytest.mark.parametrize("alpha", [0.0, 1.0, float("nan")])
def test_label_spreading_alpha_refusal(alpha):
    X, y = _data()
    with pytest.raises(ValueError, match=r"^alpha must be in \(0, 1\)$"):
        mojolearn.LabelSpreading(alpha=alpha).fit(X, y)


def test_label_propagation_kernel_refusal():
    X, y = _data()
    with pytest.raises(NotImplementedError) as native:
        mojolearn.LabelPropagation(kernel="poly").fit(X, y)
    with pytest.raises(NotImplementedError) as old:
        legacy.lp_fit(mojolearn.LabelPropagation(kernel="poly"), X, y)
    assert str(native.value) == str(old.value)


def test_label_propagation_row_mismatch_refusal():
    X, y = _data()
    with pytest.raises(ValueError, match="X and y have different numbers of rows"):
        mojolearn.LabelPropagation().fit(X, y[:-1])


@pytest.mark.parametrize("steps,interval", [(1, None), (2, None), (3, None), (4, 0.3)])
def test_additive_chi2_methods_match_former_path(steps, interval):
    X, _ = _data()
    native = mojolearn.AdditiveChi2Sampler(sample_steps=steps, sample_interval=interval)
    old = mojolearn.AdditiveChi2Sampler(sample_steps=steps, sample_interval=interval)
    assert native.fit(X) is native
    legacy.achi2_fit(old, X)
    assert vars(native) == vars(old)
    assert _bits(native.transform(X)) == _bits(legacy.achi2_transform(old, X))
    assert _bits(mojolearn.AdditiveChi2Sampler(sample_steps=steps, sample_interval=interval).fit_transform(X)) == \
        _bits(legacy.achi2_fit_transform(mojolearn.AdditiveChi2Sampler(sample_steps=steps, sample_interval=interval), X))


def test_additive_chi2_sparse_output_matches_former_path():
    sp = pytest.importorskip("scipy.sparse")
    X, _ = _data()
    X[X < 0.5] = 0
    S = sp.csr_matrix(X)
    native = mojolearn.AdditiveChi2Sampler().fit(S)
    old = legacy.achi2_fit(mojolearn.AdditiveChi2Sampler(), S)
    assert _bits(native.transform(S)) == _bits(legacy.achi2_transform(old, S))


def test_additive_chi2_refusals():
    X, _ = _data()
    with pytest.raises(ValueError, match="^Negative values in data passed to AdditiveChi2Sampler$"):
        mojolearn.AdditiveChi2Sampler().fit(-X)
    with pytest.raises(ValueError, match=r"^If sample_steps is not in \[1, 2, 3\], you need to provide sample_interval$"):
        mojolearn.AdditiveChi2Sampler(sample_steps=5).fit(X)
    with pytest.raises(ValueError, match="^Negative values in data passed to AdditiveChi2Sampler$"):
        mojolearn.AdditiveChi2Sampler().transform(-X)
    _audit(mojolearn.AdditiveChi2Sampler().fit, X)


_SCHI2_STATE = ("_random_weights", "random_offset_", "n_features_in_")


@pytest.mark.parametrize("seed", [0, 7, -3, 2 ** 40])
def test_skewed_chi2_methods_match_former_path(seed):
    X, _ = _data()
    params = dict(skewedness=0.5, n_components=16, random_state=seed)
    native, old = mojolearn.SkewedChi2Sampler(**params), mojolearn.SkewedChi2Sampler(**params)
    assert native.fit(X) is native
    legacy.schi2_fit(old, X)
    assert _state(native, _SCHI2_STATE) == _state(old, _SCHI2_STATE)
    assert _bits(native.transform(X)) == _bits(legacy.schi2_transform(old, X))
    assert _bits(mojolearn.SkewedChi2Sampler(**params).fit_transform(X)) == \
        _bits(legacy.schi2_fit_transform(mojolearn.SkewedChi2Sampler(**params), X))


def test_skewed_chi2_refusals():
    X, _ = _data()
    with pytest.raises(ValueError, match="cannot be used to seed a numpy.random.RandomState instance"):
        mojolearn.SkewedChi2Sampler(random_state="seed").fit(X)
    est = mojolearn.SkewedChi2Sampler(skewedness=0.5, random_state=0).fit(X)
    with pytest.raises(ValueError, match=r"^X may not contain entries smaller than -skewedness\.$"):
        est.transform(X - 1)
    _audit(est.transform, X)
