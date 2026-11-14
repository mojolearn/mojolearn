# SPDX-License-Identifier: Apache-2.0
"""Query-door boundary of the classical infer rows (lane fix-classical-infer-xq).

Each public predict/transform is ONE call into its compiled owner, and the PCA
query and fit doors consult the model-selection fold provider only for a
DeviceRows fold: a plain array never reaches it (the per-call provider probe
the wave-9 doors ran and 0.8.66 did not)."""
import numpy as np
import pytest
import mojolearn as ml


def _x(rows=64, cols=6, seed=3):
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(rng.standard_normal((rows, cols)), dtype=np.float32)


def _counting(monkeypatch, module, name):
    calls = []
    entry = getattr(module, name)

    def counted(*args):
        calls.append(args[1])
        return entry(*args)

    monkeypatch.setattr(module, name, counted)
    return calls


def test_pca_transform_is_one_native_call(monkeypatch):
    from mojolearn import decomposition

    x = _x()
    model = ml.PCA(n_components=3, svd_solver="covariance_eigh", numeric_mode="identical").fit(x)
    calls = _counting(monkeypatch, decomposition, "_PCA_ENTRY")
    z = np.asarray(model.transform(x))
    assert calls == ["transform"]
    assert z.shape == (64, 3)
    assert z.dtype == np.float32 and z.flags["C_CONTIGUOUS"] and z.flags["WRITEABLE"]


def test_pca_transform_output_is_fully_written():
    # The output is allocated uninitialised: every word must come from the
    # projection, so two calls (fresh allocations) give the same bits.
    x = _x(rows=257, cols=5)
    model = ml.PCA(n_components=4, svd_solver="covariance_eigh", numeric_mode="identical").fit(x)
    first = np.asarray(model.transform(x)).copy()
    second = np.asarray(model.transform(x))
    assert np.isfinite(first).all()
    assert first.tobytes() == second.tobytes()


@pytest.mark.parametrize("operation", ["fit", "transform"])
def test_pca_plain_array_never_consults_the_fold_provider(operation, monkeypatch):
    from mojolearn.decomposition import _PCA_CONTEXT

    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("fold provider consulted for a plain array")

    x = _x()
    model = ml.PCA(n_components=3, svd_solver="covariance_eigh", numeric_mode="identical").fit(x)
    monkeypatch.setitem(_PCA_CONTEXT, "pca_fold_factory", forbidden)
    if operation == "fit":
        model.fit(x)
    else:
        assert np.asarray(model.transform(x)).shape == (64, 3)
    assert calls == []


def test_pca_nonfinite_query_still_refused():
    x = _x()
    model = ml.PCA(n_components=3, svd_solver="covariance_eigh", numeric_mode="identical").fit(x)
    bad = x.copy()
    bad[5, 2] = np.nan
    with pytest.raises(Exception, match="NaN or infinity"):
        model.transform(bad)


@pytest.mark.parametrize("method", ["predict", "decision_function"])
def test_svc_query_is_one_native_call(method, monkeypatch):
    from mojolearn import _svm_impl

    x = _x(rows=80)
    y = (x[:, 0] + 0.25 * x[:, 1] > 0).astype(np.float32)
    model = ml.SVC(C=1.0, kernel="rbf", gamma=0.2).fit(x, y)
    getattr(model, method)(x)  # the first call takes the resident lease
    calls = _counting(monkeypatch, _svm_impl, "_NATIVE")
    out = np.asarray(getattr(model, method)(x))
    assert calls == ["family_svc_svc_" + ("predict" if method == "predict" else "decision")]
    assert out.shape == (80,)
