# SPDX-License-Identifier: Apache-2.0
"""PageRank.fit, Louvain.fit/fit_predict, connected_components and
SVGP.fit/predict/score are one native call each
(bindings/neighbors_graph_native.mojo); every output, fitted attribute and
refusal equals the former Python path's (tests/_legacy_neighbors_graph.py),
bit for bit."""
import numpy as np
import pytest

import mojolearn
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests import _legacy_neighbors_graph as legacy


def _bits(value):
    """A comparable record of an output: arrays by dtype, shape and bytes,
    floats by their binary64 bits, tuples elementwise."""
    if isinstance(value, tuple):
        return tuple(_bits(v) for v in value)
    if isinstance(value, float):
        return ("float", np.float64(value).tobytes())
    if hasattr(value, "__array__") and not isinstance(value, (int, str)):
        a = np.asarray(value)
        return (a.dtype.str, a.shape, a.tobytes())
    return value


def _state(model, names):
    return {k: _bits(getattr(model, k)) for k in names if hasattr(model, k)}


def _graph(n=40, seed=0, symmetric=True, density=0.12):
    rng = np.random.default_rng(seed)
    m = (rng.random((n, n)) < density)
    if symmetric:
        m = m | m.T
    np.fill_diagonal(m, False)
    return (m * rng.random((n, n))).astype(np.float32) if not symmetric else m.astype(np.float32)


def _csr(a):
    sparse = pytest.importorskip("scipy.sparse")
    return sparse.csr_matrix(a)


def _tuple(a):
    s = _csr(a)
    return (s.indptr.copy(), s.indices.copy(), a.shape[0])


def _outcome(fn, *args, **kwargs):
    try:
        return ("ok", fn(*args, **kwargs))
    except Exception as error:  # noqa: BLE001 - the refusal itself is compared
        return ("raised", type(error), str(error))


PR_ATTRS = ("pagerank_", "n_iter_")


@pytest.mark.parametrize("layout", ["dense", "csr", "tuple"])
@pytest.mark.parametrize("params", [
    {}, {"weight": False}, {"weight": None}, {"alpha": 0.5, "tol": 1e-4},
    {"personalization": "ramp"}, {"nstart": "ramp", "dangling": "ramp"},
])
def test_pagerank_fit_matches_former_path(layout, params):
    a = _graph(symmetric=False)
    n = a.shape[0]
    params = {k: (np.arange(1, n + 1, dtype=np.float32) if v == "ramp" else v) for k, v in params.items()}
    graph = {"dense": a, "csr": _csr(a) if layout == "csr" else None, "tuple": _tuple(a) if layout == "tuple" else None}[layout]
    native = mojolearn.PageRank(**params)
    old = mojolearn.PageRank(**params)
    got = audit_native_call(native.fit, graph).assert_native()
    assert got is native
    legacy.pagerank_fit(old, graph)
    assert _state(native, PR_ATTRS) == _state(old, PR_ATTRS)


@pytest.mark.parametrize("params,graph", [
    ({"max_iter": 1, "tol": 1e-12}, "dense"),
    ({"personalization": np.zeros(40, np.float32)}, "dense"),
    ({"personalization": np.ones(3, np.float32)}, "dense"),
    ({}, "rect"),
    ({}, "bad_tuple"),
])
def test_pagerank_refusals_match_former_path(params, graph):
    a = _graph(symmetric=False)
    source = {"dense": a, "rect": a[:, :-1],
              "bad_tuple": (np.arange(5, dtype=np.int32), np.arange(4, dtype=np.int32), 40)}[graph]
    native = _outcome(mojolearn.PageRank(**params).fit, source)
    old = _outcome(legacy.pagerank_fit, mojolearn.PageRank(**params), source)
    assert native[0] == old[0]
    if native[0] == "raised":
        assert native == old


LV_ATTRS = ("labels_", "n_communities_", "modularity_", "n_levels_")


@pytest.mark.parametrize("layout", ["dense", "csr", "tuple"])
@pytest.mark.parametrize("params", [{}, {"resolution": 0.5}, {"max_level": 1}, {"threshold": 1e-3}])
def test_louvain_fit_and_fit_predict_match_former_path(layout, params):
    a = _graph()
    graph = a if layout == "dense" else _csr(a) if layout == "csr" else _tuple(a)
    native, old = mojolearn.Louvain(**params), mojolearn.Louvain(**params)
    assert audit_native_call(native.fit, graph).assert_native() is native
    legacy.louvain_fit(old, graph)
    assert _state(native, LV_ATTRS) == _state(old, LV_ATTRS)
    native, old = mojolearn.Louvain(**params), mojolearn.Louvain(**params)
    got = audit_native_call(native.fit_predict, graph).assert_native()
    assert _bits(got) == _bits(legacy.louvain_fit_predict(old, graph))
    assert _state(native, LV_ATTRS) == _state(old, LV_ATTRS)


@pytest.mark.parametrize("params,graph", [
    ({"max_level": 0}, "dense"), ({"max_level": -2}, "dense"),
    ({}, "asymmetric"), ({}, "empty"), ({}, "asymmetric_csr"),
])
def test_louvain_refusals_match_former_path(params, graph):
    a = _graph()
    source = {"dense": a, "asymmetric": _graph(symmetric=False), "empty": np.zeros((6, 6), np.float32),
              "asymmetric_csr": None}[graph]
    if graph == "asymmetric_csr":
        source = _csr(_graph(symmetric=False))
    assert _outcome(mojolearn.Louvain(**params).fit, source) == _outcome(
        legacy.louvain_fit, mojolearn.Louvain(**params), source)


@pytest.mark.parametrize("layout", ["dense", "csr", "tuple"])
@pytest.mark.parametrize("directed", [True, False])
@pytest.mark.parametrize("return_labels", [True, False])
def test_connected_components_matches_former_path(layout, directed, return_labels):
    a = _graph(n=60, density=0.02, symmetric=False)
    graph = a if layout == "dense" else _csr(a) if layout == "csr" else _tuple(a)
    got = audit_native_call(mojolearn.connected_components, graph, directed=directed,
                            return_labels=return_labels).assert_native()
    want = legacy.connected_components(graph, directed=directed, return_labels=return_labels)
    assert _bits(got) == _bits(want)


@pytest.mark.parametrize("kwargs", [
    {"connection": "strong"}, {"connection": "strong", "directed": False}, {"connection": "medium"},
])
def test_connected_components_refusals_match_former_path(kwargs):
    a = _graph(n=10)
    assert _bits(_outcome(mojolearn.connected_components, a, **kwargs)) == _bits(
        _outcome(legacy.connected_components, a, **kwargs))


SV_ATTRS = ("Z_", "_alpha", "_C", "q_mu_", "q_sqrt_", "elbo_", "precision_", "n_features_in_")


def _regression(n=120, d=3, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.random((n, d), dtype=np.float32)
    y = np.sin(6 * x[:, 0]).astype(np.float32)
    return x, y


@pytest.mark.parametrize("params", [
    {"n_inducing": 16, "lengthscale": 0.3, "noise_variance": 0.01}, {"n_inducing": 500},
    {"inducing_points": "first"}, {"kernel_variance": 2.0, "jitter": 1e-4},
])
def test_svgp_fit_predict_score_match_former_path(params):
    x, y = _regression()
    params = {k: (x[:8].copy() if v == "first" else v) for k, v in params.items()}
    native, old = mojolearn.SVGP(**params), mojolearn.SVGP(**params)
    assert audit_native_call(native.fit, x, y).assert_native() is native
    legacy.svgp_fit(old, x, y)
    assert _state(native, SV_ATTRS) == _state(old, SV_ATTRS)
    q = x[::3]
    assert _bits(audit_native_call(native.predict, q).assert_native()) == _bits(legacy.svgp_predict(old, q))
    w = np.linspace(0.5, 1.5, x.shape[0], dtype=np.float32)
    for sw in (None, w):
        got = audit_native_call(native.score, x, y, sample_weight=sw).assert_native()
        assert _bits(got) == _bits(legacy.svgp_score(old, x, y, sample_weight=sw))


@pytest.mark.parametrize("params,data", [
    ({"n_inducing": 0}, "ok"), ({"lengthscale": 0.0}, "ok"), ({}, "short_y"),
    ({"noise_variance": 0.0, "jitter": 0.0, "inducing_points": "dup"}, "ok"),
])
def test_svgp_refusals_match_former_path(params, data):
    x, y = _regression()
    if data == "short_y":
        y = y[:-1]
    params = {k: (np.repeat(x[:1], 4, axis=0) if v == "dup" else v) for k, v in params.items()}
    native, old = mojolearn.SVGP(**params), mojolearn.SVGP(**params)
    assert _outcome(native.fit, x, y) == _outcome(legacy.svgp_fit, old, x, y)
    assert getattr(native, "precision_", None) == getattr(old, "precision_", None)
