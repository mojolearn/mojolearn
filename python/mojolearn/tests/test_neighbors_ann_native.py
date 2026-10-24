# SPDX-License-Identifier: Apache-2.0
"""The approximate indexes' `fit` and `refine` are one native call each
(bindings/neighbors_ann_native.mojo): the fitted state and the outputs equal
the frozen pre-migration Python path's bit for bit
(`_legacy_neighbors_ann`), refusals keep their class and wording, and the
public call runs no package Python (`audit_native_call`).

Needs the compiled bindings (`_mojolearn_runtime_classical_host`, the x_ann
and ivf bindings); skipped where they are not built."""
import numpy as np
import pytest

import mojolearn as ml
from mojolearn.tests import _legacy_neighbors_ann as legacy

_RNG = np.random.default_rng(7)
X = _RNG.standard_normal((257, 12)).astype(np.float32)
Q = _RNG.standard_normal((9, 12)).astype(np.float32)


def _built(make):
    try:
        return make().fit(X)
    except (ImportError, ModuleNotFoundError) as error:  # noqa: BLE001 - a binding not built here
        pytest.skip(f"compiled binding unavailable: {error}")


def _state(est):
    """The instance dict with arrays as (dtype, shape, bytes) and the
    resident handle reduced to its key's shape (ids and addresses differ
    between two fits by construction)."""
    out = {}
    for name, value in vars(est).items():
        if name == "_resident":
            key = value[0]
            arrays = key[0]
            out[name] = (len(key), tuple(a if a == ("device",) else (len(a), tuple(a[2])) for a in arrays), key[1:-1])
        elif hasattr(value, "__array__") or isinstance(value, np.ndarray):
            a = np.asarray(value)
            out[name] = (type(value).__name__, a.dtype.str, a.shape, a.tobytes())
        else:
            out[name] = value
    return out


_MAKERS = {
    "ivf": (lambda **kw: ml.IVFIndex(n_lists=8, n_probes=3, n_neighbors=5, random_state=3, **kw), legacy.ivf_fit),
    "ivf_euclid": (lambda **kw: ml.IVFIndex(n_lists=8, n_probes=3, metric="euclidean", **kw), legacy.ivf_fit),
    "pq": (lambda **kw: ml.IVFPQIndex(n_lists=8, n_probes=3, pq_dim=5, pq_bits=4, pq_kmeans_n_iters=4,
                                      random_state=3, **kw), legacy.pq_fit),
    "sq": (lambda **kw: ml.IVFSQIndex(n_lists=8, n_probes=3, random_state=3, **kw), legacy.sq_fit),
    "sq_train": (lambda **kw: ml.IVFSQIndex(n_lists=8, n_probes=3, max_train_points_per_cluster=16, **kw),
                 legacy.sq_fit),
    "rabitq": (lambda **kw: ml.IVFRaBitQIndex(n_lists=8, n_probes=3, random_state=5, **kw), legacy.rabitq_fit),
    "cagra": (lambda **kw: ml.CagraIndex(graph_degree=8, intermediate_graph_degree=16, **kw), legacy.cagra_fit),
}


@pytest.mark.parametrize("which", sorted(_MAKERS))
@pytest.mark.parametrize("data", ["f32", "f64", "fortran"])
def test_fit_state_equals_former_path(which, data):
    make, old = _MAKERS[which]
    x = {"f32": X, "f64": X.astype(np.float64), "fortran": np.asfortranarray(X)}[data]
    native = _built(make)
    native = make().fit(x)
    former = make()
    old(former, x)
    assert _state(native) == _state(former)
    # a refit drops the previous handle and builds the same state again
    native.fit(x)
    assert _state(native) == _state(former)
    # and the fitted index answers the same queries identically
    got, want = native.search(Q), former.search(Q)
    for g, w in zip(got, want):
        assert np.asarray(g).tobytes() == np.asarray(w).tobytes()


@pytest.mark.parametrize("which", sorted(_MAKERS))
def test_fit_is_one_native_call(which):
    from mojolearn._verify_native_runtime import audit_native_call
    make, _ = _MAKERS[which]
    est = _built(make)
    assert audit_native_call(est.fit, X).assert_native() is est


_BAD = {
    "ivf": [dict(n_lists=0), dict(n_lists=2.0), dict(n_lists=True), dict(metric="cosine"), dict(metric=2),
            dict(metric=True), dict(kmeans_n_iters="5"), dict(random_state=None)],
    "pq": [dict(pq_dim=0), dict(pq_dim=99), dict(pq_bits=9), dict(n_lists=10**6), dict(pq_bits=1.0),
           dict(pq_kmeans_n_iters=None)],
    "sq": [dict(n_lists=0), dict(n_lists=10**6), dict(max_train_points_per_cluster=0),
           dict(max_train_points_per_cluster=1.5)],
    "rabitq": [dict(n_lists=0), dict(random_state="x")],
    "cagra": [dict(graph_degree=-4), dict(intermediate_graph_degree=1.0)],
}
_CLASS = {"ivf": ("IVFIndex", legacy.ivf_fit), "pq": ("IVFPQIndex", legacy.pq_fit), "sq": ("IVFSQIndex", legacy.sq_fit),
          "rabitq": ("IVFRaBitQIndex", legacy.rabitq_fit), "cagra": ("CagraIndex", legacy.cagra_fit)}
_REQUIRED = {"IVFIndex": dict(n_lists=8, n_probes=3), "IVFPQIndex": dict(n_lists=8, n_probes=3),
             "IVFSQIndex": dict(n_lists=8, n_probes=3), "IVFRaBitQIndex": dict(n_lists=8, n_probes=3),
             "CagraIndex": {}}


def _outcome(fn, *args):
    try:
        fn(*args)
    except Exception as error:  # noqa: BLE001 - the refusal is the result
        return type(error).__name__, str(error)
    return None


@pytest.mark.parametrize("which,bad", [(w, b) for w, rows in _BAD.items() for b in rows])
def test_refusals_keep_class_and_wording(which, bad):
    name, old = _CLASS[which]
    cls = getattr(ml, name)
    _built(lambda: cls(**_REQUIRED[name]))
    kw = dict(_REQUIRED[name], **bad)
    got = _outcome(cls(**kw).fit, X)
    want = _outcome(old, cls(**kw), X)
    assert got is not None and got == want


@pytest.mark.parametrize("bad_x", [np.zeros((0, 3), np.float32), np.ones(5, np.float32), [[1.0, 2.0], [3.0]]])
def test_input_refusals_keep_class_and_wording(bad_x):
    _built(lambda: ml.IVFSQIndex(n_lists=2, n_probes=1))
    got = _outcome(ml.IVFSQIndex(n_lists=2, n_probes=1).fit, bad_x)
    want = _outcome(legacy.sq_fit, ml.IVFSQIndex(n_lists=2, n_probes=1), bad_x)
    assert got is not None and got[1] == want[1]


_CAND = ((np.arange(9 * 20, dtype=np.int64).reshape(9, 20) * 7919 + 13) % 300) - 20   # ids outside [0, 257) too


@pytest.mark.parametrize("metric", ["sqeuclidean", "euclidean"])
@pytest.mark.parametrize("cands", ["i64", "i32", "u16", "list", "big_endian", "strided", "empty"])
@pytest.mark.parametrize("k", [1, 5, 25])
def test_refine_equals_former_path(metric, cands, k):
    c = {"i64": _CAND, "i32": _CAND.astype(np.int32), "u16": np.abs(_CAND).astype(np.uint16),
         "list": _CAND.tolist(), "big_endian": _CAND.astype(">i8"), "strided": np.repeat(_CAND, 2, axis=1)[:, ::2],
         "empty": np.zeros((9, 0), np.int64)}[cands]
    try:
        got = ml.refine(X, Q, c, k, metric=metric)
    except (ImportError, ModuleNotFoundError) as error:  # noqa: BLE001
        pytest.skip(f"compiled binding unavailable: {error}")
    want = legacy.refine(X, Q, c, k, metric=metric)
    assert len(got) == len(want) == 2
    for g, w in zip(got, want):
        assert type(g) is type(w)
        assert (g.dtype, g.shape) == (w.dtype, w.shape)
        assert np.asarray(g).tobytes() == np.asarray(w).tobytes()


@pytest.mark.parametrize("args", [
    (X, Q, _CAND, 5, None, "cosine"),
    (X, Q[:, :5], _CAND, 5, None, "sqeuclidean"),
    (X, Q, _CAND[:4], 5, None, "sqeuclidean"),
    (X, Q, _CAND.astype(np.float32), 5, None, "sqeuclidean"),
    (X, Q, _CAND.ravel(), 5, None, "sqeuclidean"),
    (X, Q, _CAND, 5.0, None, "sqeuclidean"),
    (X, Q, _CAND, True, None, "sqeuclidean"),
    (X, Q, _CAND * (1 << 33), 5, None, "sqeuclidean"),
    (X, Q, _CAND, -1, None, "sqeuclidean"),
    (np.zeros((0, 12), np.float32), Q, _CAND, 5, None, "sqeuclidean"),
])
def test_refine_refusals_keep_class_and_wording(args):
    try:
        ml.refine(X, Q, _CAND, 1)
    except (ImportError, ModuleNotFoundError) as error:  # noqa: BLE001
        pytest.skip(f"compiled binding unavailable: {error}")
    got = _outcome(ml.refine, *args)
    want = _outcome(legacy.refine, *args)
    assert got is not None and got == want


def test_refine_is_one_native_call():
    from mojolearn._verify_native_runtime import audit_native_call
    try:
        ml.refine(X, Q, _CAND, 1)
    except (ImportError, ModuleNotFoundError) as error:  # noqa: BLE001
        pytest.skip(f"compiled binding unavailable: {error}")
    audit_native_call(ml.refine, X, Q, _CAND, 5).assert_native()
