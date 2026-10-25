# SPDX-License-Identifier: Apache-2.0
"""The four ways the public parallel drivers refused correct calls at 0.8.57.

1. "parallel estimator methods or storage hooks changed": the drivers'
   class/storage snapshots were taken before the package's last import-time
   class connections (native attribute slots), so every estimator looked
   modified. The snapshots must equal the final class dictionaries.
2. "shard arrays require ... got float32" / "Scaler input must have dtype
   float32": an ndarray admitted in place had its dtype read as its name
   ("float32") instead of its typestr ("<f4").
3. "'classes_'": a fit_forest classifier lost its public class labels on the
   way back from the shard workers, so predict refused.
4. "invalid global IVF list offsets": DistributedIVFIndex.from_index planned
   its shards from int32 copies that were already released.

Small shapes, one device by default (MOJOLEARN_PAR_DEVICES, e.g. "0,1").

    cd python && python3 -m pytest mojolearn/tests/test_parallel_api_regression.py
"""
import os

import numpy as np
import pytest

import mojolearn


def _devices():
    raw = os.environ.get("MOJOLEARN_PAR_DEVICES", "0")
    return tuple(int(v) for v in raw.split(",") if v.strip())


def _data(rows=256, cols=8, seed=0):
    rng = np.random.default_rng(seed)
    X = np.ascontiguousarray(rng.standard_normal((rows, cols)).astype(np.float32))
    yc = (X[:, 0] + X[:, 1] > 0).astype(np.int64)
    yr = np.ascontiguousarray((X[:, 0] - 0.5 * X[:, 2]).astype(np.float32))
    return X, yc, yr


def _same(label, actual, expected):
    a, e = np.asarray(actual), np.asarray(expected)
    assert a.shape == e.shape, (label, a.shape, e.shape)
    assert a.tobytes() == e.tobytes(), label


def _run(case):
    try:
        for label, actual, expected in case(_devices()):
            _same(label, actual, expected)
    except NotImplementedError as exc:
        if "no CPU implementation" in str(exc):
            pytest.skip(f"this package has no route for the driver: {exc}")
        raise
    except ValueError as exc:
        if "device selection is unavailable" in str(exc):
            pytest.skip(f"device group unavailable here: {exc}")
        raise
    except (ImportError, OSError) as exc:
        pytest.skip(f"binding not installed: {exc}")


# ------------------------------------------------- 1. class/storage snapshots

def test_parallel_snapshots_match_the_imported_classes():
    meta = mojolearn.parallel_ensemble._FAMILY_CONTEXT.get("parallel_public")
    if meta is None:
        pytest.skip("the compiled parallel boundary is not connected in this package")
    namespace = type.__dict__["__dict__"].__get__
    layouts = list(meta["classes"].items())
    if "predictor_class" in meta:
        layouts.append((meta["predictor_class"], meta["predictor_layout"]))
    assert layouts
    for cls, layout in layouts:
        assert layout["mro"] == type.__dict__["__mro__"].__get__(cls), cls
        for base, name, value in layout["hooks"]:
            assert namespace(base).get(name) is value, (cls.__name__, base.__name__, name)


def _logistic_and_svm(dev):
    from mojolearn.parallel_classical import fit_logistic, fit_svm
    X, yc, _ = _data(128, 6)
    lr = fit_logistic(mojolearn.LogisticRegression(), X, yc, devices=dev)
    plain_lr = mojolearn.LogisticRegression().fit(X, yc)
    svc = fit_svm(mojolearn.SVC(), X, yc, devices=dev)
    plain_svc = mojolearn.SVC().fit(X, yc)
    return [("fit_logistic predict_proba", lr.predict_proba(X), plain_lr.predict_proba(X)),
            ("fit_svm predict", svc.predict(X), plain_svc.predict(X))]


def test_estimator_drivers_admit_a_plain_estimator():
    _run(_logistic_and_svm)


# ------------------------------------------------- 2. ndarray typestr

def _scalers(dev):
    from mojolearn.parallel_preprocessing import fit_scaler, transform_scaler
    X, _, _ = _data(64, 10)
    out = []
    for cls in (mojolearn.StandardScaler, mojolearn.MinMaxScaler):
        par = fit_scaler(cls(), X, devices=dev, columns_per_shard=4)
        plain = cls().fit(X)
        out.append((cls.__name__ + " transform_scaler", transform_scaler(par, X, devices=dev, columns_per_shard=4),
                    plain.transform(X)))
    return out


def test_scaler_driver_admits_a_float32_ndarray():
    _run(_scalers)


def _reference(dev):
    from mojolearn.parallel_neighbors_reference import ReferenceShardedNeighbors
    X, yc, _ = _data(320, 8)
    m = mojolearn.KNeighborsClassifier(n_neighbors=5).fit(X[:256], yc[:256])
    q = np.ascontiguousarray(X[256:])
    with ReferenceShardedNeighbors(m, devices=dev, reference_rows_per_shard=64,
                                   query_rows_per_shard=32) as rs:
        d, i = rs.kneighbors(q)
        pr = rs.predict(q)
    d0, i0 = m.kneighbors(q)
    return [("reference distances", d, d0), ("reference indices", i, i0), ("reference predict", pr, m.predict(q))]


def test_reference_shards_admit_a_float32_ndarray_query():
    _run(_reference)


def _queries(dev):
    from mojolearn.parallel_neighbors import ParallelQueries
    X, yc, _ = _data(320, 8)
    m = mojolearn.KNeighborsClassifier(n_neighbors=5).fit(X[:256], yc[:256])
    q = np.ascontiguousarray(X[256:])
    with ParallelQueries(m, devices=dev, rows_per_shard=16) as pq:
        d, i = pq.query(q, method="kneighbors")
    d0, i0 = m.kneighbors(q)
    return [("queries distances", d, d0), ("queries indices", i, i0)]


def test_parallel_queries_admit_a_float32_ndarray_query():
    _run(_queries)


# ------------------------------------------------- 3. fit_forest class labels

def _forests(dev):
    from mojolearn.parallel_ensemble import fit_forest
    X, yc, _ = _data(256, 8)
    kw = dict(n_estimators=8, max_depth=4, random_state=7)
    out = []
    for cls in (mojolearn.RandomForestClassifier, mojolearn.ExtraTreesClassifier):
        par = fit_forest(cls(**kw), X, yc, devices=dev, trees_per_shard=2)
        plain = cls(**kw).fit(X, yc)
        assert list(par.classes_) == list(plain.classes_), cls.__name__
        out.append((cls.__name__ + " predict", par.predict(X), plain.predict(X)))
        out.append((cls.__name__ + " predict_proba", par.predict_proba(X), plain.predict_proba(X)))
    return out


def test_fit_forest_classifier_keeps_its_class_labels():
    _run(_forests)


# ------------------------------------------------- 4. distributed IVF plan

def _ivf(dev):
    from mojolearn.parallel_ivf import DistributedIVFIndex
    X, _, _ = _data(576, 8)
    kw = dict(n_lists=16, n_probes=4, n_neighbors=8, random_state=3)
    q = np.ascontiguousarray(X[512:])
    m = mojolearn.IVFIndex(**kw).fit(X[:512])
    d0, i0 = m.search(q)
    with DistributedIVFIndex.from_index(m, devices=dev) as shards:
        d, i = shards.search(q)
    return [("ivf distances", d, d0), ("ivf indices", i, i0)]


def test_distributed_ivf_plans_a_fitted_index():
    _run(_ivf)
