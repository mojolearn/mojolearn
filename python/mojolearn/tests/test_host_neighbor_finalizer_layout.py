# SPDX-License-Identifier: Apache-2.0
"""Saved neighbor models answer on the CPU through the native layout check.

Each explicit CPU neighbor class carries the host archive's native finalizer
in place of its GPU base's. The query context it shares with the public class
must admit that connected descriptor; `verify --cross-check` on knn was
refused with "Python model finalizer hooks are not accepted" when it did not.
A real Python finalizer stays refused.
"""
import numpy as np
import pytest

import mojolearn
from mojolearn import _backend, _classical_host


def _data():
    rng = np.random.default_rng(7)
    x = rng.standard_normal((48, 5)).astype(np.float32)
    q = rng.standard_normal((6, 5)).astype(np.float32)
    yc = (x[:, 0] > 0).astype(np.int64)
    yr = x[:, 1].astype(np.float32)
    return x, q, yc, yr


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert (a.dtype, a.shape, a.tobytes()) == (b.dtype, b.shape, b.tobytes())


def _fitted(name):
    x, q, yc, yr = _data()
    if name == "NearestNeighbors":
        model = mojolearn.NearestNeighbors(n_neighbors=3, numeric_mode="identical").fit(x)
    elif name == "KNeighborsClassifier":
        model = mojolearn.KNeighborsClassifier(n_neighbors=3, numeric_mode="identical").fit(x, yc)
    elif name == "KNeighborsRegressor":
        model = mojolearn.KNeighborsRegressor(n_neighbors=3, numeric_mode="identical").fit(x, yr)
    elif name == "RadiusNeighbors":
        model = mojolearn.RadiusNeighbors(radius=2.5, numeric_mode="identical").fit(x)
    else:
        try:
            _backend.load_host_module("_mojolearn_ivf_search_host")
        except ImportError:
            pytest.skip("_mojolearn_ivf_search_host is not built here")
        model = mojolearn.IVFIndex(n_lists=3, n_probes=2, n_neighbors=4, random_state=1, numeric_mode="identical").fit(x)
    return model, q


def _answers(model, q, name):
    if name == "IVFIndex":
        return list(model.search(q))
    if name == "RadiusNeighbors":
        distances, indices = model.radius_neighbors(q, sort_results=True)
        return [np.concatenate([np.asarray(r) for r in distances]),
                np.concatenate([np.asarray(r) for r in indices])]
    out = list(model.kneighbors(q))
    if name in ("KNeighborsClassifier", "KNeighborsRegressor"):
        out.append(model.predict(q))
    return out


NAMES = ["NearestNeighbors", "KNeighborsClassifier", "KNeighborsRegressor", "RadiusNeighbors", "IVFIndex"]


@pytest.mark.parametrize("name", NAMES)
def test_saved_neighbor_model_queries_on_the_cpu(tmp_path, name):
    model, q = _fitted(name)
    path = tmp_path / (name + ".npz")
    model.save(path)
    host = mojolearn.host_model(path)
    assert type(host) is getattr(_classical_host, "Host" + name)
    # The host class keeps its own connected native finalizer. RadiusNeighbors
    # holds no resident device handle, so neither class has a finalizer.
    public_finalizer = getattr(getattr(mojolearn, name), "__del__", None)
    if public_finalizer is None:
        assert not hasattr(type(host), "__del__")
    else:
        assert type(host).__del__ is not public_finalizer
    assert host.n_features_in_ == model.n_features_in_
    for wanted, got in zip(_answers(model, q, name), _answers(host, q, name)):
        _same(wanted, got)
    if name == "KNeighborsClassifier":
        _same(model.classes_, host.classes_)


def test_python_finalizer_on_a_host_neighbor_class_is_still_refused(tmp_path):
    model, q = _fitted("KNeighborsClassifier")
    path = tmp_path / "knn.npz"
    model.save(path)
    host = mojolearn.host_model(path)
    calls = []

    def finalize(self):
        calls.append("finalize")

    cls = _classical_host.HostKNeighborsClassifier
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cls, "__del__", finalize)
        with pytest.raises(TypeError, match="finalizer hooks are not accepted"):
            host.kneighbors(q)
    # Restored: the connected native finalizer answers again.
    _same(model.kneighbors(q)[1], host.kneighbors(q)[1])
    assert calls == []
