# SPDX-License-Identifier: Apache-2.0
"""Sparse connectivity storage admission; native plans own numerical oracles.

Independent parametrized boundary cases are scheduled by the test controller.
SciPy is used only to prepare input storage, before the public call under test.
Product conversion must not invoke its conversion methods.
"""
import numpy as np
import pytest

import mojolearn as ml


def _graph(format):
    sparse = pytest.importorskip("scipy.sparse")
    dense = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]], dtype=np.float32)
    return sparse.csr_matrix(dense).asformat(format)


def _model(graph):
    return ml.AgglomerativeClustering(
        n_clusters=2, linkage="single", connectivity=graph, numeric_mode="identical",
    )


def _x():
    return np.array([[0, 0], [1, 0], [5, 0], [6, 0]], dtype=np.float32)


@pytest.mark.parametrize("format", ["csr", "csc", "bsr", "coo", "dia", "dok", "lil"])
def test_sparse_connectivity_does_not_call_python_conversion_methods(format, monkeypatch):
    graph = _graph(format)
    calls = []

    def forbidden(*args, **kwargs):
        calls.append("conversion")
        raise AssertionError("Python sparse conversion entered product execution")

    # Preparation is finished. Admitted storage may be read, but these
    # Python methods must not provide any conversion or numerical work.
    for name in ("tocoo", "toarray", "todense", "tocsr"):
        monkeypatch.setattr(type(graph), name, forbidden)
    model = _model(graph)
    assert model.fit(_x()) is model
    assert np.asarray(model.children_).shape == (3, 2)
    assert np.asarray(model.labels_).shape == (4,)
    assert calls == []


@pytest.mark.parametrize("format", ["dok", "lil"])
@pytest.mark.parametrize("field", ["coordinate", "value"])
def test_sparse_container_admission_refuses_scalar_hooks(format, field):
    graph = _graph(format)
    calls = []

    class Protocol:
        def __int__(self):
            calls.append("int")
            raise AssertionError("Python int hook")

        def __index__(self):
            calls.append("index")
            raise AssertionError("Python index hook")

        def __float__(self):
            calls.append("float")
            raise AssertionError("Python float hook")

    item = Protocol()
    if format == "dok":
        graph._dict = {(0, item if field == "coordinate" else 1): item if field == "value" else 1.0}
    else:
        graph.rows[0] = [item if field == "coordinate" else 1]
        graph.data[0] = [item if field == "value" else 1.0]
    model = _model(graph)
    message = "sparse coordinates must be native integers" if field == "coordinate" else "sparse values require native numeric scalars"
    with pytest.raises(TypeError, match=message):
        model.fit(_x())
    assert calls == []
    assert vars(model).get("labels_") is None


@pytest.mark.parametrize("format", ["dok", "lil"])
def test_sparse_container_preserves_value_refusal_before_later_coordinate(format):
    graph = _graph(format)
    calls = []

    class LaterCoordinate:
        def __index__(self):
            calls.append("index")
            raise AssertionError("later coordinate hook")

    later = LaterCoordinate()
    if format == "dok":
        graph._dict = {(0, 1): object(), (0, later): 1.0}
    else:
        graph.rows[0] = [1, later]
        graph.data[0] = [object(), 1.0]
    model = _model(graph)
    with pytest.raises(TypeError, match="sparse values require native numeric scalars"):
        model.fit(_x())
    assert calls == []
