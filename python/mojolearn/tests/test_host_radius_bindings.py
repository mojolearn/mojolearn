"""Saved radius queries must sort through the native CPU helper."""
import numpy as np
import pytest

import mojolearn
from mojolearn import _backend, _classical_host


def test_radius_sort_helper_routes_only_to_the_host(monkeypatch):
    loaded = []
    monkeypatch.setattr(_backend, 'load_host_module', lambda name: loaded.append(name) or name)
    host = object.__new__(_classical_host.HostRadiusNeighbors)
    host._bind('_mojolearn')
    host._bind('_mojolearn_x_neighbors')
    assert loaded == ['_mojolearn_core_host', '_mojolearn_x_neighbors_host']
    with pytest.raises(ImportError):
        host._bind('_mojolearn_x_linear')
    other = object.__new__(_classical_host.HostNearestNeighbors)
    with pytest.raises(ImportError):
        other._bind('_mojolearn_x_neighbors')


@pytest.mark.parametrize('metric,extra', [
    ('euclidean', {}), ('manhattan', {}), ('chebyshev', {}),
    ('minkowski', {'p': 3}),
])
def test_saved_radius_sorted_ties_and_empty_rows(tmp_path, metric, extra):
    x = np.array([[0, 0], [1, 0], [-1, 0], [0, 2], [1, 1]], dtype=np.float32)
    query = np.array([[0, 0], [10, 10], [1, 0]], dtype=np.float32)
    model = mojolearn.RadiusNeighbors(radius=1.5, metric=metric, **extra).fit(x)
    path = tmp_path / 'radius.npz'
    model.save(path)
    host = mojolearn.host_model(path)
    expected = model.radius_neighbors(query, sort_results=True)
    actual = host.radius_neighbors(query, sort_results=True)
    for wanted_rows, actual_rows in zip(expected, actual):
        assert len(wanted_rows) == len(actual_rows) == len(query)
        for wanted, got in zip(wanted_rows, actual_rows):
            wanted, got = np.asarray(wanted), np.asarray(got)
            assert (wanted.dtype, wanted.shape, wanted.tobytes()) == (got.dtype, got.shape, got.tobytes())
    assert len(actual[0][1]) == len(actual[1][1]) == 0
    np.testing.assert_array_equal(np.asarray(actual[1][0])[:3], [0, 1, 2])
