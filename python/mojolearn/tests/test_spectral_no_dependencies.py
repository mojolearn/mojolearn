"""The affinity boundary must not import a scientific Python dependency."""
import array
import builtins
from unittest.mock import patch

import pytest

from mojolearn._spectral_impl import _coo_triples


def test_dense_and_sparse_protocol_without_scipy_or_numpy():
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split('.')[0] in ('numpy', 'scipy'):
            raise AssertionError('Unexpected runtime dependency: ' + name)
        return real_import(name, *args, **kwargs)

    called = []

    class COO:
        shape = (2, 2)
        row = array.array('i', [0, 1])
        col = array.array('i', [1, 0])
        data = array.array('f', [2, 3])

        def tocoo(self):
            called.append(True)
            return self

    with patch('builtins.__import__', guarded):
        dense = _coo_triples([[0., 2.], [3., 0.]])
        # A duck-typed sparse object is not SciPy storage: its tocoo() is
        # never called; the dense admission refuses it.
        with pytest.raises((TypeError, ValueError)):
            _coo_triples(COO())
    assert called == []
    assert dense[3] == 2
    assert dense[0].tolist() == [0, 1]
    assert dense[1].tolist() == [1, 0]
    assert dense[2].tolist() == [2.0, 3.0]
