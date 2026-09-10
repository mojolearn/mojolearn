"""The affinity boundary must not import a scientific Python dependency."""
import array
import builtins
from unittest.mock import patch

from mojolearn._spectral_impl import _coo_triples


def test_dense_and_sparse_protocol_without_scipy_or_numpy():
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split('.')[0] in ('numpy', 'scipy'):
            raise AssertionError('Unexpected runtime dependency: ' + name)
        return real_import(name, *args, **kwargs)

    class COO:
        shape = (2, 2)
        row = array.array('i', [0, 1])
        col = array.array('i', [1, 0])
        data = array.array('f', [2, 3])

        def tocoo(self):
            return self

    with patch('builtins.__import__', guarded):
        dense = _coo_triples([[0., 2.], [3., 0.]])
        sparse = _coo_triples(COO())
    assert dense[3] == sparse[3] == 2
    for a, b in zip(dense[:3], sparse[:3]):
        assert a.tolist() == b.tolist()
