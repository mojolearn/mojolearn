# SPDX-License-Identifier: Apache-2.0
"""Python admission/return/refusal checks for native dense-to-COO scans.

The arithmetic matrix, IEEE edge cases, scan order and integer RNE oracle
live in checks/native_tests/nonzero_scan.mojo. This file checks the Python
boundary only: exact sparse inputs, refusal before Python callbacks, buffer
shapes/dtypes, and error translation from the native functions.
"""
import array

import pytest

from mojolearn import Array, _buffer
from mojolearn._spectral_impl import _coo_triples


def test_dense_boundary_returns_native_typed_buffers():
    rows, cols, values, n = _coo_triples([[0.0, 2.5], [-1.0, -0.0]])
    assert n == 2
    assert rows.dtype == cols.dtype == "<i4" and values.dtype == "<f4"
    assert rows.shape == cols.shape == values.shape == (2,)
    assert rows.tolist() == [0, 1] and cols.tolist() == [1, 0]
    assert values.tolist() == [2.5, -1.0]


def test_sparse_input_refuses_python_protocol_without_imports_or_callbacks(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in ("numpy", "scipy"):
            raise AssertionError("unexpected runtime dependency: " + name)
        return real_import(name, *args, **kwargs)

    called = []

    class COO:
        shape = (3, 3)
        row = array.array("i", [0, 1, 2])
        col = array.array("i", [2, 0, 1])
        data = array.array("d", [1.5, -2.0, 1e-300])  # float64 in, f32 out

        def tocoo(self):
            called.append("tocoo")
            pytest.fail("product called a Python sparse protocol")

    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(TypeError, match="native input types"):
        _coo_triples(COO())
    assert called == []


def test_sparse_input_reads_exact_scipy_storage_without_protocol(monkeypatch):
    sp = pytest.importorskip("scipy.sparse")
    source = sp.coo_matrix(([1.5, -2.0, 1e-300], ([0, 1, 2], [2, 0, 1])), shape=(3, 3))
    not_square = sp.coo_matrix(([1.5], ([0], [1])), shape=(3, 2))
    monkeypatch.setattr(sp.coo_matrix, "tocoo",
                        lambda *a, **k: pytest.fail("product called SciPy tocoo"))
    rows, cols, vals, n = _coo_triples(source)
    assert n == 3
    assert rows.tolist() == [0, 1, 2] and cols.tolist() == [2, 0, 1]
    assert vals.tobytes() == array.array("f", [1.5, -2.0, 1e-300]).tobytes()

    with pytest.raises(ValueError, match="must be square"):
        _coo_triples(not_square)


def test_direct_binding_refusals():
    count_fn = _buffer._native("nonzero_f64_count")
    fill_fn = _buffer._native("nonzero_f64_fill")
    src = Array.from_list([1.0, 0.0, 2.0, 3.0], "<f8")
    a = _buffer.addr_ro(src, name="src")
    assert count_fn(a, 0) == 0
    assert count_fn(a, 4) == 3
    with pytest.raises(Exception, match="non-negative"):
        count_fn(a, -1)
    out = [_buffer.empty((3,), "<i4"), _buffer.empty((3,), "<i4"),
           _buffer.empty((3,), "<f4")]
    addrs = [_buffer.addr(o, name="o") for o in out]
    assert fill_fn(a, 2, 2, addrs, 3) == 3
    assert out[0].tolist() == [0, 1, 1] and out[1].tolist() == [0, 0, 1]
    assert out[2].tolist() == [1.0, 2.0, 3.0]
    # Stale, too-small capacity: refused, and nothing written past it.
    small = [_buffer.empty((1,), "<i4"), _buffer.empty((1,), "<i4"),
             _buffer.empty((1,), "<f4")]
    saddrs = [_buffer.addr(o, name="o") for o in small]
    with pytest.raises(Exception, match="more than 1 nonzero"):
        fill_fn(a, 2, 2, saddrs, 1)
    assert fill_fn(a, 0, 5, addrs, 3) == 0
    with pytest.raises(Exception, match="non-negative"):
        fill_fn(a, -1, 2, addrs, 3)
    with pytest.raises(Exception, match="3 addresses"):
        fill_fn(a, 2, 2, addrs[:2], 3)
