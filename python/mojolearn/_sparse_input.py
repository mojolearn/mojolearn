# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Transitional buffer adapter for native sparse normalization.

No SciPy conversion, Python element loop, dtype conversion or data reduction
is used here. This file nevertheless executes Python and is NOT compliant
with the owner's final no-product-Python interface requirement. The independent
native APIs live in x_prep/sparse_input_device.mojo and x_prep/host/sparse_input.mojo.
"""
from contextlib import ExitStack

from ._array import Array
from ._buffer import Buf, _output_store

_FORMATS = {"csr": 0, "csc": 1, "coo": 2, "bsr": 3, "dia": 4}
_MAX = 2**31 - 1
_EMPTY = (0, 2, 4, 0, 1, 1, 4, 0, 0, 0, 0)
_DTYPES = {("f", 4): 0, ("d", 8): 1, ("i", 4): 2, ("q", 8): 3,
           ("l", 4): 2, ("l", 8): 3, ("I", 4): 4, ("L", 4): 4,
           ("B", 1): 5, ("H", 2): 6, ("b", 1): 7, ("e", 2): 8,
           ("h", 2): 9, ("Q", 8): 10, ("L", 8): 10, ("?", 1): 11}


class NativeCSR:
    """Owned CSR buffers produced by the native converter; no SciPy object."""
    __slots__ = ("_parts", "shape", "nnz")

    def __init__(self, ip, ix, dv, n, d):
        self._parts = ip, ix, dv, n, d
        self.shape = n, d
        self.nnz = dv.size


def _storage(code, count, dtype):
    return Array._owned(_output_store(code, count), (count,), dtype, "C")


def _descriptor(buf, *, indices=False):
    fmt = buf.format
    endian = fmt[0] if fmt and fmt[0] in "@=<>!" else "="
    letter = fmt[1:] if fmt and fmt[0] in "@=<>!" else fmt
    code = _DTYPES.get((letter, buf.itemsize))
    if code is None or (indices and code in (0, 1, 8)):
        raise TypeError(f"mojolearn: unsupported sparse {'index' if indices else 'value'} buffer {fmt!r}")
    if buf.ndim < 1 or buf.ndim > 3:
        raise ValueError("mojolearn: sparse storage buffers must have one to three axes")
    dims = buf.shape + (1,) * (3 - buf.ndim)
    strides = buf.strides
    if strides is None:
        strides = (dims[1] * dims[2] * buf.itemsize, dims[2] * buf.itemsize, buf.itemsize)
    else:
        strides = strides + (0,) * (3 - buf.ndim)
    if any(d < 0 for d in dims) or dims[0] * dims[1] * dims[2] > _MAX:
        raise ValueError("mojolearn: sparse storage exceeds Int32 indexing")
    # Only buffer-axis metadata is visited; no element values are read.
    extent = tuple((d - 1) * s for d, s in zip(dims, strides))
    lo = sum(min(0, e) for e in extent)
    hi = sum(max(0, e) for e in extent)
    span = 0 if 0 in dims else hi - lo + buf.itemsize
    width = -buf.itemsize if endian in (">", "!") else buf.itemsize
    # An empty trailing axis is represented as an empty leading axis, so the
    # native flattened-index decoder never divides by zero.
    if 0 in dims:
        dims = (0, max(dims[1], 1), max(dims[2], 1))
        lo = 0
    return (buf.addr + lo, code, width, *dims, *strides, -lo, span)


def _object_parts(X, binding, n, kind):
    """Legacy LIL/DOK marshalling is compiled, but still depends on CPython."""
    if (X.dtype.char, int(X.dtype.itemsize)) not in _DTYPES:
        raise TypeError("mojolearn: unsupported sparse object value dtype")
    cap = int(binding.x_prep_sparse_object_size(kind, X, n))
    if cap < 0 or cap > _MAX:
        raise ValueError("mojolearn: sparse object exceeds Int32 indexing")
    rows, cols = _storage("q", cap, "<i8"), _storage("q", cap, "<i8")
    vals = _storage("d", cap, "<f8")
    binding.x_prep_sparse_object_pack(kind, X, [n, cap], [rows._addr, cols._addr, vals._addr])
    return rows, cols, vals


def as_csr(X, binding, *, dense=False):
    if isinstance(X, NativeCSR):
        return X
    fmt = "dense" if dense else getattr(X, "format", None)
    if fmt not in (*_FORMATS, "lil", "dok", "dense"):
        if hasattr(X, "tocsr"):
            raise TypeError("mojolearn: sparse providers must expose CSR/CSC/COO/BSR/DIA/LIL/DOK storage; "
                            "calling a provider's Python conversion is not supported")
        return None
    shape = tuple(X.shape)
    if len(shape) != 2:
        raise ValueError("mojolearn: sparse X must be two-dimensional")
    n, d = map(int, shape)
    if n < 1 or n >= _MAX or d < 1 or d > _MAX:
        raise ValueError("mojolearn: sparse X shape is empty or exceeds Int32 indexing")
    a = b = None
    segments, br, bc = 0, 1, 1
    kind = _FORMATS.get(fmt, 5)
    if fmt in ("csr", "csc", "bsr"):
        a, b, values = X.indptr, X.indices, X.data
        if fmt == "bsr":
            br, bc = map(int, X.blocksize)
        segments = d if fmt == "csc" else n // br
    elif fmt == "coo":
        a, b, values = X.row, X.col, X.data
    elif fmt == "dia":
        a, values = X.offsets, X.data
    elif fmt in ("lil", "dok"):
        a, b, values = _object_parts(X, binding, n, 0 if fmt == "lil" else 1)
        kind = 6 if fmt == "lil" else 2
    else:
        values = X
    with ExitStack() as pins:
        av = _descriptor(pins.enter_context(Buf(a)), indices=True) if a is not None else _EMPTY
        bv = _descriptor(pins.enter_context(Buf(b)), indices=True) if b is not None else _EMPTY
        vv = _descriptor(pins.enter_context(Buf(values)))
        entries = vv[3] * vv[4] * vv[5]
        sizes = [kind, n, d, entries, segments, br, bc]
        cap = int(binding.x_prep_sparse_capacity(sizes, vv)) if dense else entries
        ip, ix, dv = _storage("i", n + 1, "<i4"), _storage("i", cap, "<i4"), _storage("f", cap, "<f4")
        nnz = int(binding.x_prep_sparse_csr(sizes, av, bv, vv, [ip._addr, ix._addr, dv._addr, cap]))
    if nnz < 0 or nnz > cap:
        raise RuntimeError("mojolearn: native sparse converter returned an invalid storage length")
    # Relabel initialized prefixes; no Python slicing/copying over data.
    if nnz != cap:
        # Export from the owning ctypes allocation, not the unowned address
        # view in Array._mv, so the prefix keeps its allocation alive.
        ix = Array.from_buffer(memoryview(ix._store).cast("B")[:nnz * 4].cast("i"))
        dv = Array.from_buffer(memoryview(dv._store).cast("B")[:nnz * 4].cast("f"))
    return NativeCSR(ip, ix, dv, n, d)
