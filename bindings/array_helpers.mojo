# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Layout helpers of the NumPy-free layer (lane pyglue-sweep, Oct 3 2026:
Python = glue only). Each replaces a Python loop over rows or columns that
moved bytes between host buffers: `_array._reorder` (C <-> F), the strided
`Array.__getitem__` copy, `parallel_forecasting`'s column gather and
scatter, `_ragged`'s per-row padding passes, and `_portable_math.nsum`.

Byte moves only (plus `nsum_f64`, the same Neumaier fold in the same order
as the Python routine it replaces), so no result bit depends on them. The
same file is compiled into the base binding (`_mojolearn`) and the core
host binding (`_mojolearn_core_host`), so a CPU-only install runs the same
code.
"""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from core.array_helpers_impl import (
    check_lengths_i64,
    class_ratio_f64,
    fsum_f64,
    nsum_f64,
    ragged_rows_bytes,
    row_means_f64,
    row_stds_f64,
    strided_copy_bytes,
    strided_total,
)
from core.array_helpers_run import _addr_ptr


def strided_copy_bytes_binding(
    src_addr: PythonObject, dst_addr: PythonObject, dims_addr: PythonObject,
    ndim: PythonObject, itemsize: PythonObject,
) raises -> PythonObject:
    """Copy an n-d strided selection between two flat host buffers.

    `dims` is int64[3 * ndim + 2]: the shape, the source strides, the
    destination strides (all in ELEMENTS, a stride may be negative), then
    the source and destination base offsets. Elements are `itemsize` bytes
    (1, 2, 4 or 8) and are moved as bits. Returns the element count."""
    var nd = Int(py=ndim)
    var isz = Int(py=itemsize)
    if nd < 1:
        raise Error("strided_copy_bytes: ndim must be positive")
    var dp = _addr_ptr[DType.int64](Int(py=dims_addr))
    var total = strided_total(dp, nd)
    if total == 0:
        return PythonObject(0)
    var src = Int(py=src_addr)
    var dst = Int(py=dst_addr)
    if src == 0 or dst == 0:
        raise Error("strided_copy_bytes: null buffer address")
    if isz != 1 and isz != 2 and isz != 4 and isz != 8:
        raise Error("strided_copy_bytes: itemsize must be 1, 2, 4 or 8")
    with GILReleased(Python()):
        strided_copy_bytes(src, dst, dp, nd, isz, total)
    return PythonObject(total)


def check_lengths_i64_binding(
    lengths_addr: PythonObject, b: PythonObject, length: PythonObject
) raises -> PythonObject:
    """The first i whose length is outside [1, length], or -1."""
    var n = Int(py=b)
    var hi = Int64(Int(py=length))
    if n <= 0:
        return PythonObject(-1)
    var lp = _addr_ptr[DType.int64](Int(py=lengths_addr))
    return PythonObject(check_lengths_i64(lp, n, hi))


def ragged_rows_bytes_binding(
    src_addr: PythonObject, dst_addr: PythonObject, lengths_addr: PythonObject,
    b: PythonObject, row_bytes: PythonObject, pos_bytes: PythonObject, mode: PythonObject,
) raises -> PythonObject:
    """Per-row ragged passes over a (B, L, ...) block of B rows of
    `row_bytes`, `pos_bytes` per position, int64 lengths:
      mode 0: dst row i = src row i's first lengths[i] positions, then zero
              bytes (dst must be zeroed or is fully written here);
      mode 1: zero dst row i from position lengths[i] on (src unused);
      mode 2: dst row i (row_bytes = one position, V values) = src position
              lengths[i] - 1 of row i, where src rows are pos_bytes long."""
    var n = Int(py=b)
    var row = Int(py=row_bytes)
    var pos = Int(py=pos_bytes)
    var m = Int(py=mode)
    if n <= 0 or row <= 0:
        return PythonObject(0)
    var lp = _addr_ptr[DType.int64](Int(py=lengths_addr))
    var dp = _addr_ptr[DType.uint8](Int(py=dst_addr))
    var sp = _addr_ptr[DType.uint8](Int(py=src_addr))
    with GILReleased(Python()):
        ragged_rows_bytes(dp, sp, lp, n, row, pos, m)
    return PythonObject(0)


def nsum_f64_binding(addr: PythonObject, n: PythonObject) raises -> PythonObject:
    """`_portable_math.nsum`: CPython 3.12+'s float `sum`, `0 + x0`, then
    Neumaier's compensated sum in order, the compensation added once at the
    end when it is nonzero and finite. 0.0 for an empty input."""
    var count = Int(py=n)
    if count <= 0:
        return PythonObject(0.0)
    var p = _addr_ptr[DType.float64](Int(py=addr))
    return PythonObject(nsum_f64(p, count))


def row_means_f64_binding(
    src_addr: PythonObject, rows: PythonObject, cols: PythonObject, out_addr: PythonObject,
) raises -> PythonObject:
    """The mean of each row of a C-order float64 (rows, cols) table:
    `_portable_math.fsum(row) / cols` (lane py-runtime-b: the search
    results' mean test and train scores, one row per candidate and one
    column per split; they were Python `fsum` over lists). The sum is
    CPython's math.fsum (Shewchuk's exact partials, rounded once; a zero sum
    is +0.0; glm/impl/lm_finish.mojo `_exact_sum`) with `fsum`'s special
    values: a row holding +inf and -inf writes nothing and is counted, a
    NaN gives the row's last NaN, one kind of infinity gives it. Returns the
    number of +inf/-inf rows (the caller raises fsum's ValueError)."""
    var nr = Int(py=rows)
    var nc = Int(py=cols)
    if nr <= 0 or nc <= 0:
        return PythonObject(0)
    var sp = _addr_ptr[DType.float64](Int(py=src_addr))
    var op = _addr_ptr[DType.float64](Int(py=out_addr))
    return PythonObject(row_means_f64(sp, op, nr, nc))


def fsum_f64_binding(src_addr: PythonObject, n: PythonObject, out_addr: PythonObject) raises -> PythonObject:
    """`_portable_math.fsum` (lane py-runtime round 3: it was `math.fsum`
    plus a Python big-integer fold): out[0] = the exact sum of n float64
    terms (see `_fsum_terms`). Returns 1 for +inf with -inf (the caller
    raises fsum's ValueError), also 1 with out[0] = +-inf when the exact sum
    of finite terms is outside the binary64 range (fsum's OverflowError),
    else 0."""
    var count = Int(py=n)
    var op = _addr_ptr[DType.float64](Int(py=out_addr))
    if count <= 0:
        op.unsafe_store(0, Float64(0.0))
        return PythonObject(0)
    var sp = _addr_ptr[DType.float64](Int(py=src_addr))
    return PythonObject(fsum_f64(sp, op, count))


def row_stds_f64_binding(
    src_addr: PythonObject, rows: PythonObject, cols: PythonObject, means_addr: PythonObject,
    out_addr: PythonObject,
) raises -> PythonObject:
    """The population standard deviation of each row of a C-order float64
    (rows, cols) score table about its mean (lane py-runtime round 3; it was
    Python's sqrt(fsum((v - m) * (v - m)) / cols) per search candidate): the
    same binary64 subtraction and product per score, the exact fsum, one
    division and the correctly rounded sqrt."""
    var nr = Int(py=rows)
    var nc = Int(py=cols)
    if nr <= 0 or nc <= 0:
        return PythonObject(0)
    var sp = _addr_ptr[DType.float64](Int(py=src_addr))
    var mp = _addr_ptr[DType.float64](Int(py=means_addr))
    var op = _addr_ptr[DType.float64](Int(py=out_addr))
    row_stds_f64(sp, mp, op, nr, nc)
    return PythonObject(0)


def class_ratio_f64_host_binding(
    counts_addr: PythonObject, k: PythonObject, n: PythonObject, mode: PythonObject, out_addr: PythonObject,
) raises -> PythonObject:
    """Per-class ratios of the k int64 class counts (lane py-runtime-b: the
    Python list arithmetic of the class-weight and prior glue). mode 0:
    scikit-learn's 'balanced' weight n / (k * count_c); mode 1: the
    empirical prior count_c / n. Both operands are exact integers (below
    2^53) and the quotient is one IEEE binary64 division, so the words are
    Python's `int / int`. Returns the number of zero counts in mode 0 (the
    caller raises Python's ZeroDivisionError; those cells are left 0)."""
    var kk = Int(py=k)
    var nn = Int(py=n)
    var m = Int(py=mode)
    if kk <= 0:
        return PythonObject(0)
    if m != 0 and m != 1:
        raise Error("class_ratio_f64: mode is 0 (balanced) or 1 (prior)")
    if m == 1 and nn == 0:
        raise Error("class_ratio_f64: the prior of zero rows")
    var cp = _addr_ptr[DType.int64](Int(py=counts_addr))
    var op = _addr_ptr[DType.float64](Int(py=out_addr))
    return PythonObject(class_ratio_f64(cp, op, kk, nn, m))
