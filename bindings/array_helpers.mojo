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


@always_inline
def _addr_ptr[dt: DType](addr: Int) raises -> MutPointer[Scalar[dt], MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null buffer address")
    return MutPointer[Scalar[dt], MutUntrackedOrigin](unsafe_from_address=addr)


def _strided_copy[dt: DType](
    src: Int, dst: Int, dims: MutPointer[Int64, MutUntrackedOrigin], nd: Int, total: Int
) raises:
    """Walk the index space of `shape` in C order; element (i_0..i_k) moves
    from src[src_base + sum i_j * s_j] to dst[dst_base + sum i_j * d_j]."""
    var sp = _addr_ptr[dt](src)
    var dp = _addr_ptr[dt](dst)
    var sbase = Int(dims[3 * nd])
    var dbase = Int(dims[3 * nd + 1])
    var inner = Int(dims[nd - 1])
    var sstep = Int(dims[2 * nd - 1])
    var dstep = Int(dims[3 * nd - 1])
    var idx = List[Int](length=nd, fill=0)
    var so = sbase
    var do_ = dbase
    var done = 0
    while done < total:
        for j in range(inner):
            dp.unsafe_store(do_ + j * dstep, sp.unsafe_load(so + j * sstep))
        done += inner
        # advance the outer mixed-radix counter (row-major)
        var k = nd - 2
        while k >= 0:
            idx[k] += 1
            so += Int(dims[nd + k])
            do_ += Int(dims[2 * nd + k])
            if idx[k] < Int(dims[k]):
                break
            so -= idx[k] * Int(dims[nd + k])
            do_ -= idx[k] * Int(dims[2 * nd + k])
            idx[k] = 0
            k -= 1


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
    var total = 1
    for k in range(nd):
        var s = Int(dp[k])
        if s < 0:
            raise Error("strided_copy_bytes: negative dimension")
        total *= s
    if total == 0:
        return PythonObject(0)
    var src = Int(py=src_addr)
    var dst = Int(py=dst_addr)
    with GILReleased(Python()):
        if isz == 1:
            _strided_copy[DType.uint8](src, dst, dp, nd, total)
        elif isz == 2:
            _strided_copy[DType.uint16](src, dst, dp, nd, total)
        elif isz == 4:
            _strided_copy[DType.uint32](src, dst, dp, nd, total)
        elif isz == 8:
            _strided_copy[DType.uint64](src, dst, dp, nd, total)
        else:
            raise Error("strided_copy_bytes: itemsize must be 1, 2, 4 or 8")
    return PythonObject(total)


def check_lengths_i64_binding(
    lengths_addr: PythonObject, b: PythonObject, length: PythonObject
) raises -> PythonObject:
    """The first i whose length is outside [1, length], or -1."""
    var n = Int(py=b)
    var hi = Int64(Int(py=length))
    var lp = _addr_ptr[DType.int64](Int(py=lengths_addr)) if n > 0 else MutPointer[
        Int64, MutUntrackedOrigin
    ]()
    for i in range(n):
        var v = lp.unsafe_load(i)
        if v < 1 or v > hi:
            return PythonObject(i)
    return PythonObject(-1)


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
    with GILReleased(Python()):
        if m == 0:
            var sp = _addr_ptr[DType.uint8](Int(py=src_addr))
            for i in range(n):
                var keep = Int(lp.unsafe_load(i)) * pos
                var base = i * row
                for k in range(keep):
                    dp.unsafe_store(base + k, sp.unsafe_load(base + k))
                for k in range(keep, row):
                    dp.unsafe_store(base + k, 0)
        elif m == 1:
            for i in range(n):
                var keep = Int(lp.unsafe_load(i)) * pos
                var base = i * row
                for k in range(keep, row):
                    dp.unsafe_store(base + k, 0)
        else:
            var sp = _addr_ptr[DType.uint8](Int(py=src_addr))
            for i in range(n):
                var at = i * pos + (Int(lp.unsafe_load(i)) - 1) * row
                for k in range(row):
                    dp.unsafe_store(i * row + k, sp.unsafe_load(at + k))
    return PythonObject(0)


def nsum_f64_binding(addr: PythonObject, n: PythonObject) raises -> PythonObject:
    """`_portable_math.nsum`: CPython 3.12+'s float `sum`, `0 + x0`, then
    Neumaier's compensated sum in order, the compensation added once at the
    end when it is nonzero and finite. 0.0 for an empty input."""
    var count = Int(py=n)
    if count <= 0:
        return PythonObject(0.0)
    var p = _addr_ptr[DType.float64](Int(py=addr))
    var total = 0.0 + p.unsafe_load(0)
    var c = Float64(0.0)
    for i in range(1, count):
        var x = p.unsafe_load(i)
        var t = total + x
        if abs(total) >= abs(x):
            c += (total - t) + x
        else:
            c += (x - t) + total
        total = t
    if c != 0.0 and c - c == 0.0:
        total += c
    return PythonObject(total)
