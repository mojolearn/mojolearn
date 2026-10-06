# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""CPU-only sparse normalization. The GPU route uses sparse_input_device.

The public native entry accepts buffer descriptors, writes caller-owned CSR
storage of capacity shape.entries, and returns the actual nonzero count.
No Python objects or Python runtime are involved in this entry point.
"""
from std.builtin.sort import sort
from x_prep.sparse_input import (
    SparseShape, SparseView, SP_BAD, SP_CSR, SP_DENSE, UP, IP, FP, WP, validate_shape, validate_parts,
    sparse_pointer_unit, sparse_expand_unit, sparse_count_unit,
    sparse_write_unit, sparse_coalesce, sparse_direct_count_unit, sparse_direct_write_unit, validate_view,
)


def sparse_dense_nnz_host(s: SparseShape, v: SparseView) raises -> Int:
    validate_shape(s)
    validate_view(v)
    if s.kind != SP_DENSE or v.items != s.entries:
        raise Error("sparse input: dense capacity expects a matching dense buffer")
    var counts = List[Int32](length=s.rows + 1, fill=0)
    var nnz = 0
    for r in range(s.rows):
        sparse_direct_count_unit(r, s, v, v, counts.unsafe_ptr())
        nnz += Int(counts[r])
    return nnz


def sparse_to_csr_host(s: SparseShape, a: SparseView, b: SparseView, v: SparseView,
                       indptr: IP, indices: IP, data: FP, capacity: Int) raises -> Int:
    validate_parts(s, a, b, v)
    var ptr_bad = List[Int32](length=s.segments + 1, fill=0)
    for t in range(s.segments + 1):
        sparse_pointer_unit(t, s, a, ptr_bad.unsafe_ptr())
        if ptr_bad[t] != 0:
            raise Error("sparse input: invalid compressed row/column pointers")
    if s.kind == SP_CSR or s.kind == SP_DENSE:
        var nnz = 0
        for r in range(s.rows + 1):
            sparse_direct_count_unit(r, s, a, v, indptr)
            if s.kind == SP_DENSE:
                var count = Int(indptr.unsafe_load(r))
                indptr.unsafe_store(r, Int32(nnz))
                nnz += count
        nnz = Int(indptr.unsafe_load(s.rows))
        if nnz > capacity:
            raise Error("sparse input: CSR output capacity is too small")
        var bad = List[Int32](length=s.rows, fill=0)
        for r in range(s.rows):
            sparse_direct_write_unit(r, s, b, v, indptr, indices, data, bad.unsafe_ptr())
            if bad[r] != 0:
                raise Error("sparse input: invalid index or value outside signed-int64 range")
        return nnz
    if capacity < s.entries:
        raise Error("sparse input: CSR output capacity is too small")
    var rows = List[UInt32](length=s.entries, fill=SP_BAD)
    var cols = List[UInt32](length=s.entries, fill=SP_BAD)
    var vals = List[UInt64](length=s.entries, fill=0)
    var bad = List[Int32](length=s.entries, fill=0)
    var keys = List[UInt64](length=s.entries, fill=0)
    for t in range(s.entries):
        sparse_expand_unit(t, s, a, b, v, rows.unsafe_ptr(), cols.unsafe_ptr(), vals.unsafe_ptr(), bad.unsafe_ptr())
        if bad[t] != 0:
            raise Error("sparse input: invalid index or value outside signed-int64 range")
        keys[t] = (UInt64(cols[t] if sparse_coalesce(s) else rows[t]) << 32) | UInt64(t)
    sort(keys)
    var order = List[UInt32](length=s.entries, fill=0)
    if sparse_coalesce(s):
        # Low word is the position in the first (column, source-slot) sort,
        # so row sorting is stable without relying on the sort's stability.
        for t in range(s.entries):
            order[t] = UInt32(keys[t] & UInt64(0xFFFFFFFF))
            keys[t] = (UInt64(rows[Int(order[t])]) << 32) | UInt64(t)
        sort(keys)
        var by_col = order
        for t in range(s.entries):
            order[t] = by_col[Int(keys[t] & UInt64(0xFFFFFFFF))]
            rows[t] = UInt32(keys[t] >> 32)
    else:
        for t in range(s.entries):
            order[t] = UInt32(keys[t] & UInt64(0xFFFFFFFF))
            rows[t] = UInt32(keys[t] >> 32)
    for r in range(s.rows + 1):
        sparse_count_unit(r, s, rows.unsafe_ptr(), order.unsafe_ptr(), cols.unsafe_ptr(), indptr)
    var nnz = 0
    for r in range(s.rows + 1):
        var count = Int(indptr.unsafe_load(r))
        indptr.unsafe_store(r, Int32(nnz))
        nnz += count
    for r in range(s.rows):
        sparse_write_unit(r, s, v, rows.unsafe_ptr(), order.unsafe_ptr(), cols.unsafe_ptr(), vals.unsafe_ptr(),
                          indptr, indices, data)
    return nnz
