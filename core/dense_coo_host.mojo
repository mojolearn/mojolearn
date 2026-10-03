# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`knn_affinity_f32` on the host (the CPU route `bindings/_mojolearn_core_host.mojo`'s body; HOST ONLY). The GPU binding runs `core/dense_coo_device.mojo`, the same bytes. Moved out of core/dense_coo.mojo (cpu-gpu-cleanup c-core) so the GPU binding imports no host threads."""
from std.builtin.sort import sort
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_chunk, host_predict_task_count

comptime F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime I32P = MutPointer[Int32, MutUntrackedOrigin]


@always_inline
def _key(v: Float32, j: Int) -> UInt64:
    var b = bitcast[DType.uint32](v)
    if b == UInt32(0x80000000):
        b = UInt32(0)
    return (UInt64(Int(b)) << 32) | UInt64(j)


def knn_affinity_f32(
    dense: F32P, rows: I32P, cols: I32P, vals: F32P, nnz: Int, sparse: Bool,
    n: Int, k: Int, aff: F32P, status: I32P,
) raises:
    """A (n x n, the caller's zeroed float32 buffer) = 0.5 (C + C^T), C the
    k-smallest (value, column) candidates of each row: every column of the
    dense matrix, or a sparse row's stored COO entries (duplicates kept, as
    the Python lists kept them). `status` (2 int32): [0, 0] done; [1, i] row
    i holds a NaN or negative distance; [2, i] row i has fewer than k
    candidates; the FIRST failing row in row order, the checks in the
    Python loop's order (NaN or negative first, then the count). A COO
    entry outside [0, n) is refused by name."""
    # candidates per row (sparse: COO order, grouped by a counting pass)
    var start = List[Int](length=n + 1, fill=0)
    var order = List[Int32](capacity=nnz if sparse else 0)
    if sparse:
        for e in range(nnz):
            var r = Int(rows.unsafe_load(e))
            var c = Int(cols.unsafe_load(e))
            if r < 0 or r >= n or c < 0 or c >= n:
                raise Error("knn_affinity_f32: a COO entry lies outside the square graph")
            start[r + 1] += 1
        for i in range(n):
            start[i + 1] += start[i]
        var fill = List[Int](length=n, fill=0)
        for _ in range(nnz):
            order.append(Int32(0))
        for e in range(nnz):
            var r = Int(rows.unsafe_load(e))
            order[start[r] + fill[r]] = Int32(e)
            fill[r] += 1
    var C = List[UInt8](length=n * n if n > 0 else 1, fill=UInt8(0))
    var cptr = C.unsafe_ptr()
    var bad_kind = List[Int32](length=n if n > 0 else 1, fill=Int32(0))
    var bptr = bad_kind.unsafe_ptr()
    var sptr = start.unsafe_ptr()
    var optr = order.unsafe_ptr()
    var tasks = host_predict_task_count(n)
    var chunk = host_predict_chunk(n, tasks)

    def _rows(task: Int) {imm dense, imm rows, imm cols, imm vals, imm sparse, imm n, imm k, imm chunk,
                           imm cptr, imm bptr, imm sptr, imm optr}:
        var lo = task * chunk
        var hi = min(lo + chunk, n)
        var keys = List[UInt64]()
        for i in range(lo, hi):
            keys.clear()
            var m = (sptr[i + 1] - sptr[i]) if sparse else n
            var bad = False
            for q in range(m):
                var v: Float32
                var j: Int
                if sparse:
                    var e = Int(optr[sptr[i] + q])
                    v = vals.unsafe_load(e)
                    j = Int(cols.unsafe_load(e))
                else:
                    v = dense.unsafe_load(i * n + q)
                    j = q
                if v != v or v < Float32(0):
                    bad = True
                    break
                keys.append(_key(v, j))
            if bad:
                bptr.unsafe_store(i, Int32(1))
                continue
            if m < k:
                bptr.unsafe_store(i, Int32(2))
                continue
            sort(keys)
            for q in range(k):
                var j = Int(keys[q] & UInt64(0xFFFFFFFF))
                cptr.unsafe_store(i * n + j, UInt8(1))

    if tasks <= 1:
        _rows(0)
    else:
        host_parallelize(_rows, tasks)
    for i in range(n):
        var b = bad_kind[i]
        if b != Int32(0):
            status.unsafe_store(0, b)
            status.unsafe_store(1, Int32(i))
            return
    for i in range(n):
        for j in range(n):
            var t = Int(C[i * n + j]) + Int(C[j * n + i])
            # the Python loop: 0.5 if t == 1 else t / 2 (t in {0, 2})
            aff.unsafe_store(i * n + j, Float32(0.5) if t == 1 else (Float32(1) if t == 2 else Float32(0)))
    status.unsafe_store(0, Int32(0))
    status.unsafe_store(1, Int32(0))
    _ = C^
    _ = bad_kind^
    _ = start^
    _ = order^
