# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""PageRank's power iteration over the nonzero cells of the dense adjacency
(lane neural-pass30, 2026-10-01): the same chains as the dense items with
their zero terms left out.

`PageRank.fit` took a dense n x n adjacency, row-normalized it into a dense
Q on the device (an upload, a kernel and a download of the whole matrix),
scanned it again for dangling rows (another upload), then uploaded Q once
more and ran `pagerank_step_item` for every node per iteration: a chain of n
multiply-adds per node over a column of Q, plus a chain of n adds for the
dangling mass in EVERY node's thread. At the board's 20,000 nodes that is
four 1.6 GB transfers and a 400-million-cell step per iteration for a graph
of 300,000 edges (1,358 ms on the L40S; cuGraph 4.7 ms).

Here the dense matrix is scanned ONCE on the host (rows over host tasks):
each row's nonzero cells in ascending column order, the row sum folded over
them, the normalized values; then the transpose to columns (rows ascending
within a column); then the iteration reads only those cells. EVERY CHAIN IS
THE ITEM'S WITH ITS ZERO TERMS LEFT OUT, which moves no bit:
- the row sum `s = _add(s, a[j])` ascending j: a +0.0 or -0.0 term leaves
  `s` as it is (s starts at +0.0 and +0.0 + -0.0 is +0.0, so s is never
  -0.0); a denormal is not zero and stays in;
- `row_all_zero` (dangling) is "no cell with a nonzero bit pattern", the
  same test as "no nonzero cell";
- the node's fold `acc = fma(ftz(x[i]), ftz(q[i, t]), acc)` ascending i: a
  cell with q = 0 contributes x[i] * 0 = +-0.0 and leaves `acc` as it is
  (acc starts at +0.0 and never becomes -0.0 from a +0.0 start); a cell
  whose q is denormal flushes to 0 in the item and contributes the same
  nothing, so it is left out here as well (the stored values are the
  flushed quotients, exactly what the item reads);
- the dangling mass `dsum = _add(dsum, x[i])` over the dangling nodes
  ascending is the same chain, computed once per iteration instead of once
  per node.
A non-finite x[i] times a zero cell would be NaN in the item and nothing
here, but a non-finite iterate never converges in either path (NaN and inf
compare false against the threshold), so `fit` raises before the bits are
seen.
"""
from std.memory import bitcast

from checks.numerics import ftz, identical_div, identical_mul, identical_mul_add
from core.host_lanes import HOST_FW, U32V, host_row_tasks
from core.host_parallel import host_parallelize
from x_neighbors.items import FP, IP, _add, _sub

comptime _MAG = U32V(0x7FFFFFFF)
comptime _IntPtr = MutPointer[Int, MutUntrackedOrigin]
comptime _I32Ptr = MutPointer[Int32, MutUntrackedOrigin]
comptime _F32Ptr = MutPointer[Float32, MutUntrackedOrigin]


def _ip(xs: List[Int]) -> _IntPtr:
    return _IntPtr(unsafe_from_address=Int(xs.unsafe_ptr()))


def _i32p(xs: List[Int32]) -> _I32Ptr:
    return _I32Ptr(unsafe_from_address=Int(xs.unsafe_ptr()))


def _f32p(xs: List[Float32]) -> _F32Ptr:
    return _F32Ptr(unsafe_from_address=Int(xs.unsafe_ptr()))


struct PrGraph(Movable):
    """The row-normalized adjacency by columns: column t's cells are
    `rows[indptr[t] .. indptr[t + 1])` ascending with values `vals`;
    `dangling[i]` is 1 for a row with no nonzero cell."""
    var n: Int
    var nnz: Int
    var indptr: List[Int32]
    var rows: List[Int32]
    var vals: List[Float32]
    var dangling: List[Int32]

    def __init__(out self, n: Int):
        self.n = n
        self.nnz = 0
        self.indptr = List[Int32](length=n + 1, fill=Int32(0))
        self.rows = List[Int32](length=1, fill=Int32(0))
        self.vals = List[Float32](length=1, fill=Float32(0))
        self.dangling = List[Int32](length=n if n > 0 else 1, fill=Int32(0))


def pr_graph_from_dense(a: FP, n: Int, binary: Bool) -> PrGraph:
    """The scan of the dense adjacency: rows over host tasks (a count pass,
    the prefix, a fill pass), then the transpose. `binary` reads every
    nonzero cell as 1.0 (PageRank's `weight=False`), as the dense path's
    rewrite of the matrix did."""
    var g = PrGraph(n)
    var counts = List[Int](length=n if n > 0 else 1, fill=0)
    var sums = List[Float32](length=n if n > 0 else 1, fill=Float32(0))
    var cp = _ip(counts)
    var sp = _f32p(sums)
    var tasks = host_row_tasks(n, 2 * n)
    var chunk = (n + tasks - 1) // tasks
    def _count(task: Int) {imm a, imm cp, imm sp, imm n, imm chunk, imm binary}:
        for t in range(task * chunk, min((task + 1) * chunk, n)):
            var c = 0
            var s = Float32(0)
            var base = t * n
            # a chunk of HOST_FW cells with no nonzero bit pattern holds only
            # +-0.0 cells, which the fold leaves out; a chunk with one is
            # walked cell by cell in order
            var j = 0
            while j + HOST_FW <= n:
                var bits = bitcast[DType.uint32](a.unsafe_load[width=HOST_FW](base + j)) & _MAG
                if bits.reduce_or() != UInt32(0):
                    for q in range(j, j + HOST_FW):
                        var v = a.unsafe_load(base + q)
                        if v != Float32(0):
                            c += 1
                            s = _add(s, Float32(1) if binary else v)
                j += HOST_FW
            while j < n:
                var v = a.unsafe_load(base + j)
                if v != Float32(0):
                    c += 1
                    s = _add(s, Float32(1) if binary else v)
                j += 1
            cp.unsafe_store(t, c)
            sp.unsafe_store(t, s)
    if tasks <= 1:
        _count(0)
    else:
        host_parallelize(_count, tasks)
    var rowptr = List[Int](length=n + 1, fill=0)
    var total = 0
    for t in range(n):
        rowptr[t] = total
        total += counts[t]
        g.dangling[t] = Int32(1 if counts[t] == 0 else 0)
    rowptr[n] = total
    g.nnz = total
    var cols = List[Int32](length=total if total > 0 else 1, fill=Int32(0))
    var rvals = List[Float32](length=total if total > 0 else 1, fill=Float32(0))
    var colp = _i32p(cols)
    var rvp = _f32p(rvals)
    var rpp = _ip(rowptr)
    def _fill(task: Int) {imm a, imm colp, imm rvp, imm rpp, imm sp, imm n, imm chunk, imm binary}:
        for t in range(task * chunk, min((task + 1) * chunk, n)):
            var w = rpp.unsafe_load(t)
            var s = sp.unsafe_load(t)
            if s == Float32(0):
                s = Float32(1)
            var base = t * n
            var j = 0
            while j + HOST_FW <= n:
                var bits = bitcast[DType.uint32](a.unsafe_load[width=HOST_FW](base + j)) & _MAG
                if bits.reduce_or() != UInt32(0):
                    for q in range(j, j + HOST_FW):
                        var v = a.unsafe_load(base + q)
                        if v != Float32(0):
                            colp.unsafe_store(w, Int32(q))
                            rvp.unsafe_store(w, ftz(identical_div(ftz(Float32(1) if binary else v), s)))
                            w += 1
                j += HOST_FW
            while j < n:
                var v = a.unsafe_load(base + j)
                if v != Float32(0):
                    colp.unsafe_store(w, Int32(j))
                    rvp.unsafe_store(w, ftz(identical_div(ftz(Float32(1) if binary else v), s)))
                    w += 1
                j += 1
    if tasks <= 1:
        _fill(0)
    else:
        host_parallelize(_fill, tasks)
    # the transpose: cells of column t in ascending row order
    var ccount = List[Int](length=n if n > 0 else 1, fill=0)
    for e in range(total):
        ccount[Int(cols[e])] += 1
    var run = 0
    for t in range(n):
        g.indptr[t] = Int32(run)
        run += ccount[t]
        ccount[t] = Int(g.indptr[t])
    g.indptr[n] = Int32(run)
    g.rows = List[Int32](length=total if total > 0 else 1, fill=Int32(0))
    g.vals = List[Float32](length=total if total > 0 else 1, fill=Float32(0))
    for t in range(n):
        for e in range(rowptr[t], rowptr[t + 1]):
            var j = Int(cols[e])
            var pos = ccount[j]
            g.rows[pos] = Int32(t)
            g.vals[pos] = rvals[e]
            ccount[j] = pos + 1
    _ = counts^
    _ = sums^
    _ = rowptr^
    _ = cols^
    _ = rvals^
    _ = ccount^
    return g^


@always_inline
def pagerank_dangling_sum(x: FP, dangling: IP, n: Int) -> Float32:
    """The item's `dsum` chain: the dangling nodes' x ascending."""
    var dsum = Float32(0)
    for i in range(n):
        if Int(dangling.unsafe_load(i)) != 0:
            dsum = _add(dsum, x.unsafe_load(i))
    return dsum


@always_inline
def pagerank_step_sparse_item(
    t: Int, indptr: IP, rows: IP, vals: FP, x: FP, p: FP, dw: FP, dsum: Float32, res: FP, alpha: Float32,
):
    """`pagerank_step_item` for node t over column t's nonzero cells."""
    var acc = Float32(0)
    var lo = Int(indptr.unsafe_load(t))
    var hi = Int(indptr.unsafe_load(t + 1))
    for e in range(lo, hi):
        acc = ftz(identical_mul_add(ftz(x.unsafe_load(Int(rows.unsafe_load(e)))), ftz(vals.unsafe_load(e)), acc))
    var pt = ftz(p.unsafe_load(t))
    var inner = ftz(identical_mul_add(dsum, ftz(dw.unsafe_load(t)), acc))
    var teleport = ftz(identical_mul(_sub(Float32(1), alpha), pt))
    res.unsafe_store(t, ftz(identical_mul_add(alpha, inner, teleport)))
