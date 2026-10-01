# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`op_cc_iterate` (connected_components' min-label iteration of
`cc_step_item`, x_neighbors/items.mojo) on the host as a SPARSE walk: the
same rounds over the graph's edges instead of every cell of the n x n matrix
(lane neural-pass22, 2026-10-01).

The board's race is a 20,000-node kNN graph (`GRAPH_SMALL`) stored dense, and
`cc_step_item(t)` reads row t AND column t of the matrix for every node in
every round: 800 million cell reads a round, tens of rounds, for 1 to 3.5
seconds on every box where networkx takes 5 to 13 ms. The step is a MINIMUM
over integer labels (the node's own, then every neighbor's in either
direction), so neither the order of the neighbors nor the zero cells touch
the result: a node's new label is the same integer whichever way its
neighborhood is enumerated. This walk builds, once, the row adjacency (the
nonzero cells of row t) and the column adjacency (the nonzero cells of column
t), rows and column blocks over host tasks, and then runs the rounds over
those lists, nodes over tasks, counting a round exactly as the dense loop
does (the round that changes nothing is counted, then the loop stops). The
dense item stays the device kernel's body and the reference;
`x_neighbors/checks/cc_sparse_check.mojo` holds this walk to it, labels and
step count, on rings, directed random graphs and graphs with isolated nodes.
"""

from core.host_lanes import host_row_tasks
from core.host_parallel import host_parallelize
from x_neighbors.items import FP, IP

comptime _IntPtr = MutPointer[Int, MutUntrackedOrigin]
comptime _I32Ptr = MutPointer[Int32, MutUntrackedOrigin]


def _ip(xs: List[Int]) -> _IntPtr:
    return _IntPtr(unsafe_from_address=Int(xs.unsafe_ptr()))


def _i32p(xs: List[Int32]) -> _I32Ptr:
    return _I32Ptr(unsafe_from_address=Int(xs.unsafe_ptr()))


struct _Adj(Movable):
    """The nonzero positions of each row (or each column) of a dense n x n
    matrix: `cols[indptr[t] .. indptr[t + 1])`, ascending."""

    var indptr: List[Int]
    var cols: List[Int32]

    def __init__(out self, n: Int):
        self.indptr = List[Int](length=n + 1, fill=0)
        self.cols = List[Int32]()


def _row_adj(a: FP, n: Int) -> _Adj:
    """The nonzero cells of every row, rows over host tasks: a count pass,
    the prefix, a fill pass."""
    var g = _Adj(n)
    var counts = List[Int](length=n, fill=0)
    var cp = _ip(counts)
    var tasks = host_row_tasks(n, 2 * n)
    var chunk = (n + tasks - 1) // tasks

    def _count(task: Int) {imm a, imm cp, imm n, imm chunk}:
        for t in range(task * chunk, min((task + 1) * chunk, n)):
            var c = 0
            var base = t * n
            for j in range(n):
                if a.unsafe_load(base + j) != Float32(0):
                    c += 1
            cp.unsafe_store(t, c)

    if tasks <= 1:
        _count(0)
    else:
        host_parallelize(_count, tasks)
    var total = 0
    for t in range(n):
        g.indptr[t] = total
        total += counts[t]
    g.indptr[n] = total
    g.cols = List[Int32](length=total if total > 0 else 1, fill=Int32(0))
    var colp = _i32p(g.cols)
    var ipp = _ip(g.indptr)

    def _fill(task: Int) {imm a, imm colp, imm ipp, imm n, imm chunk}:
        for t in range(task * chunk, min((task + 1) * chunk, n)):
            var w = ipp.unsafe_load(t)
            var base = t * n
            for j in range(n):
                if a.unsafe_load(base + j) != Float32(0):
                    colp.unsafe_store(w, Int32(j))
                    w += 1

    if tasks <= 1:
        _fill(0)
    else:
        host_parallelize(_fill, tasks)
    _ = counts^
    return g^


def _col_adj(a: FP, n: Int) -> _Adj:
    """The nonzero cells of every column (the rows i with a[i, t] != 0),
    column BLOCKS over host tasks so every count and cursor is owned by one
    task; each task walks the matrix row-major over its own columns."""
    var g = _Adj(n)
    var counts = List[Int](length=n, fill=0)
    var cp = _ip(counts)
    var tasks = host_row_tasks(n, 2 * n)
    var chunk = (n + tasks - 1) // tasks

    def _count(task: Int) {imm a, imm cp, imm n, imm chunk}:
        var c0 = task * chunk
        var c1 = min((task + 1) * chunk, n)
        if c1 <= c0:
            return
        for i in range(n):
            var base = i * n
            for j in range(c0, c1):
                if a.unsafe_load(base + j) != Float32(0):
                    cp.unsafe_store(j, cp.unsafe_load(j) + 1)

    if tasks <= 1:
        _count(0)
    else:
        host_parallelize(_count, tasks)
    var total = 0
    for t in range(n):
        g.indptr[t] = total
        total += counts[t]
    g.indptr[n] = total
    g.cols = List[Int32](length=total if total > 0 else 1, fill=Int32(0))
    var colp = _i32p(g.cols)
    var cursor = List[Int](length=n, fill=0)
    for t in range(n):
        cursor[t] = g.indptr[t]
    var curp = _ip(cursor)

    def _fill(task: Int) {imm a, imm colp, imm curp, imm n, imm chunk}:
        var c0 = task * chunk
        var c1 = min((task + 1) * chunk, n)
        if c1 <= c0:
            return
        for i in range(n):
            var base = i * n
            for j in range(c0, c1):
                if a.unsafe_load(base + j) != Float32(0):
                    var w = curp.unsafe_load(j)
                    colp.unsafe_store(w, Int32(i))
                    curp.unsafe_store(j, w + 1)

    if tasks <= 1:
        _fill(0)
    else:
        host_parallelize(_fill, tasks)
    _ = counts^
    _ = cursor^
    return g^


def cc_iterate_sparse(a: FP, lab: IP, info: IP, n: Int):
    """`op_cc_iterate`'s contract (module note): `lab` in 0..n-1 (any start
    labels), out the fixed point of the min-label rounds; `info[0]` the
    number of rounds run, the last one the round that changed nothing."""
    var rows = _row_adj(a, n)
    var cols = _col_adj(a, n)
    var l0 = List[Int32](length=n if n > 0 else 1, fill=Int32(0))
    var l1 = List[Int32](length=n if n > 0 else 1, fill=Int32(0))
    for i in range(n):
        l0[i] = lab.unsafe_load(i)
    var cur = _i32p(l0)
    var nxt = _i32p(l1)
    var rip = _ip(rows.indptr)
    var rcol = _i32p(rows.cols)
    var cip = _ip(cols.indptr)
    var ccol = _i32p(cols.cols)
    var edges = rows.indptr[n] + cols.indptr[n]
    var tasks = host_row_tasks(n, 2 + (2 * edges) // max(n, 1))
    var chunk = (n + tasks - 1) // tasks
    var steps = 0
    while True:
        var src = cur
        var dst = nxt

        def _round(task: Int) {imm src, imm dst, imm rip, imm rcol, imm cip, imm ccol, imm n, imm chunk}:
            for t in range(task * chunk, min((task + 1) * chunk, n)):
                var best = src.unsafe_load(t)
                for e in range(rip.unsafe_load(t), rip.unsafe_load(t + 1)):
                    var l = src.unsafe_load(Int(rcol.unsafe_load(e)))
                    if l < best:
                        best = l
                for e in range(cip.unsafe_load(t), cip.unsafe_load(t + 1)):
                    var l = src.unsafe_load(Int(ccol.unsafe_load(e)))
                    if l < best:
                        best = l
                dst.unsafe_store(t, best)

        if tasks <= 1:
            _round(0)
        else:
            host_parallelize(_round, tasks)
        steps += 1
        var same = True
        for i in range(n):
            if nxt.unsafe_load(i) != cur.unsafe_load(i):
                same = False
                break
        if same:
            break
        var tmp = cur
        cur = nxt
        nxt = tmp
    for i in range(n):
        lab.unsafe_store(i, cur.unsafe_load(i))
    info.unsafe_store(0, Int32(steps))
    _ = l0^
    _ = l1^
    _ = rows^
    _ = cols^
