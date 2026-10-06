"""PageRank and Louvain as parallel items in a fixed order (lane hr-graph,
2026-10-02, docs/plans/HOST_ROUTE_REMOVAL.md).

Every stage is an item `t` of a launch; the device runs the items as threads
(x_neighbors/graph_dev.mojo), the host column runs the SAME items in
ascending t (x_neighbors/graph_host.mojo), and ONE driver (`pr_drive`,
`lv_drive`, generic over `GExec`) issues the stages in the same order with the
same control decisions on both. No item reads what another item of the same
launch writes (or reads a value that is the same whichever it sees), so the
two agree bit for bit by construction. Nothing here imports a GPU module.

Stage arguments: `GA` holds three arenas (float32, int32, int64), the layout
(the offset of every named array, `Lay`), the dense input and eight integer
and two float scalars. A stage's scalars are documented on its item.

PageRank (replaces lane neural-pass30's host scan): the dense
adjacency is read on the device, rows for the row sums (each row's nonzero
cells ascending, the dense item's chain with its zero terms left out),
columns for the transposed lists (column t's nonzero cells, rows
ascending), a fixed integer scan for the column offsets. Each step is one
item per node over its column list ascending (`pr_step_csr`, the dense
`pagerank_step_item` with its zero terms left out); the dangling mass and
the convergence sum |x' - x| are blocked folds (GP_FOLD_BLOCK elements per
item ascending from zero, then the partials ascending: the order of lane
neural-pass141's `absdiff_sum`).

Louvain (replaces the one sequential sweep, DEVIATION 5204): local moving
by a distance-1 graph colouring (Jones-Plassmann rounds, a fixed integer
hash priority, ties to the larger id); the vertices of one colour move
together (no two are adjacent, so each sees its neighbours' communities
exactly), each to the candidate community of strictly larger gain scanned
in ascending id (equal gains: the smallest id). Its k_i,in per community
is the fold of its edges to that community in ascending column order
(an in-thread heap sort of (community, column) keys). The community
totals are recomputed after every colour by a segmented fold: the vertices
sorted by (community, id) (a merge sort by rank, unique keys), each
community's degrees folded ascending by its first position, no atomics.
A sweep that lowers the modularity is undone and ends the level; one that
raises it by no more than `threshold` ends it. Aggregation sorts the edge
records by (community pair, edge index) and folds each pair's weights in
that order (each undirected edge once, an edge inside a community once as
its self-loop). Levels, the renumbering by ascending old id and the
`threshold` test on the level's modularity are networkx's
`louvain_partitions`. NEW BITS versus the sequential sweep (labels,
modularity and level count).
"""
from experiments.classical_identical_ideas.graph_controls import C43_RESIDENT_NORMALIZATION
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632

from checks.numerics import ftz, identical_div, identical_mul, identical_mul_add
from x_neighbors.items import FP, IP, _add, _sub

comptime LP = MutPointer[Int64, MutAnyOrigin]
comptime FPU = MutPointer[Float32, MutUntrackedOrigin]
comptime IPU = MutPointer[Int32, MutUntrackedOrigin]
comptime LPU = MutPointer[Int64, MutUntrackedOrigin]

comptime GP_FOLD_BLOCK = 2048
"""Elements per item of a blocked float fold (lane neural-pass141's
XN_FOLD_BLOCK: the same order as its `absdiff_sum`)."""
comptime GP_SCAN_BLOCK = 256
"""Elements per item of the integer scans."""
comptime GP_SORT_RUN = 16
"""Keys per item of the merge sort's first (insertion) pass."""
comptime LV_MAX_SWEEPS = 100
"""Sweeps per Louvain level at most (each must raise the modularity)."""
comptime LV_COLOR_CHUNK = 8
"""Colouring rounds launched between two reads of the uncoloured flag."""


@always_inline
def gp_fold_blocks(count: Int) -> Int:
    return (count + GP_FOLD_BLOCK - 1) // GP_FOLD_BLOCK if count > 0 else 0


@always_inline
def gp_scan_blocks(count: Int) -> Int:
    return (count + GP_SCAN_BLOCK - 1) // GP_SCAN_BLOCK if count > 0 else 0


@fieldwise_init
struct GA(TrivialRegisterPassable):
    """One stage's arguments (module note): the arenas, the layout and the
    dense input as typed pointers. Never an integer address: on Metal a
    pointer rebuilt from an integer inside a kernel does not reach device
    memory (every load reads 0, every store is lost). And every function that
    takes a GA or a pointer from it is @always_inline: across a real call
    the pointer is a generic-address-space argument, which Metal's AIR has
    no lowering for (the metallib compiler crashes on any such stage)."""
    var f: FPU
    var i: IPU
    var l: LPU
    var lay: LPU
    var a: FPU
    var n0: Int
    var n1: Int
    var n2: Int
    var n3: Int
    var n4: Int
    var n5: Int
    var n6: Int
    var n7: Int
    var x0: Float32
    var x1: Float32


struct Lay(Movable):
    """The arenas' layout: slot -> offset in its arena, and the arena sizes."""
    var off: List[Int64]
    var nf: Int
    var ni: Int
    var nl: Int

    def __init__(out self, nslots: Int):
        self.off = List[Int64](length=nslots, fill=Int64(0))
        self.nf = 0
        self.ni = 0
        self.nl = 0

    def f(mut self, slot: Int, size: Int):
        self.off[slot] = Int64(self.nf)
        self.nf += max(size, 1)

    def i(mut self, slot: Int, size: Int):
        self.off[slot] = Int64(self.ni)
        self.ni += max(size, 1)

    def l(mut self, slot: Int, size: Int):
        self.off[slot] = Int64(self.nl)
        self.nl += max(size, 1)


trait GExec:
    """Where the stages run: the device (graph_dev.mojo) or the host column
    (graph_host.mojo). `alloc` replaces the arenas (the previous GA's
    pointers die with them)."""

    def alloc(mut self, lay: Lay) raises -> GA:
        ...

    def run(mut self, stage: Int, count: Int, g: GA) raises:
        ...

    def get_i(mut self, slot: Int, idx: Int) raises -> Int:
        ...

    def get_f(mut self, slot: Int, idx: Int) raises -> Float32:
        ...

    def get_is(mut self, slot: Int, count: Int) raises -> List[Int32]:
        ...

    def up_f(mut self, slot: Int, addr: Int, count: Int) raises:
        ...

    def down_f(mut self, slot: Int, addr: Int, count: Int) raises:
        ...

    def down_i(mut self, slot: Int, addr: Int, count: Int) raises:
        ...


@always_inline
def _off(g: GA, slot: Int) -> Int:
    return Int(g.lay.unsafe_load(slot))


@always_inline
def _fs(g: GA, slot: Int) -> FP:
    return g.f.unsafe_offset(_off(g, slot)).unsafe_origin_cast[MutAnyOrigin]()


@always_inline
def _is(g: GA, slot: Int) -> IP:
    return g.i.unsafe_offset(_off(g, slot)).unsafe_origin_cast[MutAnyOrigin]()


@always_inline
def _ls(g: GA, slot: Int) -> LP:
    return g.l.unsafe_offset(_off(g, slot)).unsafe_origin_cast[MutAnyOrigin]()


@always_inline
def _ain(g: GA) -> FP:
    return g.a.unsafe_origin_cast[MutAnyOrigin]()


# ------------------------------------------------------------------ stage ids
comptime GP_SCAN_PART = 0
comptime GP_SCAN_MID = 1
comptime GP_SCAN_FIN = 2
comptime GP_SORT_RUN_ST = 3
comptime GP_SORT_MERGE = 4
comptime GP_FPART = 5
comptime GP_FFIN = 6
comptime PR_ROWSUM = 7
comptime PR_COLCNT = 8
comptime PR_COLFILL = 9
comptime PR_DPART = 10
comptime PR_STEP = 11
comptime PR_APART = 12
comptime LV_ROWCNT = 13
comptime LV_FILL = 14
comptime LV_MROW = 15
comptime LV_DEG = 16
comptime LV_COL0 = 17
comptime LV_COLR = 18
comptime LV_ICZ = 19
comptime LV_UNCOL = 20
comptime LV_CKEY = 21
comptime LV_COFF = 22
comptime LV_COPYL = 23
comptime LV_SAVE = 24
comptime LV_MOVE = 25
comptime LV_SKEY = 26
comptime LV_SFOLD = 27
comptime LV_QV = 28
comptime LV_RESTORE = 29
comptime LV_FLAGZ = 30
comptime LV_USED = 31
comptime LV_RENUM = 32
comptime LV_LAB = 33
comptime LV_EKEY = 34
comptime LV_HEAD = 35
comptime LV_AGG = 36
comptime LV_AIP = 37
comptime LV_COPYI = 38
comptime GP_NST = 39


# ------------------------------------------------------------------ generic stages
@always_inline
def _scan_part(t: Int, g: GA):
    """Integer exclusive scan, stage 1 of 3, one item per GP_SCAN_BLOCK
    block: n0 count, n4 the array (count + 1 entries), n5 the parts (blocks
    + 1): parts[t] = the block's sum."""
    var x = _is(g, g.n4)
    var lo = t * GP_SCAN_BLOCK
    var hi = min(lo + GP_SCAN_BLOCK, g.n0)
    var s = 0
    for k in range(lo, hi):
        s += Int(x.unsafe_load(k))
    _is(g, g.n5).unsafe_store(t, Int32(s))


@always_inline
def _scan_mid(t: Int, g: GA):
    """Stage 2 of 3, ONE item: the parts scanned exclusively in place, the
    total into x[count]. n1 the number of blocks."""
    var p = _is(g, g.n5)
    var run = 0
    for b in range(g.n1):
        var v = Int(p.unsafe_load(b))
        p.unsafe_store(b, Int32(run))
        run += v
    _is(g, g.n4).unsafe_store(g.n0, Int32(run))


@always_inline
def _scan_fin(t: Int, g: GA):
    """Stage 3 of 3, one item per block: the block's exclusive prefix."""
    var x = _is(g, g.n4)
    var run = Int(_is(g, g.n5).unsafe_load(t))
    var lo = t * GP_SCAN_BLOCK
    var hi = min(lo + GP_SCAN_BLOCK, g.n0)
    for k in range(lo, hi):
        var v = Int(x.unsafe_load(k))
        x.unsafe_store(k, Int32(run))
        run += v


@always_inline
def _sort_run(t: Int, g: GA):
    """Merge sort of UNIQUE int64 keys, pass 0, one item per GP_SORT_RUN
    keys: an insertion sort of the run in place. n0 count, n4 the keys."""
    var k = _ls(g, g.n4)
    var lo = t * GP_SORT_RUN
    var hi = min(lo + GP_SORT_RUN, g.n0)
    for a in range(lo + 1, hi):
        var v = k.unsafe_load(a)
        var b = a
        while b > lo and k.unsafe_load(b - 1) > v:
            k.unsafe_store(b, k.unsafe_load(b - 1))
            b -= 1
        k.unsafe_store(b, v)


@always_inline
def _sort_merge(t: Int, g: GA):
    """One merge pass, one item per key: n0 count, n1 the run width, n4 the
    source, n5 the destination. A key's place in the merged pair is its
    index in its run plus the number of the other run's keys below it (the
    keys are unique, so every key lands on its own place)."""
    var src = _ls(g, g.n4)
    var dst = _ls(g, g.n5)
    var w = g.n1
    var key = src.unsafe_load(t)
    var run = t // w
    var lo = run * w
    var plo: Int
    var phi: Int
    var base: Int
    if run % 2 == 0:
        plo = lo + w
        phi = min(plo + w, g.n0)
        base = lo
    else:
        plo = lo - w
        phi = lo
        base = plo
    if phi < plo:
        phi = plo
    var a = plo
    var b = phi
    while a < b:
        var mid = (a + b) // 2
        if src.unsafe_load(mid) < key:
            a = mid + 1
        else:
            b = mid
    dst.unsafe_store(base + (t - lo) + (a - plo), key)


@always_inline
def _fpart(t: Int, g: GA):
    """Blocked float fold, stage 1: n0 count, n4 the values, n5 the parts:
    block t ascending from zero."""
    var x = _fs(g, g.n4)
    var lo = t * GP_FOLD_BLOCK
    var hi = min(lo + GP_FOLD_BLOCK, g.n0)
    var acc = Float32(0)
    for k in range(lo, hi):
        acc = _add(acc, x.unsafe_load(k))
    _fs(g, g.n5).unsafe_store(t, acc)


@always_inline
def _ffin(t: Int, g: GA):
    """Blocked float fold, stage 2, ONE item: n1 parts in slot n5 folded
    ascending from zero into slot n6 at index n7."""
    var p = _fs(g, g.n5)
    var acc = Float32(0)
    for b in range(g.n1):
        acc = _add(acc, p.unsafe_load(b))
    _fs(g, g.n6).unsafe_store(g.n7, acc)


# ------------------------------------------------------------------ PageRank
comptime P_CNT = 0
comptime P_PART = 1
comptime P_RS = 2
comptime P_DG = 3
comptime P_ROWS = 4
comptime P_VALS = 5
comptime P_XA = 6
comptime P_XB = 7
comptime P_P = 8
comptime P_DW = 9
comptime P_FP = 10
comptime P_SUM = 11
comptime P_NSLOT = 12


@always_inline
def _pr_rowsum(t: Int, g: GA):
    """Row t's sum over its nonzero cells ascending (1.0 each when n1, the
    unweighted graph) and its dangling flag. n0 = n."""
    var n = g.n0
    var s = Float32(0)
    var c = 0
    for j in range(n):
        var v = _ain(g).unsafe_load(t * n + j)
        if v != Float32(0):
            c += 1
            s = _add(s, Float32(1) if g.n1 != 0 else v)
    _fs(g, P_RS).unsafe_store(t, s)
    _is(g, P_DG).unsafe_store(t, Int32(1 if c == 0 else 0))


@always_inline
def _pr_colcnt(t: Int, g: GA):
    """Column t's nonzero cells counted (consecutive items read consecutive
    words of a row)."""
    var n = g.n0
    var c = 0
    for i in range(n):
        if _ain(g).unsafe_load(i * n + t) != Float32(0):
            c += 1
    _is(g, P_CNT).unsafe_store(t, Int32(c))


@always_inline
def _pr_colfill(t: Int, g: GA):
    """Column t's list from its offset: the rows of its nonzero cells
    ascending and the row-normalized values: the flushed quotient by the
    row sum (by 1 for an all-zero sum)."""
    var n = g.n0
    var rows = _is(g, P_ROWS)
    var vals = _fs(g, P_VALS)
    var rs = _fs(g, P_RS)
    var w = Int(_is(g, P_CNT).unsafe_load(t))
    for i in range(n):
        var v = _ain(g).unsafe_load(i * n + t)
        if v != Float32(0):
            var s = rs.unsafe_load(i)
            if s == Float32(0):
                s = Float32(1)
            rows.unsafe_store(w, Int32(i))
            comptime if C43_RESIDENT_NORMALIZATION:
                vals.unsafe_store(w,ftz(Float32(1) if g.n1!=0 else v))
            else:
                vals.unsafe_store(w, ftz(identical_div(ftz(Float32(1) if g.n1 != 0 else v), s)))
            w += 1


@always_inline
def _pr_dpart(t: Int, g: GA):
    """The dangling mass, block t: the dangling nodes' x (slot n4)
    ascending from zero into P_FP."""
    var x = _fs(g, g.n4)
    var dg = _is(g, P_DG)
    var lo = t * GP_FOLD_BLOCK
    var hi = min(lo + GP_FOLD_BLOCK, g.n0)
    var acc = Float32(0)
    for i in range(lo, hi):
        if Int(dg.unsafe_load(i)) != 0:
            acc = _add(acc, x.unsafe_load(i))
    _fs(g, P_FP).unsafe_store(t, acc)


@always_inline
def pr_step_csr(
    t: Int, indptr: IP, rows: IP, vals: FP, x: FP, p: FP, dw: FP, dsum: Float32, res: FP, alpha: Float32,
):
    """`pagerank_step_item` for node t over column t's nonzero cells
    ascending: the dense item's chain with its zero terms left out (a zero
    cell adds x * 0 = +0.0 to a +0.0-seeded chain; a denormal cell is
    flushed to 0 in the item, and its stored quotient is flushed here)."""
    var acc = Float32(0)
    var lo = Int(indptr.unsafe_load(t))
    var hi = Int(indptr.unsafe_load(t + 1))
    for e in range(lo, hi):
        acc = ftz(identical_mul_add(ftz(x.unsafe_load(Int(rows.unsafe_load(e)))), ftz(vals.unsafe_load(e)), acc))
    var pt = ftz(p.unsafe_load(t))
    var inner = ftz(identical_mul_add(dsum, ftz(dw.unsafe_load(t)), acc))
    var teleport = ftz(identical_mul(_sub(Float32(1), alpha), pt))
    res.unsafe_store(t, ftz(identical_mul_add(alpha, inner, teleport)))


@always_inline
def _pr_step(t: Int, g: GA):
    """One node of a step: x in slot n4, the next iterate into slot n5,
    alpha x0, the dangling mass from P_SUM[0]."""
    comptime if C43_RESIDENT_NORMALIZATION:
        var offsets=_is(g,P_CNT)
        var rows=_is(g,P_ROWS)
        var values=_fs(g,P_VALS)
        var degrees=_fs(g,P_RS)
        var x=_fs(g,g.n4)
        var acc=Float32(0)
        for edge in range(Int(offsets[t]),Int(offsets[t+1])):
            var row=Int(rows[edge])
            var degree=degrees[row]
            if degree==Float32(0):
                degree=Float32(1)
            var value=ftz(identical_div(ftz(values[edge]),degree))
            acc=ftz(identical_mul_add(ftz(x[row]),value,acc))
        var inner=ftz(identical_mul_add(_fs(g,P_SUM)[0],ftz(_fs(g,P_DW)[t]),acc))
        var teleport=ftz(identical_mul(_sub(Float32(1),g.x0),ftz(_fs(g,P_P)[t])))
        _fs(g,g.n5)[t]=ftz(identical_mul_add(g.x0,inner,teleport))
        return
    pr_step_csr(t, _is(g, P_CNT), _is(g, P_ROWS), _fs(g, P_VALS), _fs(g, g.n4), _fs(g, P_P), _fs(g, P_DW),
                _fs(g, P_SUM).unsafe_load(0), _fs(g, g.n5), g.x0)


@always_inline
def _pr_apart(t: Int, g: GA):
    """sum |x' - x|, block t (x' slot n5, x slot n4) ascending from zero
    into P_FP (neural-pass141's `_absdiff_block(x', x)`)."""
    var a = _fs(g, g.n5)
    var b = _fs(g, g.n4)
    var lo = t * GP_FOLD_BLOCK
    var hi = min(lo + GP_FOLD_BLOCK, g.n0)
    var acc = Float32(0)
    for i in range(lo, hi):
        acc = _add(acc, abs(_sub(a.unsafe_load(i), b.unsafe_load(i))))
    _fs(g, P_FP).unsafe_store(t, acc)


# ------------------------------------------------------------------ Louvain
# A graph is four consecutive slots: indptr (int32, nodes + 1), the source
# row of every entry, the column, the weight (rows ascending, each row's
# columns ascending, both directions of an undirected edge stored).
comptime L_G0 = 0
comptime L_GA = 4
comptime L_GB = 8
comptime L_PART = 12
comptime L_COMM = 13
comptime L_SAVE = 14
comptime L_COLOR = 15
comptime L_LAB = 16
comptime L_FLAG = 17
comptime L_SEGH = 18
comptime L_COFF = 19
comptime L_IC = 20
comptime L_DEG = 21
comptime L_STOT = 22
comptime L_SSAVE = 23
comptime L_VQ = 24
comptime L_FP = 25
comptime L_FV = 26
comptime L_KA = 27
comptime L_KB = 28
comptime L_EK = 29
comptime L_ORD = 30
comptime L_NSLOT = 31


@always_inline
def _gip(g: GA, b: Int) -> IP:
    return _is(g, b)


@always_inline
def _gsrc(g: GA, b: Int) -> IP:
    return _is(g, b + 1)


@always_inline
def _gcol(g: GA, b: Int) -> IP:
    return _is(g, b + 2)


@always_inline
def _gval(g: GA, b: Int) -> FP:
    return _fs(g, b + 3)


@always_inline
def _lv_rowcnt(t: Int, g: GA):
    """Row t of the dense n0 x n0 input: its nonzero cells counted into
    G0's indptr; labels[t] = t."""
    var n = g.n0
    var c = 0
    for j in range(n):
        if _ain(g).unsafe_load(t * n + j) != Float32(0):
            c += 1
    _gip(g, L_G0).unsafe_store(t, Int32(c))
    _is(g, L_LAB).unsafe_store(t, Int32(t))


@always_inline
def _lv_fill(t: Int, g: GA):
    """Row t's nonzero cells, columns ascending, into G0 from its offset."""
    var n = g.n0
    var w = Int(_gip(g, L_G0).unsafe_load(t))
    var src = _gsrc(g, L_G0)
    var col = _gcol(g, L_G0)
    var val = _gval(g, L_G0)
    for j in range(n):
        var v = _ain(g).unsafe_load(t * n + j)
        if v != Float32(0):
            src.unsafe_store(w, Int32(t))
            col.unsafe_store(w, Int32(j))
            val.unsafe_store(w, v)
            w += 1


@always_inline
def _lv_mrow(t: Int, g: GA):
    """Row t's share of m (each undirected edge once, a self-loop once):
    its weights at columns >= t ascending from zero, into VQ. n3 graph."""
    var ip = _gip(g, g.n3)
    var col = _gcol(g, g.n3)
    var val = _gval(g, g.n3)
    var s = Float32(0)
    for e in range(Int(ip.unsafe_load(t)), Int(ip.unsafe_load(t + 1))):
        if Int(col.unsafe_load(e)) >= t:
            s = _add(s, val.unsafe_load(e))
    _fs(g, L_VQ).unsafe_store(t, s)


@always_inline
def _lv_deg(t: Int, g: GA):
    """Node t's degree on graph n3 (its row ascending, then its self-loop
    once more, networkx's degree); when n1: its own community (comm = t,
    stot = the degree)."""
    var ip = _gip(g, g.n3)
    var col = _gcol(g, g.n3)
    var val = _gval(g, g.n3)
    var dg = Float32(0)
    var self_w = Float32(0)
    for e in range(Int(ip.unsafe_load(t)), Int(ip.unsafe_load(t + 1))):
        var wv = val.unsafe_load(e)
        dg = _add(dg, wv)
        if Int(col.unsafe_load(e)) == t:
            self_w = wv
    dg = _add(dg, self_w)
    _fs(g, L_DEG).unsafe_store(t, dg)
    if g.n1 != 0:
        _is(g, L_COMM).unsafe_store(t, Int32(t))
        _fs(g, L_STOT).unsafe_store(t, _add(Float32(0), dg))


@always_inline
def _prio(u: Int) -> UInt32:
    """The colouring's fixed priority of node u (an integer hash)."""
    var x = UInt32(u)
    x ^= x >> 16
    x *= UInt32(0x7FEB352D)
    x ^= x >> 15
    x *= UInt32(0x846CA68B)
    x ^= x >> 16
    return x


@always_inline
def _beats(v: Int, u: Int) -> Bool:
    var pv = _prio(v)
    var pu = _prio(u)
    return pv > pu or (pv == pu and v > u)


@always_inline
def _lv_col0(t: Int, g: GA):
    _is(g, L_COLOR).unsafe_store(t, Int32(-1))


@always_inline
def _lv_colr(t: Int, g: GA):
    """Colouring round n1 on graph n3: an uncoloured node whose priority
    beats every neighbour uncoloured at the round's start takes colour n1.
    A neighbour coloured in this same round reads as uncoloured (its colour
    is n1), so the decision is the same whichever value is seen."""
    var color = _is(g, L_COLOR)
    if Int(color.unsafe_load(t)) != -1:
        return
    var r = g.n1
    var ip = _gip(g, g.n3)
    var col = _gcol(g, g.n3)
    for e in range(Int(ip.unsafe_load(t)), Int(ip.unsafe_load(t + 1))):
        var v = Int(col.unsafe_load(e))
        if v == t:
            continue
        var cv = Int(color.unsafe_load(v))
        if (cv == -1 or cv == r) and _beats(v, t):
            return
    color.unsafe_store(t, Int32(r))


@always_inline
def _lv_icz(t: Int, g: GA):
    _is(g, L_IC).unsafe_store(g.n1, Int32(0))


@always_inline
def _lv_uncol(t: Int, g: GA):
    """IC[1] = 1 when node t is uncoloured (every writer stores 1)."""
    if Int(_is(g, L_COLOR).unsafe_load(t)) == -1:
        _is(g, L_IC).unsafe_store(1, Int32(1))


@always_inline
def _lv_ckey(t: Int, g: GA):
    """The key (colour, node) of node t into KA. n0 nodes."""
    _ls(g, L_KA).unsafe_store(t, Int64(_is(g, L_COLOR).unsafe_load(t)) * Int64(g.n0) + Int64(t))


@always_inline
def _lower_bound(k: LP, lo: Int, hi: Int, key: Int64) -> Int:
    var a = lo
    var b = hi
    while a < b:
        var mid = (a + b) // 2
        if k.unsafe_load(mid) < key:
            a = mid + 1
        else:
            b = mid
    return a


@always_inline
def _lv_coff(t: Int, g: GA):
    """Colour t's first position among the sorted keys (slot n4, n0 nodes)."""
    _is(g, L_COFF).unsafe_store(t, Int32(_lower_bound(_ls(g, g.n4), 0, g.n0, Int64(t) * Int64(g.n0))))


@always_inline
def _lv_copyl(t: Int, g: GA):
    _ls(g, L_ORD).unsafe_store(t, _ls(g, g.n4).unsafe_load(t))


@always_inline
def _lv_save(t: Int, g: GA):
    _is(g, L_SAVE).unsafe_store(t, _is(g, L_COMM).unsafe_load(t))
    _fs(g, L_SSAVE).unsafe_store(t, _fs(g, L_STOT).unsafe_load(t))


@always_inline
def _lv_restore(t: Int, g: GA):
    _is(g, L_COMM).unsafe_store(t, _is(g, L_SAVE).unsafe_load(t))
    _fs(g, L_STOT).unsafe_store(t, _fs(g, L_SSAVE).unsafe_load(t))


@always_inline
def _heap_sift(k: LP, lo: Int, start: Int, end: Int):
    var root = start
    while True:
        var child = 2 * root + 1
        if child >= end:
            return
        if child + 1 < end and k.unsafe_load(lo + child) < k.unsafe_load(lo + child + 1):
            child += 1
        if k.unsafe_load(lo + root) < k.unsafe_load(lo + child):
            var tmp = k.unsafe_load(lo + root)
            k.unsafe_store(lo + root, k.unsafe_load(lo + child))
            k.unsafe_store(lo + child, tmp)
            root = child
        else:
            return


@always_inline
def _heap_sort(k: LP, lo: Int, cnt: Int):
    """k[lo .. lo + cnt) ascending (in-thread heap sort)."""
    var s = cnt // 2 - 1
    while s >= 0:
        _heap_sift(k, lo, s, cnt)
        s -= 1
    var end = cnt - 1
    while end > 0:
        var tmp = k.unsafe_load(lo)
        k.unsafe_store(lo, k.unsafe_load(lo + end))
        k.unsafe_store(lo + end, tmp)
        _heap_sift(k, lo, 0, end)
        end -= 1


@always_inline
def lv_move_item(t: Int, g: GA):
    """One node of the current colour moves (networkx `_one_level`'s step
    for one node): n0 nodes, n1 the colour's first position in ORD, n3 the
    graph, x0 the resolution; m in FV[0]. Its neighbours' (community,
    column) keys are heap-sorted in its own edge range of EK, so each
    community's k_i,in is its edges folded in ascending column order and
    the candidates come in ascending community id; a strictly larger gain
    moves (equal gains: the smallest id; no gain: stay). The community
    totals are the colour's start values (the colour's other movers are
    not neighbours, so k_i,in is exact). Writes comm[u] only; IC[0] = 1 on
    a move."""
    var nn = g.n0
    var u = Int(_ls(g, L_ORD).unsafe_load(g.n1 + t) % Int64(nn))
    var ip = _gip(g, g.n3)
    var col = _gcol(g, g.n3)
    var val = _gval(g, g.n3)
    var comm = _is(g, L_COMM)
    var stot = _fs(g, L_STOT)
    var ek = _ls(g, L_EK)
    var lo = Int(ip.unsafe_load(u))
    var d = Int(ip.unsafe_load(u + 1)) - lo
    var cnt = 0
    for j in range(d):
        var v = Int(col.unsafe_load(lo + j))
        if v != u and val.unsafe_load(lo + j) != Float32(0):
            ek.unsafe_store(lo + cnt, Int64(comm.unsafe_load(v)) * Int64(d) + Int64(j))
            cnt += 1
    if cnt == 0:
        return
    _heap_sort(ek, lo, cnt)
    var m = _fs(g, L_FV).unsafe_load(0)
    var res = g.x0
    var two_m2 = ftz(identical_mul(Float32(2), ftz(identical_mul(m, m))))
    var cu = Int(comm.unsafe_load(u))
    var du = _fs(g, L_DEG).unsafe_load(u)
    var stot_cu = _sub(stot.unsafe_load(cu), du)
    # k_i,in of u's own community
    var kcu = Float32(0)
    var d64 = Int64(d)
    for r in range(cnt):
        var key = ek.unsafe_load(lo + r)
        if key // d64 == Int64(cu):
            kcu = _add(kcu, val.unsafe_load(lo + Int(key % d64)))
    var remove_cost = _add(
        -ftz(identical_div(kcu, m)),
        ftz(identical_div(ftz(identical_mul(res, ftz(identical_mul(stot_cu, du)))), two_m2)),
    )
    var best = cu
    var best_gain = Float32(0)
    var r = 0
    while r < cnt:
        var c = Int(ek.unsafe_load(lo + r) // d64)
        var kc = Float32(0)
        while r < cnt and ek.unsafe_load(lo + r) // d64 == Int64(c):
            kc = _add(kc, val.unsafe_load(lo + Int(ek.unsafe_load(lo + r) % d64)))
            r += 1
        if kc == Float32(0):
            continue
        var sc = stot_cu if c == cu else stot.unsafe_load(c)
        var gain = _sub(
            _add(remove_cost, ftz(identical_div(kc, m))),
            ftz(identical_div(ftz(identical_mul(res, ftz(identical_mul(sc, du)))), two_m2)),
        )
        if gain > best_gain:
            best_gain = gain
            best = c
    if best != cu:
        comm.unsafe_store(u, Int32(best))
        _is(g, L_IC).unsafe_store(0, Int32(1))


@always_inline
def _lv_skey(t: Int, g: GA):
    """The key (community, node) of node t into KA; stot[t] = 0 (an empty
    community's total). n0 nodes."""
    _ls(g, L_KA).unsafe_store(t, Int64(_is(g, L_COMM).unsafe_load(t)) * Int64(g.n0) + Int64(t))
    _fs(g, L_STOT).unsafe_store(t, Float32(0))


@always_inline
def _lv_sfold(t: Int, g: GA):
    """At a community's first position among the sorted keys (slot n4):
    its members' degrees folded in ascending node id into stot."""
    var nn = g.n0
    var k = _ls(g, g.n4)
    var n64 = Int64(nn)
    var c = k.unsafe_load(t) // n64
    if t > 0 and k.unsafe_load(t - 1) // n64 == c:
        return
    var deg = _fs(g, L_DEG)
    var acc = Float32(0)
    var r = t
    while r < nn and k.unsafe_load(r) // n64 == c:
        acc = _add(acc, deg.unsafe_load(Int(k.unsafe_load(r) % n64)))
        r += 1
    _fs(g, L_STOT).unsafe_store(Int(c), acc)


@always_inline
def _lv_qv(t: Int, g: GA):
    """Index t's modularity term on graph n3 (x0 the resolution, m in
    FV[0]): node t's weight inside its community (its self-loop once, its
    edges to larger ids in its community, ascending) over m, minus the
    resolution times community t's (stot / 2m)^2. Their blocked fold is the
    modularity."""
    var ip = _gip(g, g.n3)
    var col = _gcol(g, g.n3)
    var val = _gval(g, g.n3)
    var comm = _is(g, L_COMM)
    var cu = Int(comm.unsafe_load(t))
    var inner = Float32(0)
    for e in range(Int(ip.unsafe_load(t)), Int(ip.unsafe_load(t + 1))):
        var v = Int(col.unsafe_load(e))
        if v == t or (v > t and Int(comm.unsafe_load(v)) == cu):
            inner = _add(inner, val.unsafe_load(e))
    var m = _fs(g, L_FV).unsafe_load(0)
    var two_m = ftz(identical_mul(Float32(2), m))
    var lc = ftz(identical_div(inner, m))
    var fr = ftz(identical_div(_fs(g, L_STOT).unsafe_load(t), two_m))
    _fs(g, L_VQ).unsafe_store(t, _sub(lc, ftz(identical_mul(g.x0, ftz(identical_mul(fr, fr))))))


@always_inline
def _lv_flagz(t: Int, g: GA):
    _is(g, L_FLAG).unsafe_store(t, Int32(0))


@always_inline
def _lv_used(t: Int, g: GA):
    """FLAG[comm[t]] = 1 (every writer stores 1)."""
    _is(g, L_FLAG).unsafe_store(Int(_is(g, L_COMM).unsafe_load(t)), Int32(1))


@always_inline
def _lv_renum(t: Int, g: GA):
    """comm[t] -> its community's rank among the used ids (FLAG scanned)."""
    var comm = _is(g, L_COMM)
    comm.unsafe_store(t, _is(g, L_FLAG).unsafe_load(Int(comm.unsafe_load(t))))


@always_inline
def _lv_lab(t: Int, g: GA):
    """Original node t's label through this level's communities."""
    var lab = _is(g, L_LAB)
    lab.unsafe_store(t, _is(g, L_COMM).unsafe_load(Int(lab.unsafe_load(t))))


@always_inline
def _lv_ekey(t: Int, g: GA):
    """Aggregation key of entry t of graph n3 (n0 entries, n1 communities):
    (comm[u] * n1 + comm[v]) * n0 + t, an entry inside a community only from
    its u <= v direction (each internal edge once, a self-loop once); the
    other entries (and zero weights) get keys past every pair."""
    var nnz = g.n0
    var nc = g.n1
    var u = Int(_gsrc(g, g.n3).unsafe_load(t))
    var v = Int(_gcol(g, g.n3).unsafe_load(t))
    var comm = _is(g, L_COMM)
    var cu = Int(comm.unsafe_load(u))
    var cv = Int(comm.unsafe_load(v))
    var keep = (cu != cv or u <= v) and _gval(g, g.n3).unsafe_load(t) != Float32(0)
    # int64 throughout: the key exceeds 2^31 (Int is narrower on some GPUs)
    var pk = Int64(cu) * Int64(nc) + Int64(cv) if keep else Int64(nc) * Int64(nc)
    _ls(g, L_KA).unsafe_store(t, pk * Int64(nnz) + Int64(t))


@always_inline
def _ehead(k: LP, t: Int, nnz: Int, nc: Int) -> Bool:
    var pk = k.unsafe_load(t) // Int64(nnz)
    if pk >= Int64(nc) * Int64(nc):
        return False
    return t == 0 or k.unsafe_load(t - 1) // Int64(nnz) != pk


@always_inline
def _lv_head(t: Int, g: GA):
    """SEGH[t] = 1 at the first sorted key (slot n4) of a community pair."""
    _is(g, L_SEGH).unsafe_store(t, Int32(1 if _ehead(_ls(g, g.n4), t, g.n0, g.n1) else 0))


@always_inline
def _lv_agg(t: Int, g: GA):
    """At a pair's first sorted key: the pair's weights folded in ascending
    entry order into entry SEGH[t] (scanned) of graph n5: row c, column d."""
    var nnz = g.n0
    var nc = g.n1
    var k = _ls(g, g.n4)
    if not _ehead(k, t, nnz, nc):
        return
    var pk = k.unsafe_load(t) // Int64(nnz)
    var val = _gval(g, g.n3)
    var w = Float32(0)
    var r = t
    while r < nnz and k.unsafe_load(r) // Int64(nnz) == pk:
        w = _add(w, val.unsafe_load(Int(k.unsafe_load(r) % Int64(nnz))))
        r += 1
    var s = Int(_is(g, L_SEGH).unsafe_load(t))
    _gsrc(g, g.n5).unsafe_store(s, Int32(pk // Int64(nc)))
    _gcol(g, g.n5).unsafe_store(s, Int32(pk % Int64(nc)))
    _gval(g, g.n5).unsafe_store(s, w)


@always_inline
def _lv_aip(t: Int, g: GA):
    """Graph n5's indptr[t]: the first of its n2 entries whose row is >= t."""
    var src = _gsrc(g, g.n5)
    var a = 0
    var b = g.n2
    while a < b:
        var mid = (a + b) // 2
        if Int(src.unsafe_load(mid)) < t:
            a = mid + 1
        else:
            b = mid
    _gip(g, g.n5).unsafe_store(t, Int32(a))


@always_inline
def _lv_copyi(t: Int, g: GA):
    """comm[t] = labels[t] (the final modularity on the input graph)."""
    _is(g, L_COMM).unsafe_store(t, _is(g, L_LAB).unsafe_load(t))


@always_inline
def gp_item[S: Int](t: Int, g: GA):
    """Stage S's item t."""
    comptime if S == GP_SCAN_PART:
        _scan_part(t, g)
    elif S == GP_SCAN_MID:
        _scan_mid(t, g)
    elif S == GP_SCAN_FIN:
        _scan_fin(t, g)
    elif S == GP_SORT_RUN_ST:
        _sort_run(t, g)
    elif S == GP_SORT_MERGE:
        _sort_merge(t, g)
    elif S == GP_FPART:
        _fpart(t, g)
    elif S == GP_FFIN:
        _ffin(t, g)
    elif S == PR_ROWSUM:
        _pr_rowsum(t, g)
    elif S == PR_COLCNT:
        _pr_colcnt(t, g)
    elif S == PR_COLFILL:
        _pr_colfill(t, g)
    elif S == PR_DPART:
        _pr_dpart(t, g)
    elif S == PR_STEP:
        _pr_step(t, g)
    elif S == PR_APART:
        _pr_apart(t, g)
    elif S == LV_ROWCNT:
        _lv_rowcnt(t, g)
    elif S == LV_FILL:
        _lv_fill(t, g)
    elif S == LV_MROW:
        _lv_mrow(t, g)
    elif S == LV_DEG:
        _lv_deg(t, g)
    elif S == LV_COL0:
        _lv_col0(t, g)
    elif S == LV_COLR:
        _lv_colr(t, g)
    elif S == LV_ICZ:
        _lv_icz(t, g)
    elif S == LV_UNCOL:
        _lv_uncol(t, g)
    elif S == LV_CKEY:
        _lv_ckey(t, g)
    elif S == LV_COFF:
        _lv_coff(t, g)
    elif S == LV_COPYL:
        _lv_copyl(t, g)
    elif S == LV_SAVE:
        _lv_save(t, g)
    elif S == LV_MOVE:
        lv_move_item(t, g)
    elif S == LV_SKEY:
        _lv_skey(t, g)
    elif S == LV_SFOLD:
        _lv_sfold(t, g)
    elif S == LV_QV:
        _lv_qv(t, g)
    elif S == LV_RESTORE:
        _lv_restore(t, g)
    elif S == LV_FLAGZ:
        _lv_flagz(t, g)
    elif S == LV_USED:
        _lv_used(t, g)
    elif S == LV_RENUM:
        _lv_renum(t, g)
    elif S == LV_LAB:
        _lv_lab(t, g)
    elif S == LV_EKEY:
        _lv_ekey(t, g)
    elif S == LV_HEAD:
        _lv_head(t, g)
    elif S == LV_AGG:
        _lv_agg(t, g)
    elif S == LV_AIP:
        _lv_aip(t, g)
    elif S == LV_COPYI:
        _lv_copyi(t, g)


# ------------------------------------------------------------------ drivers (shared by both columns)
def gp_scan[E: GExec](mut ex: E, g: GA, xs: Int, ps: Int, count: Int) raises:
    """Exclusive scan of int32 slot xs (count entries; the total lands at
    xs[count]) with parts slot ps."""
    var nb = gp_scan_blocks(count)
    var q = g
    q.n0 = count
    q.n1 = nb
    q.n4 = xs
    q.n5 = ps
    ex.run(GP_SCAN_PART, nb, q)
    ex.run(GP_SCAN_MID, 1, q)
    ex.run(GP_SCAN_FIN, nb, q)


def gp_sort[E: GExec](mut ex: E, g: GA, ka: Int, kb: Int, count: Int) raises -> Int:
    """Sort the unique int64 keys of slot ka (count) using kb; returns the
    slot that holds them sorted."""
    var q = g
    q.n0 = count
    q.n4 = ka
    ex.run(GP_SORT_RUN_ST, (count + GP_SORT_RUN - 1) // GP_SORT_RUN, q)
    var src = ka
    var dst = kb
    var w = GP_SORT_RUN
    while w < count:
        q.n1 = w
        q.n4 = src
        q.n5 = dst
        ex.run(GP_SORT_MERGE, count, q)
        var tmp = src
        src = dst
        dst = tmp
        w *= 2
    return src


def gp_fold[E: GExec](mut ex: E, g: GA, xs: Int, ps: Int, count: Int, out_slot: Int, out_idx: Int) raises:
    """The blocked float fold of slot xs (count) into out_slot[out_idx]."""
    var nb = gp_fold_blocks(count)
    var q = g
    q.n0 = count
    q.n1 = nb
    q.n4 = xs
    q.n5 = ps
    q.n6 = out_slot
    q.n7 = out_idx
    ex.run(GP_FPART, nb, q)
    ex.run(GP_FFIN, 1, q)


def pr_drive[E: GExec](
    mut ex: E, n: Int, max_iter: Int, thr: Float64, binary: Int, alpha: Float32,
    x: Int, p: Int, dw: Int, info: Int,
) raises:
    """PageRank's power iteration (`op_pr_iterate_sparse`'s contract): the
    column lists built from the dense adjacency on the executor, then per
    iteration the dangling fold, the step, the |x' - x| fold; the sum is
    read back and compared in double against thr (Python's n * tol). `x`
    in: the start, out: the last iterate. info (int32 x 2): iterations,
    converged."""
    var nbs = gp_scan_blocks(n)
    var la = Lay(P_NSLOT)
    la.i(P_CNT, n + 1)
    la.i(P_PART, nbs + 1)
    var g = ex.alloc(la)
    var q = g
    q.n0 = n
    ex.run(PR_COLCNT, n, q)
    gp_scan(ex, g, P_CNT, P_PART, n)
    var nnz = ex.get_i(P_CNT, n)
    var nbf = gp_fold_blocks(n)
    var lb = Lay(P_NSLOT)
    lb.i(P_CNT, n + 1)
    lb.i(P_PART, nbs + 1)
    lb.i(P_DG, n)
    lb.i(P_ROWS, nnz)
    lb.f(P_RS, n)
    lb.f(P_VALS, nnz)
    lb.f(P_XA, n)
    lb.f(P_XB, n)
    lb.f(P_P, n)
    lb.f(P_DW, n)
    lb.f(P_FP, nbf)
    lb.f(P_SUM, 2)
    g = ex.alloc(lb)
    q = g
    q.n0 = n
    q.n1 = binary
    ex.run(PR_COLCNT, n, q)
    gp_scan(ex, g, P_CNT, P_PART, n)
    ex.run(PR_ROWSUM, n, q)
    ex.run(PR_COLFILL, n, q)
    ex.up_f(P_XA, x, n)
    ex.up_f(P_P, p, n)
    ex.up_f(P_DW, dw, n)
    var cur = P_XA
    var nxt = P_XB
    var n_iter = 0
    var converged = False
    for it in range(max_iter):
        var s = g
        s.n0 = n
        s.n4 = cur
        s.n5 = nxt
        s.x0 = alpha
        var f = g
        f.n1 = nbf
        f.n5 = P_FP
        f.n6 = P_SUM
        f.n7 = 0
        ex.run(PR_DPART, nbf, s)
        ex.run(GP_FFIN, 1, f)
        ex.run(PR_STEP, n, s)
        ex.run(PR_APART, nbf, s)
        f.n7 = 1
        ex.run(GP_FFIN, 1, f)
        var sm = ex.get_f(P_SUM, 1)
        var tmp = cur
        cur = nxt
        nxt = tmp
        n_iter = it + 1
        if Float64(sm) < thr:
            converged = True
            break
    ex.down_f(cur, x, n)
    var inf = IP(unsafe_from_address=info)
    inf.unsafe_store(0, Int32(n_iter))
    inf.unsafe_store(1, Int32(1 if converged else 0))


def _lv_modularity[E: GExec](mut ex: E, g: GA, gb: Int, nn: Int, res: Float32) raises -> Float32:
    """The modularity of comm on graph gb (stot current) into FV[1], read."""
    var q = g
    q.n0 = nn
    q.n3 = gb
    q.x0 = res
    ex.run(LV_QV, nn, q)
    gp_fold(ex, g, L_VQ, L_FP, nn, L_FV, 1)
    return ex.get_f(L_FV, 1)


def _lv_totals[E: GExec](mut ex: E, g: GA, nn: Int) raises:
    """stot from comm and deg: the (community, node) keys sorted, each
    community's degrees folded ascending."""
    var q = g
    q.n0 = nn
    ex.run(LV_SKEY, nn, q)
    q.n4 = gp_sort(ex, g, L_KA, L_KB, nn)
    ex.run(LV_SFOLD, nn, q)


def lv_drive[E: GExec](
    mut ex: E, n: Int, max_level: Int, res: Float32, thr: Float32, labels: Int, info: Int,
) raises:
    """Louvain on the dense symmetric n x n input (module note). labels[u]
    ends as original node u's community; info = [modularity, levels]."""
    var la = Lay(L_NSLOT)
    la.i(L_G0, n + 1)
    la.i(L_PART, gp_scan_blocks(n) + 1)
    la.i(L_LAB, n)
    var g = ex.alloc(la)
    var q = g
    q.n0 = n
    ex.run(LV_ROWCNT, n, q)
    gp_scan(ex, g, L_G0, L_PART, n)
    var nnz = ex.get_i(L_G0, n)
    var cap = max(n, nnz)
    var lb = Lay(L_NSLOT)
    for b in range(3):
        var gb = L_G0 + 4 * b
        lb.i(gb, n + 1)
        lb.i(gb + 1, nnz)
        lb.i(gb + 2, nnz)
        lb.f(gb + 3, nnz)
    lb.i(L_PART, gp_scan_blocks(cap) + 1)
    lb.i(L_COMM, n)
    lb.i(L_SAVE, n)
    lb.i(L_COLOR, n)
    lb.i(L_LAB, n)
    lb.i(L_FLAG, n + 1)
    lb.i(L_SEGH, nnz + 1)
    lb.i(L_COFF, n + LV_COLOR_CHUNK + 2)
    lb.i(L_IC, 8)
    lb.f(L_DEG, n)
    lb.f(L_STOT, n)
    lb.f(L_SSAVE, n)
    lb.f(L_VQ, n)
    lb.f(L_FP, gp_fold_blocks(n) + 1)
    lb.f(L_FV, 4)
    lb.l(L_KA, cap)
    lb.l(L_KB, cap)
    lb.l(L_EK, nnz)
    lb.l(L_ORD, n)
    g = ex.alloc(lb)
    q = g
    q.n0 = n
    ex.run(LV_ROWCNT, n, q)
    gp_scan(ex, g, L_G0, L_PART, n)
    ex.run(LV_FILL, n, q)
    # m: each row's share at columns >= its own, blocked fold
    q.n3 = L_G0
    ex.run(LV_MROW, n, q)
    gp_fold(ex, g, L_VQ, L_FP, n, L_FV, 0)
    var gcur = L_G0
    var gnext = L_GA
    var nn = n
    var nnzc = nnz
    var levels = 0
    var mod = Float32(0)
    var first = True
    while max_level <= 0 or levels < max_level:
        # ---- the level's start: singletons on the current graph
        q = g
        q.n0 = nn
        q.n1 = 1
        q.n3 = gcur
        q.x0 = res
        ex.run(LV_DEG, nn, q)
        var qprev = _lv_modularity(ex, g, gcur, nn, res)
        if first:
            mod = qprev
            first = False
        # ---- colouring
        ex.run(LV_COL0, nn, q)
        var rounds = 0
        while True:
            for _ in range(LV_COLOR_CHUNK):
                var c = q
                c.n1 = rounds
                ex.run(LV_COLR, nn, c)
                rounds += 1
            var z = q
            z.n1 = 1
            ex.run(LV_ICZ, 1, z)
            ex.run(LV_UNCOL, nn, q)
            if ex.get_i(L_IC, 1) == 0:
                break
        ex.run(LV_CKEY, nn, q)
        var ks = q
        ks.n4 = gp_sort(ex, g, L_KA, L_KB, nn)
        ex.run(LV_COFF, rounds + 1, ks)
        ex.run(LV_COPYL, nn, ks)
        var coff = ex.get_is(L_COFF, rounds + 1)
        # ---- local moving, one colour at a time
        var improvement = False
        for _ in range(LV_MAX_SWEEPS):
            ex.run(LV_SAVE, nn, q)
            var z0 = q
            z0.n1 = 0
            ex.run(LV_ICZ, 1, z0)
            for k in range(rounds):
                var bk = Int(coff[k + 1]) - Int(coff[k])
                if bk == 0:
                    continue
                var mv = q
                mv.n1 = Int(coff[k])
                ex.run(LV_MOVE, bk, mv)
                _lv_totals(ex, g, nn)
            if ex.get_i(L_IC, 0) == 0:
                break
            var qn = _lv_modularity(ex, g, gcur, nn, res)
            if qn > qprev:
                improvement = True
                var gain = _sub(qn, qprev)
                qprev = qn
                if not (gain > thr):
                    break
            else:
                ex.run(LV_RESTORE, nn, q)
                break
        if levels > 0 and not improvement:
            break
        # ---- renumber the communities by ascending old id
        ex.run(LV_FLAGZ, nn + 1, q)
        ex.run(LV_USED, nn, q)
        gp_scan(ex, g, L_FLAG, L_PART, nn)
        var nc = ex.get_i(L_FLAG, nn)
        ex.run(LV_RENUM, nn, q)
        var ql = g
        ql.n0 = n
        ex.run(LV_LAB, n, ql)
        levels += 1
        if not (_sub(qprev, mod) > thr):
            break
        mod = qprev
        # ---- aggregate: (pair, entry) keys sorted, each pair folded
        if nc > 0 and nc * nc + 1 > ((Int(1) << 62) // max(nnzc, 1)):
            raise Error("Louvain: the graph is too large for the aggregation keys")
        var qa = g
        qa.n0 = nnzc
        qa.n1 = nc
        qa.n3 = gcur
        qa.n5 = gnext
        ex.run(LV_EKEY, nnzc, qa)
        qa.n4 = gp_sort(ex, g, L_KA, L_KB, nnzc)
        ex.run(LV_HEAD, nnzc, qa)
        gp_scan(ex, g, L_SEGH, L_PART, nnzc)
        var nnz2 = ex.get_i(L_SEGH, nnzc)
        ex.run(LV_AGG, nnzc, qa)
        qa.n2 = nnz2
        ex.run(LV_AIP, nc + 1, qa)
        var tg = gnext
        gnext = L_GB if tg == L_GA else L_GA
        gcur = tg
        nn = nc
        nnzc = nnz2
    # ---- the report: the modularity of the input graph under the labels
    q = g
    q.n0 = n
    q.n1 = 1
    q.n3 = L_G0
    q.x0 = res
    ex.run(LV_DEG, n, q)
    ex.run(LV_COPYI, n, q)
    _lv_totals(ex, g, n)
    var qf = _lv_modularity(ex, g, L_G0, n, res)
    ex.down_i(L_LAB, labels, n)
    var inf = FP(unsafe_from_address=info)
    inf.unsafe_store(0, qf)
    inf.unsafe_store(1, Float32(levels))
