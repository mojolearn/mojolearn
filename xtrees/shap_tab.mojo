# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""TreeSHAP's leaf-table row unit with one decision word per (row, tree)
(wherever the leaf table is on: FAST + Apple and IDENTICAL on every
vendor; the default since 2026-10-04 (rollback
`-D MOJOLEARN_SHAP_TREE_TAB_OFF`), on top of the default
leaf table MOJOLEARN_TREESHAP_FAST_TABLE of xtrees/shap.mojo).

Recovered from lane/apple-fast-shap@13343dd51 (`shap_tab.mojo` there). Main
has since tabulated the per-leaf terms per one-fraction pattern
(`shap_table_unit`); the part of the old candidate main does not have is
the row unit. `shap_table_row_unit` finds each leaf's pattern by walking the
leaf's root path and reading x at every ancestor, so a tree with L leaves
at depth D costs L x D reads and compares per row. Here:

  rank   one unit per tree: each internal node's rank among the tree's
         internal nodes (ascending node index) and the tree's internal
         count. A 64-bit decision word holds at most 64 ranks.
  need   one unit per leaf: for element e of the leaf's merged path
         (`leaf_mf`'s walk order, deepest first, as `shap_table_unit`
         writes it), the internal ranks where the row must go LEFT and the
         ranks where it must go RIGHT for the element's one fraction to be 1.
  row    one unit per (row, tree): one pass over the tree's internal nodes
         gives the row's decisions as one word (one x read and compare per
         internal node); each leaf's pattern is then n mask tests, and the
         stored terms are added exactly as `shap_table_row_unit` adds them
         (same cells, same order, same statement): the same bits.

A tree with more than SHAP_TAB_WORD internal nodes (the word's width: a
kernel limit, not a data-shape window) takes `shap_table_row_unit` in the
same unit, so there is no host wait and no refusal path."""
from max.gpu.memory import AddressSpace
from checks.numerics import ftz
from xtrees.shap import F32P, I32P, SHAP_META_BAD, _a, _m, _tree_of, shap_table_row_unit

#: the decision word's width: internal ranks 0 .. 63
comptime SHAP_TAB_WORD = 64
comptime U64P = MutPointer[UInt64, MutAnyOrigin]


@always_inline
def shap_tab_rank_unit(t: Int, offsets: I32P, left: I32P, rank: I32P, nint: I32P):
    """Tree t: rank[g] = the count of the tree's internal nodes before g
    (-1 for a leaf); nint[t] = the tree's internal count."""
    var lo = Int(offsets[unsafe_offset=t])
    var hi = Int(offsets[unsafe_offset=t + 1])
    var c = 0
    for g in range(lo, hi):
        if left[unsafe_offset=g] != -1:
            rank[unsafe_offset=g] = Int32(c)
            c += 1
        else:
            rank[unsafe_offset=g] = -1
    nint[unsafe_offset=t] = Int32(c)


@always_inline
def shap_tab_need_unit(
    g: Int, NM: Int, offsets: I32P, n_trees: Int, colid: I32P, left: I32P, parent: I32P, rank: I32P,
    leaf_mf: I32P, leaf_n: I32P, nint: I32P, needl: U64P, needr: U64P,
):
    """Leaf g (leaf_n[g] >= 0, written by `shap_table_unit`'s m == 0 unit,
    launched before this one on the same stream): needl/needr[g * NM + e]
    = the ranks of the ancestors merged into element e where the leaf's
    branch is the left / the right child. The element's one fraction is 1
    exactly when the row's decision bit is 1 at every needl rank and 0 at
    every needr rank (`shap_table_row_unit`'s `c != hot` test)."""
    var n = Int(leaf_n[unsafe_offset=g])
    if n < 0:
        return
    var t = _tree_of(offsets, n_trees, g)
    if Int(nint[unsafe_offset=t]) > SHAP_TAB_WORD:
        return  # the row unit takes `shap_table_row_unit` for this tree
    var lo = Int(offsets[unsafe_offset=t])
    for e in range(n):
        needl[unsafe_offset=g * NM + e] = UInt64(0)
        needr[unsafe_offset=g * NM + e] = UInt64(0)
    var c = g - lo
    var p = Int(parent[unsafe_offset=g])
    while p != -1:
        var a = lo + p
        var f = colid[unsafe_offset=a]
        var bit = UInt64(1) << UInt64(Int(rank[unsafe_offset=a]))
        var e = 0
        while e < n and leaf_mf[unsafe_offset=g * NM + e] != f:
            e += 1
        if e < n:
            if c == Int(left[unsafe_offset=a]):
                needl[unsafe_offset=g * NM + e] = needl[unsafe_offset=g * NM + e] | bit
            else:
                needr[unsafe_offset=g * NM + e] = needr[unsafe_offset=g * NM + e] | bit
        c = p
        p = Int(parent[unsafe_offset=a])


@always_inline
def shap_tab_row_unit[ACC: Int](
    u: Int, rows: Int, d: Int, k: Int, slots: Int, M: Int, NM: Int,
    offsets: I32P, colid: I32P, quesval: F32P, left: I32P, leaves: F32P, parent: I32P, slot: I32P,
    leaf_mf: I32P, leaf_n: I32P, table: F32P, dead: I32P, nint: I32P, needl: U64P, needr: U64P,
    x: F32P, buf: F32P, meta: I32P,
):
    """Unit u = t * rows + r: `shap_table_row_unit`'s cells, the patterns
    from one decision word. ACC as there."""
    var t = u // rows
    var r = u - t * rows
    if Int(nint[unsafe_offset=t]) > SHAP_TAB_WORD:
        shap_table_row_unit[ACC](u, rows, d, k, slots, M, NM, offsets, colid, quesval, left, leaves, parent, slot,
                                 leaf_mf, leaf_n, table, dead, x, buf, meta)
        return
    var lo = Int(offsets[unsafe_offset=t])
    var hi = Int(offsets[unsafe_offset=t + 1])
    var xr = r * d
    var dec = UInt64(0)
    var rk = 0
    for g in range(lo, hi):
        if left[unsafe_offset=g] != -1:
            if ftz(x[unsafe_offset=xr + Int(colid[unsafe_offset=g])]) <= ftz(quesval[unsafe_offset=g]):
                dec = dec | (UInt64(1) << UInt64(rk))
            rk += 1
    var acc = InlineArray[Float32, max(ACC, 1)](fill=Float32(0.0))
    comptime if ACC == 0:
        for s in range(slots):
            for j in range(k):
                buf[unsafe_offset=((t * slots + s) * k + j) * rows + r] = 0.0
    for g in range(lo, hi):
        var n = Int(leaf_n[unsafe_offset=g])
        if n < 0:
            continue
        var mask = 0
        for e in range(n):
            var nl = needl[unsafe_offset=g * NM + e]
            var nr = needr[unsafe_offset=g * NM + e]
            if (dec & nl) == nl and (dec & nr) == UInt64(0):
                mask |= 1 << e
        if dead[unsafe_offset=g * M + mask] != 0:
            continue
        var base = (g * M + mask) * NM
        for i in range(1, n + 1):
            var s = table[unsafe_offset=base + i - 1]
            var sl = Int(slot[unsafe_offset=t * d + Int(leaf_mf[unsafe_offset=g * NM + n - i])])
            if sl < 0 or sl >= slots:
                meta[unsafe_offset=SHAP_META_BAD] = 6
                return
            for j in range(k):
                comptime if ACC > 0:
                    var o = sl * k + j
                    acc[o] = _a(acc[o], _m(s, ftz(leaves[unsafe_offset=g * k + j])))
                else:
                    var o = ((t * slots + sl) * k + j) * rows + r
                    buf[unsafe_offset=o] = _a(buf[unsafe_offset=o], _m(s, ftz(leaves[unsafe_offset=g * k + j])))
    comptime if ACC > 0:
        for s in range(slots):
            for j in range(k):
                buf[unsafe_offset=((t * slots + s) * k + j) * rows + r] = acc[s * k + j]


comptime SHF32 = UnsafePointer[Float32, MutUntrackedOrigin, address_space=AddressSpace.SHARED]


@always_inline
def shap_tab_row_unit_sh[ACC: Int](
    u: Int, rows: Int, d: Int, k: Int, slots: Int, M: Int, NM: Int,
    offsets: I32P, colid: I32P, quesval: F32P, left: I32P, leaves: F32P, parent: I32P, slot: I32P,
    leaf_mf: I32P, leaf_n: I32P, table: F32P, dead: I32P, nint: I32P, needl: U64P, needr: U64P,
    x: F32P, buf: F32P, meta: I32P, sh: SHF32, lane: Int, lanes: Int,
):
    """`shap_tab_row_unit[ACC]` (ACC > 0) with the accumulator in the block's
    threadgroup memory instead of a dynamically indexed per-thread array
    (lane gap-shap-nb, -D MOJOLEARN_TREESHAP_ACC_SHARED, default off: the
    plan's per-block tile; a per-thread InlineArray indexed by a runtime slot
    is likely spilled to local memory on NVIDIA, unverified). Cell o of this
    thread is sh[o * lanes + lane]: each thread owns a column, so adjacent
    lanes touch adjacent words (no bank conflict) and no barrier is needed.
    The same cells, the same adds in the same order: the same bits."""
    var t = u // rows
    var r = u - t * rows
    if Int(nint[unsafe_offset=t]) > SHAP_TAB_WORD:
        shap_table_row_unit[ACC](u, rows, d, k, slots, M, NM, offsets, colid, quesval, left, leaves, parent, slot,
                                 leaf_mf, leaf_n, table, dead, x, buf, meta)
        return
    var lo = Int(offsets[unsafe_offset=t])
    var hi = Int(offsets[unsafe_offset=t + 1])
    var xr = r * d
    var dec = UInt64(0)
    var rk = 0
    for g in range(lo, hi):
        if left[unsafe_offset=g] != -1:
            if ftz(x[unsafe_offset=xr + Int(colid[unsafe_offset=g])]) <= ftz(quesval[unsafe_offset=g]):
                dec = dec | (UInt64(1) << UInt64(rk))
            rk += 1
    for o in range(slots * k):
        sh[o * lanes + lane] = Float32(0.0)
    for g in range(lo, hi):
        var n = Int(leaf_n[unsafe_offset=g])
        if n < 0:
            continue
        var mask = 0
        for e in range(n):
            var nl = needl[unsafe_offset=g * NM + e]
            var nr = needr[unsafe_offset=g * NM + e]
            if (dec & nl) == nl and (dec & nr) == UInt64(0):
                mask |= 1 << e
        if dead[unsafe_offset=g * M + mask] != 0:
            continue
        var base = (g * M + mask) * NM
        for i in range(1, n + 1):
            var s = table[unsafe_offset=base + i - 1]
            var sl = Int(slot[unsafe_offset=t * d + Int(leaf_mf[unsafe_offset=g * NM + n - i])])
            if sl < 0 or sl >= slots:
                meta[unsafe_offset=SHAP_META_BAD] = 6
                return
            for j in range(k):
                var o = (sl * k + j) * lanes + lane
                sh[o] = _a(sh[o], _m(s, ftz(leaves[unsafe_offset=g * k + j])))
    for s in range(slots):
        for j in range(k):
            buf[unsafe_offset=((t * slots + s) * k + j) * rows + r] = sh[(s * k + j) * lanes + lane]
