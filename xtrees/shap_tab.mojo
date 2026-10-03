# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""TreeSHAP's per-leaf polynomial tabulated (lane/apple-fast-shap,
2026-10-02; FAST + Apple only, `-D MOJOLEARN_SHAP_TREE_TAB`).

`shap_tree_unit` (xtrees/shap.mojo) rebuilds every leaf's merged root path,
extends it and unwinds it for EVERY explained row: 10,000 rows x 100 trees
x 64 leaves of extend + unwound sums (about 120 divisions and 300 indexed
register-array accesses per leaf) on the M3 Ultra took 338 ms on taxi and
816 ms on Istella against LightGBM's 73 / 148 ms. The row enters that
arithmetic ONLY through the one fractions: element i of a leaf's merged
path has o_i = 1 when the row takes the leaf's branch at every node merged
into it, else 0. So a leaf's contribution to its path features is a
function of the bit pattern (o_1 .. o_n), n <= depth, and this module
tabulates it once per leaf and pattern:

  rank     each internal node's rank among its tree's internal nodes
           (ascending node index; at most 64 per tree, i.e. depth <= 6).
  leaf     each leaf's merged path (the walk-up and merge of
           `shap_tree_unit`, row-free): the slot of its feature, its zero
           fraction, and two 64-bit masks over the internal ranks: the nodes
           the row must take LEFT at, and the nodes it must take RIGHT at,
           for o_i = 1.
  pattern  for each leaf and each pattern p < 2^n: `extend_path` and
           `unwound_path_sum` exactly as `shap_tree_unit` spells them, the
           term s_i = w_i (o_i - z_i) scale per element.
  row      per (row, tree): one walk over the tree's internal nodes gives
           the row's left/right decisions as a 64-bit word; per leaf, the
           pattern is n mask tests, and the n tabulated terms are added into
           the (row, tree, slot) cells in the SAME order and with the SAME
           statement as `shap_tree_unit` (leaves ascending, `_a(buf, _m(s,
           leaf))`), so every SHAP value is the same bits as the untabulated
           kernel's.

A forest deeper than 6, a tree with more than 64 internal nodes, a merged
path longer than 6 elements, or a table larger than SHAP_TAB_MAX_BYTES
raises `flag`, and the caller runs the untabulated kernels instead."""
from checks.numerics import ftz
from xtrees.shap import F32P, I32P, _a, _m, _q, _s, _unwound_sum

#: the tabulated depth: leaves at depth <= 6, so <= 64 internal nodes per
#: tree (one 64-bit decision word) and <= 6 merged elements per leaf path
comptime SHAP_TAB_DEPTH = 6
comptime SHAP_TAB_D = SHAP_TAB_DEPTH
comptime SHAP_TAB_P = 1 << SHAP_TAB_DEPTH
#: the register width of the path arrays (>= SHAP_TAB_D + 1), as
#: `shap_path_width` would pick for depth 6
comptime SHAP_TAB_W = 8
comptime SHAP_TAB_MAX_INTERNAL = 64
#: the table (nodes x patterns x elements float32) is refused past this
comptime SHAP_TAB_MAX_BYTES = 512 * 1024 * 1024
comptime U64P = MutPointer[UInt64, MutAnyOrigin]


@always_inline
def _tab_tree_of(offsets: I32P, n_trees: Int, g: Int) -> Int:
    """The tree holding node g: the largest t with offsets[t] <= g."""
    var lo = 0
    var hi = n_trees - 1
    while lo < hi:
        var mid = (lo + hi + 1) // 2
        if Int(offsets[unsafe_offset=mid]) <= g:
            lo = mid
        else:
            hi = mid - 1
    return lo


def shap_tab_rank_unit(u: Int, offsets: I32P, n_trees: Int, left: I32P, rank: I32P, flag: I32P):
    """Node u: rank[u] = the count of internal nodes of its tree before it
    (-1 for a leaf). A tree with more than SHAP_TAB_MAX_INTERNAL internal
    nodes raises the flag."""
    var t = _tab_tree_of(offsets, n_trees, u)
    var lo = Int(offsets[unsafe_offset=t])
    var hi = Int(offsets[unsafe_offset=t + 1])
    if left[unsafe_offset=u] == -1:
        rank[unsafe_offset=u] = -1
        if u != lo:
            return
        # the tree's root counts its internal nodes for the limit
    var c = 0
    for g in range(lo, hi):
        if g == u and left[unsafe_offset=u] != -1:
            rank[unsafe_offset=u] = Int32(c)
        if left[unsafe_offset=g] != -1:
            c += 1
    if c > SHAP_TAB_MAX_INTERNAL:
        flag[unsafe_offset=0] = 1


def shap_tab_leaf_unit(
    g: Int, d: Int, slots: Int, offsets: I32P, n_trees: Int, colid: I32P, left: I32P, parent: I32P, cover: I32P,
    rank: I32P, slot: I32P, pn: I32P, pslot: I32P, pz: F32P, needl: U64P, needr: U64P, flag: I32P,
):
    """Leaf g: its merged root path in the recursion's order (ascending
    last appearance), element i's slot, zero fraction and decision masks at
    pn[g] = n, pslot/pz/needl/needr[g * SHAP_TAB_D + i]. The walk and the
    merge are `shap_tree_unit`'s with the row factored out: the zero
    fractions are multiplied deeper * shallower in the same order, and the
    one fraction of a merged element is the AND of its nodes' one
    fractions, i.e. the row takes the leaf's branch at each of them."""
    if left[unsafe_offset=g] != -1:
        return
    var t = _tab_tree_of(offsets, n_trees, g)
    var lo = Int(offsets[unsafe_offset=t])
    var mf = InlineArray[Int32, SHAP_TAB_W](fill=Int32(-1))
    var mz = InlineArray[Float32, SHAP_TAB_W](fill=Float32(0.0))
    var ml = InlineArray[UInt64, SHAP_TAB_W](fill=UInt64(0))
    var mr = InlineArray[UInt64, SHAP_TAB_W](fill=UInt64(0))
    var n = 0
    var c = g - lo
    var p = Int(parent[unsafe_offset=g])
    while p != -1:
        var a = lo + p
        var f = colid[unsafe_offset=a]
        var ca = Int(cover[unsafe_offset=a])
        var z = Float32(0.0)
        if ca != 0:
            z = _q(Float32(Int(cover[unsafe_offset=lo + c])), Float32(ca))
        var l = Int(left[unsafe_offset=a])
        var rk = Int(rank[unsafe_offset=a])
        if rk < 0 or rk >= SHAP_TAB_MAX_INTERNAL:
            flag[unsafe_offset=0] = 1
            return
        var bit = UInt64(1) << UInt64(rk)
        var i = 0
        while i < n and mf[i] != f:
            i += 1
        if i < n:
            mz[i] = _m(mz[i], z)
        else:
            if n >= SHAP_TAB_D:
                flag[unsafe_offset=0] = 1
                return
            mf[i] = f
            mz[i] = z
            ml[i] = UInt64(0)
            mr[i] = UInt64(0)
            n += 1
        if c == l:
            ml[i] = ml[i] | bit
        else:
            mr[i] = mr[i] | bit
        c = p
        p = Int(parent[unsafe_offset=a])
    pn[unsafe_offset=g] = Int32(n)
    for i in range(n):
        var src = n - 1 - i
        var sl = Int(slot[unsafe_offset=t * d + Int(mf[src])])
        if sl < 0 or sl >= slots:
            flag[unsafe_offset=0] = 1
            return
        pslot[unsafe_offset=g * SHAP_TAB_D + i] = Int32(sl)
        pz[unsafe_offset=g * SHAP_TAB_D + i] = mz[src]
        needl[unsafe_offset=g * SHAP_TAB_D + i] = ml[src]
        needr[unsafe_offset=g * SHAP_TAB_D + i] = mr[src]


def shap_tab_pattern_unit(u: Int, offsets: I32P, n_trees: Int, left: I32P, tscale: F32P, pn: I32P, pz_in: F32P,
                          tab: F32P):
    """Unit u = g * SHAP_TAB_P + p: leaf g's terms under one-fraction
    pattern p (bit i = o of element i + 1) at tab[u * SHAP_TAB_D + i]:
    `shap_tree_unit`'s dead check, `extend_path` and `unwound_path_sum`,
    statement for statement."""
    var g = u // SHAP_TAB_P
    var p = u - g * SHAP_TAB_P
    if left[unsafe_offset=g] != -1:
        return
    var n = Int(pn[unsafe_offset=g])
    if p >= (1 << n):
        return
    var t = _tab_tree_of(offsets, n_trees, g)
    var sc = ftz(tscale[unsafe_offset=t])
    var base = u * SHAP_TAB_D
    var pz = InlineArray[Float32, SHAP_TAB_W](fill=Float32(0.0))
    var po = InlineArray[Float32, SHAP_TAB_W](fill=Float32(0.0))
    var pw = InlineArray[Float32, SHAP_TAB_W](fill=Float32(0.0))
    pz[0] = 1.0
    po[0] = 1.0
    var dead = False
    for i in range(n):
        pz[i + 1] = pz_in[unsafe_offset=g * SHAP_TAB_D + i]
        po[i + 1] = 1.0 if ((p >> i) & 1) != 0 else 0.0
        if pz[i + 1] == 0 and po[i + 1] == 0:
            dead = True
    if dead:
        for i in range(n):
            tab[unsafe_offset=base + i] = 0.0
        return
    # `extend_path` for elements 0 .. n
    for dd in range(n + 1):
        var zf = pz[dd]
        var of = po[dd]
        pw[dd] = 1.0 if dd == 0 else 0.0
        var d1 = Float32(dd + 1)
        var i = dd - 1
        while i >= 0:
            pw[i + 1] = _a(pw[i + 1], _q(_m(_m(of, pw[i]), Float32(i + 1)), d1))
            pw[i] = _q(_m(_m(zf, pw[i]), Float32(dd - i)), d1)
            i -= 1
    for i in range(1, n + 1):
        var w = _unwound_sum[SHAP_TAB_W](pz, po, pw, n, i)
        tab[unsafe_offset=base + i - 1] = _m(_m(w, _s(po[i], pz[i])), sc)


def shap_tab_row_unit(
    u: Int, rows: Int, d: Int, k: Int, slots: Int, offsets: I32P, colid: I32P, quesval: F32P, left: I32P,
    leaves: F32P, pn: I32P, pslot: I32P, needl: U64P, needr: U64P, tab: F32P, x: F32P, buf: F32P,
):
    """Unit u = t * rows + r, the cells and their order as `shap_tree_unit`:
    the row's decisions over tree t's internal nodes as one word, then each
    leaf's pattern and tabulated terms, leaves ascending."""
    var t = u // rows
    var r = u - t * rows
    for s in range(slots):
        for j in range(k):
            buf[unsafe_offset=((t * slots + s) * k + j) * rows + r] = 0.0
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
    for g in range(lo, hi):
        if left[unsafe_offset=g] != -1:
            continue
        var n = Int(pn[unsafe_offset=g])
        var p = 0
        for i in range(n):
            var nl = needl[unsafe_offset=g * SHAP_TAB_D + i]
            var nr = needr[unsafe_offset=g * SHAP_TAB_D + i]
            if (dec & nl) == nl and (dec & nr) == UInt64(0):
                p |= 1 << i
        var base = (g * SHAP_TAB_P + p) * SHAP_TAB_D
        for i in range(n):
            var s = tab[unsafe_offset=base + i]
            if s == 0:
                # a dead pattern, or a zero term: `_a(buf, +-0)` leaves buf
                # (never -0) unchanged, as the untabulated unit's skip does
                continue
            var sl = Int(pslot[unsafe_offset=g * SHAP_TAB_D + i])
            for j in range(k):
                var o = ((t * slots + sl) * k + j) * rows + r
                buf[unsafe_offset=o] = _a(buf[unsafe_offset=o], _m(s, ftz(leaves[unsafe_offset=g * k + j])))
