# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""TreeSHAP over this library's flat forests: the per-unit chains that the
device kernels (xtrees/shap_device.mojo, the GPU binding) and the host runner
(xtrees/shap_host.mojo, the CPU-only binding) both call (lane gap-treeshap,
2026-10-02). One spelling of every arithmetic statement, so the host column
and every GPU column produce the same bits.

Reference: the `shap` package's exact path-dependent algorithm,
`shap/cext/tree_shap.h` (`extend_path`, `unwound_path_sum`; Lundberg et al.
2020, Algorithm 2), which `shap/explainers/_tree.py` calls for
`feature_perturbation="tree_path_dependent"`, in the per-leaf formulation of
GPUTreeShap (Mitchell et al. 2022, `gpu_treeshap.h`): the recursion's state
at a leaf depends only on that leaf's root path, so each leaf rebuilds its
path (the repeated features merged, as `unwind_path` merges them), extends
it, and adds its unwound sums. The same polynomial as the recursion; the
association differs from shap's in the last bits only.

FLOAT32 (Metal has no float64 on the device; mojolearn hardware limits).
Every product is `identical_mul` (never contracted) and every result is
flushed (`ftz`, row 10); every quotient is `identical_div`. Inputs are
flushed where they are read, including both sides of the split compare
(Metal compares flush their operands).

THE UNITS AND THEIR FIXED ORDERS:
  * `shap_parent_unit` (node u): the tree-relative parent of u's children,
    and u's split feature marked in its tree's feature row.
  * `shap_depth_unit` (node u): u's depth, folded by an integer atomic max
    (order free).
  * `shap_cover_unit` (tree t, background row r): +1 on every node r's walk
    visits, integer atomics (order free). cover = the count of background
    rows reaching the node (the flat forest stores no instance counts).
  * `shap_slot_unit` (tree t): the tree's split features, ascending, numbered
    0, 1, ... (its slots); -1 elsewhere. The widest tree's slot count by an
    integer atomic max.
  * `shap_ev_part_unit` (tree t, output j): sum over the tree's leaves,
    ascending node index, of value * cover / root cover * scale.
    `shap_ev_fold_unit` (output j): ev[j] (the caller's init) plus those
    partials, ascending tree index.
  * `shap_tree_unit` (row r, tree t): for each leaf, ascending node index,
    the leaf's terms added into the (row, tree, slot) cells of its path's
    features, each cell a chain from +0. `shap_fold_unit` (row r, feature
    f, output j): +0 plus the (r, t, slot of f in t) cells of every tree t
    that splits on f, ASCENDING TREE INDEX. These are the only folds, so a
    SHAP value is the same sum on every vendor and on the host, at every
    chunking and launch geometry.

DEVIATIONS. The node cover is the background count (above); a node no
background row reaches gives both children a zero fraction where shap would
divide 0 / 0; a leaf whose merged path holds an element with zero and one
fractions both zero contributes exactly zero and is skipped (the recursion
returns there; its unwind would divide 0 / 0). Conditioning (`condition !=
0`, the interaction values) is not carried.

The flat forest (ensemble/flatnode.mojo): tree t is nodes offsets[t] ..
offsets[t+1]; a leaf has left == -1; children are left and left + 1,
tree-relative; `x[colid] <= quesval` goes LEFT.
"""
from std.atomic import Atomic
from checks.numerics import ftz, identical_div, identical_mul

comptime F32P = MutPointer[Float32, MutAnyOrigin]
comptime I32P = MutPointer[Int32, MutAnyOrigin]

#: meta words written by the preparation units: [widest slot count, deepest
#: leaf depth, malformed-forest flag]
comptime SHAP_META_SLOTS = 0
comptime SHAP_META_DEPTH = 1
comptime SHAP_META_BAD = 2
comptime SHAP_META_WORDS = 3

#: the path widths the tree unit is compiled for (the merged path of a leaf
#: holds at most min(depth, slot count) features)
comptime SHAP_MAX_PATH = 256


@always_inline
def _m(a: Float32, b: Float32) -> Float32:
    return ftz(identical_mul(a, b))


@always_inline
def _a(a: Float32, b: Float32) -> Float32:
    return ftz(a + b)


@always_inline
def _s(a: Float32, b: Float32) -> Float32:
    return ftz(a - b)


@always_inline
def _q(a: Float32, b: Float32) -> Float32:
    return ftz(identical_div(a, b))


def shap_path_width(depth: Int, slots: Int) -> Int:
    """The compiled path width for a forest whose deepest leaf is `depth`
    and widest tree has `slots` split features; 0 = too wide."""
    var need = min(depth, slots) + 1
    var w = 8
    while w < need:
        w *= 2
    return w if w <= SHAP_MAX_PATH else 0


@always_inline
def _tree_of(offsets: I32P, n_trees: Int, g: Int) -> Int:
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


def shap_parent_unit(u: Int, offsets: I32P, n_trees: Int, colid: I32P, left: I32P, d: Int,
                     parent: I32P, mark: I32P, meta: I32P):
    """Node u: parent[child] = u (tree-relative) for both children, and
    mark[t * d + colid[u]] = 1. parent is -1 filled by the caller (a root
    keeps it), mark zero filled."""
    var t = _tree_of(offsets, n_trees, u)
    var lo = Int(offsets[unsafe_offset=t])
    var count = Int(offsets[unsafe_offset=t + 1]) - lo
    var l = Int(left[unsafe_offset=u])
    if l == -1:
        return
    var c = Int(colid[unsafe_offset=u])
    if l < 1 or l + 1 >= count or c < 0 or c >= d:
        meta[unsafe_offset=SHAP_META_BAD] = 1
        return
    parent[unsafe_offset=lo + l] = Int32(u - lo)
    parent[unsafe_offset=lo + l + 1] = Int32(u - lo)
    mark[unsafe_offset=t * d + c] = 1


def shap_depth_unit(u: Int, offsets: I32P, n_trees: Int, parent: I32P, meta: I32P):
    """Node u's depth into meta[SHAP_META_DEPTH] by integer max."""
    var t = _tree_of(offsets, n_trees, u)
    var lo = Int(offsets[unsafe_offset=t])
    var count = Int(offsets[unsafe_offset=t + 1]) - lo
    var p = Int(parent[unsafe_offset=u])
    var depth = 0
    while p != -1:
        depth += 1
        if depth > count:
            meta[unsafe_offset=SHAP_META_BAD] = 1
            return
        p = Int(parent[unsafe_offset=lo + p])
    _ = Atomic[DType.int32].max(meta.unsafe_offset(SHAP_META_DEPTH), Int32(depth))


def shap_cover_unit(u: Int, nb: Int, offsets: I32P, colid: I32P, quesval: F32P, left: I32P,
                    bg: F32P, d: Int, cover: I32P, meta: I32P):
    """Unit u = t * nb + r: background row r's walk down tree t, +1 on every
    node it visits (cover zero filled by the caller)."""
    var t = u // nb
    var r = u - t * nb
    var lo = Int(offsets[unsafe_offset=t])
    var count = Int(offsets[unsafe_offset=t + 1]) - lo
    var node = 0
    var steps = 0
    while True:
        _ = Atomic.fetch_add(cover.unsafe_offset(lo + node), Int32(1))
        var l = Int(left[unsafe_offset=lo + node])
        if l == -1:
            return
        var c = Int(colid[unsafe_offset=lo + node])
        if c < 0 or c >= d or l < 1 or l + 1 >= count:
            meta[unsafe_offset=SHAP_META_BAD] = 1
            return
        if ftz(bg[unsafe_offset=r * d + c]) <= ftz(quesval[unsafe_offset=lo + node]):
            node = l
        else:
            node = l + 1
        steps += 1
        if steps > count:
            meta[unsafe_offset=SHAP_META_BAD] = 1
            return


def shap_slot_unit(t: Int, d: Int, slot: I32P, meta: I32P):
    """Tree t's row of `slot` (the marks of `shap_parent_unit`) becomes its
    slot numbers: marked features ascending 0, 1, ..., the rest -1."""
    var c = 0
    for f in range(d):
        if slot[unsafe_offset=t * d + f] != 0:
            slot[unsafe_offset=t * d + f] = Int32(c)
            c += 1
        else:
            slot[unsafe_offset=t * d + f] = -1
    _ = Atomic[DType.int32].max(meta.unsafe_offset(SHAP_META_SLOTS), Int32(c))


def shap_ev_part_unit(u: Int, k: Int, offsets: I32P, left: I32P, leaves: F32P, cover: I32P, tscale: F32P,
                      part: F32P):
    """Unit u = t * k + j: tree t's share of the expected value of output j."""
    var t = u // k
    var j = u - t * k
    var lo = Int(offsets[unsafe_offset=t])
    var hi = Int(offsets[unsafe_offset=t + 1])
    var root = Int(cover[unsafe_offset=lo])
    var e = Float32(0.0)
    if root != 0:
        var rootf = Float32(root)
        var sc = ftz(tscale[unsafe_offset=t])
        for g in range(lo, hi):
            if left[unsafe_offset=g] != -1:
                continue
            var frac = _q(Float32(Int(cover[unsafe_offset=g])), rootf)
            e = _a(e, _m(_m(ftz(leaves[unsafe_offset=g * k + j]), frac), sc))
    part[unsafe_offset=u] = e


def shap_ev_fold_unit(j: Int, n_trees: Int, k: Int, part: F32P, ev: F32P):
    """ev[j] (the caller's init) plus every tree's share, ascending."""
    var acc = ftz(ev[unsafe_offset=j])
    for t in range(n_trees):
        acc = _a(acc, part[unsafe_offset=t * k + j])
    ev[unsafe_offset=j] = acc


@always_inline
def _unwound_sum[W: Int](pz: InlineArray[Float32, W], po: InlineArray[Float32, W], pw: InlineArray[Float32, W],
                         dd: Int, pi: Int) -> Float32:
    """`unwound_path_sum` of path element pi on the path 0 .. dd."""
    var of = po[pi]
    var zf = pz[pi]
    var next_one = pw[dd]
    var total = Float32(0.0)
    var d1 = Float32(dd + 1)
    var i = dd - 1
    while i >= 0:
        if of != 0:
            var tmp = _q(_m(next_one, d1), _m(Float32(i + 1), of))
            total = _a(total, tmp)
            next_one = _s(pw[i], _m(_m(tmp, zf), _q(Float32(dd - i), d1)))
        elif zf != 0:
            total = _a(total, _q(_q(pw[i], zf), _q(Float32(dd - i), d1)))
        i -= 1
    return total


def shap_tree_unit[W: Int](
    u: Int, rows: Int, d: Int, k: Int, slots: Int,
    offsets: I32P, colid: I32P, quesval: F32P, left: I32P, leaves: F32P,
    parent: I32P, cover: I32P, tscale: F32P, slot: I32P, x: F32P, buf: F32P, meta: I32P,
):
    """Unit u = t * rows + r: row r of `x` (rows x d) through tree t. The
    (r, t, s, j) cell is buf[((t * slots + s) * k + j) * rows + r]; this unit
    zeroes the tree's cells, then adds each leaf's terms, leaves ascending.
    W (>= the merged path length + 1) is a register width only: no bit
    depends on it."""
    var t = u // rows
    var r = u - t * rows
    for s in range(slots):
        for j in range(k):
            buf[unsafe_offset=((t * slots + s) * k + j) * rows + r] = 0.0
    var lo = Int(offsets[unsafe_offset=t])
    var hi = Int(offsets[unsafe_offset=t + 1])
    var sc = ftz(tscale[unsafe_offset=t])
    var xr = r * d
    var pf = InlineArray[Int32, W](fill=Int32(-1))
    var pz = InlineArray[Float32, W](fill=Float32(0.0))
    var po = InlineArray[Float32, W](fill=Float32(0.0))
    var pw = InlineArray[Float32, W](fill=Float32(0.0))
    var mf = InlineArray[Int32, W](fill=Int32(-1))
    var mz = InlineArray[Float32, W](fill=Float32(0.0))
    var mo = InlineArray[Float32, W](fill=Float32(0.0))
    for g in range(lo, hi):
        if left[unsafe_offset=g] != -1:
            continue
        # the leaf's root path, walked up: a feature seen again (shallower)
        # merges into its deepest element (`unwind_path`'s merge: the
        # zero fractions multiplied deeper * shallower, the one fractions
        # both required)
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
            var hot = l if ftz(x[unsafe_offset=xr + Int(f)]) <= ftz(quesval[unsafe_offset=a]) else l + 1
            var o = Float32(1.0) if c == hot else Float32(0.0)
            var i = 0
            while i < n and mf[i] != f:
                i += 1
            if i < n:
                mz[i] = _m(mz[i], z)
                if o == 0:
                    mo[i] = 0.0
            else:
                if n + 1 >= W:
                    # wider than the compiled path: refused by the caller
                    meta[unsafe_offset=SHAP_META_BAD] = 1
                    return
                mf[i] = f
                mz[i] = z
                mo[i] = o
                n += 1
            c = p
            p = Int(parent[unsafe_offset=a])
        var dead = False
        for i in range(n):
            if mz[i] == 0 and mo[i] == 0:
                dead = True
        if dead:
            continue
        # the path in the recursion's order (ascending last appearance),
        # after the root element (feature -1, fractions 1, 1)
        pf[0] = -1
        pz[0] = 1.0
        po[0] = 1.0
        for i in range(n):
            pf[i + 1] = mf[n - 1 - i]
            pz[i + 1] = mz[n - 1 - i]
            po[i + 1] = mo[n - 1 - i]
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
            var w = _unwound_sum[W](pz, po, pw, n, i)
            var s = _m(_m(w, _s(po[i], pz[i])), sc)
            var sl = Int(slot[unsafe_offset=t * d + Int(pf[i])])
            if sl < 0 or sl >= slots:
                meta[unsafe_offset=SHAP_META_BAD] = 1
                return
            for j in range(k):
                var o = ((t * slots + sl) * k + j) * rows + r
                buf[unsafe_offset=o] = _a(buf[unsafe_offset=o], _m(s, ftz(leaves[unsafe_offset=g * k + j])))


def shap_fold_unit(u: Int, r0: Int, rows: Int, n_trees: Int, d: Int, k: Int, slots: Int, slot: I32P, buf: F32P,
                   phi: F32P):
    """Unit u = (f * k + j) * rows + r: phi[r0 + r, f, j] = +0 plus the
    (r, t, slot of f in t, j) cells, ascending tree index."""
    var r = u % rows
    var fj = u // rows
    var j = fj % k
    var f = fj // k
    var acc = Float32(0.0)
    for t in range(n_trees):
        var s = Int(slot[unsafe_offset=t * d + f])
        if s >= 0:
            acc = _a(acc, buf[unsafe_offset=((t * slots + s) * k + j) * rows + r])
    phi[unsafe_offset=((r0 + r) * d + f) * k + j] = acc


# ------------------------------------------- the model-agnostic explainers
def block_mean(
    y: MutPointer[Float32, MutUntrackedOrigin], m: Int, nb: Int, k: Int,
    res: MutPointer[Float64, MutUntrackedOrigin],
):
    """res[s, j] = mean over the nb background rows of y[(s * nb + r), j],
    summed in row order."""
    for s in range(m):
        for j in range(k):
            var acc: Float64 = 0.0
            for r in range(nb):
                # DEVIATION 5602: background rows folded in order.
                acc = acc + Float64(y[unsafe_offset=(s * nb + r) * k + j])
            res[unsafe_offset=s * k + j] = acc / Float64(nb)
