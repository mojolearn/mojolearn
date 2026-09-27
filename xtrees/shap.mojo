# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""TreeSHAP over this library's flat forests (the trees lane, 2026-09-27).

Reference: the `shap` package's exact path-dependent algorithm,
`shap/cext/tree_shap.h` (`extend_path`, `unwind_path`, `unwound_path_sum`,
`tree_shap_recursive`; Lundberg et al. 2020, Algorithm 2), which
`shap/explainers/_tree.py` calls for `feature_perturbation=
"tree_path_dependent"`. Every arithmetic statement is theirs in their
association order, in binary64 on the host; each product that meets an add
is `identical_mul64` (the build contracts `a * b + c` otherwise).

DEVIATIONS. The node cover (`node_sample_weight`) is the count of BACKGROUND
rows reaching each node (`node_cover`), since the flat forest stores no
instance counts; a node no background row reaches gives both children a zero
fraction where theirs would divide 0 / 0. Conditioning (`condition != 0`,
the interaction values) is not carried.

The flat forest (ensemble/flatnode.mojo): tree t is nodes offsets[t] ..
offsets[t+1]; a leaf has left == -1; children are left and left + 1,
tree-relative; `x[colid] <= quesval` goes LEFT.
"""
from checks.numerics import identical_mul64


struct _Path(Movable):
    var feature: List[Int]
    var zero: List[Float64]
    var one: List[Float64]
    var pw: List[Float64]

    def __init__(out self, n: Int):
        self.feature = List[Int](length=n, fill=-1)
        self.zero = List[Float64](length=n, fill=0.0)
        self.one = List[Float64](length=n, fill=0.0)
        self.pw = List[Float64](length=n, fill=0.0)


def _extend(mut p: _Path, b: Int, d: Int, zf: Float64, of: Float64, fi: Int):
    """`extend_path`, on the path starting at `b`."""
    p.feature[b + d] = fi
    p.zero[b + d] = zf
    p.one[b + d] = of
    p.pw[b + d] = 1.0 if d == 0 else 0.0
    var i = d - 1
    while i >= 0:
        p.pw[b + i + 1] = p.pw[b + i + 1] + identical_mul64(identical_mul64(of, p.pw[b + i]), Float64(i + 1)) / Float64(d + 1)
        p.pw[b + i] = identical_mul64(identical_mul64(zf, p.pw[b + i]), Float64(d - i)) / Float64(d + 1)
        i -= 1


def _unwind(mut p: _Path, b: Int, d: Int, path_index: Int):
    """`unwind_path`."""
    var of = p.one[b + path_index]
    var zf = p.zero[b + path_index]
    var next_one = p.pw[b + d]
    var i = d - 1
    while i >= 0:
        if of != 0:
            var tmp = p.pw[b + i]
            p.pw[b + i] = identical_mul64(next_one, Float64(d + 1)) / identical_mul64(Float64(i + 1), of)
            next_one = tmp - identical_mul64(identical_mul64(p.pw[b + i], zf), Float64(d - i)) / Float64(d + 1)
        else:
            p.pw[b + i] = identical_mul64(p.pw[b + i], Float64(d + 1)) / identical_mul64(zf, Float64(d - i))
        i -= 1
    for j in range(path_index, d):
        p.feature[b + j] = p.feature[b + j + 1]
        p.zero[b + j] = p.zero[b + j + 1]
        p.one[b + j] = p.one[b + j + 1]


def _unwound_sum(p: _Path, b: Int, d: Int, path_index: Int) -> Float64:
    """`unwound_path_sum`."""
    var of = p.one[b + path_index]
    var zf = p.zero[b + path_index]
    var next_one = p.pw[b + d]
    var total: Float64 = 0.0
    var i = d - 1
    while i >= 0:
        if of != 0:
            var tmp = identical_mul64(next_one, Float64(d + 1)) / identical_mul64(Float64(i + 1), of)
            total = total + tmp
            next_one = p.pw[b + i] - identical_mul64(identical_mul64(tmp, zf), Float64(d - i) / Float64(d + 1))
        elif zf != 0:
            total = total + (p.pw[b + i] / zf) / (Float64(d - i) / Float64(d + 1))
        i -= 1
    return total


def _recurse(
    colid: MutPointer[Int32, MutUntrackedOrigin], quesval: MutPointer[Float32, MutUntrackedOrigin],
    left: MutPointer[Int32, MutUntrackedOrigin], leaves: MutPointer[Float32, MutUntrackedOrigin],
    cover: MutPointer[Float64, MutUntrackedOrigin], lo: Int, k: Int,
    x: MutPointer[Float32, MutUntrackedOrigin], xoff: Int,
    phi: MutPointer[Float64, MutUntrackedOrigin], phioff: Int, scale: Float64,
    mut p: _Path, node: Int, depth_in: Int, parent_b: Int,
    pzero: Float64, pone: Float64, pfeature: Int,
):
    """`tree_shap_recursive` with condition 0. `phi[phioff + f * k + j]`
    receives feature f's share of output j, times `scale`."""
    var depth = depth_in
    var b = parent_b + depth + 1
    for j in range(depth + 1):
        p.feature[b + j] = p.feature[parent_b + j]
        p.zero[b + j] = p.zero[parent_b + j]
        p.one[b + j] = p.one[parent_b + j]
        p.pw[b + j] = p.pw[parent_b + j]
    _extend(p, b, depth, pzero, pone, pfeature)
    var g = lo + node
    var l = Int(left[unsafe_offset=g])
    if l == -1:
        for i in range(1, depth + 1):
            var w = _unwound_sum(p, b, depth, i)
            var s = identical_mul64(identical_mul64(w, p.one[b + i] - p.zero[b + i]), scale)
            var f = p.feature[b + i]
            for j in range(k):
                var o = phioff + f * k + j
                phi[unsafe_offset=o] = phi[unsafe_offset=o] + identical_mul64(s, Float64(leaves[unsafe_offset=g * k + j]))
        return
    var split = Int(colid[unsafe_offset=g])
    var hot: Int
    var cold: Int
    if x[unsafe_offset=xoff + split] <= quesval[unsafe_offset=g]:
        hot = l
        cold = l + 1
    else:
        hot = l + 1
        cold = l
    var w = cover[unsafe_offset=g]
    var hot_zero: Float64 = 0.0
    var cold_zero: Float64 = 0.0
    if w != 0:
        hot_zero = cover[unsafe_offset=lo + hot] / w
        cold_zero = cover[unsafe_offset=lo + cold] / w
    var in_zero: Float64 = 1.0
    var in_one: Float64 = 1.0
    var path_index = 0
    while path_index <= depth:
        if p.feature[b + path_index] == split:
            break
        path_index += 1
    if path_index != depth + 1:
        in_zero = p.zero[b + path_index]
        in_one = p.one[b + path_index]
        _unwind(p, b, depth, path_index)
        depth -= 1
    _recurse(colid, quesval, left, leaves, cover, lo, k, x, xoff, phi, phioff, scale, p, hot, depth + 1, b,
             identical_mul64(hot_zero, in_zero), in_one, split)
    _recurse(colid, quesval, left, leaves, cover, lo, k, x, xoff, phi, phioff, scale, p, cold, depth + 1, b,
             identical_mul64(cold_zero, in_zero), 0.0, split)


def _tree_depth(left: MutPointer[Int32, MutUntrackedOrigin], lo: Int, count: Int) -> Int:
    var depth = List[Int](length=count, fill=0)
    var best = 0
    for i in range(count):
        var c = Int(left[unsafe_offset=lo + i])
        if c != -1 and c + 1 < count:
            depth[c] = depth[i] + 1
            depth[c + 1] = depth[i] + 1
            if depth[i] + 1 > best:
                best = depth[i] + 1
    return best


def node_cover(
    offsets: MutPointer[Int32, MutUntrackedOrigin], colid: MutPointer[Int32, MutUntrackedOrigin],
    quesval: MutPointer[Float32, MutUntrackedOrigin], left: MutPointer[Int32, MutUntrackedOrigin],
    x: MutPointer[Float32, MutUntrackedOrigin], n: Int, d: Int, n_trees: Int,
    cover: MutPointer[Float64, MutUntrackedOrigin],
) raises:
    """cover[node] = the number of rows of `x` whose walk visits the node
    (zeroed by the caller)."""
    for t in range(n_trees):
        var lo = Int(offsets[unsafe_offset=t])
        var count = Int(offsets[unsafe_offset=t + 1]) - lo
        for i in range(n):
            var node = 0
            var steps = 0
            while True:
                cover[unsafe_offset=lo + node] = cover[unsafe_offset=lo + node] + 1.0
                var l = Int(left[unsafe_offset=lo + node])
                if l == -1:
                    break
                var c = Int(colid[unsafe_offset=lo + node])
                if c < 0 or c >= d or l < 1 or l + 1 >= count:
                    raise Error("x_trees node_cover: malformed tree")
                node = l if x[unsafe_offset=i * d + c] <= quesval[unsafe_offset=lo + node] else l + 1
                steps += 1
                if steps > count:
                    raise Error("x_trees node_cover: cycle in tree")


def tree_shap(
    offsets: MutPointer[Int32, MutUntrackedOrigin], colid: MutPointer[Int32, MutUntrackedOrigin],
    quesval: MutPointer[Float32, MutUntrackedOrigin], left: MutPointer[Int32, MutUntrackedOrigin],
    leaves: MutPointer[Float32, MutUntrackedOrigin], cover: MutPointer[Float64, MutUntrackedOrigin],
    x: MutPointer[Float32, MutUntrackedOrigin], n: Int, d: Int, n_trees: Int, k: Int, scale: Float64,
    phi: MutPointer[Float64, MutUntrackedOrigin],
) raises:
    """phi[i, f, j] += scale * the TreeSHAP value of feature f for output j
    of row i, trees in order (phi zeroed by the caller). `leaves` holds k
    values per node."""
    for t in range(n_trees):
        var lo = Int(offsets[unsafe_offset=t])
        var count = Int(offsets[unsafe_offset=t + 1]) - lo
        if count < 1:
            raise Error("x_trees tree_shap: empty tree")
        var maxd = _tree_depth(left, lo, count) + 2
        var p = _Path((maxd * (maxd + 1)) // 2 + maxd + 2)
        for i in range(n):
            _recurse(colid, quesval, left, leaves, cover, lo, k, x, i * d, phi, i * d * k, scale, p, 0, 0, 0,
                     1.0, 1.0, -1)


def expected_value(
    offsets: MutPointer[Int32, MutUntrackedOrigin], left: MutPointer[Int32, MutUntrackedOrigin],
    leaves: MutPointer[Float32, MutUntrackedOrigin], cover: MutPointer[Float64, MutUntrackedOrigin],
    n_trees: Int, k: Int, scale: Float64, res: MutPointer[Float64, MutUntrackedOrigin],
):
    """res[j] += scale * sum over each tree's leaves (node order) of
    value[j] * cover / root cover."""
    for t in range(n_trees):
        var lo = Int(offsets[unsafe_offset=t])
        var hi = Int(offsets[unsafe_offset=t + 1])
        var root = cover[unsafe_offset=lo]
        if root == 0:
            continue
        for g in range(lo, hi):
            if left[unsafe_offset=g] != -1:
                continue
            var frac = cover[unsafe_offset=g] / root
            for j in range(k):
                res[unsafe_offset=j] = res[unsafe_offset=j] + identical_mul64(identical_mul64(Float64(leaves[unsafe_offset=g * k + j]), frac), scale)


# ------------------------------------------- the model-agnostic explainers
def mask_expand(
    x: MutPointer[Float32, MutUntrackedOrigin], bg: MutPointer[Float32, MutUntrackedOrigin], nb: Int, d: Int,
    masks: MutPointer[Int32, MutUntrackedOrigin], m: Int, res: MutPointer[Float32, MutUntrackedOrigin],
):
    """res[(s * nb + r), f] = x[f] if masks[s, f] else bg[r, f]: every
    coalition over every background row (the synthetic dataset of cuML's
    `kernel_dataset` / `permutation_shap_dataset`, `explainer/*.cu`). A copy."""
    for s in range(m):
        for r in range(nb):
            var o = (s * nb + r) * d
            for f in range(d):
                if masks[unsafe_offset=s * d + f] != 0:
                    res[unsafe_offset=o + f] = x[unsafe_offset=f]
                else:
                    res[unsafe_offset=o + f] = bg[unsafe_offset=r * d + f]


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
                acc = acc + Float64(y[unsafe_offset=(s * nb + r) * k + j])
            res[unsafe_offset=s * k + j] = acc / Float64(nb)


def kernel_solve(
    masks: MutPointer[Int32, MutUntrackedOrigin], w: MutPointer[Float64, MutUntrackedOrigin], m: Int, d: Int,
    ey: MutPointer[Float64, MutUntrackedOrigin], k: Int, fx: MutPointer[Float64, MutUntrackedOrigin],
    fnull: MutPointer[Float64, MutUntrackedOrigin], phi: MutPointer[Float64, MutUntrackedOrigin],
):
    """shap `KernelExplainer.solve` without l1 selection: the last feature
    eliminated through the efficiency constraint, the weighted normal
    equations solved by Gaussian elimination with partial pivoting (first
    largest pivot), phi[f, j] written for every output j."""
    for j in range(k):
        var total = fx[unsafe_offset=j] - fnull[unsafe_offset=j]
        if d == 1:
            phi[unsafe_offset=j] = total
            continue
        var q = d - 1
        var a = List[Float64](length=q * q, fill=0.0)
        var bv = List[Float64](length=q, fill=0.0)
        for s in range(m):
            var last = Float64(Int(masks[unsafe_offset=s * d + q]))
            var y2 = (ey[unsafe_offset=s * k + j] - fnull[unsafe_offset=j]) - identical_mul64(last, total)
            var ws = w[unsafe_offset=s]
            for r in range(q):
                var er = Float64(Int(masks[unsafe_offset=s * d + r])) - last
                if er == 0:
                    continue
                var wer = identical_mul64(ws, er)
                bv[r] = bv[r] + identical_mul64(wer, y2)
                for c in range(q):
                    var ec = Float64(Int(masks[unsafe_offset=s * d + c])) - last
                    a[r * q + c] = a[r * q + c] + identical_mul64(wer, ec)
        # Gaussian elimination, partial pivoting
        var perm = List[Int](length=q, fill=0)
        for r in range(q):
            perm[r] = r
        for col in range(q):
            var piv = col
            var best = abs(a[perm[col] * q + col])
            for r in range(col + 1, q):
                var v = abs(a[perm[r] * q + col])
                if v > best:
                    best = v
                    piv = r
            var t = perm[col]
            perm[col] = perm[piv]
            perm[piv] = t
            var pr = perm[col]
            var pv = a[pr * q + col]
            if pv == 0:
                continue
            for r in range(col + 1, q):
                var rr = perm[r]
                var f = a[rr * q + col] / pv
                if f == 0:
                    continue
                for c in range(col, q):
                    a[rr * q + c] = a[rr * q + c] - identical_mul64(f, a[pr * q + c])
                bv[rr] = bv[rr] - identical_mul64(f, bv[pr])
        var sol = List[Float64](length=q, fill=0.0)
        var r = q - 1
        while r >= 0:
            var pr = perm[r]
            var acc = bv[pr]
            for c in range(r + 1, q):
                acc = acc - identical_mul64(a[pr * q + c], sol[c])
            var pv = a[pr * q + r]
            sol[r] = acc / pv if pv != 0 else 0.0
            r -= 1
        var ssum: Float64 = 0.0
        for c in range(q):
            phi[unsafe_offset=c * k + j] = sol[c]
            ssum = ssum + sol[c]
        phi[unsafe_offset=q * k + j] = total - ssum
