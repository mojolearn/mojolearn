# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The random forest's out-of-bag pass on the device
(fam2-forests, 2026-10-04, `IDN_RF_OOB_DEVICE`).

Before (DEVIATION 311): `compute_oob_score` downloaded X and the in-bag
masks and walked every (tree, row) on the host with
`DecisionTree.predict_one`, accumulating in Float64.

Now one thread per row walks, in increasing tree order, every tree whose
mask says the row is OUT of bag, adds the leaf's outputs into a binary64
accumulator and divides by the row's count. Metal has no Float64, so the
accumulator is the soft binary64 of `checks/soft_f64.mojo` (correctly
rounded add and divide on UInt64 words): the SAME IEEE results the host's
hardware Float64 chain produces, in the same order, so the
`oob_decision_function_` / `oob_prediction_` words do not move and every
vendor agrees by construction. X and the masks never leave the device.

cpu3-trees (2026-10-04): this is the only OOB route, in IDENTICAL and FAST
alike; the host walk, the host epilogue and their `_OFF` defines are gone.

THE SCORE EPILOGUE (fix-r1-rescue, audit B6, `IDN_RF_OOB_EPILOGUE_DEVICE`).
Before: the averaged predictions, the counts and y came back and the host
ran the argmax/accuracy (classifier) or the R^2 sums (regressor, a serial
Float64 fma chain). Now the device does both and returns two integers
(valid rows, correct rows) and the score's binary64 word:
  * classifier: argmax with a strict `>` from class 0 (first maximum, the
    host's rule), integer counts by order-free atomics, score =
    correct / n_valid correctly rounded. SAME BITS as before.
  * regressor: `xtrees/oob.mojo`'s exact binary64 sums (each term formed
    with correctly rounded soft ops, the sum rounded once): mean =
    fsum(y) / n_valid, den = fsum((y - mean)^2), num = fsum((y - pred)^2),
    score = 1 - num / den with sklearn's force_finite rules. NEW BITS for
    `oob_score_` (old: a sequential fma chain), the same on every vendor by
    construction (integer limb sums); a non-finite term or an overflowed
    sum gives NaN.
y and the counts no longer cross the bus.
"""
from std.atomic import Atomic
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.sys.compile import is_defined
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz
from checks.soft_f64 import (
    SF64_NAN,
    SF64_ONE,
    SF64_SIGN,
    SF64_ZERO,
    sf64_add,
    sf64_div,
    sf64_from_f32,
    sf64_from_int,
    sf64_gt,
    sf64_is_nan,
    sf64_mul,
    sf64_sub,
)
from xtrees.oob import E64_LIMBS, E64_THREADS, e64_add
from ensemble.flatnode import SparseTreeNode

comptime OOB_TPB = 256
#: words slots of the epilogue: [0] fsum(y), [1] den, [2] num, [3] mean,
#: [4] the score
comptime OOB_WORDS = 5
#: stats: [0] valid rows, [1] correct rows (classifier)
comptime OOB_STATS = 2


def rf_oob_append_tree_kernel[
    dtype: DType
](
    tree: MutPointer[SparseTreeNode[dtype], MutAnyOrigin],
    tree_leaves: MutPointer[Scalar[dtype], MutAnyOrigin],
    offsets: MutPointer[Int32, MutAnyOrigin],
    colid: MutPointer[Int32, MutAnyOrigin],
    quesval: MutPointer[Float32, MutAnyOrigin],
    left: MutPointer[Int32, MutAnyOrigin],
    leaves: MutPointer[Float32, MutAnyOrigin],
    n_nodes: Int32,
    n_out: Int32,
    base: Int32,
    tree_idx: Int32,
):
    """One finished tree into the forest's flat OOB model (cpu4-forest),
    straight from the builder's device tree (`leaf_d_tree`,
    `leaf_d_leaves`), one thread per node: the column, the left child and
    the threshold and leaf row narrowed to float32, at node `base + j`;
    thread 0 records the tree's base. What the host flatten wrote, field
    for field (`ColumnId`, `LeftChildId`, `QueryValue().cast`,
    `vector_leaf[...].cast`), without the model crossing the bus."""
    var j = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if j == 0:
        offsets[unsafe_offset = Int(tree_idx)] = base
    if j >= Int(n_nodes):
        return
    # A reference, not `var nd = tree[j]`: a whole SparseTreeNode copy out of
    # the device pointer crashes Apple's Metal compiler ("failed to compile
    # metallib"); the fields read are the same words.
    ref nd = tree[unsafe_offset=j]
    var g = Int(base) + j
    colid[unsafe_offset=g] = nd.ColumnId()
    left[unsafe_offset=g] = Int32(Int(nd.LeftChildId()))
    quesval[unsafe_offset=g] = nd.QueryValue().cast[DType.float32]()
    var no = Int(n_out)
    for k in range(no):
        leaves[unsafe_offset = g * no + k] = tree_leaves[
            unsafe_offset = j * no + k
        ].cast[DType.float32]()


def rf_oob_rows_kernel(
    x: MutPointer[Float32, MutAnyOrigin],
    masks: MutPointer[UInt8, MutAnyOrigin],
    offsets: MutPointer[Int32, MutAnyOrigin],
    colid: MutPointer[Int32, MutAnyOrigin],
    quesval: MutPointer[Float32, MutAnyOrigin],
    left: MutPointer[Int32, MutAnyOrigin],
    leaves: MutPointer[Float32, MutAnyOrigin],
    acc: MutPointer[UInt64, MutAnyOrigin],
    counts: MutPointer[Int32, MutAnyOrigin],
    n_rows: Int32,
    n_cols: Int32,
    n_trees: Int32,
    n_out: Int32,
    row_major: Int32,
    in_bag_scores: Int32,
):
    """`compute_oob_score`'s `:715-738` for one row per thread.

    `masks` is IN-BAG (`store_bootstrap_mask`): a tree scores the row only
    when its byte is 0. `in_bag_scores` is the check hook's inversion
    (sabotage 1) and is 0 on every shipping call. The walk is
    `predict_one`'s: `ftz(x) <= quesval` goes left, right is `left + 1`, a
    leaf has `left == -1`. `acc[r * n_out + k]` leaves as the binary64
    word of sum / count (the bare sum's zero when the count is zero, as
    the host leaves it); `counts[r]` is the number of trees that scored
    the row."""
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if r >= Int(n_rows):
        return
    var no = Int(n_out)
    var nr = Int(n_rows)
    var nc = Int(n_cols)
    for k in range(no):
        acc.unsafe_store(r * no + k, UInt64(0))
    var cnt = 0
    for t in range(Int(n_trees)):
        var in_bag = masks.unsafe_load(t * nr + r) != UInt8(0)
        if in_bag_scores != Int32(0):
            in_bag = not in_bag
        if in_bag:
            continue
        var base = Int(offsets.unsafe_load(t))
        var node = 0
        var child = Int(left.unsafe_load(base))
        while child != -1:
            var c = Int(colid.unsafe_load(base + node))
            var v: Float32
            if row_major != Int32(0):
                v = x.unsafe_load(r * nc + c)
            else:
                v = x.unsafe_load(c * nr + r)
            if ftz(v) <= quesval.unsafe_load(base + node):
                node = child
            else:
                node = child + 1
            child = Int(left.unsafe_load(base + node))
        for k in range(no):
            var cell = r * no + k
            acc.unsafe_store(
                cell,
                sf64_add(
                    acc.unsafe_load(cell),
                    sf64_from_f32(leaves.unsafe_load((base + node) * no + k)),
                ),
            )
        cnt += 1
    if cnt > 0:
        var d = sf64_from_int(cnt)
        for k in range(no):
            var cell = r * no + k
            acc.unsafe_store(cell, sf64_div(acc.unsafe_load(cell), d))
    counts.unsafe_store(r, Int32(cnt))


def rf_oob_rows_outputs_kernel(
    x: MutPointer[Float32, MutAnyOrigin], masks: MutPointer[UInt8, MutAnyOrigin],
    offsets: MutPointer[Int32, MutAnyOrigin], colid: MutPointer[Int32, MutAnyOrigin],
    quesval: MutPointer[Float32, MutAnyOrigin], left: MutPointer[Int32, MutAnyOrigin],
    leaves: MutPointer[Float32, MutAnyOrigin], acc: MutPointer[UInt64, MutAnyOrigin],
    counts: MutPointer[Int32, MutAnyOrigin], n_rows: Int32, n_cols: Int32,
    n_trees: Int32, n_out: Int32, row_major: Int32, in_bag_scores: Int32,
):
    """T15: one register fold per (row,output), fixed ascending tree order.

    NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
    Reuses fit-owned membership/model; one final output store replaces one
    read/write per voter. Multiclass repeats walks across outputs, a declared
    tradeoff to measure. Zero voters preserve +0 and an exact zero count.
    """
    var cell = Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if cell >= Int(n_rows)*Int(n_out):
        return
    var row = cell // Int(n_out)
    var output = cell % Int(n_out)
    var total = UInt64(0)
    var count = 0
    for t in range(Int(n_trees)):
        var in_bag = masks.unsafe_load(t*Int(n_rows)+row) != UInt8(0)
        if in_bag_scores != Int32(0):
            in_bag = not in_bag
        if in_bag:
            continue
        var base = Int(offsets.unsafe_load(t))
        var node = 0
        var child = Int(left.unsafe_load(base))
        while child != -1:
            var column = Int(colid.unsafe_load(base+node))
            var pos = row*Int(n_cols)+column if row_major != Int32(0) else column*Int(n_rows)+row
            node = child if ftz(x.unsafe_load(pos)) <= quesval.unsafe_load(base+node) else child+1
            child = Int(left.unsafe_load(base+node))
        total = sf64_add(total,sf64_from_f32(leaves.unsafe_load((base+node)*Int(n_out)+output)))
        count += 1
    if count > 0:
        total = sf64_div(total,sf64_from_int(count))
    acc.unsafe_store(cell,total)
    if output == 0:
        counts.unsafe_store(row,Int32(count))


# ------------------------------------------------- score epilogue (B6) --


@always_inline
def _gthread() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


def rf_oob_clf_stats_kernel[label_dt: DType](
    acc: MutPointer[UInt64, MutAnyOrigin],
    counts: MutPointer[Int32, MutAnyOrigin],
    y: MutPointer[Scalar[label_dt], MutAnyOrigin],
    n_rows: Int32,
    n_out: Int32,
    stats: MutPointer[Int32, MutAnyOrigin],
):
    """stats[0] += valid rows, stats[1] += valid rows whose argmax (strict
    `>` from class 0: the first maximum, as the host loop and `cp.argmax`)
    equals the label. A NaN probability never wins and a NaN running
    maximum is never replaced (`>` with a NaN is false on the host).
    Per-thread integer counts, one atomic each: order-free, exact."""
    var r = _gthread()
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    var no = Int(n_out)
    var valid = Int32(0)
    var correct = Int32(0)
    while r < Int(n_rows):
        if counts.unsafe_load(r) > Int32(0):
            valid += 1
            var best = 0
            var best_p = acc.unsafe_load(r * no)
            for k in range(1, no):
                var p = acc.unsafe_load(r * no + k)
                if (
                    not sf64_is_nan(p)
                    and not sf64_is_nan(best_p)
                    and sf64_gt(p, best_p)
                ):
                    best_p = p
                    best = k
            if Int(y.unsafe_load(r)) == best:
                correct += 1
        r += stride
    if valid != Int32(0):
        _ = Atomic.fetch_add(stats, valid)
    if correct != Int32(0):
        _ = Atomic.fetch_add(stats + 1, correct)


def rf_oob_reg_ysum_kernel(
    counts: MutPointer[Int32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    n_rows: Int32,
    part: MutPointer[Int64, MutAnyOrigin],
    stats: MutPointer[Int32, MutAnyOrigin],
    flags: MutPointer[Int32, MutAnyOrigin],
):
    """Thread t's limb row of part: the exact sum of y over its valid rows
    (E64_THREADS threads, grid-stride); stats[0] += its valid rows."""
    var t = _gthread()
    var row = part + t * E64_LIMBS
    for k in range(E64_LIMBS):
        row[unsafe_offset=k] = 0
    var valid = Int32(0)
    var i = t
    while i < Int(n_rows):
        if counts.unsafe_load(i) > Int32(0):
            e64_add(row, sf64_from_f32(y.unsafe_load(i)), flags)
            valid += 1
        i += E64_THREADS
    if valid != Int32(0):
        _ = Atomic.fetch_add(stats, valid)


def rf_oob_reg_mean_kernel(
    words: MutPointer[UInt64, MutAnyOrigin],
    stats: MutPointer[Int32, MutAnyOrigin],
):
    """words[3] = words[0] / n_valid. Its own kernel: the rounding stays in
    `round_kernel` (the gfx942 e64_round + sf64_div codegen fault,
    `xtrees/oob.mojo`)."""
    if _gthread() == 0:
        var nv = Int(stats.unsafe_load(0))
        if nv > 0:
            words.unsafe_store(3, sf64_div(words.unsafe_load(0), sf64_from_int(nv)))
        else:
            words.unsafe_store(3, SF64_ZERO)


def rf_oob_reg_sq_kernel(
    acc: MutPointer[UInt64, MutAnyOrigin],
    counts: MutPointer[Int32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    n_rows: Int32,
    n_out: Int32,
    words: MutPointer[UInt64, MutAnyOrigin],
    part_tot: MutPointer[Int64, MutAnyOrigin],
    part_res: MutPointer[Int64, MutAnyOrigin],
    flags: MutPointer[Int32, MutAnyOrigin],
):
    """Thread t's limb rows over its valid rows: the exact sums of
    (y - mean)^2 and (y - pred)^2, pred = acc[r * n_out] (output 0, as the
    host's r2), each term two correctly rounded binary64 operations."""
    var t = _gthread()
    var rt = part_tot + t * E64_LIMBS
    var rr = part_res + t * E64_LIMBS
    for k in range(E64_LIMBS):
        rt[unsafe_offset=k] = 0
        rr[unsafe_offset=k] = 0
    var mu = words.unsafe_load(3)
    var no = Int(n_out)
    var i = t
    while i < Int(n_rows):
        if counts.unsafe_load(i) > Int32(0):
            var v = sf64_from_f32(y.unsafe_load(i))
            var d = sf64_sub(v, mu)
            e64_add(rt, sf64_mul(d, d), flags)
            var e = sf64_sub(v, acc.unsafe_load(i * no))
            e64_add(rr, sf64_mul(e, e), flags)
        i += E64_THREADS


def rf_oob_score_kernel(
    words: MutPointer[UInt64, MutAnyOrigin],
    stats: MutPointer[Int32, MutAnyOrigin],
    flags: MutPointer[Int32, MutAnyOrigin],
    is_classifier: Int32,
):
    """words[4] = the score. Classifier: correct / n_valid. Regressor
    (sklearn r2_score, force_finite=True): num == 0 -> 1; den == 0 -> 0;
    else 1 - num / den; a non-finite term or an overflowed sum -> NaN."""
    if _gthread() != 0:
        return
    var nv = Int(stats.unsafe_load(0))
    if is_classifier != Int32(0):
        if nv > 0:
            words.unsafe_store(
                4,
                sf64_div(sf64_from_int(Int(stats.unsafe_load(1))), sf64_from_int(nv)),
            )
        else:
            words.unsafe_store(4, SF64_ZERO)
        return
    if flags.unsafe_load(0) != Int32(0) or flags.unsafe_load(1) != Int32(0):
        words.unsafe_store(4, SF64_NAN)
        return
    var den = words.unsafe_load(1)
    var num = words.unsafe_load(2)
    if (num & ~SF64_SIGN) == UInt64(0):
        words.unsafe_store(4, SF64_ONE)
    elif (den & ~SF64_SIGN) == UInt64(0):
        words.unsafe_store(4, SF64_ZERO)
    else:
        words.unsafe_store(4, sf64_sub(SF64_ONE, sf64_div(num, den)))
