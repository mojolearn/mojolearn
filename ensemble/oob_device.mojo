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

`-D MOJOLEARN_IDN_RF_OOB_DEVICE_OFF` (or `MOJOLEARN_IDN_ALL_OFF`) restores
the host walk.
"""
from std.gpu import block_dim, block_idx, thread_idx
from std.sys.compile import is_defined
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz
from checks.soft_f64 import sf64_add, sf64_div, sf64_from_f32, sf64_from_int

comptime IDN_RF_OOB_DEVICE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not (
        is_defined["MOJOLEARN_IDN_RF_OOB_DEVICE_OFF"]()
        or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
    )
)

comptime OOB_TPB = 256


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
