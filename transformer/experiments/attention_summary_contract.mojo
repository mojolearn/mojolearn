# SPDX-License-Identifier: Apache-2.0
"""Pure host/device NN20 summary graph; numerical profile constants shared by all columns."""
from std.memory import bitcast
from std.sys.compile import is_defined
from checks.numerics import (GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_div, identical_exp, identical_fmax, identical_mul, identical_mul_add)

comptime NN20_BALANCED_SUMMARY_TREE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and is_defined["MOJOLEARN_NN20_BALANCED_SUMMARY_TREE"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
# Fixed numerical profile constant, never a dimension-targeting route.
comptime NN20_KEY_LEAF = 32


def summary_levels(keys: Int) -> Int:
    var tiles = (keys + NN20_KEY_LEAF - 1) // NN20_KEY_LEAF
    var levels = 1
    var capacity = 1
    while capacity < tiles:
        capacity *= 2
        levels += 1
    return levels


def summary_scratch_elements(rows: Int, keys: Int, width: Int, key_origin: Int = 0) -> Int:
    # Two temporaries plus one summary per binary level, for every row.
    return rows * (summary_levels(key_origin + keys) + 2) * (width + 2)


def _empty(summary: MutPointer[Float32, MutAnyOrigin], width: Int):
    summary.unsafe_store(0, bitcast[DType.float32](UInt32(0xFF800000)))
    summary.unsafe_store(1, Float32(0.0))
    for d in range(width):
        summary.unsafe_store(2 + d, Float32(0.0))


def _copy_summary(dst: MutPointer[Float32, MutAnyOrigin],
                  src: MutPointer[Float32, MutAnyOrigin], width: Int):
    for i in range(width + 2):
        dst.unsafe_store(i, src.unsafe_load(i))


def merge_summaries(dst: MutPointer[Float32, MutAnyOrigin],
                    left: MutPointer[Float32, MutAnyOrigin],
                    right: MutPointer[Float32, MutAnyOrigin], width: Int):
    """Exact graph of (max, scaled denominator, weighted numerator) merge.

    Empty summaries are bit-copy identities; no exp(-inf-(-inf)) occurs.
    Each rescale multiply rounds before the add. No FMA contraction crosses
    that boundary. A and B differ solely in which logical summaries meet.
    """
    if left.unsafe_load(1) == Float32(0.0):
        _copy_summary(dst, right, width)
        return
    if right.unsafe_load(1) == Float32(0.0):
        _copy_summary(dst, left, width)
        return
    var lm = left.unsafe_load(0)
    var rm = right.unsafe_load(0)
    var m = identical_fmax(lm, rm)
    var a = ftz(identical_exp(ftz(lm - m)))
    var b = ftz(identical_exp(ftz(rm - m)))
    dst.unsafe_store(0, m)
    for i in range(1, width + 2):
        var lval = ftz(identical_mul(ftz(left.unsafe_load(i)), a))
        var rval = ftz(identical_mul(ftz(right.unsafe_load(i)), b))
        dst.unsafe_store(i, ftz(lval + rval))


def _score(q: MutPointer[Float32, MutAnyOrigin],
           k: MutPointer[Float32, MutAnyOrigin], row: Int, group: Int,
           key: Int, keys: Int, head_dim: Int, scale: Float32) -> Float32:
    var score = Float32(0.0)
    for d in range(head_dim):
        score = ftz(identical_mul_add(ftz(q.unsafe_load(row * head_dim + d)),
            ftz(k.unsafe_load((group * keys + key) * head_dim + d)), score))
    return ftz(identical_mul(score, scale))


def leaf_summary(dst: MutPointer[Float32, MutAnyOrigin],
                 q: MutPointer[Float32, MutAnyOrigin],
                 k: MutPointer[Float32, MutAnyOrigin],
                 v: MutPointer[Float32, MutAnyOrigin], row: Int, group: Int,
                 lo: Int, hi: Int, keys: Int, head_dim: Int,
                 width: Int, scale: Float32):
    _empty(dst, width)
    if lo >= hi:
        return
    var m = _score(q, k, row, group, lo, keys, head_dim, scale)
    for j in range(lo + 1, hi):
        m = identical_fmax(m, _score(q, k, row, group, j, keys, head_dim, scale))
    dst.unsafe_store(0, m)
    var z = Float32(0.0)
    for j in range(lo, hi):
        var w = ftz(identical_exp(ftz(_score(q, k, row, group, j, keys, head_dim, scale) - m)))
        z = ftz(z + w)
        for d in range(width):
            var value = ftz(v.unsafe_load((group * keys + j) * width + d))
            dst.unsafe_store(2 + d, ftz(identical_mul_add(w, value, dst.unsafe_load(2 + d))))
    dst.unsafe_store(1, z)


def summary_attention_forward_row[TREE: Bool](
    q: MutPointer[Float32, MutAnyOrigin], k: MutPointer[Float32, MutAnyOrigin],
    v: MutPointer[Float32, MutAnyOrigin], lo: MutPointer[Int32, MutAnyOrigin],
    hi: MutPointer[Int32, MutAnyOrigin], out: MutPointer[Float32, MutAnyOrigin],
    maxes: MutPointer[Float32, MutAnyOrigin], denoms: MutPointer[Float32, MutAnyOrigin],
    scratch: MutPointer[Float32, MutAnyOrigin], status: MutPointer[Int32, MutAnyOrigin],
    row: Int, keys: Int, head_dim: Int, width: Int, queries_per_group: Int,
    scale: Float32, key_origin: Int = 0,
):
    var begin = Int(lo.unsafe_load(row))
    var end = Int(hi.unsafe_load(row))
    if begin < 0 or begin > end or end > keys:
        status.unsafe_store(row, Int32(1))
        return
    status.unsafe_store(row, Int32(0))
    var group = row // queries_per_group
    var fields = width + 2
    var levels = summary_levels(key_origin + keys)
    var stack = scratch + row * (levels + 2) * fields
    var current = stack + levels * fields
    var work = current + fields
    _empty(current, width)
    # Include empty leading absolute leaves to anchor EVERY tree level at
    # absolute key zero. Reindexing only the visible leaves would change
    # pair membership when a sliding window crosses a 32-key boundary.
    # Empty summaries are exact bit-copy identities at every merge.
    var tiles = (key_origin + keys + NN20_KEY_LEAF - 1) // NN20_KEY_LEAF
    comptime if TREE:
        for tile in range(tiles):
            # Leaves are aligned to absolute key index, not the visible
            # interval's start or the device's scheduling block size.
            leaf_summary(current, q, k, v, row, group,
                max(begin, tile * NN20_KEY_LEAF - key_origin),
                min(end, min(keys, (tile + 1) * NN20_KEY_LEAF - key_origin)),
                keys, head_dim, width, scale)
            var carry = tile
            var level = 0
            while (carry & 1) != 0:
                merge_summaries(work, stack + level * fields, current, width)
                var swap = current
                current = work
                work = swap
                carry >>= 1
                level += 1
            _copy_summary(stack + level * fields, current, width)
        var have = False
        # Low -> high folds the unmatched right fringe. For seven leaves
        # this is merge(0..3, merge(4..5, 6)), the adjacent-pair odd-carry tree.
        for level in range(levels):
            if ((tiles >> level) & 1) != 0:
                if not have:
                    _copy_summary(current, stack + level * fields, width)
                    have = True
                else:
                    merge_summaries(work, stack + level * fields, current, width)
                    var swap = current
                    current = work
                    work = swap
    else:
        for tile in range(tiles):
            leaf_summary(work, q, k, v, row, group,
                max(begin, tile * NN20_KEY_LEAF - key_origin),
                min(end, min(keys, (tile + 1) * NN20_KEY_LEAF - key_origin)),
                keys, head_dim, width, scale)
            merge_summaries(stack, current, work, width)
            _copy_summary(current, stack, width)
    var z = current.unsafe_load(1)
    maxes.unsafe_store(row, current.unsafe_load(0))
    denoms.unsafe_store(row, z)
    for d in range(width):
        var value = Float32(0.0)
        if z != Float32(0.0):
            value = ftz(identical_div(current.unsafe_load(2 + d), z))
        out.unsafe_store(row * width + d, value)


def _probability(q: MutPointer[Float32, MutAnyOrigin],
                 k: MutPointer[Float32, MutAnyOrigin],
                 maxes: MutPointer[Float32, MutAnyOrigin],
                 denoms: MutPointer[Float32, MutAnyOrigin],
                 row: Int, group: Int, j: Int, keys: Int,
                 head_dim: Int, scale: Float32) -> Float32:
    var z = denoms.unsafe_load(row)
    if z == Float32(0.0):
        return Float32(0.0)
    var e = ftz(identical_exp(ftz(_score(q, k, row, group, j, keys, head_dim, scale) - maxes.unsafe_load(row))))
    return ftz(identical_div(e, z))


def _dyv(v: MutPointer[Float32, MutAnyOrigin],
         dy: MutPointer[Float32, MutAnyOrigin], row: Int, group: Int,
         j: Int, keys: Int, width: Int) -> Float32:
    var dot = Float32(0.0)
    for d in range(width):
        dot = ftz(identical_mul_add(ftz(dy.unsafe_load(row * width + d)),
            ftz(v.unsafe_load((group * keys + j) * width + d)), dot))
    return dot


def summary_attention_rowdot(
    q: MutPointer[Float32, MutAnyOrigin], k: MutPointer[Float32, MutAnyOrigin],
    v: MutPointer[Float32, MutAnyOrigin], dy: MutPointer[Float32, MutAnyOrigin],
    lo: MutPointer[Int32, MutAnyOrigin], hi: MutPointer[Int32, MutAnyOrigin],
    maxes: MutPointer[Float32, MutAnyOrigin], denoms: MutPointer[Float32, MutAnyOrigin],
    zdot: MutPointer[Float32, MutAnyOrigin], status: MutPointer[Int32, MutAnyOrigin],
    row: Int, keys: Int, head_dim: Int, width: Int, queries_per_group: Int,
    scale: Float32,
):
    var z = Float32(0.0)
    if status.unsafe_load(row) == Int32(0):
        var group = row // queries_per_group
        for j in range(Int(lo.unsafe_load(row)), Int(hi.unsafe_load(row))):
            var p = _probability(q, k, maxes, denoms, row, group, j, keys, head_dim, scale)
            z = ftz(identical_mul_add(p, _dyv(v, dy, row, group, j, keys, width), z))
    zdot.unsafe_store(row, z)


def summary_attention_dq_cell(
    q: MutPointer[Float32, MutAnyOrigin], k: MutPointer[Float32, MutAnyOrigin],
    v: MutPointer[Float32, MutAnyOrigin], dy: MutPointer[Float32, MutAnyOrigin],
    lo: MutPointer[Int32, MutAnyOrigin], hi: MutPointer[Int32, MutAnyOrigin],
    maxes: MutPointer[Float32, MutAnyOrigin], denoms: MutPointer[Float32, MutAnyOrigin],
    zdot: MutPointer[Float32, MutAnyOrigin], status: MutPointer[Int32, MutAnyOrigin],
    dq: MutPointer[Float32, MutAnyOrigin], cell: Int,
    keys: Int, head_dim: Int, width: Int, queries_per_group: Int, scale: Float32,
):
    var row = cell // head_dim
    var d = cell % head_dim
    var acc = Float32(0.0)
    if status.unsafe_load(row) == Int32(0):
        var group = row // queries_per_group
        for j in range(Int(lo.unsafe_load(row)), Int(hi.unsafe_load(row))):
            var p = _probability(q, k, maxes, denoms, row, group, j, keys, head_dim, scale)
            var ds = ftz(identical_mul(p, ftz(_dyv(v, dy, row, group, j, keys, width) - zdot.unsafe_load(row))))
            var dscore = ftz(identical_mul(ds, scale))
            acc = ftz(identical_mul_add(dscore, ftz(k.unsafe_load((group * keys + j) * head_dim + d)), acc))
    dq.unsafe_store(cell, acc)


def summary_attention_dkdv_cell(
    q: MutPointer[Float32, MutAnyOrigin], k: MutPointer[Float32, MutAnyOrigin],
    v: MutPointer[Float32, MutAnyOrigin], dy: MutPointer[Float32, MutAnyOrigin],
    lo: MutPointer[Int32, MutAnyOrigin], hi: MutPointer[Int32, MutAnyOrigin],
    maxes: MutPointer[Float32, MutAnyOrigin], denoms: MutPointer[Float32, MutAnyOrigin],
    zdot: MutPointer[Float32, MutAnyOrigin], status: MutPointer[Int32, MutAnyOrigin],
    dk: MutPointer[Float32, MutAnyOrigin], dv: MutPointer[Float32, MutAnyOrigin],
    cell: Int, rows: Int, keys: Int, head_dim: Int, width: Int,
    queries_per_group: Int, scale: Float32,
):
    var columns = max(head_dim, width)
    var d = cell % columns
    var j = (cell // columns) % keys
    var group = cell // (columns * keys)
    var ak = Float32(0.0)
    var av = Float32(0.0)
    for row in range(group * queries_per_group, min(rows, (group + 1) * queries_per_group)):
        if status.unsafe_load(row) != Int32(0):
            continue
        if j < Int(lo.unsafe_load(row)) or j >= Int(hi.unsafe_load(row)):
            continue
        var p = _probability(q, k, maxes, denoms, row, group, j, keys, head_dim, scale)
        if d < head_dim:
            var ds = ftz(identical_mul(p, ftz(_dyv(v, dy, row, group, j, keys, width) - zdot.unsafe_load(row))))
            var dscore = ftz(identical_mul(ds, scale))
            ak = ftz(identical_mul_add(dscore, ftz(q.unsafe_load(row * head_dim + d)), ak))
        if d < width:
            av = ftz(identical_mul_add(p, ftz(dy.unsafe_load(row * width + d)), av))
    if d < head_dim:
        dk.unsafe_store((group * keys + j) * head_dim + d, ak)
    if d < width:
        dv.unsafe_store((group * keys + j) * width + d, av)


def summary_attention_host_forward(
    q: MutPointer[Float32, MutAnyOrigin], k: MutPointer[Float32, MutAnyOrigin],
    v: MutPointer[Float32, MutAnyOrigin], lo: MutPointer[Int32, MutAnyOrigin],
    hi: MutPointer[Int32, MutAnyOrigin], out: MutPointer[Float32, MutAnyOrigin],
    maxes: MutPointer[Float32, MutAnyOrigin], denoms: MutPointer[Float32, MutAnyOrigin],
    scratch: MutPointer[Float32, MutAnyOrigin], status: MutPointer[Int32, MutAnyOrigin],
    rows: Int, keys: Int, head_dim: Int, width: Int, queries_per_group: Int, scale: Float32,
):
    for row in range(rows):
        summary_attention_forward_row[NN20_BALANCED_SUMMARY_TREE](q, k, v, lo, hi,
            out, maxes, denoms, scratch, status, row, keys, head_dim, width,
            queries_per_group, scale)


def summary_attention_host_backward(
    q: MutPointer[Float32, MutAnyOrigin], k: MutPointer[Float32, MutAnyOrigin],
    v: MutPointer[Float32, MutAnyOrigin], dy: MutPointer[Float32, MutAnyOrigin],
    lo: MutPointer[Int32, MutAnyOrigin], hi: MutPointer[Int32, MutAnyOrigin],
    maxes: MutPointer[Float32, MutAnyOrigin], denoms: MutPointer[Float32, MutAnyOrigin],
    zdot: MutPointer[Float32, MutAnyOrigin], status: MutPointer[Int32, MutAnyOrigin],
    dq: MutPointer[Float32, MutAnyOrigin], dk: MutPointer[Float32, MutAnyOrigin],
    dv: MutPointer[Float32, MutAnyOrigin], rows: Int, keys: Int,
    head_dim: Int, width: Int, queries_per_group: Int, scale: Float32,
):
    for row in range(rows):
        summary_attention_rowdot(q, k, v, dy, lo, hi, maxes, denoms,
            zdot, status, row, keys, head_dim, width, queries_per_group, scale)
    for cell in range(rows * head_dim):
        summary_attention_dq_cell(q, k, v, dy, lo, hi, maxes, denoms,
            zdot, status, dq, cell, keys, head_dim, width, queries_per_group, scale)
    var groups = (rows + queries_per_group - 1) // queries_per_group
    for cell in range(groups * keys * max(head_dim, width)):
        summary_attention_dkdv_cell(q, k, v, dy, lo, hi, maxes, denoms,
            zdot, status, dk, dv, cell, rows, keys, head_dim, width,
            queries_per_group, scale)


