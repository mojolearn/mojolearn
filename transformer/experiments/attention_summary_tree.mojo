# SPDX-License-Identifier: Apache-2.0
"""NN20 standalone versioned stable-summary graph, not a model dispatch.

A: fixed 32-key leaves, adjacent-pair tree with odd carries, streamed through
an O(log(key_tiles)) stack. B: the SAME leaves combined left-to-right. The
B arm is a component control, not a claim of v1 or existing v2 equality.
Existing attention_v2's documented large-shape nonpromotion still stands.

Shared Mojo functions define host and device forward, row-dot and dQ/dK/dV.
Arithmetic graph, logical leaf coordinates and empty-row behavior are
independent of vendor/warp width. Public model/profile/checkpoint/decode
integration remains pending; this file is not imported into a model default.
No compile, identity/quality check, benchmark or candidate execution was run.

Finite Q/K/V and cotangents with a finite score regime must be admitted by
the future caller. Portable arithmetic alone is not a NaN-payload policy.
The component checks mask metadata on device with per-row status words.
Gradient is the explicitly prescribed softmax closed form using the saved
final summary normalizer, as in existing v2; we do not differentiate floating
rounding or inject a tie-dependent derivative through the max. This graph
requires independent gradient/task quality admission before model use.
"""
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import (
    GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_div,
    identical_exp, identical_fmax, identical_mul, identical_mul_add,
)

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


def summary_scratch_elements(rows: Int, keys: Int, width: Int) -> Int:
    # Two temporaries plus one summary per binary level, for every row.
    return rows * (summary_levels(keys) + 2) * (width + 2)


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
    scale: Float32,
):
    var begin = Int(lo.unsafe_load(row))
    var end = Int(hi.unsafe_load(row))
    if begin < 0 or begin > end or end > keys:
        status.unsafe_store(row, Int32(1))
        return
    status.unsafe_store(row, Int32(0))
    var group = row // queries_per_group
    var fields = width + 2
    var levels = summary_levels(keys)
    var stack = scratch + row * (levels + 2) * fields
    var current = stack + levels * fields
    var work = current + fields
    _empty(current, width)
    var tiles = (keys + NN20_KEY_LEAF - 1) // NN20_KEY_LEAF
    comptime if TREE:
        for tile in range(tiles):
            # Leaves are aligned to absolute key index, not the visible
            # interval's start or the device's scheduling block size.
            leaf_summary(current, q, k, v, row, group,
                max(begin, tile * NN20_KEY_LEAF),
                min(end, min(keys, (tile + 1) * NN20_KEY_LEAF)),
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
                max(begin, tile * NN20_KEY_LEAF),
                min(end, min(keys, (tile + 1) * NN20_KEY_LEAF)),
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


def summary_attention_forward_kernel(
    q: MutPointer[Float32, MutAnyOrigin], k: MutPointer[Float32, MutAnyOrigin],
    v: MutPointer[Float32, MutAnyOrigin], lo: MutPointer[Int32, MutAnyOrigin],
    hi: MutPointer[Int32, MutAnyOrigin], out: MutPointer[Float32, MutAnyOrigin],
    maxes: MutPointer[Float32, MutAnyOrigin], denoms: MutPointer[Float32, MutAnyOrigin],
    scratch: MutPointer[Float32, MutAnyOrigin], status: MutPointer[Int32, MutAnyOrigin],
    rows: Int32, keys: Int32, head_dim: Int32, width: Int32,
    queries_per_group: Int32, scale: Float32,
):
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if row < Int(rows):
        summary_attention_forward_row[NN20_BALANCED_SUMMARY_TREE](q, k, v, lo, hi,
            out, maxes, denoms, scratch, status, row, Int(keys), Int(head_dim),
            Int(width), Int(queries_per_group), scale)


def summary_attention_backward_kernel[STAGE: Int](
    q: MutPointer[Float32, MutAnyOrigin], k: MutPointer[Float32, MutAnyOrigin],
    v: MutPointer[Float32, MutAnyOrigin], dy: MutPointer[Float32, MutAnyOrigin],
    lo: MutPointer[Int32, MutAnyOrigin], hi: MutPointer[Int32, MutAnyOrigin],
    maxes: MutPointer[Float32, MutAnyOrigin], denoms: MutPointer[Float32, MutAnyOrigin],
    zdot: MutPointer[Float32, MutAnyOrigin], status: MutPointer[Int32, MutAnyOrigin],
    dq: MutPointer[Float32, MutAnyOrigin], dk: MutPointer[Float32, MutAnyOrigin],
    dv: MutPointer[Float32, MutAnyOrigin], rows: Int32, keys: Int32,
    head_dim: Int32, width: Int32, queries_per_group: Int32, scale: Float32,
):
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var r = Int(rows)
    var kcount = Int(keys)
    var hd = Int(head_dim)
    var w = Int(width)
    var qpg = Int(queries_per_group)
    comptime if STAGE == 0:
        if cell < r:
            summary_attention_rowdot(q, k, v, dy, lo, hi, maxes, denoms,
                zdot, status, cell, kcount, hd, w, qpg, scale)
    elif STAGE == 1:
        if cell < r * hd:
            summary_attention_dq_cell(q, k, v, dy, lo, hi, maxes, denoms,
                zdot, status, dq, cell, kcount, hd, w, qpg, scale)
    else:
        var groups = (r + qpg - 1) // qpg
        if cell < groups * kcount * max(hd, w):
            summary_attention_dkdv_cell(q, k, v, dy, lo, hi, maxes, denoms,
                zdot, status, dk, dv, cell, r, kcount, hd, w, qpg, scale)


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


def _require_shape(rows: Int, keys: Int, head_dim: Int, width: Int,
                   queries_per_group: Int) raises:
    if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("NN20 is an IDENTICAL-only component")
    if rows <= 0 or keys <= 0 or head_dim <= 0 or width <= 0 or queries_per_group <= 0:
        raise Error("NN20: positive dimensions/grouping required")
    # Device scalar ABI is Int32. This is an ABI bound, not a performance
    # cap; neighboring legal shapes always use the same numerical graph.
    if max(max(rows, keys), max(max(head_dim, width), queries_per_group)) > 2147483647:
        raise Error("NN20: dimension exceeds kernel Int32 ABI")


def enqueue_summary_attention_forward(
    ctx: DeviceContext,
    mut q: DeviceBuffer[DType.float32], mut k: DeviceBuffer[DType.float32],
    mut v: DeviceBuffer[DType.float32], mut lo: DeviceBuffer[DType.int32],
    mut hi: DeviceBuffer[DType.int32], mut out: DeviceBuffer[DType.float32],
    mut maxes: DeviceBuffer[DType.float32], mut denoms: DeviceBuffer[DType.float32],
    mut scratch: DeviceBuffer[DType.float32], mut status: DeviceBuffer[DType.int32],
    rows: Int, keys: Int, head_dim: Int, width: Int, queries_per_group: Int,
    scale: Float32,
) raises:
    _require_shape(rows, keys, head_dim, width, queries_per_group)
    var groups = (rows + queries_per_group - 1) // queries_per_group
    if len(q) < rows * head_dim or len(k) < groups * keys * head_dim or len(v) < groups * keys * width:
        raise Error("NN20: short Q/K/V operand")
    if len(out) < rows * width or len(maxes) < rows or len(denoms) < rows or len(status) < rows or len(lo) < rows or len(hi) < rows:
        raise Error("NN20: short output/row state")
    if len(scratch) < summary_scratch_elements(rows, keys, width):
        raise Error("NN20: bounded summary-stack scratch is too small")
    ctx.enqueue_function[summary_attention_forward_kernel](
        q.unsafe_ptr(), k.unsafe_ptr(), v.unsafe_ptr(), lo.unsafe_ptr(), hi.unsafe_ptr(),
        out.unsafe_ptr(), maxes.unsafe_ptr(), denoms.unsafe_ptr(), scratch.unsafe_ptr(), status.unsafe_ptr(),
        Int32(rows), Int32(keys), Int32(head_dim), Int32(width), Int32(queries_per_group), scale,
        grid_dim=((rows + 63) // 64, 1, 1), block_dim=(64, 1, 1),
    )


def enqueue_summary_attention_backward(
    ctx: DeviceContext,
    mut q: DeviceBuffer[DType.float32], mut k: DeviceBuffer[DType.float32],
    mut v: DeviceBuffer[DType.float32], mut dy: DeviceBuffer[DType.float32],
    mut lo: DeviceBuffer[DType.int32], mut hi: DeviceBuffer[DType.int32],
    mut maxes: DeviceBuffer[DType.float32], mut denoms: DeviceBuffer[DType.float32],
    mut zdot: DeviceBuffer[DType.float32], mut status: DeviceBuffer[DType.int32],
    mut dq: DeviceBuffer[DType.float32], mut dk: DeviceBuffer[DType.float32],
    mut dv: DeviceBuffer[DType.float32], rows: Int, keys: Int, head_dim: Int,
    width: Int, queries_per_group: Int, scale: Float32,
) raises:
    """Saved max/denom/mask/status must belong to this exact forward generation.

    Caller owns all buffers through completion and consumes/refuses nonzero
    status before accepting outputs. No source-data comparison is performed.
    Three in-order kernels establish z before its independent consumers.
    """
    _require_shape(rows, keys, head_dim, width, queries_per_group)
    var groups = (rows + queries_per_group - 1) // queries_per_group
    if len(q) < rows * head_dim or len(k) < groups * keys * head_dim or len(v) < groups * keys * width or len(dy) < rows * width:
        raise Error("NN20: short backward operand")
    if len(dq) < rows * head_dim or len(dk) < groups * keys * head_dim or len(dv) < groups * keys * width:
        raise Error("NN20: short gradient output")
    if len(maxes) < rows or len(denoms) < rows or len(zdot) < rows or len(status) < rows or len(lo) < rows or len(hi) < rows:
        raise Error("NN20: short backward row state")
    comptime for stage in range(3):
        var tasks = rows
        comptime if stage == 1:
            tasks = rows * head_dim
        elif stage == 2:
            tasks = groups * keys * max(head_dim, width)
        ctx.enqueue_function[summary_attention_backward_kernel[stage]](
            q.unsafe_ptr(), k.unsafe_ptr(), v.unsafe_ptr(), dy.unsafe_ptr(), lo.unsafe_ptr(), hi.unsafe_ptr(),
            maxes.unsafe_ptr(), denoms.unsafe_ptr(), zdot.unsafe_ptr(), status.unsafe_ptr(),
            dq.unsafe_ptr(), dk.unsafe_ptr(), dv.unsafe_ptr(), Int32(rows), Int32(keys),
            Int32(head_dim), Int32(width), Int32(queries_per_group), scale,
            grid_dim=((tasks + 255) // 256, 1, 1), block_dim=(256, 1, 1),
        )
