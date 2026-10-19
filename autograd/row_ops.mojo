# SPDX-License-Identifier: Apache-2.0
# Row bodies of the softmax family and the AUC sort's in-block stages (lane
# ag-pass6-rows, pass-6 ideas C and I), shared by HostExec and DeviceExec.
#
# IDEA C, MOJOLEARN_IDN_ROW_COOP (IDENTICAL default; -D
# MOJOLEARN_IDN_ROW_COOP_OFF or MOJOLEARN_IDN_ALL_OFF restores the one-chain
# rows and their bits). Softmax, log-softmax, their VJP, the cross-entropy
# row loss and VJP and the fused mean-SCE VJP were one thread per row, a
# serial chain over the columns of software exponentials (~100 dependent
# operations each): 512 x 256 softmax 105 / 248 us NV / AMD, CE 512 x 256
# 48.8 / 127.5 (opbench, pass-6 ideas section 1 C). Under ROW_COOP every
# row sum is the fold32 order (sequence/fold32.mojo): slot j adds columns
# j, j + 32, j + 64, ... ascending from +0.0, then `tree32`. The device runs
# a row on one 32-lane coop cell (sequence/coop.mojo coop_row_*: lane j
# owns slot j, `coop_rw_total` is tree32); the host column and the device's
# one-thread rows (fewer than 32 columns, exec_device's coop rule) run the
# same 32 slots serially here. BIT VERSION for the sums (every column moves
# together: host, Metal, NVIDIA, AMD); the exponentials, the division, the
# max and the argmax are elementwise or order-free, so their words are the
# serial row's. The max is the serial `>` scan's (the first maximum; a NaN
# in column 0 sticks, a later NaN is skipped); the coop cell reproduces it
# exactly with (value, index) partials (`row_lane_takes`, `row_best_takes`).
# Gated on IDENTICAL like SEQ_LN_COOP: an Apple FAST build keeps today's
# rows until a FAST A/B decides.
#
# IDEA I, MOJOLEARN_IDN_AUC_BLOCK_SORT (default on, identical: same network,
# same pairs, same stage order; -D MOJOLEARN_IDN_AUC_BLOCK_SORT_OFF or
# MOJOLEARN_IDN_ALL_OFF restores one launch per stage). roc_auc ran one
# OP_LS_BITONIC launch per (k, j) stage: 53 / 86 / 145 launches at n 512 /
# 4096 / 65536, all at the launch floor (205 / 318 / 565 us NV, 192 / 361 /
# 714 AMD). OP_ROW_BITONIC_BLOCK runs every stage whose partner distance is
# below ROW_SORT_BLOCK inside one block of ROW_SORT_BLOCK elements: one
# launch for all merge sizes k <= ROW_SORT_BLOCK, then per larger k one
# OP_LS_BITONIC launch per stage with j >= ROW_SORT_BLOCK and one block
# launch for its stages j < ROW_SORT_BLOCK. Launches at 65536 rows: 136 ->
# 1 + 21 + 6 = 28; at 4096: 78 -> 6; at 512: 45 -> 1. A stage's pairs are
# disjoint and a pair (i, i + j) with j < ROW_SORT_BLOCK stays inside its
# aligned block, so running the block's stages behind barriers is the same
# as launch boundaries: the same compare-exchanges on the same values.
#
# Each body reads and writes only the slots in its docstring and loads
# through branches, never through a pointer select (Metal rule,
# autograd/ops.mojo _ld_right). Every function a kernel reaches is
# @always_inline.
from std.math import nan
from std.sys.compile import is_defined
from sequence.ops import Args, FP, ld, st, add, sub, mul
from sequence.fold32 import FOLD_L, tree32
from checks.numerics import (
    GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_div, identical_exp, identical_log,
)

comptime ROW_COOP = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not (is_defined["MOJOLEARN_IDN_ROW_COOP_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]())
)
comptime AUC_BLOCK_SORT = not (
    is_defined["MOJOLEARN_IDN_AUC_BLOCK_SORT_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)

#: slots of a row fold (the coop cell's lanes; sequence/coop.mojo asserts
#: COOP_W == ROW_L)
comptime ROW_L = FOLD_L

# Pass-6 rows family, 2200.. (sequence/dispatch.mojo asserts it free).
comptime OP_ROW_BITONIC_BLOCK = 2200
comptime OP_ROW_FIRST = OP_ROW_BITONIC_BLOCK
comptime OP_ROW_LAST = OP_ROW_BITONIC_BLOCK

#: elements of one in-block sort: two arrays of 1024 floats are 8 KB of
#: threadgroup memory and ROW_SORT_BLOCK / 2 = 512 threads, inside every
#: vendor's block limits (Metal and AMD 1024 threads, 32 KB+ of shared
#: memory); a hardware bound, not a data size.
comptime ROW_SORT_BLOCK = 1024

#: label value of a padding slot in the AUC sort (real labels are 0 or 1);
#: autograd/loss_ops.mojo AUC_PAD is this value
comptime ROW_SORT_PAD = Float32(2)


@always_inline
def row_exp(x: Float32, m: Float32) -> Float32:
    """exp(x - m), the rows' exponential (the serial bodies' expression)."""
    return ftz(identical_exp(sub(x, m)))


@always_inline
def row_div(a: Float32, b: Float32) -> Float32:
    return ftz(identical_div(a, b))


@always_inline
def row_first_max(p: FP, base: Int, cols: Int) -> Int:
    """Column of the first maximum of the row: the serial `>` scan from
    column 0 (a NaN in column 0 is kept, a later NaN never replaces)."""
    var best = 0
    var m = ld(p, base)
    for c in range(1, cols):
        var v = ld(p, base + c)
        if v > m:
            m = v
            best = c
    return best


@always_inline
def row_lane_takes(m: Float32, i: Int, v: Float32, c: Int) -> Bool:
    """A coop lane's scan step over its columns: does column c (value v)
    replace the lane's candidate (m, i)? i < 0 is no candidate yet. Column 0
    is always taken (it starts the serial scan, NaN or not); another column
    starts a candidate only when it is not NaN, then replaces by `>`."""
    if i < 0:
        return c == 0 or v == v
    return v > m


@always_inline
def row_best_takes(ma: Float32, ia: Int32, mb: Float32, ib: Int32) -> Bool:
    """Does partial (mb, ib) win over (ma, ia)? The serial scan's answer is
    column 0 when it is NaN, else the lowest column holding the largest
    non-NaN value (== ties, so -0.0 and +0.0 tie). This is a total order on
    distinct columns, so any tree of these choices gives that column, and
    the two lanes of a butterfly pair pick the same winner."""
    if ib < 0:
        return False
    if ia < 0:
        return True
    if ia == 0 and ma != ma:
        return False
    if ib == 0 and mb != mb:
        return True
    if mb > ma:
        return True
    return mb == ma and ib < ia


@always_inline
def row_expsum32(p: FP, base: Int, cols: Int, m: Float32) -> Float32:
    """sum_c exp(p[base + c] - m) in the fold32 order."""
    var s = InlineArray[Float32, ROW_L](fill=Float32(0.0))
    var k = 0
    while k < cols:
        var n = min(ROW_L, cols - k)
        for j in range(n):
            s[j] = add(s[j], row_exp(ld(p, base + k + j), m))
        k += ROW_L
    return tree32(s)


@always_inline
def row_softmax_fwd(t: Int, a: Args):
    """OP_AG_SOFTMAX_FWD under ROW_COOP: p0 x, p1 y, i0 = 1 for log-softmax,
    i1 cols. Thread t = row t. Softmax parks each exponential in p1 (its own
    row) and reads it back for the division; p1 must not overlap p0."""
    var cols = a.i1
    var base = t * cols
    var m = ld(a.p0, base + row_first_max(a.p0, base, cols))
    if a.i0 == 0:
        var s = InlineArray[Float32, ROW_L](fill=Float32(0.0))
        var k = 0
        while k < cols:
            var n = min(ROW_L, cols - k)
            for j in range(n):
                var e = row_exp(ld(a.p0, base + k + j), m)
                s[j] = add(s[j], e)
                st(a.p1, base + k + j, e)
            k += ROW_L
        var total = tree32(s)
        for c in range(cols):
            st(a.p1, base + c, row_div(ld(a.p1, base + c), total))
    else:
        var log_total = ftz(identical_log(row_expsum32(a.p0, base, cols, m)))
        for c in range(cols):
            st(a.p1, base + c, sub(sub(ld(a.p0, base + c), m), log_total))


@always_inline
def row_softmax_vjp_term(a: Args, i: Int) -> Float32:
    """The dot product's term of element i: g, times y for softmax."""
    var g = ld(a.p1, i)
    if a.i0 == 0:
        g = mul(g, ld(a.p0, i))
    return g


@always_inline
def row_softmax_vjp_out(a: Args, i: Int, total: Float32):
    if a.i0 == 0:
        st(a.p2, i, mul(ld(a.p0, i), sub(ld(a.p1, i), total)))
    else:
        st(a.p2, i, sub(ld(a.p1, i), mul(ftz(identical_exp(ld(a.p0, i))), total)))


@always_inline
def row_softmax_vjp(t: Int, a: Args):
    """OP_AG_SOFTMAX_VJP under ROW_COOP: p0 forward output y, p1 upstream, p2
    destination, i0 = 1 for log-softmax, i1 cols. Thread t = row t; the row's
    dot product in the fold32 order."""
    var cols = a.i1
    var base = t * cols
    var s = InlineArray[Float32, ROW_L](fill=Float32(0.0))
    var k = 0
    while k < cols:
        var n = min(ROW_L, cols - k)
        for j in range(n):
            s[j] = add(s[j], row_softmax_vjp_term(a, base + k + j))
        k += ROW_L
    var total = tree32(s)
    for c in range(cols):
        row_softmax_vjp_out(a, base + c, total)


@always_inline
def row_ce_fwd(t: Int, a: Args):
    """OP_LS_CE_FWD under ROW_COOP: p0 logits (rows x i1), p1 codes, p2 row
    loss = log(sum exp(z - m)) - (z_y - m). A code outside [0, i1) gives NaN."""
    var cols = a.i1
    var base = t * cols
    var y = Int(ld(a.p1, t))
    if y < 0 or y >= cols:
        a.p2.unsafe_store(t, nan[DType.float32]())
        return
    var m = ld(a.p0, base + row_first_max(a.p0, base, cols))
    var ls = ftz(identical_log(row_expsum32(a.p0, base, cols, m)))
    st(a.p2, t, sub(ls, sub(ld(a.p0, base + y), m)))


@always_inline
def row_ce_grad(t: Int, a: Args, g: Float32):
    """Row t of (softmax - onehot) * g into p3: p0 logits (rows x i1), p1
    codes. The cross-entropy VJP's and the fused SCE VJP's row."""
    var cols = a.i1
    var base = t * cols
    var y = Int(ld(a.p1, t))
    var m = ld(a.p0, base + row_first_max(a.p0, base, cols))
    var s = row_expsum32(a.p0, base, cols, m)
    for c in range(cols):
        var p = row_div(row_exp(ld(a.p0, base + c), m), s)
        if c == y:
            p = sub(p, Float32(1))
        st(a.p3, base + c, mul(p, g))


@always_inline
def row_sort_after(ka: Float32, la: Float32, kb: Float32, lb: Float32) -> Bool:
    """(ka, la) sorts after (kb, lb): padding last, then by score."""
    if la == ROW_SORT_PAD:
        return lb != ROW_SORT_PAD
    if lb == ROW_SORT_PAD:
        return False
    return ka > kb


@always_inline
def row_bitonic_exchange(keys: FP, labels: FP, i: Int, j: Int, k: Int):
    """The compare-exchange of pair (i, i + j) at merge size k: ascending
    when i & k == 0 (op_ls_bitonic's body)."""
    var l = i + j
    var ka = ld(keys, i)
    var la = ld(labels, i)
    var kb = ld(keys, l)
    var lb = ld(labels, l)
    var swap: Bool
    if (i & k) == 0:
        swap = row_sort_after(ka, la, kb, lb)
    else:
        swap = row_sort_after(kb, lb, ka, la)
    if swap:
        st(keys, i, kb)
        st(labels, i, lb)
        st(keys, l, ka)
        st(labels, l, la)


@always_inline
def op_row_bitonic_block(t: Int, a: Args):
    """The in-block stages of the AUC sort, block t: p0 keys, p1 labels
    (exchanged together), i0 the first merge size, i1 the last (powers of
    two, i0 <= i1), i2 the block's elements B (a power of two, at most
    ROW_SORT_BLOCK, B <= the padded length). For k = i0, 2 i0, ..., i1 the
    stages j = min(k, B) / 2, ..., 1 over every pair of elements
    [t B, (t + 1) B), in the network's order. Global thread T = t B / 2 + q
    of a stage owns i = 2 j (T // j) + T % j = t B + 2 j (q // j) + q % j,
    since j divides B / 2. The device runs this block in one threadgroup
    (autograd/row_sort.mojo), the same exchanges behind barriers."""
    var B = a.i2
    var lo = t * B
    var k = a.i0
    while k <= a.i1:
        var j = min(k, B) // 2
        while j > 0:
            for q in range(B // 2):
                row_bitonic_exchange(a.p0, a.p1, lo + 2 * j * (q // j) + q % j, j, k)
            j //= 2
        k *= 2


@always_inline
def apply_rows[OP: Int](t: Int, a: Args):
    comptime if OP == OP_ROW_BITONIC_BLOCK:
        op_row_bitonic_block(t, a)
