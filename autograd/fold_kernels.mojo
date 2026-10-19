# SPDX-License-Identifier: Apache-2.0
# Lane ag-pass6-fold (2026-10-19, PASS6_IDEAS ideas D(1), E, F(1)): device
# kernels for three autograd body families, all IDENTICAL (each output word
# is the scalar body's word, so HostExec, Apple, NVIDIA and AMD keep one set
# of digests). DeviceExec._launch (sequence/exec_device.mojo) hands a launch
# to ag_pass6_launch when ag_pass6_op[OP]() holds; ag_pass6_launch returns
# False where its route rule says no, and the launch takes seq_kernel as
# before.
#
# D(1) STAGED FOLD LEAVES (AG_FOLD_STAGED). OP_AG_REDUCE_PART,
#   OP_AGK_BIN_VJP_PART and OP_AGS_LOSS_PART over AXIS_ALL or AXIS_COLS fold
#   64-term leaves that are CONTIGUOUS (autograd/ops.mojo ag_leaf_span), one
#   leaf a thread: neighbouring threads sit 256 B apart, so every load
#   instruction of a warp touches 32 (NVIDIA) / 64 (AMD) cache lines (opbench:
#   sum axis=None 65536x4096 2388 / 512 us = 52% / 35% of bandwidth against
#   axis=0's 92% / 65%; sum 512x64 axis=1 NV 15.0 us vs AMD 4.4). Here a
#   block of 64 threads owns 64 consecutive leaves: for each leaf in turn the
#   64 threads load (or compute) its words j = 0 .. 63 lane by lane
#   (coalesced) into threadgroup memory, then each thread adds its own
#   leaf's staged words ascending from +0.0 and finishes as the scalar body
#   (ag_part_finish / the BIN_VJP_PART store). Same words (the staged word
#   is the scalar body's ld / term), same adds, same order: same bits.
#   Threadgroup memory 64 x 65 floats = 16.6 KB a block (Metal's page is
#   32 KB); the pad column keeps the column-wise reads on distinct banks.
#   AXIS_ROWS keeps seq_kernel: its leaves already run down a column with
#   neighbouring threads on neighbouring columns (coalesced, 92%).
#   -D MOJOLEARN_IDN_FOLD_STAGED_OFF (or MOJOLEARN_IDN_ALL_OFF).
# E FOUR ELEMENTS A THREAD (AG_ELEM_VEC4). The binary, unary, reduce-VJP,
#   accumulate and dropout bodies on ag_vec4_kernel (autograd/ops.mojo
#   "FOUR ELEMENTS A THREAD"). Route rule: every pointer slot 16-byte
#   aligned (the tape's buffers are whole allocations; a sub-buffer at an
#   odd offset keeps seq_kernel) and n >= 4 x TPB elements (a cost rule:
#   below one block of four-element threads the launch floor is the cost and
#   a quarter of the threads would idle a block). -D MOJOLEARN_IDN_ELEM_VEC4_OFF.
# F(1) TRANSPOSE TILE (AG_TRANSPOSE_TILE). OP_AG_TRANSPOSE wrote lane-
#   contiguous words read at stride cols (opbench 65536x4096 7032 / 5214 us
#   = 35% / 7% of bandwidth). Here a block of 256 threads moves a 32 x 32
#   tile: coalesced loads into threadgroup memory (33-float rows), barrier,
#   coalesced stores of the transposed tile. Each destination word is
#   st(ld(source word)), the scalar body's. -D MOJOLEARN_IDN_TRANSPOSE_TILE_OFF.
# None of the three keys on a shape beyond the stated cost and alignment
# rules; every rule covers neighbouring shapes alike.
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast, stack_allocation
from std.sys.compile import is_defined
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz
from sequence.ops import Args, FP, ld, st, add
from autograd.ops import (
    OP_AG_FIRST, OP_AG_LAST, OP_AG_REDUCE_PART, OP_AG_TRANSPOSE, OP_AG_BIN_FWD, OP_AG_BIN_VJP,
    OP_AG_UNARY_FWD, OP_AG_UNARY_VJP, OP_AG_REDUCE_VJP, OP_AG_ACCUMULATE, OP_AGK_BIN_VJP_PART,
    AXIS_ROWS, AG_FOLD_LEAF, AG_VEC, AUTOGRAD_FOLD_V2, ag_leaf_span, ag_part_finish, _bin_vjp_at,
    apply_autograd, apply_autograd_vec4,
)
from autograd.data_ops import (
    OP_AG_DROPOUT_FWD, OP_AG_DROPOUT_VJP, apply_autograd_data, apply_autograd_data_vec4,
)
from autograd.fold_ops import OP_AGS_LOSS_PART, ags_loss_term

comptime _IDN_ALL_OFF = is_defined["MOJOLEARN_IDN_ALL_OFF"]()
comptime AG_FOLD_STAGED = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and AUTOGRAD_FOLD_V2
    and not (is_defined["MOJOLEARN_IDN_FOLD_STAGED_OFF"]() or _IDN_ALL_OFF)
)
comptime AG_ELEM_VEC4 = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not (is_defined["MOJOLEARN_IDN_ELEM_VEC4_OFF"]() or _IDN_ALL_OFF)
)
comptime AG_TRANSPOSE_TILE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not (is_defined["MOJOLEARN_IDN_TRANSPOSE_TILE_OFF"]() or _IDN_ALL_OFF)
)

# the staged leaves: one leaf a thread, a block of AG_FOLD_LEAF threads
comptime AGS_TPB = AG_FOLD_LEAF
comptime AGS_PAD = AG_FOLD_LEAF + 1
# the four-element kernel's block (sequence/exec_device.mojo TPB)
comptime AGV_TPB = 128
# the transpose tile
comptime AGT_TILE = 32
comptime AGT_PAD = AGT_TILE + 1
comptime AGT_TPB = 256
comptime AGT_ROWS = AGT_TPB // AGT_TILE


# The integer and float slots travel packed two to an Int64 word, as
# seq_kernel's (sequence/exec_device.mojo _pack_ii / _pack_ff, Metal's
# 31-argument limit); these are that file's helpers, here to keep the
# import one way.
@always_inline
def _pk_ii(lo: Int, hi: Int) -> Int64:
    return Int64(Int(UInt32(Int32(lo))) | (Int(UInt32(Int32(hi))) << 32))


@always_inline
def _pk_ff(lo: Float32, hi: Float32) -> Int64:
    return Int64(Int(bitcast[DType.uint32](lo)) | (Int(bitcast[DType.uint32](hi)) << 32))


@always_inline
def _lo(w: Int64) -> Int:
    return Int(Int32(w & 0xFFFFFFFF))


@always_inline
def _hi(w: Int64) -> Int:
    return Int(Int32((w >> 32) & 0xFFFFFFFF))


@always_inline
def _flo(w: Int64) -> Float32:
    return bitcast[DType.float32](UInt32(w & 0xFFFFFFFF))


@always_inline
def _fhi(w: Int64) -> Float32:
    return bitcast[DType.float32](UInt32((w >> 32) & 0xFFFFFFFF))


# ---------------------------------------------------------------- D(1)
@always_inline
def _ags_span[OP: Int](t: Int, a: Args) -> Tuple[Int, Int]:
    """Leaf t's (first index, count): OP_AGK_BIN_VJP_PART's slots are i2 rows,
    i3 cols, i9 leaf length; the others' i1, i2, i3. i0 the axis."""
    comptime if OP == OP_AGK_BIN_VJP_PART:
        return ag_leaf_span(t, a.i2, a.i3, a.i0, a.i9)
    else:
        return ag_leaf_span(t, a.i1, a.i2, a.i0, a.i3)


@always_inline
def _ags_term[OP: Int](a: Args, i: Int) -> Float32:
    """Term i, the word the scalar body adds."""
    comptime if OP == OP_AG_REDUCE_PART:
        return ld(a.p0, i)
    elif OP == OP_AGS_LOSS_PART:
        return ags_loss_term(a, i)
    else:
        return ftz(_bin_vjp_at(i, a))


@always_inline
def _ags_finish[OP: Int](t: Int, a: Args, s: Float32):
    """The scalar body's store of leaf t's chain s."""
    comptime if OP == OP_AGK_BIN_VJP_PART:
        if a.i11 == 1:
            st(a.p5, t, add(Float32(0), s))
        else:
            st(a.p4, t, s)
    else:
        ag_part_finish(t, a, s)


def ag_fold_staged_kernel[OP: Int](
    p0: FP, p1: FP, p2: FP, p3: FP, p4: FP, p5: FP,
    p6: FP, p7: FP, p8: FP, p9: FP, p10: FP, p11: FP,
    i01: Int64, i23: Int64, i45: Int64, i67: Int64, i89: Int64, i1011: Int64,
    f01: Int64, f23: Int64, f45: Int64, f67: Int64,
    n: Int64,
):
    """Leaves first .. first + 63 of OP (n leaves in all) on one block of
    AGS_TPB threads; thread `lane` folds leaf first + lane. Every thread
    reaches the barrier (the leaf count is block-uniform)."""
    var lane = Int(thread_idx.x)
    var first = Int(block_idx.x) * AGS_TPB
    var a = Args(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11,
                 _lo(i01), _hi(i01), _lo(i23), _hi(i23), _lo(i45), _hi(i45),
                 _lo(i67), _hi(i67), _lo(i89), _hi(i89), _lo(i1011), _hi(i1011),
                 _flo(f01), _fhi(f01), _flo(f23), _fhi(f23),
                 _flo(f45), _fhi(f45), _flo(f67), _fhi(f67))
    var tile = stack_allocation[AGS_TPB * AGS_PAD, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var leaves = min(AGS_TPB, Int(n) - first)
    # stage: leaf i's word `lane` (neighbouring lanes, neighbouring words)
    for i in range(leaves):
        var span = _ags_span[OP](first + i, a)
        if lane < span[1]:
            tile[i * AGS_PAD + lane] = _ags_term[OP](a, span[0] + lane)
    barrier()
    if lane < leaves:
        var span = _ags_span[OP](first + lane, a)
        var s = Float32(0)
        for k in range(span[1]):
            s = add(s, tile[lane * AGS_PAD + k])
        _ags_finish[OP](first + lane, a, s)


# ---------------------------------------------------------------- E
def ag_vec4_kernel[OP: Int](
    p0: FP, p1: FP, p2: FP, p3: FP, p4: FP, p5: FP,
    p6: FP, p7: FP, p8: FP, p9: FP, p10: FP, p11: FP,
    i01: Int64, i23: Int64, i45: Int64, i67: Int64, i89: Int64, i1011: Int64,
    f01: Int64, f23: Int64, f45: Int64, f67: Int64,
    n: Int64,
):
    """Elements 4q .. 4q + 3 of OP on thread q; the last n % 4 elements on
    the scalar body."""
    comptime IS_AG = OP >= OP_AG_FIRST and OP <= OP_AG_LAST
    var q = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var t0 = q * AG_VEC
    var count = Int(n)
    if t0 < count:
        var a = Args(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11,
                     _lo(i01), _hi(i01), _lo(i23), _hi(i23), _lo(i45), _hi(i45),
                     _lo(i67), _hi(i67), _lo(i89), _hi(i89), _lo(i1011), _hi(i1011),
                     _flo(f01), _fhi(f01), _flo(f23), _fhi(f23),
                     _flo(f45), _fhi(f45), _flo(f67), _fhi(f67))
        if t0 + AG_VEC <= count:
            comptime if IS_AG:
                apply_autograd_vec4[OP](t0, a)
            else:
                apply_autograd_data_vec4[OP](t0, a)
        else:
            var t = t0
            while t < count:
                comptime if IS_AG:
                    apply_autograd[OP](t, a)
                else:
                    apply_autograd_data[OP](t, a)
                t += 1


# ---------------------------------------------------------------- F(1)
def ag_transpose_tile_kernel(src: FP, dst: FP, rows: Int32, cols: Int32):
    """dst (cols x rows) = the transpose of src (rows x cols), one 32 x 32
    tile a block of AGT_TPB threads."""
    var R = Int(rows)
    var C = Int(cols)
    var ntc = (C + AGT_TILE - 1) // AGT_TILE
    var b = Int(block_idx.x)
    var tr = b // ntc
    var tc = b - tr * ntc
    var r0 = tr * AGT_TILE
    var c0 = tc * AGT_TILE
    var tid = Int(thread_idx.x)
    var lx = tid % AGT_TILE
    var ly = tid // AGT_TILE
    var tile = stack_allocation[AGT_TILE * AGT_PAD, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var j = 0
    while j < AGT_TILE:
        var r = r0 + ly + j
        var c = c0 + lx
        if r < R and c < C:
            tile[(ly + j) * AGT_PAD + lx] = ld(src, r * C + c)
        j += AGT_ROWS
    barrier()
    j = 0
    while j < AGT_TILE:
        # destination row = a source column, destination column = a source row
        var dr = c0 + ly + j
        var dc = r0 + lx
        if dr < C and dc < R:
            st(dst, dr * R + dc, tile[lx * AGT_PAD + ly + j])
        j += AGT_ROWS


# ---------------------------------------------------------------- routing
@always_inline
def ag_pass6_op[OP: Int]() -> Bool:
    """The codes ag_pass6_launch may take."""
    return (
        (AG_FOLD_STAGED and (OP == OP_AG_REDUCE_PART or OP == OP_AGK_BIN_VJP_PART or OP == OP_AGS_LOSS_PART))
        or (AG_ELEM_VEC4 and (OP == OP_AG_BIN_FWD or OP == OP_AG_BIN_VJP or OP == OP_AG_UNARY_FWD
                              or OP == OP_AG_UNARY_VJP or OP == OP_AG_REDUCE_VJP or OP == OP_AG_ACCUMULATE
                              or OP == OP_AG_DROPOUT_FWD or OP == OP_AG_DROPOUT_VJP))
        or (AG_TRANSPOSE_TILE and OP == OP_AG_TRANSPOSE)
    )


@always_inline
def _aligned16(a: Args) -> Bool:
    """Every pointer slot on a 16-byte boundary (the placeholder's address,
    64, is)."""
    var bits = (Int(a.p0) | Int(a.p1) | Int(a.p2) | Int(a.p3) | Int(a.p4) | Int(a.p5)
                | Int(a.p6) | Int(a.p7) | Int(a.p8) | Int(a.p9) | Int(a.p10) | Int(a.p11))
    return (bits & 15) == 0


def ag_pass6_launch[OP: Int](ctx: DeviceContext, a: Args, n: Int) raises -> Bool:
    """Launch OP (n threads of seq_kernel's mapping) on its pass-6 kernel and
    return True, or return False (nothing launched) where the route rule
    says seq_kernel."""
    comptime if OP == OP_AG_TRANSPOSE:
        var blocks = ((a.i0 + AGT_TILE - 1) // AGT_TILE) * ((a.i1 + AGT_TILE - 1) // AGT_TILE)
        ctx.enqueue_function[ag_transpose_tile_kernel](
            a.p0, a.p1, Int32(a.i0), Int32(a.i1),
            grid_dim=(blocks, 1, 1), block_dim=(AGT_TPB, 1, 1),
        )
        return True
    elif OP == OP_AG_REDUCE_PART or OP == OP_AGK_BIN_VJP_PART or OP == OP_AGS_LOSS_PART:
        # stride-1 leaves of v2's fixed length only (AXIS_ROWS is coalesced)
        var leaf = a.i9 if OP == OP_AGK_BIN_VJP_PART else a.i3
        if a.i0 == AXIS_ROWS or leaf != AG_FOLD_LEAF:
            return False
        ctx.enqueue_function[ag_fold_staged_kernel[OP]](
            a.p0, a.p1, a.p2, a.p3, a.p4, a.p5, a.p6, a.p7, a.p8, a.p9, a.p10, a.p11,
            _pk_ii(a.i0, a.i1), _pk_ii(a.i2, a.i3), _pk_ii(a.i4, a.i5),
            _pk_ii(a.i6, a.i7), _pk_ii(a.i8, a.i9), _pk_ii(a.i10, a.i11),
            _pk_ff(a.f0, a.f1), _pk_ff(a.f2, a.f3), _pk_ff(a.f4, a.f5), _pk_ff(a.f6, a.f7),
            Int64(n),
            grid_dim=((n + AGS_TPB - 1) // AGS_TPB, 1, 1), block_dim=(AGS_TPB, 1, 1),
        )
        return True
    else:
        if n < AG_VEC * AGV_TPB or not _aligned16(a):
            return False
        var threads = (n + AG_VEC - 1) // AG_VEC
        ctx.enqueue_function[ag_vec4_kernel[OP]](
            a.p0, a.p1, a.p2, a.p3, a.p4, a.p5, a.p6, a.p7, a.p8, a.p9, a.p10, a.p11,
            _pk_ii(a.i0, a.i1), _pk_ii(a.i2, a.i3), _pk_ii(a.i4, a.i5),
            _pk_ii(a.i6, a.i7), _pk_ii(a.i8, a.i9), _pk_ii(a.i10, a.i11),
            _pk_ff(a.f0, a.f1), _pk_ff(a.f2, a.f3), _pk_ff(a.f4, a.f5), _pk_ff(a.f6, a.f7),
            Int64(n),
            grid_dim=((threads + AGV_TPB - 1) // AGV_TPB, 1, 1), block_dim=(AGV_TPB, 1, 1),
        )
        return True
