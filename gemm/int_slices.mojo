# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""fp32 GEMM on the integer matrix units by exact integer slicing: the device
plans of the research profile `mojolearn.identical.gemm.fp32.v2-int`.

PROTOTYPE (lane/gemm-int-slices, 2026-10-28, REPORT idea 13). Switch
`-D MOJOLEARN_IDN_GEMM_INT_SLICES`, default OFF; off, nothing here is reached
and the linalg GEMM is fp32.v1 byte for byte (the hook in
`gemm/host_entry.mojo` is a `comptime if` on this file's constant). Contract:
`docs/plans/GEMM_INT_SLICES.md`. Seams: `gemm/int_slices_seams.mojo`. Host
column: `gemm/host/gemm_int_slices_oracle.mojo`. Gate:
`gemm/checks/gemm_int_slices_check.mojo`.

PRIOR ART IN THIS TREE, AND WHAT IS DIFFERENT. Two Ozaki routes were measured
and deleted: `MOJOLEARN_IDN_GEMM_OZAKI_LINALG` (2026-10-08, gemm 4096^3 S=4
NV 3.97x / AMD 2.06x slower) and `MOJOLEARN_IDN_NEURAL_GEMM_OZAKI_SLICES`
(2026-10-11), `docs/TOMBSTONES.md`. Both ran their digit products on the
REFERENCE int8 unit kernel (`gemm_int8_mma.mojo`: one warp one 16x16 tile,
every fragment read from device memory at every k step, measured at 1.38 to
1.66x fp32.v1's time on an H100 by lane/lowbit-units) with Int32 atomics
across a k-split, and kept only the high triangle of digit products. This
prototype runs every product on the TUNED four-plane kernel
(`identical_gemm_int8_pieces_tuned_with_plan`: both planes of each operand
staged once per window in threadgroup memory, four unit steps per tile; the
L40S read it at 67 TFLOPS-equivalent on qkv 512x4096x4096,
`docs/plans/IDENTICAL_NEURAL_ROADMAP.md:161`), with no atomics, and keeps
every product, so a cell is the correctly rounded exact dot product of the
codes. Whether that is faster than fp32.v1 is the A/B the plan doc names.

THE PLANS (every one is the profile: same integers, same seams)
  UNIT   NVIDIA and AMD (`lib_int8_matrix_unit_for`): quantize op(A) and op(B)
         to four int8 planes each (two halves x two planes), four launches of
         the tuned four-plane kernel (hi.hi, hi.lo, lo.hi, lo.lo), each three
         exact Int32 sums per cell, then one epilogue launch, one thread per
         cell: `int15_recombine` per launch's sums, then the seams.
  FLAT   every other GPU column (Apple): the same planes, one thread per
         cell, the four half products as Int64 sums `p` ascending. No bounded
         launch slicing on Apple yet (clause W-14's rule): OWED before Apple
         runs it at large shapes.
  HOST   the CPU column: the host oracle file named above, on the caller's memory.

THE QUANTIZER is one launch per operand, one block per row along k: the
absmax by a tree in threadgroup memory (a maximum is order-free), then the
codes and planes. It reads the operand through `(row_stride, k_stride)`, so
the transposed orientations read with a stride (NOT coalesced for op(B) of
OP_NN). SCHEDULING ONLY; a coalesced transposing quantizer (the int15 file's
32x32 tile) is owed if the A/B shows the quantizer matters.

THE WORKSPACE is allocated per call (planes 4 (m + n) k bytes, exponents,
four sums buffers of 12 m n bytes in all). Pooling it is owed; it moves no
bit.

THE SABOTAGE ARM. `-D MOJOLEARN_INT_SLICES_SABOTAGE=1` drops the lo.lo
product in the UNIT and FLAT epilogues (never in the host oracle): the gate
must print DIFFER under it on shapes whose low halves are nonzero.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.memory import stack_allocation
from std.sys import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.kernel_matrix import COLUMN_CPU, TARGET_COLUMN, lib_int8_matrix_unit_for
from gemm.contract import OP_NN, OP_NT, OP_TN
from gemm.checks.gemm_int8_mma_tuned import (
    INT8_PIECES_MAX_K,
    int15_box_low_bandwidth,
    int8_pieces_dispatch_for,
    identical_gemm_int8_pieces_tuned_with_plan,
)
from gemm.int_slices_seams import (
    INT_SLICES_MAX_K,
    int_slices_abs_ftz,
    int_slices_cell,
    int_slices_code,
    int_slices_half_from_planes,
    int_slices_half_hi,
    int_slices_half_lo,
    int_slices_plane_h,
    int_slices_plane_l,
    int_slices_product_from_sums,
    int_slices_row_exponent,
    int_slices_strides,
)

#: THE SWITCH. Default OFF.
comptime IDN_GEMM_INT_SLICES = is_defined["MOJOLEARN_IDN_GEMM_INT_SLICES"]()

#: The sabotage arm (see the module docstring).
comptime INT_SLICES_SABOTAGE = is_defined["MOJOLEARN_INT_SLICES_SABOTAGE"]()

#: Forces the FLAT plan on a column that has the unit, so one box can compare
#: UNIT against FLAT (the gate's plan-agreement arm).
comptime INT_SLICES_FORCE_FLAT = is_defined["MOJOLEARN_INT_SLICES_FORCE_FLAT"]()

#: Whether this build has the integer matrix unit plan.
comptime INT_SLICES_UNIT = lib_int8_matrix_unit_for[TARGET_COLUMN]() and not INT_SLICES_FORCE_FLAT

#: Threads of the quantizer's and the cell kernels' blocks. SCHEDULING.
comptime INT_SLICES_TPB = 256


def int_slices_admits(m: Int, n: Int, k: Int) -> Bool:
    """Whether the profile serves the shape: positive extents and `k` within
    the exactness bound (clause S4; the tuned kernel's own bound is the
    same 65536 for this plane domain). From the arithmetic only."""
    return m > 0 and n > 0 and k > 0 and k <= INT_SLICES_MAX_K and k <= INT8_PIECES_MAX_K


def int_slices_quantize_kernel(
    hh: MutPointer[Int8, MutAnyOrigin],
    hl: MutPointer[Int8, MutAnyOrigin],
    lh: MutPointer[Int8, MutAnyOrigin],
    ll: MutPointer[Int8, MutAnyOrigin],
    e: MutPointer[Int32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    rows_in: Int32,
    k_in: Int32,
    rs_in: Int32,
    ps_in: Int32,
):
    """One block per row along k (clauses S1 to S3): the row's absmax by a
    tree, its exponent, then per value the code and its four planes, stored
    `r * k + p` (k-contiguous, the layout the four-plane kernel reads).
    `hh, hl` are the high half's planes, `lh, ll` the low half's."""
    var rows = Int(rows_in)
    var k = Int(k_in)
    var rs = Int(rs_in)
    var ps = Int(ps_in)
    var r = Int(block_idx.x)
    if r >= rows:
        return
    var red = stack_allocation[INT_SLICES_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    var best = Float32(0.0)
    var p = tid
    while p < k:
        var v = int_slices_abs_ftz(x.unsafe_load(r * rs + p * ps))
        if v > best:
            best = v
        p += INT_SLICES_TPB
    red[unsafe_offset=tid] = best
    barrier()
    var active = INT_SLICES_TPB // 2
    while active > 0:
        if tid < active:
            var o = red[unsafe_offset=tid + active]
            if o > red[unsafe_offset=tid]:
                red[unsafe_offset=tid] = o
        barrier()
        active = active // 2
    var ex = int_slices_row_exponent(red[unsafe_offset=0])
    if tid == 0:
        e.unsafe_store(r, Int32(ex))
    p = tid
    while p < k:
        var q = int_slices_code(x.unsafe_load(r * rs + p * ps), ex)
        var hi = int_slices_half_hi(q)
        var lo = int_slices_half_lo(q)
        var o = r * k + p
        hh.unsafe_store(o, int_slices_plane_h(hi))
        hl.unsafe_store(o, int_slices_plane_l(hi))
        lh.unsafe_store(o, int_slices_plane_h(lo))
        ll.unsafe_store(o, int_slices_plane_l(lo))
        p += INT_SLICES_TPB


def int_slices_epilogue_kernel(
    c: MutPointer[Float32, MutAnyOrigin],
    s_hh: MutPointer[Int32, MutAnyOrigin],
    s_hl: MutPointer[Int32, MutAnyOrigin],
    s_lh: MutPointer[Int32, MutAnyOrigin],
    s_ll: MutPointer[Int32, MutAnyOrigin],
    ea: MutPointer[Int32, MutAnyOrigin],
    eb: MutPointer[Int32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
):
    """UNIT plan, last launch, one thread per cell: each half product from
    its launch's three sums at `3 (i n + j)` (`int15_recombine`), then
    clauses S5 to S7."""
    var m = Int(m_in)
    var n = Int(n_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= m * n:
        return
    var i = cell // n
    var j = cell - i * n
    var o = 3 * cell
    var p_hh = int_slices_product_from_sums(s_hh.unsafe_load(o), s_hh.unsafe_load(o + 1), s_hh.unsafe_load(o + 2))
    var p_hl = int_slices_product_from_sums(s_hl.unsafe_load(o), s_hl.unsafe_load(o + 1), s_hl.unsafe_load(o + 2))
    var p_lh = int_slices_product_from_sums(s_lh.unsafe_load(o), s_lh.unsafe_load(o + 1), s_lh.unsafe_load(o + 2))
    var p_ll = int_slices_product_from_sums(s_ll.unsafe_load(o), s_ll.unsafe_load(o + 1), s_ll.unsafe_load(o + 2))
    comptime if INT_SLICES_SABOTAGE:
        p_ll = Int64(0)
    c.unsafe_store(cell, int_slices_cell(p_hh, p_hl, p_lh, p_ll, Int(ea.unsafe_load(i)) + Int(eb.unsafe_load(j))))


def int_slices_flat_kernel(
    c: MutPointer[Float32, MutAnyOrigin],
    ahh: MutPointer[Int8, MutAnyOrigin],
    ahl: MutPointer[Int8, MutAnyOrigin],
    alh: MutPointer[Int8, MutAnyOrigin],
    all_: MutPointer[Int8, MutAnyOrigin],
    ea: MutPointer[Int32, MutAnyOrigin],
    bhh: MutPointer[Int8, MutAnyOrigin],
    bhl: MutPointer[Int8, MutAnyOrigin],
    blh: MutPointer[Int8, MutAnyOrigin],
    bll: MutPointer[Int8, MutAnyOrigin],
    eb: MutPointer[Int32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    k_in: Int32,
):
    """FLAT plan, one thread per cell: the four half products from the
    planes as Int64 sums of Int32 products, `p` ascending, then S5 to S7."""
    var m = Int(m_in)
    var n = Int(n_in)
    var k = Int(k_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= m * n:
        return
    var i = cell // n
    var j = cell - i * n
    var p_hh = Int64(0)
    var p_hl = Int64(0)
    var p_lh = Int64(0)
    var p_ll = Int64(0)
    for p in range(k):
        var oa = i * k + p
        var ob = j * k + p
        var xh = int_slices_half_from_planes(ahh.unsafe_load(oa), ahl.unsafe_load(oa))
        var xl = int_slices_half_from_planes(alh.unsafe_load(oa), all_.unsafe_load(oa))
        var yh = int_slices_half_from_planes(bhh.unsafe_load(ob), bhl.unsafe_load(ob))
        var yl = int_slices_half_from_planes(blh.unsafe_load(ob), bll.unsafe_load(ob))
        p_hh += Int64(xh * yh)
        p_hl += Int64(xh * yl)
        p_lh += Int64(xl * yh)
        p_ll += Int64(xl * yl)
    comptime if INT_SLICES_SABOTAGE:
        p_ll = Int64(0)
    c.unsafe_store(cell, int_slices_cell(p_hh, p_hl, p_lh, p_ll, Int(ea.unsafe_load(i)) + Int(eb.unsafe_load(j))))


def int_slices_plan_name() -> String:
    comptime if TARGET_COLUMN == COLUMN_CPU:
        return String("host")
    elif INT_SLICES_UNIT:
        return String("unit(tuned four-plane kernel x4 + epilogue)")
    else:
        return String("flat")


def _quantize(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    mut hh: DeviceBuffer[DType.int8],
    mut hl: DeviceBuffer[DType.int8],
    mut lh: DeviceBuffer[DType.int8],
    mut ll: DeviceBuffer[DType.int8],
    mut e: DeviceBuffer[DType.int32],
    rows: Int,
    k: Int,
    rs: Int,
    ps: Int,
) raises:
    ctx.enqueue_function[int_slices_quantize_kernel](
        hh.unsafe_ptr(),
        hl.unsafe_ptr(),
        lh.unsafe_ptr(),
        ll.unsafe_ptr(),
        e.unsafe_ptr(),
        x.unsafe_ptr(),
        Int32(rows),
        Int32(k),
        Int32(rs),
        Int32(ps),
        grid_dim=(rows, 1, 1),
        block_dim=(INT_SLICES_TPB, 1, 1),
    )


def identical_gemm_int_slices_into(
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
    op: Int,
) raises:
    """`C[m x n] = op(A) . op(B)` under the prototype profile on a GPU
    column, from device buffers (row-major, contiguous; contract section
    0.1's orientations). Asynchronous on the one in-order context; the
    workspace is released with the frees enqueued behind the launches.
    Refuses by name a shape outside `int_slices_admits` and the CPU column
    (whose answer is `gemm/host/gemm_int_slices_oracle.mojo` on host memory)."""
    if not int_slices_admits(m, n, k):
        raise Error(
            "identical_gemm_int_slices_into: m, n, k must be positive and k at most "
            + String(INT_SLICES_MAX_K) + ", got m=" + String(m) + " n=" + String(n) + " k=" + String(k)
        )
    if op != OP_NN and op != OP_NT and op != OP_TN:
        raise Error("identical_gemm_int_slices_into: op must be 0, 1 or 2, got " + String(op))
    comptime if TARGET_COLUMN == COLUMN_CPU:
        raise Error("identical_gemm_int_slices_into: the CPU column's answer is gemm_int_slices_oracle")
    else:
        var st = int_slices_strides(op, m, n, k)
        var ahh = ctx.enqueue_create_buffer[DType.int8](m * k)
        var ahl = ctx.enqueue_create_buffer[DType.int8](m * k)
        var alh = ctx.enqueue_create_buffer[DType.int8](m * k)
        var all_ = ctx.enqueue_create_buffer[DType.int8](m * k)
        var bhh = ctx.enqueue_create_buffer[DType.int8](n * k)
        var bhl = ctx.enqueue_create_buffer[DType.int8](n * k)
        var blh = ctx.enqueue_create_buffer[DType.int8](n * k)
        var bll = ctx.enqueue_create_buffer[DType.int8](n * k)
        var ea = ctx.enqueue_create_buffer[DType.int32](m)
        var eb = ctx.enqueue_create_buffer[DType.int32](n)
        _quantize(ctx, a, ahh, ahl, alh, all_, ea, m, k, st[0], st[1])
        _quantize(ctx, b, bhh, bhl, blh, bll, eb, n, k, st[2], st[3])
        var cells = m * n
        var blocks = (cells + INT_SLICES_TPB - 1) // INT_SLICES_TPB
        comptime if INT_SLICES_UNIT:
            var plan = int8_pieces_dispatch_for(int15_box_low_bandwidth(String(ctx.name())), m, n, k)
            var s_hh = ctx.enqueue_create_buffer[DType.int32](3 * cells)
            var s_hl = ctx.enqueue_create_buffer[DType.int32](3 * cells)
            var s_lh = ctx.enqueue_create_buffer[DType.int32](3 * cells)
            var s_ll = ctx.enqueue_create_buffer[DType.int32](3 * cells)
            identical_gemm_int8_pieces_tuned_with_plan(ctx, s_hh, ahh, ahl, bhh, bhl, m, n, k, plan)
            identical_gemm_int8_pieces_tuned_with_plan(ctx, s_hl, ahh, ahl, blh, bll, m, n, k, plan)
            identical_gemm_int8_pieces_tuned_with_plan(ctx, s_lh, alh, all_, bhh, bhl, m, n, k, plan)
            identical_gemm_int8_pieces_tuned_with_plan(ctx, s_ll, alh, all_, blh, bll, m, n, k, plan)
            ctx.enqueue_function[int_slices_epilogue_kernel](
                c.unsafe_ptr(),
                s_hh.unsafe_ptr(),
                s_hl.unsafe_ptr(),
                s_lh.unsafe_ptr(),
                s_ll.unsafe_ptr(),
                ea.unsafe_ptr(),
                eb.unsafe_ptr(),
                Int32(m),
                Int32(n),
                grid_dim=(blocks, 1, 1),
                block_dim=(INT_SLICES_TPB, 1, 1),
            )
            _ = s_hh^
            _ = s_hl^
            _ = s_lh^
            _ = s_ll^
        else:
            ctx.enqueue_function[int_slices_flat_kernel](
                c.unsafe_ptr(),
                ahh.unsafe_ptr(),
                ahl.unsafe_ptr(),
                alh.unsafe_ptr(),
                all_.unsafe_ptr(),
                ea.unsafe_ptr(),
                bhh.unsafe_ptr(),
                bhl.unsafe_ptr(),
                blh.unsafe_ptr(),
                bll.unsafe_ptr(),
                eb.unsafe_ptr(),
                Int32(m),
                Int32(n),
                Int32(k),
                grid_dim=(blocks, 1, 1),
                block_dim=(INT_SLICES_TPB, 1, 1),
            )
        # The workspace's frees are enqueued behind the launches above on the
        # same in-order context (gemm/linalg_native.mojo drains them, DEVIATION 3010).
        _ = ahh^
        _ = ahl^
        _ = alh^
        _ = all_^
        _ = bhh^
        _ = bhl^
        _ = blh^
        _ = bll^
        _ = ea^
        _ = eb^

