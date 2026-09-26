# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The device realizations of the two low-bit profiles.

    mojolearn.identical.gemm.bf16f32.v1   `identical_gemm_bf16w_into`
    mojolearn.identical.gemm.int8i32.v1   `identical_gemm_int8_into`

Contract `gemm/IDENTICAL_LOWBIT_CONTRACT.md`; answers
`gemm/host/gemm_lowbit_oracle.mojo`; gates `gemm/checks/gemm_lowbit_check.mojo`.
Lane lane/identical-lowbit-inference, 2026-09-17, DEVIATIONS 2906 to 2909.

TWO EXECUTION PLANS FOR bf16, ONE ANSWER (DEVIATION 2906)
---------------------------------------------------------
A bf16 right operand can reach the fp32.v1 arithmetic two ways, and both are
the profile because the widening is exact:

  FUSED   `identical_gemm_bf16w_flat_kernel`: the fp32 profile's flat plan
          with the right operand loaded as bf16 and widened in a register.
          Reads half the weight bytes. Chosen for the decode shape, where a
          projection is bandwidth-bound and the fused read is the point of
          storing bf16 at all.
  WIDEN   `bf16_widen_kernel` into a float32 scratch, then
          `identical_gemm_into[False]`, every plan of the fp32 profile
          available. Chosen for the prefill shape, where compute dominates.

`BF16W_FUSED_MAX_CELLS` decides between them and is SCHEDULING: the checks
run both plans on every shape and require the same bits, so the threshold
cannot move a bit any more than a block size can. Contract L-8.

int8 (DEVIATION 2907)
---------------------
`identical_gemm_int8_flat_kernel` accumulates in Int32, `p` ascending, one
thread per cell, and dequantizes through `dequant_int8_pinned`. Because the
sum is exact the fold tree is not needed and not used; the only floating
seam is the dequantization, and it is one exact multiply plus the flush.
`quantize_rows_int8_kernel` is one thread per row: the absmax, the exponent,
the codes, in the oracle's order.

TWO EXECUTION PLANS FOR int8, ONE ANSWER (DEVIATION 2910, contract L-9)
-----------------------------------------------------------------------
  FLAT    `identical_gemm_int8_flat_kernel`, above. Every column.
  MMA     `gemm/checks/gemm_int8_mma.mojo::identical_gemm_int8_mma_kernel`,
          the vendor's integer matrix unit (NVIDIA IMMA m16n8k32 s8/s32,
          AMD CDNA MFMA i32_16x16x32_i8), Int32 accumulation, the same
          `dequant_int8_pinned` epilogue. Columns whose kernel-matrix row
          `lib_int8_matrix_unit_for` says True: NVIDIA and AMD.
Both are the profile because an int8 product is exact and an Int32 sum of
exact integers is order-free: no tile shape and no internal summation order
can move a bit, so the choice is SCHEDULING and `check_int8_mma_matches_flat`
requires the two plans' bits to match on every shape.
`identical_gemm_int8_into` takes the MMA plan when the row says True, the
shape is admitted (`int8_mma_admits`, every legal shape) and the build does
not carry `-D MOJOLEARN_INT8_FORCE_FLAT=1`; the flat plan otherwise. Apple
stays on the flat plan (Metal has no integer matrix unit).

THE SABOTAGE ARM (DEVIATION 2908)
---------------------------------
`-D MOJOLEARN_LOWBIT_SABOTAGE=1` flips the value of every cell both kernels
store (the fp32 profile's own `gemm_oracle_sabotage_value_flip`), so a build
carrying it must fail every gate in `gemm_lowbit_check.mojo`. A value arm,
not an order arm: the int8 sum is exact and would fold an order arm away.
"""

from max.gpu import block_dim, block_idx, thread_idx
from std.sys import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    bf16_bits_to_f32,
    dequant_int8_pinned,
    f32_to_bf16_bits_rne,
    ftz,
    identical_mul_add,
    int8_row_exponent,
    quantize_int8_value,
)
from checks.rtf_seam import rtf_mul_add
from gemm.checks.gemm_identical import (
    GEMM_FOLD_SLOTS,
    _fold_drain,
    _fold_push,
    _leaf_at,
    _leaf_bounds,
    contract_partition,
    gemm_operand_strides,
    identical_gemm_into,
    identical_gemm_workspace_max_floats,
    step_count_device_alloc,
    step_count_sync,
)
from checks.kernel_matrix import TARGET_COLUMN, lib_int8_matrix_unit_for
from gemm.checks.gemm_int8_mma import (
    identical_gemm_int8_mma_into,
    int8_mma_admits,
)
from gemm.host.gemm_lowbit_oracle import INT8_MAX_K
from gemm.host.gemm_oracle import (
    OP_NN,
    OP_NT,
    OP_TN,
    gemm_oracle_sabotage_value_flip,
)

#: DEVIATION 2908, the value arm. Off in every build that does not name it.
comptime LOWBIT_SABOTAGE = is_defined["MOJOLEARN_LOWBIT_SABOTAGE"]()

#: DEVIATION 2910: `-D MOJOLEARN_INT8_FORCE_FLAT=1` keeps the flat int8 plan
#: on every column, the unit's column included, so a box that has the unit
#: can run the whole gate file through the flat plan and compare. SCHEDULING;
#: cannot move a bit (contract L-9).
comptime INT8_FORCE_FLAT = is_defined["MOJOLEARN_INT8_FORCE_FLAT"]()

#: DEVIATION 2910: whether this build's dispatcher may pick the MMA plan at
#: all. The row and the define; the shape is asked per call.
comptime INT8_MMA_ENABLED = lib_int8_matrix_unit_for[TARGET_COLUMN]() and not INT8_FORCE_FLAT


def int8_plan_dispatch_name() -> String:
    """What `identical_gemm_int8_into` runs on this build: for the gate
    banner, so a log says which plan produced the oracle comparison."""
    comptime if INT8_FORCE_FLAT:
        return String("flat (MOJOLEARN_INT8_FORCE_FLAT)")
    elif lib_int8_matrix_unit_for[TARGET_COLUMN]():
        return String("mma (lib_int8_matrix_unit_for)")
    else:
        return String("flat (no int8 matrix unit on this column)")

#: Threads per block for every kernel in this file. SCHEDULING; no float
#: crosses a thread boundary in any of them.
comptime LOWBIT_TPB = 256

#: DEVIATION 2906: the fused plan serves outputs of at most this many cells,
#: the widen plan the rest. SCHEDULING (contract L-8): both plans are the
#: profile and `check_bf16_plans_agree` requires their bits to match.
comptime BF16W_FUSED_MAX_CELLS = 16384


def lowbit_sabotage_name() -> String:
    if LOWBIT_SABOTAGE:
        return String("LOWBIT_VALUE_FLIP")
    return String("none")


# ===========================================================================
# bf16 conversions on the device
# ===========================================================================


def bf16_widen_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    src: MutPointer[UInt16, MutAnyOrigin],
    n_in: Int32,
):
    """Contract L-1, one element per thread."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_in):
        return
    dst.unsafe_store(i, bf16_bits_to_f32(src.unsafe_load(i)))


def bf16_narrow_kernel(
    dst: MutPointer[UInt16, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """Contract L-2, one element per thread."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_in):
        return
    dst.unsafe_store(i, f32_to_bf16_bits_rne(src.unsafe_load(i)))


def bf16_widen(
    ctx: DeviceContext,
    mut dst: DeviceBuffer[DType.float32],
    mut src: DeviceBuffer[DType.uint16],
    count: Int,
) raises:
    if count <= 0:
        return
    ctx.enqueue_function[bf16_widen_kernel](
        dst.unsafe_ptr(),
        src.unsafe_ptr(),
        Int32(count),
        grid_dim=((count + LOWBIT_TPB - 1) // LOWBIT_TPB, 1, 1),
        block_dim=(LOWBIT_TPB, 1, 1),
    )


def bf16_narrow(
    ctx: DeviceContext,
    mut dst: DeviceBuffer[DType.uint16],
    mut src: DeviceBuffer[DType.float32],
    count: Int,
) raises:
    if count <= 0:
        return
    ctx.enqueue_function[bf16_narrow_kernel](
        dst.unsafe_ptr(),
        src.unsafe_ptr(),
        Int32(count),
        grid_dim=((count + LOWBIT_TPB - 1) // LOWBIT_TPB, 1, 1),
        block_dim=(LOWBIT_TPB, 1, 1),
    )


# ===========================================================================
# bf16f32.v1, the fused plan
# ===========================================================================


def identical_gemm_bf16w_flat_kernel(
    c: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Float32, MutAnyOrigin],
    b: MutPointer[UInt16, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    k_in: Int32,
    leaf_in: Int32,
    p_in: Int32,
    a_si_in: Int32,
    a_sp_in: Int32,
    b_sp_in: Int32,
    b_sj_in: Int32,
):
    """`identical_gemm_flat_kernel` with the right operand widened from
    bf16 as it is loaded. Every other character is the fp32 profile's:
    ascending `p` inside a leaf, one `identical_mul_add` per step, the
    accumulator flushed after every step, the register fold tree across
    leaves. `leaf_in` and `p_in` come from `contract_partition(k)`."""
    var m = Int(m_in)
    var n = Int(n_in)
    var k = Int(k_in)
    var leaf = Int(leaf_in)
    var p_count = Int(p_in)
    var a_si = Int(a_si_in)
    var a_sp = Int(a_sp_in)
    var b_sp = Int(b_sp_in)
    var b_sj = Int(b_sj_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= m * n:
        return
    var i = cell // n
    var j = cell - i * n
    var stack = SIMD[DType.float32, GEMM_FOLD_SLOTS](0.0)
    var occ = 0
    var a_row = i * a_si
    var b_col = j * b_sj
    for t in range(p_count):
        var bounds = _leaf_bounds(_leaf_at(t, p_count), leaf, k)
        var acc = Float32(0.0)
        for p in range(bounds[0], bounds[1]):
            acc = rtf_mul_add(
                ftz(a.unsafe_load(a_row + p * a_sp)),
                ftz(bf16_bits_to_f32(b.unsafe_load(p * b_sp + b_col))),
                acc,
            )
        _ = _fold_push(stack, occ, ftz(acc))
    var out = ftz(_fold_drain(stack, occ))
    comptime if LOWBIT_SABOTAGE:
        out = gemm_oracle_sabotage_value_flip(out)
    c.unsafe_store(cell, out)


struct LowbitWorkspace(Movable):
    """Scratch for the low-bit GEMMs on ONE in-order context: the fp32
    profile's workspace and a float32 image of the widened right operand.
    Grown on demand; growth drains the context first, as `GemmWorkspace`
    does, because a kernel still in flight may be reading the buffer that
    is about to be replaced."""

    var ws: DeviceBuffer[DType.float32]
    var wide: DeviceBuffer[DType.float32]

    def __init__(out self, ctx: DeviceContext) raises:
        step_count_device_alloc()
        self.ws = ctx.enqueue_create_buffer[DType.float32](1)
        step_count_device_alloc()
        self.wide = ctx.enqueue_create_buffer[DType.float32](1)

    def ensure(mut self, ctx: DeviceContext, ws_floats: Int, wide_floats: Int) raises:
        var need_ws = ws_floats
        if need_ws < 1:
            need_ws = 1
        var need_wide = wide_floats
        if need_wide < 1:
            need_wide = 1
        if need_ws > len(self.ws) or need_wide > len(self.wide):
            step_count_sync()
            ctx.synchronize()
            if need_ws > len(self.ws):
                step_count_device_alloc()
                self.ws = ctx.enqueue_create_buffer[DType.float32](need_ws)
            if need_wide > len(self.wide):
                step_count_device_alloc()
                self.wide = ctx.enqueue_create_buffer[DType.float32](need_wide)


def identical_gemm_bf16w_fused_into(
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.uint16],
    m: Int,
    n: Int,
    k: Int,
    op: Int,
) raises:
    """The FUSED plan, always. The checks call this directly to compare it
    with the widen plan; production goes through `identical_gemm_bf16w_into`."""
    _refuse_shape(m, n, k, op)
    var lp = contract_partition(k)
    var st = gemm_operand_strides(op, m, n, k)
    ctx.enqueue_function[identical_gemm_bf16w_flat_kernel](
        c.unsafe_ptr(),
        a.unsafe_ptr(),
        b.unsafe_ptr(),
        Int32(m),
        Int32(n),
        Int32(k),
        Int32(lp[0]),
        Int32(lp[1]),
        Int32(st[0]),
        Int32(st[1]),
        Int32(st[2]),
        Int32(st[3]),
        grid_dim=((m * n + LOWBIT_TPB - 1) // LOWBIT_TPB, 1, 1),
        block_dim=(LOWBIT_TPB, 1, 1),
    )


def identical_gemm_bf16w_widen_into(
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.uint16],
    mut work: LowbitWorkspace,
    m: Int,
    n: Int,
    k: Int,
    op: Int,
) raises:
    """The WIDEN plan, always: the right operand to float32, then the fp32
    profile's own dispatch with the vendor route closed."""
    _refuse_shape(m, n, k, op)
    var nb = n * k
    work.ensure(ctx, identical_gemm_workspace_max_floats(m, n, k), nb)
    bf16_widen(ctx, work.wide, b, nb)
    identical_gemm_into[False](ctx, c, a, work.wide, work.ws, m, n, k, op)


def identical_gemm_bf16w_into(
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.uint16],
    mut work: LowbitWorkspace,
    m: Int,
    n: Int,
    k: Int,
    op: Int,
) raises:
    """**THE ENTRY POINT of `mojolearn.identical.gemm.bf16f32.v1` with a
    float32 left operand.** `C[m x n] = op(A) . op(B)`, `B` stored as bf16
    bits in a `uint16` buffer, row-major and contiguous. Asynchronous: the
    caller owns `work` and waits."""
    if m * n <= BF16W_FUSED_MAX_CELLS:
        identical_gemm_bf16w_fused_into(ctx, c, a, b, m, n, k, op)
        return
    identical_gemm_bf16w_widen_into(ctx, c, a, b, work, m, n, k, op)


def identical_gemm_bf16w(
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.uint16],
    m: Int,
    n: Int,
    k: Int,
    op: Int,
) raises:
    """The synchronizing form: owns its workspace and waits before it
    returns, for the reason `identical_gemm` gives."""
    var work = LowbitWorkspace(ctx)
    identical_gemm_bf16w_into(ctx, c, a, b, work, m, n, k, op)
    step_count_sync()
    ctx.synchronize()
    _ = work^


def _refuse_shape(m: Int, n: Int, k: Int, op: Int) raises:
    if m <= 0 or n <= 0 or k <= 0:
        raise Error(
            "identical_gemm lowbit: m, n and k must all be positive, got m="
            + String(m) + " n=" + String(n) + " k=" + String(k)
        )
    if op != OP_NN and op != OP_NT and op != OP_TN:
        raise Error(
            "identical_gemm lowbit: op must be 0 (OP_NN), 1 (OP_NT) or 2"
            " (OP_TN), got " + String(op)
        )


# ===========================================================================
# int8i32.v1
# ===========================================================================


def quantize_rows_int8_kernel(
    q: MutPointer[Int8, MutAnyOrigin],
    e: MutPointer[Int32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    rows_in: Int32,
    cols_in: Int32,
):
    """One thread per row: absmax, exponent, codes. Contract L-3, L-4, in
    `quantize_rows_int8`'s order."""
    var rows = Int(rows_in)
    var cols = Int(cols_in)
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if r >= rows:
        return
    var best = Float32(0.0)
    for c in range(cols):
        var v = ftz(x.unsafe_load(r * cols + c))
        if v < Float32(0.0):
            v = -v
        if v > best:
            best = v
    var ex = int8_row_exponent(best)
    e.unsafe_store(r, Int32(ex))
    for c in range(cols):
        q.unsafe_store(r * cols + c, quantize_int8_value(x.unsafe_load(r * cols + c), ex))


def dequantize_rows_int8_kernel(
    y: MutPointer[Float32, MutAnyOrigin],
    q: MutPointer[Int8, MutAnyOrigin],
    e: MutPointer[Int32, MutAnyOrigin],
    rows_in: Int32,
    cols_in: Int32,
):
    """`q * 2^e`, exact, one element per thread: the float32 matrix an int8
    store stands for (the `int8w` block formats)."""
    var rows = Int(rows_in)
    var cols = Int(cols_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= rows * cols:
        return
    var r = i // cols
    y.unsafe_store(i, dequant_int8_pinned(Int32(q.unsafe_load(i)), Int(e.unsafe_load(r))))


def identical_gemm_int8_flat_kernel(
    c: MutPointer[Float32, MutAnyOrigin],
    qa: MutPointer[Int8, MutAnyOrigin],
    ea: MutPointer[Int32, MutAnyOrigin],
    qb: MutPointer[Int8, MutAnyOrigin],
    eb: MutPointer[Int32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    k_in: Int32,
):
    """One thread per cell, OP_NT, Int32 accumulation, `p` ascending, then
    the dequantization seam. Contract L-5 to L-7."""
    var m = Int(m_in)
    var n = Int(n_in)
    var k = Int(k_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= m * n:
        return
    var i = cell // n
    var j = cell - i * n
    var acc = Int32(0)
    var a_row = i * k
    var b_row = j * k
    for p in range(k):
        acc += Int32(qa.unsafe_load(a_row + p)) * Int32(qb.unsafe_load(b_row + p))
    var out = dequant_int8_pinned(acc, Int(ea.unsafe_load(i)) + Int(eb.unsafe_load(j)))
    comptime if LOWBIT_SABOTAGE:
        out = gemm_oracle_sabotage_value_flip(out)
    c.unsafe_store(cell, out)


def quantize_rows_int8_device(
    ctx: DeviceContext,
    mut q: DeviceBuffer[DType.int8],
    mut e: DeviceBuffer[DType.int32],
    mut x: DeviceBuffer[DType.float32],
    rows: Int,
    cols: Int,
) raises:
    if rows <= 0 or cols <= 0:
        raise Error("quantize_rows_int8: rows and cols must be positive")
    ctx.enqueue_function[quantize_rows_int8_kernel](
        q.unsafe_ptr(),
        e.unsafe_ptr(),
        x.unsafe_ptr(),
        Int32(rows),
        Int32(cols),
        grid_dim=((rows + LOWBIT_TPB - 1) // LOWBIT_TPB, 1, 1),
        block_dim=(LOWBIT_TPB, 1, 1),
    )


def dequantize_rows_int8_device(
    ctx: DeviceContext,
    mut y: DeviceBuffer[DType.float32],
    mut q: DeviceBuffer[DType.int8],
    mut e: DeviceBuffer[DType.int32],
    rows: Int,
    cols: Int,
) raises:
    if rows <= 0 or cols <= 0:
        raise Error("dequantize_rows_int8: rows and cols must be positive")
    var count = rows * cols
    ctx.enqueue_function[dequantize_rows_int8_kernel](
        y.unsafe_ptr(),
        q.unsafe_ptr(),
        e.unsafe_ptr(),
        Int32(rows),
        Int32(cols),
        grid_dim=((count + LOWBIT_TPB - 1) // LOWBIT_TPB, 1, 1),
        block_dim=(LOWBIT_TPB, 1, 1),
    )


def identical_gemm_int8_into(
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut qa: DeviceBuffer[DType.int8],
    mut ea: DeviceBuffer[DType.int32],
    mut qb: DeviceBuffer[DType.int8],
    mut eb: DeviceBuffer[DType.int32],
    m: Int,
    n: Int,
    k: Int,
) raises:
    """**THE ENTRY POINT of `mojolearn.identical.gemm.int8i32.v1`**, OP_NT:
    `C[m x n] = Q_a[m x k] . Q_b[n x k]^T` dequantized. Asynchronous.
    DEVIATION 2910: the MMA plan when the column's row says True, the shape
    is admitted and the build does not force the flat plan; the flat plan
    otherwise. Both are the profile (contract L-9)."""
    comptime if INT8_MMA_ENABLED:
        if int8_mma_admits(m, n, k):
            identical_gemm_int8_mma_into(ctx, c, qa, ea, qb, eb, m, n, k)
            return
    identical_gemm_int8_flat_into(ctx, c, qa, ea, qb, eb, m, n, k)


def identical_gemm_int8_flat_into(
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut qa: DeviceBuffer[DType.int8],
    mut ea: DeviceBuffer[DType.int32],
    mut qb: DeviceBuffer[DType.int8],
    mut eb: DeviceBuffer[DType.int32],
    m: Int,
    n: Int,
    k: Int,
) raises:
    """The FLAT plan, always: one thread per cell. The checks call this
    directly to compare it with the MMA plan; production goes through
    `identical_gemm_int8_into`. Asynchronous."""
    if m <= 0 or n <= 0 or k <= 0:
        raise Error(
            "identical_gemm_int8: m, n and k must all be positive, got m="
            + String(m) + " n=" + String(n) + " k=" + String(k)
        )
    if k > INT8_MAX_K:
        raise Error(
            "identical_gemm_int8: k must be at most " + String(INT8_MAX_K)
            + " so the Int32 accumulator cannot overflow (contract L-7), got "
            + String(k)
        )
    ctx.enqueue_function[identical_gemm_int8_flat_kernel](
        c.unsafe_ptr(),
        qa.unsafe_ptr(),
        ea.unsafe_ptr(),
        qb.unsafe_ptr(),
        eb.unsafe_ptr(),
        Int32(m),
        Int32(n),
        Int32(k),
        grid_dim=((m * n + LOWBIT_TPB - 1) // LOWBIT_TPB, 1, 1),
        block_dim=(LOWBIT_TPB, 1, 1),
    )


def identical_gemm_int8_from_f32(
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
) raises:
    """Quantize both float32 operands on the device by the profile's rule,
    then the product. Synchronizes before it returns."""
    step_count_device_alloc()
    var qa = ctx.enqueue_create_buffer[DType.int8](m * k)
    step_count_device_alloc()
    var ea = ctx.enqueue_create_buffer[DType.int32](m)
    step_count_device_alloc()
    var qb = ctx.enqueue_create_buffer[DType.int8](n * k)
    step_count_device_alloc()
    var eb = ctx.enqueue_create_buffer[DType.int32](n)
    quantize_rows_int8_device(ctx, qa, ea, a, m, k)
    quantize_rows_int8_device(ctx, qb, eb, b, n, k)
    identical_gemm_int8_into(ctx, c, qa, ea, qb, eb, m, n, k)
    step_count_sync()
    ctx.synchronize()
    _ = qa
    _ = ea
    _ = qb
    _ = eb
