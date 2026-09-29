# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The device realizations of the fifteen-bit profile.

    mojolearn.identical.gemm.int15i64.v1   `identical_gemm_int15_into`
                                           `identical_gemm_int15_planes_into`

Contract `gemm/IDENTICAL_LOWBIT_CONTRACT.md`, the section headed THE
FIFTEEN-BIT PROFILE; answers `gemm/host/gemm_int15_oracle.mojo`; seams
`checks/numerics_int15.mojo`; gates `gemm/checks/gemm_int15_check.mojo`.
Lane lane/lowbit-int15, 2026-09-29, DEVIATIONS 2965 to 2973.

THREE EXECUTION PLANS, ONE ANSWER (DEVIATION 2972, clause W-8)
---------------------------------------------------------------
  FLAT    `identical_gemm_int15_flat_kernel`: one thread per cell, the
          Int16 codes as stored, one Int32 product per step, the sum in
          Int64, `p` ascending. Every column. THE REFERENCE DEVICE PLAN:
          it is the oracle's `int15_dot_cell` on the device and it does not
          know the pieces exist.
  PIECES  `identical_gemm_int15_pieces_kernel`: one thread per cell, the
          two int8 planes of each operand, three Int32 accumulators
          (`HH`, `HL + LH`, `LL`), the recombination in Int64. Every
          column. It is what a matrix unit computes with the unit taken
          away, so a column that has no unit (Apple) still runs the
          construction of clauses W-3 to W-5 on its own integer hardware.
  MMA     `identical_gemm_int15_mma_kernel`: the vendor's integer matrix
          unit through `gemm_int8_mma.mojo`'s own fragment loads and step,
          FOUR products per k-tile of 32 into the same three Int32
          accumulators, zero-code padding, the same epilogue. Columns whose
          kernel-matrix row `lib_int8_matrix_unit_for` says True: NVIDIA
          and AMD.
All three are the profile because every piece product is exact, every
Int32 piece sum is exact under the bound on `k` (clause W-4) and therefore
order-free, and the recombination is exact in Int64: no tile shape and no
summation order can move a bit, so the choice is SCHEDULING and
`check_int15_plans_agree` requires the plans' bits to match on every shape.

THE STORED FORMS. An operand is its Int16 codes and one Int32 exponent per
row (what the quantizer writes and what the simulation's codes are), or its
two int8 PLANES and the same exponents (what the pieces and the matrix
unit read). `split_int15_device` makes the second from the first. A weight
is split once and kept; an activation is split per call.

A LAUNCH ON APPLE IS BOUNDED IN WORK (DEVIATION 2979, clause W-14)
-------------------------------------------------------------------
macOS aborts a Metal command buffer that holds the GPU for seconds
(`kIOGPUCommandBufferCallbackErrorImpactingInteractivity`). The cells
written before the abort stay, the rest keep what the buffer held, and the
wait returns as if the launch had finished: a product that is partly
another product, and nothing says so. Measured on the M2 Pro, 2026-09-29
(request 1790655769241): the pieces kernel at 512 x 4096 x 14336, one
launch of 3e10 multiply-accumulates, left cells unwritten in two runs of
three, at a different cell each time.
So on the Apple column the two one-thread-per-cell plans launch their cells
in SLICES of at most `INT15_APPLE_SLICE_MAC` multiply-accumulates and wait
between slices, and the dispatchers send a product above
`INT15_APPLE_UNIT_MIN_MAC` to the float-unit plan
(`gemm/checks/gemm_int15_apple.mojo`), whose launches take milliseconds.
Which cells a launch covers is SCHEDULING: a cell is computed by one thread
from its own two rows whatever the slice. The gate runs the row that failed.
The wait between slices makes these two launches SYNCHRONOUS on Apple when
they are sliced; on every other column they are one launch and asynchronous
as before.

THE CONVERSIONS, AND TWO SCHEDULES OF ONE QUANTIZER (DEVIATION 2974)
--------------------------------------------------------------------
  ROWS      `quantize_rows_int15_kernel` and `quantize_cols_int15_kernel`:
            one thread per row (per column, for an operand stored the
            other way). `quantize_rows_int15`'s own order. THE REFERENCE.
  PARALLEL  three launches on the one in-order context: the absmax of each
            row in chunks of `INT15_ABSMAX_CHUNK` values, one thread per
            chunk; the chunk maxima of a row reduced to its exponent, one
            thread per row; the codes (or the two planes directly), one
            thread per VALUE. No thread waits on another and nothing is
            shared inside a launch.
Both are the quantizer because the absmax is a MAXIMUM of magnitudes that a
NaN never enters: it is the same float under every grouping, so the
exponent is, and a code is a function of its own value and its row's
exponent. The choice is SCHEDULING and
`check_int15_device_conversions_match_host` requires the two schedules'
codes, exponents and planes to be the host's on every planted row. A row
that is one thread's work takes the row's whole length in time whatever the
device; lane/lowbit-units measured that the row quantizer takes longer than
the product it feeds.

THE SABOTAGE ARMS (DEVIATION 2973)
----------------------------------
`-D MOJOLEARN_LOWBIT_SABOTAGE=1`, the low-bit family's device arm, flips the
value of every cell all three kernels store, so a build carrying it must
fail every oracle gate of `gemm_int15_check.mojo`.
`-D MOJOLEARN_INT15_PIECE_SABOTAGE=1` is the DEFECT ARM of clause W-3: the
split writes -127 where the high piece is -128, which is what a split
written for a symmetric int8 range would do. It changes no code above
-16257: a handful of the codes of a random fixture, and EVERY code of two
of the planted worst cases. It does not reach the FLAT plan, which reads
no plane.
`-D MOJOLEARN_INT15_QUANT_SABOTAGE=1` is the DEFECT ARM of the parallel
quantizer: the last chunk of every row is left out of the absmax, the
mistake a chunked loop makes at a ragged tail. It moves the exponent of a
row whose largest value lies in that tail and of no other row.
"""

from std.gpu import WARP_SIZE, block_dim, block_idx, lane_id, thread_idx
from std.sys import is_defined, llvm_intrinsic
from std.sys.info import is_amd_gpu, is_nvidia_gpu
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.kernel_matrix import (
    COLUMN_APPLE,
    TARGET_COLUMN,
    column_name,
    lib_int8_matrix_unit_for,
)
from checks.numerics import ftz
from checks.numerics_int15 import (
    dequant_int15_code,
    dequant_int15_pinned,
    int15_piece_hi,
    int15_piece_lo,
    int15_recombine,
    int15_row_exponent,
    quantize_int15_value,
)
from gemm.checks.gemm_identical import step_count_device_alloc, step_count_sync
from gemm.checks.gemm_int8_mma import (
    INT8_MMA_BLOCK_TILE_M,
    INT8_MMA_BLOCK_TILE_N,
    INT8_MMA_K_TILE,
    INT8_MMA_TILE,
    INT8_MMA_TPB,
    INT8_MMA_UNSTATED_LOADS,
    INT8_MMA_WARPS_N,
    _imma_m16n8k32,
    _pack4,
    _pack8,
    mma_operands_aligned,
)
from gemm.checks.gemm_int15_apple import (
    INT15_APPLE_FORM_FOUR,
    INT15_APPLE_FORM_TWO,
    INT15_APPLE_ROW_MAX_M,
    identical_gemm_int15_apple_into,
)
from gemm.checks.gemm_int15_epilogue import INT15_EXPONENT_SABOTAGE
from gemm.host.gemm_int15_oracle import INT15_MAX_K
from gemm.host.gemm_oracle import gemm_oracle_sabotage_value_flip

#: DEVIATION 2973, the value arm: the low-bit family's own define.
comptime INT15_SABOTAGE = is_defined["MOJOLEARN_LOWBIT_SABOTAGE"]()

#: DEVIATION 2973, the defect arm of the split (clause W-3).
comptime INT15_PIECE_SABOTAGE = is_defined["MOJOLEARN_INT15_PIECE_SABOTAGE"]()

#: DEVIATION 2973, the defect arm of the parallel quantizer.
comptime INT15_QUANT_SABOTAGE = is_defined["MOJOLEARN_INT15_QUANT_SABOTAGE"]()

#: DEVIATION 2974: values per chunk of the parallel absmax. SCHEDULING: a
#: maximum is the same under every grouping.
comptime INT15_ABSMAX_CHUNK = 256

#: DEVIATION 2972: `-D MOJOLEARN_INT15_FORCE_FLAT=1` keeps the two
#: one-thread-per-cell plans on every column, the unit's columns included.
#: SCHEDULING; cannot move a bit (clause W-8).
comptime INT15_FORCE_FLAT = is_defined["MOJOLEARN_INT15_FORCE_FLAT"]()

#: Whether this build's dispatchers may pick the MMA plan at all. The row
#: is `int8i32.v1`'s: the unit is the same unit.
comptime INT15_MMA_ENABLED = lib_int8_matrix_unit_for[TARGET_COLUMN]() and not INT15_FORCE_FLAT

#: DEVIATION 2979: whether this build bounds the work of a launch of the
#: one-thread-per-cell plans (the Apple column), and the bound: 2^30
#: multiply-accumulates, a fraction of a second on the slowest Apple box
#: this tree runs on. `-D MOJOLEARN_INT15_ONE_LAUNCH=1` is the arm that
#: launches every cell at once, as the kernels did until 2026-09-29.
comptime INT15_BOUNDED_LAUNCHES = TARGET_COLUMN == COLUMN_APPLE and not is_defined["MOJOLEARN_INT15_ONE_LAUNCH"]()
comptime INT15_APPLE_SLICE_MAC = 1073741824

#: DEVIATION 2979: on Apple the dispatchers send a product above this many
#: multiply-accumulates to the float-unit plan (2^28; below it the
#: one-thread-per-cell kernels took less time on the M3 Ultra, run 2).
#: `-D MOJOLEARN_INT15_NO_APPLE_UNIT=1` keeps the dispatchers off the unit.
#: SCHEDULING.
comptime INT15_APPLE_UNIT_ENABLED = (
    TARGET_COLUMN == COLUMN_APPLE
    and not INT15_FORCE_FLAT
    and not is_defined["MOJOLEARN_INT15_NO_APPLE_UNIT"]()
)
comptime INT15_APPLE_UNIT_MIN_MAC = 268435456

#: Threads per block for every one-thread-per-element kernel in this file.
#: SCHEDULING; no value crosses a thread boundary in any of them.
comptime INT15_TPB = 256


def int15_plan_dispatch_name() -> String:
    """What the two entry points run on this build, for the gate banner."""
    comptime if INT15_FORCE_FLAT:
        return String("flat and pieces (MOJOLEARN_INT15_FORCE_FLAT)")
    elif lib_int8_matrix_unit_for[TARGET_COLUMN]():
        return String("mma (lib_int8_matrix_unit_for)")
    elif INT15_APPLE_UNIT_ENABLED:
        return String("flat and pieces below 2^28 multiply-accumulates, the float matrix unit above (apple)")
    else:
        return String("flat and pieces (no int8 matrix unit on this column)")


def int15_slice_cells(m: Int, n: Int, k: Int) -> Int:
    """Cells per launch of a one-thread-per-cell plan: every cell where
    launches are not bounded, else whole blocks of at most
    `INT15_APPLE_SLICE_MAC` multiply-accumulates (one block at least)."""
    comptime if INT15_BOUNDED_LAUNCHES:
        var cells = ((INT15_APPLE_SLICE_MAC // k) // INT15_TPB) * INT15_TPB
        if cells < INT15_TPB:
            cells = INT15_TPB
        return cells
    else:
        return m * n


def int15_apple_takes_unit(m: Int, n: Int, k: Int) -> Bool:
    """Whether the Apple dispatchers send the shape to the float-unit
    plan. Reads the shape and may: every plan returns the same bits."""
    return m * n * k > INT15_APPLE_UNIT_MIN_MAC


def int15_apple_form(m: Int) -> Int:
    """The form of the float-unit plan the dispatchers take: TWO for the
    decode rows, FOUR above them (M3 Ultra, run 2: at the wide decode rows
    TWO took less time, at the training rows FOUR)."""
    if m <= INT15_APPLE_ROW_MAX_M:
        return INT15_APPLE_FORM_TWO
    return INT15_APPLE_FORM_FOUR


def int15_sabotage_name() -> String:
    comptime if INT15_SABOTAGE:
        return String("LOWBIT_VALUE_FLIP")
    elif INT15_PIECE_SABOTAGE:
        return String("INT15_PIECE_HI_CLAMP")
    elif INT15_QUANT_SABOTAGE:
        return String("INT15_QUANT_TAIL_DROPPED")
    else:
        return String("none")


def int15_admits(m: Int, n: Int, k: Int) -> Bool:
    """Every shape the profile accepts: positive extents and
    `k <= INT15_MAX_K`. One sentence for all three plans, so no plan
    refuses a shape another serves."""
    return m > 0 and n > 0 and k > 0 and k <= INT15_MAX_K


def _refuse(m: Int, n: Int, k: Int, who: String) raises:
    if m <= 0 or n <= 0 or k <= 0:
        raise Error(
            who + ": m, n and k must all be positive, got m=" + String(m)
            + " n=" + String(n) + " k=" + String(k)
        )
    if k > INT15_MAX_K:
        raise Error(
            who + ": k must be at most " + String(INT15_MAX_K)
            + " so no Int32 piece sum can overflow (contract W-4), got "
            + String(k)
        )


# ===========================================================================
# the conversions: quantize, dequantize, split
# ===========================================================================


def quantize_rows_int15_kernel(
    q: MutPointer[Int16, MutAnyOrigin],
    e: MutPointer[Int32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    rows_in: Int32,
    cols_in: Int32,
):
    """One thread per row: absmax, exponent, codes. Clauses W-1 and W-2, in
    `quantize_rows_int15`'s order."""
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
    var ex = int15_row_exponent(best)
    e.unsafe_store(r, Int32(ex))
    for c in range(cols):
        q.unsafe_store(r * cols + c, quantize_int15_value(x.unsafe_load(r * cols + c), ex))


def quantize_cols_int15_kernel(
    q: MutPointer[Int16, MutAnyOrigin],
    e: MutPointer[Int32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    rows_in: Int32,
    cols_in: Int32,
):
    """THE TRANSPOSING QUANTIZER. `x` is `rows x cols` row-major; one
    thread per COLUMN of it: the column's absmax, its exponent, and its
    codes written as row `c` of the `cols x rows` transpose. The values a
    thread reads are one column's and the rule is W-1 and W-2, so the codes
    are `quantize_rows_int15`'s on the transposed matrix."""
    var rows = Int(rows_in)
    var cols = Int(cols_in)
    var c = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if c >= cols:
        return
    var best = Float32(0.0)
    for r in range(rows):
        var v = ftz(x.unsafe_load(r * cols + c))
        if v < Float32(0.0):
            v = -v
        if v > best:
            best = v
    var ex = int15_row_exponent(best)
    e.unsafe_store(c, Int32(ex))
    for r in range(rows):
        q.unsafe_store(c * rows + r, quantize_int15_value(x.unsafe_load(r * cols + c), ex))


def dequantize_rows_int15_kernel(
    y: MutPointer[Float32, MutAnyOrigin],
    q: MutPointer[Int16, MutAnyOrigin],
    e: MutPointer[Int32, MutAnyOrigin],
    rows_in: Int32,
    cols_in: Int32,
):
    """`q * 2^e`, one element per thread: the float32 matrix a fifteen-bit
    store stands for."""
    var rows = Int(rows_in)
    var cols = Int(cols_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= rows * cols:
        return
    var r = i // cols
    y.unsafe_store(i, dequant_int15_code(q.unsafe_load(i), Int(e.unsafe_load(r))))


def dequantize_planes_int15_kernel(
    y: MutPointer[Float32, MutAnyOrigin],
    hi: MutPointer[Int8, MutAnyOrigin],
    lo: MutPointer[Int8, MutAnyOrigin],
    e: MutPointer[Int32, MutAnyOrigin],
    rows_in: Int32,
    cols_in: Int32,
):
    """`(hi * 128 + lo) * 2^e`, one element per thread: the float32 matrix
    a store of planes stands for. The code is rebuilt by clause W-5's own
    recombination with no high-high and no low-low term."""
    var rows = Int(rows_in)
    var cols = Int(cols_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= rows * cols:
        return
    var r = i // cols
    var code = int15_recombine(Int32(0), Int32(hi.unsafe_load(i)), Int32(lo.unsafe_load(i)))
    y.unsafe_store(i, dequant_int15_pinned(code, Int(e.unsafe_load(r))))


def split_int15_kernel(
    hi: MutPointer[Int8, MutAnyOrigin],
    lo: MutPointer[Int8, MutAnyOrigin],
    q: MutPointer[Int16, MutAnyOrigin],
    n_in: Int32,
):
    """Clause W-3, one code per thread: a mask and an arithmetic shift."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_in):
        return
    var c = q.unsafe_load(i)
    var h = int15_piece_hi(c)
    comptime if INT15_PIECE_SABOTAGE:
        # THE DEFECT ARM: a split that believes an int8 piece is symmetric.
        if h == Int8(-128):
            h = Int8(-127)
    hi.unsafe_store(i, h)
    lo.unsafe_store(i, int15_piece_lo(c))


# ---------------------------------------------------------------------------
# the PARALLEL schedule of the quantizer (DEVIATION 2974)
# ---------------------------------------------------------------------------
# The operand is addressed by two strides so one set of kernels serves an
# operand stored either way: value `(r, c)` of the `rows x cols` matrix
# being quantized is `x[r * s_row + c * s_col]`, with `(s_row, s_col)`
# `(cols, 1)` when `x` is that matrix row-major and `(1, rows)` when `x` is
# its transpose row-major (`cols x rows`).


def int15_absmax_chunk_kernel(
    part: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    rows_in: Int32,
    cols_in: Int32,
    chunks_in: Int32,
    s_row_in: Int32,
    s_col_in: Int32,
):
    """One thread per chunk of a row: the largest magnitude of its values
    after the flush, `row_absmax`'s own comparison, into `part[r * chunks +
    chunk]`."""
    var rows = Int(rows_in)
    var cols = Int(cols_in)
    var chunks = Int(chunks_in)
    var s_row = Int(s_row_in)
    var s_col = Int(s_col_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= rows * chunks:
        return
    var r = t // chunks
    var ch = t - r * chunks
    var c0 = ch * INT15_ABSMAX_CHUNK
    var c1 = c0 + INT15_ABSMAX_CHUNK
    if c1 > cols:
        c1 = cols
    comptime if INT15_QUANT_SABOTAGE:
        # THE DEFECT ARM: the ragged tail of the row is never read.
        if ch == chunks - 1:
            c1 = c0
    var best = Float32(0.0)
    var base = r * s_row
    for c in range(c0, c1):
        var v = ftz(x.unsafe_load(base + c * s_col))
        if v < Float32(0.0):
            v = -v
        if v > best:
            best = v
    part.unsafe_store(t, best)


def int15_exponent_kernel(
    e: MutPointer[Int32, MutAnyOrigin],
    part: MutPointer[Float32, MutAnyOrigin],
    rows_in: Int32,
    chunks_in: Int32,
):
    """One thread per row: the largest of the row's chunk maxima, then
    clause W-1."""
    var rows = Int(rows_in)
    var chunks = Int(chunks_in)
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if r >= rows:
        return
    var best = Float32(0.0)
    for ch in range(chunks):
        var v = part.unsafe_load(r * chunks + ch)
        if v > best:
            best = v
    e.unsafe_store(r, Int32(int15_row_exponent(best)))


def int15_codes_kernel(
    q: MutPointer[Int16, MutAnyOrigin],
    e: MutPointer[Int32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    rows_in: Int32,
    cols_in: Int32,
    s_row_in: Int32,
    s_col_in: Int32,
):
    """One thread per value: clause W-2 under the row's exponent."""
    var rows = Int(rows_in)
    var cols = Int(cols_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= rows * cols:
        return
    var r = i // cols
    var c = i - r * cols
    var v = x.unsafe_load(r * Int(s_row_in) + c * Int(s_col_in))
    q.unsafe_store(i, quantize_int15_value(v, Int(e.unsafe_load(r))))


def int15_planes_kernel(
    hi: MutPointer[Int8, MutAnyOrigin],
    lo: MutPointer[Int8, MutAnyOrigin],
    e: MutPointer[Int32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    rows_in: Int32,
    cols_in: Int32,
    s_row_in: Int32,
    s_col_in: Int32,
):
    """One thread per value: clause W-2, then clause W-3, the code never
    stored. What a call pays for an activation whose product reads planes."""
    var rows = Int(rows_in)
    var cols = Int(cols_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= rows * cols:
        return
    var r = i // cols
    var c = i - r * cols
    var v = x.unsafe_load(r * Int(s_row_in) + c * Int(s_col_in))
    var code = quantize_int15_value(v, Int(e.unsafe_load(r)))
    var h = int15_piece_hi(code)
    comptime if INT15_PIECE_SABOTAGE:
        if h == Int8(-128):
            h = Int8(-127)
    hi.unsafe_store(i, h)
    lo.unsafe_store(i, int15_piece_lo(code))


struct Int15QuantWorkspace(Movable):
    """The chunk maxima of the parallel quantizer, on ONE in-order
    context. Grown on demand; growth drains the context first."""

    var part: DeviceBuffer[DType.float32]

    def __init__(out self, ctx: DeviceContext) raises:
        step_count_device_alloc()
        self.part = ctx.enqueue_create_buffer[DType.float32](1)

    def ensure(mut self, ctx: DeviceContext, floats: Int) raises:
        if floats > len(self.part):
            step_count_sync()
            ctx.synchronize()
            step_count_device_alloc()
            self.part = ctx.enqueue_create_buffer[DType.float32](floats)


def int15_quant_chunks(cols: Int) -> Int:
    return (cols + INT15_ABSMAX_CHUNK - 1) // INT15_ABSMAX_CHUNK


def _parallel_exponents(
    ctx: DeviceContext,
    mut e: DeviceBuffer[DType.int32],
    mut x: DeviceBuffer[DType.float32],
    mut work: Int15QuantWorkspace,
    rows: Int,
    cols: Int,
    s_row: Int,
    s_col: Int,
) raises:
    """The first two launches: chunk maxima, then one exponent per row."""
    var chunks = int15_quant_chunks(cols)
    work.ensure(ctx, rows * chunks)
    ctx.enqueue_function[int15_absmax_chunk_kernel](
        work.part.unsafe_ptr(),
        x.unsafe_ptr(),
        Int32(rows),
        Int32(cols),
        Int32(chunks),
        Int32(s_row),
        Int32(s_col),
        grid_dim=((rows * chunks + INT15_TPB - 1) // INT15_TPB, 1, 1),
        block_dim=(INT15_TPB, 1, 1),
    )
    ctx.enqueue_function[int15_exponent_kernel](
        e.unsafe_ptr(),
        work.part.unsafe_ptr(),
        Int32(rows),
        Int32(chunks),
        grid_dim=((rows + INT15_TPB - 1) // INT15_TPB, 1, 1),
        block_dim=(INT15_TPB, 1, 1),
    )


def quantize_int15_parallel_device(
    ctx: DeviceContext,
    mut q: DeviceBuffer[DType.int16],
    mut e: DeviceBuffer[DType.int32],
    mut x: DeviceBuffer[DType.float32],
    mut work: Int15QuantWorkspace,
    rows: Int,
    cols: Int,
    transposed: Bool,
) raises:
    """The PARALLEL schedule, to codes: `rows x cols` codes and `rows`
    exponents. `x` is that matrix row-major, or with `transposed` its
    transpose row-major (`cols x rows`). Asynchronous."""
    if rows <= 0 or cols <= 0:
        raise Error("quantize_int15_parallel: rows and cols must be positive")
    var s_row = cols
    var s_col = 1
    if transposed:
        s_row = 1
        s_col = rows
    _parallel_exponents(ctx, e, x, work, rows, cols, s_row, s_col)
    ctx.enqueue_function[int15_codes_kernel](
        q.unsafe_ptr(),
        e.unsafe_ptr(),
        x.unsafe_ptr(),
        Int32(rows),
        Int32(cols),
        Int32(s_row),
        Int32(s_col),
        grid_dim=((rows * cols + INT15_TPB - 1) // INT15_TPB, 1, 1),
        block_dim=(INT15_TPB, 1, 1),
    )


def quantize_planes_int15_parallel_device(
    ctx: DeviceContext,
    mut hi: DeviceBuffer[DType.int8],
    mut lo: DeviceBuffer[DType.int8],
    mut e: DeviceBuffer[DType.int32],
    mut x: DeviceBuffer[DType.float32],
    mut work: Int15QuantWorkspace,
    rows: Int,
    cols: Int,
    transposed: Bool,
) raises:
    """The PARALLEL schedule, straight to planes. Asynchronous."""
    if rows <= 0 or cols <= 0:
        raise Error("quantize_planes_int15_parallel: rows and cols must be positive")
    var s_row = cols
    var s_col = 1
    if transposed:
        s_row = 1
        s_col = rows
    _parallel_exponents(ctx, e, x, work, rows, cols, s_row, s_col)
    ctx.enqueue_function[int15_planes_kernel](
        hi.unsafe_ptr(),
        lo.unsafe_ptr(),
        e.unsafe_ptr(),
        x.unsafe_ptr(),
        Int32(rows),
        Int32(cols),
        Int32(s_row),
        Int32(s_col),
        grid_dim=((rows * cols + INT15_TPB - 1) // INT15_TPB, 1, 1),
        block_dim=(INT15_TPB, 1, 1),
    )


def quantize_rows_int15_device(
    ctx: DeviceContext,
    mut q: DeviceBuffer[DType.int16],
    mut e: DeviceBuffer[DType.int32],
    mut x: DeviceBuffer[DType.float32],
    rows: Int,
    cols: Int,
) raises:
    if rows <= 0 or cols <= 0:
        raise Error("quantize_rows_int15: rows and cols must be positive")
    ctx.enqueue_function[quantize_rows_int15_kernel](
        q.unsafe_ptr(),
        e.unsafe_ptr(),
        x.unsafe_ptr(),
        Int32(rows),
        Int32(cols),
        grid_dim=((rows + INT15_TPB - 1) // INT15_TPB, 1, 1),
        block_dim=(INT15_TPB, 1, 1),
    )


def quantize_cols_int15_device(
    ctx: DeviceContext,
    mut q: DeviceBuffer[DType.int16],
    mut e: DeviceBuffer[DType.int32],
    mut x: DeviceBuffer[DType.float32],
    rows: Int,
    cols: Int,
) raises:
    """`x` is `rows x cols`; `q` receives `cols x rows` codes and `e` one
    exponent per column of `x`."""
    if rows <= 0 or cols <= 0:
        raise Error("quantize_cols_int15: rows and cols must be positive")
    ctx.enqueue_function[quantize_cols_int15_kernel](
        q.unsafe_ptr(),
        e.unsafe_ptr(),
        x.unsafe_ptr(),
        Int32(rows),
        Int32(cols),
        grid_dim=((cols + INT15_TPB - 1) // INT15_TPB, 1, 1),
        block_dim=(INT15_TPB, 1, 1),
    )


def dequantize_rows_int15_device(
    ctx: DeviceContext,
    mut y: DeviceBuffer[DType.float32],
    mut q: DeviceBuffer[DType.int16],
    mut e: DeviceBuffer[DType.int32],
    rows: Int,
    cols: Int,
) raises:
    if rows <= 0 or cols <= 0:
        raise Error("dequantize_rows_int15: rows and cols must be positive")
    var count = rows * cols
    ctx.enqueue_function[dequantize_rows_int15_kernel](
        y.unsafe_ptr(),
        q.unsafe_ptr(),
        e.unsafe_ptr(),
        Int32(rows),
        Int32(cols),
        grid_dim=((count + INT15_TPB - 1) // INT15_TPB, 1, 1),
        block_dim=(INT15_TPB, 1, 1),
    )


def dequantize_planes_int15_device(
    ctx: DeviceContext,
    mut y: DeviceBuffer[DType.float32],
    mut hi: DeviceBuffer[DType.int8],
    mut lo: DeviceBuffer[DType.int8],
    mut e: DeviceBuffer[DType.int32],
    rows: Int,
    cols: Int,
) raises:
    if rows <= 0 or cols <= 0:
        raise Error("dequantize_planes_int15: rows and cols must be positive")
    var count = rows * cols
    ctx.enqueue_function[dequantize_planes_int15_kernel](
        y.unsafe_ptr(),
        hi.unsafe_ptr(),
        lo.unsafe_ptr(),
        e.unsafe_ptr(),
        Int32(rows),
        Int32(cols),
        grid_dim=((count + INT15_TPB - 1) // INT15_TPB, 1, 1),
        block_dim=(INT15_TPB, 1, 1),
    )


def split_int15_device(
    ctx: DeviceContext,
    mut hi: DeviceBuffer[DType.int8],
    mut lo: DeviceBuffer[DType.int8],
    mut q: DeviceBuffer[DType.int16],
    count: Int,
) raises:
    if count <= 0:
        raise Error("split_int15: count must be positive")
    ctx.enqueue_function[split_int15_kernel](
        hi.unsafe_ptr(),
        lo.unsafe_ptr(),
        q.unsafe_ptr(),
        Int32(count),
        grid_dim=((count + INT15_TPB - 1) // INT15_TPB, 1, 1),
        block_dim=(INT15_TPB, 1, 1),
    )


# ===========================================================================
# the shared epilogue
# ===========================================================================


@always_inline
def _store_cell15(
    c: MutPointer[Float32, MutAnyOrigin],
    ea: MutPointer[Int32, MutAnyOrigin],
    eb: MutPointer[Int32, MutAnyOrigin],
    acc: Int64,
    i: Int,
    j: Int,
    m: Int,
    n: Int,
):
    """The dequantization seam (W-6, W-7) and the store, masked to the
    output. One spelling for all three plans."""
    if i >= m or j >= n:
        return
    var jb = j
    comptime if INT15_EXPONENT_SABOTAGE:
        # THE DEFECT ARM (-D MOJOLEARN_INT15_EXPONENT_SABOTAGE=1): the
        # column exponent read at the row index.
        jb = i % n
    var out = dequant_int15_pinned(acc, Int(ea.unsafe_load(i)) + Int(eb.unsafe_load(jb)))
    comptime if INT15_SABOTAGE:
        out = gemm_oracle_sabotage_value_flip(out)
    c.unsafe_store(i * n + j, out)


# ===========================================================================
# FLAT: the codes, Int64
# ===========================================================================


def identical_gemm_int15_flat_kernel(
    c: MutPointer[Float32, MutAnyOrigin],
    qa: MutPointer[Int16, MutAnyOrigin],
    ea: MutPointer[Int32, MutAnyOrigin],
    qb: MutPointer[Int16, MutAnyOrigin],
    eb: MutPointer[Int32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    k_in: Int32,
    cell0_in: Int32,
    cell1_in: Int32,
):
    """One thread per cell, OP_NT, one Int32 product per step, the sum in
    Int64, `p` ascending, then the dequantization seam. The accumulator
    sees additions only: no 64-bit multiply and no 64-bit division is asked
    of any backend. The launch covers the cells `[cell0, cell1)`."""
    var m = Int(m_in)
    var n = Int(n_in)
    var k = Int(k_in)
    var cell = Int(cell0_in) + Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= Int(cell1_in):
        return
    var i = cell // n
    var j = cell - i * n
    var acc = Int64(0)
    var a_row = i * k
    var b_row = j * k
    for p in range(k):
        acc += Int64(Int32(qa.unsafe_load(a_row + p)) * Int32(qb.unsafe_load(b_row + p)))
    _store_cell15(c, ea, eb, acc, i, j, m, n)


# ===========================================================================
# PIECES: the planes, three Int32 accumulators, no unit
# ===========================================================================


def identical_gemm_int15_pieces_kernel(
    c: MutPointer[Float32, MutAnyOrigin],
    ah: MutPointer[Int8, MutAnyOrigin],
    al: MutPointer[Int8, MutAnyOrigin],
    ea: MutPointer[Int32, MutAnyOrigin],
    bh: MutPointer[Int8, MutAnyOrigin],
    bl: MutPointer[Int8, MutAnyOrigin],
    eb: MutPointer[Int32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    k_in: Int32,
    cell0_in: Int32,
    cell1_in: Int32,
):
    """One thread per cell, OP_NT, the four piece products of each step
    into `HH`, `HL + LH` and `LL` in Int32, then clause W-5 and the seam.
    The launch covers the cells `[cell0, cell1)`."""
    var m = Int(m_in)
    var n = Int(n_in)
    var k = Int(k_in)
    var cell = Int(cell0_in) + Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= Int(cell1_in):
        return
    var i = cell // n
    var j = cell - i * n
    var hh = Int32(0)
    var mid = Int32(0)
    var ll = Int32(0)
    var a_row = i * k
    var b_row = j * k
    for p in range(k):
        var a_hi = Int32(ah.unsafe_load(a_row + p))
        var a_lo = Int32(al.unsafe_load(a_row + p))
        var b_hi = Int32(bh.unsafe_load(b_row + p))
        var b_lo = Int32(bl.unsafe_load(b_row + p))
        hh += a_hi * b_hi
        mid += a_hi * b_lo
        mid += a_lo * b_hi
        ll += a_lo * b_lo
    _store_cell15(c, ea, eb, int15_recombine(hh, mid, ll), i, j, m, n)


# ===========================================================================
# MMA: the planes on the integer matrix units, four products per k-tile
# ===========================================================================


@always_inline
def _mfma_i32_16x16x32_i8(
    a: Int64, b: Int64, acc: SIMD[DType.int32, 4]
) -> SIMD[DType.int32, 4]:
    """One `v_mfma_i32_16x16x32_i8`, the instruction `gemm_int8_mma.mojo`
    issues, with `cbsz`, `abid` and `blgp` 0."""
    return llvm_intrinsic[
        "llvm.amdgcn.mfma.i32.16x16x32.i8",
        SIMD[DType.int32, 4],
        has_side_effect=False,
    ](a, b, acc, Int32(0), Int32(0), Int32(0))


@always_inline
def _nvidia_warp_tile15(
    c: MutPointer[Float32, MutAnyOrigin],
    ah: MutPointer[Int8, MutAnyOrigin],
    al: MutPointer[Int8, MutAnyOrigin],
    ea: MutPointer[Int32, MutAnyOrigin],
    bh: MutPointer[Int8, MutAnyOrigin],
    bl: MutPointer[Int8, MutAnyOrigin],
    eb: MutPointer[Int32, MutAnyOrigin],
    lane: Int,
    row0: Int,
    col0: Int,
    m: Int,
    n: Int,
    k: Int,
    aligned: Bool,
):
    """`gemm_int8_mma.mojo::_nvidia_warp_tile` with two planes on each side:
    the same fragment layout, the same two n8 halves, eight IMMA steps per
    k-tile in place of two. `HL` and `LH` chain through one accumulator."""
    var g = lane >> 2
    var t = lane & 3
    var ra0 = row0 + g
    var ra1 = row0 + g + 8
    var cb0 = col0 + g
    var cb1 = col0 + 8 + g
    var hh0 = SIMD[DType.int32, 4](0)
    var hh1 = SIMD[DType.int32, 4](0)
    var mid0 = SIMD[DType.int32, 4](0)
    var mid1 = SIMD[DType.int32, 4](0)
    var ll0 = SIMD[DType.int32, 4](0)
    var ll1 = SIMD[DType.int32, 4](0)
    for kt in range(0, k, INT8_MMA_K_TILE):
        var ka = kt + t * 4
        var ah0 = _pack4(ah, ra0, ka, m, k, aligned)
        var ah1 = _pack4(ah, ra1, ka, m, k, aligned)
        var ah2 = _pack4(ah, ra0, ka + 16, m, k, aligned)
        var ah3 = _pack4(ah, ra1, ka + 16, m, k, aligned)
        var al0 = _pack4(al, ra0, ka, m, k, aligned)
        var al1 = _pack4(al, ra1, ka, m, k, aligned)
        var al2 = _pack4(al, ra0, ka + 16, m, k, aligned)
        var al3 = _pack4(al, ra1, ka + 16, m, k, aligned)
        var bh0 = _pack4(bh, cb0, ka, n, k, aligned)
        var bh1 = _pack4(bh, cb0, ka + 16, n, k, aligned)
        var bl0 = _pack4(bl, cb0, ka, n, k, aligned)
        var bl1 = _pack4(bl, cb0, ka + 16, n, k, aligned)
        hh0 = _imma_m16n8k32(ah0, ah1, ah2, ah3, bh0, bh1, hh0)
        mid0 = _imma_m16n8k32(ah0, ah1, ah2, ah3, bl0, bl1, mid0)
        mid0 = _imma_m16n8k32(al0, al1, al2, al3, bh0, bh1, mid0)
        ll0 = _imma_m16n8k32(al0, al1, al2, al3, bl0, bl1, ll0)
        var bh2 = _pack4(bh, cb1, ka, n, k, aligned)
        var bh3 = _pack4(bh, cb1, ka + 16, n, k, aligned)
        var bl2 = _pack4(bl, cb1, ka, n, k, aligned)
        var bl3 = _pack4(bl, cb1, ka + 16, n, k, aligned)
        hh1 = _imma_m16n8k32(ah0, ah1, ah2, ah3, bh2, bh3, hh1)
        mid1 = _imma_m16n8k32(ah0, ah1, ah2, ah3, bl2, bl3, mid1)
        mid1 = _imma_m16n8k32(al0, al1, al2, al3, bh2, bh3, mid1)
        ll1 = _imma_m16n8k32(al0, al1, al2, al3, bl2, bl3, ll1)
    var jc = col0 + t * 2
    _store_cell15(c, ea, eb, int15_recombine(hh0[0], mid0[0], ll0[0]), ra0, jc, m, n)
    _store_cell15(c, ea, eb, int15_recombine(hh0[1], mid0[1], ll0[1]), ra0, jc + 1, m, n)
    _store_cell15(c, ea, eb, int15_recombine(hh0[2], mid0[2], ll0[2]), ra1, jc, m, n)
    _store_cell15(c, ea, eb, int15_recombine(hh0[3], mid0[3], ll0[3]), ra1, jc + 1, m, n)
    _store_cell15(c, ea, eb, int15_recombine(hh1[0], mid1[0], ll1[0]), ra0, jc + 8, m, n)
    _store_cell15(c, ea, eb, int15_recombine(hh1[1], mid1[1], ll1[1]), ra0, jc + 9, m, n)
    _store_cell15(c, ea, eb, int15_recombine(hh1[2], mid1[2], ll1[2]), ra1, jc + 8, m, n)
    _store_cell15(c, ea, eb, int15_recombine(hh1[3], mid1[3], ll1[3]), ra1, jc + 9, m, n)


@always_inline
def _amd_warp_tile15(
    c: MutPointer[Float32, MutAnyOrigin],
    ah: MutPointer[Int8, MutAnyOrigin],
    al: MutPointer[Int8, MutAnyOrigin],
    ea: MutPointer[Int32, MutAnyOrigin],
    bh: MutPointer[Int8, MutAnyOrigin],
    bl: MutPointer[Int8, MutAnyOrigin],
    eb: MutPointer[Int32, MutAnyOrigin],
    lane: Int,
    row0: Int,
    col0: Int,
    m: Int,
    n: Int,
    k: Int,
    aligned: Bool,
):
    """`gemm_int8_mma.mojo::_amd_warp_tile` with two planes on each side:
    the same operand layout, four MFMA steps per k-tile in place of one."""
    var i16 = lane & 15
    var kq = (lane >> 4) * 8
    var hh = SIMD[DType.int32, 4](0)
    var mid = SIMD[DType.int32, 4](0)
    var ll = SIMD[DType.int32, 4](0)
    for kt in range(0, k, INT8_MMA_K_TILE):
        var a_hi = _pack8(ah, row0 + i16, kt + kq, m, k, aligned)
        var a_lo = _pack8(al, row0 + i16, kt + kq, m, k, aligned)
        var b_hi = _pack8(bh, col0 + i16, kt + kq, n, k, aligned)
        var b_lo = _pack8(bl, col0 + i16, kt + kq, n, k, aligned)
        hh = _mfma_i32_16x16x32_i8(a_hi, b_hi, hh)
        mid = _mfma_i32_16x16x32_i8(a_hi, b_lo, mid)
        mid = _mfma_i32_16x16x32_i8(a_lo, b_hi, mid)
        ll = _mfma_i32_16x16x32_i8(a_lo, b_lo, ll)
    var j = col0 + i16
    var ir = row0 + (lane >> 4) * 4
    _store_cell15(c, ea, eb, int15_recombine(hh[0], mid[0], ll[0]), ir, j, m, n)
    _store_cell15(c, ea, eb, int15_recombine(hh[1], mid[1], ll[1]), ir + 1, j, m, n)
    _store_cell15(c, ea, eb, int15_recombine(hh[2], mid[2], ll[2]), ir + 2, j, m, n)
    _store_cell15(c, ea, eb, int15_recombine(hh[3], mid[3], ll[3]), ir + 3, j, m, n)


def identical_gemm_int15_mma_kernel(
    c: MutPointer[Float32, MutAnyOrigin],
    ah: MutPointer[Int8, MutAnyOrigin],
    al: MutPointer[Int8, MutAnyOrigin],
    ea: MutPointer[Int32, MutAnyOrigin],
    bh: MutPointer[Int8, MutAnyOrigin],
    bl: MutPointer[Int8, MutAnyOrigin],
    eb: MutPointer[Int32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    k_in: Int32,
    aligned_in: Int32,
):
    """OP_NT, one 16 x 16 output tile per warp, four int8 products per
    k-tile on the vendor's integer matrix unit, then clause W-5 and the
    seam. `identical_gemm_int8_mma_kernel`'s grid and block, and its
    `aligned_in` (DEVIATION 2975): 1 when the launch found the bases of
    all four planes aligned. On a target with no integer matrix unit both
    branches are dead and the kernel stores nothing; the dispatchers never
    launch it there."""
    var m = Int(m_in)
    var n = Int(n_in)
    var k = Int(k_in)
    var aligned = aligned_in != Int32(0)
    var warp = Int(thread_idx.x) // WARP_SIZE
    var lane = Int(lane_id())
    var wm = warp // INT8_MMA_WARPS_N
    var wn = warp - wm * INT8_MMA_WARPS_N
    var row0 = Int(block_idx.y) * INT8_MMA_BLOCK_TILE_M + wm * INT8_MMA_TILE
    var col0 = Int(block_idx.x) * INT8_MMA_BLOCK_TILE_N + wn * INT8_MMA_TILE
    # Uniform across the warp, as in the int8 kernel.
    if row0 >= m or col0 >= n:
        return
    comptime if is_nvidia_gpu():
        _nvidia_warp_tile15(c, ah, al, ea, bh, bl, eb, lane, row0, col0, m, n, k, aligned)
    elif is_amd_gpu():
        _amd_warp_tile15(c, ah, al, ea, bh, bl, eb, lane, row0, col0, m, n, k, aligned)
    else:
        return


# ===========================================================================
# the launches
# ===========================================================================


def identical_gemm_int15_flat_into(
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut qa: DeviceBuffer[DType.int16],
    mut ea: DeviceBuffer[DType.int32],
    mut qb: DeviceBuffer[DType.int16],
    mut eb: DeviceBuffer[DType.int32],
    m: Int,
    n: Int,
    k: Int,
) raises:
    """The FLAT plan, always. Asynchronous where it is one launch; on
    Apple a product of more than one slice waits between slices (DEVIATION
    2979)."""
    _refuse(m, n, k, String("identical_gemm_int15"))
    var total = m * n
    var per = int15_slice_cells(m, n, k)
    var cell0 = 0
    while cell0 < total:
        var cell1 = cell0 + per
        if cell1 > total:
            cell1 = total
        ctx.enqueue_function[identical_gemm_int15_flat_kernel](
            c.unsafe_ptr(),
            qa.unsafe_ptr(),
            ea.unsafe_ptr(),
            qb.unsafe_ptr(),
            eb.unsafe_ptr(),
            Int32(m),
            Int32(n),
            Int32(k),
            Int32(cell0),
            Int32(cell1),
            grid_dim=((cell1 - cell0 + INT15_TPB - 1) // INT15_TPB, 1, 1),
            block_dim=(INT15_TPB, 1, 1),
        )
        cell0 = cell1
        if cell0 < total:
            step_count_sync()
            ctx.synchronize()


def identical_gemm_int15_pieces_into(
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut ah: DeviceBuffer[DType.int8],
    mut al: DeviceBuffer[DType.int8],
    mut ea: DeviceBuffer[DType.int32],
    mut bh: DeviceBuffer[DType.int8],
    mut bl: DeviceBuffer[DType.int8],
    mut eb: DeviceBuffer[DType.int32],
    m: Int,
    n: Int,
    k: Int,
) raises:
    """The PIECES plan, always. Asynchronous where it is one launch; on
    Apple a product of more than one slice waits between slices (DEVIATION
    2979). A slice is a quarter of the FLAT plan's: a step is four
    products."""
    _refuse(m, n, k, String("identical_gemm_int15_pieces"))
    var total = m * n
    var per = int15_slice_cells(m, n, 4 * k)
    var cell0 = 0
    while cell0 < total:
        var cell1 = cell0 + per
        if cell1 > total:
            cell1 = total
        ctx.enqueue_function[identical_gemm_int15_pieces_kernel](
            c.unsafe_ptr(),
            ah.unsafe_ptr(),
            al.unsafe_ptr(),
            ea.unsafe_ptr(),
            bh.unsafe_ptr(),
            bl.unsafe_ptr(),
            eb.unsafe_ptr(),
            Int32(m),
            Int32(n),
            Int32(k),
            Int32(cell0),
            Int32(cell1),
            grid_dim=((cell1 - cell0 + INT15_TPB - 1) // INT15_TPB, 1, 1),
            block_dim=(INT15_TPB, 1, 1),
        )
        cell0 = cell1
        if cell0 < total:
            step_count_sync()
            ctx.synchronize()


def identical_gemm_int15_mma_into(
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut ah: DeviceBuffer[DType.int8],
    mut al: DeviceBuffer[DType.int8],
    mut ea: DeviceBuffer[DType.int32],
    mut bh: DeviceBuffer[DType.int8],
    mut bl: DeviceBuffer[DType.int8],
    mut eb: DeviceBuffer[DType.int32],
    m: Int,
    n: Int,
    k: Int,
) raises:
    """The matrix-unit plan, always. Refuses by name on a column whose
    kernel-matrix row says it has no integer matrix unit. Asynchronous."""
    comptime if not lib_int8_matrix_unit_for[TARGET_COLUMN]():
        raise Error(
            "identical_gemm_int15_mma: column " + column_name(TARGET_COLUMN)
            + " has no int8 matrix unit (kernel_matrix row"
            " lib_int8_matrix_unit_for); the flat and pieces kernels serve it"
        )
    else:
        _refuse(m, n, k, String("identical_gemm_int15_mma"))
        var grid_x = (n + INT8_MMA_BLOCK_TILE_N - 1) // INT8_MMA_BLOCK_TILE_N
        var grid_y = (m + INT8_MMA_BLOCK_TILE_M - 1) // INT8_MMA_BLOCK_TILE_M
        # DEVIATION 2975: the fragment loads state their alignment only
        # when the bases of ALL FOUR planes are aligned, read here.
        var aligned = Int32(0)
        comptime if not INT8_MMA_UNSTATED_LOADS:
            if mma_operands_aligned(Int(ah.unsafe_ptr()), Int(al.unsafe_ptr())) and mma_operands_aligned(Int(bh.unsafe_ptr()), Int(bl.unsafe_ptr())):
                aligned = Int32(1)
        ctx.enqueue_function[identical_gemm_int15_mma_kernel](
            c.unsafe_ptr(),
            ah.unsafe_ptr(),
            al.unsafe_ptr(),
            ea.unsafe_ptr(),
            bh.unsafe_ptr(),
            bl.unsafe_ptr(),
            eb.unsafe_ptr(),
            Int32(m),
            Int32(n),
            Int32(k),
            aligned,
            grid_dim=(grid_x, grid_y, 1),
            block_dim=(INT8_MMA_TPB, 1, 1),
        )


# ===========================================================================
# the entry points
# ===========================================================================


struct Int15Workspace(Movable):
    """The four planes of one product on ONE in-order context, for a caller
    who holds codes. Grown on demand; growth drains the context first, as
    `LowbitWorkspace` does."""

    var ah: DeviceBuffer[DType.int8]
    var al: DeviceBuffer[DType.int8]
    var bh: DeviceBuffer[DType.int8]
    var bl: DeviceBuffer[DType.int8]

    def __init__(out self, ctx: DeviceContext) raises:
        step_count_device_alloc()
        self.ah = ctx.enqueue_create_buffer[DType.int8](1)
        step_count_device_alloc()
        self.al = ctx.enqueue_create_buffer[DType.int8](1)
        step_count_device_alloc()
        self.bh = ctx.enqueue_create_buffer[DType.int8](1)
        step_count_device_alloc()
        self.bl = ctx.enqueue_create_buffer[DType.int8](1)

    def ensure(mut self, ctx: DeviceContext, a_codes: Int, b_codes: Int) raises:
        if a_codes > len(self.ah) or b_codes > len(self.bh):
            step_count_sync()
            ctx.synchronize()
            if a_codes > len(self.ah):
                step_count_device_alloc()
                self.ah = ctx.enqueue_create_buffer[DType.int8](a_codes)
                step_count_device_alloc()
                self.al = ctx.enqueue_create_buffer[DType.int8](a_codes)
            if b_codes > len(self.bh):
                step_count_device_alloc()
                self.bh = ctx.enqueue_create_buffer[DType.int8](b_codes)
                step_count_device_alloc()
                self.bl = ctx.enqueue_create_buffer[DType.int8](b_codes)


def identical_gemm_int15_planes_into(
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut ah: DeviceBuffer[DType.int8],
    mut al: DeviceBuffer[DType.int8],
    mut ea: DeviceBuffer[DType.int32],
    mut bh: DeviceBuffer[DType.int8],
    mut bl: DeviceBuffer[DType.int8],
    mut eb: DeviceBuffer[DType.int32],
    m: Int,
    n: Int,
    k: Int,
) raises:
    """**THE ENTRY POINT of `mojolearn.identical.gemm.int15i64.v1` for
    operands held as planes**, OP_NT: `C[m x n] = Q_a[m x k] . Q_b[n x k]^T`
    dequantized. The MMA plan when the column's row says True and the build
    does not force the flat plans; on Apple the float-unit plan above
    `INT15_APPLE_UNIT_MIN_MAC`; the PIECES plan otherwise. All are the
    profile (clause W-8). Asynchronous, but see DEVIATION 2979."""
    comptime if INT15_MMA_ENABLED:
        identical_gemm_int15_mma_into(ctx, c, ah, al, ea, bh, bl, eb, m, n, k)
    elif INT15_APPLE_UNIT_ENABLED:
        _refuse(m, n, k, String("identical_gemm_int15"))
        if int15_apple_takes_unit(m, n, k):
            identical_gemm_int15_apple_into(
                ctx, c, ah, al, ea, bh, bl, eb, m, n, k, int15_apple_form(m)
            )
        else:
            identical_gemm_int15_pieces_into(ctx, c, ah, al, ea, bh, bl, eb, m, n, k)
    else:
        identical_gemm_int15_pieces_into(ctx, c, ah, al, ea, bh, bl, eb, m, n, k)


def identical_gemm_int15_into(
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut qa: DeviceBuffer[DType.int16],
    mut ea: DeviceBuffer[DType.int32],
    mut qb: DeviceBuffer[DType.int16],
    mut eb: DeviceBuffer[DType.int32],
    mut work: Int15Workspace,
    m: Int,
    n: Int,
    k: Int,
) raises:
    """**THE ENTRY POINT of `mojolearn.identical.gemm.int15i64.v1` for
    operands held as codes**, OP_NT. On a column that has the unit: both
    operands split into `work`'s planes, then the MMA plan. On Apple above
    `INT15_APPLE_UNIT_MIN_MAC`: the split, then the float-unit plan. On any
    other column or shape, and under `MOJOLEARN_INT15_FORCE_FLAT`: the FLAT
    plan on the codes, no split. Asynchronous (but see DEVIATION 2979): the
    caller owns `work` and waits."""
    _refuse(m, n, k, String("identical_gemm_int15"))
    comptime if INT15_MMA_ENABLED:
        work.ensure(ctx, m * k, n * k)
        split_int15_device(ctx, work.ah, work.al, qa, m * k)
        split_int15_device(ctx, work.bh, work.bl, qb, n * k)
        identical_gemm_int15_mma_into(
            ctx, c, work.ah, work.al, ea, work.bh, work.bl, eb, m, n, k
        )
    elif INT15_APPLE_UNIT_ENABLED:
        if int15_apple_takes_unit(m, n, k):
            work.ensure(ctx, m * k, n * k)
            split_int15_device(ctx, work.ah, work.al, qa, m * k)
            split_int15_device(ctx, work.bh, work.bl, qb, n * k)
            identical_gemm_int15_apple_into(
                ctx, c, work.ah, work.al, ea, work.bh, work.bl, eb, m, n, k, int15_apple_form(m)
            )
        else:
            identical_gemm_int15_flat_into(ctx, c, qa, ea, qb, eb, m, n, k)
    else:
        identical_gemm_int15_flat_into(ctx, c, qa, ea, qb, eb, m, n, k)


def identical_gemm_int15_from_f32(
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
) raises:
    """Quantize both float32 operands on the device by the profile's rule,
    then the product. The PARALLEL schedule of the quantizer, straight to
    the form this column's plan reads: planes where the column has the
    unit, codes elsewhere. Synchronizes before it returns."""
    _refuse(m, n, k, String("identical_gemm_int15"))
    step_count_device_alloc()
    var ea = ctx.enqueue_create_buffer[DType.int32](m)
    step_count_device_alloc()
    var eb = ctx.enqueue_create_buffer[DType.int32](n)
    var quant = Int15QuantWorkspace(ctx)
    comptime if INT15_MMA_ENABLED:
        var work = Int15Workspace(ctx)
        work.ensure(ctx, m * k, n * k)
        quantize_planes_int15_parallel_device(ctx, work.ah, work.al, ea, a, quant, m, k, False)
        quantize_planes_int15_parallel_device(ctx, work.bh, work.bl, eb, b, quant, n, k, False)
        identical_gemm_int15_mma_into(
            ctx, c, work.ah, work.al, ea, work.bh, work.bl, eb, m, n, k
        )
        step_count_sync()
        ctx.synchronize()
        _ = work^
    else:
        var unit = False
        comptime if INT15_APPLE_UNIT_ENABLED:
            unit = int15_apple_takes_unit(m, n, k)
        if unit:
            var work = Int15Workspace(ctx)
            work.ensure(ctx, m * k, n * k)
            quantize_planes_int15_parallel_device(ctx, work.ah, work.al, ea, a, quant, m, k, False)
            quantize_planes_int15_parallel_device(ctx, work.bh, work.bl, eb, b, quant, n, k, False)
            identical_gemm_int15_planes_into(
                ctx, c, work.ah, work.al, ea, work.bh, work.bl, eb, m, n, k
            )
            step_count_sync()
            ctx.synchronize()
            _ = work^
        else:
            step_count_device_alloc()
            var qa = ctx.enqueue_create_buffer[DType.int16](m * k)
            step_count_device_alloc()
            var qb = ctx.enqueue_create_buffer[DType.int16](n * k)
            quantize_int15_parallel_device(ctx, qa, ea, a, quant, m, k, False)
            quantize_int15_parallel_device(ctx, qb, eb, b, quant, n, k, False)
            identical_gemm_int15_flat_into(ctx, c, qa, ea, qb, eb, m, n, k)
            step_count_sync()
            ctx.synchronize()
            _ = qa
            _ = qb
    _ = ea
    _ = eb
    _ = quant^
