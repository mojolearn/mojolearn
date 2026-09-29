# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The PARALLEL quantizer of `mojolearn.identical.gemm.int8i32.v1`: the same
codes and the same exponents as `gemm_lowbit.mojo::quantize_rows_int8_kernel`,
written by a block of threads per row instead of one thread per row.

Lane lane/lowbit-mma-speed, 2026-09-29. Contract clauses L-3 and L-4 of
`gemm/IDENTICAL_LOWBIT_CONTRACT.md`; the answer is
`gemm/host/gemm_lowbit_oracle.mojo::quantize_rows_int8`; the gate is
`gemm/checks/gemm_int8_mma_tuned_check.mojo::check_par_quantizer_matches`.
`gemm_lowbit.mojo` is not edited: its quantizer stays the reference device
plan and this file is a second plan of the same seam.

WHY IT EXISTS. lane/lowbit-units timed the one-thread-per-row quantizer on
an H100: 2.47 ms for 512 x 4096 activations against 1.44 ms for the int8
product they feed, and 0.49 ms for ONE row of 4096. A row is one serial
chain of loads there, so its time is the row's length whatever the box.

WHY THE CODES CANNOT MOVE (the construction argument).
  L-3  The row exponent is a function of the row's absmax. The absmax is a
       MAXIMUM of `|ftz(x)|` over the row's values that are not NaN, and a
       maximum is exact and the same under every order and every grouping.
       Each thread here takes the maximum of the values it owns with the
       reference's own comparison (`v > best` from `+0.0`, so a NaN is
       never taken and the result is never `-0.0`); the block then takes
       the maximum of the threads' maxima in threadgroup memory. The
       exponent is `int8_row_exponent` of that, the reference's call.
  L-4  A code is a function of ONE value and the row exponent:
       `quantize_int8_value(x, e)`, the reference's call, character for
       character. Which thread computes it is scheduling.
So the threads per block, the four-wide loads and the packed store of four
codes decide who computes what and in how many transactions, and none of
them reaches an expression that produces a code or an exponent.

THE SCHEDULE. One block per row. A SLOT is four consecutive columns; thread
`tid` owns slots `tid, tid + NT, ...`, so consecutive threads read
consecutive sixteen bytes of the row and the loads coalesce. Pass one reduces
the absmax; the block's tree runs in threadgroup memory with one barrier per
level; thread 0 stores the exponent; pass two writes the codes, four to a
store where the row is a whole number of slots. No thread reads device
memory another thread wrote, so nothing here leans on what a barrier orders
in device memory.

THE SABOTAGE ARMS.
  `-D MOJOLEARN_QUANT_PAR_SABOTAGE=1` skips the first level of the
      block's tree: the maxima held by the upper half of the threads never
      reach slot 0, so a row whose absmax one of them owns takes the wrong
      exponent. A SCHEDULING defect, the kind this file can have; it must
      fail `check_par_quantizer_matches` and its planted rows say where.
  `-D MOJOLEARN_LOWBIT_SABOTAGE=1`, the value arm the int8 kernels read
      (DEVIATION 2908), flips the lowest bit of every code stored here.
"""

from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation
from std.sys import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.numerics import ftz, int8_row_exponent, quantize_int8_value

#: The scheduling arm: the tree's first level is skipped. Off in every build
#: that does not name it.
comptime QUANT_PAR_SABOTAGE = is_defined["MOJOLEARN_QUANT_PAR_SABOTAGE"]()

#: DEVIATION 2908, the value arm, read from the define the int8 kernels read.
comptime QUANT_PAR_VALUE_SABOTAGE = is_defined["MOJOLEARN_LOWBIT_SABOTAGE"]()

#: Columns per slot: one four-wide load and one four-wide store. SCHEDULING.
comptime QUANT_PAR_SLOT = 4

#: Threads per block of the two instantiations. A row of at most
#: `QUANT_PAR_NARROW_MAX_COLS` columns takes the narrow block, where the
#: wide one would launch threads that own no slot. SCHEDULING: the gate runs
#: both on every shape.
comptime QUANT_PAR_TPB_NARROW = 64
comptime QUANT_PAR_TPB_WIDE = 256
comptime QUANT_PAR_NARROW_MAX_COLS = 1024

comptime QUANT_PAR_BLOCK_NARROW = 0
comptime QUANT_PAR_BLOCK_WIDE = 1


def quantize_par_sabotage_name() -> String:
    comptime if QUANT_PAR_SABOTAGE:
        return String("TREE_SKIPS_ITS_FIRST_LEVEL")
    elif QUANT_PAR_VALUE_SABOTAGE:
        return String("LOWBIT_VALUE_FLIP")
    else:
        return String("none")


def quantize_par_block(cols: Int) -> Int:
    """The block the launcher picks. Reads `cols` and may: the block is a
    schedule and every code is the same on either."""
    if cols <= QUANT_PAR_NARROW_MAX_COLS:
        return QUANT_PAR_BLOCK_NARROW
    return QUANT_PAR_BLOCK_WIDE


def quantize_par_block_name(block: Int) -> String:
    if block == QUANT_PAR_BLOCK_NARROW:
        return String("NARROW ") + String(QUANT_PAR_TPB_NARROW) + " threads per row"
    return String("WIDE ") + String(QUANT_PAR_TPB_WIDE) + " threads per row"


@always_inline
def _absmax_step(x: Float32, best: Float32) -> Float32:
    """One value into a running absmax: the reference's three lines
    (`quantize_rows_int8_kernel`, `row_absmax`)."""
    var v = ftz(x)
    if v < Float32(0.0):
        v = -v
    if v > best:
        return v
    return best


@always_inline
def _code(x: Float32, ex: Int) -> Int8:
    """Contract L-4, the reference's call; the value arm flips the lowest
    bit of what is stored."""
    var code = quantize_int8_value(x, ex)
    comptime if QUANT_PAR_VALUE_SABOTAGE:
        code = code ^ Int8(1)
    return code


def quantize_rows_int8_par_kernel[NT: Int](
    q: MutPointer[Int8, MutAnyOrigin],
    e: MutPointer[Int32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    rows_in: Int32,
    cols_in: Int32,
):
    """One block of `NT` threads per row: the absmax reduced by the block,
    the exponent stored once, the codes written by every thread. Grid
    `(rows, 1, 1)`, block `NT`.

    Every thread of the block reaches every `barrier()`: the one early
    return is block-uniform, and the tree's level count is a function of the
    comptime `NT`."""
    comptime assert NT >= 2 and (NT & (NT - 1)) == 0, (
        "quantize_rows_int8_par_kernel: NT must be a power of two, because"
        " the block's tree halves it"
    )
    var rows = Int(rows_in)
    var cols = Int(cols_in)
    var r = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    if r >= rows:
        return
    var best_s = stack_allocation[
        NT, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var base = r * cols
    var slots = (cols + QUANT_PAR_SLOT - 1) // QUANT_PAR_SLOT

    # ---- PASS ONE: the absmax of the slots this thread owns.
    var best = Float32(0.0)
    var s = tid
    while s < slots:
        var c0 = s * QUANT_PAR_SLOT
        if c0 + QUANT_PAR_SLOT <= cols:
            var v = x.unsafe_load[width=QUANT_PAR_SLOT, alignment=4](base + c0)
            comptime for i in range(QUANT_PAR_SLOT):
                best = _absmax_step(v[i], best)
        else:
            comptime for i in range(QUANT_PAR_SLOT):
                if c0 + i < cols:
                    best = _absmax_step(x.unsafe_load(base + c0 + i), best)
        s += NT
    best_s.unsafe_store(tid, best)
    barrier()

    # ---- THE BLOCK'S TREE, in threadgroup memory. A maximum of maxima.
    var half = NT // 2
    comptime if QUANT_PAR_SABOTAGE:
        # SABOTAGE: the first level is never taken, so slot 0 holds the
        # maximum of the LOWER half of the threads only.
        half = NT // 4
    while half > 0:
        if tid < half:
            var mine = best_s.unsafe_load(tid)
            var other = best_s.unsafe_load(tid + half)
            if other > mine:
                best_s.unsafe_store(tid, other)
        barrier()
        half = half // 2

    var ex = int8_row_exponent(best_s.unsafe_load(0))
    if tid == 0:
        e.unsafe_store(r, Int32(ex))

    # ---- PASS TWO: the codes of the slots this thread owns. The packed
    # store is taken only where it is aligned: a row of whole slots starts
    # on a multiple of four bytes (`gemm_int8_mma.mojo::_pack4`'s
    # discipline, on the store).
    var whole = (cols & (QUANT_PAR_SLOT - 1)) == 0
    s = tid
    while s < slots:
        var c0 = s * QUANT_PAR_SLOT
        if whole:
            var v = x.unsafe_load[width=QUANT_PAR_SLOT, alignment=4](base + c0)
            var codes = SIMD[DType.int8, QUANT_PAR_SLOT](0)
            comptime for i in range(QUANT_PAR_SLOT):
                codes[i] = _code(v[i], ex)
            q.unsafe_store[alignment=QUANT_PAR_SLOT](base + c0, codes)
        else:
            comptime for i in range(QUANT_PAR_SLOT):
                if c0 + i < cols:
                    q.unsafe_store(
                        base + c0 + i, _code(x.unsafe_load(base + c0 + i), ex)
                    )
        s += NT


def quantize_rows_int8_par_with_block(
    ctx: DeviceContext,
    mut q: DeviceBuffer[DType.int8],
    mut e: DeviceBuffer[DType.int32],
    mut x: DeviceBuffer[DType.float32],
    rows: Int,
    cols: Int,
    block: Int,
) raises:
    """The parallel quantizer on a NAMED block. The gate calls this to
    compare the two; callers go through `quantize_rows_int8_par_device`.
    Asynchronous."""
    if rows <= 0 or cols <= 0:
        raise Error("quantize_rows_int8_par: rows and cols must be positive")
    if block == QUANT_PAR_BLOCK_NARROW:
        comptime kern_narrow = quantize_rows_int8_par_kernel[QUANT_PAR_TPB_NARROW]
        ctx.enqueue_function[kern_narrow](
            q.unsafe_ptr(),
            e.unsafe_ptr(),
            x.unsafe_ptr(),
            Int32(rows),
            Int32(cols),
            grid_dim=(rows, 1, 1),
            block_dim=(QUANT_PAR_TPB_NARROW, 1, 1),
        )
        return
    if block != QUANT_PAR_BLOCK_WIDE:
        raise Error(
            "quantize_rows_int8_par: block must be 0 (NARROW) or 1 (WIDE), got "
            + String(block)
        )
    comptime kern_wide = quantize_rows_int8_par_kernel[QUANT_PAR_TPB_WIDE]
    ctx.enqueue_function[kern_wide](
        q.unsafe_ptr(),
        e.unsafe_ptr(),
        x.unsafe_ptr(),
        Int32(rows),
        Int32(cols),
        grid_dim=(rows, 1, 1),
        block_dim=(QUANT_PAR_TPB_WIDE, 1, 1),
    )


def quantize_rows_int8_par_device(
    ctx: DeviceContext,
    mut q: DeviceBuffer[DType.int8],
    mut e: DeviceBuffer[DType.int32],
    mut x: DeviceBuffer[DType.float32],
    rows: Int,
    cols: Int,
) raises:
    """`quantize_rows_int8_device`'s signature and its codes, by a block of
    threads per row. Asynchronous."""
    quantize_rows_int8_par_with_block(
        ctx, q, e, x, rows, cols, quantize_par_block(cols)
    )
