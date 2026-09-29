# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`mojolearn.identical.gemm.int15i64.v1` on the integer matrix units, the
TUNED plan: lane/lowbit-mma-speed's four products with one staging, then
this profile's own seams.

Lane lane/lowbit-int15, 2026-09-29, DEVIATION 2978. Contract clause W-13 of
`gemm/IDENTICAL_LOWBIT_CONTRACT.md`. The reference unit plan, the flat
plans and the dispatchers are `gemm/checks/gemm_int15.mojo`; the answer is
`gemm/host/gemm_int15_oracle.mojo::gemm_int15_oracle`; the gate is
`gemm/checks/gemm_int15_tuned_check.mojo`. NOT DISPATCHED: the entry points
still take the reference unit plan.

TWO LAUNCHES, ONE ANSWER.
  SUMS      `gemm_int8_mma_tuned.mojo::identical_gemm_int8_pieces_tuned_into`,
            another lane's kernel and not edited here: the two planes of
            each operand staged once per window in threadgroup memory, four
            unit steps per tile, three Int32 sums per cell stored side by
            side (`HH`, `HL + LH`, `LL`). It has no float in it.
  EPILOGUE  `int15_sums_epilogue_kernel`, this file: one thread per cell,
            clause W-5's recombination in Int64, clause W-6's conversion,
            clause W-7's scale. The seams are `checks/numerics_int15.mojo`'s,
            called with the arguments every other plan calls them with.
It is the profile because the three sums are exact integers under the bound
on `k`, so they are the three integers the reference unit plan holds in its
registers, whatever the tile, the staging or the order (clause W-8's
argument); and what is done with them is the same two functions.

THE BOUND ON `k`. The sums kernel refuses `k` above 65535
(`INT8_PIECES_MAX_K`): it allows a low piece of -128, which this profile
never makes, so its cross term is 32768 a step where this profile's is
32512. The profile admits `k = 65536`. At that one extent this plan is not
available and `identical_gemm_int15_tuned_into` takes the reference unit
plan, which is the same bits.

WHAT IT COSTS BESIDE THE REFERENCE PLAN. One more launch, and twelve bytes
written and read per output cell that the reference plan keeps in
registers. What it saves is the reference plan's fragment loads from device
memory at every step. Which is more is a measurement
(`bench/gemm_int15_price_main.mojo`).

THE SABOTAGE ARMS. `-D MOJOLEARN_LOWBIT_SABOTAGE=1` flips the value of
every cell the epilogue stores (and the lowest bit of every sum the sums
kernel stores). `-D MOJOLEARN_INT8_PIECES_SABOTAGE=1` is the sums kernel's
own scheduling defect: its middle sum takes `HL` twice and never `LH`.
`-D MOJOLEARN_INT15_EPILOGUE_SABOTAGE=1` is this file's: the epilogue reads
the middle sum where the high one belongs.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.sys import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.kernel_matrix import COLUMN_AMD, TARGET_COLUMN, column_name, lib_int8_matrix_unit_for
from checks.numerics_int15 import dequant_int15_pinned, int15_recombine
from gemm.checks.gemm_identical import step_count_device_alloc, step_count_sync
from gemm.checks.gemm_int15 import INT15_TPB, identical_gemm_int15_mma_into
from gemm.checks.gemm_int8_mma_amd import identical_gemm_int8_pieces_amd_into
from gemm.checks.gemm_int8_mma_tuned import (
    INT8_PIECES_MAX_K,
    identical_gemm_int8_pieces_tuned_into,
    identical_gemm_int8_pieces_tuned_with_plan,
)
from gemm.host.gemm_int15_oracle import INT15_MAX_K
from gemm.host.gemm_oracle import gemm_oracle_sabotage_value_flip

#: DEVIATION 2973, the value arm, the define every fifteen-bit plan reads.
comptime INT15_TUNED_VALUE_SABOTAGE = is_defined["MOJOLEARN_LOWBIT_SABOTAGE"]()

#: The epilogue's own defect arm: the high sum is read from the middle slot.
comptime INT15_EPILOGUE_SABOTAGE = is_defined["MOJOLEARN_INT15_EPILOGUE_SABOTAGE"]()

#: Whether this build can run the plan: a column with the integer unit.
comptime INT15_TUNED_AVAILABLE = lib_int8_matrix_unit_for[TARGET_COLUMN]()


def int15_tuned_sabotage_name() -> String:
    comptime if INT15_EPILOGUE_SABOTAGE:
        return String("INT15_EPILOGUE_READS_THE_WRONG_SUM")
    else:
        return String("none")


def int15_tuned_admits(m: Int, n: Int, k: Int) -> Bool:
    """Whether the SUMS kernel serves the shape. Every shape the profile
    admits but `k = 65536`."""
    return m > 0 and n > 0 and k > 0 and k <= INT8_PIECES_MAX_K


def int15_sums_epilogue_kernel(
    c: MutPointer[Float32, MutAnyOrigin],
    s: MutPointer[Int32, MutAnyOrigin],
    ea: MutPointer[Int32, MutAnyOrigin],
    eb: MutPointer[Int32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
):
    """One thread per cell: the three sums at `3 (i n + j)` to the cell.
    Clauses W-5, W-6 and W-7, the epilogue every plan shares."""
    var m = Int(m_in)
    var n = Int(n_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= m * n:
        return
    var i = cell // n
    var j = cell - i * n
    var hh = s.unsafe_load(3 * cell)
    var mid = s.unsafe_load(3 * cell + 1)
    var ll = s.unsafe_load(3 * cell + 2)
    comptime if INT15_EPILOGUE_SABOTAGE:
        # THE DEFECT ARM: a slot read one place off.
        hh = mid
    var out = dequant_int15_pinned(
        int15_recombine(hh, mid, ll), Int(ea.unsafe_load(i)) + Int(eb.unsafe_load(j))
    )
    comptime if INT15_TUNED_VALUE_SABOTAGE:
        out = gemm_oracle_sabotage_value_flip(out)
    c.unsafe_store(cell, out)


struct Int15SumsWorkspace(Movable):
    """The three Int32 sums per cell of one product, on ONE in-order
    context. Grown on demand; growth drains the context first."""

    var sums: DeviceBuffer[DType.int32]

    def __init__(out self, ctx: DeviceContext) raises:
        step_count_device_alloc()
        self.sums = ctx.enqueue_create_buffer[DType.int32](3)

    def ensure(mut self, ctx: DeviceContext, cells: Int) raises:
        if 3 * cells > len(self.sums):
            step_count_sync()
            ctx.synchronize()
            step_count_device_alloc()
            self.sums = ctx.enqueue_create_buffer[DType.int32](3 * cells)


def _refuse_tuned(m: Int, n: Int, k: Int) raises:
    if m <= 0 or n <= 0 or k <= 0 or k > INT15_MAX_K:
        raise Error(
            "identical_gemm_int15_tuned: m, n and k must be positive and k at"
            " most " + String(INT15_MAX_K) + " (contract W-4), got m=" + String(m)
            + " n=" + String(n) + " k=" + String(k)
        )


def _epilogue(
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut work: Int15SumsWorkspace,
    mut ea: DeviceBuffer[DType.int32],
    mut eb: DeviceBuffer[DType.int32],
    m: Int,
    n: Int,
) raises:
    ctx.enqueue_function[int15_sums_epilogue_kernel](
        c.unsafe_ptr(),
        work.sums.unsafe_ptr(),
        ea.unsafe_ptr(),
        eb.unsafe_ptr(),
        Int32(m),
        Int32(n),
        grid_dim=((m * n + INT15_TPB - 1) // INT15_TPB, 1, 1),
        block_dim=(INT15_TPB, 1, 1),
    )


def identical_gemm_int15_tuned_with_plan(
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut ah: DeviceBuffer[DType.int8],
    mut al: DeviceBuffer[DType.int8],
    mut ea: DeviceBuffer[DType.int32],
    mut bh: DeviceBuffer[DType.int8],
    mut bl: DeviceBuffer[DType.int8],
    mut eb: DeviceBuffer[DType.int32],
    mut work: Int15SumsWorkspace,
    m: Int,
    n: Int,
    k: Int,
    plan: Int,
) raises:
    """The TUNED plan on a NAMED plan of the sums kernel, for the gate.
    Refuses by name on a column with no integer matrix unit, and a `k` the
    sums kernel does not admit. Asynchronous: the caller owns `work`."""
    comptime if not INT15_TUNED_AVAILABLE:
        raise Error(
            "identical_gemm_int15_tuned: column " + column_name(TARGET_COLUMN)
            + " has no int8 matrix unit (kernel_matrix row lib_int8_matrix_unit_for)"
        )
    else:
        _refuse_tuned(m, n, k)
        if not int15_tuned_admits(m, n, k):
            raise Error(
                "identical_gemm_int15_tuned: the sums kernel admits k up to "
                + String(INT8_PIECES_MAX_K) + ", got " + String(k)
                + "; identical_gemm_int15_tuned_into takes the reference unit plan there"
            )
        work.ensure(ctx, m * n)
        identical_gemm_int8_pieces_tuned_with_plan(ctx, work.sums, ah, al, bh, bl, m, n, k, plan)
        _epilogue(ctx, c, work, ea, eb, m, n)


def identical_gemm_int15_tuned_into(
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut ah: DeviceBuffer[DType.int8],
    mut al: DeviceBuffer[DType.int8],
    mut ea: DeviceBuffer[DType.int32],
    mut bh: DeviceBuffer[DType.int8],
    mut bl: DeviceBuffer[DType.int8],
    mut eb: DeviceBuffer[DType.int32],
    mut work: Int15SumsWorkspace,
    m: Int,
    n: Int,
    k: Int,
) raises:
    """The TUNED plan on the plan the sums kernel's own dispatcher names;
    the reference unit plan at `k = 65536`. Asynchronous."""
    comptime if not INT15_TUNED_AVAILABLE:
        raise Error(
            "identical_gemm_int15_tuned: column " + column_name(TARGET_COLUMN)
            + " has no int8 matrix unit (kernel_matrix row lib_int8_matrix_unit_for)"
        )
    else:
        _refuse_tuned(m, n, k)
        if not int15_tuned_admits(m, n, k):
            identical_gemm_int15_mma_into(ctx, c, ah, al, ea, bh, bl, eb, m, n, k)
            return
        work.ensure(ctx, m * n)
        comptime if TARGET_COLUMN == COLUMN_AMD:
            # THE AMD COLUMN (lane/lowbit-amd-tuned): the sums from the
            # plan the MI325X measured least for a wavefront of 64, which
            # at the decode rows is a kernel the tuned file does not have.
            # The same three integers per cell (that file's gate).
            identical_gemm_int8_pieces_amd_into(ctx, work.sums, ah, al, bh, bl, m, n, k)
        else:
            identical_gemm_int8_pieces_tuned_into(ctx, work.sums, ah, al, bh, bl, m, n, k)
        _epilogue(ctx, c, work, ea, eb, m, n)
