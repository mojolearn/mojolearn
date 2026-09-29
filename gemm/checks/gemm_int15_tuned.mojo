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

ONE LAUNCH (FUSED), AND THE TWO-LAUNCH PATH KEPT AS AN ARM (2026-09-29,
the epilogue fold, the orchestrator's approval of the five-point interface).
  SUMS      `gemm_int8_mma_tuned.mojo`, another lane's kernel and not
            edited here: the two planes of each operand staged once per
            window in threadgroup memory, four unit steps per tile, three
            Int32 sums per cell (`HH`, `HL + LH`, `LL`). No float in it.
  EPILOGUE  `gemm_int15_epilogue.mojo::int15_store_cell`: clause W-5's
            recombination in Int64, W-6's conversion, W-7's scale, through
            `checks/numerics_int15.mojo`'s seams.
  FUSED     the sums kernel with `FUSED = True` calls `int15_store_cell` at
            its store: one launch, no sums in device memory. THE ENTRY
            POINTS TAKE IT. Until lane/lowbit-mma-speed's half is pushed
            the fused path is a STUB (`INT15_FUSED_IS_STUB`): the two-launch
            path under the fused name.
  TWO LAUNCH the sums stored side by side at `3 (i n + j)`, then
            `int15_sums_epilogue_kernel` calls `int15_store_cell` on them.
            `identical_gemm_int15_tuned_two_launch_*`: a gate arm (fused and
            two-launch must print one digest) and a timing arm.
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

WHAT THE FOLD SAVES. The two-launch path pays one more launch and twelve
bytes written and read per output cell. Whether the fused kernel is faster
is a measurement (`bench/gemm_int15_price_main.mojo`, run 6).

THE SABOTAGE ARMS. `-D MOJOLEARN_LOWBIT_SABOTAGE=1` flips the value of
every cell `int15_store_cell` stores (and the lowest bit of every sum the
two-launch sums kernel stores). `-D MOJOLEARN_INT8_PIECES_SABOTAGE=1` is
the sums kernel's own defect: its middle sum takes `HL` twice and never
`LH`. `-D MOJOLEARN_INT15_EPILOGUE_SABOTAGE=1`: the epilogue reads the
middle sum where the high one belongs. `-D MOJOLEARN_INT15_EXPONENT_SABOTAGE=1`:
the column exponent is read at the row index. Every one reaches both paths.
"""

from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.kernel_matrix import TARGET_COLUMN, column_name, lib_int8_matrix_unit_for
from gemm.checks.gemm_identical import step_count_device_alloc, step_count_sync
from gemm.checks.gemm_int15 import INT15_TPB, identical_gemm_int15_mma_into
from gemm.checks.gemm_int15_epilogue import int15_epilogue_sabotage_name, int15_store_cell
from gemm.checks.gemm_int8_mma_tuned import (
    INT8_PIECES_MAX_K,
    int8_pieces_dispatch,
    identical_gemm_int8_pieces_tuned_with_plan,
)
from gemm.host.gemm_int15_oracle import INT15_MAX_K

#: Whether this build can run the plan: a column with the integer unit.
comptime INT15_TUNED_AVAILABLE = lib_int8_matrix_unit_for[TARGET_COLUMN]()

#: Whether the FUSED path is lane/lowbit-mma-speed's fused kernel (True) or
#: the STUB below, which is the two-launch path under the fused name, so the
#: gate and the timing harness build before that lane's half is pushed. A
#: stub build's fused and two-launch digests agree by construction and its
#: fused time is the two-launch time: neither says anything of the fold.
comptime INT15_FUSED_IS_STUB = True


def int15_tuned_sabotage_name() -> String:
    return int15_epilogue_sabotage_name()


def int15_fused_name() -> String:
    comptime if INT15_FUSED_IS_STUB:
        return String("STUB(two-launch; lane/lowbit-mma-speed's fused kernel not yet pushed)")
    else:
        return String("fused(lane/lowbit-mma-speed identical_gemm_int8_pieces_tuned_fused_with_plan)")


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
    """The TWO-LAUNCH path's second launch, one thread per cell: the three
    sums at `3 (i n + j)` to the cell through `int15_store_cell`, the
    function the fused kernel calls at its store."""
    var m = Int(m_in)
    var n = Int(n_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= m * n:
        return
    var i = cell // n
    var j = cell - i * n
    int15_store_cell(
        c, ea, eb, s.unsafe_load(3 * cell), s.unsafe_load(3 * cell + 1), s.unsafe_load(3 * cell + 2), i, j, m, n
    )


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


def _fused_with_plan_stub(
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
    """THE STUB of lane/lowbit-mma-speed's
    `identical_gemm_int8_pieces_tuned_fused_with_plan(ctx, c, ah, al, ea, bh,
    bl, eb, m, n, k, plan)`: the two-launch path. Removed when that lane's
    half is pushed (`INT15_FUSED_IS_STUB`)."""
    work.ensure(ctx, m * n)
    identical_gemm_int8_pieces_tuned_with_plan(ctx, work.sums, ah, al, bh, bl, m, n, k, plan)
    _epilogue(ctx, c, work, ea, eb, m, n)


def _check_tuned_named(m: Int, n: Int, k: Int) raises:
    comptime if not INT15_TUNED_AVAILABLE:
        raise Error(
            "identical_gemm_int15_tuned: column " + column_name(TARGET_COLUMN)
            + " has no int8 matrix unit (kernel_matrix row lib_int8_matrix_unit_for)"
        )
    _refuse_tuned(m, n, k)
    if not int15_tuned_admits(m, n, k):
        raise Error(
            "identical_gemm_int15_tuned: the sums kernel admits k up to "
            + String(INT8_PIECES_MAX_K) + ", got " + String(k)
            + "; identical_gemm_int15_tuned_into takes the reference unit plan there"
        )


def identical_gemm_int15_tuned_two_launch_with_plan(
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
    """The TWO-LAUNCH path on a NAMED plan of the sums kernel: the sums to
    `work`, then `int15_sums_epilogue_kernel`. A gate arm and a timing arm;
    no entry point dispatches it. Refuses by name on a column with no
    integer matrix unit and a `k` the sums kernel does not admit.
    Asynchronous: the caller owns `work`."""
    _check_tuned_named(m, n, k)
    comptime if INT15_TUNED_AVAILABLE:
        work.ensure(ctx, m * n)
        identical_gemm_int8_pieces_tuned_with_plan(ctx, work.sums, ah, al, bh, bl, m, n, k, plan)
        _epilogue(ctx, c, work, ea, eb, m, n)


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
    """The TUNED plan, FUSED, on a NAMED plan of the sums kernel: one
    launch, the epilogue at the sums kernel's store. `work` is untouched on
    the fused path (the stub uses it). Refuses as the two-launch path does.
    Asynchronous."""
    _check_tuned_named(m, n, k)
    comptime if INT15_TUNED_AVAILABLE:
        _fused_with_plan_stub(ctx, c, ah, al, ea, bh, bl, eb, work, m, n, k, plan)


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
    """THE ENTRY POINT of the tuned plan: FUSED, on the plan the sums
    kernel's own dispatcher names; the reference unit plan at `k = 65536`.
    Asynchronous."""
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
        identical_gemm_int15_tuned_with_plan(ctx, c, ah, al, ea, bh, bl, eb, work, m, n, k, int8_pieces_dispatch(m, n, k))


def identical_gemm_int15_tuned_two_launch_into(
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
    """The TWO-LAUNCH path on the dispatched plan, the timing harness's
    other arm of the fold. The reference unit plan at `k = 65536`."""
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
        identical_gemm_int15_tuned_two_launch_with_plan(
            ctx, c, ah, al, ea, bh, bl, eb, work, m, n, k, int8_pieces_dispatch(m, n, k)
        )
