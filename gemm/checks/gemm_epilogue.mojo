# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The GEMM store epilogue arm (lane fx-gemm-epi, 2026-11-18).

`-D MOJOLEARN_GEMM_FAST_EPI=1`, default OFF, OPEN (no A/B yet). A product
whose output is joined to the value already in its destination stores the
join, so the follower add launch is not issued. The kinds and the argument
for unchanged bits are beside `gemm_epi_cell` in `gemm_identical.mojo`.

What takes a call (execution plan only, contract 6.1; no rule reads a board
shape): exactly the calls the shipped dispatch runs on the packed body
(`_shipped_body_kpack_hg`: kernel body row 1, the plan chooser answering the
TUNED 128x128 plan), in the launch form that dispatch would use.

  all leaves in one launch   `identical_gemm_kpack_kernel[..., EPI]`: the
                             epilogue in the packed store. ONE launch where
                             the two-launch form has two.
  leaf groups + fold         the SAME group launch into the caller's
                             workspace (`_kpack_launch`, untouched), then
                             `identical_gemm_fold_stack_epi_kernel[EPI]`: the
                             fold kernel's body with the epilogue in its
                             store. TWO launches where there were three.

Everything else declines (`identical_gemm_epi_takes` answers False) and the
caller runs the plain product and its follower launch, unchanged: every other
plan, an output of at most `SPLIT_BLOCK_FOLD_MAX_CELLS` cells under groups
(the block fold), `k == 0`, a trial or sabotage build, and every column whose
kernel body row is not 1 (AMD, Apple, the host today). Declining moves no
bit: the epilogue IS the follower's expression.

Parallel plan: one thread per output cell in the fold (grid
`ceil(m n / 256)` blocks), the packed kernel's own tile grid otherwise; no
host loop over data, nothing serial added.
"""
from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from std.sys.compile import is_defined
from checks.kernel_matrix import (
    COLUMN_NVIDIA,
    TARGET_COLUMN,
    lib_smem_page_fits_for,
    lib_smem_pages_for,
)
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz
from core.step_phase import step_count_launch
from gemm.checks.gemm_identical import (
    ANY_SABOTAGE,
    FLAT_TPB,
    GEMM_ARM_TRIAL,
    GEMM_BODY_KPACK_HG,
    GEMM_EPI_C_PLUS_CELL,
    GEMM_EPI_CELL_PLUS_C,
    GEMM_FOLD_SLOTS,
    GEMM_IDENTICAL_LEAF_SPLIT,
    GEMM_IDENTICAL_MFMA,
    GEMM_KPACK_ALIGN,
    GEMM_KPACK_CPT,
    GEMM_KPACK_FS,
    GEMM_KPACK_KS,
    GEMM_KPACK_PAD,
    GEMM_KPACK_PAGE_GUARD_BYTES,
    GEMM_KPACK_RPT,
    GEMM_REUSE_GROUP_WS,
    IDN_GEMM_FILL_TILES,
    NI06_ONE_PAGE,
    PLAN_TUNED_128_8X8,
    SPLIT_BLOCK_FOLD_MAX_CELLS,
    TUNED_TC,
    TUNED_TPB,
    _fold_drain,
    _fold_push,
    _kpack_launch,
    _ksplit_resolve_leaves,
    _tile_grid,
    choose_gemm_plan,
    contract_partition,
    gemm_default_ksplit_leaves,
    gemm_epi_cell,
    gemm_fill_note_device,
    gemm_kpack_fold_slots_for,
    gemm_operand_strides,
    identical_gemm_kpack_kernel,
)

#: The arm's define.
comptime GEMM_FAST_EPI = is_defined["MOJOLEARN_GEMM_FAST_EPI"]()
#: Whether this build has a fused store at all: the define, IDENTICAL, and the
#: shipped dispatch on the packed body with the reusable group workspace (the
#: NVIDIA column today). A trial build, a GEMM sabotage build and the
#: matrix-core column keep the two-launch form.
comptime GEMM_EPI_FUSES = (
    GEMM_FAST_EPI
    and GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and TARGET_COLUMN == COLUMN_NVIDIA
    and GEMM_BODY_KPACK_HG
    and GEMM_REUSE_GROUP_WS
    and not GEMM_IDENTICAL_MFMA
    and not GEMM_IDENTICAL_LEAF_SPLIT
    and not GEMM_ARM_TRIAL
    and not ANY_SABOTAGE
    and not is_defined["MOJOLEARN_GEMM_SABOTAGE_WS_FOLD"]()
)


def identical_gemm_fold_stack_epi_kernel[EPI: Int](
    c: MutPointer[Float32, MutAnyOrigin],
    ws: MutPointer[Float32, MutAnyOrigin],
    mn_in: Int32,
    p_in: Int32,
):
    """`identical_gemm_fold_stack_kernel` with the epilogue in its store.

    The same loads (leaf-major, batches of eight issued before their
    pushes), the same `_fold_push` in ascending `t` and the same
    `_fold_drain`: the contract's tree (7.2, 7.3). The plain kernel stores
    `ftz(root)`; this one stores `gemm_epi_cell[EPI](ftz(root), c[cell])`,
    the follower add's expression on the word the plain kernel would have
    stored and the destination word read here, before the store. One thread
    owns one cell, so no other thread reads or writes `c[cell]`.
    """
    var mn = Int(mn_in)
    var p_count = Int(p_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= mn:
        return
    var stack = SIMD[DType.float32, GEMM_FOLD_SLOTS](0.0)
    var occ = 0
    comptime FOLD_BATCH = 8
    var t = 0
    while t + FOLD_BATCH <= p_count:
        var vals = SIMD[DType.float32, FOLD_BATCH](0.0)
        comptime for e in range(FOLD_BATCH):
            vals[e] = ws.unsafe_load((t + e) * mn + cell)
        comptime for e2 in range(FOLD_BATCH):
            _ = _fold_push(stack, occ, vals[e2])
        t += FOLD_BATCH
    while t < p_count:
        var part = ws.unsafe_load(t * mn + cell)
        _ = _fold_push(stack, occ, part)
        t += 1
    var root = _fold_drain(stack, occ)
    # 5g, then the epilogue.
    c.unsafe_store(cell, gemm_epi_cell[EPI](ftz(root), c.unsafe_load(cell)))


def _kpack_epi_launch[FS: Int, EPI: Int](
    ctx: DeviceContext,
    mut dst: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
    leaf: Int,
    p_count: Int,
    st: Tuple[Int, Int, Int, Int],
) raises:
    """`_kpack_launch` in all-leaves mode at the shipped `kpack_hg` geometry
    (padded aligned page, hardware fold flush, gather staging), with the
    store epilogue `EPI`. The page, the grid and every argument are
    `_kpack_launch`'s; `dst` is read by the store."""
    comptime NTH = TUNED_TPB
    comptime TR = NTH // TUNED_TC
    comptime BM = GEMM_KPACK_RPT * TR
    comptime BN = GEMM_KPACK_CPT * TUNED_TC
    comptime PAGE_BYTES = (
        BM * GEMM_KPACK_KS + TR * GEMM_KPACK_PAD + BN * GEMM_KPACK_KS + TUNED_TC * GEMM_KPACK_PAD
    ) * 4
    comptime PAGES = 1 if (NI06_ONE_PAGE or is_defined["MOJOLEARN_GEMM_ONE_PAGE"]()) else lib_smem_pages_for[
        TARGET_COLUMN, PAGE_BYTES + GEMM_KPACK_PAGE_GUARD_BYTES
    ]()
    comptime assert lib_smem_page_fits_for[TARGET_COLUMN, PAGE_BYTES](), (
        "_kpack_epi_launch: one packed page of this geometry exceeds the column's"
        " shared limit"
    )
    comptime kern = identical_gemm_kpack_kernel[
        GEMM_KPACK_RPT, GEMM_KPACK_CPT, TUNED_TC, GEMM_KPACK_KS, FS, PAGES, False, False,
        GEMM_KPACK_PAD, GEMM_KPACK_ALIGN, 0, True, True, EPI
    ]
    var g = _tile_grid(m, n, BM, BN, False)
    # No launch count here: `_kpack_launch` counts none, and the arm's
    # counters must read against the default's.
    ctx.enqueue_function[kern](
        dst.unsafe_ptr(),
        a.unsafe_ptr(),
        b.unsafe_ptr(),
        Int32(m),
        Int32(n),
        Int32(k),
        Int32(leaf),
        Int32(p_count),
        Int32(st[0]),
        Int32(st[1]),
        Int32(st[2]),
        Int32(st[3]),
        Int32(0),
        Int32(0),
        grid_dim=(g[0], 1, 1),
        block_dim=(NTH, 1, 1),
    )


def identical_gemm_epi_takes(m: Int, n: Int, k: Int) -> Bool:
    """Whether `identical_gemm_epi_into` fuses the join at `(m, n, k)` on
    this build (module docstring). Reads sizes and the plan chooser only, so
    a caller can decide where its operands go before any launch. Always
    False on a build without a fused store."""
    comptime if not GEMM_EPI_FUSES:
        return False
    if m <= 0 or n <= 0:
        return False
    if choose_gemm_plan(m, n, k) != PLAN_TUNED_128_8X8:
        return False
    var p_count = contract_partition(k)[1]
    if p_count <= 0:
        return False
    if gemm_default_ksplit_leaves(m, n, k) > 0 and m * n <= SPLIT_BLOCK_FOLD_MAX_CELLS:
        # Groups whose fold is one BLOCK per cell: not given an epilogue.
        return False
    return True


def identical_gemm_epi_into[EPI: Int](
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    mut ws: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
    op: Int,
) raises:
    """`c[i, j] = gemm_epi_cell[EPI]((op(A) . op(B))[i, j], c[i, j])`, the
    product's leaves, fold tree and stored cell exactly as
    `identical_gemm_into` computes them on this build. ASYNCHRONOUS, as
    that entry. Call only where `identical_gemm_epi_takes(m, n, k)` is
    True (RAISES otherwise, before any launch). `ws` holds at least
    `identical_gemm_workspace_max_floats(m, n, k)` floats, as for
    `identical_gemm_into`; `c` overlaps neither operand nor `ws`."""
    comptime assert EPI == GEMM_EPI_C_PLUS_CELL or EPI == GEMM_EPI_CELL_PLUS_C, (
        "identical_gemm_epi_into: kind 1 (prior + cell) or 2 (cell + prior)"
    )
    comptime if not GEMM_EPI_FUSES:
        raise Error("identical_gemm_epi_into: this build has no fused GEMM store (identical_gemm_epi_takes is False)")
    else:
        comptime if IDN_GEMM_FILL_TILES:
            # As `identical_gemm_shipped_into`: before the first plan choice.
            gemm_fill_note_device(ctx)
        if not identical_gemm_epi_takes(m, n, k):
            raise Error("identical_gemm_epi_into: identical_gemm_epi_takes(m, n, k) is False for this call")
        var part = contract_partition(k)
        var gl = gemm_default_ksplit_leaves(m, n, k)
        var st = gemm_operand_strides(op, m, n, k)
        # The fold-stack class `_shipped_body_kpack_hg` picks on this column
        # (unused local storage only; it cannot change the fold DAG).
        var fs4 = False
        comptime if not is_defined["MOJOLEARN_LEGACY_SHAPE_NV_FS4"]() and not is_defined["MOJOLEARN_GEMM_NV_FS4_OFF"]():
            fs4 = gemm_kpack_fold_slots_for(part[1], gl) == 4
        if gl > 0:
            var rg = _ksplit_resolve_leaves(gl, part[1])
            if len(ws) < m * n * rg[1]:
                raise Error("identical_gemm_epi_into: the workspace does not hold the group nodes")
            if fs4:
                _kpack_launch[
                    GEMM_KPACK_RPT, GEMM_KPACK_CPT, TUNED_TC, GEMM_KPACK_KS, 4, True, False,
                    GEMM_KPACK_PAD, GEMM_KPACK_ALIGN, 0, True, True,
                ](ctx, ws, a, b, m, n, k, part[0], part[1], st, rg[0], rg[1])
            else:
                _kpack_launch[
                    GEMM_KPACK_RPT, GEMM_KPACK_CPT, TUNED_TC, GEMM_KPACK_KS, GEMM_KPACK_FS, True, False,
                    GEMM_KPACK_PAD, GEMM_KPACK_ALIGN, 0, True, True,
                ](ctx, ws, a, b, m, n, k, part[0], part[1], st, rg[0], rg[1])
            step_count_launch()
            ctx.enqueue_function[identical_gemm_fold_stack_epi_kernel[EPI]](
                c.unsafe_ptr(),
                ws.unsafe_ptr(),
                Int32(m * n),
                Int32(rg[1]),
                grid_dim=((m * n + FLAT_TPB - 1) // FLAT_TPB, 1, 1),
                block_dim=(FLAT_TPB, 1, 1),
            )
            return
        if fs4:
            _kpack_epi_launch[4, EPI](ctx, c, a, b, m, n, k, part[0], part[1], st)
        else:
            _kpack_epi_launch[GEMM_KPACK_FS, EPI](ctx, c, a, b, m, n, k, part[0], part[1], st)
