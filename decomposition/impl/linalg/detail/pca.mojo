"""PCA by covariance eigendecomposition. The `input`-unchanged CONTRACT is the one to not drop: `input` is an in-out parameter that must end the call unchanged, and a fit that leaves the caller's matrix centered is wrong in a way nothing in the fit itself will reveal."""
from experiments.classical_identical_ideas.linear_controls import PCA_COV_C04, PCA_COV_C23, PCA_COV_LEGAL
from core.blocked_moments import bm_centered_gram_panels, bm_onepass_covariance
from gemm.contract import contract_leaf_size
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632


from std.gpu import block_dim, block_idx, thread_idx
from std.math import sqrt
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from max.gpu.memory import AddressSpace
from core.pinned_reduce import (
    pinned_block_max as block_max,
    pinned_block_min as block_min,
)
from max.gpu.sync import barrier
from std.memory import stack_allocation

from core.gemm import gemm_nt, gemm_tn
from std.sys.compile import is_defined
from std.ffi import _Global
from gemm.afn_apple_fast import (
    AFN_EPI_NONE,
    AFN_GEMM_APPLE,
    AFN_GEMM_KB,
    AFN_TILE_SQUARE,
    AFN_ZERO_TPB,
    afn_launch_tile,
    afn_strides,
    afn_zero_kernel,
)
from experiments.apple_fast.gemm.scoped_dispatch import try_scoped_gemm
from gemm.contract import OP_TN

#: lane/apple-fast-gap-linalg2-pca (2026-10-03): PCA_FAST_GRAM_MMA, the FAST +
#: Apple default (M3 A/B, one run per arm: pca istella 606 -> 490 ms, tag
#: gl2p-pca-grammma-istella-r; M2 quality-only check gl2p-pca-quality on a
#: seeded 200,000 x 220 matrix: explained_variance_ max relative difference
#: 3.4e-06, largest component subspace angle 1.8e-06 rad).
#: -D MOJOLEARN_PCA_FAST_GRAM_MMA_OFF restores the IDENTICAL Gram (the A/B arm).
#: Past the split-K Gram's capacity (d > 128: Istella's 220 features) the
#: FAST covariance ran `gemm_tn_identical_v1`, the IDENTICAL profile's
#: one-chain-per-cell Gram, plus a second centering pass to restore the
#: caller's copy. Here: center in place, then X^T X on the Apple matrix unit
#: (`afn_gemm_mma_kernel`, 64 x 64 tiles, `k` split across PCA_GRAM_SPLITS
#: row ranges summed with f32 atomics into a zeroed output), then the
#: 1 / (n - 1) scale. The device copy is the fit's own (`pca_fit_host`
#: uploads it), so the restore pass is skipped. FAST + Apple only.
comptime PCA_FAST_GRAM_MMA = AFN_GEMM_APPLE and not is_defined["MOJOLEARN_PCA_FAST_GRAM_MMA_OFF"]()
#: blocks the split Gram aims for (tiles x splits), about 8 per M3 Ultra core
comptime PCA_GRAM_BLOCK_TARGET = 640
from checks.numerics import ftz, identical_div, identical_mul
from core.device_zero import enqueue_fill
from decomposition.pca_rr_switch import PCA_RR_EIGH, PCA_RR_FLAG_TEST, PCA_RR_ONE_BLOCK, PCA_RR_SWEEPS
from x_decomp.rr_one_block import RR_ONE_TPB, rr_eigh_one_block_kernel, rr_one_block_applies
from x_decomp.jacobi_par import (
    PJ_TPB,
    eigh_par_off_fold_kernel,
    eigh_par_off_part_kernel,
    pj_identity_kernel,
)
from x_decomp.cells import F32Ptr
from x_decomp.eigh_scale import enqueue_es_scale, enqueue_es_unscale_diag
from x_decomp.rr import RR_OFF_TPB, rr_block, rr_converged, rr_cs, rr_fro_kept, rr_gate_state, rr_vrow
from core.gram_splitk import (
    gram_centered_splitk_into,
    gram_splitk_applies,
    gram_splitk_chunk_count,
    gram_splitk_scratch_covers,
)
from gemm.checks.gemm_identical import identical_gemm_workspace_max_floats
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.xtdz_coalesced import column_mean_launch
from core.column_stats import (
    STATS_TPB,
    scale_in_place_kernel,
    shift_columns_kernel,
)
from decomposition.checks.jacobi_eigh import jacobi_eigh
from decomposition.checks.jacobi_eigh_device import (
    JACOBI_SWEEPS,
    JACOBI_TOL,
    JACOBI_ROT_TPB,
    jacobi_eigh_kernel,
)


# Default-off FAST experiment: compensate only centered covariance products.
# One writer per covariance cell; no atomic/split-policy race, host arithmetic,
# or input shift/restore pass. Performance admission belongs to full PCA fit.
# MEASURED M3 FAST; candidate remains OFF. Broader workload coverage pending.
# Scored FAST quality: 12/12 metrics within the existing bands; PASS.
# F11/compensated-pca M3 2026-10-06: 12 retained public-caller timings;
# B/A range 0.8987..2.1360, mixed/regressing; FAST candidate stays OFF.
# One excluded warmup/one score; caller67d0efb29; compile/identity reused.
# Scored quality metrics and per-arm build/hash provenance retained at
# ~/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F11/compensated-pca.
# No combined-switch or full-board default claim from these component cases.
# Full Istella PCA operation, M3 FAST, source ec3c8c850 (2026-10-06):
# F11/compensated-pca A=716.473 ms, B=1955.869 ms, B/A=2.72986.
# All 2,043,304 training rows and 500,000 query rows; preparation, fit,
# transform, inverse and consumed outputs included. One excluded warmup and
# one scored sample per arm; reconstruction gate passed, model/output hashes
# retained. Full taxi also measured all 5,250,086 training/500,000 query rows:
# A=107.864 ms, B=1777.204 ms, B/A=16.47638; reconstruction gate passed.
# These full-workload regressions keep the switch OFF. Downstream LLE and
# interactions remain pending; no campaign-wide claim.
# Evidence: experiments/performance_ideas/measurements/full_ab_20261006/
# pca-full-summary.json (full receipts linked by the measurement board).
comptime PCA_COMPENSATED_COV = AFN_GEMM_APPLE and is_defined["MOJOLEARN_PCA_FAST_COMPENSATED_COV"]()

# AFCL-L07: NEVER RUN — PENDING MEASUREMENT. Uncompiled/unverified, OFF.
# Requires the compensated covariance route in BOTH arms. Each partial
# centers before multiplying and uses compensation; the finish compensates
# the partial fold too. 2048-row nominal chunks expose row parallelism.
# Partial scratch is bounded by 16 MiB or one covariance matrix, whichever
# is larger; fewer chunks handle wider matrices without a shape-specific gate.
comptime AFCL_L07 = AFN_GEMM_APPLE and is_defined["MOJOLEARN_AFCL_L07"]()
comptime AFCL_PCA_PART_WORDS = 4 * 1024 * 1024


def pca_compensated_cov_part_kernel(
    x: F32Ptr, mu: F32Ptr, part: F32Ptr, rows_in: Int32,
    cols_in: Int32, chunk_rows_in: Int32,
):
    var rows = Int(rows_in)
    var cols = Int(cols_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= cols * cols:
        return
    var ch = Int(block_idx.y)
    var lo = ch * Int(chunk_rows_in)
    var hi = min(rows, lo + Int(chunk_rows_in))
    var i = cell // cols
    var j = cell % cols
    var mi = mu[i]
    var mj = mu[j]
    var total = Float32(0)
    var correction = Float32(0)
    for r in range(lo, hi):
        var product = (x[r * cols + i] - mi) * (x[r * cols + j] - mj)
        var adjusted = product - correction
        var updated = total + adjusted
        correction = (updated - total) - adjusted
        total = updated
    part[ch * cols * cols + cell] = total


def pca_compensated_cov_finish_kernel(
    part: F32Ptr, cov: F32Ptr, rows_in: Int32, cells_in: Int32, chunks_in: Int32,
):
    var cells = Int(cells_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= cells:
        return
    var total = Float32(0)
    var correction = Float32(0)
    for ch in range(Int(chunks_in)):
        var adjusted = part[ch * cells + cell] - correction
        var updated = total + adjusted
        correction = (updated - total) - adjusted
        total = updated
    cov[cell] = total / Float32(Int(rows_in) - 1)

struct PcaCovAudit(Defaultable, Movable):
    var calls: Int
    def __init__(out self):
        self.calls = 0
comptime PCA_COV_STATE = _Global[StorageType=PcaCovAudit, name="PcaCompensatedCovAudit", init_fn=PcaCovAudit.__init__]

def pca_compensated_cov_count() raises -> Int:
    return PCA_COV_STATE.get_or_create_ptr()[].calls

def pca_compensated_cov_kernel(x: F32Ptr, mu: F32Ptr, cov: F32Ptr, rows_in: Int32, cols_in: Int32):
    var rows = Int(rows_in)
    var cols = Int(cols_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= cols * cols:
        return
    var i = cell // cols
    var j = cell % cols
    var total = Float32(0)
    var correction = Float32(0)
    for r in range(rows):
        var product = (x[r * cols + i] - mu[i]) * (x[r * cols + j] - mu[j])
        var adjusted = product - correction
        var updated = total + adjusted
        correction = (updated - total) - adjusted
        total = updated
    cov[cell] = total / Float32(rows - 1)


@fieldwise_init
struct PCAResult(Movable):
    """What `pca_fit` writes back on the host side."""

    var components: List[Float64]
    var explained_var: List[Float64]
    var explained_var_ratio: List[Float64]
    var singular_vals: List[Float64]
    var noise_var: Float64


def compute_covariance(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    mut x_alias: DeviceBuffer[DType.float32],
    mut x_alias2: DeviceBuffer[DType.float32],
    mut mu: DeviceBuffer[DType.float32],
    mut cov: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_cols: Int,
    restore_input: Bool = True,
) raises:
    """Steps 1, 2, 3 and 6. The branch below must take the fused arm exactly when `gemm_tn` would take split-K for this shape, so it asks the SAME `gram_splitk_applies(m, n, k)` that `gemm_tn` asks -- one predicate, both readers, no target test of our own."""
    comptime assert PCA_COV_LEGAL, "MOJOLEARN_CLASSICAL_PCA_COV must be 4 (C04 arm) or 23 (C23 arm)"
    # MOJOLEARN_CLASSICAL_PCA_COV (lane classical-decomp, 2026-10-07; default
    # absent = the incumbent below). One switch, two named arms, each
    # replacing the incumbent's routing at every width; X is never modified.
    # NOT MEASURED.
    comptime if PCA_COV_C23:
        # =23: mean and covariance from ONE blocked read (per-leaf centering,
        # Chan merge in the binary-counter order), scaled by 1 / (n - 1).
        bm_onepass_covariance(
            ctx, mu.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
            cov.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
            x.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), n_rows, n_cols,
        )
        return
    # column_mean_kernel's value, read row-coalesced where it applies
    # (core/xtdz_coalesced.mojo::column_mean_launch).
    column_mean_launch(ctx, mu, x, n_rows, n_cols)
    var cells = n_rows * n_cols
    comptime if PCA_COV_C04:
        # =4: the centered Gram around the mean read straight from X (no
        # shift/unshift passes), leaves of contract_leaf_size(n) rows, the
        # binary-counter fold: the old C04 cell's value
        # (core/classical_centered.mojo, the host column), row-parallel.
        bm_centered_gram_panels(
            ctx, cov.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
            x.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
            mu.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
            n_rows, n_cols, contract_leaf_size(n_rows),
        )
        ctx.enqueue_function[scale_in_place_kernel](
            cov.unsafe_ptr(), Int32(n_cols * n_cols), Float32(1.0) / Float32(n_rows - 1),
            grid_dim=((n_cols * n_cols + 255) // 256, 1, 1), block_dim=(256, 1, 1),
        )
        ctx.synchronize()
        return
    comptime if PCA_COMPENSATED_COV:
        PCA_COV_STATE.get_or_create_ptr()[].calls += 1
        comptime if AFCL_L07:
            var cov_cells = n_cols * n_cols
            var max_chunks = max(1, AFCL_PCA_PART_WORDS // max(cov_cells, 1))
            var chunks = min(max(1, (n_rows + 2047) // 2048), max_chunks)
            var chunk_rows = (n_rows + chunks - 1) // chunks
            var part = ctx.enqueue_create_buffer[DType.float32](chunks * cov_cells)
            ctx.enqueue_function[pca_compensated_cov_part_kernel](
                x.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
                mu.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
                part.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
                Int32(n_rows), Int32(n_cols), Int32(chunk_rows),
                grid_dim=((cov_cells + 255) // 256, chunks, 1), block_dim=(256, 1, 1),
            )
            ctx.enqueue_function[pca_compensated_cov_finish_kernel](
                part.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
                cov.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
                Int32(n_rows), Int32(cov_cells), Int32(chunks),
                grid_dim=((cov_cells + 255) // 256, 1, 1), block_dim=(256, 1, 1),
            )
            ctx.synchronize()
            _ = part^
            return
        ctx.enqueue_function[pca_compensated_cov_kernel](
            x.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
            mu.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
            cov.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), Int32(n_rows), Int32(n_cols),
            grid_dim=((n_cols * n_cols + 255) // 256, 1, 1), block_dim=(256, 1, 1),
        )
        ctx.synchronize()
        return
    comptime if PCA_FAST_GRAM_MMA:
        if not gram_splitk_applies(n_cols, n_cols, n_rows):
            ctx.enqueue_function[shift_columns_kernel](
                x.unsafe_ptr(),
                mu.unsafe_ptr(),
                Int32(n_rows),
                Int32(n_cols),
                Float32(-1.0),
                grid_dim=((cells + 255) // 256, 1, 1),
                block_dim=(256, 1, 1),
            )
            var mm = n_cols * n_cols
            var cp = cov.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
            var xp = x.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
            ctx.enqueue_function[afn_zero_kernel](
                cp, Int32(mm),
                grid_dim=((mm + 4 * AFN_ZERO_TPB - 1) // (4 * AFN_ZERO_TPB), 1, 1),
                block_dim=(AFN_ZERO_TPB, 1, 1),
            )
            var tiles = ((n_cols + 63) // 64) * ((n_cols + 63) // 64)
            var splits = max(1, PCA_GRAM_BLOCK_TARGET // tiles)
            var per = (n_rows + splits - 1) // splits
            per = ((per + AFN_GEMM_KB - 1) // AFN_GEMM_KB) * AFN_GEMM_KB
            splits = (n_rows + per - 1) // per
            # Opt-in scoped G1 fit remains HOLD (2026-10-04), source201fe736,
            # scoped-pca-fit-istella-q-v1: singular/noise oracle errors worsen
            # under zero allowance. Report serialization incomplete; no PASS
            # or timing admission. Existing accepted AFN default is fallback.
            # PCA transform/inverse evidence uses another route/contract;
            # see docs/apple-fast/GEMM_INLINE_OUTCOMES.md before any retry.
            if not try_scoped_gemm[True, 2](
                ctx, cp, xp, xp, n_cols, n_cols, n_rows, 1, n_cols, n_cols, 1, splits, per,
            ):
                afn_launch_tile[DType.float32, DType.float32, True, AFN_EPI_NONE](
                    ctx, AFN_TILE_SQUARE, cp, xp, xp, cp, cp,
                    n_cols, n_cols, n_rows, afn_strides(OP_TN, n_cols, n_cols, n_rows), splits, per,
                )
            ctx.enqueue_function[scale_in_place_kernel](
                cov.unsafe_ptr(),
                Int32(n_cols * n_cols),
                Float32(1.0) / Float32(n_rows - 1),
                grid_dim=((n_cols * n_cols + 255) // 256, 1, 1),
                block_dim=(256, 1, 1),
            )
            if restore_input:
                ctx.enqueue_function[shift_columns_kernel](
                    x.unsafe_ptr(),
                    mu.unsafe_ptr(),
                    Int32(n_rows),
                    Int32(n_cols),
                    Float32(1.0),
                    grid_dim=((cells + 255) // 256, 1, 1),
                    block_dim=(256, 1, 1),
                )
            ctx.synchronize()
            return
    var fused = gram_splitk_applies(n_cols, n_cols, n_rows)
    if fused:
        gram_centered_splitk_into(
            ctx, cov, x, mu, x_alias, n_cols, n_rows
        )
    else:
        ctx.enqueue_function[shift_columns_kernel](
            x.unsafe_ptr(),
            mu.unsafe_ptr(),
            Int32(n_rows),
            Int32(n_cols),
            Float32(-1.0),
            grid_dim=((cells + 255) // 256, 1, 1),
            block_dim=(256, 1, 1),
        )
        gemm_tn(ctx, cov, x, x_alias, x_alias2, n_cols, n_cols, n_rows)
    ctx.enqueue_function[scale_in_place_kernel](
        cov.unsafe_ptr(),
        Int32(n_cols * n_cols),
        Float32(1.0) / Float32(n_rows - 1),
        grid_dim=((n_cols * n_cols + 255) // 256, 1, 1),
        block_dim=(256, 1, 1),
    )
    if restore_input and not fused:
        ctx.enqueue_function[shift_columns_kernel](
            x.unsafe_ptr(),
            mu.unsafe_ptr(),
            Int32(n_rows),
            Int32(n_cols),
            Float32(1.0),
            grid_dim=((cells + 255) // 256, 1, 1),
            block_dim=(256, 1, 1),
        )
    ctx.synchronize()


def pca_validate(n_rows: Int, n_cols: Int, n_components: Int) raises:
    """The four shape refusals `pca_fit` makes first (cuML's `pcaFit` asserts, in their order)."""
    if n_cols <= 1:
        raise Error("Parameter n_cols: number of columns cannot be less than two")
    if n_rows <= 1:
        raise Error("Parameter n_rows: number of rows cannot be less than two")
    if n_components <= 0:
        raise Error(
            "Parameter n_components: number of components cannot be less than one"
        )
    if n_components > n_cols:
        raise Error("n_components cannot exceed n_cols")


#: P2, lane fg-pca (2026-10-09), IDENTICAL default; rollback
#: `-D MOJOLEARN_IDN_PCA_LEAN_SCRATCH_OFF`. `pca_fit_host` uploads X into a
#: device buffer that is the fit's own and nothing reads after the Gram, so
#: (a) the restore pass (`shift_columns_kernel` +mu: one n*d read and one
#: n*d write, 2 x 1.8 GB at 2M x 220) is dead and is skipped, and (b) the two
#: n*d alias buffers are sized to what the route below actually reads
#: (`pca_cov_scratch_floats`): the split-K centered Gram reads only `x_alias`
#: as its n_chunks*d*d partials (when `gram_splitk_scratch_covers`), the v1
#: OP_TN Gram past the split-K width reads only `x_alias2` as the
#: identical_gemm workspace (when that fits the old k*m contract); every other
#: case allocated its own workspace before and still does. Cost reasoning:
#: 2 x n*d floats of allocation and 2 x n*d words of traffic per fit at any
#: width, against a workspace of at most n_chunks*d*d or the plan's own size.
#: Bits: none (the words the Gram and the eigensolver read are unchanged;
#: the restored copy was never read). IDENTICAL only: the FAST NVIDIA/AMD
#: route (`gemm_tn_via_transpose`) reads both alias buffers at full size.
comptime PCA_LEAN_SCRATCH = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not is_defined["MOJOLEARN_IDN_PCA_LEAN_SCRATCH_OFF"]()
)


def pca_cov_scratch_floats(n_rows: Int, n_cols: Int) raises -> Tuple[Int, Int]:
    """(x_alias, x_alias2) float counts `compute_covariance` reads on the
    IDENTICAL default route for this shape; 1 for a buffer the route never
    reads. Mirrors the two predicates the route itself asks
    (`gram_splitk_applies`, then the scratch-cover tests in
    `gram_centered_splitk_into` / `gemm_tn_identical_v1`), so a buffer is
    either big enough for the route's reuse branch or unread."""
    var k = n_rows
    var m = n_cols
    if gram_splitk_applies(m, m, k):
        if gram_splitk_scratch_covers(m, k):
            return (max(1, gram_splitk_chunk_count() * m * m), 1)
        return (1, 1)
    var need = identical_gemm_workspace_max_floats(m, m, k)
    if need <= k * m:
        # gemm_tn_identical_v1 tests `need <= k * m` against the buffer's
        # nominal k*m contract; a buffer of `need` floats serves that branch
        return (1, max(1, need))
    return (1, 1)


def pca_fit(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    mut x_alias: DeviceBuffer[DType.float32],
    mut x_alias2: DeviceBuffer[DType.float32],
    mut mu: DeviceBuffer[DType.float32],
    mut cov: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_cols: Int,
    n_components: Int,
    restore_input: Bool = True,
) raises -> PCAResult:
    """`pca_fit`, all six steps."""
    pca_validate(n_rows, n_cols, n_components)

    compute_covariance(
        ctx, x, x_alias, x_alias2, mu, cov, n_rows, n_cols, restore_input
    )

    return eig_and_truncate(
        ctx, cov, n_cols, n_components, n_rows - 1
    )


comptime SIGNFLIP_TPB = 32


def sign_flip_kernel(
    v: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """Reference: `signFlipKernel`, `raft/matrix/detail/math.cuh:367`. `decomposition/checks/pca_check.mojo` holds it to that: the device answer must equal a fold-free host scan BITWISE, and the tie, the zero cases and the NaN case are each planted rather than hoped for."""
    var n = Int(n_in)
    var col = Int(block_idx.x)
    var tid = Int(thread_idx.x)

    var sh = stack_allocation[
        3,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()

    var local = Float32(0.0)
    var f = tid
    while f < n:
        var m = abs(v.unsafe_load(f * n + col))
        if m > local:
            local = m
        f += SIGNFLIP_TPB
    var reduced_max = block_max[block_size=SIGNFLIP_TPB](local)
    if tid == 0:
        sh[0] = reduced_max
    barrier()
    var biggest = sh[0]

    var cand = Float32(n)
    f = tid
    while f < n:
        var fv = Float32(f)
        if abs(v.unsafe_load(f * n + col)) == biggest:
            if fv < cand:
                cand = fv
        f += SIGNFLIP_TPB
    var reduced_first = block_min[block_size=SIGNFLIP_TPB](cand)
    if tid == 0:
        sh[1] = reduced_first
        sh[2] = Float32(1.0)
    barrier()
    var first = sh[1]

    f = tid
    while f < n:
        if Float32(f) == first:
            if v.unsafe_load(f * n + col) < Float32(0.0):
                sh[2] = Float32(-1.0)
            else:
                sh[2] = Float32(1.0)
        f += SIGNFLIP_TPB
    barrier()

    if sh[2] < Float32(0.0):
        f = tid
        while f < n:
            v.unsafe_store(f * n + col, -v.unsafe_load(f * n + col))
            f += SIGNFLIP_TPB


def order_truncate_spectrum(
    diag: List[Float64],
    vecs: List[Float64],
    n_cols: Int,
    n_components: Int,
    singular_scale: Int,
    spectrum_count: Int = 0,
) raises -> PCAResult:
    """`colReverse` + `truncCompExpVars`: the HOST tail of a fit, shared."""

    var count = n_cols if spectrum_count == 0 else spectrum_count
    var order = List[Int]()
    for i in range(count):
        order.append(i)
    for i in range(count):
        for j in range(i + 1, count):
            if diag[order[j]] > diag[order[i]]:
                var t = order[i]
                order[i] = order[j]
                order[j] = t

    var total = 0.0
    for i in range(count):
        total += diag[i]

    var components = List[Float64]()
    var explained_var = List[Float64]()
    var explained_var_ratio = List[Float64]()
    var singular_vals = List[Float64]()

    for c in range(n_components):
        var src = order[c]
        var lam = diag[src]

        for f in range(n_cols):
            components.append(vecs[f * n_cols + src])
        explained_var.append(lam)
        explained_var_ratio.append(lam / total if total != 0.0 else 0.0)
        singular_vals.append(sqrt(lam * Float64(singular_scale)))

    var noise = 0.0
    if n_components < count and n_components <= singular_scale:
        for c in range(n_components, count):
            noise += diag[order[c]]
        noise /= Float64(count - n_components)

    return PCAResult(
        components^, explained_var^, explained_var_ratio^, singular_vals^, noise
    )


#: rounds between synchronizes of the round-robin solve (x_decomp/device.mojo
#: PJ_SYNC_ROUNDS: a bound on the enqueued launches, no bit).
comptime PCA_RR_SYNC_ROUNDS = 512
#: slots of the round-robin solve's device state (`pca_rr_gate_kernel`)
comptime PCA_RR_STATE = 6


def pca_rr_gate_kernel(fold: F32Ptr, state: F32Ptr, tol: Float32):
    """The round-robin solve's convergence decision, on the device (no
    readback between sweeps): from the folded test sums fold = (off, diag,
    ran mark) (`eigh_par_off_fold_kernel`), state[0] = 1 once `rr_converged`
    holds (sticky: every later round is then a no-op and the state stays
    the converged test's), state[1] = the off-diagonal sum, state[2] = the
    first test's ||A||_F^2 (the caller fills -1), state[3] = this test's,
    state[4] = -1 when a block of the test did not run, state[5] = sweeps
    started. `host_eigh_rr` (x_decomp/rr.mojo) decides the same from the
    same sums. One thread."""
    if Int(block_idx.x) == 0 and Int(thread_idx.x) == 0:
        # x_decomp/rr.mojo `rr_gate_state`: these statements, moved there
        # unchanged so the one-block sweep (PCA_RR_ONE_BLOCK) runs the same
        rr_gate_state(fold.unsafe_load(0), fold.unsafe_load(1), fold.unsafe_load(2), state, tol)


def pca_rr_cs_kernel(a: F32Ptr, cs: F32Ptr, state: F32Ptr, n_in: Int32, m_in: Int32, round_in: Int32):
    """`eigh_par_cs_kernel` (x_decomp/jacobi_par.mojo) behind the device's
    convergence flag: pair b's rotation (`rr_cs`), nothing once converged."""
    var m = Int(m_in)
    var b = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if state.unsafe_load(0) == Float32(0.0) and b < m // 2:
        var got = rr_cs(a, Int(n_in), m, Int(round_in), b)
        cs.unsafe_store(2 * b, got[0])
        cs.unsafe_store(2 * b + 1, got[1])


def pca_rr_update_kernel(
    a: F32Ptr, v: F32Ptr, cs: F32Ptr, state: F32Ptr, n_in: Int32, m_in: Int32, round_in: Int32
):
    """`eigh_par_update_kernel` behind the device's convergence flag: the 2
    x 2 blocks (`rr_block`) then V's (row, pair) (`rr_vrow`), nothing once
    converged."""
    var n = Int(n_in)
    var m = Int(m_in)
    var h = m // 2
    var r = Int(round_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if state.unsafe_load(0) == Float32(0.0):
        if t < h * h:
            var i = t // h
            var j = t - i * h
            if i <= j:
                rr_block(a, cs, n, m, r, i, j)
        elif t < h * h + n * h:
            var u = t - h * h
            var k = u // h
            rr_vrow(v, cs, n, m, r, k, u - k * h)


def pca_rr_finish_kernel(state: F32Ptr, info: F32Ptr):
    """`jacobi_eigh_kernel`'s info slots from the round-robin state: info[0]
    = 1 converged (and ||A||_F kept, `rr_fro_kept`), 0 not, -1 a test block
    did not run; info[1] = the last off-diagonal sum; info[2] = sweeps
    started. One thread."""
    if Int(block_idx.x) == 0 and Int(thread_idx.x) == 0:
        var ok = Float32(0.0)
        if state.unsafe_load(0) == Float32(1.0) and rr_fro_kept(state.unsafe_load(2), state.unsafe_load(3)):
            ok = Float32(1.0)
        if state.unsafe_load(4) < Float32(0.0):
            ok = Float32(-1.0)
        info.unsafe_store(0, ok)
        info.unsafe_store(1, state.unsafe_load(1))
        info.unsafe_store(2, state.unsafe_load(5))


def _pca_rr_done(
    ctx: DeviceContext, mut dstate: DeviceBuffer[DType.float32], mut hstate: HostBuffer[DType.float32]
) raises -> Bool:
    """PCA_RR_FLAG_TEST: reads the device's verdict of the test just enqueued
    (`pca_rr_gate_kernel`'s PCA_RR_STATE flag words, the only words that
    cross; x_decomp/device.mojo `_eigh_par_test` reads its three the same
    way) and returns True when the solve is over: converged (state[0]), or a
    test block did not run (state[4]). The decision itself was made on the
    device; the host only stops enqueuing rounds that would be no-ops."""
    ctx.enqueue_copy(dst_ptr=hstate.unsafe_ptr(), src_buf=dstate)
    ctx.synchronize()
    return hstate.unsafe_ptr().unsafe_load(0) == Float32(1.0) or hstate.unsafe_ptr().unsafe_load(4) < Float32(0.0)


def _eig_rr_device(
    ctx: DeviceContext,
    mut cov: DeviceBuffer[DType.float32],
    mut vec_buf: DeviceBuffer[DType.float32],
    mut info_buf: DeviceBuffer[DType.float32],
    n: Int,
) raises:
    """PCA_RR_EIGH: the round-robin two-sided Jacobi on `cov` in place (its
    diagonal the eigenvalues), the vectors in the columns of `vec_buf`, the
    outcome in `info_buf` (`pca_rr_finish_kernel`). The rounds are
    x_decomp/rr.mojo's (`rr_cs`, `rr_block`, `rr_vrow`), every rotation of a
    round in parallel; the convergence test before each sweep is folded AND
    decided on the device (`pca_rr_gate_kernel`). PCA_RR_FLAG_TEST (lane
    idn-all): the host reads the device's PCA_RR_STATE flag words after each
    test (no matrix data crosses) and stops enqueuing at the converged test,
    so a solve costs its own sweeps, not the budget's; without it every
    budgeted sweep is enqueued up front (no-ops once converged). The host
    column `host_eigh_rr` runs the same rounds and the same test."""
    var m = n + (n % 2)
    var h = m // 2
    var nb = max((n + RR_OFF_TPB - 1) // RR_OFF_TPB, 1)
    var dcs = ctx.enqueue_create_buffer[DType.float32](2 * h)
    var doff = ctx.enqueue_create_buffer[DType.float32](3 * n)
    var dpart = ctx.enqueue_create_buffer[DType.float32](3 * nb)
    var dfold = ctx.enqueue_create_buffer[DType.float32](3)
    var dstate = ctx.enqueue_create_buffer[DType.float32](PCA_RR_STATE)
    enqueue_fill(ctx, info_buf, Float32(-1.0))
    enqueue_fill(ctx, dpart, Float32(-1.0))
    enqueue_fill(ctx, dfold, Float32(-1.0))
    enqueue_fill(ctx, dstate, Float32(0.0))
    ctx.enqueue_function[pca_rr_state_init_kernel](dstate.unsafe_ptr(), grid_dim=1, block_dim=1)
    ctx.enqueue_function[pj_identity_kernel](
        vec_buf.unsafe_ptr(), Int32(n), grid_dim=(n * n + PJ_TPB - 1) // PJ_TPB, block_dim=PJ_TPB
    )
    var hstate = ctx.enqueue_create_host_buffer[DType.float32](PCA_RR_STATE)
    var launched = 0
    var one_block = False
    comptime if PCA_RR_ONE_BLOCK:
        one_block = rr_one_block_applies(n)
    if one_block:
        # P1 + P1b (MOJOLEARN_IDN_PCA_RR_ONE_BLOCK): every sweep, its test,
        # gate and rounds in one launch of one block; the same cells, the
        # same order, the same state words as the loop below
        ctx.enqueue_function[rr_eigh_one_block_kernel](  # small-launch(n: n bounded by rr_one_block_applies, h h + n h <= RR_ONE_BLOCK_STEPS x RR_ONE_TPB cells a round): 1024 threads share every round's cells in parallel, default off, wider n keeps the grid launches
            cov.unsafe_ptr(), vec_buf.unsafe_ptr(), dcs.unsafe_ptr(), doff.unsafe_ptr(),
            dpart.unsafe_ptr(), dfold.unsafe_ptr(), dstate.unsafe_ptr(),
            Int32(n), Int32(PCA_RR_SWEEPS), Float32(JACOBI_TOL),
            grid_dim=1, block_dim=RR_ONE_TPB,
        )
    for sweep in range(PCA_RR_SWEEPS + 1):
        if one_block:
            break
        ctx.enqueue_function[eigh_par_off_part_kernel](
            cov.unsafe_ptr(), doff.unsafe_ptr(), dpart.unsafe_ptr(), Int32(n),
            grid_dim=nb, block_dim=RR_OFF_TPB,
        )
        ctx.enqueue_function[eigh_par_off_fold_kernel](
            dpart.unsafe_ptr(), dfold.unsafe_ptr(), Int32(nb), grid_dim=1, block_dim=RR_OFF_TPB
        )
        ctx.enqueue_function[pca_rr_gate_kernel](
            dfold.unsafe_ptr(), dstate.unsafe_ptr(), Float32(JACOBI_TOL), grid_dim=1, block_dim=1
        )
        comptime if PCA_RR_FLAG_TEST:
            # the device's verdict: converged, or a test block that did not
            # run, ends the enqueuing here
            if _pca_rr_done(ctx, dstate, hstate):
                break
        if sweep < PCA_RR_SWEEPS:
            for rd in range(m - 1):
                ctx.enqueue_function[pca_rr_cs_kernel](
                    cov.unsafe_ptr(), dcs.unsafe_ptr(), dstate.unsafe_ptr(), Int32(n), Int32(m), Int32(rd),
                    grid_dim=(h + PJ_TPB - 1) // PJ_TPB, block_dim=PJ_TPB,
                )
                ctx.enqueue_function[pca_rr_update_kernel](
                    cov.unsafe_ptr(), vec_buf.unsafe_ptr(), dcs.unsafe_ptr(), dstate.unsafe_ptr(),
                    Int32(n), Int32(m), Int32(rd),
                    grid_dim=(h * h + n * h + PJ_TPB - 1) // PJ_TPB, block_dim=PJ_TPB,
                )
                launched += 1
                if launched % PCA_RR_SYNC_ROUNDS == 0:
                    ctx.synchronize()
    ctx.enqueue_function[pca_rr_finish_kernel](
        dstate.unsafe_ptr(), info_buf.unsafe_ptr(), grid_dim=1, block_dim=1
    )
    ctx.synchronize()
    _ = hstate^
    _ = dcs^
    _ = doff^
    _ = dpart^
    _ = dfold^
    _ = dstate^


def pca_rr_state_init_kernel(state: F32Ptr):
    """state[2] = -1: no test has run yet (`pca_rr_gate_kernel`)."""
    if Int(block_idx.x) == 0 and Int(thread_idx.x) == 0:
        state.unsafe_store(2, Float32(-1.0))


def eig_and_truncate(
    ctx: DeviceContext,
    mut cov: DeviceBuffer[DType.float32],
    n_cols: Int,
    n_components: Int,
    singular_scale: Int,
) raises -> PCAResult:
    """`calEig` + `truncCompExpVars`, shared by PCA and truncated SVD."""
    var vec_buf = ctx.enqueue_create_buffer[DType.float32](n_cols * n_cols)
    var info_buf = ctx.enqueue_create_buffer[DType.float32](3)
    # lane idn-cov-overflow: the power-of-two range scale of
    # x_decomp/eigh_scale.mojo (no write while max |cov| is in
    # [2^-33, 2^32)), so the Jacobi's folded squares stay finite (an
    # uncentered TruncatedSVD Gram overflowed them); the diagonal is
    # unscaled after the solve. `host_eig_and_truncate` takes the same.
    var cov_p = F32Ptr(unsafe_from_address=Int(cov.unsafe_ptr()))
    var dfac = enqueue_es_scale(ctx, cov_p, 1, n_cols)
    ctx.synchronize()
    comptime if PCA_RR_EIGH:
        # the round-robin rounds (every rotation of a round in parallel),
        # converged or not decided on the device into the info slots
        _eig_rr_device(ctx, cov, vec_buf, info_buf, n_cols)
    else:
        ctx.enqueue_function[jacobi_eigh_kernel[JACOBI_ROT_TPB]](
            cov.unsafe_ptr(),
            vec_buf.unsafe_ptr(),
            info_buf.unsafe_ptr(),
            Int32(n_cols),
            Int32(JACOBI_SWEEPS),
            Float32(JACOBI_TOL),
            grid_dim=(1, 1, 1),
            block_dim=(JACOBI_ROT_TPB, 1, 1),
        )

    enqueue_es_unscale_diag(ctx, cov_p, dfac, n_cols)
    ctx.enqueue_function[sign_flip_kernel](
        vec_buf.unsafe_ptr(),
        Int32(n_cols),
        grid_dim=(n_cols, 1, 1),
        block_dim=(SIGNFLIP_TPB, 1, 1),
    )
    ctx.synchronize()
    _ = dfac^

    var h_cov = ctx.enqueue_create_host_buffer[DType.float32](n_cols * n_cols)
    var h_vec = ctx.enqueue_create_host_buffer[DType.float32](n_cols * n_cols)
    var h_info = ctx.enqueue_create_host_buffer[DType.float32](3)
    ctx.enqueue_copy(dst_ptr=h_cov.unsafe_ptr(), src_buf=cov)
    ctx.enqueue_copy(dst_ptr=h_vec.unsafe_ptr(), src_buf=vec_buf)
    ctx.enqueue_copy(dst_ptr=h_info.unsafe_ptr(), src_buf=info_buf)
    ctx.synchronize()

    comptime if PCA_RR_EIGH:
        if h_info.unsafe_ptr().unsafe_load(0) < Float32(0.0):
            raise Error(
                "the device Jacobi's convergence test did not run at n_cols = "
                + String(n_cols)
                + " (a block's mark is still -1): a launch failure, not a"
                " convergence failure. Check that the binding is built for this device."
            )
        # non-convergence is the refusal of the cyclic solver this replaced:
        # the same error, in its words, on the device and the host column
        if h_info.unsafe_ptr().unsafe_load(0) != Float32(1.0):
            raise Error(
                "the device Jacobi did not converge in "
                + String(PCA_RR_SWEEPS)
                + " sweeps at n_cols = "
                + String(n_cols)
                + " (round-robin order; off-diagonal mass "
                + String(h_info.unsafe_ptr().unsafe_load(1))
                + ", or ||A||_F moved) against a tolerance of "
                + String(JACOBI_TOL)
                + ". cuSOLVER's syevj has the same failure mode and the same"
                " remedy, which is more sweeps. A non-symmetric covariance"
                " produces this too; see check_covariance_is_symmetric."
            )
    if h_info.unsafe_ptr().unsafe_load(0) == Float32(0.0):
        raise Error(
            "the device Jacobi did not converge in "
            + String(JACOBI_SWEEPS)
            + " sweeps at n_cols = "
            + String(n_cols)
            + ": ||offdiag(A)||_F / ||A||_F is still "
            + String(h_info.unsafe_ptr().unsafe_load(1))
            + " against a tolerance of "
            + String(JACOBI_TOL)
            + ". cuSOLVER's syevj has the same failure mode and the same"
            " remedy, which is more sweeps. A non-symmetric covariance"
            " produces this too; see check_covariance_is_symmetric."
        )

    var diag = List[Float64]()
    for i in range(n_cols):
        diag.append(Float64(h_cov.unsafe_ptr().unsafe_load(i * n_cols + i)))
    var vecs = List[Float64]()
    for i in range(n_cols * n_cols):
        vecs.append(Float64(h_vec.unsafe_ptr().unsafe_load(i)))

    return order_truncate_spectrum(
        diag, vecs, n_cols, n_components, singular_scale
    )


def pca_transform(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    mut mu: DeviceBuffer[DType.float32],
    mut components: DeviceBuffer[DType.float32],
    mut output: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_cols: Int,
    n_components: Int,
) raises:
    """`pca_transform`: center, then project onto the components."""
    var cells = n_rows * n_cols
    ctx.enqueue_function[shift_columns_kernel](
        x.unsafe_ptr(),
        mu.unsafe_ptr(),
        Int32(n_rows),
        Int32(n_cols),
        Float32(-1.0),
        grid_dim=((cells + 255) // 256, 1, 1),
        block_dim=(256, 1, 1),
    )
    gemm_nt(
        ctx,
        output,
        x,
        components,
        n_rows,
        n_components,
        n_cols,
    )
    ctx.enqueue_function[shift_columns_kernel](
        x.unsafe_ptr(),
        mu.unsafe_ptr(),
        Int32(n_rows),
        Int32(n_cols),
        Float32(1.0),
        grid_dim=((cells + 255) // 256, 1, 1),
        block_dim=(256, 1, 1),
    )
    ctx.synchronize()




comptime WHITEN_SKIP_ZERO = 1.0e-10


def whiten_scalar(n_fit_rows: Int, inverse: Bool) -> Float32:
    """`sqrt(n_fit_rows - 1)` forward, `1 / sqrt(n_fit_rows - 1)` inverse. `pca_validate` already refuses `n_rows <= 1` so the guard cannot fire on a fitted model; it is kept because dropping a guard that the reference wrote is a silent change of behavior on the one input it was written for."""
    var d = Float64(n_fit_rows - 1)
    if d <= 0.0:
        return Float32(0.0)
    var r = sqrt(d)
    if inverse:
        return Float32(1.0 / r)
    return Float32(r)


def whiten_scale_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin],
    singular: MutPointer[Float32, MutAnyOrigin],
    n_cells_in: Int32,
    n_cols_in: Int32,
    scalar_in: Float32,
    divide_in: Int32,
):
    """`dst = whiten(src)`: cuML's two passes over the components copy, fused."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_cells_in):
        var c = i // Int(n_cols_in)
        var s = singular.unsafe_load(c)
        var v = ftz(identical_mul(src.unsafe_load(i), scalar_in))
        if abs(s) < Float32(WHITEN_SKIP_ZERO):
            dst.unsafe_store(i, v)
        elif divide_in != Int32(0):
            dst.unsafe_store(i, ftz(identical_div(v, s)))
        else:
            dst.unsafe_store(i, ftz(identical_mul(v, s)))


def whiten_components(
    ctx: DeviceContext,
    mut src: DeviceBuffer[DType.float32],
    mut dst: DeviceBuffer[DType.float32],
    mut singular: DeviceBuffer[DType.float32],
    n_components: Int,
    n_cols: Int,
    n_fit_rows: Int,
    inverse: Bool,
) raises:
    """Launch `whiten_scale_kernel` over the whole components matrix. `dst` is a separate buffer because cuML makes a copy too (`rmm::device_uvector<math_t> components_copy` at `pca.cuh:229` and `:289`): the caller's `components_` must survive the transform unchanged, the same contract `check_input_restored` holds the fit to."""
    var cells = n_components * n_cols
    var scalar = whiten_scalar(n_fit_rows, inverse)
    var divide = Int32(1)
    if inverse:
        divide = Int32(0)
    ctx.enqueue_function[whiten_scale_kernel](
        dst.unsafe_ptr(),
        src.unsafe_ptr(),
        singular.unsafe_ptr(),
        Int32(cells),
        Int32(n_cols),
        scalar,
        divide,
        grid_dim=((cells + 255) // 256, 1, 1),
        block_dim=(256, 1, 1),
    )
    ctx.synchronize()
