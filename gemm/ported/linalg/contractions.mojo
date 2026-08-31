# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Derivative work: the upstream file and its pinned commit are recorded in this lane's PORTED_MAP.tsv and in this file's own docstring. See NOTICE.
"""`raft/linalg/contractions.cuh`: the contraction tile policy. **COPY, DO NOT
IMPROVE.**

Upstream: RAFT at `661a3b8`, `cpp/include/raft/linalg/contractions.cuh`
(`KernelPolicy`, `ColKernelPolicy`, `Policy4x4`, `Policy4x4Skinny`,
`Policy2x8`). The load/accumulate machinery those policies parameterize is
`cpp/include/raft/linalg/detail/contractions.cuh`, and the loop that drives
it is `cpp/include/raft/distance/detail/pairwise_distance_base.cuh:128-152`.

WHY THIS FILE EXISTS SEPARATELY FROM THE CONSTANTS IN `core/gemm.mojo`
----------------------------------------------------------------------
`core/gemm.mojo` carries `Policy4x4<float, 4>` FLATTENED into seven
`comptime` integers, because the one kernel instantiated at that policy
(`cluster/gbdt/distance/fused_distance_nn/simt_kernel.mojo`) needs exactly
that instantiation and nothing else. Those constants are correct and this
file does not replace them; it transcribes the POLICY ITSELF, parameterized,
so a second instantiation (the skinny policy for small `k`, the column-major
policy, the `Policy2x8` shape) is a type argument rather than a second set of
hand-copied integers.

WHAT THE POLICY IS AND IS NOT, IN THIS LANE'S TERMS
----------------------------------------------------
Every field below is an EXECUTION PLAN quantity in the sense of
`IDENTICAL_GEMM_PLAN.md`'s table: tile sizes, thread counts, how many loads
each thread issues, and the shared-memory page layout. **None of it is a
numerical plan quantity**, and that is a statement about RAFT's kernel, not a
hope about ours: their main loop walks `kidx` from `0` to `k` ASCENDING in
steps of `Kblk`, and inside each `Kblk` walks `ki` from `0` to `Kblk`
ascending in steps of `Veclen`, with ONE block owning the entire `k` range for
its output tile (`pairwise_distance_base.cuh:139-149`, `:223-241`). There is
no split-K and no cross-block combination anywhere in it. So changing a
policy changes WHICH thread accumulates a cell and in what register, never
the SEQUENCE of values accumulated into it.

That is the property `gemm/IDENTICAL_FP32_CONTRACT.md` requires of an
execution plan, and RAFT's own contraction already has it. It is the reason
the contract's k ordering is stated as "ascending, one leaf at a time": it
mirrors upstream rather than inventing a rule.

`SmemStride = Kblk + Veclen` is theirs and is not a rounding: the padding
staggers each row's start so threads reading down a column of shared memory
do not all land in the same bank.

NOT PORTED FROM THIS HEADER: nothing. `contractions.cuh` is the policy
structs and the three policy families; `detail/contractions.cuh`'s loader is
`UNPORTED.tsv`'s first row and lives, for the one instantiation this
repository needs, in `simt_kernel.mojo`.
"""


struct KernelPolicy[
    VECLEN: Int,
    KBLK: Int,
    RPT: Int,
    CPT: Int,
    TR: Int,
    TC: Int,
    DBYTES: Int = 4,
](Copyable, Movable):
    """`template <typename DataT, int _veclen, int _kblk, int _rpt, int _cpt,
    int _tr, int _tc> struct KernelPolicy`, `:62-103`. The ROW-MAJOR policy.

    `DataT` is not a Mojo type parameter here because this repository is
    FP32 on device throughout (`mojo_only/hardware_matrix.mojo`: no float64
    on Metal); `DBYTES` carries the one place the element type appears, in
    `SmemSize`.

    Their parameter documentation, kept verbatim in substance:

    - `_veclen`: k-elements loaded by each thread per LDG. Configure from `k`
      and the data type; for float and `k` a multiple of 4, 4 gives the best
      LDG pattern. Possible values {1, 2, 4}.
    - `_kblk`: k-elements per main-loop iteration, so `ceil(k/_kblk)`
      iterations. Must be a multiple of `_veclen`. Bigger costs shared memory.
    - `_rpt`: rows a thread accumulates on. Register pressure, and it sets
      the block's m extent.
    - `_cpt`: cols a thread accumulates on. Register pressure, and it sets
      the block's n extent.
    - `_tr`: threads working the same output column.
    - `_tc`: threads working the same output row.
    """

    #: `Kblk = _kblk` -- elements along K per main loop iteration.
    comptime kblk = Self.KBLK
    #: `Veclen = _veclen` -- elements loaded per LDG.
    comptime veclen = Self.VECLEN
    #: `AccRowsPerTh = _rpt`.
    comptime acc_rows_per_th = Self.RPT
    #: `AccColsPerTh = _cpt`.
    comptime acc_cols_per_th = Self.CPT
    #: `AccThRows = _tr`.
    comptime acc_th_rows = Self.TR
    #: `AccThCols = _tc`.
    comptime acc_th_cols = Self.TC
    #: `Nthreads = AccThRows * AccThCols`.
    comptime nthreads = Self.TR * Self.TC
    #: `Mblk = AccRowsPerTh * AccThRows` -- output tile size along rows.
    comptime mblk = Self.RPT * Self.TR
    #: `Nblk = AccColsPerTh * AccThCols` -- output tile size along cols.
    comptime nblk = Self.CPT * Self.TC
    #: `LdgThRow = Kblk / Veclen` -- threads loading a single row.
    comptime ldg_th_row = Self.KBLK // Self.VECLEN
    #: `LdgPerThX = Mblk * LdgThRow / Nthreads`.
    comptime ldg_per_th_x = (
        (Self.RPT * Self.TR) * (Self.KBLK // Self.VECLEN)
    ) // (Self.TR * Self.TC)
    #: `LdgPerThY = Nblk * LdgThRow / Nthreads`.
    comptime ldg_per_th_y = (
        (Self.CPT * Self.TC) * (Self.KBLK // Self.VECLEN)
    ) // (Self.TR * Self.TC)
    #: `LdgRowsX = Mblk / LdgPerThX`.
    comptime ldg_rows_x = (Self.RPT * Self.TR) // (
        ((Self.RPT * Self.TR) * (Self.KBLK // Self.VECLEN))
        // (Self.TR * Self.TC)
    )
    #: `LdgRowsY = Nblk / LdgPerThY`.
    comptime ldg_rows_y = (Self.CPT * Self.TC) // (
        ((Self.CPT * Self.TC) * (Self.KBLK // Self.VECLEN))
        // (Self.TR * Self.TC)
    )
    #: `SmemStride = Kblk + Veclen`. THEIRS, and it is bank-conflict padding.
    comptime smem_stride = Self.KBLK + Self.VECLEN
    #: `SmemPageX = SmemStride * Mblk`.
    comptime smem_page_x = (Self.KBLK + Self.VECLEN) * (Self.RPT * Self.TR)
    #: `SmemPageY = SmemStride * Nblk`.
    comptime smem_page_y = (Self.KBLK + Self.VECLEN) * (Self.CPT * Self.TC)
    #: `SmemPage = SmemPageX + SmemPageY`.
    comptime smem_page = (Self.KBLK + Self.VECLEN) * (
        Self.RPT * Self.TR + Self.CPT * Self.TC
    )
    #: `SmemSize = 2 * SmemPage * sizeof(DataT)` -- BYTES, double buffered.
    comptime smem_size = (
        2
        * ((Self.KBLK + Self.VECLEN) * (Self.RPT * Self.TR + Self.CPT * Self.TC))
        * Self.DBYTES
    )


struct ColKernelPolicy[
    VECLEN: Int,
    KBLK: Int,
    RPT: Int,
    CPT: Int,
    TR: Int,
    TC: Int,
    DBYTES: Int = 4,
](Copyable, Movable):
    """`template <...> struct ColKernelPolicy`, `:105-148`. The COLUMN-MAJOR
    policy.

    Differs from `KernelPolicy` in the loader geometry only: `LdgThRow` is
    `Mblk / Veclen` rather than `Kblk / Veclen`, both pages are `Kblk` rows
    tall, and `SmemStride` pads `Mblk` rather than `Kblk`. Their
    `static_assert(Mblk == Nblk)` is transcribed as `assert_col_policy_square`
    below, because Mojo has no `static_assert` in a struct body here.
    """

    comptime kblk = Self.KBLK
    comptime veclen = Self.VECLEN
    comptime acc_rows_per_th = Self.RPT
    comptime acc_cols_per_th = Self.CPT
    comptime acc_th_rows = Self.TR
    comptime acc_th_cols = Self.TC
    comptime nthreads = Self.TR * Self.TC
    comptime mblk = Self.RPT * Self.TR
    comptime nblk = Self.CPT * Self.TC
    #: `LdgThRow = Mblk / Veclen` -- threads loading a single COLUMN.
    comptime ldg_th_row = (Self.RPT * Self.TR) // Self.VECLEN
    #: `LdgPerThX = Kblk * LdgThRow / Nthreads`.
    comptime ldg_per_th_x = (
        Self.KBLK * ((Self.RPT * Self.TR) // Self.VECLEN)
    ) // (Self.TR * Self.TC)
    #: `LdgPerThY = Kblk * LdgThRow / Nthreads`.
    comptime ldg_per_th_y = (
        Self.KBLK * ((Self.RPT * Self.TR) // Self.VECLEN)
    ) // (Self.TR * Self.TC)
    #: `LdgRowsX = Kblk / LdgPerThX`.
    comptime ldg_rows_x = Self.KBLK // (
        (Self.KBLK * ((Self.RPT * Self.TR) // Self.VECLEN))
        // (Self.TR * Self.TC)
    )
    #: `LdgRowsY = Kblk / LdgPerThY`.
    comptime ldg_rows_y = Self.KBLK // (
        (Self.KBLK * ((Self.RPT * Self.TR) // Self.VECLEN))
        // (Self.TR * Self.TC)
    )
    #: `SmemStride = Mblk + Veclen`.
    comptime smem_stride = (Self.RPT * Self.TR) + Self.VECLEN
    #: `SmemPageX = SmemStride * Kblk`.
    comptime smem_page_x = ((Self.RPT * Self.TR) + Self.VECLEN) * Self.KBLK
    #: `SmemPageY = SmemStride * Kblk`.
    comptime smem_page_y = ((Self.RPT * Self.TR) + Self.VECLEN) * Self.KBLK
    #: `SmemPage = SmemPageX + SmemPageY`.
    comptime smem_page = (
        2 * (((Self.RPT * Self.TR) + Self.VECLEN) * Self.KBLK)
    )
    #: `SmemSize = 2 * SmemPage * sizeof(DataT)`.
    comptime smem_size = (
        2
        * (2 * (((Self.RPT * Self.TR) + Self.VECLEN) * Self.KBLK))
        * Self.DBYTES
    )


def assert_col_policy_square[RPT: Int, CPT: Int, TR: Int, TC: Int]() raises:
    """`static_assert(Mblk == Nblk, "Mblk should be equal to Nblk")`, `:147`.

    Their assertion, relocated rather than dropped. Call it beside any
    `ColKernelPolicy` instantiation.
    """
    if RPT * TR != CPT * TC:
        raise Error(
            "ColKernelPolicy: Mblk should be equal to Nblk (raft"
            " contractions.cuh:147); got Mblk="
            + String(RPT * TR)
            + " Nblk="
            + String(CPT * TC)
        )


# ---------------------------------------------------------------------------
# The three policy families, float instantiations only.
# ---------------------------------------------------------------------------
# `Policy4x4<float, _veclen> { KernelPolicy<float, _veclen, 32, 4, 4, 16, 16> }`
# `:155-166`. This is the one RAFT's float distance kernels instantiate and
# the one `core/gemm.mojo` flattens; `_veclen` is chosen from `k` by the
# caller, which is why it stays a parameter here.
comptime Policy4x4Float = KernelPolicy[4, 32, 4, 4, 16, 16]
comptime Policy4x4FloatCol = ColKernelPolicy[4, 32, 4, 4, 16, 16]

# `Policy4x4Skinny<float, _veclen> { KernelPolicy<float, _veclen, 8, 4, 4, 8,
# 8> }`, `:175-185`. Their comment: a smaller k-block (8 instead of 32) with
# fewer threads per block (8x8 instead of 16x16), FASTER FOR `fusedL2NN` ON
# SKINNY MATRICES -- matrices with a small k dimension.
comptime Policy4x4SkinnyFloat = KernelPolicy[4, 8, 4, 4, 8, 8]
comptime Policy4x4SkinnyFloatCol = ColKernelPolicy[4, 8, 4, 4, 8, 8]

# `Policy2x8<float, _veclen> { KernelPolicy<float, _veclen, 16, 2, 8, 8, 32> }`
# `:192-202`. 16 elements per thread with k-block 16. No `ColPolicy` upstream.
comptime Policy2x8Float = KernelPolicy[1, 16, 2, 8, 8, 32]
