# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The matrix product. One call to MAX's tuned matmul, plus RAFT's policy.

WHAT IS LEFT IN THIS FILE, AND WHY THE KERNEL IS NOT
-----------------------------------------------------
Where cuVS and cuML call cuBLAS for a STANDALONE matrix product, they call a
library with no source, so there is nothing to port and `linalg.matmul` is
the default mirror. The Gram product PCA, truncated SVD and OLS need goes
through `gemm_tn` here, which dispatches: the tuned matmul where the output
has enough tiles to fill the device, and `core/gram_splitk.mojo` where it
does not (measured 13x off bandwidth on the vendor kernel at the shipped
shapes; the closed-library exception, argued in that file's header).

**It is not the rule for a distance step.** `archive/reference/VENDOR_LIBS.md` opens with why:
their dispatch for pairwise distance under an argmin or a top-k does not call
cuBLAS at all, it calls a FUSED kernel that never materializes the distance
matrix, and a device-wide matmul cannot be fused into anything. Callers whose
product is an intermediate inside a reduction belong on the fused kernel
(`cluster/gbdt/distance/fused_distance_nn/simt_kernel.mojo`,
`neighbors/.../fused_l2_knn.mojo`), not here.

A STANDALONE register-tiled contraction used to sit here as well, a port of
`raft/linalg/contractions.cuh::KernelPolicy` and the load/accumulate structure
of `raft/linalg/detail/contractions.cuh` at RAFT `9aa17e5` (FLAGGED, NOT PROVEN, 2026-08-31: `gemm/DERIVATION_MAP.tsv` pins RAFT at `661a3b8` and that lane's one testable citation resolves exactly there. This citation carries no line numbers, so nothing settles it either way; do not 'fix' it toward the map without evidence), unfused and
writing its product to memory. It is DELETED. Nothing called it once OLS's
step 6 went to `gemv_gpu`, it was measured at about 15 GFLOP/s against
`linalg.matmul`'s 248 at 1M x 128, and the FUSED instantiation of the same
policy, which is the one RAFT's dispatch actually runs, lives in
`simt_kernel.mojo` and is untouched.

The vendor matmul's 248 GFLOP/s is a SQUARE-shape number. On the Gram shape
PCA/tSVD/OLS actually ship (tiny output, k in the millions) it delivers ~25,
because its only parallelism is output tiles and a 32 x 32 output is one
tile. That regime now goes to `core/gram_splitk.mojo`, a hand-written
split-K Gram kernel under the closed-library exception -- see its header
for the measurements and for why every MAX route was exhausted first.
`gemm_tn` below dispatches between the two.

THEIR POLICY IS STILL HERE, BECAUSE THE FUSED KERNEL NEEDS IT
--------------------------------------------------------------
`KernelPolicy<float, Veclen=4, Kblk=32, AccRowsPerTh=4, AccColsPerTh=4,
AccThRows=16, AccThCols=16>`, RAFT's `Policy4x4<float>` and the one their float
distance kernels instantiate:

    Nthreads = AccThRows * AccThCols    = 256
    Mblk     = AccRowsPerTh * AccThRows = 64
    Nblk     = AccColsPerTh * AccThCols = 64
    SmemStride = Kblk + Veclen          = 36   (padding, not a rounding)

These constants stay because `cluster/gbdt/distance/fused_distance_nn/
simt_kernel.mojo` is instantiated at this policy and its callers compute their
launch geometry from it. That kernel is NOT a substitution candidate under any
version of the rule: RAFT fuses the argmin epilogue into the contraction and
never writes the distance tile, which is a thing no BLAS call can do and is
the whole point of `fusedDistanceNN`.

`SmemStride = Kblk + Veclen` is theirs and is not arbitrary: the padding
staggers each row's start so that threads reading down a column of shared
memory do not all land in the same bank.
"""

from layout import TileTensor
from layout.tile_layout import row_major
from linalg.matmul import matmul
from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import block_dim, block_idx, thread_idx

from core.gram_splitk import (
    GRAM_MAX_CELLS_PER_THREAD,
    GRAM_MAX_COLS,
    GRAM_TPB,
    gemm_tn_splitk_into,
    gram_splitk_applies,
)
from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_mul_add,
)
from std.sys.compile import is_defined

# DEVIATION 537: these two imports exist only for the flag-guarded swap
# below. `gemm/checks/gemm_identical.mojo` does not import `core.gemm`
# (checked 2026-09-02), so no cycle; the cost is that every build of this
# file now compiles that module even with the flag off. THE COMPILE RISK IS
# CLEARED: this file built first try in both flag states on ONE APPLE M4,
# 2026-09-02.
from gemm.checks.gemm_identical import identical_gemm
from gemm.checks.gemm_oracle import OP_NT


# ===========================================================================
# THE PINNED PRODUCTS (IDENTITY_PATHS row 28, DEVIATION 526)
# ===========================================================================
#
# `linalg.matmul` and `linalg.gemv.gemv_gpu` are CLOSED vendor libraries, and
# that is the correct default: where cuVS and cuML call cuBLAS for a
# standalone product they call a library with no source, so there is nothing
# to port. **A closed library cannot carry an identity claim.** Its tile
# shape, its k-split and, on some backends, its mantissa width are chosen per
# vendor and per shape, and a k-split IS a summation order. IDENTITY_PATHS
# row 24 reached this verdict for the k-NN distance step and paid for it
# (`neighbors/checks/pinned_distance_tile.mojo`, measured 2.85x); these
# two are the same verdict for the two products OLS and PCA run through.
#
# THE CONSTRUCTION, and it is deliberately the simplest correct one:
# ONE THREAD PER OUTPUT CELL, contraction axis ASCENDING, every step through
# `identical_mul_add` (one rounding) with `ftz` at the seams. Therefore
#
#     the sequence of arithmetic operations that produces cell (i, j) is a
#     pure function of k, and of NOTHING ELSE.
#
# Not of the grid, not of the block size, not of how many other cells were
# computed in the same launch, not of the lane width, not of the vendor. A
# staged or split-k version would be faster and would be a SECOND thing to
# pin, which is the trade row 24 already made once and made the same way.
#
# THIS IS THE PROPERTY THE SERVING WORLD CALLS BATCH INVARIANCE. There, the
# observation is that changing the batch size changes the reduction order
# inside attention and normalization kernels, so the same prompt returns
# different logits depending on what else was in the batch. It is the same
# defect as IDENTITY_PATHS rows 3 and 7 -- a block count is a summation order
# -- reached from a different direction by a different community. A product
# whose per-cell order does not depend on launch geometry is the shared fix,
# and it is worth knowing that the two problems are one problem.
#
# THE COST IS A MEASUREMENT, NOT AN ARGUMENT: `tools/price_linalg_identity.sh`
# runs the arms interleaved and reports the ratio. Under `NUMERIC_FAST`
# neither kernel is reachable and the vendor call is what runs, bit for bit.


def pinned_gemm_nt_kernel(
    z: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    k_in: Int32,
):
    """`z[m x n] = x[m x k] . y[n x k]^T`, one thread per output cell.

    The whole numeric content is the loop below: `p` ascending from 0 to
    k-1, one `fma` per step, no partials, no cross-thread combination, no
    atomics. Two threads never contribute to one cell, so nothing about the
    launch can reach the answer.
    """
    var m = Int(m_in)
    var n = Int(n_in)
    var k = Int(k_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= m * n:
        return
    var i = cell // n
    var j = cell % n
    var acc = Float32(0.0)
    for p in range(k):
        acc = ftz(
            identical_mul_add(
                ftz(x.unsafe_load(i * k + p)),
                ftz(y.unsafe_load(j * k + p)),
                acc,
            )
        )
    # THE FOLD IS UNCONDITIONAL, and at one partition it is a one-term
    # fold rather than a bypass. `gemm/IDENTICAL_FP32_CONTRACT.md` section
    # 9.2(b): seeding at `+0.0` and always folding is what keeps the SIGN OF
    # A ZERO from being a function of the partition count. `0.0 + x` is `x`
    # for every x except `-0.0`, where it is `+0.0` -- so a cell that
    # accumulates to negative zero comes out `+0.0` here AND under a
    # partitioned arm, instead of `-0.0` here and `+0.0` there. Inert today
    # because this kernel is always one partition; it stops being inert the
    # moment the generalized kernel lands, and IDENTITY_PATHS row 13 is what
    # a signed zero does downstream when a min or a max consumes it.
    z.unsafe_store(cell, ftz(Float32(0.0) + ftz(acc)))


def pinned_gemm_nt_gram_kernel(
    z: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    k_in: Int32,
):
    """`z[m x n] = x[m x k] . x[n x k]^T`: `pinned_gemm_nt_kernel` with ONE
    operand pointer instead of two.

    WHY THIS EXISTS, AND WHY IT IS A SECOND KERNEL RATHER THAN A CALL.
    `gemm_nt_gram` needs the Gram product under IDENTICAL, which means one
    buffer feeding BOTH operands. Handing the same pointer to
    `pinned_gemm_nt_kernel`'s `x` and `y` does not compile:
    `enqueue_function` treats every pointer argument as a mutable borrow
    and refuses two that alias --

        aliasing values passed mutably to 'args' argument and passed
        mutably to 'args' argument in 'enqueue_function' call

    -- and that refusal survives materialising the pointer into a single
    local first, so "a pointer is not a borrow" is not true of this API.
    The alternatives were worse: DEVIATION 1873 deleted the second
    transpose precisely to stop writing and reading `k * m` floats for
    nothing, and re-adding it under IDENTICAL would buy compilability with
    the cost 1873 was written to remove.

    So the operand is collapsed in the SIGNATURE, where the aliasing rule
    cannot see it, and the body below is `pinned_gemm_nt_kernel`'s body
    with `y` spelled `x`. Nothing else differs -- same `p` ascending 0..k-1,
    same single `identical_mul_add` per step, same `ftz` on both loads and
    on the accumulator, same unconditional `+0.0` fold at the end for the
    signed-zero reason `gemm/IDENTICAL_FP32_CONTRACT.md` section 9.2(b)
    gives. The two kernels must be read side by side and changed together;
    that duplication is deliberate and is cheaper than a wrapper that would
    have to defeat the same rule.

    BITS: for `x is y` this computes what `pinned_gemm_nt_kernel` computes,
    operation for operation and in the same order.
    """
    var m = Int(m_in)
    var n = Int(n_in)
    var k = Int(k_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= m * n:
        return
    var i = cell // n
    var j = cell % n
    var acc = Float32(0.0)
    for p in range(k):
        acc = ftz(
            identical_mul_add(
                ftz(x.unsafe_load(i * k + p)),
                ftz(x.unsafe_load(j * k + p)),
                acc,
            )
        )
    z.unsafe_store(cell, ftz(Float32(0.0) + ftz(acc)))


def pinned_gemv_n_kernel(
    z: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    k_in: Int32,
):
    """`z[m] = x[m x k] . y[k]`, one thread per output element.

    The `n == 1` case of the kernel above, kept separate only because the
    caller's `y` is a k-vector rather than an `n x k` matrix; the index
    arithmetic collapses to the same thing and the loop is character for
    character the same, so the two cannot round differently.
    """
    var m = Int(m_in)
    var k = Int(k_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= m:
        return
    var acc = Float32(0.0)
    for p in range(k):
        acc = ftz(
            identical_mul_add(
                ftz(x.unsafe_load(i * k + p)), ftz(y.unsafe_load(p)), acc
            )
        )
    # Section 9.2(b)'s one-term fold, as in the kernel above and for the
    # same reason: the sign of a zero must not depend on the partition
    # count.
    z.unsafe_store(i, ftz(Float32(0.0) + ftz(acc)))


# ===========================================================================
# DEVIATION 537 -- THE FLAG-GUARDED v1 SWAP. OFF BY DEFAULT.
# LADDER STEPS 1-3 GREEN ON ONE APPLE M4; STEPS 4 AND 5 OWED.
# ===========================================================================
#
# `-D MOJOLEARN_537_GEMM_IDENT_SWAP=1` routes `gemm_nt`'s IDENTICAL arm (the
# `n > 1` branch only) to `gemm/checks/gemm_identical.mojo::identical_gemm`,
# profile `mojolearn.identical.gemm.fp32.v1`, instead of
# `pinned_gemm_nt_kernel`. This is the swap the MLSys paper's sentence names:
# the newer portable matrix product exists (v1, eight execution plans,
# three-vendor bit-certified at its own card, leg 11) and is not the released
# implementation; k-NN inherits the pinned kernel through
# `neighbors/impl/neighbors/detail/knn_brute_force.mojo:453,498`.
#
# **THIS IS A BIT-MOVING SWAP, NOT AN OPTIMIZATION OF THE SAME BITS.**
# Established from source, 2026-09-02:
#   - `pinned_gemm_nt_kernel` folds the WHOLE k range serially ascending
#     (this file, the `for p in range(k)` loop) and stores a `+0.0`-seeded
#     one-term fold (`ftz(Float32(0.0) + ftz(acc))`, below). That is
#     `gemm_oracle_serial` plus a seed -- the DIAGNOSTIC spelling, per
#     `gemm/IDENTICAL_FP32_CONTRACT.md` section 7.6 item 2.
#   - `identical_gemm` partitions k into `P = contract_leaf_count(k)` leaves
#     (`CONTRACT_K_LEAF_MIN = 128`, `gemm/checks/gemm_oracle.mojo:95`) and
#     folds a seedless fixed balanced tree. Fixture F1 measures the
#     difference at k = 1024: serial 0x4b800000, tree 0x4b8001c0
#     (`gemm/README.md`, clause-5 table). At k <= 128 (P == 1) the two
#     differ on exactly one value: the seed here launders a `-0.0` leaf
#     partial to `+0.0`; v1 stores `-0.0` (measured, IDENTITY_PATHS row 28:
#     gemm_nt 0x00000000, v1 0x80000000).
#
# THEREFORE THE FLAG NEVER FLIPS ON A PERFORMANCE NUMBER ALONE, and never on
# fingerprint equality -- flag-on and flag-off DIFFER by design. Flipping is
# a PROFILE MIGRATION of the released standalone product onto v1 (the "last
# step of Phase 2" IDENTITY_PATHS row 28 already names) and requires, at ONE
# commit: (1) the identity lane's sign-off, because rows 27/28 and the
# E1U/E2 vendor cards are certified on the current serial bits; (2)
# re-certification of the swapped caller path on all three vendor columns
# (Apple, NVIDIA, AMD); (3) a measured timing win at large shapes. The
# paper's own rule applies: changing the floor is a new profile of the
# released path, not a patch.
#
# SCOPE, deliberately narrow: `gemv_n` (the `n == 1` route, taken FIRST in
# both modes) and `gemm_nt_gram` are NOT swapped -- the gram entry hands one
# pointer to two kernel operands, which `identical_gemm`'s two-`mut`-buffer
# signature refuses, the exact aliasing rule `gemm_nt_gram`'s docstring
# records. A FLIP must migrate all of contract 7.6's kernels together or
# record that the identical tier runs two profiles at once; flag-on today is
# a MEASUREMENT configuration, not a shippable state.
#
# SEMANTICS CAVEAT: `identical_gemm` allocates its own workspace and
# SYNCHRONIZES before returning; the pinned enqueue is asynchronous. Correct
# for every caller (they synchronize later), but the timing leg measures the
# sync and the allocation too. A flip would want `identical_gemm_into` with
# a caller-owned workspace; that is follow-up work, not this deviation.
#
# RUN STATE, 2026-09-02, ONE APPLE M4 (exact commands and measured bits in
# `gemm/README.md`, DEVIATION 537 section). RUN AND GREEN: flag-off
# inertness in both modes; the flag-on gates, which compiled first try and
# moved exactly one arm, the identity check's reference cell (0x40d15787
# off -> 0x40d15798 on), with no kNN arm moved; and the reach proof
# `gemm/checks/gemm_537_reach_probe.mojo`, which returns 0x00000000 with
# the flag off and 0x80000000 with it on, so the flag reaches this branch.
# STILL OWED, NOT RUN: the timing window (there is no timing number) and
# the three-vendor legs. The flip rule is unchanged and UNMET -- one Apple
# box is not three columns.
comptime GEMM_IDENT_SWAP_537 = is_defined["MOJOLEARN_537_GEMM_IDENT_SWAP"]()


#: Threads per block for the two pinned products. SCHEDULING and provably so:
#: each thread owns a whole output cell, so this number moves WHICH thread
#: computes a cell and never the order any cell is accumulated in. That is
#: the distinction `numerics.mojo` opens by drawing, and it is the reason
#: this row does not belong in `lib_block_bounds_a_float_fold` beside the
#: rows whose block size IS a fold width.
comptime PINNED_GEMM_TPB = 256


# `KernelPolicy<float, 4, 32, 4, 4, 16, 16>`, RAFT's Policy4x4<float>.
comptime GEMM_VECLEN = 4
comptime GEMM_KBLK = 32
comptime GEMM_ACC_ROWS_PER_TH = 4
comptime GEMM_ACC_COLS_PER_TH = 4
comptime GEMM_ACC_TH_ROWS = 16
comptime GEMM_ACC_TH_COLS = 16

comptime GEMM_THREADS = GEMM_ACC_TH_ROWS * GEMM_ACC_TH_COLS
comptime GEMM_MBLK = GEMM_ACC_ROWS_PER_TH * GEMM_ACC_TH_ROWS
comptime GEMM_NBLK = GEMM_ACC_COLS_PER_TH * GEMM_ACC_TH_COLS
comptime GEMM_SMEM_STRIDE = GEMM_KBLK + GEMM_VECLEN
comptime GEMM_SMEM_PAGE_X = GEMM_SMEM_STRIDE * GEMM_MBLK
comptime GEMM_SMEM_PAGE_Y = GEMM_SMEM_STRIDE * GEMM_NBLK


def gemm_nt(
    ctx: DeviceContext,
    mut z: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    mut y: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
) raises:
    """`z[m x n] = x[m x k] . y[n x k]^T` through MAX's tuned matmul.

    `transpose_b=True` is exactly the shape every algorithm here wants: rows
    against centroids, index points or candidates, with neither operand ever
    materialized in another layout.

    **`n == 1` GOES TO GEMV, BECAUSE `transpose_b=True` DOES NOT WRITE THERE.**
    Measured 2026-08-19 through this wrapper (`checks/
    vendor_correctness_check.mojo`): m=64, n=1, k=32, output poisoned before
    the call, **63 of the 64 rows still held the poison afterwards**. Nothing
    was written wrong; 63 rows were not written at all, so a caller reusing a
    buffer reads whatever was in it last time. That is worse than zeros, and
    `archive/reference/VENDOR_LIBRARIES.md` used to say "returns zeros for some outputs", which
    is why this was mistaken for a benign edge case for as long as it was.

    The fault is `transpose_b=True`, not `n == 1`: the same product with
    `transpose_b=False` is correct at the identical shape, which is what the
    sabotage in that check established.

    `n == 1` is not exotic. It is a one-column design matrix in `lstsq.mojo`
    and `pca.mojo`, a single index point in `knn_brute_force.mojo`, and in
    `min_cluster_distance_compute.mojo:197` it is any run where
    `n_clusters % centroid_batch == 1` -- a REMAINDER, so ordinary cluster
    counts reach it.

    Routing to `gemv_n` needs no reshaping and is what RAFT does for the same
    degenerate case: at `n == 1` this product IS a matrix-vector product, `y`
    is already a contiguous k-vector whether it is read as `(1, k)` or
    `(k, 1)`, and `z` is already a contiguous m-vector. `gemv_n` wraps
    `gemv_gpu`, which tests correct at every m from 1 to 100,003.
    """
    if n == 1:
        gemv_n(ctx, z, x, y, m, k)
        return
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        comptime if GEMM_IDENT_SWAP_537:
            # DEVIATION 537, see the block above PINNED_GEMM_TPB: profile
            # v1 through `identical_gemm`, DIFFERENT BITS from the pinned
            # kernel below by design, OFF in every build that does not name
            # `-D MOJOLEARN_537_GEMM_IDENT_SWAP=1`. The `n == 1` route
            # above stays on `gemv_n` in both flag states.
            identical_gemm(ctx, z, x, y, m, n, k, OP_NT)
            return
        # DEVIATION 526. `matmul` is a closed vendor library; see the
        # PINNED PRODUCTS block above. The `n == 1` route is taken FIRST and
        # deliberately, so the degenerate shape keeps landing on ONE
        # implementation in both modes rather than acquiring a second
        # spelling that only the identical arm sees.
        ctx.enqueue_function[pinned_gemm_nt_kernel](
            z.unsafe_ptr(),
            x.unsafe_ptr(),
            y.unsafe_ptr(),
            Int32(m),
            Int32(n),
            Int32(k),
            grid_dim=((m * n + PINNED_GEMM_TPB - 1) // PINNED_GEMM_TPB, 1, 1),
            block_dim=(PINNED_GEMM_TPB, 1, 1),
        )
        return
    var tz = TileTensor(z, row_major(m, n))
    var tx = TileTensor(x, row_major(m, k))
    var ty = TileTensor(y, row_major(n, k))
    matmul[transpose_b=True, target="gpu"](tz, tx, ty, ctx)


def gemm_nt_gram(
    ctx: DeviceContext,
    mut z: DeviceBuffer[DType.float32],
    xt: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
) raises:
    """`z[m x n] = xt[m x k] . xt[n x k]^T`: `gemm_nt` with ONE operand.

    DEVIATION 1873. This exists for a reason that is about the LANGUAGE and
    not about arithmetic, and it is worth stating plainly because the cost it
    removes is large.

    `gemm_nt` takes `mut x` and `mut y`. Neither is written -- only `z` is --
    but they are declared `mut`, so the borrow checker refuses the same
    buffer for both, and the Gram case IS the same buffer for both.
    `gemm_tn_via_transpose` worked around that by transposing X into TWO
    destination buffers with byte-identical kernel calls and handing one to
    each parameter. The second transpose is a full redundant pass over
    `k * m` floats and a second `k * m` of device memory, bought purely to
    satisfy an aliasing rule.

    At the shipped `gram.32x32x1M` row that is 128 MB written and 128 MB read
    for nothing, against a product whose entire useful traffic is one read of
    the same 128 MB.

    So this entry takes the operand ONCE and builds both views from it. The
    FAST arm builds two `TileTensor`s over one buffer, which is a read-only
    aliasing the vendor kernel is perfectly happy with; the IDENTICAL arm
    passes the same pointer to both kernel arguments, which it already
    supports because a pointer is not a borrow.

    THE PINNED ARM IS UNCHANGED BIT FOR BIT. Same kernel, same launch
    geometry, same fold order; only the pointer it reads its second operand
    from moves, and it moves to a buffer holding the identical bytes.
    """
    if n == 1:
        # The Gram case at n == 1 is a 1 x 1 output and m == n == 1, so the
        # gemv route `gemm_nt` takes here cannot be reached with two
        # DIFFERENT operands. Refused by name rather than routed, because
        # gemv_n has the same two-mut-parameter aliasing problem and
        # papering over it here would hide it.
        raise Error(
            "gemm_nt_gram: n == 1 is not a Gram shape this entry serves."
            " gemm_nt's gemv route takes two mut buffers and cannot be"
            " handed one buffer twice; a 1 x 1 Gram is a dot product and"
            " belongs somewhere else."
        )
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        # ONE POINTER, TAKEN ONCE, PASSED TWICE -- AND BOTH HALVES OF THAT
        # SENTENCE ARE LOAD-BEARING.
        #
        # DEVIATION 1873 wrote `xt.unsafe_ptr()` twice here and the file did
        # not compile under IDENTICAL from the day it landed (6e384b2,
        # 2026-08-25) until 2026-08-28. `pinned_gemm_nt_kernel` declares
        # `MutPointer[Float32, MutAnyOrigin]` for all three operands, which
        # RESUME.md records as a requirement of the kernel ABI rather than a
        # choice; `xt` is a read-only parameter, so `xt.unsafe_ptr()` is an
        # immutable pointer, and the enqueue failed the constraint:
        #
        #   argument #2 of type 'Pointer[..., mut=False]' does not match the
        #   declared function argument type 'Pointer[..., mut=True]'
        #
        # NEITHER OBVIOUS REPAIR WORKS. Declaring the parameter `mut xt`
        # satisfies the constraint and then trips the aliasing rule at this
        # very call -- "aliasing values passed mutably to 'args' argument" --
        # AND at the FAST arm's two TileTensor views below, which is the
        # exact rule the docstring above says this entry point exists to
        # dodge. Widening the kernel's operands to immutable pointers is not
        # available for the same reason the ABI note gives.
        #
        # So the pointer is materialised ONCE into a local and that local is
        # handed to both arguments. `xt` is borrowed a single time; what the
        # enqueue sees twice is a value, and a pointer is not a borrow --
        # which is what the docstring claimed all along and what this line
        # finally spells. `unsafe_origin_cast[MutUntrackedOrigin]` is the
        # repo's existing spelling for the same coercion (see
        # ensemble/checks/sampled_cols_check.mojo).
        #
        # BITS: the kernel, its launch geometry and its fold order are
        # untouched, and the FAST arm below is not compiled in this branch.
        var xtm = xt
        ctx.enqueue_function[pinned_gemm_nt_gram_kernel](
            z.unsafe_ptr(),
            xtm.unsafe_ptr(),
            Int32(m),
            Int32(n),
            Int32(k),
            grid_dim=((m * n + PINNED_GEMM_TPB - 1) // PINNED_GEMM_TPB, 1, 1),
            block_dim=(PINNED_GEMM_TPB, 1, 1),
        )
        return
    var tz = TileTensor(z, row_major(m, n))
    var tx = TileTensor(xt, row_major(m, k))
    var ty = TileTensor(xt, row_major(n, k))
    matmul[transpose_b=True, target="gpu"](tz, tx, ty, ctx)


def gemm_tn(
    ctx: DeviceContext,
    mut z: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    mut xt: DeviceBuffer[DType.float32],
    mut xt2: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
) raises:
    """`z[m x n] = x[k x m]^T . x[k x n]`: the Gram shape, DISPATCHED.

    Two arms, picked by `gram_splitk_applies` (a computed predicate, not a
    constant -- see its docstring):

    - **Small output, any k** -> `core/gram_splitk.mojo::gemm_tn_splitk_into`
      (the `xt`-scratch-reusing entry over the same kernel pair).
      The vendor matmul's only parallelism is output tiles, and a 32 x 32
      output is ONE tile: measured 322.9 ms (~25 GFLOP/s) at 32 x 32 x 4M
      against a ~10-15 ms bandwidth floor (`bench/results/
      LANE_covariance-unblock_2026-08-19.md`, orchestrator postscript). The
      split-K kernel parallelizes the k axis instead, reads X once, and
      needs neither transpose nor the alias buffers.
    - **Output big enough to fill the device** -> `gemm_tn_via_transpose`
      below, the tuned vendor matmul behind two device transposes.

    The split-K arm is the APPLE target column's only: on nvidia/amd
    `gram_splitk_applies` is False at every shape
    (`hardware_matrix.gram_splitk_is_target_arm`), because MAX's matmul has
    its own split-K machinery there -- it is comptime-gated
    `not has_apple_gpu_accelerator()` in Modular's source, verified by
    LANE_gram-splitk -- so those targets hand the Gram shape back to
    `linalg.matmul` through the transpose route.

    Both arms are exercised by name: `checks/gram_splitk_check.mojo`
    runs each directly against a Float64 host oracle, and the vendor table
    covers each through this wrapper (`check_matmul_colmajor`'s tail rows).

    **UNDER `NUMERIC_IDENTICAL` THERE IS ONE ARM AND A REFUSAL, NOT TWO
    ARMS** (IDENTITY_PATHS row 27, DEVIATION 521). `gram_splitk_applies`
    resolves the column to `COLUMN_BIT_IDENTICAL` and stops consulting the
    starvation test, so it answers True on every column wherever the
    kernel's capacity allows. Where capacity does NOT allow, the shape
    would otherwise fall through to `gemm_tn_via_transpose` and
    `linalg.matmul`, a CLOSED vendor library whose tile shape and k-split
    are per-vendor and whose k-split IS a summation order -- an identity
    build would then return a model that is not identical and say nothing.
    So it raises by name instead. REFUSE is the third and only other
    legitimate move (IDENTITY_PATHS' opening rule); the closure that
    upgrades this refusal to a run is a wider split-K kernel, and it is
    named in the error rather than left for a reader to infer.
    """
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        if not gram_splitk_applies(m, n, k):
            raise Error(
                "gemm_tn: NUMERIC_IDENTICAL refuses the Gram shape "
                + String(m)
                + " x "
                + String(n)
                + " x "
                + String(k)
                + ". The split-K kernel (core/gram_splitk.mojo) is the only"
                + " arm with a pinned summation order, and this shape is"
                + " outside its capacity (needs m == n <= "
                + String(GRAM_MAX_COLS)
                + " and m*n <= "
                + String(GRAM_TPB * GRAM_MAX_CELLS_PER_THREAD)
                + " register cells). The other arm is linalg.matmul, whose"
                + " k-split is a per-vendor summation order, so running it"
                + " would return a NON-identical model under a mode that"
                + " promises one. IDENTITY_PATHS row 27. To close this"
                + " refusal, widen the split-K kernel's staging tile; to"
                + " work around it today, reduce the feature count or run"
                + " NUMERIC_FAST and drop the cross-vendor claim."
            )
    if gram_splitk_applies(m, n, k):
        # `xt` is pure scratch on both arms and is sized `k * m`, so the
        # split-K arm reuses it as the partials workspace whenever it
        # covers (`gram_splitk_scratch_covers`) -- no per-call allocation
        # on the shipped fit shapes.
        gemm_tn_splitk_into(ctx, z, x, xt, m, k)
        return
    gemm_tn_via_transpose(ctx, z, x, xt, xt2, m, n, k)


def gemm_tn_via_transpose(
    ctx: DeviceContext,
    mut z: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    mut xt: DeviceBuffer[DType.float32],
    mut xt2: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
) raises:
    """`z[m x n] = x[k x m]^T . x[k x n]`, ON MAX'S TUNED MATMUL.

    **This route was abandoned twice and both times for a bad reason.**
    First because `matmul` refuses `transpose_a`; then because
    `linalg.transpose` compiles and signals on device buffers. Neither of
    those blocks the actual identity:

        Xt = transpose(X)      Xt . Xt^T == X^T X

    which is the N-T shape MAX does support. The only missing piece was a
    device transpose, and `core/column_stats.mojo::transpose_kernel` is
    twenty lines.

    WHERE THE VENDOR KERNEL WINS, AND WHERE IT STARVES -- BOTH MEASURED.
    At 1M x 128 the vendor matmul ran ~248 GFLOP/s against ~15 on the
    hand-written contraction this file used to carry, which is why that
    contraction was deleted. But its throughput is per-OUTPUT-TILE
    parallelism (64 x 64 tiles on Apple; `max/kernels/src/linalg/matmul/
    gpu/__init__.mojo:663-688` at `max/v26.5.0`), so on a small output it
    starves: 322.9 ms, ~25 GFLOP/s, at 32 x 32 x 4M (LANE_covariance-unblock
    postscript). So `gemm_tn` sends tile-starved outputs to the split-K
    kernel and this route serves the rest.

    `transpose_a` is STILL unsupported and this does not use it. The transpose
    is two extra passes over `k x m` floats against a product that is
    `O(k * m * n)`, and it buys the tuned kernel.
    """
    from core.column_stats import (
        CUDA_MAX_GRID_YZ,
        TRANSPOSE_TILE,
        transpose_kernel,
    )

    # DEVIATION 1874 -- `m != n` WAS SILENTLY WRONG AND IS NOW REFUSED.
    #
    # The contract is `z[m x n] = x[k x m]^T . x[k x n]`, one operand read at
    # two widths. Both transposes below took `Int32(m)` as the width, so at
    # `m != n` the second operand was an `m x k` block being read as `n x k`:
    # off the end of the shorter, or a truncation of the longer. Every
    # shipped call site passes `m, m, k`, so it never fired -- which is
    # exactly the latent case that surfaces the day someone reuses the entry.
    if m != n:
        raise Error(
            "gemm_tn_via_transpose: m="
            + String(m)
            + " n="
            + String(n)
            + ". This entry is the GRAM case: one operand, one width. Its"
            " transpose is built at width m, and reading that block at width"
            " n runs off it. A genuine two-operand TN needs a second"
            " transpose at width n, which this does not do."
        )

    # DEVIATION 1873 -- ONE TRANSPOSE, NOT TWO.
    #
    # This used to transpose X into `xt` AND into `xt2` with byte-identical
    # kernel calls: same source, same dims, same launch geometry, different
    # destination. The comment that stood here named the reason correctly and
    # then accepted it -- `matmul` refuses one buffer as two mutable
    # arguments (archive/reference/PORTING.md 24), and `Xt . Xt^T` names it twice.
    #
    # THAT IS A LANGUAGE CONSTRAINT, NOT AN ARITHMETIC ONE, and it was being
    # paid for in bandwidth. `gemm_nt` writes only `z`; its two operands are
    # declared `mut` and never mutated. So the fix is an entry that takes the
    # operand ONCE and builds both views from it, which is `gemm_nt_gram`.
    #
    # MEASURED COST OF THE WORKAROUND, H100, 2026-08-25: the four TN rows
    # were the worst in the whole GEMM lane at 1.9x to 4.0x cuBLAS, while
    # every NT row sat at or near parity. At `gram.32x32x1M` the redundant
    # pass is 128 MB written and 128 MB read, against a product whose entire
    # useful traffic is one 128 MB read.
    #
    # `xt2` is kept in the signature and is now UNUSED. Dropping it changes
    # the arity of a function three other lanes' drivers call and this
    # checkout is shared, so that cleanup is OWED and named here rather than
    # done silently under them.
    ctx.enqueue_function[transpose_kernel](
        xt.unsafe_ptr(),
        x.unsafe_ptr(),
        Int32(k),
        Int32(m),
        # DEVIATION 1883 -- Y IS CAPPED AT CUDA'S GRID LIMIT.
        #
        # `k` is the ROW COUNT of the operand and it is the big dimension
        # here: `ols` and `pca` ship 4,000,000 x 32. One block per row tile
        # in Y needs 125,000 blocks, and CUDA caps grid_dim.y at 65,535, so
        # the LAUNCH was rejected outright -- CUDA_ERROR_INVALID_VALUE, both
        # lanes dead on an H100 on 2026-08-26 with a cuML row and no row of
        # ours. `transpose_kernel` now walks its row tiles with a grid-stride
        # loop, so this cap costs occupancy at absurd sizes and nothing else.
        grid_dim=(
            (m + TRANSPOSE_TILE - 1) // TRANSPOSE_TILE,
            min((k + TRANSPOSE_TILE - 1) // TRANSPOSE_TILE, CUDA_MAX_GRID_YZ),
            1,
        ),
        block_dim=(TRANSPOSE_TILE, TRANSPOSE_TILE, 1),
    )
    # DEVIATION 1899 -- BOTH UNCONDITIONAL SYNCS DELETED (the class DEVIATION
    # 1877 names in gemm/checks/gemm_identical.mojo). This function
    # allocates NOTHING: `z`, `x`, `xt`, `xt2` are all caller-owned, so no
    # local buffer's lifetime hangs on a wait here. The transpose above and
    # `gemm_nt_gram` below are enqueued on the SAME ctx and are
    # stream-ordered, so the gram kernel already sees the finished
    # transpose without a host round trip; and a caller that reads `z` does
    # it with an enqueue_copy + synchronize ordered on the same stream.
    # Fences only -- IDENTICAL's bits do not move.
    gemm_nt_gram(ctx, z, xt, m, n, k)
    _ = xt2


# THE GRAM SHAPE'S UPSTREAM ROUTE IS CLOSED. `transpose_a` IS STILL REFUSED.
#
# `raft::stats::cov`, `lstsqEig`'s first step and cuML's `tsvd_fit` all ask
# cuBLAS for `CUBLAS_OP_T, CUBLAS_OP_N`: the Gram shape `A^T A`, contracting
# down the ROW axis. Handing that shape to MAX's matmul directly fails to
# compile, and the constraint is still live:
#
#     max/kernels/src/linalg/matmul/__init__.mojo:110:9:
#     note: constraint failed: transpose_a not yet supported
#
# `transpose(X) . transpose(X)^T` is the same matrix in the N-T shape, so
# one twenty-line transpose puts T-N users on the tuned matmul -- and that
# is `gemm_tn_via_transpose`, the arm for outputs with enough tiles to fill
# the device. On the tile-starved outputs the shipped fits actually have,
# the tuned matmul is measured 13x off the bandwidth floor, and `gemm_tn`
# dispatches those to `core/gram_splitk.mojo` instead.
#
# The N-T route is still the one to prefer where a caller can choose it: it
# skips the transpose entirely.

from linalg.gemv import gemv_gpu


def gemv_n(
    ctx: DeviceContext,
    mut z: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    mut y: DeviceBuffer[DType.float32],
    m: Int,
    k: Int,
) raises:
    """`z[m] = x[m x k] . y[k]`, on MAX's tuned GEMV.

    `raft::linalg::gemv` is what RAFT calls for `w <- covA Ab`, and
    `linalg.gemv.gemv` is HOST-ONLY (no ctx, no target). `gemv_gpu` is the
    GPU sibling. See archive/reference/VENDOR_LIBRARIES.md.

    Orientation read from MAX's `gemv.mojo`, not guessed: `GemmShape.get`
    takes m and n from `c` and k from `a`, ignoring `b` entirely, so
    `c = (m, 1)`, `a = (m, k)`, `b = (k, 1)`. `transpose_b` must be False;
    its True arm swaps a and b and at n=1 would write one output instead of
    m. `b` is `(k, 1)` and not `(1, k)` because the MATMUL_NAIVE fallback in
    the same dispatcher indexes B as `(k, n)`.

    REACH PROVED BY SABOTAGE, 2026-08-19. A green OLS run does NOT show this
    is reached: `w <- inv Ab` is 8x8 in the checks, and the scale-invariance
    check in particular would pass under any wrong-but-linear kernel. So the
    orientation was flipped to `transpose_b=True` and the run failed with
    `coefficient 1 = 0.0` — only the first output written, exactly what that
    arm predicts at n=1, since it swaps a and b and passes `(n, m, k)`.
    Reverted; that failure is the evidence.
    """
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        # DEVIATION 526. `gemv_gpu` is a closed vendor library and its
        # reduction over k is its own business: on a wide-k gemv the usual
        # shape is one BLOCK per output row with a cross-lane fold, whose
        # width is the hardware's (32 on Apple and NVIDIA, 64 on AMD) --
        # IDENTITY_PATHS row 20's defect, inside a library we cannot reach.
        # The pinned kernel gives the row to ONE thread instead, so k is
        # walked ascending with no fold at all.
        ctx.enqueue_function[pinned_gemv_n_kernel](
            z.unsafe_ptr(),
            x.unsafe_ptr(),
            y.unsafe_ptr(),
            Int32(m),
            Int32(k),
            grid_dim=((m + PINNED_GEMM_TPB - 1) // PINNED_GEMM_TPB, 1, 1),
            block_dim=(PINNED_GEMM_TPB, 1, 1),
        )
        return
    var tz = TileTensor(z, row_major(m, Int(1)))
    var tx = TileTensor(x, row_major(m, k))
    var ty = TileTensor(y, row_major(k, Int(1)))
    gemv_gpu[transpose_b=False](tz, tx, ty, ctx)
