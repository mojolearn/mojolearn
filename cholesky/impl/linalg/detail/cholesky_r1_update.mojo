# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Rank-one update of a Cholesky factor.

PORT of `raft/linalg/detail/cholesky_r1_update.cuh::choleskyRank1Update` at
RAFT `ebf9268` (`upstream/raft-v26.08.00`), lower-triangular arm. **COPY, DO
NOT IMPROVE**, with DEVIATIONS 1632, 1633 and 1646 below.

**THIS IS THE ONLY PORTABLE CHOLESKY SOURCE IN THE THREE RAPIDS CHECKOUTS.**
Everything else is a closed-library call: cuVS's only factorization from
scratch is `cusolverDnpotrf` + `cusolverDnpotrs`
(`cuvs/src/neighbors/scann/detail/scann_avq.cuh:179-200`) and cuML has none
at all. So this file, and `raft/matrix/detail/matrix.cuh`'s four triangular
and diagonal helpers, are the whole of what `cholesky/impl/` can contain,
and `cholesky/DERIVATION_MAP.tsv` says so rather than implying a wider mirror.

WHO CALLS IT UPSTREAM, corrected against the checkout
------------------------------------------------------
`ML::Solver::Lars::updateCholesky` (`cuml/src/solver/lars_impl.cuh:315-320`),
which is LARS -- least-angle regression -- growing the Gram matrix of its
active set one column per step. **NOT the SVM.** `grep -rn cholesky
cuml/cpp/src/svm/` at this pin returns nothing; cuML's SVM solver is SMO and
touches no factorization. The brief that opened this lane said SVM, the
checkout says LARS, and PORTING_RULES rule 1 says the file wins.

`raft::linalg::cholesky_r1_update.cuh:20-21` also states, in the public
header, that the new mdspan API will NOT be provided for this function -- it
is a legacy raw-pointer entry that RAFT has frozen rather than modernized.

WHAT THEIRS DOES, and it is four vendor calls around a host square root
-----------------------------------------------------------------------
On entry `L` is the factor of the leading `(n-1) x (n-1)` block and the new
row of `A` sits in row `n-1` (lower arm). Then (`:60-118`):

    A_new <- copy of row n-1               cublasCopy, strided, into workspace
    L_12  <- solve L_11 x = A_new          cublastrsm
    s     <- L_12 . L_12                   cublasdot
    row n-1 <- L_12                        cublasCopy back
    L_22  <- sqrt(A_22 - s)                HOST: update_host x2, std::sqrt,
                                           update_device x1
    if eps >= 0 and (isnan(L_22) or L_22 < eps): L_22 = eps
    ASSERT(!isnan(L_22))

# =========================================================================
# DEVIATION 1632: THE THREE cuBLAS CALLS ARE REPLACED BY THIS LANE'S OWN
# PINNED KERNELS, AND THE HOST ROUND TRIP IS KEPT.
#
# `cublasCopy`, `cublastrsm` and `cublasdot` are CLOSED. `VENDOR_LIBS.md`'s
# surviving exception says call the platform equivalent because there is
# nothing to port; here the equivalents are already in this tree and are
# pinned, so:
#
#   cublasCopy  -> `copy_row_to_vector_kernel` / `copy_vector_to_row_kernel`
#                  below. A copy performs no arithmetic, so there is nothing
#                  to pin and nothing to deviate about.
#   cublastrsm  -> `cholesky/checks/trsm.mojo::trsm_lower` at `nrhs = 1`
#                  and `ld` = the factor's stride. That is what the `ld`
#                  parameter of that kernel exists for and this is its only
#                  caller.
#   cublasdot   -> `r1_dot_kernel` below: ONE thread, ascending, one `fma`
#                  per term through `identical_mul_add`, flushed through
#                  `ftz`. A device-wide dot is a fold shape and a fold shape
#                  is a summation order (IDENTITY_PATHS row 21); at the
#                  sizes LARS uses this (one column per active feature) a
#                  serial chain is a pure function of `n` and is the whole
#                  of the pin.
#
# THE HOST ROUND TRIP IS COPIED RATHER THAN OPTIMIZED AWAY. Theirs reads two
# scalars to the host, computes `sqrt` THERE, and writes one back. That is
# two drains and a launch per rank, it is the shape of their algorithm, and
# PORTING_RULES rule 2 says the host/device split is part of the algorithm
# and not an implementation detail to re-decide. It also makes the pivot
# decision a HOST compare on a value already flushed and pinned on the
# device, which is exactly the shape `potrf_lower` uses for `info`.
#
# The host `sqrt` is `identical_sqrt`, not `std::sqrt`: row 18's class says
# a host libm is not one arithmetic across hosts, and under IDENTICAL
# `identical_sqrt` is `portable_sqrtf`, which is float32 basic operations
# only and therefore the same bits on every host and on the device.
# =========================================================================

# =========================================================================
# DEVIATION 1633: THEIR `eps` CLAMP IS A NUMERICAL POLICY, SO UNDER
# `NUMERIC_IDENTICAL` AN UNPINNED `eps` IS REFUSED, AND THE PIVOT IS TESTED
# BEFORE THE SQUARE ROOT RATHER THAN AFTER IT.
#
# Theirs (`:110-115`):
#
#     L_22_host = std::sqrt(L_22_host - s_host);
#     if (eps >= 0 && (std::isnan(L_22_host) || L_22_host < eps)) L_22_host = eps;
#     ASSERT(!std::isnan(L_22_host), "Error during Cholesky rank one update");
#
# TWO CHANGES, both stated:
#
# (a) THE ORDER OF THE TEST. Ours tests `not (v > 0)` on `v = A_22 - s`
#     BEFORE taking the root, exactly as `panel_factor_kernel` does, and
#     only then roots it. Theirs roots first and asks whether the answer is
#     NaN. On a negative `v` the two agree. On a SUBNORMAL `v` they do not:
#     `sqrt` of a subnormal is a small positive number on a column that
#     keeps subnormals and `+0.0` on a column that flushes, and neither is
#     NaN, so their test passes on both columns with two different answers.
#     Ours flushes `v` first (`ftz`) and refuses it on every column.
#     DEVIATION 1634's argument, applied to their file.
#
# (b) `eps` IS PINNED UNDER IDENTICAL. Their default `eps = -1` means "do
#     not clamp, assert instead", and that arm survives verbatim -- ours
#     raises by name with the rank instead of aborting, because this tree
#     raises rather than asserts. A non-negative `eps` REPLACES a refused
#     pivot with a chosen constant, which is DEVIATION 1637's jitter
#     question wearing a different name: it is a number that changes the
#     answer and that no caller writes down. So under IDENTICAL the only
#     accepted values are `eps < 0` (their default, no clamp) and exactly
#     `CHOL_JITTER_PINNED`; under FAST any finite value is honored, which
#     is what a LARS port would need.
#
# Their own header agrees with the direction of this, for what it is worth:
# "for an iterative solver it is probably better to stop early in case of
# error, rather than relying on the eps parameter" (`cholesky_r1_update.cuh
# :47-49`).
# =========================================================================

# =========================================================================
# DEVIATION 1646: THE STRIDED COPY IS A CONTIGUOUS COPY HERE, AND IT IS KEPT
# ANYWAY.
#
# Theirs copies row `n-1` into a contiguous workspace because their storage
# is COLUMN-major, which makes a row non-contiguous (`:66-70`, and the
# comment there says exactly that). This tree is row-major, so row `n-1` IS
# contiguous and the copy could be replaced by a view.
#
# It is kept. The copy is bit-exact and costs `n-1` floats of traffic once
# per rank; deleting it would make the trsm write its solution over the
# input row in place, so a FAILED update (DEVIATION 1633's refusal) would
# leave the caller's matrix holding a half-solved row instead of the
# untouched `A_new` it handed in. Theirs has the same property by accident
# of its layout, and preserving a failure's observable state is worth one
# copy.
#
# The `align = 256` workspace padding (`:50-52`) is NOT ported: it exists so
# cuBLAS gets an aligned scalar and there is no such requirement here.
# `cholesky_rank1_update_workspace_floats` returns `n` floats -- `n-1` for
# the vector and one for the dot -- rather than their byte count.
# =========================================================================
"""

from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace
from cholesky.checks.chol_sabotage import CHOL_SAB_NONE
from cholesky.checks.potrf import (
    CHOL_ELEM_TPB,
    CHOL_PROFILE,
    chol_hex32_bits,
    chol_jitter_pinned,
)
from cholesky.checks.trsm import CHOL_SOLVE_TPB, trsm_lower
from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_mul_add,
    identical_sqrt,
)


def copy_row_to_vector_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin],
    row_in: Int32,
    ld_in: Int32,
    count_in: Int32,
):
    """`cublasCopy(n-1, A_row, ld, A_new, 1)` (`:69-70`), row-major.

    No arithmetic. Row-major makes the source contiguous too, so this is a
    straight block copy; DEVIATION 1646 says why it is kept anyway.
    """
    var row = Int(row_in)
    var ld = Int(ld_in)
    var count = Int(count_in)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= count:
        return
    dst.unsafe_store(idx, src.unsafe_load(row * ld + idx))


def copy_vector_to_row_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin],
    row_in: Int32,
    ld_in: Int32,
    count_in: Int32,
):
    """`cublasCopy(n-1, A_new, 1, A_row, ld)` (`:96-97`), the way back."""
    var row = Int(row_in)
    var ld = Int(ld_in)
    var count = Int(count_in)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= count:
        return
    dst.unsafe_store(row * ld + idx, src.unsafe_load(idx))


def r1_dot_kernel(
    x: MutPointer[Float32, MutAnyOrigin],
    out_scalar: MutPointer[Float32, MutAnyOrigin],
    count_in: Int32,
):
    """`cublasdot(n-1, A_new, 1, A_new, 1, s)` (`:92-93`). ONE thread.

    `s = sum_k x_k^2`, ascending, one `fma` per term, `ftz` at every step.
    DEVIATION 1632 says why it is a serial chain and not a block fold: a
    fold shape is a summation order, and a serial chain is a pure function
    of `count`.

    Spelled `identical_mul_add(v, v, acc)` and NOT `identical_mul_add(-v, v,
    acc)` -- this accumulates the dot product, where `panel_factor_kernel`
    accumulates its NEGATION into the diagonal. The two are different
    expressions and the caller here subtracts, once, on the host.
    """
    if Int(block_idx.x) != 0 or Int(thread_idx.x) != 0:
        return
    var count = Int(count_in)
    var acc = Float32(0.0)
    for k in range(count):
        var v = ftz(x.unsafe_load(k))
        acc = ftz(identical_mul_add(v, v, acc))
    out_scalar.unsafe_store(0, acc)


def cholesky_rank1_update_workspace_floats(n: Int) -> Int:
    """`*n_bytes` (`:53-56`), in floats and without their 256-byte align.

    `n - 1` floats for `A_new` plus one for the dot's scalar. Never less
    than 1, so the buffer is always constructible. DEVIATION 1646.
    """
    var need = n
    if need < 1:
        return 1
    return need


def cholesky_rank1_update(
    ctx: DeviceContext,
    mut l: DeviceBuffer[DType.float32],
    mut workspace: DeviceBuffer[DType.float32],
    n: Int,
    ld: Int,
    eps: Float32,
    mut trace: IdentityTrace,
    tag: StringSlice = "chol.r1",
    tpb: Int = CHOL_SOLVE_TPB,
    elem_tpb: Int = CHOL_ELEM_TPB,
    sabotage: Int = CHOL_SAB_NONE,
) raises:
    """`choleskyRank1Update(handle, L, n, ld, workspace, n_bytes,
    CUBLAS_FILL_MODE_LOWER, stream, eps)` (`:22-118`).

    On entry `l` holds the factor of the leading `(n-1) x (n-1)` block and
    the new row of `A` in row `n-1`, columns `[0, n)`. On exit row `n-1`
    holds the new row of `L` and `L[n-1][n-1]` its new diagonal, so `l` is
    the factor of the whole `n x n` matrix. `ld` is the row stride.

    Raises by name when the new diagonal would not be positive, naming the
    rank -- their `ASSERT` (`:117`), turned into this tree's refusal.

    UPPER is NOT PORTED. Their `uplo == CUBLAS_FILL_MODE_UPPER` arm stores
    `A_new` as a COLUMN and solves `U^T x = A_12` with `CUBLAS_OP_T`; it is
    the arm LARS actually uses (`lars_impl.cuh:271`, `fillmode =
    CUBLAS_FILL_MODE_UPPER`). Nothing in this tree stores an upper factor,
    so porting it would create a second storage convention with no caller.
    `cholesky/NOT_IMPLEMENTED.tsv` records it.

    SYNCHRONIZES, twice per call, exactly as theirs does. Records two card
    stages under `tag`: `<tag>.l12` (the solved new row) and `<tag>.diag`
    (the whole factor after the new diagonal lands).
    """
    if n <= 0:
        raise Error(
            "cholesky_rank1_update: n must be positive, got " + String(n)
        )
    if ld < n:
        raise Error(
            "cholesky_rank1_update: ld="
            + String(ld)
            + " is smaller than n="
            + String(n)
        )
    if len(workspace) < cholesky_rank1_update_workspace_floats(n):
        raise Error(
            "cholesky_rank1_update: the workspace holds "
            + String(len(workspace))
            + " floats, rank "
            + String(n)
            + " needs "
            + String(cholesky_rank1_update_workspace_floats(n))
            + " (cholesky_rank1_update_workspace_floats)"
        )
    _validate_eps(eps)

    var m = n - 1
    var s = Float32(0.0)
    if m > 0:
        var xvec = workspace.create_sub_buffer[DType.float32](0, m)
        var sbuf = workspace.create_sub_buffer[DType.float32](m, 1)
        # `:69-70`
        ctx.enqueue_function[copy_row_to_vector_kernel](
            xvec.unsafe_ptr(),
            l.unsafe_ptr(),
            Int32(n - 1),
            Int32(ld),
            Int32(m),
            grid_dim=((m + elem_tpb - 1) // elem_tpb, 1, 1),
            block_dim=(elem_tpb, 1, 1),
        )
        # `:77-88`: solve L_11 x = A_12, in place over the workspace.
        trsm_lower(
            ctx,
            l,
            xvec,
            m,
            1,
            trace,
            String(tag) + ".l12",
            tpb,
            sabotage,
            ld,
        )
        # `:92-93`
        ctx.enqueue_function[r1_dot_kernel](
            xvec.unsafe_ptr(),
            sbuf.unsafe_ptr(),
            Int32(m),
            grid_dim=(1, 1, 1),
            block_dim=(1, 1, 1),
        )
        # `:95-98`
        ctx.enqueue_function[copy_vector_to_row_kernel](
            l.unsafe_ptr(),
            xvec.unsafe_ptr(),
            Int32(n - 1),
            Int32(ld),
            Int32(m),
            grid_dim=((m + elem_tpb - 1) // elem_tpb, 1, 1),
            block_dim=(elem_tpb, 1, 1),
        )
        var hs = ctx.enqueue_create_host_buffer[DType.float32](1)
        ctx.enqueue_copy(dst_ptr=hs.unsafe_ptr(), src_buf=sbuf)
        ctx.synchronize()
        s = hs.unsafe_ptr().unsafe_load(0)
        _ = hs^
        _ = xvec^
        _ = sbuf^
    # else: `:99-101`, `cudaMemsetAsync(s, 0, ...)` -- the n == 1 case.

    # `:103-117`, on the HOST, and DEVIATION 1633 changes the order of the
    # test but not which values it looks at.
    var hd = ctx.enqueue_create_host_buffer[DType.float32](1)
    var l22 = l.create_sub_buffer[DType.float32]((n - 1) * ld + n - 1, 1)
    ctx.enqueue_copy(dst_ptr=hd.unsafe_ptr(), src_buf=l22)
    ctx.synchronize()
    var a22 = ftz(hd.unsafe_ptr().unsafe_load(0))
    var v = ftz(a22 - ftz(s))
    var newdiag = Float32(0.0)
    if not (v > Float32(0.0)):
        if eps >= Float32(0.0):
            # `:115`: their clamp. `eps` stands in for the refused pivot.
            newdiag = eps
        else:
            raise Error(
                "cholesky_rank1_update: the new diagonal of rank "
                + String(n)
                + " is not positive (A_22 - L_12.L_12 has bits 0x"
                + chol_hex32_bits(v)
                + "), so the extended matrix is not positive definite."
                " Their ASSERT at cholesky_r1_update.cuh:117 aborts here;"
                " this raises. Pass eps >= 0 to clamp instead, and read"
                " DEVIATION 1633 first -- under NUMERIC_IDENTICAL the only"
                " clamp value the profile "
                + CHOL_PROFILE
                + " accepts is the pinned ridge."
            )
    else:
        newdiag = ftz(identical_sqrt(v))
        if eps >= Float32(0.0) and newdiag < eps:
            # `:115`, the second half of their condition.
            newdiag = eps
    hd.unsafe_ptr().unsafe_store(0, newdiag)
    ctx.enqueue_copy(dst_buf=l22, src_ptr=hd.unsafe_ptr())
    ctx.synchronize()
    trace.record_device(ctx, String(tag) + ".diag", l, ld * n)
    _ = hd^
    _ = l22^


def _validate_eps(eps: Float32) raises:
    """DEVIATION 1633 (b). Under IDENTICAL, `eps` is either their default
    "do not clamp" (any negative value) or exactly the pinned ridge."""
    if eps != eps:
        raise Error("cholesky_rank1_update: eps is NaN; refused by name")
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        if eps < Float32(0.0):
            return
        var pinned = chol_jitter_pinned()
        if eps != pinned:
            raise Error(
                "cholesky_rank1_update: NUMERIC_IDENTICAL refuses the"
                " unpinned eps 0x"
                + chol_hex32_bits(eps)
                + ". A non-negative eps REPLACES a refused pivot with a"
                " chosen constant, so it changes the factor by a number the"
                " caller picked and nobody recorded -- DEVIATION 1637's"
                " question about the jitter, wearing RAFT's name for it."
                " The accepted values under IDENTICAL are any negative eps"
                " (their default: do not clamp, refuse instead) and exactly"
                " 0x"
                + chol_hex32_bits(pinned)
                + ". DEVIATION 1633."
            )
