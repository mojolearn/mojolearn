# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Tall-skinny Householder QR on the device, block parallel, ROW MAJOR.

WHY THIS FILE EXISTS AND WHAT IT DID NOT COPY
=============================================
`decomposition/NOT_IMPLEMENTED.tsv` named ONE missing primitive behind two
refusals, `PCA(svd_solver='full')` and `TruncatedSVD(algorithm='randomized')`:
"a tall-skinny Householder QR, bit-identical across three vendors". The arima
lane wrote a Householder QR on 2026-09-01 as DEVIATION 678
(`arima/impl/linalg/batched/least_squares.mojo`) and gated it
(`check_qr_device_equals_oracle` bitwise against a separately spelled host
replay, and `check_qr_beats_normal_equations_on_ill_conditioning`, worst QR
error 7.4e-07 against 1.5e-04 for the normal equations, strictly better on 6
of 6 series).

THE QUESTION THIS FILE ANSWERS IS WHICH PART OF THAT IS REUSABLE, and the
answer is not "all of it" and not "none of it".

WHAT IS REUSED (DEVIATION 586). The three scalar decisions DEVIATION 678
made and wrote down live here now, ONCE, and nowhere else:

    `qr_reflector_r`     the reflector's SIGN. `s = -sign(a_jj)` with
                         `sign(+-0)` taken as `+1`, so `r_jj = s * normx`
                         and the `u1` below ADDS two same-signed terms
                         instead of subtracting near-equal ones. This is the
                         classical cancellation the formulation exists to
                         avoid; `arima/SABOTAGES.md` arm (l) flips it and
                         `check_full_reflector_sign_earns_its_place` in this
                         tree's decomposition lane flips it again on a
                         near-collinear fixture.
    `qr_reflector_u1`    `u1 = a_jj - r_jj`.
    `qr_reflector_tau`   `tau = (-s * u1) / normx`.

A gated numeric choice copied into a second file is how two spellings drift,
and this tree has a standing rule about exactly that. Lifting the THREE LINES
that are a CHOICE, rather than the loop that is a launch geometry, is the
largest share that can be lifted without changing anybody's bits.

WHAT IS NOT REUSED, AND THIS IS NOT A DEFECT IN EITHER FILE. arima's routine
is ONE THREAD PER SERIES, serial ascending in `m`, with `m` a few hundred and
`n <= 17`, and its own banner says so ("at `m = 1e5` and `n = 16` it is about
5e7 flops in ONE thread and it will be visibly slow"). PCA's shape is ONE
matrix, `n_samples x n_features`, and `n_samples` is the shipped size
(`large-data-runs-default`): thousands to millions of rows against tens to
a couple of hundred of columns. Importing the serial routine where it sits
would compile, run, and be unusable at the size the library ships, which is
a capability in name only.

So the ARITHMETIC PER OPERATION is shared and the FOLD IS NOT.

DEVIATION 587: THE FOLD IS A BLOCK FOLD, NOT SERIAL ASCENDING.
--------------------------------------------------------------
Every inner product here is a per-thread strided partial (serial ascending
within the thread, `identical_mul_add`, `ftz`) folded by
`core.pinned_reduce.pinned_block_sum`, which is a halving tree with no lane
primitive in it under IDENTICAL and the library call bit for bit under FAST.

THIS IS NOT arima's SUM AND IS NOT MEANT TO BE. A halving tree over `QR_TPB`
partials and a single ascending scan combine different partials, so the two
routines return different bits on the same input. What is promised is what
IDENTICAL promises everywhere in this tree: the bits are the SAME BITS on
Apple, NVIDIA and AMD, because the fold's association is a pure function of
`QR_TPB` and nothing in it consults the hardware's lane width.

`QR_TPB` IS A NUMERIC ROW WEARING A SCHEDULING ROW'S CLOTHES, exactly as
`JACOBI_TPB` is in `decomposition/checks/jacobi_eigh_device.mojo`. It is the
WIDTH OF THE FOLD, so two vendor columns carrying two values would be two
summation orders. It is therefore a FLAT constant here and deliberately not
read from `checks/kernel_matrix.mojo`, the same shape as `SIGNFLIP_TPB` in
`decomposition/impl/linalg/detail/pca.mojo`.
`check_qr_fold_width_is_pinned` asserts the value and that it is a power of
two, which `pinned_block_sum`'s halving fold requires.

DEVIATION 588: THERE IS NO RANK TEST ON THIS PATH, AND THAT IS DELIBERATE.
--------------------------------------------------------------------------
DEVIATION 678's routine returns `info = j + 1` when a diagonal falls below
`LS_RANK_TOL` times the largest, and it is RIGHT to: it then divides by that
diagonal in a back substitution, and a solution divided by two significant
bits is noise. THIS FILE NEVER BACK SUBSTITUTES. Its output is `R`, and its
consumer takes an SVD of `R`, where a rank-deficient `R` is not a failure but
the correct representation of a ZERO SINGULAR VALUE.

The difference is a capability, not a nicety. A constant feature column is
centered to exactly zero, and PCA on such a column is an ordinary thing to
ask for; the covariance arm this lane already ships returns a zero eigenvalue
for it without complaint. Importing arima's refusal here would make
`svd_solver='full'` raise on data `svd_solver='jacobi'` accepts, which is a
regression wearing a safety check's clothes.

So a column whose norm is zero (after `ftz`, which is the flush the branch
is decided on -- IDENTITY_PATHS row 10, and the reason the test is on
`normx` and not on `sigma`) gets `r_jj = 0`, no reflector, `H_j = I`, and the
loop continues. LAPACK's `geqrf` does the same thing and leaves the rank
question to the caller. `check_full_survives_a_constant_column` is the gate
and its sabotage arm is arima's rank test reinstated, which must raise.

DEVIATION 589: TSQR, AND THE SLICING IS A PURE FUNCTION OF THE SHAPE.
---------------------------------------------------------------------
One block over `m` rows is `O(m n^2 / QR_TPB)` on ONE core of the GPU. The
standard fix for a tall-skinny matrix is TSQR: cut the rows into slices,
factor each slice in its own block, stack the slices' `R` factors and factor
THAT. Every slice's `R` satisfies `R_b^T R_b = A_b^T A_b`, so the stacked
factorization's `R` satisfies `R^T R = A^T A` -- which is every property the
SVD consumer needs, and it is reached without ever FORMING `A^T A`, which is
the entire point of taking this route instead of the covariance one.

THE R IS NOT THE R A SINGLE HOUSEHOLDER PASS WOULD PRODUCE. It is a valid
`R` for the same matrix, differing by the signs of its rows and by rounding.
Nothing downstream reads `R` itself; the gates are on `R^T R`, on the
spectrum, and on the reconstruction error, none of which can see the
difference. `check_tsqr_agrees_with_one_block` measures it anyway, because a
claim that two routes agree is worth a number.

THE SLICE COUNT MUST NOT DEPEND ON THE MACHINE. `qr_slice_count` is a pure
function of `(n_rows, n_cols)` -- no core count, no occupancy table, no
`TARGET_COLUMN`. A vendor-keyed slice count would be a different reduction
tree on every vendor, which is DEVIATION 587's hazard moved up one level and
made much larger.

ROW MAJOR, WHERE arima IS COLUMN MAJOR
---------------------------------------
arima's `A` is column major because `cublasgelsBatched`'s contract is.
This tree's `X` is row major everywhere (`core/column_stats.mojo` reads
`x[r * n_cols + col]`), so a column-major QR here would cost a transpose of
the FULL data matrix, which is the only `O(rows)` buffer in the fit. The
indexing changes; not one arithmetic operation does.

THROUGHPUT, NAMED SO IT IS NOT DISCOVERED AS A SURPRISE. TSQR spreads the
row pass over at most `QR_MAX_SLICES = 64` blocks, and each block is
`QR_TPB = 32` threads, so the arithmetic is `O(m n^2)` over at most 2,048
threads with `n^2 / 2` block folds per slice. At the shipped PCA shapes
(`m` in the thousands to low millions, `n <= 128`) that is a real pass over
the data and not a placeholder, but it is NOT a tuned kernel and it is a
fraction of a modern GPU: the covariance arm reaches the same data through a
GEMM that saturates it. This arm is carried for ACCURACY, not for speed, and
the two are not benchmarked against each other here because nothing has been
run yet. If it ever matters, the fix is a panel/blocked update (apply several
reflectors at once through a small `T` factor, which is what LAPACK's `geqrf`
does above its unblocked `geqr2`) and a wider block, neither of which changes
the arithmetic this file gates -- both change the FOLD, which is DEVIATION
587's ground and would need its numbers re-taken. THROUGHPUT DEVIATION,
CORRECTNESS UNAFFECTED, UNMEASURED.

WHAT THIS FILE DOES NOT CARRY YET, NAMED SO IT IS NOT DISCOVERED AS A
SURPRISE: `orgqr`, the explicit thin `Q`. Nothing on the `svd_solver='full'`
route needs it -- R-SVD's right singular vectors and singular values come
from `R` alone -- but `raft`'s randomized SVD does (`rsvd.cuh:198`, `:218`,
`:241` call `qrGetQ`), so the randomized arm stays refused and
`decomposition/NOT_IMPLEMENTED.tsv` names `qr_get_q` as one of the three
pieces it now waits on instead of the whole QR.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation

from checks.numerics import (
    ftz,
    identical_div,
    identical_mul_add,
    identical_sqrt,
)
from core.pinned_reduce import pinned_block_sum


#: THE WIDTH OF THE FOLD. See DEVIATION 587 in the banner: this is a numeric
#: constant, not a scheduling one, and it is flat on purpose.
#: `pinned_block_sum`'s halving tree requires a power of two.
comptime QR_TPB = 32

#: The most row slices TSQR will cut, and the smallest slice it will accept,
#: expressed in COLUMNS so the rule scales with the problem rather than with
#: a row count somebody guessed. A slice narrower than its own column count
#: cannot produce a full-rank `R`, and one only a little wider produces a
#: badly conditioned one, so four columns' worth of rows is the floor.
comptime QR_MAX_SLICES = 64
comptime QR_SLICE_ROWS_PER_COL = 4


def qr_slice_count(n_rows: Int, n_cols: Int) -> Int:
    """How many row slices TSQR cuts, from the SHAPE ALONE.

    DEVIATION 589. Deliberately not a function of the device: two vendors
    that sliced differently would build two different reduction trees and
    IDENTICAL would be a claim about nothing. Halving from `QR_MAX_SLICES`
    keeps the answer a power of two, which makes the row boundaries
    `(b * m) // s` land on the same integers a doubled or halved run would,
    and makes `check_tsqr_agrees_with_one_block` a sweep over a short ladder
    rather than over every integer.
    """
    var s = QR_MAX_SLICES
    while s > 1 and n_rows < s * (QR_SLICE_ROWS_PER_COL * n_cols):
        s //= 2
    return s


# ===========================================================================
# DEVIATION 586: THE REFLECTOR. THE ONLY COPY IN THE TREE OF DEVIATION 678's
# THREE CHOICES.
# ===========================================================================


@always_inline
def qr_reflector_sign(ajj: Float32) -> Float32:
    """`s = -sign(a_jj)`, with `sign(+-0)` taken as `+1` so `s = -1` there.

    Written as a comparison against `+0.0` rather than through a `sign`
    call because `a_jj` can be a negative zero and `-0.0 >= 0.0` is TRUE in
    IEEE-754, which is the answer this wants: a zero column is about to be
    caught by the `normx == 0` test anyway, and until then the branch must
    be the same branch on every backend.
    """
    return Float32(-1.0) if ajj >= Float32(0.0) else Float32(1.0)


@always_inline
def qr_reflector_r(ajj: Float32, normx: Float32) -> Float32:
    """`r_jj = s * normx`, the diagonal of `R` at column `j`."""
    return ftz(qr_reflector_sign(ajj) * normx)


@always_inline
def qr_reflector_u1(ajj: Float32, r_jj: Float32) -> Float32:
    """`u1 = a_jj - r_jj`. THE POINT OF THE SIGN: `r_jj` carries the sign
    OPPOSITE `a_jj`, so this subtraction is an ADDITION of magnitudes and
    `|u1| >= normx`. With the other sign it is a difference of near-equal
    numbers exactly when the column is already nearly axis-aligned."""
    return ftz(ajj - r_jj)


@always_inline
def qr_reflector_tau(ajj: Float32, normx: Float32, u1: Float32) -> Float32:
    """`tau = (-s * u1) / normx`, the reflector's scalar.

    THE DIVISION IS `identical_div` HERE AND A BARE `/` IN arima's COPY,
    and that difference is recorded rather than glossed. `identical_div`
    routes through `portable_divf` under IDENTICAL, which is bit-inert on
    Apple and on every column that flushes denormals in hardware, and moves
    only on a denormal-honoring column with a subnormal operand. New code
    on the identity path pays the pinned spelling by default
    (IDENTITY_PATHS row 49); arima's is a SHIPPED GATED spelling and moving
    it means moving `arima/checks/fit_oracle.mojo::householder_qr_solve_host`
    in the same commit so `check_qr_device_equals_oracle` still holds. That
    patch is written down in this pass's lane report and is NOT applied
    here.
    """
    var s = qr_reflector_sign(ajj)
    return ftz(identical_div(ftz(ftz(-s) * u1), normx))


@always_inline
def fold_and_broadcast[tpb: Int](value: Float32) -> Float32:
    """`pinned_block_sum` plus the broadcast every branch below needs.

    Same construction and same reason as
    `decomposition/checks/jacobi_eigh_device.mojo::_folded_and_broadcast`:
    `pinned_block_sum` promises only thread 0's return (its FAST arm is
    `block_sum` with no `broadcast` argument at all), and the value folded
    here DECIDES A BRANCH -- the zero-column test of DEVIATION 588 -- so a
    branch taken on a value only thread 0 holds is a non-uniform branch, and
    a non-uniform branch over the barriers below deadlocks.

    Duplicated rather than imported because `core/` must not import
    `decomposition/`, and because what is duplicated is a barrier pattern
    with no arithmetic in it. The FOLD itself is the shared
    `pinned_block_sum`, which is the part that could drift.
    """
    var s = pinned_block_sum[tpb](value)
    var slot = stack_allocation[
        1,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    if Int(thread_idx.x) == 0:
        slot.unsafe_store(0, s)
    barrier()
    var out = slot.unsafe_load(0)
    barrier()
    return out


def qr_panel_kernel(
    a: MutPointer[Float32, MutAnyOrigin],
    r_out: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    lda_in: Int32,
    n_slices_in: Int32,
):
    """One block per ROW SLICE. Factors the slice in place and writes its
    `n x n` upper-triangular `R` into `r_out` at `block_idx.x * n * n`.

    `a` is `m x lda` row major and the factorization DESTROYS it, which is
    `geqrf`'s contract and arima's ("THE CALLER MUST PASS A COPY").
    `r_out` must hold `n_slices * n * n` floats.

    LAUNCH WITH EXACTLY `QR_TPB` THREADS. `pinned_block_sum` writes one
    threadgroup slot per thread into a `QR_TPB`-wide slab, so a wider block
    writes past it and a narrower one folds a slot nobody wrote. This is the
    same contract `jacobi_eigh_kernel` states, for the same reason.

    NOTHING `n`-SIZED LIVES IN THREADGROUP MEMORY. The diagonal of `R` is
    written straight into `r_out`, so there is no `QR_MAX_COLS` and no
    repeat of the `JACOBI_MAX_N = 32` cap this section spent a round
    removing.
    """
    var m = Int(m_in)
    var n = Int(n_in)
    var lda = Int(lda_in)
    var n_slices = Int(n_slices_in)
    var b = Int(block_idx.x)
    var tid = Int(thread_idx.x)

    # The slice boundaries. Integer arithmetic on the shape only; every
    # block computes every other block's boundaries the same way, and so
    # does the host.
    var rb = (b * m) // n_slices
    var re = ((b + 1) * m) // n_slices
    var ms = re - rb
    var rbase = b * n * n

    for j in range(n):
        # --- the column norm of A[j:ms, j], strided partials then a fold ---
        var acc = Float32(0.0)
        var i = j + tid
        while i < ms:
            var v = ftz(a.unsafe_load((rb + i) * lda + j))
            acc = ftz(identical_mul_add(v, v, acc))
            i += QR_TPB
        var sigma = fold_and_broadcast[QR_TPB](acc)
        # THE TEST IS ON `normx`, NOT ON `sigma`. `identical_sqrt` of a
        # subnormal `sigma` can flush to zero through `ftz`, and then a
        # `sigma != 0` guard would let a zero `normx` reach the division in
        # `qr_reflector_tau`. Testing the value that is actually divided by
        # is the guard that cannot be skipped past.
        var normx = ftz(identical_sqrt(sigma))
        var ajj = ftz(a.unsafe_load((rb + j) * lda + j))

        # `normx` came out of a BROADCAST fold, so every thread of the block
        # holds the same bits and this branch is uniform. That is what makes
        # the barriers inside both arms legal.
        if normx == Float32(0.0):
            # DEVIATION 588: rank deficiency is a ZERO SINGULAR VALUE here,
            # not a refusal. `H_j = I`, `R_jj = 0`, carry on.
            if tid == 0:
                r_out.unsafe_store(rbase + j * n + j, Float32(0.0))
            barrier()
        else:
            var r_jj = qr_reflector_r(ajj, normx)
            var u1 = qr_reflector_u1(ajj, r_jj)
            var tau = qr_reflector_tau(ajj, normx, u1)
            if tid == 0:
                r_out.unsafe_store(rbase + j * n + j, r_jj)

            # Pack `w` into the subdiagonal with `w_j = 1` implicit, which
            # is LAPACK's layout and arima's.
            #
            # `u1` cannot be zero when `normx` is not: with `s = -sign(ajj)`
            # the two terms of `ajj - s*normx` have the same sign and
            # `|u1| >= normx > 0`. That is the sign's whole job, so the
            # division needs no second guard -- and if the sign is ever
            # sabotaged, this is one of the places it shows.
            var i2 = j + 1 + tid
            while i2 < ms:
                var cur = ftz(a.unsafe_load((rb + i2) * lda + j))
                a.unsafe_store(
                    (rb + i2) * lda + j, ftz(identical_div(cur, u1))
                )
                i2 += QR_TPB
            barrier()

            # Apply `H = I - tau w w'` to the trailing columns.
            for c in range(j + 1, n):
                var dacc = Float32(0.0)
                var i3 = j + 1 + tid
                while i3 < ms:
                    var w = ftz(a.unsafe_load((rb + i3) * lda + j))
                    var x = ftz(a.unsafe_load((rb + i3) * lda + c))
                    dacc = ftz(identical_mul_add(w, x, dacc))
                    i3 += QR_TPB
                var tail = fold_and_broadcast[QR_TPB](dacc)
                var ajc = ftz(a.unsafe_load((rb + j) * lda + c))
                # arima SEEDS the fold with `a[j][c]` and adds upward; a
                # block fold cannot, so the implicit `w_j = 1` term is added
                # to the folded tail instead. Same multiset, different
                # association: DEVIATION 587, stated at the one line where
                # the two routines visibly part.
                var total = ftz(ajc + tail)
                var td = ftz(tau * total)
                if tid == 0:
                    a.unsafe_store((rb + j) * lda + c, ftz(ajc - td))
                var i4 = j + 1 + tid
                while i4 < ms:
                    var w2 = ftz(a.unsafe_load((rb + i4) * lda + j))
                    var cur2 = ftz(a.unsafe_load((rb + i4) * lda + c))
                    a.unsafe_store(
                        (rb + i4) * lda + c,
                        ftz(identical_mul_add(-td, w2, cur2)),
                    )
                    i4 += QR_TPB
                barrier()

    # The strict upper triangle of `R` is row `j` of the factored slice; the
    # diagonal was written above and is not touched here. A slice with fewer
    # rows than columns (only reachable at `n_slices == 1`, where the host
    # refuses it, or on the second TSQR pass, where `ms = n_slices * n >= n`)
    # would read past its own rows, so those cells are zeroed instead.
    barrier()
    var t = tid
    while t < n * n:
        var rr = t // n
        var cc = t - rr * n
        if cc > rr:
            if rr < ms:
                r_out.unsafe_store(
                    rbase + t, ftz(a.unsafe_load((rb + rr) * lda + cc))
                )
            else:
                r_out.unsafe_store(rbase + t, Float32(0.0))
        elif cc < rr:
            r_out.unsafe_store(rbase + t, Float32(0.0))
        t += QR_TPB


def qr_factor(
    ctx: DeviceContext,
    mut a: DeviceBuffer[DType.float32],
    mut r_scratch: DeviceBuffer[DType.float32],
    mut r_out: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_cols: Int,
    slices_override: Int = 0,
) raises -> Int:
    """`R` of the `n_rows x n_cols` row-major `a`, into `r_out` (`n x n`).

    DESTROYS `a`. `r_scratch` must hold `qr_slice_count(...) * n_cols^2`
    floats and is untouched when the slice count is 1. Returns the slice
    count that ran, which is what `check_tsqr_agrees_with_one_block` sweeps
    and what the identity card records.

    `slices_override` is for the gates only. It is NOT a tuning knob: a
    caller that passes a machine-dependent value has broken DEVIATION 589's
    whole argument, so the shipped callers pass nothing.
    """
    if n_rows < n_cols:
        raise Error(
            "qr_factor needs at least as many rows as columns, got "
            + String(n_rows)
            + " x "
            + String(n_cols)
            + ". The route for a wide matrix is an LQ factorization of the"
            " transpose, which this file does not carry; see DEVIATION 593"
            " in decomposition/impl/linalg/detail/svd_full.mojo"
        )
    var ns = qr_slice_count(n_rows, n_cols) if slices_override <= 0 else slices_override
    if ns < 1:
        raise Error("qr_factor slice count must be at least 1")
    if ns == 1:
        ctx.enqueue_function[qr_panel_kernel](
            a.unsafe_ptr(),
            r_out.unsafe_ptr(),
            Int32(n_rows),
            Int32(n_cols),
            Int32(n_cols),
            Int32(1),
            grid_dim=(1, 1, 1),
            block_dim=(QR_TPB, 1, 1),
        )
        ctx.synchronize()
        return 1
    ctx.enqueue_function[qr_panel_kernel](
        a.unsafe_ptr(),
        r_scratch.unsafe_ptr(),
        Int32(n_rows),
        Int32(n_cols),
        Int32(n_cols),
        Int32(ns),
        grid_dim=(ns, 1, 1),
        block_dim=(QR_TPB, 1, 1),
    )
    # The `ns` tiles are contiguous `n x n` row-major blocks, so the stack
    # of them IS an `(ns * n) x n` row-major matrix with leading dimension
    # `n`. No copy and no transpose: TSQR's second pass is the SAME kernel
    # on the SAME layout, which is the reason the tiles are stored this way.
    ctx.enqueue_function[qr_panel_kernel](
        r_scratch.unsafe_ptr(),
        r_out.unsafe_ptr(),
        Int32(ns * n_cols),
        Int32(n_cols),
        Int32(n_cols),
        Int32(1),
        grid_dim=(1, 1, 1),
        block_dim=(QR_TPB, 1, 1),
    )
    ctx.synchronize()
    return ns
