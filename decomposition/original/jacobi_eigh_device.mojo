# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Cyclic Jacobi eigendecomposition ON THE DEVICE.

NOT A PORT of a file, and NOT THE ARM THEIR DISPATCH TAKES. Both halves of
that sentence matter.

`calEig` (`cuml/cpp/src/tsvd/tsvd.cuh:99`) branches on `prms.algorithm`.
**Its default is `solver::COV_EIG_DQ`**
(`cuml/cpp/include/cuml/decomposition/params.hpp:53`), which is
`raft::linalg::eigDC` -> cuSOLVER `syevd`, a divide-and-conquer tridiagonal
eigensolver. `COV_EIG_JACOBI` is the OPT-IN arm, reached only from
`svd_solver='jacobi'` (`cuml/python/cuml/cuml/decomposition/pca.pyx:392-404`,
where `'auto'` and `'full'` both map to `COV_EIG_DQ`).

So this file ports the shape of their SECOND arm, not their default. The
reason is that both arms end in a closed NVIDIA library. `syevd` and `syevj`
are cuSOLVER, there is no source to transliterate, and MAX ships no symmetric
eigensolver at all (`linalg` has `matmul`, `bmm`, `gemv`, `transpose`,
`qr_factorization` and no eigen or SVD entry point). When the path their
dispatch takes calls a closed library and no equivalent exists to call, the
only remaining option is to write one, and between their two named algorithms
Jacobi is the one whose per-rotation arithmetic is small enough to be checked
against a host reference line by line. **That is a substitution and it is
recorded as one**, in `decomposition/NOT_IMPLEMENTED.tsv` and in the lane report.

This is the path a fit takes. `jacobi_eigh.mojo` is the host Float64 oracle
it is checked against, not a CPU fallback: there is no CPU path in this
repository.

WHAT IS PARALLEL AND WHAT IS NOT
--------------------------------
The rotation SEQUENCE is identical to the host version: cyclic by `(p, q)`
pairs, in the same order, so the two agree rotation for rotation. What is
parallel is each rotation's O(n) column, row and basis updates, which run one
thread per index.

The textbook parallel Jacobi instead schedules `n/2` DISJOINT pairs per round
in a round-robin tournament, which is a genuinely different rotation order
and therefore a different (equally valid) answer. Not doing that yet is a
deliberate choice: this version can be diffed against the host reference
rotation by rotation, and a tournament version cannot.

FLOAT32, BECAUSE APPLE HAS NO FLOAT64
-------------------------------------
The host version accumulates in Float64. This one cannot: Metal has no
double, which is why `gpu_portability` refuses device float64 across the
board. Jacobi is unusually forgiving here, because every rotation is exact
orthogonal arithmetic and the basis stays orthonormal to rounding no matter
how many are applied, which is the property that makes it worth its extra
passes on a small matrix.

**The two therefore do not agree bit for bit and are not meant to.**
`jacobi_check.mojo` compares them at a stated tolerance and prints it.

WHAT `IDENTICAL` PROMISES HERE, AND WHY IT IS NOT A ROUNDING QUESTION
---------------------------------------------------------------------
DEVIATION 524, IDENTITY_PATHS row 27. This file had NO ledger row and no
identity construction in it at all: two `block.sum` folds whose cross-lane
stage follows the hardware warp width, and a rotation written as plain
`a*b + c` at every seam.

The fold is the one that is worse than a drift. `off` is compared against
`limit` to decide whether to stop sweeping, so a last-bit disagreement
between a 32-wide fold and AMD's 64-wide one does not perturb the answer,
it changes THE NUMBER OF SWEEPS -- and one more sweep is n(n-1)/2 more
rotations applied to `a` and `v`. That is row 25's shape (a loop whose exit
is decided by a quantity the machine gets a vote in) with a smaller basin,
and `check_jacobi_sweep_count_is_a_knife_edge` measures the basin rather
than asserting it is small.

The three moves, in the ledger's vocabulary:

  REPLACE  both folds, with `core/pinned_reduce.pinned_block_sum` -- a
           halving tree with no lane primitive in it under IDENTICAL, the
           library call bit for bit under FAST. The broadcast the old call
           got from `broadcast=True` is done explicitly here; see
           `_folded_and_broadcast`.
  PIN      every multiply-add through `identical_mul_add`, every
           intermediate through `ftz`, and every `sqrt` through
           `identical_sqrt`: the two partial sums, the rotation's
           `1 + theta^2` and `1 + t^2`, the four update lines (`_rot_sub`,
           `_rot_add`), the `limit` product, and the reported residual. The
           sqrt pin is the NVIDIA column's, not a precaution -- their
           `std.math.sqrt` is one ulp off on 176,577 NORMAL inputs of 2^20
           (E2's H100 `check-ieee-arith`), and this kernel calls it twice
           per rotation.
  REFUSE   the unconverged exit -- and it was ALREADY WIRED, at both
           callers, in BOTH modes: `eig_and_truncate` (`pca.mojo`) and
           `lstsq_eig` (`lstsq.mojo`) raise on `info_out[0] == 0`
           rather than return a snapshot of a sweep budget. DEVIATION
           BLOCK 2 below is that refusal's argument. It needs no mode gate,
           because unlike DBSCAN's cap ours never returns the truncated
           answer to anyone.

WHAT THIS FILE DOES NOT PROMISE: that the ANSWER is identical to some other
implementation's. A halving tree is not CUB's warp-then-block shape and an
`fma` is not a multiply-then-add, so IDENTICAL's bits differ from FAST's on
Apple BY DESIGN. What is bought is that IDENTICAL's bits are the same bits
on every vendor -- and, above the last bit, that the SWEEP COUNT is.

THE BLOCK DIM IS A CONTRACT, NOT A SUGGESTION. Launch this with exactly
`JACOBI_TPB` threads. `pinned_block_sum` writes one threadgroup slot per
thread into a `JACOBI_TPB`-wide slab, so a wider block writes past it and a
narrower one folds a slot nobody wrote. All three call sites pass
`JACOBI_TPB` (`pca.mojo`, `lstsq.mojo`, `jacobi_check.mojo`) and
`check_jacobi_fold_width_is_pinned` gates the value itself.

THERE IS NO SIZE CAP
--------------------
There used to be. The matrix and the basis were two `32 x 32` THREADGROUP
arrays, so `JACOBI_MAX_N = 32` and a PCA at 33 or more features silently
returned something that was not an eigendecomposition of anything. Both
arrays now live in global memory and `JACOBI_MAX_N` is gone. Verified across
16, 32, 33, 64, 128 and 256 by `check_jacobi_device_sizes`, and the check has
reach: re-imposing the cap makes n = 33 fail at `||V^T V - I|| = 0.61`.
"""

from core.pinned_reduce import pinned_block_sum
from original.kernel_matrix import (
    K_LIB_JACOBI_EIGH,
    TARGET_COLUMN,
    lib_block_size_for,
)
from original.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_mul_add,
    identical_sqrt,
)


from std.gpu import thread_idx
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.memory import stack_allocation


# LAUNCH GEOMETRY, NOT A PROBLEM BOUND. One block of 32 threads, and every
# loop over the matrix below is strided by exactly this, so any `n` is
# covered. Nothing `n`-sized lives in threadgroup memory, so no value of `n`
# can exhaust Metal's 32 KB budget (`PORTING.md 1`).
# READ FROM THE MATRIX, not restated here. `original/kernel_matrix.mojo`
# owns every tunable in this tree; changing TARGET_COLUMN there rebuilds
# this kernel for another vendor with no edit in this file.
#
# AND IT IS A NUMERIC ROW WEARING A SCHEDULING ROW'S CLOTHES, which is the
# mistake `numerics.mojo` opens by warning about. This number is the WIDTH
# OF THE FOLD below AND the stride that partitions the matrix into
# per-thread partials, so two columns carrying two values would be two
# summation orders and -- at a knife edge -- two sweep counts. It is safe
# today for a reason that is checkable rather than argued:
# `lib_block_size_for` returns a FLAT 32 for `K_LIB_JACOBI_EIGH` in every
# column, so `TARGET_COLUMN` cannot move it. `check_jacobi_fold_width_is_pinned`
# asserts exactly that and fails if a vendor number ever lands on this row.
# `lib_block_bounds_a_float_fold` does NOT list this kernel and says so
# explicitly ("the fix there is the fold, not the row"); the fold fix is
# done, and the residual row hazard is gated here instead of silently.
comptime JACOBI_TPB = lib_block_size_for[K_LIB_JACOBI_EIGH, TARGET_COLUMN]()

# `raft::linalg::eigJacobi`'s own defaults (`raft/linalg/eig.cuh:108-109`),
# which are also what cuML's Python layer passes for the Jacobi arm
# (`pca.pyx:358` `tol=1e-7`, `pca.pyx:356` `iterated_power=15`).
comptime JACOBI_TOL = 1.0e-7
comptime JACOBI_SWEEPS = 15


@always_inline
def _folded_and_broadcast[tpb: Int](value: Float32) -> Float32:
    """`pinned_block_sum` plus the broadcast the sweep loop cannot do without.

    DEVIATION 524, and the broadcast half is not decoration. The fold this
    replaced was `block_sum[block_size, broadcast=True]`, and the `True`
    was load bearing: the convergence `break` is taken on the folded value
    and a `break` that is not UNIFORM across the block deadlocks on the
    barriers inside the rotation loop.

    `pinned_block_sum` promises only thread 0's return (its docstring says
    so, and its FAST arm is `block_sum` with no `broadcast` argument at
    all, which is exactly the guarantee that would be lost by swapping it
    in naively). So the broadcast is done HERE, explicitly, through one
    threadgroup slot: thread 0 stores, barrier, everyone loads, barrier.
    Relying on the IDENTICAL arm's `red[0]` being readable by every thread
    would be relying on an implementation detail of a file this lane may
    not edit.

    The trailing barrier is what lets the caller write to `a` immediately
    afterwards: no thread reaches a rotation store while another is still
    reading `a` into its partial. That is the barrier the old comment
    credited to the library reduction.
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


@always_inline
def _rot_sub(c: Float32, x: Float32, s: Float32, y: Float32) -> Float32:
    """`c*x - s*y`, ONE spelling on every backend under IDENTICAL.

    DEVIATION 524, IDENTITY_PATHS rows 9 and 10. Written out, this is two
    products and a subtract, and a codegen may contract EITHER product
    into the subtract or neither -- three spellings of one line, chosen
    per backend. Under IDENTICAL the rule is stated rather than left to
    the compiler: **fuse the FIRST product, round the second**, so the
    value is `fma(c, x, -fl(s*y))` with every intermediate flushed through
    `ftz`. The `ftz` on the second product is also what makes the pin
    hold: a rounded, flushed local is not an operand a contraction can
    reach.

    The FAST arm is the ORIGINAL EXPRESSION, character for character, so
    the shipped bits do not move. That is why this wrapper exists instead
    of a bare `identical_mul_add` call at the seam: `identical_mul_add`'s
    FAST arm would have spelled the same line with its addends swapped,
    and an operand swap is free only until a contraction picks a different
    product to fuse.
    """
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return ftz(identical_mul_add(c, x, -ftz(s * y)))
    return c * x - s * y


@always_inline
def _rot_add(s: Float32, x: Float32, c: Float32, y: Float32) -> Float32:
    """`s*x + c*y`, the sibling of `_rot_sub` and the same rule: fuse the
    first product, round the second, flush both. See `_rot_sub`."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return ftz(identical_mul_add(s, x, ftz(c * y)))
    return s * x + c * y


def jacobi_eigh_kernel(
    a_io: MutPointer[Float32, MutAnyOrigin],
    v_out: MutPointer[Float32, MutAnyOrigin],
    info_out: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    max_sweeps_in: Int32,
    tol_in: Float32,
):
    """`a_io` in: the symmetric matrix. Out: diagonal holds the eigenvalues.

    `v_out` ends with eigenvector `i` in COLUMN `i`, LAPACK's convention and
    therefore cuSOLVER's, which is what the caller's ordering code expects.

    `info_out` is three slots and it is not optional bookkeeping.
    `info_out[0]` is 1 if the sweep loop converged and 0 if it hit
    `max_sweeps_in`; `info_out[1]` is the last measured
    `||offdiag(A)||_F / ||A||_F`; `info_out[2]` is the number of sweeps
    executed, which is `cusolverDnXsyevjGetSweeps`. See the DEVIATION BLOCK 2
    note below for why an unchecked sweep limit is not acceptable here even
    though RAFT's Jacobi arm leaves one unchecked.
    """
    var n = Int(n_in)
    var tid = Int(thread_idx.x)

    # THE MATRIX AND THE BASIS LIVE IN GLOBAL MEMORY, NOT SHARED.
    #
    # They used to be two `JACOBI_MAX_N x JACOBI_MAX_N` shared arrays, and
    # that imposed a HARD CAP OF 32 FEATURES on PCA, truncated SVD and OLS.
    # Not a property of Jacobi, not a property of Metal: a consequence of
    # choosing to hold the whole problem in threadgroup memory. At 32 that is
    # 8 KB, at 64 it is 32 KB which is Metal's entire budget, and at 128 it
    # is 128 KB which is impossible. A 128-feature PCA is an ordinary thing
    # to ask for and this refused it.
    #
    # Streaming from global costs bandwidth per rotation and removes the cap
    # entirely. cuSOLVER has no such limit for the same reason: it does not
    # try to hold the matrix on chip.
    #
    # `JACOBI_MAX_N` is gone. There is no maximum.
    var a = a_io
    var v = v_out
    var rot = stack_allocation[
        2,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()

    var idx = tid
    while idx < n * n:
        var r = idx // n
        var c = idx % n
        v.unsafe_store(idx, Float32(1.0) if r == c else Float32(0.0))
        idx += JACOBI_TPB
    barrier()

    # DEVIATION BLOCK 1 -- THE CONVERGENCE TEST IS RELATIVE, NOT ABSOLUTE.
    #
    # THEIRS: cuSOLVER's `syevj` takes a `tol` through
    # `cusolverDnXsyevjSetTolerance` (`raft/linalg/detail/eig.cuh:276`) and
    # stops on the off-diagonal norm measured AGAINST THE MATRIX. RAFT's
    # default is `1.e-7` (`raft/linalg/eig.cuh:108`), a number that only
    # means anything as a relative quantity.
    #
    # OURS, BEFORE: `off <= tol_in` with `tol_in = 1e-10`, where `off` is a
    # sum of SQUARES of the strict upper triangle. That is an ABSOLUTE test
    # on a quantity that scales with the square of the data. On a covariance
    # whose eigenvalues are around 100 it is unreachable, so the loop always
    # ran its full sweep budget; multiply the same data by 1000 and it is
    # unreachable by another six orders of magnitude. **The same matrix in
    # different units converged differently**, which is not a tolerance, it
    # is a bug with a tolerance-shaped name.
    #
    # OURS, NOW: `||offdiag||_F <= tol * ||A||_F`, which is cuSOLVER's
    # quantity and is invariant to the units of the data. `off` is the sum of
    # squares over the STRICT UPPER triangle, so `||offdiag||_F^2 = 2 * off`
    # and the test is `2 * off <= tol^2 * ||A||_F^2`.
    #
    # MEASURED (`check_jacobi_scale_invariance`, and the sweep column of
    # `check_jacobi_device_sizes`): a matrix and the SAME matrix times 1000
    # now both converge in 7 sweeps to a relative off-diagonal of 3.1e-12,
    # and their spectra agree to 3.9e-07 after dividing the scale out. Across
    # n = 16, 32, 33, 64, 128, 256 the executed sweep counts are 5, 6, 6, 8,
    # 8, 9, all inside RAFT's default budget of 15 and none of them near the
    # 80 this file used to be given. Accuracy is UNCHANGED on that fixture
    # (n = 128 orthogonality error 2.7999649614418587e-05 both before and
    # after), so this is a correctness fix and a work reduction, not an
    # accuracy claim.
    var local_f = Float32(0.0)
    var fe = tid
    while fe < n * n:
        var fv = ftz(a.unsafe_load(fe))
        local_f = ftz(identical_mul_add(fv, fv, local_f))
        fe += JACOBI_TPB
    var fro2 = _folded_and_broadcast[JACOBI_TPB](local_f)
    # `limit` is the RIGHT-HAND SIDE OF THE EXIT TEST, so its denormal
    # policy is a sweep count. `tol^2` is 1e-14 at RAFT's default and
    # `fro2` can be anything, so the product reaches the denormal range on
    # any small-scale matrix -- where CUDA keeps a number Metal has already
    # flushed to zero, and the two machines then disagree about whether the
    # matrix ever converges. Both intermediates flushed (row 10's
    # checklist), not just the result.
    var limit = ftz(ftz(tol_in * tol_in) * fro2)

    var executed = 0
    var converged = False
    var last_off = Float32(0.0)

    for _sweep in range(Int(max_sweeps_in)):
        # `cub::BlockReduce`'s counterpart. This REPLACED a `tid == 0` serial
        # double loop over the strict upper triangle, run once per sweep with
        # 31 of the 32 threads idle. Every thread now accumulates a strided
        # slice of the SAME triangle and the block reduces them. See
        # VENDOR_LIBRARIES.md.
        #
        # DEVIATION BLOCK 3 -- THE FOLD THAT DECIDES A SWEEP COUNT.
        #
        # **THE COMMENT THAT STOOD HERE WAS WRONG AND IS DELETED.** It said
        # the summation order can move `off`'s last bits but that "a last-bit
        # difference can only move which sweep a knife-edge matrix stops on,
        # never the answer it stops at". The second half does not follow from
        # the first. `off` is compared against `limit` to decide whether to
        # `break`, so a last-bit difference does not perturb the answer, it
        # changes THE NUMBER OF SWEEPS, and one more sweep is n(n-1)/2 more
        # rotations applied to both `a` and `v`. A different sweep count is a
        # different matrix -- not in the last bit, in the fifth decimal --
        # and where two eigenvalues are close it can also reorder the
        # spectrum, which sends `eig_and_truncate`'s sort a DIFFERENT
        # component.
        #
        # MEASURED, TWICE, AND THE NUMBERS ARE THE POINT.
        # `check_jacobi_sweep_count_is_a_knife_edge`: at n = 64 there are two
        # ADJACENT Float32 tolerances -- one ulp apart, bits 877130934 and
        # 877130935 -- across which the executed sweeps go 7 -> 6 and 1,325
        # of the 4,096 eigenvector cells come back with different BITS, worst
        # |delta| 1.9e-06. Both answers pass every tolerance check in this
        # file (5e-5), which is precisely why no tolerance check was ever
        # going to catch this.
        # `check_jacobi_fold_shape_decides_the_sweep_count`: at n = 32 the
        # halving fold and a sequential fold of the same partials change
        # their mind at tolerances ONE ULP apart, and at a tolerance between
        # them they execute 6 and 7 sweeps on the same input and end 190 of
        # 1,024 eigenvector cells apart. The fold shape is not a rounding
        # detail here, it is a control-flow input.
        #
        # This is IDENTITY_PATHS row 25's shape (a loop whose exit condition
        # is decided by a quantity the machine gets a vote in) with a smaller
        # basin, and it is why the fold below is not a library call any more:
        # `max.gpu.primitives.block.sum` folds its cross-lane stage at the
        # HARDWARE warp width -- 32 on Apple and NVIDIA, 64 on AMD's CDNA
        # wavefront -- so Apple and AMD compute two different sums of the
        # same multiset, and at a knife edge that is two different answers.
        # `pinned_block_sum` is a halving tree in threadgroup memory with no
        # lane primitive in it under IDENTICAL, and the library call bit for
        # bit under FAST.
        #
        # The per-thread partial is pinned too, and has to be: `acc + v*v` is
        # one rounding or two at the codegen's whim (row 9), so an unpinned
        # partial would put the vendors back where the fold just took them
        # out of.
        var local_off = Float32(0.0)
        var e = tid
        while e < n * n:
            var i = e // n
            var j = e - i * n
            if j > i:
                var av = ftz(a.unsafe_load(e))
                local_off = ftz(identical_mul_add(av, av, local_off))
            e += JACOBI_TPB
        var off = _folded_and_broadcast[JACOBI_TPB](local_off)
        last_off = off
        if Float32(2.0) * off <= limit:
            converged = True
            break
        executed += 1

        for p in range(n):
            for q in range(p + 1, n):
                # DEVIATION BLOCK 4 -- THE ROTATION ITSELF.
                #
                # Every line here is a float seam and three of them decide a
                # BRANCH, which is the shape that turns a last bit into a
                # different rotation rather than a different last bit:
                #
                #   * `apq == 0` selects the IDENTITY rotation. A DENORMAL
                #     `apq` is zero on Metal and nonzero on a CUDA default
                #     build, so without the flush the two vendors apply
                #     different rotations, not differently-rounded ones.
                #     `ftz` aligns them (row 10).
                #   * `theta >= 0` picks which of the two tan formulas runs.
                #     `theta` is a quotient of flushed operands, so it is one
                #     value everywhere and the branch follows it.
                #   * `1 + theta*theta` and `1 + t*t` are multiply-adds and go
                #     through `identical_mul_add` (row 9).
                #
                # `1.0 / sqrt(x)` is written with the sqrt's result stored and
                # flushed FIRST, deliberately: an unpinned `1/sqrt` is exactly
                # the expression a backend is entitled to lower to a
                # reciprocal-sqrt approximation, and the flushed local makes
                # that rewrite unavailable under IDENTICAL.
                #
                # THE SQRT ITSELF GOES THROUGH `identical_sqrt`, AND THAT IS
                # NOT BELT AND BRACES. The sentence that would have stood
                # here -- "division and sqrt are IEEE-correct on normals on
                # every backend measured" -- was true of two vendors and
                # false of the third: `check-ieee-arith` on an H100 (E2,
                # 2026-08-23) found Mojo's `std.math.sqrt` lowering to an
                # APPROXIMATE PTX sqrt, 180,714 of 2^20 hashed patterns off
                # by one ulp and 176,577 of those on NORMAL inputs. Two
                # sqrts per rotation at one ulp is a different `t`, a
                # different `c`, a different `s` -- and at a knife edge a
                # different SWEEP COUNT. `identical_sqrt` is row 10's seam
                # call: `portable_sqrtf` under IDENTICAL (correctly rounded
                # by construction, so Metal and HIP bits do not move), the
                # stdlib verbatim under FAST.
                if tid == 0:
                    var apq = ftz(a.unsafe_load(p * n + q))
                    if apq == Float32(0.0):
                        rot[0] = Float32(1.0)
                        rot[1] = Float32(0.0)
                    else:
                        var aqq = ftz(a.unsafe_load(q * n + q))
                        var app = ftz(a.unsafe_load(p * n + p))
                        var theta = ftz(
                            ftz(aqq - app) / ftz(Float32(2.0) * apq)
                        )
                        var root = ftz(
                            identical_sqrt(
                                ftz(
                                    identical_mul_add(
                                        theta, theta, Float32(1.0)
                                    )
                                )
                            )
                        )
                        var t = Float32(0.0)
                        if theta >= Float32(0.0):
                            t = ftz(Float32(1.0) / ftz(theta + root))
                        else:
                            t = ftz(Float32(-1.0) / ftz(root - theta))
                        var croot = ftz(
                            identical_sqrt(
                                ftz(identical_mul_add(t, t, Float32(1.0)))
                            )
                        )
                        var c = ftz(Float32(1.0) / croot)
                        rot[0] = c
                        rot[1] = ftz(t * c)
                barrier()

                var c = rot[0]
                var s = rot[1]

                # Columns p and q, one thread per row. `_rot_sub` / `_rot_add`
                # are the pinned spelling of these four lines; every load is
                # flushed on the way in and every store on the way out, which
                # is row 10's seam rule applied to a buffer this kernel hands
                # back to the caller.
                var k = tid
                while k < n:
                    var akp = ftz(a.unsafe_load(k * n + p))
                    var akq = ftz(a.unsafe_load(k * n + q))
                    a.unsafe_store(k * n + p, _rot_sub(c, akp, s, akq))
                    a.unsafe_store(k * n + q, _rot_add(s, akp, c, akq))
                    k += JACOBI_TPB
                barrier()

                # Rows p and q, one thread per column.
                k = tid
                while k < n:
                    var apk = ftz(a.unsafe_load(p * n + k))
                    var aqk = ftz(a.unsafe_load(q * n + k))
                    a.unsafe_store(p * n + k, _rot_sub(c, apk, s, aqk))
                    a.unsafe_store(q * n + k, _rot_add(s, apk, c, aqk))
                    k += JACOBI_TPB
                barrier()

                # The accumulated basis.
                k = tid
                while k < n:
                    var vkp = ftz(v.unsafe_load(k * n + p))
                    var vkq = ftz(v.unsafe_load(k * n + q))
                    v.unsafe_store(k * n + p, _rot_sub(c, vkp, s, vkq))
                    v.unsafe_store(k * n + q, _rot_add(s, vkp, c, vkq))
                    k += JACOBI_TPB
                barrier()

    # DEVIATION BLOCK 2 -- WE REPORT NON-CONVERGENCE. RAFT'S JACOBI ARM
    # DOES NOT.
    #
    # THEIRS: `detail::eigJacobi` calls `cusolverDnXsyevjGetSweeps` into a
    # local `int executed_sweeps` and then **never reads it**
    # (`raft/linalg/detail/eig.cuh:310-311`). It does not check `dev_info`
    # either. A `syevj` that exhausts its 15 sweeps returns an unconverged
    # answer to cuML with no signal at all.
    #
    # THEIRS, ON THE DEFAULT PATH: `eigDC` -- the arm `svd_solver='auto'`
    # actually reaches -- DOES check, and aborts with "eigensolver couldn't
    # converge to a solution" (`raft/linalg/detail/eig.cuh:79-82`).
    #
    # OURS: we follow the DEFAULT arm's behaviour, because a silently
    # unconverged eigendecomposition is the same class of defect as the
    # 32-feature cap this file just lost: a wrong answer with no error. The
    # host reference in `jacobi_eigh.mojo` already raises on the same
    # condition. Two slots are written and `eig_and_truncate` raises on them.
    if tid == 0:
        info_out.unsafe_store(0, Float32(1.0) if converged else Float32(0.0))
        var rel = Float32(0.0)
        if fro2 > Float32(0.0):
            rel = ftz(
                identical_sqrt(ftz(ftz(Float32(2.0) * last_off) / fro2))
            )
        info_out.unsafe_store(1, rel)
        info_out.unsafe_store(2, Float32(executed))

    # No write-back: `a` and `v` ARE the caller's buffers now.
