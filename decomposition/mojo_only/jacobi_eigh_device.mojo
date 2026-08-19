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
recorded as one**, in `decomposition/UNPORTED.tsv` and in the lane report.

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

THERE IS NO SIZE CAP
--------------------
There used to be. The matrix and the basis were two `32 x 32` THREADGROUP
arrays, so `JACOBI_MAX_N = 32` and a PCA at 33 or more features silently
returned something that was not an eigendecomposition of anything. Both
arrays now live in global memory and `JACOBI_MAX_N` is gone. Verified across
16, 32, 33, 64, 128 and 256 by `check_jacobi_device_sizes`, and the check has
reach: re-imposing the cap makes n = 33 fail at `||V^T V - I|| = 0.61`.
"""

from mojo_only.kernel_matrix import (
    K_LIB_JACOBI_EIGH,
    TARGET_COLUMN,
    lib_block_size_for,
)


from std.gpu import thread_idx
from std.math import sqrt
from max.gpu.memory import AddressSpace
from max.gpu.primitives.block import sum as block_sum
from max.gpu.sync import barrier
from std.memory import stack_allocation


# LAUNCH GEOMETRY, NOT A PROBLEM BOUND. One block of 32 threads, and every
# loop over the matrix below is strided by exactly this, so any `n` is
# covered. Nothing `n`-sized lives in threadgroup memory, so no value of `n`
# can exhaust Metal's 32 KB budget (`PORTING.md 1`).
# READ FROM THE MATRIX, not restated here. `mojo_only/kernel_matrix.mojo`
# owns every tunable in this tree; changing TARGET_COLUMN there rebuilds
# this kernel for another vendor with no edit in this file.
comptime JACOBI_TPB = lib_block_size_for[K_LIB_JACOBI_EIGH, TARGET_COLUMN]()

# `raft::linalg::eigJacobi`'s own defaults (`raft/linalg/eig.cuh:108-109`),
# which are also what cuML's Python layer passes for the Jacobi arm
# (`pca.pyx:358` `tol=1e-7`, `pca.pyx:356` `iterated_power=15`).
comptime JACOBI_TOL = 1.0e-7
comptime JACOBI_SWEEPS = 15


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
        var fv = a.unsafe_load(fe)
        local_f += fv * fv
        fe += JACOBI_TPB
    var fro2 = block_sum[block_size=JACOBI_TPB, broadcast=True](local_f)
    var limit = tol_in * tol_in * fro2

    var executed = 0
    var converged = False
    var last_off = Float32(0.0)

    for _sweep in range(Int(max_sweeps_in)):
        # `cub::BlockReduce`'s counterpart from `max.gpu.primitives.block`.
        # This REPLACED a `tid == 0` serial double loop over the strict upper
        # triangle, run once per sweep with 31 of the 32 threads idle. Every
        # thread now accumulates a strided slice of the SAME triangle and the
        # block reduces them. See VENDOR_LIBRARIES.md.
        #
        # `broadcast=True` is load bearing rather than decoration: every
        # thread has to receive the same `off`, because the `break` below is
        # taken on it and a divergent break deadlocks on the barriers inside
        # the rotation loop. It is also what lets the shared `offdiag` slot
        # this replaced disappear.
        #
        # The block reduction ends block-wide, so it doubles as the barrier
        # that used to sit between the sum and the test: no thread reaches a
        # rotation write to `a` while another is still reading `a` here.
        #
        # NUMERIC: the summation ORDER changes, so `off` can differ from the
        # serial version in its last bits. The CONVERGENCE SEMANTICS do not:
        # same test against the same `tol_in`, and a last-bit difference can
        # only move which sweep a knife-edge matrix stops on, never the
        # answer it stops at.
        var local_off = Float32(0.0)
        var e = tid
        while e < n * n:
            var i = e // n
            var j = e - i * n
            if j > i:
                local_off += a.unsafe_load(e) * a.unsafe_load(e)
            e += JACOBI_TPB
        var off = block_sum[block_size=JACOBI_TPB, broadcast=True](local_off)
        last_off = off
        if Float32(2.0) * off <= limit:
            converged = True
            break
        executed += 1

        for p in range(n):
            for q in range(p + 1, n):
                if tid == 0:
                    var apq = a.unsafe_load(p * n + q)
                    if apq == Float32(0.0):
                        rot[0] = Float32(1.0)
                        rot[1] = Float32(0.0)
                    else:
                        var theta = (a.unsafe_load(q * n + q) - a.unsafe_load(p * n + p)) / (
                            Float32(2.0) * apq
                        )
                        var t = Float32(0.0)
                        if theta >= Float32(0.0):
                            t = Float32(1.0) / (
                                theta + sqrt(Float32(1.0) + theta * theta)
                            )
                        else:
                            t = Float32(-1.0) / (
                                -theta + sqrt(Float32(1.0) + theta * theta)
                            )
                        var c = Float32(1.0) / sqrt(Float32(1.0) + t * t)
                        rot[0] = c
                        rot[1] = t * c
                barrier()

                var c = rot[0]
                var s = rot[1]

                # Columns p and q, one thread per row.
                var k = tid
                while k < n:
                    var akp = a.unsafe_load(k * n + p)
                    var akq = a.unsafe_load(k * n + q)
                    a.unsafe_store(k * n + p, c * akp - s * akq)
                    a.unsafe_store(k * n + q, s * akp + c * akq)
                    k += JACOBI_TPB
                barrier()

                # Rows p and q, one thread per column.
                k = tid
                while k < n:
                    var apk = a.unsafe_load(p * n + k)
                    var aqk = a.unsafe_load(q * n + k)
                    a.unsafe_store(p * n + k, c * apk - s * aqk)
                    a.unsafe_store(q * n + k, s * apk + c * aqk)
                    k += JACOBI_TPB
                barrier()

                # The accumulated basis.
                k = tid
                while k < n:
                    var vkp = v.unsafe_load(k * n + p)
                    var vkq = v.unsafe_load(k * n + q)
                    v.unsafe_store(k * n + p, c * vkp - s * vkq)
                    v.unsafe_store(k * n + q, s * vkp + c * vkq)
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
            rel = sqrt(Float32(2.0) * last_off / fro2)
        info_out.unsafe_store(1, rel)
        info_out.unsafe_store(2, Float32(executed))

    # No write-back: `a` and `v` ARE the caller's buffers now.
