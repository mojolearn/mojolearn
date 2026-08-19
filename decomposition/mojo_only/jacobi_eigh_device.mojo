"""Cyclic Jacobi eigendecomposition ON THE DEVICE.

NOT A PORT of a file: RAFT calls cuSOLVER's `syevj` and cuSOLVER is closed.
But **it runs on the device in RAFT, so it runs on the device here**, which
is the standing rule for this repository: mirror their host/device split, and
where they use the GPU we use the GPU.

This REPLACES the host version in `jacobi_eigh.mojo` as the path the fit
takes. The host one stays because it is the reference the device one is
checked against, and because a host implementation of a small dense solver is
the right thing for a bring-up harness to have.

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

**The two therefore do not agree bit for bit and are not meant to.** The
check compares them at a stated tolerance.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.math import sqrt
from max.gpu.memory import AddressSpace
from max.gpu.primitives.block import sum as block_sum
from max.gpu.sync import barrier
from std.memory import stack_allocation


# One block, and the matrix plus the basis both live in shared memory, so
# this is the cap. 32 x 32 x 4 bytes x 2 arrays = 8 KB against Metal's 32 KB
# threadgroup budget (`PORTING.md 1`).
comptime JACOBI_MAX_N = 32
comptime JACOBI_TPB = 32


def jacobi_eigh_kernel(
    a_io: MutPointer[Float32, MutAnyOrigin],
    v_out: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    max_sweeps_in: Int32,
    tol_in: Float32,
):
    """`a_io` in: the symmetric matrix. Out: diagonal holds the eigenvalues.

    `v_out` ends with eigenvector `i` in COLUMN `i`, LAPACK's convention and
    therefore cuSOLVER's, which is what the caller's ordering code expects.
    """
    var n = Int(n_in)
    var tid = Int(thread_idx.x)

    var a = stack_allocation[
        JACOBI_MAX_N * JACOBI_MAX_N,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var v = stack_allocation[
        JACOBI_MAX_N * JACOBI_MAX_N,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var rot = stack_allocation[
        2,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()

    var idx = tid
    while idx < n * n:
        a[idx] = a_io.unsafe_load(idx)
        var r = idx // n
        var c = idx % n
        v[idx] = Float32(1.0) if r == c else Float32(0.0)
        idx += JACOBI_TPB
    barrier()

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
        # same `off <= tol_in` test against the same `tol_in`, and a last-bit
        # difference can only move which sweep a knife-edge matrix stops on,
        # never the answer it stops at.
        var local_off = Float32(0.0)
        var e = tid
        while e < n * n:
            var i = e // n
            var j = e - i * n
            if j > i:
                local_off += a[e] * a[e]
            e += JACOBI_TPB
        var off = block_sum[block_size=JACOBI_TPB, broadcast=True](local_off)
        if off <= tol_in:
            break

        for p in range(n):
            for q in range(p + 1, n):
                if tid == 0:
                    var apq = a[p * n + q]
                    if apq == Float32(0.0):
                        rot[0] = Float32(1.0)
                        rot[1] = Float32(0.0)
                    else:
                        var theta = (a[q * n + q] - a[p * n + p]) / (
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
                    var akp = a[k * n + p]
                    var akq = a[k * n + q]
                    a[k * n + p] = c * akp - s * akq
                    a[k * n + q] = s * akp + c * akq
                    k += JACOBI_TPB
                barrier()

                # Rows p and q, one thread per column.
                k = tid
                while k < n:
                    var apk = a[p * n + k]
                    var aqk = a[q * n + k]
                    a[p * n + k] = c * apk - s * aqk
                    a[q * n + k] = s * apk + c * aqk
                    k += JACOBI_TPB
                barrier()

                # The accumulated basis.
                k = tid
                while k < n:
                    var vkp = v[k * n + p]
                    var vkq = v[k * n + q]
                    v[k * n + p] = c * vkp - s * vkq
                    v[k * n + q] = s * vkp + c * vkq
                    k += JACOBI_TPB
                barrier()

    idx = tid
    while idx < n * n:
        a_io.unsafe_store(idx, a[idx])
        v_out.unsafe_store(idx, v[idx])
        idx += JACOBI_TPB
