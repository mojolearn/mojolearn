# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Cyclic Jacobi eigendecomposition ON THE DEVICE."""

from core.pinned_reduce import pinned_block_sum
from checks.kernel_matrix import (
    K_LIB_JACOBI_EIGH,
    TARGET_COLUMN,
    lib_block_size_for,
)
from checks.numerics import (
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


comptime JACOBI_TPB = lib_block_size_for[K_LIB_JACOBI_EIGH, TARGET_COLUMN]()

comptime JACOBI_TOL = 1.0e-7
comptime JACOBI_SWEEPS = 15


@always_inline
def _folded_and_broadcast[tpb: Int](value: Float32) -> Float32:
    """`pinned_block_sum` plus the broadcast the sweep loop cannot do without."""
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
    """`c*x - s*y`, ONE spelling on every backend under IDENTICAL."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return ftz(identical_mul_add(c, x, -ftz(s * y)))
    return c * x - s * y


@always_inline
def _rot_add(s: Float32, x: Float32, c: Float32, y: Float32) -> Float32:
    """`s*x + c*y`, the sibling of `_rot_sub` and the same rule: fuse the first product, round the second, flush both."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return ftz(identical_mul_add(s, x, ftz(c * y)))
    return s * x + c * y


@always_inline
def jacobi_rotation_cs(
    app_in: Float32, aqq_in: Float32, apq_in: Float32
) -> SIMD[DType.float32, 2]:
    """The Jacobi rotation `(c, s)` that annihilates `apq` in `[[app, apq], [apq, aqq]]`."""
    var apq = ftz(apq_in)
    if apq == Float32(0.0):
        return SIMD[DType.float32, 2](Float32(1.0), Float32(0.0))
    var aqq = ftz(aqq_in)
    var app = ftz(app_in)
    var theta = ftz(ftz(aqq - app) / ftz(Float32(2.0) * apq))
    var root = ftz(
        identical_sqrt(ftz(identical_mul_add(theta, theta, Float32(1.0))))
    )
    var t = Float32(0.0)
    if theta >= Float32(0.0):
        t = ftz(Float32(1.0) / ftz(theta + root))
    else:
        t = ftz(Float32(-1.0) / ftz(root - theta))
    var croot = ftz(
        identical_sqrt(ftz(identical_mul_add(t, t, Float32(1.0))))
    )
    var c = ftz(Float32(1.0) / croot)
    return SIMD[DType.float32, 2](c, ftz(t * c))


@always_inline
def _rotate_pair_block(
    a: MutPointer[Float32, MutAnyOrigin],
    n: Int,
    p: Int,
    q: Int,
    c: Float32,
    s: Float32,
):
    """The 2 x 2 block `(p, p), (p, q), (q, p), (q, q)` of one rotation, in the four-phase kernel's order.

    Column stage for k = p and k = q, then row stage for k = p and k = q,
    each value loaded from memory and stored back exactly as the column and
    row loops of `jacobi_eigh_kernel_four_phase` do. Only the lane that owns
    `k == p` calls this, and no other lane of the same phase touches these
    four cells, so the order written here is the only order there is.
    """
    var app = ftz(a.unsafe_load(p * n + p))
    var apq = ftz(a.unsafe_load(p * n + q))
    a.unsafe_store(p * n + p, _rot_sub(c, app, s, apq))
    a.unsafe_store(p * n + q, _rot_add(s, app, c, apq))
    var aqp = ftz(a.unsafe_load(q * n + p))
    var aqq = ftz(a.unsafe_load(q * n + q))
    a.unsafe_store(q * n + p, _rot_sub(c, aqp, s, aqq))
    a.unsafe_store(q * n + q, _rot_add(s, aqp, c, aqq))
    var rpp = ftz(a.unsafe_load(p * n + p))
    var rqp = ftz(a.unsafe_load(q * n + p))
    a.unsafe_store(p * n + p, _rot_sub(c, rpp, s, rqp))
    a.unsafe_store(q * n + p, _rot_add(s, rpp, c, rqp))
    var rpq = ftz(a.unsafe_load(p * n + q))
    var rqq = ftz(a.unsafe_load(q * n + q))
    a.unsafe_store(p * n + q, _rot_sub(c, rpq, s, rqq))
    a.unsafe_store(q * n + q, _rot_add(s, rpq, c, rqq))


def jacobi_eigh_kernel(
    a_io: MutPointer[Float32, MutAnyOrigin],
    v_out: MutPointer[Float32, MutAnyOrigin],
    info_out: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    max_sweeps_in: Int32,
    tol_in: Float32,
):
    """`a_io` in: the symmetric matrix.

    DEVIATION 2671 (2026-09-11, linear-cluster-istella): TWO BARRIERS PER
    ROTATION INSTEAD OF FOUR, THE SAME ARITHMETIC IN THE SAME ORDER.
    `jacobi_eigh_kernel_four_phase` (the kernel this replaced, kept for the
    equality gate) runs each rotation as four phases, the `(c, s)` pick, the
    column update, the row update and the eigenvector update, each closed by
    a barrier. At Istella-S's 220 columns that is 24,090 rotations and 96,360
    barriers per sweep, and OLS takes 12 sweeps there.

    The last three phases only need an order where they write the same cell.
    The column update writes cells `(k, p), (k, q)`, the row update writes
    `(p, k), (q, k)`, and the two sets meet only in the 2 x 2 block
    `{p, q} x {p, q}`. The eigenvector update writes `v`, which neither reads.
    So one phase does, per lane and per k outside `{p, q}`, that k's column
    pair then its row pair; the lane owning `k == p` does the 2 x 2 block in
    the old column-then-row order (`_rotate_pair_block`); every lane does its
    eigenvector pair. Every store reads the same inputs through the same
    pinned spelling as before, and the `(c, s)` pick, the rotation order,
    both folds and the sweep test are untouched, so the output bits cannot
    move. `check_jacobi_merged_phases_equal_four_phase` holds both kernels
    equal at every cell, and `check_jacobi_is_a_pure_function_of_its_input`
    still holds this one to the host replay. A parallel (round-robin)
    rotation order would drop more barriers, and it is a different
    algorithm with different bits, so it is not what this is.
    """
    var n = Int(n_in)
    var tid = Int(thread_idx.x)

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
        var cc = idx % n
        v.unsafe_store(idx, Float32(1.0) if r == cc else Float32(0.0))
        idx += JACOBI_TPB
    barrier()

    var local_f = Float32(0.0)
    var fe = tid
    while fe < n * n:
        var fv = ftz(a.unsafe_load(fe))
        local_f = ftz(identical_mul_add(fv, fv, local_f))
        fe += JACOBI_TPB
    var fro2 = _folded_and_broadcast[JACOBI_TPB](local_f)
    var limit = ftz(ftz(tol_in * tol_in) * fro2)

    var executed = 0
    var converged = False
    var last_off = Float32(0.0)

    for _sweep in range(Int(max_sweeps_in)):
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
                if tid == 0:
                    var cs = jacobi_rotation_cs(
                        a.unsafe_load(p * n + p),
                        a.unsafe_load(q * n + q),
                        a.unsafe_load(p * n + q),
                    )
                    rot[0] = cs[0]
                    rot[1] = cs[1]
                barrier()

                var c = rot[0]
                var s = rot[1]

                var k = tid
                while k < n:
                    if k != p and k != q:
                        var akp = ftz(a.unsafe_load(k * n + p))
                        var akq = ftz(a.unsafe_load(k * n + q))
                        a.unsafe_store(k * n + p, _rot_sub(c, akp, s, akq))
                        a.unsafe_store(k * n + q, _rot_add(s, akp, c, akq))
                        var apk = ftz(a.unsafe_load(p * n + k))
                        var aqk = ftz(a.unsafe_load(q * n + k))
                        a.unsafe_store(p * n + k, _rot_sub(c, apk, s, aqk))
                        a.unsafe_store(q * n + k, _rot_add(s, apk, c, aqk))
                    elif k == p:
                        _rotate_pair_block(a, n, p, q, c, s)
                    var vkp = ftz(v.unsafe_load(k * n + p))
                    var vkq = ftz(v.unsafe_load(k * n + q))
                    v.unsafe_store(k * n + p, _rot_sub(c, vkp, s, vkq))
                    v.unsafe_store(k * n + q, _rot_add(s, vkp, c, vkq))
                    k += JACOBI_TPB
                barrier()

    if tid == 0:
        info_out.unsafe_store(0, Float32(1.0) if converged else Float32(0.0))
        var rel = Float32(0.0)
        if fro2 > Float32(0.0):
            rel = ftz(
                identical_sqrt(ftz(ftz(Float32(2.0) * last_off) / fro2))
            )
        info_out.unsafe_store(1, rel)
        info_out.unsafe_store(2, Float32(executed))


def jacobi_eigh_kernel_four_phase(
    a_io: MutPointer[Float32, MutAnyOrigin],
    v_out: MutPointer[Float32, MutAnyOrigin],
    info_out: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    max_sweeps_in: Int32,
    tol_in: Float32,
):
    """The shipped kernel before DEVIATION 2671, four barriers per rotation. No library path launches it; `check_jacobi_merged_phases_equal_four_phase` and the lane's stage probe do."""
    var n = Int(n_in)
    var tid = Int(thread_idx.x)

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

    var local_f = Float32(0.0)
    var fe = tid
    while fe < n * n:
        var fv = ftz(a.unsafe_load(fe))
        local_f = ftz(identical_mul_add(fv, fv, local_f))
        fe += JACOBI_TPB
    var fro2 = _folded_and_broadcast[JACOBI_TPB](local_f)
    var limit = ftz(ftz(tol_in * tol_in) * fro2)

    var executed = 0
    var converged = False
    var last_off = Float32(0.0)

    for _sweep in range(Int(max_sweeps_in)):
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
                if tid == 0:
                    var cs = jacobi_rotation_cs(
                        a.unsafe_load(p * n + p),
                        a.unsafe_load(q * n + q),
                        a.unsafe_load(p * n + q),
                    )
                    rot[0] = cs[0]
                    rot[1] = cs[1]
                barrier()

                var c = rot[0]
                var s = rot[1]

                var k = tid
                while k < n:
                    var akp = ftz(a.unsafe_load(k * n + p))
                    var akq = ftz(a.unsafe_load(k * n + q))
                    a.unsafe_store(k * n + p, _rot_sub(c, akp, s, akq))
                    a.unsafe_store(k * n + q, _rot_add(s, akp, c, akq))
                    k += JACOBI_TPB
                barrier()

                k = tid
                while k < n:
                    var apk = ftz(a.unsafe_load(p * n + k))
                    var aqk = ftz(a.unsafe_load(q * n + k))
                    a.unsafe_store(p * n + k, _rot_sub(c, apk, s, aqk))
                    a.unsafe_store(q * n + k, _rot_add(s, apk, c, aqk))
                    k += JACOBI_TPB
                barrier()

                k = tid
                while k < n:
                    var vkp = ftz(v.unsafe_load(k * n + p))
                    var vkq = ftz(v.unsafe_load(k * n + q))
                    v.unsafe_store(k * n + p, _rot_sub(c, vkp, s, vkq))
                    v.unsafe_store(k * n + q, _rot_add(s, vkp, c, vkq))
                    k += JACOBI_TPB
                barrier()

    if tid == 0:
        info_out.unsafe_store(0, Float32(1.0) if converged else Float32(0.0))
        var rel = Float32(0.0)
        if fro2 > Float32(0.0):
            rel = ftz(
                identical_sqrt(ftz(ftz(Float32(2.0) * last_off) / fro2))
            )
        info_out.unsafe_store(1, rel)
        info_out.unsafe_store(2, Float32(executed))
