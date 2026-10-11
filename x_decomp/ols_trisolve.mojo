# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""MOJOLEARN_IDN_OLS_R_TRISOLVE (lane ols-r-trisolve, 2026-10-11; DEFAULT OFF,
verdict OPEN): LinearRegression's solve straight from the TSQR's R.

Structural reason: the centered TSQR of [X - mu | y - ybar] already gives
R_aug = [[R, c], [0, r22]] ((d + 1) x (d + 1), row-major); the least-squares
coefficients are the back-substitution R x = c. Today's route takes the
one-sided Jacobi SVD of R (~2.5k launches at d = 220, a host readback per
sweep group) plus ~12 kit binding calls, all of it only for rank deficiency.
This file runs ONE launch (one block over the d x d factor, never over rows):
  1. the DEVIATION 2620 power-of-two column scales s_j of ||R e_j||^2 (the
     `p2scale` that `_tsqr_lstsq_core(equilibrate=True)` applies);
  2. the rank gate on the equilibrated diagonal: every |R_jj s_j| above
     rcond * max_j |R_jj s_j| with rcond = eps32 * d (the cutoff the SVD route
     applies to singular values, applied to R's diagonal instead);
  3. if the gate passes, x by back-substitution on the UNSCALED R (a power of
     two scale commutes exactly with every rounding, so solving R S and
     multiplying by S gives the same words).
The gate word goes back to Python, which keeps the SVD route when it is 0
(rank deficient, ill-conditioned or non-finite R). Expected speedup: removes
~2.5k launches and ~10 binding crossings, ~100-200 ms of OLS istella on NV
(ideas-dense-linalg-20261011 F1). Bits change versus the SVD route (same
model up to rounding, r2 unchanged), the same on NVIDIA, AMD and the host
column: every dot product is the fixed fold below on every column.

Fold (identical on device and host): row i's sum over j > i of R_ij x_j is
split over OTS_TPB lanes, lane t taking j = i + 1 + t, i + 1 + t + OTS_TPB,
... ascending with the IDENTICAL fma cell, then the lanes fold by the halving
tree p[t] += p[t + s], s = OTS_TPB / 2 .. 1; x_i = (c_i - sum) / R_ii."""
from std.gpu import thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from checks.numerics import ftz, identical_mul_add
from x_decomp.cells import F32Ptr, add, div0, mul, p2_equilibration_scale, sub
from x_decomp.tsqr_core import TS_MAX_N

comptime OTS_TPB = 256
comptime OTS_EPS = Float32(1.1920928955078125e-07)


@always_inline
def ots_fma(a: Float32, b: Float32, acc: Float32) -> Float32:
    """The IDENTICAL cell chain ftz(fma(ftz(a), ftz(b), acc))."""
    return ftz(identical_mul_add(ftz(a), ftz(b), acc))


@always_inline
def ots_col_diag(r: F32Ptr, j: Int, n: Int) -> Float32:
    """|R_jj s_j| with s_j the power-of-two scale of ||R e_j||^2, summed
    i = 0 .. j ascending (rows below the diagonal are zero)."""
    var acc = Float32(0)
    for i in range(j + 1):  # small-loop(j: rows of one R column): R is the d x d factor, never data rows
        var v = r.unsafe_load(i * n + j)
        acc = add(acc, mul(v, v))
    var dj = mul(r.unsafe_load(j * n + j), p2_equilibration_scale(ftz(acc)))
    return dj if dj >= Float32(0) else -dj


@always_inline
def ots_gate(dg: F32Ptr, d: Int) -> Bool:
    """Every equilibrated |diagonal| strictly above eps32 * d * max (max and
    min are exact, so the order is free); a NaN or inf fails it."""
    var dmax = Float32(0)
    for j in range(d):  # small-loop(d: R diagonal entries): one word per feature, never rows
        var v = dg.unsafe_load(j)
        if not (v == v):
            return False
        if v > dmax:
            dmax = v
    var cut = mul(dmax, mul(OTS_EPS, Float32(d)))
    if not (cut < Float32(3.4028234663852886e38)):
        return False
    for j in range(d):  # small-loop(d: R diagonal entries): one word per feature, never rows
        if not (dg.unsafe_load(j) > cut):
            return False
    return True


@always_inline
def ots_lane_partial(r: F32Ptr, x: F32Ptr, i: Int, t: Int, n: Int, d: Int) -> Float32:
    """Lane t's ascending partial of row i's sum over j > i of R_ij x_j."""
    var acc = Float32(0)
    var j = i + 1 + t
    while j < d:  # small-loop(d: R columns of one row): the d x d factor, never data rows
        acc = ots_fma(r.unsafe_load(i * n + j), x.unsafe_load(j), acc)
        j += OTS_TPB
    return acc


def ols_r_trisolve_kernel(r: F32Ptr, res: F32Ptr, d_in: Int32):
    """One block of OTS_TPB threads over R_aug (n = d + 1, row-major, device):
    res[0, d) = x when the gate passes, res[d] = 1 (pass) or 0 (fail)."""
    var d = Int(d_in)
    var n = d + 1
    var tid = Int(thread_idx.x)
    var sx = stack_allocation[TS_MAX_N, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var sdg = stack_allocation[TS_MAX_N, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var sp = stack_allocation[OTS_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var sok = stack_allocation[1, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var j = tid
    while j < d:
        sdg[j] = ots_col_diag(r, j, n)
        sx[j] = Float32(0)
        j += OTS_TPB
    barrier()
    if tid == 0:  # small-launch(d: R diagonal entries): the gate reads d words of the d x d factor
        # the same steps as `ots_gate` on the shared words
        var dmax = Float32(0)
        var good = True
        for q in range(d):  # small-loop(d: R diagonal entries): one word per feature, never rows
            var v = sdg[q]
            if not (v == v):
                good = False
            elif v > dmax:
                dmax = v
        var cut = mul(dmax, mul(OTS_EPS, Float32(d)))
        if not (cut < Float32(3.4028234663852886e38)):
            good = False
        for q in range(d):  # small-loop(d: R diagonal entries): one word per feature, never rows
            if not (sdg[q] > cut):
                good = False
        sok[0] = Float32(1) if good else Float32(0)
    barrier()
    if sok[0] != Float32(0):
        var i = d - 1
        while i >= 0:
            # `ots_lane_partial` on the shared x
            var acc = Float32(0)
            var q = i + 1 + tid
            while q < d:
                acc = ots_fma(r.unsafe_load(i * n + q), sx[q], acc)
                q += OTS_TPB
            sp[tid] = acc
            barrier()
            var s = OTS_TPB // 2
            while s > 0:
                if tid < s:
                    sp[tid] = add(sp[tid], sp[tid + s])
                barrier()
                s //= 2
            if tid == 0:  # small-launch(d: R row i): one division per row of the d x d factor
                sx[i] = div0(sub(r.unsafe_load(i * n + d), sp[0]), r.unsafe_load(i * n + i))
            barrier()
            i -= 1
    j = tid
    while j < d:
        res.unsafe_store(j, sx[j])
        j += OTS_TPB
    if tid == 0:  # small-launch(d: the gate word): one word written after the d x d solve
        res.unsafe_store(d, sok[0])


def ols_r_trisolve_host(r: F32Ptr, res: F32Ptr, d: Int):
    """The host column of `ols_r_trisolve_kernel`: the same cells in the
    same fold (lane partials then the halving tree)."""
    var n = d + 1
    var dg = List[Float32](length=d if d > 0 else 1, fill=Float32(0))
    var x = List[Float32](length=d if d > 0 else 1, fill=Float32(0))
    var p = List[Float32](length=OTS_TPB, fill=Float32(0))
    for j in range(d):  # small-loop(d: R columns): the d x d factor, never data rows
        dg[j] = ots_col_diag(r, j, n)
    var ok = ots_gate(F32Ptr(unsafe_from_address=Int(dg.unsafe_ptr())), d)
    if ok:
        var xp = F32Ptr(unsafe_from_address=Int(x.unsafe_ptr()))
        var i = d - 1
        while i >= 0:  # small-loop(d: R rows): back-substitution over the d x d factor
            for t in range(OTS_TPB):
                p[t] = ots_lane_partial(r, xp, i, t, n, d)
            var s = OTS_TPB // 2
            while s > 0:
                for t in range(s):
                    p[t] = add(p[t], p[t + s])
                s //= 2
            x[i] = div0(sub(r.unsafe_load(i * n + d), p[0]), r.unsafe_load(i * n + i))
            i -= 1
    for j in range(d):  # small-loop(d: coefficients): one word per feature
        res.unsafe_store(j, x[j])
    res.unsafe_store(d, Float32(1) if ok else Float32(0))
    _ = dg^
    _ = x^
    _ = p^
