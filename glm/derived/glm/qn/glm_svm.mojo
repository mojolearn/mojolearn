# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`SVCL1Loss`, `SVCL2Loss`, `SVRL1Loss`, `SVRL2Loss`: the hinge family,
per row.

PORT OF `cuml/cpp/src/glm/qn/glm_svm.cuh` at cuML `00094f7`. Whole file:
the four `Lz`/`Dlz` pairs and their `gradNorm`s (`nrm1` for the L1 pair,
`squaredNorm * 0.5` for the L2 pair -- dispatched in `glm_base.mojo::
GLMWithData.grad_norm`). Do not improve.

THEIR EIGHT FUNCTORS, copied (`glm_svm.cuh:19-128`); `s = 2 * y - 1`,
`t = y - z`, `eps` the SVR `sensitivity` (`svr_eps` in `qn_fit_x`):

    SVC-L1  lz  = max(0, 1 - s z)
            dlz = s z <= 1 ? -s : 0
    SVC-L2  lz  = t^2,  t = max(0, 1 - s z)
            dlz = s z <= 1 ? z - s : 0          <- THEIRS; ours is 2 (z - s),
                                                  DEVIATION 714 below
    SVR-L1  lz  = t > eps ? t - eps : (t < -eps ? -t - eps : 0)
            dlz = t > eps ? -1 : (t < -eps ? 1 : 0)
    SVR-L2  lz  = s^2,  s = t > eps ? t - eps : (t < -eps ? -t - eps : 0)
            dlz = -2 * (t > eps ? t - eps : (t < -eps ? t + eps : 0))

TWO MULTIPLY-ADDS AND ONE CLAMP. `2 * y - 1` and `1 - s * z` are row 9's
contraction candidates (nvcc contracts both inside a device lambda) and go
through `identical_mul_add(2, y, -1)` -- the spelling `glm_logistic.mojo`
already uses for the same `2y - 1` -- and `identical_mul_add(-s, z, 1)`.
`raft::max<T>(0, v)` is a float CLAMP and is spelled VALUE-FIRST `max(v,
0.0)` (ADDENDUM 11): `max(0, -0.0)` is `-0.0` on Apple (second operand)
and `+0.0` on NVIDIA/AMD (IEEE maximum), and `max(-0.0, 0.0)` is `+0.0`
on all three. `s * z <= 1` is a single product compared, one rounding;
`-2 * (...)` is exact. Every intermediate is stored through `ftz`.

THE LAUNCH SHAPE IS `glm_logistic.mojo`'s: one fused kernel per loss,
`loss_terms[i] = lz * normalization` and `z[i] = dlz`, one thread per row;
the sum is `glm_base.mojo::sum_terms_kernel` (DEVIATION 547).
DEVIATION 708 covers the pinned spellings above.

DEVIATION 714: `SVCL2Loss::Dlz` IS NOT THE DERIVATIVE OF `SVCL2Loss::Lz`.
Theirs (`glm_svm.cuh:58-66`) pairs `lz = t^2, t = max(0, 1 - s z)` with
`dlz = s z <= 1 ? z - s : 0`. The derivative of `(1 - s z)^2` in `z` is
`-2 s (1 - s z) = 2 (z - s)` (`s^2 = 1`), so their gradient is HALF the
slope of the value they report: the line search's Armijo test compares a
decrease in `t^2` against half its slope, `check_convergence` tests half a
gradient against the full value, and the fixed point the solver stops at is
the minimizer of `mean t^2 / 2 + l2/2 ||w||^2`, i.e. `mean t^2 + l2
||w||^2` -- TWICE the regularization the objective says (and twice what
`cuml.svm.LinearSVC(loss='squared_hinge')`, which routes here through
`svm/linear.cu:168`, documents). `SVRL2Loss` in the same file carries its
factor 2 (`dlz = -2 * (...)`), and `SquaredLoss` carries its `0.5` on the
value side, so the omission is an inconsistency, not a convention. Ours is
`dlz = s z <= 1 ? 2 (z - s) : 0` (the `2 *` exact). MEASURED by
`glm/original/qn_losses_check.mojo::check_losses_fd_gradient` with their
spelling in the float64 oracle: analytic vs central-difference relative
error 0.9999999912 on QN_LOSS_SVC_L2 (a factor of exactly 2), 1.1e-8 on
QN_LOSS_SQUARED, 6.3e-9 on QN_LOSS_SVC_L1 -- the oracle disagreed with
itself before any device number was consulted. With the fix it is below
1e-7. `lz` is left as theirs (`t^2`, the squared hinge by name), so the
reported objective is sklearn's `LinearSVC` objective divided by `C n`.
"""

from std.gpu import block_dim, block_idx, thread_idx

from original.numerics import ftz, identical_mul_add


@always_inline
def _s_of(y: Float32) -> Float32:
    """`T s = 2 * y - 1`."""
    return ftz(identical_mul_add(Float32(2.0), y, Float32(-1.0)))


@always_inline
def _hinge(s: Float32, z: Float32) -> Float32:
    """`raft::max<T>(0, 1 - s * z)`, value-first."""
    var v = ftz(identical_mul_add(-s, z, Float32(1.0)))
    return max(v, Float32(0.0))


@always_inline
def svc_l1_lz(y: Float32, z: Float32) -> Float32:
    return _hinge(_s_of(y), z)


@always_inline
def svc_l1_dlz(y: Float32, z: Float32) -> Float32:
    var s = _s_of(y)
    if ftz(s * z) <= Float32(1.0):
        return -s
    return Float32(0.0)


@always_inline
def svc_l2_lz(y: Float32, z: Float32) -> Float32:
    var t = _hinge(_s_of(y), z)
    return ftz(t * t)


@always_inline
def svc_l2_dlz(y: Float32, z: Float32) -> Float32:
    """DEVIATION 714: `2 (z - s)`, the derivative of `lz`; theirs is `z - s`."""
    var s = _s_of(y)
    if ftz(s * z) <= Float32(1.0):
        return ftz(Float32(2.0) * ftz(z - s))
    return Float32(0.0)


@always_inline
def _svr_dead_zone(t: Float32, eps: Float32) -> Float32:
    """`t > eps ? t - eps : (t < -eps ? -t - eps : 0)`: the epsilon-
    insensitive magnitude, `SVRL1Loss::Lz` and `SVRL2Loss::Lz`'s `s`."""
    if t > eps:
        return ftz(t - eps)
    if t < -eps:
        return ftz(-t - eps)
    return Float32(0.0)


@always_inline
def svr_l1_lz(y: Float32, z: Float32, eps: Float32) -> Float32:
    return _svr_dead_zone(ftz(y - z), eps)


@always_inline
def svr_l1_dlz(y: Float32, z: Float32, eps: Float32) -> Float32:
    var t = ftz(y - z)
    if t > eps:
        return Float32(-1.0)
    if t < -eps:
        return Float32(1.0)
    return Float32(0.0)


@always_inline
def svr_l2_lz(y: Float32, z: Float32, eps: Float32) -> Float32:
    var s = _svr_dead_zone(ftz(y - z), eps)
    return ftz(s * s)


@always_inline
def svr_l2_dlz(y: Float32, z: Float32, eps: Float32) -> Float32:
    """`-2 * (t > eps ? t - eps : (t < -eps ? (t + eps) : 0))`."""
    var t = ftz(y - z)
    var inner = Float32(0.0)
    if t > eps:
        inner = ftz(t - eps)
    elif t < -eps:
        inner = ftz(t + eps)
    return ftz(Float32(-2.0) * inner)


def svc_l1_loss_dz_kernel(
    loss_terms: MutPointer[Float32, MutAnyOrigin],
    z: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    normalization: Float32,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        var yi = y.unsafe_load(i)
        var zi = z.unsafe_load(i)
        loss_terms.unsafe_store(i, ftz(svc_l1_lz(yi, zi) * normalization))
        z.unsafe_store(i, svc_l1_dlz(yi, zi))


def svc_l2_loss_dz_kernel(
    loss_terms: MutPointer[Float32, MutAnyOrigin],
    z: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    normalization: Float32,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        var yi = y.unsafe_load(i)
        var zi = z.unsafe_load(i)
        loss_terms.unsafe_store(i, ftz(svc_l2_lz(yi, zi) * normalization))
        z.unsafe_store(i, svc_l2_dlz(yi, zi))


def svr_l1_loss_dz_kernel(
    loss_terms: MutPointer[Float32, MutAnyOrigin],
    z: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    normalization: Float32,
    eps: Float32,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        var yi = y.unsafe_load(i)
        var zi = z.unsafe_load(i)
        loss_terms.unsafe_store(i, ftz(svr_l1_lz(yi, zi, eps) * normalization))
        z.unsafe_store(i, svr_l1_dlz(yi, zi, eps))


def svr_l2_loss_dz_kernel(
    loss_terms: MutPointer[Float32, MutAnyOrigin],
    z: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    normalization: Float32,
    eps: Float32,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        var yi = y.unsafe_load(i)
        var zi = z.unsafe_load(i)
        loss_terms.unsafe_store(i, ftz(svr_l2_lz(yi, zi, eps) * normalization))
        z.unsafe_store(i, svr_l2_dlz(yi, zi, eps))
