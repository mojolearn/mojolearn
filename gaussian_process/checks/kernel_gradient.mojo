# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""dK/dtheta on the device (2026-09-15): the `eval_gradient` arms of
scikit-learn 1.9.0 `sklearn/gaussian_process/kernels.py`, for every node kind
`kernels.mojo` implements, in one postfix walk that also produces K.

DEVIATION 2880 (header of `gaussian_process/host/gp_theta.mojo`) defines
theta; this file pins each gradient's arithmetic. `D` is the per-feature
squared scaled difference for an ARD length scale, `diff_f = ftz(x_f/l_f -
y_f/l_f)`, `D_f = ftz(diff_f * diff_f)` by `identical_mul`; for an isotropic
length scale it is the scaled squared distance `d2` itself, exactly the value
`gp_scaled_sqdist` computes for K (so no second fold exists). `dist =
ftz(sqrt(d2))`; `k` is the leaf's own value cell.

    ConstantKernel   c everywhere                  kernels.py (np.full)
    WhiteKernel      noise on the diagonal         kernels.py (eye)
    RBF              ftz(D * k)                    kernels.py:1576-1584
    Matern nu=0.5    ftz(k * ftz(D / dist)), 0 where dist == 0
                                                   kernels.py:1757-1766
    Matern nu=1.5    ftz(ftz(3 * D) * ftz(exp(-s)))   s = ftz(dist * sqrt3)
                                                   kernels.py:1768
    Matern nu=2.5    ftz(ftz(ftz(ftz(5 * D) / 3) * ftz(1 + s5)) * ftz(exp(-s5)))
                                                   s5 = ftz(dist * sqrt5),
                                                   kernels.py:1770-1771
    Sum              the operands' gradients, unchanged
    Product          left gradients * right value, then right gradients *
                     left value, elementwise `identical_mul`, BEFORE the value
                     product overwrites the left value (kernels.py Product:
                     `dstack(K1_grad * K2, K2_grad * K1)`)

`s` and `s5` are the kernel's own `dist * sqrt3` and `dist * sqrt5` rather
than the reference's `sqrt(3 * D.sum())` and `sqrt(5 * D.sum())`: the same
real number, one rounding fewer, and the same bits K was built from.

One thread owns one cell and walks the feature axis ascending, `kernels.mojo`'s
discipline; the gradient buffer holds `n_free` matrices in theta order.
"""

from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.numerics import (
    ftz,
    identical_div,
    identical_exp,
    identical_mul,
    identical_sqrt,
)
from gaussian_process.checks.kernels import (
    GP_ELEM_TPB,
    GP_K_CONST,
    GP_K_MATERN,
    GP_K_PROD,
    GP_K_RBF,
    GP_K_SUM,
    GP_K_WHITE,
    GPKernelSpec,
    gp_combine_kernel,
    gp_const_kernel,
    gp_copy_kernel,
    gp_kernel_stack_floats,
    gp_matern_kernel,
    gp_matern_nu_selector,
    gp_rbf_kernel,
    gp_scaled_sqdist,
    gp_sqrt3,
    gp_sqrt5,
    gp_validate_kernel,
    gp_white_kernel,
)
from gaussian_process.host.gp_theta import gp_free_count

#: The length-scale gradient forms.
comptime GP_GRAD_RBF = 0
comptime GP_GRAD_MATERN05 = 1
comptime GP_GRAD_MATERN15 = 2
comptime GP_GRAD_MATERN25 = 3


def gp_ls_gradient_value(
    form: Int, dd: Float32, d2: Float32, k: Float32, sqrt3: Float32, sqrt5: Float32
) -> Float32:
    """One gradient cell from `D` (`dd`), `d2` and the leaf value `k`, this
    file's table. Shared by the device kernel and nothing else; the verifier
    restates it (`gpr_grad_oracle.mojo`)."""
    if form == GP_GRAD_RBF:
        return ftz(identical_mul(dd, k))
    var dist = ftz(identical_sqrt(d2))
    if form == GP_GRAD_MATERN05:
        if not (dist > Float32(0.0)):
            return Float32(0.0)
        return ftz(identical_mul(k, ftz(identical_div(dd, dist))))
    if form == GP_GRAD_MATERN15:
        var s = ftz(identical_mul(dist, sqrt3))
        var e = ftz(identical_exp(-s))
        return ftz(identical_mul(ftz(identical_mul(Float32(3.0), dd)), e))
    var s5 = ftz(identical_mul(dist, sqrt5))
    var a = ftz(identical_div(ftz(identical_mul(Float32(5.0), dd)), Float32(3.0)))
    var b = ftz(identical_mul(a, ftz(Float32(1.0) + s5)))
    return ftz(identical_mul(b, ftz(identical_exp(-s5))))


def gp_ls_grad_kernel(
    g: MutPointer[Float32, MutAnyOrigin],
    kval: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    ls: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    d_in: Int32,
    ls_len_in: Int32,
    form_in: Int32,
    sqrt3: Float32,
    sqrt5: Float32,
):
    """The length-scale gradient of one RBF or Matern leaf over `K(X, X)`:
    `ls_len` matrices written at `g[f * n * n + t]`."""
    var n = Int(n_in)
    var d = Int(d_in)
    var ls_len = Int(ls_len_in)
    var form = Int(form_in)
    var cells = n * n
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= cells:
        return
    var i = t // n
    var j = t - i * n
    var k = ftz(kval.unsafe_load(t))
    var d2 = gp_scaled_sqdist(x, x, ls, i, j, d, ls_len)
    if ls_len == 1:
        g.unsafe_store(t, gp_ls_gradient_value(form, d2, d2, k, sqrt3, sqrt5))
        return
    for f in range(d):
        var lv = ftz(ls.unsafe_load(f))
        var xv = ftz(identical_div(ftz(x.unsafe_load(i * d + f)), lv))
        var yv = ftz(identical_div(ftz(x.unsafe_load(j * d + f)), lv))
        var diff = ftz(xv - yv)
        var dd = ftz(identical_mul(diff, diff))
        g.unsafe_store(f * cells + t, gp_ls_gradient_value(form, dd, d2, k, sqrt3, sqrt5))


def gp_kernel_matrix_grad(
    ctx: DeviceContext,
    mut out: DeviceBuffer[DType.float32],
    x_input: DeviceBuffer[DType.float32],
    mut dls: DeviceBuffer[DType.float32],
    mut stack: DeviceBuffer[DType.float32],
    mut dgrad: DeviceBuffer[DType.float32],
    n: Int,
    d: Int,
    spec: GPKernelSpec,
    free: List[Int32],
    elem_tpb: Int = GP_ELEM_TPB,
) raises -> Int:
    """`out[n x n] = k(X, X)` (is_self, bit for bit `gp_kernel_matrix`'s
    value launches) and `dgrad[p * n * n + t]` the gradient of theta entry
    `p`. Returns the number of theta entries. ASYNCHRONOUS; the caller
    synchronizes and keeps every buffer alive."""
    # DEVIATION 2487: the self-kernel reads X through two views, one buffer.
    var x = x_input.create_sub_buffer[DType.float32](0, len(x_input))
    var x2 = x_input.create_sub_buffer[DType.float32](0, len(x_input))
    if n <= 0 or d <= 0:
        raise Error("gp_kernel_matrix_grad: n and d must be positive")
    if elem_tpb <= 0:
        raise Error("gp_kernel_matrix_grad: elem_tpb must be positive")
    gp_validate_kernel(spec, d)
    var n_free = gp_free_count(spec.kinds, spec.ls_len, free)
    var cells = n * n
    if len(out) < cells or len(stack) < gp_kernel_stack_floats(n, n):
        raise Error("gp_kernel_matrix_grad: the output or stack buffer is too small")
    if len(dgrad) < max(n_free * cells, 1):
        raise Error("gp_kernel_matrix_grad: the gradient buffer is too small")
    var grid = (cells + elem_tpb - 1) // elem_tpb
    var sqrt3 = gp_sqrt3()
    var sqrt5 = gp_sqrt5()

    var sp = 0
    var gi = 0
    var first = List[Int]()
    var count = List[Int]()
    for t in range(len(spec.kinds)):
        var kind = Int(spec.kinds[t])
        if kind == GP_K_SUM or kind == GP_K_PROD:
            var lhs = stack.create_sub_buffer[DType.float32]((sp - 2) * cells, cells)
            var rhs = stack.create_sub_buffer[DType.float32]((sp - 1) * cells, cells)
            var fb = first.pop()
            var cb = count.pop()
            var fa = first.pop()
            var ca = count.pop()
            if kind == GP_K_PROD:
                for q in range(ca):
                    var gsub = dgrad.create_sub_buffer[DType.float32]((fa + q) * cells, cells)
                    ctx.enqueue_function[gp_combine_kernel](
                        gsub.unsafe_ptr(), rhs.unsafe_ptr(), Int32(cells), Int32(1),
                        grid_dim=(grid, 1, 1), block_dim=(elem_tpb, 1, 1),
                    )
                    _ = gsub^
                for q in range(cb):
                    var gsub = dgrad.create_sub_buffer[DType.float32]((fb + q) * cells, cells)
                    ctx.enqueue_function[gp_combine_kernel](
                        gsub.unsafe_ptr(), lhs.unsafe_ptr(), Int32(cells), Int32(1),
                        grid_dim=(grid, 1, 1), block_dim=(elem_tpb, 1, 1),
                    )
                    _ = gsub^
            ctx.enqueue_function[gp_combine_kernel](
                lhs.unsafe_ptr(), rhs.unsafe_ptr(), Int32(cells),
                Int32(1) if kind == GP_K_PROD else Int32(0),
                grid_dim=(grid, 1, 1), block_dim=(elem_tpb, 1, 1),
            )
            _ = lhs^
            _ = rhs^
            first.append(fa)
            count.append(ca + cb)
            sp -= 1
            continue

        var slot = stack.create_sub_buffer[DType.float32](sp * cells, cells)
        var is_free = Int(free[t]) != 0
        first.append(gi)
        if kind == GP_K_CONST:
            ctx.enqueue_function[gp_const_kernel](
                slot.unsafe_ptr(), Int32(cells), spec.params[t],
                grid_dim=(grid, 1, 1), block_dim=(elem_tpb, 1, 1),
            )
            if is_free:
                var gsub = dgrad.create_sub_buffer[DType.float32](gi * cells, cells)
                ctx.enqueue_function[gp_const_kernel](
                    gsub.unsafe_ptr(), Int32(cells), spec.params[t],
                    grid_dim=(grid, 1, 1), block_dim=(elem_tpb, 1, 1),
                )
                _ = gsub^
                gi += 1
                count.append(1)
            else:
                count.append(0)
        elif kind == GP_K_WHITE:
            ctx.enqueue_function[gp_white_kernel](
                slot.unsafe_ptr(), Int32(n), Int32(n), spec.params[t], Int32(1), Int32(0),
                grid_dim=(grid, 1, 1), block_dim=(elem_tpb, 1, 1),
            )
            if is_free:
                var gsub = dgrad.create_sub_buffer[DType.float32](gi * cells, cells)
                ctx.enqueue_function[gp_white_kernel](
                    gsub.unsafe_ptr(), Int32(n), Int32(n), spec.params[t], Int32(1), Int32(0),
                    grid_dim=(grid, 1, 1), block_dim=(elem_tpb, 1, 1),
                )
                _ = gsub^
                gi += 1
                count.append(1)
            else:
                count.append(0)
        else:
            var ln = Int(spec.ls_len[t])
            var lsview = dls.create_sub_buffer[DType.float32](Int(spec.ls_off[t]), ln)
            var form = GP_GRAD_RBF
            if kind == GP_K_RBF:
                ctx.enqueue_function[gp_rbf_kernel](
                    slot.unsafe_ptr(), x.unsafe_ptr(), x2.unsafe_ptr(), lsview.unsafe_ptr(),
                    Int32(n), Int32(n), Int32(d), spec.ls_len[t],
                    grid_dim=(grid, 1, 1), block_dim=(elem_tpb, 1, 1),
                )
            else:
                var nu_sel = gp_matern_nu_selector(spec.params[t])
                form = GP_GRAD_MATERN05 + nu_sel
                ctx.enqueue_function[gp_matern_kernel](
                    slot.unsafe_ptr(), x.unsafe_ptr(), x2.unsafe_ptr(), lsview.unsafe_ptr(),
                    Int32(n), Int32(n), Int32(d), spec.ls_len[t], Int32(nu_sel), sqrt3, sqrt5,
                    grid_dim=(grid, 1, 1), block_dim=(elem_tpb, 1, 1),
                )
            if is_free:
                var gsub = dgrad.create_sub_buffer[DType.float32](gi * cells, ln * cells)
                ctx.enqueue_function[gp_ls_grad_kernel](
                    gsub.unsafe_ptr(), slot.unsafe_ptr(), x.unsafe_ptr(), lsview.unsafe_ptr(),
                    Int32(n), Int32(d), Int32(ln), Int32(form), sqrt3, sqrt5,
                    grid_dim=(grid, 1, 1), block_dim=(elem_tpb, 1, 1),
                )
                _ = gsub^
                gi += ln
                count.append(ln)
            else:
                count.append(0)
            _ = lsview^
        _ = slot^
        sp += 1

    var root = stack.create_sub_buffer[DType.float32](0, cells)
    ctx.enqueue_function[gp_copy_kernel](
        out.unsafe_ptr(), root.unsafe_ptr(), Int32(cells),
        grid_dim=(grid, 1, 1), block_dim=(elem_tpb, 1, 1),
    )
    _ = root^
    return n_free
