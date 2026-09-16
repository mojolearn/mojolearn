# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SHIPS: compiled into a CPU host binding (python/mojolearn/host_surface.py names which); product, not only a check.
"""`gaussian_process/estimator.mojo::gpr_lml_grad_host` on the HOST (the
CPU verifier of kernel hyperparameter optimization, 2026-09-15).

    K, dK      `gaussian_process/checks/kernel_gradient.mojo`'s walk restated:
               each leaf's value is `gpr_oracle.mojo::gpr_host_kernel_matrix`
               over that leaf alone (the device's value launch), each
               gradient cell the table in that file's header, the product
               rule's order (left gradients by the right value, right
               gradients by the left value, then the value product)
    L, alpha_  `chol_host_potrf`, `chol_host_solve`
    lml        the factor's logdet, `y^T alpha_` i ascending, `gpr_host_lml`
    K^-1       `chol_host_solve` against the identity, n right-hand sides
    grad       `gp_theta.mojo::gp_lml_gradient_fold`, the one shared spelling

The distance under a length-scale gradient is `gpr_oracle.mojo`'s
`_scaled_sqdist`, so the host sabotage define reaches it with the kernel.
"""

from checks.numerics import (
    ftz,
    identical_div,
    identical_exp,
    identical_mul,
    identical_mul_add,
    identical_sqrt,
)
from cholesky.host.chol_oracle import chol_host_potrf, chol_host_solve
from gaussian_process.host.gp_theta import gp_free_count, gp_lml_gradient_fold
from gaussian_process.host.gpr_oracle import (
    GPR_K_CONST,
    GPR_K_MATERN,
    GPR_K_PROD,
    GPR_K_RBF,
    GPR_K_SUM,
    GPR_K_WHITE,
    GPR_SQRT3_BITS,
    GPR_SQRT5_BITS,
    GPHostKernelSpec,
    _matern_selector,
    _scaled_sqdist,
    gpr_host_kernel_matrix,
    gpr_host_lml,
    gpr_host_validate_alpha,
    gpr_host_validate_data,
    gpr_host_validate_kernel,
    gpr_host_validate_targets,
)
from std.memory import bitcast


@fieldwise_init
struct GPHostLmlGrad(Movable):
    """`estimator.mojo::GPLmlGrad`."""

    var lml: Float32
    var grad: List[Float32]
    var info: Int


def _grad_cell(
    form: Int, dd: Float32, d2: Float32, k: Float32, sqrt3: Float32, sqrt5: Float32
) -> Float32:
    """`kernel_gradient.mojo::gp_ls_gradient_value`: 0 RBF, 1 Matern 0.5,
    2 Matern 1.5, 3 Matern 2.5."""
    if form == 0:
        return ftz(identical_mul(dd, k))
    var dist = ftz(identical_sqrt(d2))
    if form == 1:
        if not (dist > Float32(0.0)):
            return Float32(0.0)
        return ftz(identical_mul(k, ftz(identical_div(dd, dist))))
    if form == 2:
        var s = ftz(identical_mul(dist, sqrt3))
        var e = ftz(identical_exp(-s))
        return ftz(identical_mul(ftz(identical_mul(Float32(3.0), dd)), e))
    var s5 = ftz(identical_mul(dist, sqrt5))
    var a = ftz(identical_div(ftz(identical_mul(Float32(5.0), dd)), Float32(3.0)))
    var b = ftz(identical_mul(a, ftz(Float32(1.0) + s5)))
    return ftz(identical_mul(b, ftz(identical_exp(-s5))))


def _leaf_spec(spec: GPHostKernelSpec, t: Int) -> GPHostKernelSpec:
    var kinds = List[Int32]()
    kinds.append(spec.kinds[t])
    var params = List[Float32]()
    params.append(spec.params[t])
    var off = List[Int32]()
    off.append(Int32(0))
    var ln = List[Int32]()
    ln.append(spec.ls_len[t])
    var ls = List[Float32]()
    var o = Int(spec.ls_off[t])
    for q in range(Int(spec.ls_len[t])):
        ls.append(spec.length_scales[o + q])
    return GPHostKernelSpec(kinds^, params^, off^, ln^, ls^)


def gpr_host_kernel_matrix_grad(
    x: List[Float32],
    n: Int,
    d: Int,
    spec: GPHostKernelSpec,
    free: List[Int32],
    mut grad: List[Float32],
) raises -> List[Float32]:
    """K (returned) and the `n_free` gradient matrices (into `grad`, which is
    resized), `kernel_gradient.mojo::gp_kernel_matrix_grad` on the host."""
    gpr_host_validate_kernel(spec, d)
    var n_free = gp_free_count(spec.kinds, spec.ls_len, free)
    var cells = n * n
    grad = List[Float32](length=max(n_free * cells, 1), fill=Float32(0.0))
    var sqrt3 = bitcast[DType.float32](GPR_SQRT3_BITS)
    var sqrt5 = bitcast[DType.float32](GPR_SQRT5_BITS)
    var stack = List[List[Float32]]()
    var first = List[Int]()
    var count = List[Int]()
    var gi = 0
    for t in range(len(spec.kinds)):
        var kind = Int(spec.kinds[t])
        if kind == GPR_K_SUM or kind == GPR_K_PROD:
            var rhs = stack.pop()
            var lhs = stack.pop()
            var fb = first.pop()
            var cb = count.pop()
            var fa = first.pop()
            var ca = count.pop()
            if kind == GPR_K_PROD:
                for q in range(ca):
                    var base = (fa + q) * cells
                    for c in range(cells):
                        grad[base + c] = ftz(identical_mul(ftz(grad[base + c]), ftz(rhs[c])))
                for q in range(cb):
                    var base = (fb + q) * cells
                    for c in range(cells):
                        grad[base + c] = ftz(identical_mul(ftz(grad[base + c]), ftz(lhs[c])))
            for c in range(cells):
                var av = ftz(lhs[c])
                var bv = ftz(rhs[c])
                if kind == GPR_K_PROD:
                    lhs[c] = ftz(identical_mul(av, bv))
                else:
                    lhs[c] = ftz(av + bv)
            stack.append(lhs^)
            _ = rhs^
            first.append(fa)
            count.append(ca + cb)
            continue
        var leaf = _leaf_spec(spec, t)
        var value = gpr_host_kernel_matrix(x, n, x, n, d, leaf, True)
        first.append(gi)
        if Int(free[t]) == 0:
            count.append(0)
        elif kind == GPR_K_CONST:
            var v = ftz(spec.params[t])
            for c in range(cells):
                grad[gi * cells + c] = v
            gi += 1
            count.append(1)
        elif kind == GPR_K_WHITE:
            for i in range(n):
                grad[gi * cells + i * n + i] = ftz(spec.params[t])
            gi += 1
            count.append(1)
        else:
            var ln = Int(spec.ls_len[t])
            var form = 0
            if kind == GPR_K_MATERN:
                form = 1 + _matern_selector(spec.params[t])
            for i in range(n):
                for j in range(n):
                    var c = i * n + j
                    var k = ftz(value[c])
                    var d2 = _scaled_sqdist(x, x, leaf.length_scales, 0, ln, i, j, d)
                    if ln == 1:
                        grad[gi * cells + c] = _grad_cell(form, d2, d2, k, sqrt3, sqrt5)
                    else:
                        for f in range(d):
                            var lv = ftz(leaf.length_scales[f])
                            var xv = ftz(identical_div(ftz(x[i * d + f]), lv))
                            var yv = ftz(identical_div(ftz(x[j * d + f]), lv))
                            var diff = ftz(xv - yv)
                            var dd = ftz(identical_mul(diff, diff))
                            grad[(gi + f) * cells + c] = _grad_cell(form, dd, d2, k, sqrt3, sqrt5)
            gi += ln
            count.append(ln)
        stack.append(value^)
    return stack.pop()


def gpr_host_lml_grad(
    x: List[Float32],
    n_train: Int,
    n_features: Int,
    y: List[Float32],
    spec: GPHostKernelSpec,
    free: List[Int32],
    alpha: Float32,
) raises -> GPHostLmlGrad:
    """`estimator.mojo::gpr_lml_grad_host` on the host."""
    gpr_host_validate_data(x, n_train, n_features, String("X"))
    gpr_host_validate_targets(y, n_train)
    gpr_host_validate_kernel(spec, n_features)
    gpr_host_validate_alpha(alpha)
    var n = n_train
    var cells = n * n
    var g = List[Float32]()
    var k = gpr_host_kernel_matrix_grad(x, n, n_features, spec, free, g)
    var n_free = gp_free_count(spec.kinds, spec.ls_len, free)
    var factor = chol_host_potrf(k, n, alpha)
    _ = k^
    if factor.info != 0:
        return GPHostLmlGrad(
            Float32(0.0), List[Float32](length=n_free, fill=Float32(0.0)), factor.info
        )
    var dual = chol_host_solve(factor, y, 1)
    var acc = Float32(0.0)
    for i in range(n):
        acc = ftz(identical_mul_add(ftz(y[i]), ftz(dual[i]), acc))
    var lml = gpr_host_lml(ftz(acc), factor.logdet, n)
    var eye = List[Float32](length=cells, fill=Float32(0.0))
    for i in range(n):
        eye[i * n + i] = Float32(1.0)
    var kinv = chol_host_solve(factor, eye, n)
    var grad = gp_lml_gradient_fold(dual, kinv, g, n, n_free)
    return GPHostLmlGrad(lml, grad^, 0)
