# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SHIPS: compiled into a CPU host binding (python/mojolearn/host_surface.py names which); product, not only a check.
"""Profile `mojolearn.identical.gp.fp32.v1` on the HOST: exact dense
Gaussian process regression, fit and predict, with no device (workstream
E, the gp host lane, 2026-09-14).

WHAT THIS IS. `gaussian_process/estimator.mojo::gpr_fit_host` and
`gpr_predict_host` launch the kernels of `gaussian_process/checks/
kernels.mojo`, factor through `cholesky/estimator.mojo` and take the
posterior mean from `gemm/checks/gemm_identical.mojo::identical_gemm_into`.
This file is a SECOND spelling of that path, in the device's order, so the
`gp` CPU host binding (`bindings/_mojolearn_gp_host.mojo`) can serve
`GaussianProcessRegressor` on a CPU-only install. It imports only the
`checks/numerics.mojo` seams, `cholesky/host/chol_oracle.mojo` (the
Cholesky profile restated on the host) and `gemm/host/gemm_oracle.mojo`
(the gemm profile's normative answer, GPU-free host code already shipped).
`kernels.mojo` is not imported because it imports `std.gpu` and
`max.gpu.host`; its host-side constructors and validators are restated here
under the same refusal sentences.

THE FIT HERE IS THE ONE-SHOT FIT: one kernel matrix, one ridge, one
factorization, one solve and three scalars, at the kernel it is handed
(`gpr_fit_host`'s own contract; `optimizer` and `n_restarts_optimizer` are
not arguments of this entry). Kernel hyperparameter optimization (2026-09-15)
is `gaussian_process/host/gpr_grad_oracle.mojo` -- the likelihood and its
gradient at a candidate kernel, the verifier of
`estimator.mojo::gpr_lml_grad_host` -- driven by
`python/mojolearn/_gp_optimizer.py` (DEVIATIONS 2880 and 2881), which runs
the same state machine on every column.

THE STAGES, EACH WITH THE DEVICE LINE IT MIRRORS

    validation          estimator.mojo:613-617, in that order:
                        gp_validate_data, gp_validate_targets,
                        gp_validate_kernel, gp_validate_alpha
    K = k(X, X)         kernels.mojo:1069-1305 (gp_kernel_matrix, the
                        postfix walk) over kernels.mojo:737-958 (the scaled
                        distance with l2_unexp_core, the const, white, RBF,
                        Matern and combine kernels), is_self True
    the ridge, L        estimator.mojo:714 -> chol_oracle.mojo
                        (potrf.mojo:618-1182), alpha IS the jitter
    dual = K^-1 y       estimator.mojo:722 -> chol_host_solve (trsm.mojo)
    logdet              estimator.mojo:764-788 -> the factor's logdet
                        (potrf.mojo:780-818), never recomputed
    y^T dual            estimator.mojo:791-821, i ascending
    lml                 estimator.mojo:824-855, t1 + t2 then + t3
    info != 0           estimator.mojo:730-732: the partial factor, a zero
                        dual and zero scalars
    kss                 kernels.mojo:691-729 (gp_kernel_diag)
    K_* = k(Xtr, X*)    estimator.mojo:1010-1026, is_self False, stored
                        n_train x n_star (DEVIATION 1758)
    mean                estimator.mojo:1040-1042, identical_gemm_into at
                        OP_TN -> gemm_oracle (m = n_star, n = 1,
                        k = n_train)
    V = L^-1 K_*        estimator.mojo:1053-1062 (trsm_lower in place)
    var, std, clamp     kernels.mojo:961-1034 (gp_variance_kernel)

THE SABOTAGE (-D MOJOLEARN_HOST_SABOTAGE=1, the routed set's negative
control). The scaled squared distance walks the feature axis DESCENDING
(`GPR_ORACLE_HOST_SABOTAGE`), so every off-diagonal kernel cell of every
fit and every cross-covariance cell of every predict moves; the diagonal
stays exactly 1 (a chain of exact zeros in any order). The Cholesky
trailing update and the posterior mean move as well through
`gemm_oracle`'s own descending leaf.
"""

from std.memory import bitcast
from std.sys.compile import is_defined

from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_div,
    identical_exp,
    identical_mul,
    identical_mul_add,
    identical_sqrt,
)
from cholesky.host.chol_oracle import (
    chol_host_hex32_bits,
    chol_host_jitter_pinned,
    chol_host_potrf,
    chol_host_solve,
    chol_host_trsm_lower,
)
from gemm.host.identical_gemm import OP_TN, gemm_oracle

#: THE NEGATIVE CONTROL. See this file's header.
comptime GPR_ORACLE_HOST_SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()

#: `kernels.mojo::GP_PROFILE`.
comptime GPR_HOST_PROFILE = "mojolearn.identical.gp.fp32.v1"

# `kernels.mojo`'s node kinds, mirrored in `_gp_impl.py`.
comptime GPR_K_CONST = 0
comptime GPR_K_WHITE = 1
comptime GPR_K_RBF = 2
comptime GPR_K_MATERN = 3
comptime GPR_K_SUM = 4
comptime GPR_K_PROD = 5
comptime GPR_K_KIND_COUNT = 6

#: `kernels.mojo::GP_MAX_NODES` and `GP_MAX_STACK` (DEVIATION 1756).
comptime GPR_MAX_NODES = 16
comptime GPR_MAX_STACK = 4

#: `kernels.mojo`'s pinned constants, by their bits (DEVIATION 1767).
comptime GPR_SQRT3_BITS: UInt32 = 0x3FDDB3D7
comptime GPR_SQRT5_BITS: UInt32 = 0x400F1BBD
comptime GPR_LOG_2PI_BITS: UInt32 = 0x3FEB3F8E
comptime GPR_NU_0_5_BITS: UInt32 = 0x3F000000
comptime GPR_NU_1_5_BITS: UInt32 = 0x3FC00000
comptime GPR_NU_2_5_BITS: UInt32 = 0x40200000


# ===========================================================================
# THE KERNEL EXPRESSION (kernels.mojo:218-729, host code on both paths)
# ===========================================================================


@fieldwise_init
struct GPHostKernelSpec(Copyable, Movable):
    """`kernels.mojo::GPKernelSpec`, field for field."""

    var kinds: List[Int32]
    var params: List[Float32]
    var ls_off: List[Int32]
    var ls_len: List[Int32]
    var length_scales: List[Float32]


def _leaf(kind: Int, param: Float32, ls: List[Float32]) -> GPHostKernelSpec:
    var kinds = List[Int32]()
    kinds.append(Int32(kind))
    var params = List[Float32]()
    params.append(param)
    var off = List[Int32]()
    off.append(Int32(0))
    var ln = List[Int32]()
    ln.append(Int32(len(ls)))
    return GPHostKernelSpec(kinds^, params^, off^, ln^, ls.copy())


def gpr_host_kernel_const(constant_value: Float32) raises -> GPHostKernelSpec:
    """`kernels.mojo::gp_kernel_const`."""
    if constant_value != constant_value:
        raise Error(
            "gp_kernel_const: constant_value is NaN; refused by name"
            " (DEVIATION 1768)"
        )
    if constant_value < Float32(0.0):
        raise Error(
            "gp_kernel_const: constant_value must be non-negative, got bits"
            " 0x"
            + chol_host_hex32_bits(constant_value)
            + ". A negative constant kernel is not positive semi-definite,"
            " so the factorization would refuse at a pivot several stages"
            " downstream of the actual mistake"
        )
    return _leaf(GPR_K_CONST, constant_value, List[Float32]())


def gpr_host_kernel_white(noise_level: Float32) raises -> GPHostKernelSpec:
    """`kernels.mojo::gp_kernel_white`."""
    if noise_level != noise_level:
        raise Error(
            "gp_kernel_white: noise_level is NaN; refused by name"
            " (DEVIATION 1768)"
        )
    if noise_level < Float32(0.0):
        raise Error(
            "gp_kernel_white: noise_level must be non-negative, got bits 0x"
            + chol_host_hex32_bits(noise_level)
        )
    return _leaf(GPR_K_WHITE, noise_level, List[Float32]())


def _validate_length_scale(ls: List[Float32], what: String) raises:
    """`kernels.mojo::_validate_length_scale`."""
    if len(ls) < 1:
        raise Error(
            what
            + ": length_scale must have at least one entry (one for an"
            " isotropic kernel, n_features for ARD)"
        )
    for i in range(len(ls)):
        var v = ls[i]
        if v != v:
            raise Error(
                what
                + ": length_scale["
                + String(i)
                + "] is NaN; refused by name (DEVIATION 1768)"
            )
        if not (v > Float32(0.0)):
            raise Error(
                what
                + ": length_scale["
                + String(i)
                + "] must be strictly positive, got bits 0x"
                + chol_host_hex32_bits(v)
                + ". A zero or negative length scale divides every"
                " coordinate by it, and sklearn's own"
                " length_scale_bounds refuse the same range"
            )
        if v > Float32(3.4028234663852886e38):
            raise Error(
                what
                + ": length_scale["
                + String(i)
                + "] is infinite; refused by name"
            )


def gpr_host_kernel_rbf(length_scale: List[Float32]) raises -> GPHostKernelSpec:
    """`kernels.mojo::gp_kernel_rbf`."""
    _validate_length_scale(length_scale, String("gp_kernel_rbf"))
    return _leaf(GPR_K_RBF, Float32(0.0), length_scale)


def gpr_host_kernel_matern(
    length_scale: List[Float32], nu: Float32
) raises -> GPHostKernelSpec:
    """`kernels.mojo::gp_kernel_matern`: the three closed forms only, by
    bits (DEVIATION 1765). Every other nu is refused by name: there is no
    CPU implementation of the general Matern (the Bessel branch) because
    there is no implementation of it on any column."""
    _validate_length_scale(length_scale, String("gp_kernel_matern"))
    var nub = bitcast[DType.uint32](nu)
    if not (
        nub == GPR_NU_0_5_BITS or nub == GPR_NU_1_5_BITS or nub == GPR_NU_2_5_BITS
    ):
        raise Error(
            "gp_kernel_matern: refusing nu with bits 0x"
            + chol_host_hex32_bits(nu)
            + ". Only the three CLOSED FORMS are implemented -- nu = 0.5"
            " (bits 0x3f000000), 1.5 (0x3fc00000) and 2.5 (0x40200000),"
            " scikit-learn kernels.py:1722-1730. The general case needs kv,"
            " a modified Bessel function of the second kind, which has no"
            " implementation in this repository on any column. nu = inf is"
            " the RBF and is refused here too. DEVIATION 1765"
        )
    return _leaf(GPR_K_MATERN, nu, length_scale)


def _combine(
    a: GPHostKernelSpec, b: GPHostKernelSpec, op: Int
) raises -> GPHostKernelSpec:
    """`kernels.mojo::_combine`: a's nodes, b's nodes with their offsets
    shifted past a's table, then the operator."""
    var kinds = List[Int32]()
    var params = List[Float32]()
    var off = List[Int32]()
    var ln = List[Int32]()
    var ls = List[Float32]()
    for i in range(len(a.length_scales)):
        ls.append(a.length_scales[i])
    var shift = len(a.length_scales)
    for t in range(len(a.kinds)):
        kinds.append(a.kinds[t])
        params.append(a.params[t])
        off.append(a.ls_off[t])
        ln.append(a.ls_len[t])
    for i in range(len(b.length_scales)):
        ls.append(b.length_scales[i])
    for t in range(len(b.kinds)):
        kinds.append(b.kinds[t])
        params.append(b.params[t])
        off.append(b.ls_off[t] + Int32(shift))
        ln.append(b.ls_len[t])
    kinds.append(Int32(op))
    params.append(Float32(0.0))
    off.append(Int32(0))
    ln.append(Int32(0))
    if len(kinds) > GPR_MAX_NODES:
        raise Error(
            "gp_kernel: the composed expression has "
            + String(len(kinds))
            + " postfix nodes and GP_MAX_NODES is "
            + String(GPR_MAX_NODES)
            + ". That is a pinned CAPACITY, not a numerical parameter (no"
            " bit depends on it), and it is refused rather than grown"
            " silently because the device operand stack is allocated from"
            " it. DEVIATION 1756"
        )
    return GPHostKernelSpec(kinds^, params^, off^, ln^, ls^)


def gpr_host_kernel_sum(
    a: GPHostKernelSpec, b: GPHostKernelSpec
) raises -> GPHostKernelSpec:
    return _combine(a, b, GPR_K_SUM)


def gpr_host_kernel_prod(
    a: GPHostKernelSpec, b: GPHostKernelSpec
) raises -> GPHostKernelSpec:
    return _combine(a, b, GPR_K_PROD)


def gpr_host_kernel_stack_depth(spec: GPHostKernelSpec) raises -> Int:
    """`kernels.mojo::gp_kernel_stack_depth`."""
    if len(spec.kinds) < 1:
        raise Error("gp_kernel: the expression has no nodes")
    if len(spec.kinds) > GPR_MAX_NODES:
        raise Error(
            "gp_kernel: the expression has "
            + String(len(spec.kinds))
            + " nodes and GP_MAX_NODES is "
            + String(GPR_MAX_NODES)
        )
    var sp = 0
    var peak = 0
    for t in range(len(spec.kinds)):
        var k = Int(spec.kinds[t])
        if k == GPR_K_SUM or k == GPR_K_PROD:
            if sp < 2:
                raise Error(
                    "gp_kernel: the postfix node at index "
                    + String(t)
                    + " is an operator with "
                    + String(sp)
                    + " operand(s) on the stack. The expression is"
                    " ill-formed"
                )
            sp -= 1
        elif k >= 0 and k < GPR_K_SUM:
            sp += 1
            if sp > peak:
                peak = sp
        else:
            raise Error(
                "gp_kernel: the postfix node at index "
                + String(t)
                + " has kind "
                + String(k)
                + ", which is not one of the "
                + String(GPR_K_KIND_COUNT)
                + " GP_K_* values"
            )
    if sp != 1:
        raise Error(
            "gp_kernel: the postfix expression leaves "
            + String(sp)
            + " matrices on the stack; a well-formed one leaves exactly 1"
        )
    if peak > GPR_MAX_STACK:
        raise Error(
            "gp_kernel: the expression needs an operand stack "
            + String(peak)
            + " deep and GP_MAX_STACK is "
            + String(GPR_MAX_STACK)
            + ". That is a pinned CAPACITY (DEVIATION 1756). Refused rather"
            " than grown silently"
        )
    return peak


def gpr_host_validate_kernel(spec: GPHostKernelSpec, n_features: Int) raises:
    """`kernels.mojo::gp_validate_kernel`."""
    _ = gpr_host_kernel_stack_depth(spec)
    if n_features < 1:
        raise Error(
            "gp_validate_kernel: n_features must be positive, got "
            + String(n_features)
        )
    for t in range(len(spec.kinds)):
        var k = Int(spec.kinds[t])
        if k != GPR_K_RBF and k != GPR_K_MATERN:
            continue
        var ln = Int(spec.ls_len[t])
        var off = Int(spec.ls_off[t])
        if ln != 1 and ln != n_features:
            raise Error(
                "gp_validate_kernel: node "
                + String(t)
                + " has a length scale of "
                + String(ln)
                + " entries; scikit-learn's _check_length_scale"
                " (kernels.py:34-48) accepts 1 (isotropic) or n_features="
                + String(n_features)
                + " (ARD) and nothing between"
            )
        if off < 0 or off + ln > len(spec.length_scales):
            raise Error(
                "gp_validate_kernel: node "
                + String(t)
                + " addresses length scales ["
                + String(off)
                + ", "
                + String(off + ln)
                + ") of a table holding "
                + String(len(spec.length_scales))
            )


def gpr_host_kernel_diag(spec: GPHostKernelSpec) raises -> Float32:
    """`kernels.mojo::gp_kernel_diag` (DEVIATION 1770): one scalar, the
    postfix fold of the leaves' diagonals through `ftz` and
    `identical_mul`."""
    _ = gpr_host_kernel_stack_depth(spec)
    var stack = List[Float32]()
    for t in range(len(spec.kinds)):
        var k = Int(spec.kinds[t])
        if k == GPR_K_CONST or k == GPR_K_WHITE:
            stack.append(ftz(spec.params[t]))
        elif k == GPR_K_RBF or k == GPR_K_MATERN:
            stack.append(Float32(1.0))
        else:
            var b = stack[len(stack) - 1]
            var a = stack[len(stack) - 2]
            _ = stack.pop()
            _ = stack.pop()
            if k == GPR_K_PROD:
                stack.append(ftz(identical_mul(a, b)))
            else:
                stack.append(ftz(a + b))
    return stack[0]


def _matern_selector(nu: Float32) raises -> Int:
    """`kernels.mojo::gp_matern_nu_selector`."""
    var b = bitcast[DType.uint32](nu)
    if b == GPR_NU_0_5_BITS:
        return 0
    if b == GPR_NU_1_5_BITS:
        return 1
    if b == GPR_NU_2_5_BITS:
        return 2
    raise Error(
        "gp_matern_nu_selector: nu bits 0x"
        + chol_host_hex32_bits(nu)
        + " is not one of the three closed forms. DEVIATION 1765"
    )


# ===========================================================================
# THE KERNEL MATRIX (kernels.mojo:737-958 and 1069-1305)
# ===========================================================================


def _scaled_sqdist(
    x: List[Float32],
    y: List[Float32],
    table: List[Float32],
    ls_off: Int,
    ls_len: Int,
    i: Int,
    j: Int,
    d: Int,
) -> Float32:
    """`kernels.mojo::gp_scaled_sqdist`: per feature, both coordinates
    divided by the length scale through `identical_div`, then
    `l2_unexp_core` (`diff = ftz(xv - yv); ftz(fma(diff, diff, acc))`,
    neighbors/impl/distance/detail/distance_ops.mojo:405-408), `f`
    ascending, seeded +0.0."""
    var acc = Float32(0.0)
    for q in range(d):
        var f = q
        comptime if GPR_ORACLE_HOST_SABOTAGE:
            # THE SABOTAGE ARM: the feature axis descending.
            f = d - 1 - q
        var li = f
        if ls_len == 1:
            li = 0
        var lv = ftz(table[ls_off + li])
        var xv = ftz(identical_div(ftz(x[i * d + f]), lv))
        var yv = ftz(identical_div(ftz(y[j * d + f]), lv))
        var diff = ftz(xv - yv)
        acc = ftz(identical_mul_add(diff, diff, acc))
    return ftz(acc)


def gpr_host_kernel_matrix(
    x: List[Float32],
    m: Int,
    y: List[Float32],
    n: Int,
    d: Int,
    spec: GPHostKernelSpec,
    is_self: Bool,
) raises -> List[Float32]:
    """`kernels.mojo::gp_kernel_matrix` on one device (MOJOLEARN_GP_DEVICE_COUNT
    unset; the row-sharded driver is par-gp's and is not served here):
    `out[m x n] = k(x[m x d], y[n x d])`, row-major, the postfix list walked
    once, a leaf pushing a matrix, an operator combining the top two into
    the deeper slot, the root copied bit for bit."""
    if m <= 0 or n <= 0 or d <= 0:
        raise Error(
            "gp_kernel_matrix: m, n and d must all be positive, got "
            + String(m)
            + ", "
            + String(n)
            + ", "
            + String(d)
        )
    gpr_host_validate_kernel(spec, d)
    var cells = m * n
    var sqrt3 = bitcast[DType.float32](GPR_SQRT3_BITS)
    var sqrt5 = bitcast[DType.float32](GPR_SQRT5_BITS)
    var stack = List[List[Float32]]()
    for t in range(len(spec.kinds)):
        var kind = Int(spec.kinds[t])
        if kind == GPR_K_SUM or kind == GPR_K_PROD:
            # gp_combine_kernel: into the LEFT (deeper) operand's slot.
            var rhs = stack.pop()
            var lhs = stack.pop()
            for c in range(cells):
                var av = ftz(lhs[c])
                var bv = ftz(rhs[c])
                if kind == GPR_K_PROD:
                    lhs[c] = ftz(identical_mul(av, bv))
                else:
                    lhs[c] = ftz(av + bv)
            stack.append(lhs^)
            _ = rhs^
            continue
        var slot = List[Float32](capacity=cells)
        if kind == GPR_K_CONST:
            # gp_const_kernel
            var v = ftz(spec.params[t])
            for _c in range(cells):
                slot.append(v)
        elif kind == GPR_K_WHITE:
            # gp_white_kernel: the STRUCTURAL test (DEVIATION 1762), global
            # row start 0.
            for i in range(m):
                for j in range(n):
                    if is_self and i == j:
                        slot.append(ftz(spec.params[t]))
                    else:
                        slot.append(Float32(0.0))
        else:
            var off = Int(spec.ls_off[t])
            var ln = Int(spec.ls_len[t])
            if kind == GPR_K_RBF:
                # gp_rbf_kernel
                for i in range(m):
                    for j in range(n):
                        var d2 = _scaled_sqdist(
                            x, y, spec.length_scales, off, ln, i, j, d
                        )
                        var e = ftz(identical_mul(Float32(-0.5), d2))
                        slot.append(ftz(identical_exp(e)))
            else:
                # gp_matern_kernel, the three closed forms in sklearn's order
                var nu_sel = _matern_selector(spec.params[t])
                for i in range(m):
                    for j in range(n):
                        var d2 = _scaled_sqdist(
                            x, y, spec.length_scales, off, ln, i, j, d
                        )
                        var dist = ftz(identical_sqrt(d2))
                        if nu_sel == 0:
                            slot.append(ftz(identical_exp(-dist)))
                        elif nu_sel == 1:
                            var s = ftz(identical_mul(dist, sqrt3))
                            var pre = ftz(Float32(1.0) + s)
                            slot.append(
                                ftz(identical_mul(pre, ftz(identical_exp(-s))))
                            )
                        else:
                            var s5 = ftz(identical_mul(dist, sqrt5))
                            var ss = ftz(identical_mul(s5, s5))
                            var third = ftz(identical_div(ss, Float32(3.0)))
                            var pre5 = ftz(ftz(Float32(1.0) + s5) + third)
                            slot.append(
                                ftz(identical_mul(pre5, ftz(identical_exp(-s5))))
                            )
        stack.append(slot^)
    # gp_copy_kernel: bit for bit, no ftz.
    return stack.pop()


# ===========================================================================
# VALIDATION (estimator.mojo:248-407, host code on both paths)
# ===========================================================================


def gpr_host_validate_alpha(alpha: Float32) raises:
    """`estimator.mojo::gp_validate_alpha`."""
    if alpha != alpha:
        raise Error(
            "gpr_fit_host: alpha is NaN; refused by name (DEVIATION 1768)"
        )
    if alpha < Float32(0.0):
        raise Error(
            "gpr_fit_host: alpha must be non-negative, got bits 0x"
            + chol_host_hex32_bits(alpha)
            + ". alpha is a RIDGE added to the diagonal of the kernel"
            " matrix (scikit-learn _gpr.py:350); a negative one subtracts"
            " from the diagonal and turns a positive-definite kernel"
            " matrix indefinite"
        )
    if alpha > Float32(3.4028234663852886e38):
        raise Error("gpr_fit_host: alpha is +inf; refused by name")
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        var pinned = chol_host_jitter_pinned()
        var bits = bitcast[DType.uint32](alpha)
        if not (bits == UInt32(0) or bits == bitcast[DType.uint32](pinned)):
            raise Error(
                "gpr_fit_host: NUMERIC_IDENTICAL refuses the unpinned"
                " alpha 0x"
                + chol_host_hex32_bits(alpha)
                + ". The ridge is part of profile "
                + GPR_HOST_PROFILE
                + ", which CONTAINS mojolearn.identical.cholesky.fp32.v1,"
                " and alpha IS that profile's jitter passed through"
                " unchanged (DEVIATION 1751). The two pinned values are"
                " 0x00000000 (no ridge) and 0x"
                + chol_host_hex32_bits(pinned)
                + " (2^-20). scikit-learn's default alpha=1e-10 is a NO-OP"
                " on a unit float32 diagonal (DEVIATION 1752). DEVIATION 1637"
            )


def gpr_host_validate_data(
    x: List[Float32], rows: Int, cols: Int, what: String
) raises:
    """`estimator.mojo::gp_validate_data`."""
    if rows <= 0:
        raise Error(
            "gpr: " + what + " must have at least one row, got "
            + String(rows)
        )
    if cols <= 0:
        raise Error(
            "gpr: " + what + " must have at least one feature, got "
            + String(cols)
        )
    if len(x) != rows * cols:
        raise Error(
            "gpr: "
            + what
            + " holds "
            + String(len(x))
            + " floats, "
            + String(rows)
            + " x "
            + String(cols)
            + " needs "
            + String(rows * cols)
        )
    for i in range(len(x)):
        var v = x[i]
        if v != v:
            raise Error(
                "gpr: "
                + what
                + " contains NaN at row "
                + String(i // cols)
                + ", feature "
                + String(i % cols)
                + "; refused by name before any launch, because a NaN"
                " carries the VENDOR's payload and every stage here is a"
                " certified card stage (IDENTITY_PATHS row 39)"
            )
        if v > Float32(3.4028234663852886e38) or v < Float32(
            -3.4028234663852886e38
        ):
            raise Error(
                "gpr: "
                + what
                + " contains infinity at row "
                + String(i // cols)
                + ", feature "
                + String(i % cols)
                + " (bits 0x"
                + chol_host_hex32_bits(v)
                + "); refused by name. An infinite coordinate reaches exp"
                " and returns a zero or a NaN rather than an error"
            )


def gpr_host_validate_targets(y: List[Float32], n_train: Int) raises:
    """`estimator.mojo::gp_validate_targets`."""
    if len(y) != n_train:
        raise Error(
            "gpr_fit_host: y holds "
            + String(len(y))
            + " values for "
            + String(n_train)
            + " training rows. **MULTI-OUTPUT IS NOT IMPLEMENTED**"
            " (gaussian_process/NOT_IMPLEMENTED.tsv, DEVIATION 1763)"
        )
    for i in range(len(y)):
        var v = y[i]
        if v != v:
            raise Error(
                "gpr_fit_host: y contains NaN at index "
                + String(i)
                + "; refused by name (DEVIATION 1768)"
            )
        if v > Float32(3.4028234663852886e38) or v < Float32(
            -3.4028234663852886e38
        ):
            raise Error(
                "gpr_fit_host: y contains infinity at index "
                + String(i)
                + "; refused by name (DEVIATION 1768)"
            )


# ===========================================================================
# FIT (estimator.mojo:573-748)
# ===========================================================================


@fieldwise_init
struct GPHostFit(Movable):
    """What `gpr_fit_binding` writes back: `GPRegressor`'s l, dual_coef,
    info, nb, logdet, ydotalpha and lml."""

    var l: List[Float32]
    var dual: List[Float32]
    var info: Int
    var nb: Int
    var logdet: Float32
    var ydotalpha: Float32
    var lml: Float32


def gpr_host_lml(ydotalpha: Float32, logdet: Float32, n: Int) -> Float32:
    """`estimator.mojo::gp_log_marginal_likelihood_value`: two named
    partials, then the third term, every product `identical_mul`."""
    var t1 = ftz(identical_mul(Float32(-0.5), ftz(ydotalpha)))
    var t2 = ftz(identical_mul(Float32(-0.5), ftz(logdet)))
    var half_n = ftz(identical_mul(Float32(-0.5), Float32(n)))
    var t3 = ftz(identical_mul(half_n, bitcast[DType.float32](GPR_LOG_2PI_BITS)))
    return ftz(ftz(t1 + t2) + t3)


def gpr_host_fit(
    x: List[Float32],
    n_train: Int,
    n_features: Int,
    y: List[Float32],
    spec: GPHostKernelSpec,
    alpha: Float32,
) raises -> GPHostFit:
    """`gpr_fit_host(x, n_train, n_features, y, kernel, alpha)` at its
    defaults (optimizer 'none', no restarts, normalize_y False, no
    sabotage probe, no trace), on the host."""
    gpr_host_validate_data(x, n_train, n_features, String("X"))
    gpr_host_validate_targets(y, n_train)
    gpr_host_validate_kernel(spec, n_features)
    gpr_host_validate_alpha(alpha)

    var k = gpr_host_kernel_matrix(x, n_train, x, n_train, n_features, spec, True)
    # The ridge is applied inside the factorization, because alpha IS the
    # Cholesky profile's jitter (DEVIATION 1751).
    var factor = chol_host_potrf(k, n_train, alpha)
    _ = k^

    var dual = List[Float32]()
    var logdet = Float32(0.0)
    var ydotalpha = Float32(0.0)
    var lml = Float32(0.0)
    if factor.info == 0:
        dual = chol_host_solve(factor, y, 1)
        # _logdet_of: the factor's own value (cholesky_logdet_host).
        logdet = factor.logdet
        # _y_dot_alpha: i ascending.
        var acc = Float32(0.0)
        for i in range(n_train):
            acc = ftz(identical_mul_add(ftz(y[i]), ftz(dual[i]), acc))
        ydotalpha = ftz(acc)
        lml = gpr_host_lml(ydotalpha, logdet, n_train)
    else:
        for _i in range(n_train):
            dual.append(Float32(0.0))
    var info = factor.info
    var nb = factor.nb
    return GPHostFit(factor.l.copy(), dual^, info, nb, logdet, ydotalpha, lml)


# ===========================================================================
# PREDICT (estimator.mojo:897-1099)
# ===========================================================================


@fieldwise_init
struct GPHostPrediction(Movable):
    """`estimator.mojo::GPPrediction`'s mean, variance, std, clamp flags and
    their count."""

    var mean: List[Float32]
    var variance: List[Float32]
    var std: List[Float32]
    var clamped: List[Int32]
    var n_clamped: Int


def gpr_host_predict(
    x_train: List[Float32],
    l: List[Float32],
    dual: List[Float32],
    n_train: Int,
    n_features: Int,
    spec: GPHostKernelSpec,
    info: Int,
    x_star: List[Float32],
    n_star: Int,
    return_std: Bool,
) raises -> GPHostPrediction:
    """`gpr_predict_host(model, x_star, n_star, return_std)` at its
    defaults, on the host, with the fitted arrays handed in as the GPU
    binding hands them (`bindings/_mojolearn_gp.mojo::gpr_predict_binding`)."""
    if info != 0:
        raise Error(
            "gpr_predict_host: refusing to predict from a FAILED fit"
            " (info="
            + String(info)
            + "). The factor's columns from "
            + String(info - 1)
            + " onward are unfinished, and solving against them returns"
            " infinities that look like numbers. DEVIATION 1634"
        )
    if n_star <= 0:
        raise Error(
            "gpr_predict_host: n_star must be positive, got "
            + String(n_star)
        )
    gpr_host_validate_data(x_star, n_star, n_features, String("X_star"))
    var kss = gpr_host_kernel_diag(spec)

    # K_trans^T = k(X_train, X_star), is_self False (DEVIATION 1762).
    var kcross = gpr_host_kernel_matrix(
        x_train, n_train, x_star, n_star, n_features, spec, False
    )
    # The mean BEFORE the solve, which overwrites kcross in place.
    var mean = gemm_oracle(kcross, dual, OP_TN, n_star, 1, n_train)

    var variance = List[Float32]()
    var std = List[Float32]()
    var clamped = List[Int32]()
    var n_clamped = 0
    if return_std:
        chol_host_trsm_lower(l, kcross, n_train, n_star)
        for t in range(n_star):
            var acc = Float32(0.0)
            for i in range(n_train):
                var vv = ftz(kcross[i * n_star + t])
                acc = ftz(identical_mul_add(vv, vv, acc))
            var raw = ftz(ftz(kss) - acc)
            var outv = raw
            if not (raw > Float32(0.0)):
                outv = Float32(0.0)
            var moved = bitcast[DType.uint32](outv) != bitcast[DType.uint32](raw)
            variance.append(outv)
            std.append(ftz(identical_sqrt(outv)))
            if moved:
                clamped.append(Int32(1))
                n_clamped += 1
            else:
                clamped.append(Int32(0))
    _ = kcross^
    return GPHostPrediction(mean^, variance^, std^, clamped^, n_clamped)
