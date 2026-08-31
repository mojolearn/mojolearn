# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The covariance functions of profile `mojolearn.identical.gp.fp32.v1`.

**NOT A PORT, AND THERE IS NOTHING TO PORT.** cuML, cuVS and RAFT implement
no Gaussian process at any of the pinned commits; `gaussian_process/
DERIVATION_MAP.tsv` carries the grep that establishes it. `PORTING_RULES.md`'s
COPY DO NOT IMPROVE therefore does not apply here, because there is nothing
to copy. scikit-learn's `sklearn/gaussian_process/kernels.py` is the
SEMANTICS reference and the ORACLE and is never the design source: it is
CPU, LAPACK-shaped and float64, and it is cited below line by line so that
what our parameters MEAN is checkable against a file rather than against a
recollection.

THE SEMANTICS, CITED (scikit-learn 1.9.0, `77def0e`, at
`/Users/andrewhendel/CascadeProjects/upstream/scikit-learn`)

    RBF.__call__            kernels.py:1568-1570
        dists = cdist(X / length_scale, Y / length_scale, "sqeuclidean")
        K     = np.exp(-0.5 * dists)
    Matern.__call__         kernels.py:1718-1730
        dists = cdist(X / length_scale, Y / length_scale, "euclidean")
        nu=0.5   K = np.exp(-dists)
        nu=1.5   K = dists * sqrt(3);  K = (1.0 + K) * np.exp(-K)
        nu=2.5   K = dists * sqrt(5);  K = (1.0 + K + K**2 / 3.0) * np.exp(-K)
        general  K = 2^(1-nu)/gamma(nu) * (sqrt(2 nu) d)^nu * kv(nu, .)
                 -- the Bessel branch, REFUSED BY NAME here (DEVIATION 1765)
    ConstantKernel.__call__ kernels.py:1278-1282   K = full(constant_value)
    WhiteKernel.__call__    kernels.py:1406-1419
        Y is None -> noise_level * eye(n);   Y given -> zeros
    Sum.__call__            kernels.py:871      k1(X, Y) + k2(X, Y)
    Product.__call__        kernels.py:971      k1(X, Y) * k2(X, Y)
    Sum.diag / Product.diag kernels.py:889 / :989

THE ONE PLACE THE SEMANTICS REFERENCE IS DELIBERATELY NOT FOLLOWED, and it
is a simplification rather than a change: sklearn's `Y is None` path uses
`pdist` + `squareform` and then `np.fill_diagonal(K, 1)` (kernels.py:1564,
:1741), i.e. it FORCES the unit diagonal. The unexpanded distance below
returns exactly `+0.0` on the diagonal for any coordinate set, because
`x_f/l - x_f/l` is exactly zero at every `f` and a chain of `fma(0, 0, 0)`
is exactly `+0.0`, and `portable_expf(+0.0)` is exactly `1.0`. So the unit
diagonal is ARITHMETIC here where it is an assignment there, and
`check_kernels_vs_oracle` asserts it BY BITS rather than assuming it. That
matters downstream: `cholesky/README.md`'s first correction records that
every correlation-shaped kernel matrix has a unit diagonal and that the
Cholesky lane's absolute and relative jitter policies therefore COINCIDE on
exactly the matrices this file produces. Nothing here re-derives that and
nothing here adds a second jitter knob (DEVIATION 1751).

THE ARITHMETIC, and which pin each seam answers to
---------------------------------------------------
    scale     x_f / l_f          `identical_div`, row 49
    distance  sum_f (dx)^2       `l2_unexp_core`, IMPORTED from
                                 `kde/impl/distance/distance_ops.mojo`
                                 rather than re-spelled (DEVIATION 1754)
    root      sqrt(d2)           `identical_sqrt`, row 10
    exp       exp(e)             `identical_exp`, row 12
    products  a * b              `identical_mul`, row 9 (DEVIATION 826)
    every stored intermediate    `ftz`, row 10

THE SHAPE: ONE THREAD OWNS ONE CELL and walks the feature axis in its own
registers. No float crosses a thread boundary anywhere in this file: no
shared-memory staging, no block fold, no cross-block combination, no warp
primitive and no atomic of any kind. So the summation order is a pure
function of `d` and of the loops written here, and launch invariance and
batch invariance are properties of the kernel's SHAPE rather than of a check
that happens to pass. That is `neighbors/checks/pinned_distance_tile.mojo`'s
discipline and `kde/impl/distance/distance_ops.mojo`'s, applied to a
covariance function.

The price is stated rather than hidden, exactly as those two files state
theirs: this reads `d` floats of `X` and `d` of `Y` per output cell with no
reuse, where a Contractions tile would read each row once per tile. It is a
speed idea, it would introduce a staging shape, and this lane has measured
nothing, so it is named in `gaussian_process/README.md` under WHAT IS OWED
rather than attempted.

THE TOTAL ORDER, stated because IDENTITY_PATHS asks every tie to state one
--------------------------------------------------------------------------
There is no tie to break in a covariance function: no max, no min, no argmax
and no selection of any kind reaches an output here. The one comparison in
the lane is the predictive variance's clamp, and it is spelled
`not (v > 0.0)` so that NaN and both zeros take the same branch on every
vendor (IDENTITY_PATHS row 39, and row 39's measured Apple-versus-NVIDIA
split on `max(+0.0, -0.0)` is why it is not spelled as a `max`). What IS
pinned is the SUMMATION order, and it is: **`f` ASCENDING over the feature
axis, `i` ASCENDING over the training axis, in every loop of this lane.**

THE EXPRESSION IS EVALUATED IN POSTFIX AND NEVER REWRITTEN (DEVIATION 1756)
---------------------------------------------------------------------------
A kernel is a postfix (reverse-Polish) node list. `Sum` and `Product`
combine the top two matrices of a stack and nothing distributes, factors or
reassociates anything: `(A + B) * C` is three leaves and two nodes, and it
is NOT rewritten as `A*C + B*C`, because distributing changes both the
number of roundings and the order they happen in. sklearn's `Sum` and
`Product` are the same two lines and the same non-rewriting.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast
from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace
from gaussian_process.checks.gp_sabotage import (
    GP_SAB_NONE,
    gp_sabotage_touches_kernel_matrix,
    gp_sabotage_touches_variance,
    sabotage_combine_kernel,
    sabotage_matern_kernel,
    sabotage_rbf_kernel,
    sabotage_variance_kernel,
)
from kde.impl.distance.distance_ops import l2_unexp_core
from checks.numerics import (
    ftz,
    identical_div,
    identical_exp,
    identical_mul,
    identical_mul_add,
    identical_sqrt,
)


#: The profile's name. Changing any pinned constant below, the postfix
#: evaluation order, the feature-axis order or the variance fold creates a
#: v2; it does not amend v1. Same discipline as
#: `mojolearn.identical.cholesky.fp32.v1` and
#: `mojolearn.identical.gemm.fp32.v1`, both of which this profile CONTAINS.
comptime GP_PROFILE = "mojolearn.identical.gp.fp32.v1"

#: SCHEDULING. One thread owns one cell; the block width moves no bit, and
#: `check_launch_invariance` varies it precisely to say so out loud.
comptime GP_ELEM_TPB = 256

# ---------------------------------------------------------------------------
# THE NODE KINDS. Postfix; leaves push, operators pop two and push one.
# ---------------------------------------------------------------------------
comptime GP_K_CONST = 0
comptime GP_K_WHITE = 1
comptime GP_K_RBF = 2
comptime GP_K_MATERN = 3
comptime GP_K_SUM = 4
comptime GP_K_PROD = 5
comptime GP_K_KIND_COUNT = 6

#: **STRUCTURAL PIN. DEVIATION 1756.** The longest postfix node list and the
#: deepest operand stack accepted. Not a numerical parameter -- no bit
#: depends on either -- but a pinned CAPACITY: the device stack is allocated
#: at `GP_MAX_STACK * m * n` floats, and a kernel that needed a deeper stack
#: would otherwise silently overwrite an operand. Refused by name instead.
comptime GP_MAX_NODES = 16
comptime GP_MAX_STACK = 4

# ---------------------------------------------------------------------------
# THE PINNED CONSTANTS, WRITTEN AS FLOAT32 BITS. DEVIATION 1767.
#
# `[[mojo-string-float-roundtrip]]`: `String(Float32)` does not round trip in
# this toolchain, so a profile constant written as a decimal literal is a
# constant no log line can reproduce and no reviewer can check. Each is the
# correctly-rounded float32 of the float64 value sklearn uses, and
# `check_gp_constants` asserts exactly that by recomputing it from
# `std.math` in float64 and comparing bit patterns -- so if one of these
# digits is wrong the gate NAMES it instead of shipping it.
# ---------------------------------------------------------------------------

#: `sqrt(3)`, sklearn `kernels.py:1725`'s `math.sqrt(3)`.
comptime GP_SQRT3_BITS: UInt32 = 0x3FDDB3D7

#: `sqrt(5)`, sklearn `kernels.py:1728`'s `math.sqrt(5)`.
comptime GP_SQRT5_BITS: UInt32 = 0x400F1BBD

#: `log(2 pi)`, sklearn `_gpr.py:615`'s `np.log(2 * np.pi)`.
comptime GP_LOG_2PI_BITS: UInt32 = 0x3FEB3F8E

#: The three Matern orders this lane implements, as float32 bits. The
#: general case is refused (DEVIATION 1765).
comptime GP_NU_0_5_BITS: UInt32 = 0x3F000000
comptime GP_NU_1_5_BITS: UInt32 = 0x3FC00000
comptime GP_NU_2_5_BITS: UInt32 = 0x40200000


def gp_sqrt3() -> Float32:
    """`GP_SQRT3_BITS` as a value. A function rather than a `comptime`
    binding because the constant is defined by its BITS and the bitcast is
    the definition."""
    return bitcast[DType.float32](GP_SQRT3_BITS)


def gp_sqrt5() -> Float32:
    return bitcast[DType.float32](GP_SQRT5_BITS)


def gp_log_2pi() -> Float32:
    return bitcast[DType.float32](GP_LOG_2PI_BITS)


def gp_hex32_bits(v: Float32) -> String:
    """Eight lowercase hex digits of a float32's bit pattern. Every error
    message and every printed float in this lane names a float by its BITS
    beside its decimal, never by `String(Float32)` alone."""
    comptime DIGITS = "0123456789abcdef"
    var u = bitcast[DType.uint32](v)
    var out = String("")
    for i in range(8):
        var nib = Int((u >> UInt32(28 - 4 * i)) & UInt32(0xF))
        out += String(DIGITS[byte=nib])
    return out


# ===========================================================================
# THE KERNEL EXPRESSION
# ===========================================================================


@fieldwise_init
struct GPKernelSpec(Copyable, Movable):
    """A covariance function as a POSTFIX node list. DEVIATION 1756.

    Five parallel host lists rather than a tree of heap nodes, because the
    evaluation is a loop over `kinds` on the host that launches one kernel
    per node, and because a flat list is what an `IdentityTrace` can hash and
    what a binding layer can hand over without a graph representation.

    `kinds[t]`      one of the `GP_K_*` values
    `params[t]`     `constant_value` for CONST, `noise_level` for WHITE,
                    `nu` for MATERN, unused (`+0.0`) for RBF, SUM and PROD
    `ls_off[t]`     offset into `length_scales` for RBF and MATERN
    `ls_len[t]`     1 for an isotropic length scale, `n_features` for ARD.
                    **Anisotropic IS the ARD case**; there is no third
                    spelling, exactly as sklearn's `RBF.anisotropic` is
                    `length_scale` having more than one entry
    `length_scales` every leaf's length scales, concatenated

    Nothing here is mutable after construction and there is no `theta`, no
    bounds and no gradient: hyperparameter optimization is not implemented
    (DEVIATION 1761) and the pieces that exist only to serve it are absent
    rather than present and unused.
    """

    var kinds: List[Int32]
    var params: List[Float32]
    var ls_off: List[Int32]
    var ls_len: List[Int32]
    var length_scales: List[Float32]


def _leaf_spec(
    kind: Int, param: Float32, ls: List[Float32]
) -> GPKernelSpec:
    var kinds = List[Int32]()
    kinds.append(Int32(kind))
    var params = List[Float32]()
    params.append(param)
    var off = List[Int32]()
    off.append(Int32(0))
    var ln = List[Int32]()
    ln.append(Int32(len(ls)))
    return GPKernelSpec(kinds^, params^, off^, ln^, ls.copy())


def gp_kernel_const(constant_value: Float32) raises -> GPKernelSpec:
    """`ConstantKernel(constant_value)`, sklearn `kernels.py:1187`.

    `k(x, y) = constant_value` for every pair, including the diagonal and
    including the cross-covariance. sklearn's constructor requires a
    positive value through its `constant_value_bounds`; a NON-FINITE or
    NEGATIVE value is refused here by name, because a negative constant
    kernel is not a covariance function and the Cholesky would discover that
    as a pivot failure several stages downstream of the mistake.
    """
    if constant_value != constant_value:
        raise Error(
            "gp_kernel_const: constant_value is NaN; refused by name"
            " (DEVIATION 1768)"
        )
    if constant_value < Float32(0.0):
        raise Error(
            "gp_kernel_const: constant_value must be non-negative, got bits"
            " 0x"
            + gp_hex32_bits(constant_value)
            + ". A negative constant kernel is not positive semi-definite,"
            " so the factorization would refuse at a pivot several stages"
            " downstream of the actual mistake"
        )
    return _leaf_spec(GP_K_CONST, constant_value, List[Float32]())


def gp_kernel_white(noise_level: Float32) raises -> GPKernelSpec:
    """`WhiteKernel(noise_level)`, sklearn `kernels.py:1325`.

    `k(X, X) = noise_level * I` and `k(X, Y) = 0` for `Y` given
    (`kernels.py:1406-1419`). **THAT IS A STRUCTURAL TEST, NOT A COORDINATE
    TEST** (DEVIATION 1762): sklearn asks whether the second argument was
    passed, not whether the two rows are equal. Two IDENTICAL training rows
    therefore get the white noise on the diagonal and NOT on their
    off-diagonal cell, which is exactly what makes a duplicated input a
    rank-deficiency the ridge has to fix rather than a case the white noise
    quietly absorbs. `check_duplicate_inputs_need_the_ridge` is that gate.
    """
    if noise_level != noise_level:
        raise Error(
            "gp_kernel_white: noise_level is NaN; refused by name"
            " (DEVIATION 1768)"
        )
    if noise_level < Float32(0.0):
        raise Error(
            "gp_kernel_white: noise_level must be non-negative, got bits 0x"
            + gp_hex32_bits(noise_level)
        )
    return _leaf_spec(GP_K_WHITE, noise_level, List[Float32]())


def _validate_length_scale(ls: List[Float32], what: String) raises:
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
                + gp_hex32_bits(v)
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


def gp_kernel_rbf(length_scale: List[Float32]) raises -> GPKernelSpec:
    """`RBF(length_scale)`, sklearn `kernels.py:1448`. One entry is the
    isotropic case, `n_features` entries the ARD (anisotropic) one."""
    _validate_length_scale(length_scale, String("gp_kernel_rbf"))
    return _leaf_spec(GP_K_RBF, Float32(0.0), length_scale)


def gp_kernel_matern(
    length_scale: List[Float32], nu: Float32
) raises -> GPKernelSpec:
    """`Matern(length_scale, nu)`, sklearn `kernels.py:1601`.

    **ONLY THE THREE CLOSED FORMS. DEVIATION 1765.** `nu` in
    `{0.5, 1.5, 2.5}` (sklearn `kernels.py:1722-1730`); every other value,
    including `nu = inf` (which sklearn implements as the RBF,
    `kernels.py:1731`) and the general Bessel branch
    (`kernels.py:1733-1739`), is REFUSED BY NAME. The general branch needs
    `scipy.special.kv`, a modified Bessel function of the second kind of
    real order, which is not in this repository, is not in MAX, has no
    device implementation on any of the three vendors, and would be a new
    transcendental to certify under IDENTITY_PATHS row 12 before a single
    cell of it could be trusted. Refusing is IDENTITY_PATHS' third move and
    the closure condition is named in the error.
    """
    _validate_length_scale(length_scale, String("gp_kernel_matern"))
    var nub = bitcast[DType.uint32](nu)
    var ok = (
        nub == GP_NU_0_5_BITS
        or nub == GP_NU_1_5_BITS
        or nub == GP_NU_2_5_BITS
    )
    if not ok:
        raise Error(
            "gp_kernel_matern: refusing nu with bits 0x"
            + gp_hex32_bits(nu)
            + ". Only the three CLOSED FORMS are implemented -- nu = 0.5"
            " (bits 0x3f000000), 1.5 (0x3fc00000) and 2.5 (0x40200000),"
            " scikit-learn kernels.py:1722-1730. The general case"
            " (kernels.py:1733-1739) is 2^(1-nu)/gamma(nu) *"
            " (sqrt(2 nu) d)^nu * kv(nu, sqrt(2 nu) d), where kv is a"
            " modified Bessel function of the second kind of real order."
            " There is no kv in this repository, none in MAX, and no"
            " device implementation on any of the three vendors, so"
            " porting it would mean certifying a NEW transcendental under"
            " IDENTITY_PATHS row 12 before one cell of it could be"
            " trusted. nu = inf is the RBF and is refused here too, so"
            " that a caller who wants it says RBF and gets the kernel"
            " whose gates it actually ran. To close this refusal, add"
            " kv to checks/numerics.mojo with its own bit-level"
            " certificate; to work around it today, use one of the three"
            " orders or the RBF. DEVIATION 1765"
        )
    return _leaf_spec(GP_K_MATERN, nu, length_scale)


def _combine(
    a: GPKernelSpec, b: GPKernelSpec, op: Int
) raises -> GPKernelSpec:
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
    if len(kinds) > GP_MAX_NODES:
        raise Error(
            "gp_kernel: the composed expression has "
            + String(len(kinds))
            + " postfix nodes and GP_MAX_NODES is "
            + String(GP_MAX_NODES)
            + ". That is a pinned CAPACITY, not a numerical parameter (no"
            " bit depends on it), and it is refused rather than grown"
            " silently because the device operand stack is allocated from"
            " it. DEVIATION 1756"
        )
    return GPKernelSpec(kinds^, params^, off^, ln^, ls^)


def gp_kernel_sum(a: GPKernelSpec, b: GPKernelSpec) raises -> GPKernelSpec:
    """`Sum(k1, k2)`, sklearn `kernels.py:799`. `k1(X, Y) + k2(X, Y)`,
    `kernels.py:871`. Postfix `a b +`; NOTHING is distributed or
    reassociated (DEVIATION 1756)."""
    return _combine(a, b, GP_K_SUM)


def gp_kernel_prod(a: GPKernelSpec, b: GPKernelSpec) raises -> GPKernelSpec:
    """`Product(k1, k2)`, sklearn `kernels.py:896`. `k1(X, Y) * k2(X, Y)`,
    `kernels.py:971`. Postfix `a b *`."""
    return _combine(a, b, GP_K_PROD)


def gp_kernel_from_name(name: String) raises -> Int:
    """The kernel-name table, and every UNPORTED name refused BY NAME.

    A string entry point so a binding layer, a driver's environment knob and
    an error message all agree about what the names are. Returns a
    `GP_K_*` value for the four supported leaves.
    """
    if name == "ConstantKernel":
        return GP_K_CONST
    if name == "WhiteKernel":
        return GP_K_WHITE
    if name == "RBF":
        return GP_K_RBF
    if name == "Matern":
        return GP_K_MATERN
    if (
        name == "DotProduct"
        or name == "RationalQuadratic"
        or name == "ExpSineSquared"
        or name == "Exponentiation"
        or name == "PairwiseKernel"
        or name == "CompoundKernel"
    ):
        raise Error(
            "gp_kernel_from_name: the kernel '"
            + name
            + "' exists in scikit-learn (sklearn/gaussian_process/"
            "kernels.py) and is NOT ported. gaussian_process/NOT_IMPLEMENTED.tsv"
            " carries the row and the reason. This lane implements"
            " ConstantKernel, WhiteKernel, RBF and Matern at nu in"
            " {0.5, 1.5, 2.5}, and sums and products of them. Refused by"
            " name rather than silently substituted"
        )
    raise Error(
        "gp_kernel_from_name: unknown kernel '"
        + name
        + "'. The supported names are ConstantKernel, WhiteKernel, RBF"
        " and Matern"
    )


def gp_kernel_name(spec: GPKernelSpec) -> String:
    """A one-line rendering of the postfix expression, for the card header
    and for an error message. Not sklearn's `__repr__` and not parseable
    back: it names the nodes in evaluation order, which is what a card
    reader needs."""
    var out = String("")
    for t in range(len(spec.kinds)):
        if t > 0:
            out += " "
        var k = Int(spec.kinds[t])
        if k == GP_K_CONST:
            out += "Const(0x" + gp_hex32_bits(spec.params[t]) + ")"
        elif k == GP_K_WHITE:
            out += "White(0x" + gp_hex32_bits(spec.params[t]) + ")"
        elif k == GP_K_RBF:
            out += "RBF(ls" + String(Int(spec.ls_len[t])) + ")"
        elif k == GP_K_MATERN:
            out += (
                "Matern(ls"
                + String(Int(spec.ls_len[t]))
                + ",nu=0x"
                + gp_hex32_bits(spec.params[t])
                + ")"
            )
        elif k == GP_K_SUM:
            out += "+"
        elif k == GP_K_PROD:
            out += "*"
        else:
            out += "?"
    return out


def gp_kernel_stack_depth(spec: GPKernelSpec) raises -> Int:
    """The deepest operand stack the postfix list reaches, and the arity
    check. Raises on an ill-formed expression -- a leading operator, a
    trailing leaf, an operator with one operand -- rather than reading past
    the stack on the device."""
    if len(spec.kinds) < 1:
        raise Error("gp_kernel: the expression has no nodes")
    if len(spec.kinds) > GP_MAX_NODES:
        raise Error(
            "gp_kernel: the expression has "
            + String(len(spec.kinds))
            + " nodes and GP_MAX_NODES is "
            + String(GP_MAX_NODES)
        )
    var sp = 0
    var peak = 0
    for t in range(len(spec.kinds)):
        var k = Int(spec.kinds[t])
        if k == GP_K_SUM or k == GP_K_PROD:
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
        elif k >= 0 and k < GP_K_SUM:
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
                + String(GP_K_KIND_COUNT)
                + " GP_K_* values"
            )
    if sp != 1:
        raise Error(
            "gp_kernel: the postfix expression leaves "
            + String(sp)
            + " matrices on the stack; a well-formed one leaves exactly 1"
        )
    if peak > GP_MAX_STACK:
        raise Error(
            "gp_kernel: the expression needs an operand stack "
            + String(peak)
            + " deep and GP_MAX_STACK is "
            + String(GP_MAX_STACK)
            + ". That is a pinned CAPACITY (DEVIATION 1756): the device"
            " scratch is allocated at GP_MAX_STACK * m * n floats, and a"
            " deeper expression would overwrite an operand. Refused rather"
            " than grown silently. To close this, raise GP_MAX_STACK and"
            " re-run every gate; nothing numerical depends on the value"
        )
    return peak


def gp_validate_kernel(spec: GPKernelSpec, n_features: Int) raises:
    """Every leaf's length scale is either isotropic or exactly `n_features`
    long, every offset is in range, and the expression is well formed.

    Refused on the HOST, before any upload, naming the node. This is
    sklearn's `_check_length_scale` (`kernels.py:34-48`) plus the bound
    checking a flat representation needs and a tree does not.
    """
    _ = gp_kernel_stack_depth(spec)
    if n_features < 1:
        raise Error(
            "gp_validate_kernel: n_features must be positive, got "
            + String(n_features)
        )
    for t in range(len(spec.kinds)):
        var k = Int(spec.kinds[t])
        if k != GP_K_RBF and k != GP_K_MATERN:
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


def gp_kernel_diag(spec: GPKernelSpec) raises -> Float32:
    """`kernel.diag(X)`, sklearn `_gpr.py:480`. **DEVIATION 1770.**

    For every kernel this lane implements, `k(x, x)` is a CONSTANT that does
    not depend on the coordinates at all -- RBF and Matern are 1 (their
    distance to themselves is exactly zero), ConstantKernel is its value,
    WhiteKernel is its noise level -- so `diag` is one scalar rather than an
    `n`-vector, and it is computed once on the host and recorded as the card
    stage `gp.kss`. sklearn returns a full vector because a kernel it
    supports and this lane does not (`DotProduct`) has a coordinate-dependent
    diagonal; when that is ported this function returns a vector and the
    stage becomes an array. Named here so the change is a change and not a
    surprise.

    The fold is sklearn's own: `Sum.diag = k1.diag + k2.diag`
    (`kernels.py:889`), `Product.diag = k1.diag * k2.diag`
    (`kernels.py:989`), evaluated in the same postfix order the matrices
    are, through the same `ftz` and `identical_mul` pins.
    """
    _ = gp_kernel_stack_depth(spec)
    var stack = List[Float32]()
    for t in range(len(spec.kinds)):
        var k = Int(spec.kinds[t])
        if k == GP_K_CONST:
            stack.append(ftz(spec.params[t]))
        elif k == GP_K_WHITE:
            stack.append(ftz(spec.params[t]))
        elif k == GP_K_RBF or k == GP_K_MATERN:
            stack.append(Float32(1.0))
        else:
            var b = stack[len(stack) - 1]
            var a = stack[len(stack) - 2]
            _ = stack.pop()
            _ = stack.pop()
            if k == GP_K_PROD:
                stack.append(ftz(identical_mul(a, b)))
            else:
                stack.append(ftz(a + b))
    return stack[0]


# ===========================================================================
# THE DEVICE KERNELS
# ===========================================================================


def gp_scaled_sqdist(
    x: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    ls: MutPointer[Float32, MutAnyOrigin],
    i: Int,
    j: Int,
    d: Int,
    ls_len: Int,
) -> Float32:
    """`sum_f ((x[i,f] - y[j,f]) / l_f)^2`, `f` ASCENDING, in one thread.

    **THE ARD LENGTH SCALE IS APPLIED BY SCALING THE COORDINATES, not by
    changing the distance. DEVIATION 1753.** That is sklearn's own spelling
    -- `cdist(X / length_scale, Y / length_scale, ...)`, `kernels.py:1568`
    and `:1720` -- and it is bit-equal to materializing `X / length_scale`
    into a buffer first, because each quotient is rounded to float32 before
    the subtraction either way. Fusing it saves two `m x d` scratch buffers
    and two launches per leaf and moves no bit; the equality is asserted,
    not assumed, by `check_kernels_vs_oracle`, whose oracle DOES materialize
    the scaled copy.

    `ls_len` is 1 for an isotropic kernel and `d` for ARD, and
    `gp_validate_kernel` has already refused everything else, so the index
    below has exactly two cases and no clamp.

    **The per-feature step is `l2_unexp_core`, IMPORTED rather than
    re-spelled. DEVIATION 1754.** It lives in
    `kde/impl/distance/distance_ops.mojo`, ported from cuVS
    `distance_ops/l2_unexp.cuh:62-63`, and is `diff = ftz(x - y);
    ftz(fma(diff, diff, acc))`. Writing those two lines again here would be
    a second spelling of one arithmetic, which is exactly what
    `gemm_identical.mojo::contract_partition` records having gone wrong once
    already. What is NOT reusable is the surrounding kernel: that file's
    `pairwise_unexpanded_kernel` applies cuVS's `L2SqrtUnexpanded` epilog
    (`identical_sqrt`) unconditionally, and an RBF needs the SQUARED
    distance with no root. The right merge is a `DIST_L2_SQ_UNEXPANDED`
    metric value in that file, which is the KDE lane's to add, so it is
    named in `gaussian_process/README.md` rather than done here.
    """
    var acc = Float32(0.0)
    for f in range(d):
        var li = f
        if ls_len == 1:
            li = 0
        var lv = ftz(ls.unsafe_load(li))
        var xv = ftz(identical_div(ftz(x.unsafe_load(i * d + f)), lv))
        var yv = ftz(identical_div(ftz(y.unsafe_load(j * d + f)), lv))
        acc = l2_unexp_core(acc, xv, yv)
    return ftz(acc)


def gp_const_kernel(
    out_k: MutPointer[Float32, MutAnyOrigin],
    mn_in: Int32,
    value: Float32,
):
    """`ConstantKernel.__call__`, sklearn `kernels.py:1278-1282`. Every cell
    is the same value, including the cross-covariance."""
    var mn = Int(mn_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < mn:
        out_k.unsafe_store(t, ftz(value))


def gp_white_kernel(
    out_k: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    noise: Float32,
    is_self_in: Int32,
):
    """`WhiteKernel.__call__`, sklearn `kernels.py:1406-1419`.

    `is_self` is sklearn's `Y is None`: a STRUCTURAL flag decided by the
    caller, never a coordinate comparison (DEVIATION 1762). Two identical
    rows of `X` get the noise on their diagonal cells and zero on their
    off-diagonal cell, which is what makes a duplicated input a rank
    deficiency the ridge must fix.
    """
    var m = Int(m_in)
    var n = Int(n_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= m * n:
        return
    var i = t // n
    var j = t - i * n
    if Int(is_self_in) != 0 and i == j:
        out_k.unsafe_store(t, ftz(noise))
    else:
        out_k.unsafe_store(t, Float32(0.0))


def gp_rbf_kernel(
    out_k: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    ls: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    d_in: Int32,
    ls_len_in: Int32,
):
    """`RBF.__call__`, sklearn `kernels.py:1568-1570`.
    `K = exp(-0.5 * sqeuclidean(X / l, Y / l))`.

    `-0.5` is a power of two so the multiply is exact at every input, and it
    is spelled `identical_mul` anyway (DEVIATION 826): no bit depends on the
    spelling, and no codegen may contract it into a neighbouring add.

    THE DIAGONAL. When `x` and `y` are the same buffer and `i == j`, every
    `xv - yv` is exactly `+0.0`, so the chain is `fma(0, 0, +0.0)` repeated
    and `d2` is exactly `+0.0`; `identical_mul(-0.5, +0.0)` is `-0.0`, and
    `portable_expf(-0.0)` is exactly `1.0`. So `K_ii = 1.0` by arithmetic,
    where sklearn assigns it (`np.fill_diagonal(K, 1)`, `kernels.py:1564`).
    `check_kernels_vs_oracle` asserts it by bits.
    """
    var m = Int(m_in)
    var n = Int(n_in)
    var d = Int(d_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= m * n:
        return
    var i = t // n
    var j = t - i * n
    var d2 = gp_scaled_sqdist(x, y, ls, i, j, d, Int(ls_len_in))
    var e = ftz(identical_mul(Float32(-0.5), d2))
    out_k.unsafe_store(t, ftz(identical_exp(e)))


def gp_matern_kernel(
    out_k: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    ls: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    d_in: Int32,
    ls_len_in: Int32,
    nu_sel_in: Int32,
    sqrt3: Float32,
    sqrt5: Float32,
):
    """`Matern.__call__`, the three closed forms, sklearn
    `kernels.py:1720-1730`, transcribed in their order.

        d      = euclidean(X / l, Y / l)          = sqrt(sqeuclidean)
        nu=0.5 K = exp(-d)
        nu=1.5 K = d * sqrt(3);  K = (1 + K) * exp(-K)
        nu=2.5 K = d * sqrt(5);  K = (1 + K + K*K / 3) * exp(-K)

    `nu_sel` is `0`, `1` or `2`, resolved on the HOST from the float `nu`
    (`gp_matern_nu_selector`), so no float comparison decides a branch on
    the device and the general case cannot be reached from here at all.
    `sqrt3` and `sqrt5` arrive as kernel ARGUMENTS from `gp_sqrt3()` /
    `gp_sqrt5()` rather than as device literals, so there is exactly one
    definition of each constant in the lane and the oracle reads the same
    one.

    `K**2 / 3.0` is transcribed as a DIVIDE (`identical_div`), not as a
    multiply by a stored one-third: sklearn divides, and `x * (1/3)` is two
    roundings where a divide is one. Same argument as DEVIATION 1643 in the
    triangular solves.
    """
    var m = Int(m_in)
    var n = Int(n_in)
    var d = Int(d_in)
    var nu_sel = Int(nu_sel_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= m * n:
        return
    var i = t // n
    var j = t - i * n
    var d2 = gp_scaled_sqdist(x, y, ls, i, j, d, Int(ls_len_in))
    var dist = ftz(identical_sqrt(d2))
    if nu_sel == 0:
        out_k.unsafe_store(t, ftz(identical_exp(-dist)))
        return
    if nu_sel == 1:
        var s = ftz(identical_mul(dist, sqrt3))
        var pre = ftz(Float32(1.0) + s)
        out_k.unsafe_store(t, ftz(identical_mul(pre, ftz(identical_exp(-s)))))
        return
    var s5 = ftz(identical_mul(dist, sqrt5))
    var ss = ftz(identical_mul(s5, s5))
    var third = ftz(identical_div(ss, Float32(3.0)))
    var pre5 = ftz(ftz(Float32(1.0) + s5) + third)
    out_k.unsafe_store(t, ftz(identical_mul(pre5, ftz(identical_exp(-s5)))))


def gp_combine_kernel(
    a: MutPointer[Float32, MutAnyOrigin],
    b: MutPointer[Float32, MutAnyOrigin],
    mn_in: Int32,
    is_prod_in: Int32,
):
    """`Sum` (`kernels.py:871`) and `Product` (`kernels.py:971`), elementwise
    into the LEFT operand's slot. `a` is the deeper stack entry, so the
    operand order is the source order and not the stack order."""
    var mn = Int(mn_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= mn:
        return
    var av = ftz(a.unsafe_load(t))
    var bv = ftz(b.unsafe_load(t))
    if Int(is_prod_in) != 0:
        a.unsafe_store(t, ftz(identical_mul(av, bv)))
    else:
        a.unsafe_store(t, ftz(av + bv))


def gp_copy_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin],
    mn_in: Int32,
):
    """A bit-for-bit copy, no arithmetic and no `ftz`. The postfix
    evaluation finishes in a stack slot and the caller wants it in `out`."""
    var mn = Int(mn_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < mn:
        dst.unsafe_store(t, src.unsafe_load(t))


def gp_variance_kernel(
    var_out: MutPointer[Float32, MutAnyOrigin],
    std_out: MutPointer[Float32, MutAnyOrigin],
    clamped: MutPointer[Int32, MutAnyOrigin],
    v: MutPointer[Float32, MutAnyOrigin],
    n_train_in: Int32,
    n_star_in: Int32,
    kss: Float32,
):
    """`y_var = kernel.diag(X) - einsum("ij,ji->i", V.T, V)`, sklearn
    `_gpr.py:480-481`, then the clamp at `_gpr.py:485-491`.

    ONE THREAD PER TEST POINT, walking the training axis ASCENDING in its
    own registers. **NOT A GEMM, and DEVIATION 1759 is the argument**: only
    the DIAGONAL of `V^T V` is wanted, so a matrix product would compute
    `n_star x n_star` cells and discard all but `n_star` of them, and would
    put a fold tree where a serial chain does. sklearn makes the same choice
    for the same reason and says so in a comment (`_gpr.py:109-110`, "Use
    einsum to avoid explicitly forming the large matrix V^T @ V just to
    extract its diagonal afterward").

    No float crosses a thread boundary, so there is no fold shape to pin and
    `check_launch_invariance`'s one-point-alone-versus-in-a-batch arm is a
    property of the shape rather than an observation.

    **THE CLAMP, AND WHY IT IS COUNTED. DEVIATION 1760.** In exact
    arithmetic `k** - v^T v` is non-negative, because it is a Schur
    complement of a positive-definite matrix. In float32 it is not: at a
    test point that coincides with a training point the true value is
    exactly zero and the computed one is a few ulps either side of it. So
    the clamp FIRES in ordinary use, and a Gaussian process that clamps
    without saying so is a Gaussian process that lies quietly.

    The record is a PER-TEST-POINT FLAG, hashed as the card stage
    `gp.clamped` and summed on the host -- never a device counter. A count
    is a TOTAL and says nothing about placement (PORTING_RULES rule 7: "a
    check whose expected value is the same in every cell verifies the total
    and nothing about placement"); the flag vector is placement, and it is
    also the thing a cross-vendor diff can align. There is no atomic of any
    kind here, integer or float.

    The flag is set from a BIT COMPARISON of the stored value against the
    computed one, not from the branch: an exactly `+0.0` variance takes the
    clamp branch and changes nothing, and calling that a clamp would
    overcount. A `-0.0` variance changes bits and IS counted, which is the
    row-39 case sklearn misses (`y_var < 0` is False for `-0.0`, so sklearn
    returns a negative zero and every tolerance comparison downstream is
    blind to it).

    The comparison is spelled `not (v > 0.0)` and never as a `max`:
    IDENTITY_PATHS row 39 measured `max(+0.0, -0.0)` returning `+0` on
    NVIDIA and AMD and `-0` on Apple, so a `max` here would be a
    vendor-shaped sign in a certified stage.
    """
    var n_train = Int(n_train_in)
    var n_star = Int(n_star_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= n_star:
        return
    var acc = Float32(0.0)
    for i in range(n_train):
        var vv = ftz(v.unsafe_load(i * n_star + t))
        acc = ftz(identical_mul_add(vv, vv, acc))
    var raw = ftz(ftz(kss) - acc)
    var outv = raw
    if not (raw > Float32(0.0)):
        outv = Float32(0.0)
    var moved = bitcast[DType.uint32](outv) != bitcast[DType.uint32](raw)
    var_out.unsafe_store(t, outv)
    clamped.unsafe_store(t, Int32(1) if moved else Int32(0))
    # sklearn returns the STANDARD DEVIATION from `return_std`
    # (`_gpr.py:500`, `np.sqrt(y_var)`), so the root is taken here on the
    # already-clamped value and can never see a negative operand.
    std_out.unsafe_store(t, ftz(identical_sqrt(outv)))


# ===========================================================================
# THE DRIVER
# ===========================================================================


def gp_matern_nu_selector(nu: Float32) raises -> Int:
    """`0`, `1` or `2` for `nu` in `{0.5, 1.5, 2.5}`. Resolved on the HOST so
    no float comparison decides a device branch."""
    var b = bitcast[DType.uint32](nu)
    if b == GP_NU_0_5_BITS:
        return 0
    if b == GP_NU_1_5_BITS:
        return 1
    if b == GP_NU_2_5_BITS:
        return 2
    raise Error(
        "gp_matern_nu_selector: nu bits 0x"
        + gp_hex32_bits(nu)
        + " is not one of the three closed forms. DEVIATION 1765"
    )


def gp_kernel_stack_floats(m: Int, n: Int) -> Int:
    """Floats `gp_kernel_matrix` needs in `stack` at this shape:
    `GP_MAX_STACK` slots of `m * n`. Never less than 1, so the buffer is
    always constructible."""
    var need = GP_MAX_STACK * m * n
    if need < 1:
        return 1
    return need


def gp_kernel_matrix(
    ctx: DeviceContext,
    mut out: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    mut y: DeviceBuffer[DType.float32],
    mut dls: DeviceBuffer[DType.float32],
    mut stack: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    d: Int,
    spec: GPKernelSpec,
    is_self: Bool,
    mut trace: IdentityTrace,
    tag: StringSlice,
    elem_tpb: Int = GP_ELEM_TPB,
    sabotage: Int = GP_SAB_NONE,
) raises:
    """`out[m x n] = k(x[m x d], y[n x d])`, row-major, on the device.

    `is_self` is sklearn's `Y is None` and reaches exactly one kernel
    (`gp_white_kernel`); it is a caller's structural statement about whether
    `x` and `y` are the same rows, never a coordinate test (DEVIATION 1762).

    `stack` must hold at least `gp_kernel_stack_floats(m, n)` floats. The
    postfix list is walked ONCE, in order: a leaf launches into stack slot
    `sp` and pushes, an operator launches into slot `sp - 2` reading slot
    `sp - 1` and pops. `gp_kernel_stack_depth` has already refused any
    expression that would exceed `GP_MAX_STACK`, so no launch below can
    address past the buffer.

    ASYNCHRONOUS except for the trace record, which drains by construction
    (`core/identity_trace.mojo` rule 4). The caller keeps every buffer alive
    past its own `ctx.synchronize()`.

    `tag` is the single card stage this evaluation records, over the FINAL
    matrix. Per-node stages are deliberately NOT recorded: the tag-uniqueness
    invariant would make the stage list a function of the kernel expression,
    so two runs of one program with two kernels would produce cards a differ
    cannot align. `gaussian_process/README.md` carries that under WHAT IS
    OWED, with the fix (a per-node tag derived from the node index, the way
    `chol_panel_tag` derives one from the panel index).
    """
    if m <= 0 or n <= 0 or d <= 0:
        raise Error(
            "gp_kernel_matrix: m, n and d must all be positive, got "
            + String(m)
            + ", "
            + String(n)
            + ", "
            + String(d)
        )
    if elem_tpb <= 0:
        raise Error("gp_kernel_matrix: elem_tpb must be positive")
    gp_validate_kernel(spec, d)
    if len(out) < m * n:
        raise Error(
            "gp_kernel_matrix: the output buffer holds "
            + String(len(out))
            + " floats, an "
            + String(m)
            + " x "
            + String(n)
            + " matrix needs "
            + String(m * n)
        )
    if len(stack) < gp_kernel_stack_floats(m, n):
        raise Error(
            "gp_kernel_matrix: the operand stack holds "
            + String(len(stack))
            + " floats and this shape needs "
            + String(gp_kernel_stack_floats(m, n))
            + ". Sizing it for one expression and evaluating another is an"
            " out-of-bounds write a small shape will not show you; use"
            " gp_kernel_stack_floats"
        )

    var cells = m * n
    var grid = (cells + elem_tpb - 1) // elem_tpb
    var sab_kernels = gp_sabotage_touches_kernel_matrix(sabotage)
    var sqrt3 = gp_sqrt3()
    var sqrt5 = gp_sqrt5()
    var self_flag = Int32(1) if is_self else Int32(0)

    var sp = 0
    for t in range(len(spec.kinds)):
        var kind = Int(spec.kinds[t])
        if kind == GP_K_SUM or kind == GP_K_PROD:
            var lhs = stack.create_sub_buffer[DType.float32](
                (sp - 2) * cells, cells
            )
            var rhs = stack.create_sub_buffer[DType.float32](
                (sp - 1) * cells, cells
            )
            var is_prod = Int32(1) if kind == GP_K_PROD else Int32(0)
            if sab_kernels:
                ctx.enqueue_function[sabotage_combine_kernel](
                    lhs.unsafe_ptr(),
                    rhs.unsafe_ptr(),
                    Int32(cells),
                    is_prod,
                    Int32(sabotage),
                    grid_dim=(grid, 1, 1),
                    block_dim=(elem_tpb, 1, 1),
                )
            else:
                ctx.enqueue_function[gp_combine_kernel](
                    lhs.unsafe_ptr(),
                    rhs.unsafe_ptr(),
                    Int32(cells),
                    is_prod,
                    grid_dim=(grid, 1, 1),
                    block_dim=(elem_tpb, 1, 1),
                )
            _ = lhs^
            _ = rhs^
            sp -= 1
            continue

        var slot = stack.create_sub_buffer[DType.float32](sp * cells, cells)
        if kind == GP_K_CONST:
            ctx.enqueue_function[gp_const_kernel](
                slot.unsafe_ptr(),
                Int32(cells),
                spec.params[t],
                grid_dim=(grid, 1, 1),
                block_dim=(elem_tpb, 1, 1),
            )
        elif kind == GP_K_WHITE:
            ctx.enqueue_function[gp_white_kernel](
                slot.unsafe_ptr(),
                Int32(m),
                Int32(n),
                spec.params[t],
                self_flag,
                grid_dim=(grid, 1, 1),
                block_dim=(elem_tpb, 1, 1),
            )
        else:
            var lsview = dls.create_sub_buffer[DType.float32](
                Int(spec.ls_off[t]), Int(spec.ls_len[t])
            )
            if kind == GP_K_RBF:
                if sab_kernels:
                    ctx.enqueue_function[sabotage_rbf_kernel](
                        slot.unsafe_ptr(),
                        x.unsafe_ptr(),
                        y.unsafe_ptr(),
                        lsview.unsafe_ptr(),
                        Int32(m),
                        Int32(n),
                        Int32(d),
                        spec.ls_len[t],
                        Int32(sabotage),
                        grid_dim=(grid, 1, 1),
                        block_dim=(elem_tpb, 1, 1),
                    )
                else:
                    ctx.enqueue_function[gp_rbf_kernel](
                        slot.unsafe_ptr(),
                        x.unsafe_ptr(),
                        y.unsafe_ptr(),
                        lsview.unsafe_ptr(),
                        Int32(m),
                        Int32(n),
                        Int32(d),
                        spec.ls_len[t],
                        grid_dim=(grid, 1, 1),
                        block_dim=(elem_tpb, 1, 1),
                    )
            else:
                var nu_sel = gp_matern_nu_selector(spec.params[t])
                if sab_kernels:
                    ctx.enqueue_function[sabotage_matern_kernel](
                        slot.unsafe_ptr(),
                        x.unsafe_ptr(),
                        y.unsafe_ptr(),
                        lsview.unsafe_ptr(),
                        Int32(m),
                        Int32(n),
                        Int32(d),
                        spec.ls_len[t],
                        Int32(nu_sel),
                        sqrt3,
                        sqrt5,
                        Int32(sabotage),
                        grid_dim=(grid, 1, 1),
                        block_dim=(elem_tpb, 1, 1),
                    )
                else:
                    ctx.enqueue_function[gp_matern_kernel](
                        slot.unsafe_ptr(),
                        x.unsafe_ptr(),
                        y.unsafe_ptr(),
                        lsview.unsafe_ptr(),
                        Int32(m),
                        Int32(n),
                        Int32(d),
                        spec.ls_len[t],
                        Int32(nu_sel),
                        sqrt3,
                        sqrt5,
                        grid_dim=(grid, 1, 1),
                        block_dim=(elem_tpb, 1, 1),
                    )
            _ = lsview^
        _ = slot^
        sp += 1

    var root = stack.create_sub_buffer[DType.float32](0, cells)
    ctx.enqueue_function[gp_copy_kernel](
        out.unsafe_ptr(),
        root.unsafe_ptr(),
        Int32(cells),
        grid_dim=(grid, 1, 1),
        block_dim=(elem_tpb, 1, 1),
    )
    _ = root^
    trace.record_device(ctx, tag, out, cells)


def gp_predictive_variance(
    ctx: DeviceContext,
    mut var_out: DeviceBuffer[DType.float32],
    mut std_out: DeviceBuffer[DType.float32],
    mut clamped: DeviceBuffer[DType.int32],
    mut v: DeviceBuffer[DType.float32],
    n_train: Int,
    n_star: Int,
    kss: Float32,
    mut trace: IdentityTrace,
    elem_tpb: Int = GP_ELEM_TPB,
    sabotage: Int = GP_SAB_NONE,
) raises:
    """`gp_variance_kernel`, launched. Records `gp.var`, `gp.clamped` and
    `gp.std`, in that order, so a divergence lands on the stage that moved
    rather than on the answer.

    `v` is `n_train x n_star` row-major: the SAME buffer and the SAME
    orientation `trsm_lower` wrote its solution into, so nothing is
    transposed between the solve and the fold.
    """
    if n_train <= 0 or n_star <= 0:
        raise Error(
            "gp_predictive_variance: n_train and n_star must be positive,"
            " got "
            + String(n_train)
            + ", "
            + String(n_star)
        )
    if elem_tpb <= 0:
        raise Error("gp_predictive_variance: elem_tpb must be positive")
    if len(v) < n_train * n_star:
        raise Error(
            "gp_predictive_variance: the solve buffer holds "
            + String(len(v))
            + " floats, "
            + String(n_train)
            + " x "
            + String(n_star)
            + " needs "
            + String(n_train * n_star)
        )
    var grid = (n_star + elem_tpb - 1) // elem_tpb
    if gp_sabotage_touches_variance(sabotage):
        ctx.enqueue_function[sabotage_variance_kernel](
            var_out.unsafe_ptr(),
            std_out.unsafe_ptr(),
            clamped.unsafe_ptr(),
            v.unsafe_ptr(),
            Int32(n_train),
            Int32(n_star),
            kss,
            Int32(sabotage),
            grid_dim=(grid, 1, 1),
            block_dim=(elem_tpb, 1, 1),
        )
    else:
        ctx.enqueue_function[gp_variance_kernel](
            var_out.unsafe_ptr(),
            std_out.unsafe_ptr(),
            clamped.unsafe_ptr(),
            v.unsafe_ptr(),
            Int32(n_train),
            Int32(n_star),
            kss,
            grid_dim=(grid, 1, 1),
            block_dim=(elem_tpb, 1, 1),
        )
    trace.record_device(ctx, "gp.var", var_out, n_star)
    trace.record_device(ctx, "gp.clamped", clamped, n_star)
    trace.record_device(ctx, "gp.std", std_out, n_star)
