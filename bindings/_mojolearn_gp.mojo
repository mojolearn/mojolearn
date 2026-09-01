# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPython boundary for the exact dense Gaussian process regression lane.

A THIRTEENTH extension module, and a separate one on purpose. The header of
`bindings/_mojolearn_estimators.mojo` states the reason and it is the same
reason here: an independently changing binding must not become a merge
point. `gaussian_process/` is ORIGINAL WORK (cuML, cuVS and RAFT implement
no Gaussian process at the pinned commits; `gaussian_process/
DERIVATION_MAP.tsv` carries the grep) over the Cholesky lane's factor/solve
and the identical GEMM, and it shares no binding-level code with any
sibling. All thirteen binaries land in one wheel.

THIS FILE CLOSES THE "OWED" HALF OF COMMIT 22a5b550 -- UNDER A RENEGOTIATED
ABI, AND THE RENEGOTIATION IS A MEASURED RESULT, NOT A PREFERENCE. That
commit landed `python/mojolearn/_gp_impl.py` with `_mojolearn_gp` already
registered in `_backend.py`'s `_MODULES` and `_build_script` (both places,
DEVIATION 869) and a binding contract written at each call site in which
every buffer address was its own positional argument: TEN arguments for
`gpr_fit`, THIRTEEN for `gpr_predict`. This file's first version (8c449b70)
honored that spelling verbatim and flagged the risk: `bindings/
_mojolearn.mojo`'s header records that `PythonModuleBuilder.def_function`
infers its signature from arity "and stops being able to above roughly nine
arguments; mojotrees' widest binding takes nine and that is not a
coincidence". THE FIRST BUILD CONFIRMED IT: `bash bindings/build_gp.sh`
failed AT def_function ELABORATION on 2026-09-01 (log
/tmp/build_gp_fast.log, the stdlib error pointing at def_function's
signature), before any artifact existed. The orchestrator renegotiated BOTH
sides the same day.

THE FOLD IS THE TREE'S OWN PRECEDENT, NOT A NEW MECHANISM. When
`knn_search` needed ten arguments and `knn_classify` fourteen-plus, they
did not grow arity: every scalar folded into ONE length-checked Python
list whose order is written out in the same words on both sides
(`_mojolearn.mojo`'s SCALARS ARRIVE AS ONE LIST banner). Here the
overflowing axis is the ADDRESSES, so they get the same treatment: each
entry point below takes exactly TWO arguments,

    gpr_fit(addrs, params)         len(addrs) == 9,  len(params) == 5
    gpr_predict(addrs, params)     len(addrs) == 12, len(params) == 7

where `addrs` is a plain Python list of the NumPy buffer addresses, in an
exact order written in each docstring and mirrored at the `_gp_impl.py`
call site, and `params` is the scalar list 22a5b550's contract already
spelled, unchanged to the word. A Python LIST, deliberately not a packed
int64 array: `Int(py=addrs[i])` is the same read every sibling does on
`params`, an int needs no dtype contract and no bit-cast, and the length
check is what stands between a swapped pair and a believable address --
where a swapped pair of ADDRESSES is a crash or a wrong answer, which is
why the order comment exists twice.

Arrays cross as borrowed NumPy addresses; all device buffers and contexts
live for one call and no pointer is retained. The Python wrapper owns the
arrays and keeps every one of them alive for the duration of the call BY
BINDING THEM TO LOCALS -- an address inside a list keeps nothing alive
(`python/mojolearn/_arrays.py` is where that contract is written down).

THE KERNEL SPEC IS REBUILT THROUGH THE CONSTRUCTORS, NOT ASSEMBLED BY HAND.
`_gp_impl.py::_kernel_arrays` sends four flat postfix arrays (kinds,
params, ls_len and the concatenated length-scale table; DEVIATION 1756) and
deliberately does NOT send the offsets: `_rebuild_kernel_spec` below walks
the postfix list with a stack over `gp_kernel_const` / `gp_kernel_white` /
`gp_kernel_rbf` / `gp_kernel_matern` and `gp_kernel_sum` / `gp_kernel_prod`,
which recompute the offsets in `_combine`. That is what keeps every
constructor refusal -- the Matern closed-forms pin (DEVIATION 1765), the
non-positive length scale, the negative constant, the node-count cap --
reachable from Python, which is the reason the Python side judges none of
those values (its `_as_length_scale` docstring says so in the same words).

WHAT IS REFUSED, AND WHERE. Nothing is refused in this file except a null
address, an `addrs` or `params` list of the wrong length and a malformed
postfix expression (a stack underflow, a leftover operand, an unknown kind code, a
length-scale table the two sides disagree about -- each of which means the
two sides of THIS boundary disagree and no lane refusal exists for it).
Every model refusal lives one or two layers down and is raised there by
name: alpha (NaN, negative, +inf, and the identical tier's two-value pin,
DEVIATIONS 1768/1637), non-finite X or y with the flat index (DEVIATION
1768), the optimizer/n_restarts/normalize_y knobs (DEVIATIONS 1761/1764),
and `gpr_predict_host`'s refusal to predict from a FAILED fit (DEVIATION
1634) -- which is why `gpr_predict` takes `info` in its params list and
passes it through instead of judging it.

THE GIL is released around every device call, and nothing inside a
`GILReleased` block touches a `PythonObject`.
"""

from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.vendor import COMPILED_VENDOR

from gaussian_process.checks.kernels import (
    GP_K_CONST,
    GP_K_MATERN,
    GP_K_PROD,
    GP_K_RBF,
    GP_K_SUM,
    GP_K_WHITE,
    GPKernelSpec,
    gp_kernel_const,
    gp_kernel_matern,
    gp_kernel_prod,
    gp_kernel_rbf,
    gp_kernel_sum,
    gp_kernel_white,
)
from gaussian_process.estimator import (
    GPRegressor,
    gpr_fit_host,
    gpr_predict_host,
)


def _f32_ptr(addr: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null float32 buffer address")
    return MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=addr)


def _i32_ptr(addr: Int) raises -> MutPointer[Int32, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null int32 buffer address")
    return MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=addr)


def _f64_ptr(addr: Int) raises -> MutPointer[Float64, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null float64 buffer address")
    return MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=addr)


def gp_numeric_mode_binding() raises -> PythonObject:
    """THE BUILD'S TIER, as the `NUMERIC_*` code itself: 0 FAST, 1
    IDENTICAL, 2 DETERMINISTIC. The same shape as `gbdt_numeric_mode`, and
    for the same reason: the wrapper reads it once (`_gp_impl.py::
    GaussianProcessRegressor._extension`) and refuses to run if the binary
    it loaded disagrees with the mode the package asked for. A wrong-arm
    measurement that is correctly labelled by accident is the failure this
    prevents, and a boolean could not do that job once a third tier
    existed, because DETERMINISTIC answered 0 and read back as "fast"."""
    return PythonObject(GLOBAL_NUMERIC_MODE)


def gp_vendor_binding() raises -> PythonObject:
    """THE ACCELERATOR API THIS BINARY WAS COMPILED FOR: 'metal', 'cuda',
    'hip' or 'none'. A compile-time constant folded in from
    `checks/vendor.mojo`, the same shape as the tier read-back: the answer
    comes from the binary that actually loaded, never from the directory it
    sat in and never from the environment.
    `python/mojolearn/_backend.py` refuses at import when this disagrees
    with the vendor directory the set was loaded from."""
    return PythonObject(String(COMPILED_VENDOR))


def _rebuild_kernel_spec(
    kinds_addr: Int,
    kparams_addr: Int,
    ls_len_addr: Int,
    ls_addr: Int,
    n_nodes: Int,
    n_ls: Int,
    what: String,
) raises -> GPKernelSpec:
    """The postfix node list back into a `GPKernelSpec`, THROUGH THE
    CONSTRUCTORS (DEVIATION 1756).

    A stack machine over the four flat arrays `_gp_impl.py::_kernel_arrays`
    sends. Leaf node `t` consumes `ls_len[t]` floats from the length-scale
    table at a running offset; `gp_kernel_sum` / `gp_kernel_prod` recompute
    every offset in `_combine`, which is why the offsets are the one array
    NOT sent. Every value goes to the constructor UNJUDGED so its refusal
    fires with its own name: `gp_kernel_matern` refuses a `nu` outside the
    three closed forms BY BITS (DEVIATION 1765) and `kparams` crosses as
    float32, so the bits that arrive are the bits `np.float32(nu)` holds.

    `Int(Int32)` below SIGN-EXTENDS ([[mojo-int-widening-sign-extends]]),
    and here that is the wanted branch of the trap: a negative kind or
    ls_len must ARRIVE negative so the explicit range refusals below can
    name it, rather than be masked into a plausible small code.
    """
    if n_nodes < 1:
        raise Error(
            what
            + ": the kernel spec must have at least one postfix node, got "
            + String(n_nodes)
        )
    if n_ls < 0:
        raise Error(
            what + ": n_ls cannot be negative, got " + String(n_ls)
        )
    var kp = _i32_ptr(kinds_addr)
    var pp = _f32_ptr(kparams_addr)
    var lnp = _i32_ptr(ls_len_addr)
    var tp = _f32_ptr(ls_addr)
    var stack = List[GPKernelSpec]()
    var off = 0
    for t in range(n_nodes):
        var k = Int(kp.unsafe_load(t))
        var param = pp.unsafe_load(t)
        if k == GP_K_CONST:
            stack.append(gp_kernel_const(param))
        elif k == GP_K_WHITE:
            stack.append(gp_kernel_white(param))
        elif k == GP_K_RBF or k == GP_K_MATERN:
            var ln = Int(lnp.unsafe_load(t))
            if ln < 1:
                raise Error(
                    what
                    + ": node "
                    + String(t)
                    + " is an RBF or Matern leaf with ls_len "
                    + String(ln)
                    + "; a leaf consumes at least one length scale, so the"
                    " two sides of this boundary disagree about the spec"
                )
            if off + ln > n_ls:
                raise Error(
                    what
                    + ": node "
                    + String(t)
                    + " consumes length scales ["
                    + String(off)
                    + ", "
                    + String(off + ln)
                    + ") of a table holding "
                    + String(n_ls)
                    + "; the two sides of this boundary disagree about the"
                    " table"
                )
            var leaf_ls = List[Float32]()
            for i in range(ln):
                leaf_ls.append(tp.unsafe_load(off + i))
            off += ln
            if k == GP_K_RBF:
                stack.append(gp_kernel_rbf(leaf_ls))
            else:
                stack.append(gp_kernel_matern(leaf_ls, param))
        elif k == GP_K_SUM or k == GP_K_PROD:
            if len(stack) < 2:
                raise Error(
                    what
                    + ": node "
                    + String(t)
                    + " combines a stack of "
                    + String(len(stack))
                    + " operands; the postfix expression is malformed"
                )
            var b = stack.pop()
            var a = stack.pop()
            if k == GP_K_SUM:
                stack.append(gp_kernel_sum(a, b))
            else:
                stack.append(gp_kernel_prod(a, b))
        else:
            raise Error(
                what
                + ": node "
                + String(t)
                + " has unknown kind "
                + String(k)
                + ". The GP_K_* codes are 0 CONST, 1 WHITE, 2 RBF,"
                " 3 MATERN, 4 SUM, 5 PROD, mirrored in _gp_impl.py; a"
                " silent renumbering on either side is a WRONG KERNEL,"
                " which is why this refuses by value instead of clamping"
            )
    if len(stack) != 1:
        raise Error(
            what
            + ": the postfix expression leaves "
            + String(len(stack))
            + " operands on the stack; a well-formed kernel leaves exactly"
            " one"
        )
    if off != n_ls:
        raise Error(
            what
            + ": the leaves consumed "
            + String(off)
            + " length scales of the "
            + String(n_ls)
            + " sent; the two sides of this boundary disagree about the"
            " table"
        )
    return stack.pop()


def _gpr_fit_run(
    x: List[Float32],
    y: List[Float32],
    spec: GPKernelSpec,
    n_train: Int,
    n_features: Int,
    alpha: Float32,
    lp: MutPointer[Float32, MutUntrackedOrigin],
    dp: MutPointer[Float32, MutUntrackedOrigin],
    sp: MutPointer[Float64, MutUntrackedOrigin],
) raises -> Int:
    """The GIL-free half of `gpr_fit_binding`: everything after the
    `PythonObject`s have been read and before one is built. A separate
    function because `GPRegressor` has no default constructor to
    pre-declare across a `with` block, and an `Int` does."""
    var model = gpr_fit_host(x, n_train, n_features, y, spec, alpha)
    for i in range(n_train * n_train):
        lp.unsafe_store(i, model.l[i])
    for i in range(n_train):
        dp.unsafe_store(i, model.dual_coef[i])
    # info, nb, logdet, ydotalpha, lml -- in that order, the same five
    # words as `_gp_impl.py::fit`'s `scalars` comment. Each float32 widens
    # to float64 exactly.
    sp.unsafe_store(0, Float64(model.info))
    sp.unsafe_store(1, Float64(model.nb))
    sp.unsafe_store(2, Float64(model.logdet))
    sp.unsafe_store(3, Float64(model.ydotalpha))
    sp.unsafe_store(4, Float64(model.lml))
    return model.info


def gpr_fit_binding(
    addrs: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`GaussianProcessRegressor(kernel, alpha).fit(X, y)`
    (gaussian_process/, DEVIATIONS 1750-1771). Returns LAPACK's `info`,
    which is a RESULT and not an exception (DEVIATION 1634): 0 means the
    factor is complete, `k > 0` that the leading minor of order `k` was not
    positive definite.

    `addrs` is the NINE buffer addresses, in this exact order (mirrored in
    `python/mojolearn/_gp_impl.py`; the module header says why they are a
    list and not nine arguments):

        0  x               n_train * n_features float32, row-major, read
        1  y               n_train float32, read
        2  kinds           n_nodes int32, read
        3  kparams         n_nodes float32, read
        4  ls_len          n_nodes int32, read
        5  ls              max(n_ls, 1) float32, read
        6  l_out           n_train * n_train float32, WRITTEN
        7  dual_out        n_train float32, WRITTEN
        8  scalars_out     5 float64, WRITTEN

    `params` is, in this exact order (mirrored in
    `python/mojolearn/_gp_impl.py`):

        0  n_train
        1  n_features
        2  n_nodes         postfix nodes in the kernel spec
        3  n_ls            floats the spec's leaves consume from `ls`
                            (addrs slot 5)
        4  alpha           (float; the RIDGE, which is the Cholesky
                            profile's jitter, DEVIATION 1751. Crosses
                            UNCLAMPED so gp_validate_alpha's refusals --
                            NaN, negative, +inf, and the identical tier's
                            two-value pin -- fire by name, DEVIATIONS
                            1768/1637)

    `ls` (slot 5) holds `max(n_ls, 1)` float32 -- one unused `1.0` stands
    in when the kernel has no length scale at all, exactly as
    `estimator.mojo::_length_scale_table` spells it, and `n_ls` says which
    case this is.

    `l_out` (slot 6) receives the lower Cholesky factor of `K + alpha I`,
    sklearn's `L_`; `dual_out` (slot 7) receives `(K + alpha I)^-1 y`,
    sklearn's `alpha_`, which is NOT the ridge -- the collision is
    scikit-learn's and both sides name it. On a failed fit both still
    cross: the partial factor and the zero dual are what `gpr_fit_host`
    hands back beside a nonzero `info`.

    `scalars_out` (slot 8) is FIVE float64, written in this exact order:

        0  info
        1  nb              the Cholesky panel width that ran
        2  logdet
        3  ydotalpha
        4  lml
    """
    if len(addrs) != 9:
        raise Error(
            "gpr_fit: addrs must contain 9 addresses (x, y, kinds, kparams,"
            " ls_len, ls, l_out, dual_out, scalars_out), got "
            + String(len(addrs))
        )
    if len(params) != 5:
        raise Error(
            "gpr_fit: params must contain 5 values (n_train, n_features,"
            " n_nodes, n_ls, alpha), got "
            + String(len(params))
        )
    var xp = _f32_ptr(Int(py=addrs[0]))
    var yp = _f32_ptr(Int(py=addrs[1]))
    var lp = _f32_ptr(Int(py=addrs[6]))
    var dp = _f32_ptr(Int(py=addrs[7]))
    var sp = _f64_ptr(Int(py=addrs[8]))
    var n_train = Int(py=params[0])
    var n_features = Int(py=params[1])
    var n_nodes = Int(py=params[2])
    var n_ls = Int(py=params[3])
    var alpha = Float32(Float64(py=params[4]))
    # No positivity check on n_train/n_features here: `gp_validate_data`
    # refuses them by name on the Mojo host (a copy loop over an empty or
    # negative range below reads nothing), and a duplicate here would make
    # that refusal unreachable from Python.
    var spec = _rebuild_kernel_spec(
        Int(py=addrs[2]),
        Int(py=addrs[3]),
        Int(py=addrs[4]),
        Int(py=addrs[5]),
        n_nodes,
        n_ls,
        String("gpr_fit"),
    )
    var x = List[Float32]()
    var y = List[Float32]()
    for i in range(n_train * n_features):
        x.append(xp.unsafe_load(i))
    for i in range(n_train):
        y.append(yp.unsafe_load(i))
    var info = 0
    with GILReleased(Python()):
        info = _gpr_fit_run(
            x, y, spec, n_train, n_features, alpha, lp, dp, sp
        )
    return PythonObject(info)


def _gpr_predict_run(
    model: GPRegressor,
    x_star: List[Float32],
    n_star: Int,
    return_std: Bool,
    mean_addr: Int,
    var_addr: Int,
    std_addr: Int,
    clamped_addr: Int,
) raises -> Int:
    """The GIL-free half of `gpr_predict_binding`, for `_gpr_fit_run`'s
    reason. The variance/std/clamp addresses are taken as `Int` and only
    resolved inside the `return_std` arm, so a caller that asked for the
    mean alone never has them dereferenced -- "safe to write or skip", and
    this side skips."""
    var pred = gpr_predict_host(model, x_star, n_star, return_std)
    var mp = _f32_ptr(mean_addr)
    for i in range(n_star):
        mp.unsafe_store(i, pred.mean[i])
    if return_std:
        var vp = _f32_ptr(var_addr)
        var stp = _f32_ptr(std_addr)
        var cp = _i32_ptr(clamped_addr)
        for i in range(n_star):
            vp.unsafe_store(i, pred.variance[i])
            stp.unsafe_store(i, pred.std[i])
            cp.unsafe_store(i, pred.clamped[i])
    return pred.n_clamped


def gpr_predict_binding(
    addrs: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`predict(X_star, return_std)` on a model handed back in
    (gaussian_process/, DEVIATIONS 1758-1760). Returns `n_clamped`, the
    count of test points whose predictive variance was clamped at zero
    (DEVIATION 1760); the per-point flags are in `addrs[11]`.

    `addrs` is the TWELVE buffer addresses, in this exact order (mirrored
    in `python/mojolearn/_gp_impl.py`; the module header says why they are
    a list and not twelve arguments):

        0  xtrain          n_train * n_features float32, read
        1  l               n_train * n_train float32, read
        2  dual            n_train float32, read
        3  xstar           n_star * n_features float32, read
        4  kinds           n_nodes int32, read
        5  kparams         n_nodes float32, read
        6  ls_len          n_nodes int32, read
        7  ls              max(n_ls, 1) float32, read
        8  mean_out        n_star float32, WRITTEN on every call
        9  var_out         n_star float32, WRITTEN when return_std != 0
       10  std_out         n_star float32, WRITTEN when return_std != 0
       11  clamped_out     n_star int32, WRITTEN when return_std != 0

    `params` is, in this exact order (mirrored in
    `python/mojolearn/_gp_impl.py`):

        0  n_train
        1  n_features
        2  n_star
        3  n_nodes
        4  n_ls
        5  return_std      (0/1)
        6  info            LAPACK's info from the fit, PASSED THROUGH

    Slot 6 is the trap in this list, and it is a deliberate one. The
    `GPRegressor` is reconstructed below with `y_train`, `alpha`,
    `logdet`, `ydotalpha`, `lml` and `nb` ZERO-FILLED -- `gpr_predict_host`
    reads none of them -- but `info` goes down AS THE FIT REPORTED IT, so
    that `gpr_predict_host`'s refusal to solve against a partial factor
    (DEVIATION 1634) fires from Python exactly as it fires from Mojo. A
    binding that judged `info` here, or zero-filled it with the rest,
    would make that refusal unreachable, which is the accepted-and-ignored
    failure this surface's tables exist to prevent.

    The training-side arrays are the fit's own: `l` is `gpr_fit`'s
    `l_out`, `dual` its `dual_out`, and the kernel arrays are `gpr_fit`'s,
    rebuilt through the constructors for the same reason. The mean is
    written on every call; with `return_std` zero the var/std/clamp
    addresses are NOT TOUCHED (never dereferenced), and the return value
    is 0.
    """
    if len(addrs) != 12:
        raise Error(
            "gpr_predict: addrs must contain 12 addresses (xtrain, l,"
            " dual, xstar, kinds, kparams, ls_len, ls, mean_out, var_out,"
            " std_out, clamped_out), got "
            + String(len(addrs))
        )
    if len(params) != 7:
        raise Error(
            "gpr_predict: params must contain 7 values (n_train,"
            " n_features, n_star, n_nodes, n_ls, return_std, info), got "
            + String(len(params))
        )
    var xtp = _f32_ptr(Int(py=addrs[0]))
    var lp = _f32_ptr(Int(py=addrs[1]))
    var dp = _f32_ptr(Int(py=addrs[2]))
    var xsp = _f32_ptr(Int(py=addrs[3]))
    var mean_addr = Int(py=addrs[8])
    var var_addr = Int(py=addrs[9])
    var std_addr = Int(py=addrs[10])
    var clamped_addr = Int(py=addrs[11])
    var n_train = Int(py=params[0])
    var n_features = Int(py=params[1])
    var n_star = Int(py=params[2])
    var n_nodes = Int(py=params[3])
    var n_ls = Int(py=params[4])
    var return_std = Int(py=params[5]) != 0
    var info = Int(py=params[6])
    # No positivity check on n_star here: `gpr_predict_host` refuses it by
    # name, after the failed-fit refusal, and both must stay reachable.
    var spec = _rebuild_kernel_spec(
        Int(py=addrs[4]),
        Int(py=addrs[5]),
        Int(py=addrs[6]),
        Int(py=addrs[7]),
        n_nodes,
        n_ls,
        String("gpr_predict"),
    )
    var xt = List[Float32]()
    for i in range(n_train * n_features):
        xt.append(xtp.unsafe_load(i))
    var l = List[Float32]()
    for i in range(n_train * n_train):
        l.append(lp.unsafe_load(i))
    var dual = List[Float32]()
    for i in range(n_train):
        dual.append(dp.unsafe_load(i))
    var x_star = List[Float32]()
    for i in range(n_star * n_features):
        x_star.append(xsp.unsafe_load(i))
    # The reconstructed model. `y_train` and every fit-only scalar are
    # ZERO-FILLED (gpr_predict_host reads none of them); `info` is the
    # caller's, passed through -- the docstring above says why.
    var yzero = List[Float32]()
    for _i in range(n_train):
        yzero.append(Float32(0.0))
    var model = GPRegressor(
        xt^,
        yzero^,
        n_train,
        n_features,
        spec^,
        Float32(0.0),
        l^,
        dual^,
        Float32(0.0),
        Float32(0.0),
        Float32(0.0),
        info,
        0,
    )
    var n_clamped = 0
    with GILReleased(Python()):
        n_clamped = _gpr_predict_run(
            model,
            x_star,
            n_star,
            return_std,
            mean_addr,
            var_addr,
            std_addr,
            clamped_addr,
        )
    return PythonObject(n_clamped)


@export
def PyInit__mojolearn_gp() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_gp")
        m.def_function[gp_vendor_binding]("gp_vendor")
        m.def_function[gp_numeric_mode_binding]("gp_numeric_mode")
        m.def_function[gpr_fit_binding]("gpr_fit")
        m.def_function[gpr_predict_binding]("gpr_predict")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_gp: ", e))
