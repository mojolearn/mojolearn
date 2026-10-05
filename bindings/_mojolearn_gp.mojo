# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPython boundary for the exact dense Gaussian process regression lane.

A THIRTEENTH extension module, and a separate one on purpose. The header of
`bindings/_mojolearn_estimators.mojo` states the reason and it is the same
reason here: an independently changing binding must not become a merge
point. `gaussian_process/` is ORIGINAL WORK (cuML, cuVS and RAFT implement
no Gaussian process at the pinned commits; `gaussian_process/
 All thirteen binaries land in one wheel.

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
signature), before any artifact existed. The maintainer renegotiated BOTH
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

from gaussian_process.unnorm import GP_PY2MOJO, gpc_binary_out, gpc_ovr_targets
from std.os import abort
from bindings.hostptr import f32_ptr, f64_ptr, i32_ptr, copy_f32, read_f32, read_i32
from gaussian_process.host.gp_theta import (
    gp_log64,
    gp_restart_uniform,
    gp_theta_param,
)
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
from gaussian_process.gp_optim import GPOptResult, gpr_optimize_device
from gaussian_process.estimator import (
    GPRegressor,
    gpr_fit_host,
    gpr_lml_grad_host,
    gpr_predict_host,
    gpr_predict_cov_host,
    gpr_sample_y_host,
)
# Gaussian process classification (lane/gaussian-process-classifier,
# 2026-09-15): the binary Laplace fit, its latent prediction and (since
# cpu-gpu-cleanup c-gp-kernel, 2026-10-02) the float64 probability and the
# one-vs-rest combine on the device (DEVIATIONS 2830-2833).
from gaussian_process.classifier import (
    gpc_fit_binary_host,
    gpc_ovr_combine_host,
    gpc_predict_binary_host,
)
# lane fam2-kernel-gp (2026-10-04): every class in one device session.
from gaussian_process.gpc_ovr import GPC_IDN_OVR, gpc_fit_all_device, gpc_predict_all_device
from gaussian_process.gpr_resident import GPR_IDN_PTR, gpr_fit_ptr_device, gpr_predict_ptr_device
# The Cholesky door (workstream D, 2026-09-14). `cholesky/` is already
# linked into this binary because the GP factors through it; exposing the
# one-shot host entries here adds no kernel and no second build.
from cholesky.estimator import (
    CHOL_FAST_DEVIO,
    cholesky_factor_devio,
    CholeskyFactor,
    cholesky_factor_host,
    cholesky_profile_jitter,
    cholesky_solve_host,
)


def _f32_ptr(addr: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    return f32_ptr(addr)


def _i32_ptr(addr: Int) raises -> MutPointer[Int32, MutUntrackedOrigin]:
    return i32_ptr(addr)


def _f64_ptr(addr: Int) raises -> MutPointer[Float64, MutUntrackedOrigin]:
    return f64_ptr(addr)


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


def gp_py2mojo_binding() raises -> PythonObject:
    """1: normalize_y's un-normalization runs in this binding, on the device
    (lane apple-fast-py2mojo-cluster); 0 under
    `-D MOJOLEARN_PY2MOJO_cluster_OFF`, and `_gp_impl.py` applies it in
    Python."""
    return PythonObject(1 if GP_PY2MOJO else 0)


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
    copy_f32(model.l.unsafe_ptr(), lp, n_train * n_train)
    copy_f32(model.dual_coef.unsafe_ptr(), dp, n_train)
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
    var x = read_f32(Int(xp), n_train * n_features)
    var y = read_f32(Int(yp), n_train)
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
    unnorm: Bool = False,
    y_std: Float32 = Float32(1.0),
    y_mean: Float32 = Float32(0.0),
) raises -> Int:
    """The GIL-free half of `gpr_predict_binding`, for `_gpr_fit_run`'s
    reason. The variance/std/clamp addresses are taken as `Int` and only
    resolved inside the `return_std` arm, so a caller that asked for the
    mean alone never has them dereferenced -- "safe to write or skip", and
    this side skips."""
    var pred = gpr_predict_host(
        model, x_star, n_star, return_std, unnorm=unnorm, y_std=y_std, y_mean=y_mean
    )
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
    `GPRegressor` is reconstructed below with an empty `y_train` and
    zero `alpha`, `logdet`, `ydotalpha`, `lml` and `nb` -- `gpr_predict_host`
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
    if len(params) != 7 and len(params) != 10:
        raise Error(
            "gpr_predict: params must contain 7 values (n_train,"
            " n_features, n_star, n_nodes, n_ls, return_std, info), or 10"
            " (+ normalize_y, y_std, y_mean), got "
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
    # params 7-9 (lane apple-fast-py2mojo-cluster): normalize_y's
    # un-normalization of the mean and std, on the device (it ran in Python)
    var unnorm = len(params) == 10 and Int(py=params[7]) != 0
    var y_std = Float32(Float64(py=params[8])) if len(params) == 10 else Float32(1.0)
    var y_mean = Float32(Float64(py=params[9])) if len(params) == 10 else Float32(0.0)
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
    # DEVIATION 2486: bulk copies, including the quadratic Cholesky factor.
    # Caller-owned buffers remain live throughout these integer-address reads.
    var xt = read_f32(Int(xtp), n_train * n_features)
    var l = read_f32(Int(lp), n_train * n_train)
    var dual = read_f32(Int(dp), n_train)
    var x_star = read_f32(Int(xsp), max(0, n_star * n_features))
    # Prediction never reads y_train; retain no unused n_train-element fill.
    # Fit-only scalars remain zero and the caller's info is preserved.
    var yzero = List[Float32]()
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
            unnorm,
            y_std,
            y_mean,
        )
    return PythonObject(n_clamped)


def _gpr_sample_y_run(
    model: GPRegressor,
    x_star: List[Float32],
    n_star: Int,
    n_samples: Int,
    seed: UInt64,
    out_addr: Int,
    unnorm: Bool = False,
    y_std: Float32 = Float32(1.0),
    y_mean: Float32 = Float32(0.0),
) raises:
    """The GIL-free half of `gpr_sample_y_binding`."""
    var y = gpr_sample_y_host(
        model, x_star, n_star, n_samples, seed, unnorm=unnorm, y_std=y_std, y_mean=y_mean
    )
    copy_f32(y.unsafe_ptr(), _f32_ptr(out_addr), n_star * n_samples)


def gpr_sample_y_binding(
    addrs: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`sample_y(X, n_samples, random_state)` on a model handed back in
    (DEVIATION 2793, gaussian_process/checks/sample_y.mojo). Returns
    `n_samples`; the draws are in the model's normalized scale.

    `addrs` is the NINE buffer addresses, in this exact order (mirrored in
    `python/mojolearn/_gp_impl.py::sample_y`):

        0  xtrain          n_train * n_features float32, read
        1  l               n_train * n_train float32, read
        2  dual            n_train float32, read
        3  xstar           n_star * n_features float32, read
        4  kinds           n_nodes int32, read
        5  kparams         n_nodes float32, read
        6  ls_len          n_nodes int32, read
        7  ls              max(n_ls, 1) float32, read
        8  y_out           n_star * n_samples float32, WRITTEN (row i is
                            query row i, column s is draw s)

    `params` is, in this exact order:

        0  n_train
        1  n_features
        2  n_star
        3  n_nodes
        4  n_ls
        5  info            LAPACK's info from the fit, PASSED THROUGH so
                            the refusal to sample from a failed fit fires
                            by name (DEVIATION 1634)
        6  n_samples       refused below 1 by name in Mojo
        7  random_state's low 32 bits
        8  random_state's high 32 bits
    """
    if len(addrs) != 9:
        raise Error(
            "gpr_sample_y: addrs must contain 9 addresses (xtrain, l, dual,"
            " xstar, kinds, kparams, ls_len, ls, y_out), got "
            + String(len(addrs))
        )
    if len(params) != 9 and len(params) != 12:
        raise Error(
            "gpr_sample_y: params must contain 9 values (n_train, n_features,"
            " n_star, n_nodes, n_ls, info, n_samples, seed_lo, seed_hi), or 12"
            " (+ normalize_y, y_std, y_mean), got "
            + String(len(params))
        )
    # params 9-11 (lane apple-fast-py2mojo-cluster): every draw un-normalized
    # on the device (it ran in Python)
    var unnorm = len(params) == 12 and Int(py=params[9]) != 0
    var y_std = Float32(Float64(py=params[10])) if len(params) == 12 else Float32(1.0)
    var y_mean = Float32(Float64(py=params[11])) if len(params) == 12 else Float32(0.0)
    var out_addr = Int(py=addrs[8])
    var n_train = Int(py=params[0])
    var n_features = Int(py=params[1])
    var n_star = Int(py=params[2])
    var n_nodes = Int(py=params[3])
    var n_ls = Int(py=params[4])
    var info = Int(py=params[5])
    var n_samples = Int(py=params[6])
    var seed = (UInt64(Int(py=params[8])) << 32) | UInt64(Int(py=params[7]))
    var spec = _rebuild_kernel_spec(
        Int(py=addrs[4]),
        Int(py=addrs[5]),
        Int(py=addrs[6]),
        Int(py=addrs[7]),
        n_nodes,
        n_ls,
        String("gpr_sample_y"),
    )
    var xt = read_f32(Int(py=addrs[0]), max(0, n_train * n_features))
    var l = read_f32(Int(py=addrs[1]), max(0, n_train * n_train))
    var dual = read_f32(Int(py=addrs[2]), max(0, n_train))
    var x_star = read_f32(Int(py=addrs[3]), max(0, n_star * n_features))
    var yzero = List[Float32]()
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
    with GILReleased(Python()):
        _gpr_sample_y_run(model, x_star, n_star, n_samples, seed, out_addr, unnorm, y_std, y_mean)
    return PythonObject(n_samples)


def _gpr_predict_cov_run(
    model: GPRegressor,
    x_star: List[Float32],
    n_star: Int,
    mean_addr: Int,
    cov_addr: Int,
    unnorm: Bool = False,
    y_std: Float32 = Float32(1.0),
    y_mean: Float32 = Float32(0.0),
) raises:
    """The GIL-free half of `gpr_predict_cov_binding`."""
    var r = gpr_predict_cov_host(model, x_star, n_star, unnorm=unnorm, y_std=y_std, y_mean=y_mean)
    copy_f32(r.mean.unsafe_ptr(), _f32_ptr(mean_addr), n_star)
    copy_f32(r.cov.unsafe_ptr(), _f32_ptr(cov_addr), n_star * n_star)


def gpr_predict_cov_binding(
    addrs: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """`predict(X, return_cov=True)` on a model handed back in
    (`gpr_predict_cov_host`). `addrs`: 0 xtrain, 1 l, 2 dual, 3 xstar,
    4 kinds, 5 kparams, 6 ls_len, 7 ls (as gpr_sample_y), 8 mean_out
    (n_star float32), 9 cov_out (n_star * n_star float32). `params`:
    0 n_train, 1 n_features, 2 n_star, 3 n_nodes, 4 n_ls, 5 info (passed
    through). Returns n_star; both outputs in the normalized scale."""
    if len(addrs) != 10:
        raise Error("gpr_predict_cov: addrs must contain 10 addresses, got " + String(len(addrs)))
    if len(params) != 6 and len(params) != 9:
        raise Error("gpr_predict_cov: params must contain 6 or 9 values, got " + String(len(params)))
    # params 6-8 (lane apple-fast-py2mojo-cluster): the mean and every
    # covariance cell un-normalized on the device (it ran in Python)
    var unnorm = len(params) == 9 and Int(py=params[6]) != 0
    var y_std = Float32(Float64(py=params[7])) if len(params) == 9 else Float32(1.0)
    var y_mean = Float32(Float64(py=params[8])) if len(params) == 9 else Float32(0.0)
    var n_train = Int(py=params[0])
    var n_features = Int(py=params[1])
    var n_star = Int(py=params[2])
    var n_nodes = Int(py=params[3])
    var n_ls = Int(py=params[4])
    var info = Int(py=params[5])
    var spec = _rebuild_kernel_spec(
        Int(py=addrs[4]), Int(py=addrs[5]), Int(py=addrs[6]), Int(py=addrs[7]),
        n_nodes, n_ls, String("gpr_predict_cov"),
    )
    var xt = read_f32(Int(py=addrs[0]), max(0, n_train * n_features))
    var l = read_f32(Int(py=addrs[1]), max(0, n_train * n_train))
    var dual = read_f32(Int(py=addrs[2]), max(0, n_train))
    var x_star = read_f32(Int(py=addrs[3]), max(0, n_star * n_features))
    var mean_addr = Int(py=addrs[8])
    var cov_addr = Int(py=addrs[9])
    var yzero = List[Float32]()
    var model = GPRegressor(
        xt^, yzero^, n_train, n_features, spec^, Float32(0.0), l^, dual^,
        Float32(0.0), Float32(0.0), Float32(0.0), info, 0,
    )
    with GILReleased(Python()):
        _gpr_predict_cov_run(model, x_star, n_star, mean_addr, cov_addr, unnorm, y_std, y_mean)
    return PythonObject(n_star)


# ===========================================================================
# THE CHOLESKY DOOR (workstream D, 2026-09-14). `cholesky/estimator.mojo`'s
# one-shot host entries, reached through THIS binding because the GP build
# already links the whole of `cholesky/` (bindings/build_gp.sh's blob table
# lists its potrf, trsm, logdet and jitter kernels). Nothing new is
# compiled; a second binding would be a second copy of the same kernels.
# The ABI is the GP's: two lists, an address list and a params list, each
# length-checked, orders written out here and mirrored in
# python/mojolearn/_cholesky_impl.py.
# ===========================================================================


def cholesky_parallel_available() raises -> PythonObject:
    """1: potrf_lower and cho_solve read MOJOLEARN_CHOLESKY_DEVICE_COUNT and
    move whole trailing-update rows and right-hand-side columns
    (cholesky/multi_gpu.mojo)."""
    return PythonObject(1)


def cholesky_profile_jitter_binding() raises -> PythonObject:
    """The profile's pinned ridge (`chol_jitter_pinned`, DEVIATION 1637),
    as a Python float. The Python side reads it here so its default is the
    NAME and never a literal that can drift from the pin."""
    return PythonObject(Float64(cholesky_profile_jitter()))


def _cholesky_factor_run(
    a: List[Float32],
    n: Int,
    jitter: Float32,
    lp: MutPointer[Float32, MutUntrackedOrigin],
    sp: MutPointer[Float64, MutUntrackedOrigin],
) raises -> Int:
    """The GIL-free half of `cholesky_factor_binding`."""
    var f = cholesky_factor_host(a, n, jitter)
    copy_f32(f.l.unsafe_ptr(), lp, n * n)
    # info, nb, logdet, jitter -- in that order, mirrored in
    # `_cholesky_impl.py::Cholesky.fit`. Each widens to float64 exactly.
    sp.unsafe_store(0, Float64(f.info))
    sp.unsafe_store(1, Float64(f.nb))
    sp.unsafe_store(2, Float64(f.logdet))
    sp.unsafe_store(3, Float64(f.jitter))
    return f.info


def cholesky_factor_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`cholesky_factor_host(a, n, jitter)`: `A + jitter I = L L^T`, host
    in and host out. Returns LAPACK's `info`, a RESULT and not an exception
    (DEVIATION 1634): 0 means `l_out` holds the factor, `k > 0` that the
    leading minor of order `k` was not positive definite and `l_out` holds
    a partial result.

    `addrs`, in this exact order:

        0  a               n * n float32, row-major, read
        1  l_out           n * n float32, WRITTEN (lower triangle L, strict
                            upper +0.0)
        2  scalars_out     4 float64, WRITTEN: info, nb, logdet, jitter

    `params`, in this exact order:

        0  n
        1  jitter          (float; crosses UNCLAMPED so chol_validate_jitter
                            refuses an unpinned value by name, DEVIATION 1637)

    Non-finite and non-symmetric matrices are refused by name on the Mojo
    host BEFORE any upload (`chol_validate_matrix`, DEVIATION 1638); nothing
    is judged here except the two list lengths.
    """
    if len(addrs) != 3:
        raise Error(
            "cholesky_factor: addrs must contain 3 addresses (a, l_out,"
            " scalars_out), got "
            + String(len(addrs))
        )
    if len(params) != 2:
        raise Error(
            "cholesky_factor: params must contain 2 values (n, jitter), got "
            + String(len(params))
        )
    var ap = _f32_ptr(Int(py=addrs[0]))
    var lp = _f32_ptr(Int(py=addrs[1]))
    var sp = _f64_ptr(Int(py=addrs[2]))
    var n = Int(py=params[0])
    var jitter = Float32(Float64(py=params[1]))
    var info = 0
    comptime if CHOL_FAST_DEVIO:
        # CHOL_FAST_DEVIO (FAST + Apple default, cholesky/estimator.mojo;
        # -D MOJOLEARN_CHOL_FAST_DEVIO_OFF reverts)
        with GILReleased(Python()):
            info = cholesky_factor_devio(ap, lp, sp, n, jitter)
        return PythonObject(info)
    var a = read_f32(Int(ap), max(0, n * n))
    with GILReleased(Python()):
        info = _cholesky_factor_run(a, n, jitter, lp, sp)
    return PythonObject(info)


def _cholesky_solve_run(
    factor: CholeskyFactor,
    b: List[Float32],
    nrhs: Int,
    xp: MutPointer[Float32, MutUntrackedOrigin],
) raises:
    """The GIL-free half of `cholesky_solve_binding`."""
    var x = cholesky_solve_host(factor, b, nrhs)
    copy_f32(x.unsafe_ptr(), xp, factor.n * nrhs)


def cholesky_solve_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`cholesky_solve_host(factor, b, nrhs)`: `A X = B` from the factor
    `cholesky_factor` returned. cuSOLVER's `potrs`. Returns 0.

    `addrs`, in this exact order:

        0  l               n * n float32, the factor, read
        1  b               n * nrhs float32, row-major, read
        2  x_out           n * nrhs float32, WRITTEN

    `params`, in this exact order:

        0  n
        1  nrhs
        2  info            LAPACK's info from the factorization, PASSED
                            THROUGH so `cholesky_solve_host`'s refusal to
                            solve against a FAILED factor (DEVIATION 1634)
                            fires from Python exactly as it fires from Mojo
        3  nb              the panel width that ran (part of the profile)
        4  logdet
        5  jitter

    Slot 2 is the trap in this list and it is deliberate, for the reason
    `gpr_predict_binding` gives about its own `info` slot: a binding that
    judged it, or zero-filled it, would make that refusal unreachable.
    """
    if len(addrs) != 3:
        raise Error(
            "cholesky_solve: addrs must contain 3 addresses (l, b, x_out),"
            " got "
            + String(len(addrs))
        )
    if len(params) != 6:
        raise Error(
            "cholesky_solve: params must contain 6 values (n, nrhs, info,"
            " nb, logdet, jitter), got "
            + String(len(params))
        )
    var lp = _f32_ptr(Int(py=addrs[0]))
    var bp = _f32_ptr(Int(py=addrs[1]))
    var xp = _f32_ptr(Int(py=addrs[2]))
    var n = Int(py=params[0])
    var nrhs = Int(py=params[1])
    var info = Int(py=params[2])
    var nb = Int(py=params[3])
    var logdet = Float32(Float64(py=params[4]))
    var jitter = Float32(Float64(py=params[5]))
    var l = read_f32(Int(lp), max(0, n * n))
    var b = read_f32(Int(bp), max(0, n * nrhs))
    var factor = CholeskyFactor(l^, n, info, logdet, nb, jitter)
    with GILReleased(Python()):
        _cholesky_solve_run(factor, b, nrhs, xp)
    return PythonObject(0)


# ===========================================================================
# GAUSSIAN PROCESS CLASSIFICATION (lane/gaussian-process-classifier,
# 2026-09-15). One BINARY Laplace fit per call; the one-vs-rest loop past
# two classes is python/mojolearn/_gpc_impl.py's (DEVIATION 2833). The ABI is
# the GP's: an address list and a params list, each length-checked, orders
# written out here and mirrored at the _gpc_impl.py call sites and in
# bindings/_mojolearn_gp_host.mojo.
# ===========================================================================


def _gpc_fit_run(
    x: List[Float32],
    y: List[Float32],
    spec: GPKernelSpec,
    n_train: Int,
    n_features: Int,
    max_iter_predict: Int,
    lp: MutPointer[Float32, MutUntrackedOrigin],
    pp: MutPointer[Float32, MutUntrackedOrigin],
    wp: MutPointer[Float32, MutUntrackedOrigin],
    sp: MutPointer[Float64, MutUntrackedOrigin],
) raises -> Int:
    """The GIL-free half of `gpc_fit_binding`."""
    var fit = gpc_fit_binary_host(x, n_train, n_features, y, spec, max_iter_predict)
    for i in range(n_train * n_train):
        lp.unsafe_store(i, fit.l[i])
    for i in range(n_train):
        pp.unsafe_store(i, fit.pi[i])
        wp.unsafe_store(i, fit.wsr[i])
    # lml, n_iter, nb, in that order; each widens to float64 exactly.
    sp.unsafe_store(0, Float64(fit.lml))
    sp.unsafe_store(1, Float64(fit.n_iter))
    sp.unsafe_store(2, Float64(fit.nb))
    return fit.n_iter


def gpc_fit_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """One binary `_BinaryGaussianProcessClassifierLaplace(kernel,
    optimizer=None).fit(X, y)` with `y` encoded to 0 and 1. Returns the
    Newton iteration count (DEVIATION 2830).

    `addrs`, in this exact order:

        0  x               n_train * n_features float32, row-major, read
        1  y               n_train float32, each 0 or 1, read
        2  kinds           n_nodes int32, read
        3  kparams         n_nodes float32, read
        4  ls_len          n_nodes int32, read
        5  ls              max(n_ls, 1) float32, read
        6  l_out           n_train * n_train float32, WRITTEN (L of B)
        7  pi_out          n_train float32, WRITTEN
        8  wsr_out         n_train float32, WRITTEN
        9  scalars_out     3 float64, WRITTEN: lml, n_iter, nb

    `params`, in this exact order: 0 n_train, 1 n_features, 2 n_nodes,
    3 n_ls, 4 max_iter_predict.
    """
    if len(addrs) != 10 and not (len(addrs) == 11 and len(params) == 6):
        raise Error(
            "gpc_fit: addrs must contain 10 addresses (x, y, kinds, kparams,"
            " ls_len, ls, l_out, pi_out, wsr_out, scalars_out), got "
            + String(len(addrs))
        )
    if len(params) != 5 and len(params) != 6:
        raise Error(
            "gpc_fit: params must contain 5 values (n_train, n_features,"
            " n_nodes, n_ls, max_iter_predict), or 6 (+ the one-vs-rest class"
            " k, when addrs[1] is the n_train int32 class codes), got "
            + String(len(params))
        )
    var lp = _f32_ptr(Int(py=addrs[6]))
    var pp = _f32_ptr(Int(py=addrs[7]))
    var wp = _f32_ptr(Int(py=addrs[8]))
    var sp = _f64_ptr(Int(py=addrs[9]))
    var n_train = Int(py=params[0])
    var n_features = Int(py=params[1])
    var n_nodes = Int(py=params[2])
    var n_ls = Int(py=params[3])
    var max_iter_predict = Int(py=params[4])
    var spec = _rebuild_kernel_spec(
        Int(py=addrs[2]),
        Int(py=addrs[3]),
        Int(py=addrs[4]),
        Int(py=addrs[5]),
        n_nodes,
        n_ls,
        String("gpc_fit"),
    )
    var x = read_f32(Int(py=addrs[0]), max(0, n_train * n_features))
    # params[5] (lane apple-fast-py2mojo-cluster): addrs[1] holds the int32
    # class codes and the targets are code == k, built here, not in Python
    var y: List[Float32]
    if len(params) == 6:
        y = gpc_ovr_targets(read_i32(Int(py=addrs[1]), max(0, n_train)), n_train, Int(py=params[5]))
        if len(addrs) == 11:
            # addrs[10]: the targets, WRITTEN (the fitted model keeps them)
            var yo = f32_ptr(Int(py=addrs[10]))
            for i in range(n_train):
                yo.unsafe_store(i, y[i])
    else:
        y = read_f32(Int(py=addrs[1]), max(0, n_train))
    var n_iter = 0
    with GILReleased(Python()):
        n_iter = _gpc_fit_run(
            x, y, spec, n_train, n_features, max_iter_predict, lp, pp, wp, sp
        )
    return PythonObject(n_iter)


def _gpc_predict_run(
    xt: List[Float32],
    y: List[Float32],
    pi: List[Float32],
    wsr: List[Float32],
    l: List[Float32],
    spec: GPKernelSpec,
    x_star: List[Float32],
    n_train: Int,
    n_features: Int,
    n_star: Int,
    want_proba: Bool,
    mean_addr: Int,
    var_addr: Int,
    proba_addr: Int,
) raises -> Int:
    """The GIL-free half of `gpc_predict_binding`. The variance and
    probability addresses are resolved only when `want_proba`."""
    var lat = gpc_predict_binary_host(
        xt, y, pi, wsr, l, n_train, n_features, spec, x_star, n_star, want_proba
    )
    var mp = _f32_ptr(mean_addr)
    for t in range(n_star):
        mp.unsafe_store(t, lat.mean[t])
    if want_proba:
        var vp = _f32_ptr(var_addr)
        var pr = _f64_ptr(proba_addr)
        for t in range(n_star):
            vp.unsafe_store(t, lat.variance[t])
            pr.unsafe_store(t, lat.proba[t])
    return 0


def gpc_predict_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """The latent mean, and with `want_proba` the latent variance and the
    probability of class 1, of one binary fit at the query rows. Returns 0.

    `addrs`, in this exact order:

        0  xtrain          n_train * n_features float32, read
        1  y               n_train float32 (0 or 1), read
        2  pi              n_train float32, read
        3  wsr             n_train float32, read
        4  l               n_train * n_train float32, read
        5  xstar           n_star * n_features float32, read
        6  kinds           n_nodes int32, read
        7  kparams         n_nodes float32, read
        8  ls_len          n_nodes int32, read
        9  ls              max(n_ls, 1) float32, read
        10 mean_out        n_star float32, WRITTEN
        11 var_out         n_star float32, WRITTEN when want_proba
        12 proba_out       n_star float64, WRITTEN when want_proba

    `params`, in this exact order: 0 n_train, 1 n_features, 2 n_star,
    3 n_nodes, 4 n_ls, 5 want_proba.
    """
    # addrs[13] + params[6] (lane apple-fast-py2mojo-cluster): an output the
    # Python side computed from these, `gpc_binary_out`'s kind
    var out_kind = Int(py=params[6]) if len(params) == 7 else 0
    if len(addrs) != 13 and not (len(addrs) == 14 and out_kind != 0):
        raise Error(
            "gpc_predict: addrs must contain 13 addresses (xtrain, y, pi,"
            " wsr, l, xstar, kinds, kparams, ls_len, ls, mean_out, var_out,"
            " proba_out), got "
            + String(len(addrs))
        )
    if len(params) != 6 and len(params) != 7:
        raise Error(
            "gpc_predict: params must contain 6 values (n_train, n_features,"
            " n_star, n_nodes, n_ls, want_proba), got "
            + String(len(params))
        )
    var n_train = Int(py=params[0])
    var n_features = Int(py=params[1])
    var n_star = Int(py=params[2])
    var n_nodes = Int(py=params[3])
    var n_ls = Int(py=params[4])
    var want_proba = Int(py=params[5]) != 0
    var mean_addr = Int(py=addrs[10])
    var var_addr = Int(py=addrs[11])
    var proba_addr = Int(py=addrs[12])
    var spec = _rebuild_kernel_spec(
        Int(py=addrs[6]),
        Int(py=addrs[7]),
        Int(py=addrs[8]),
        Int(py=addrs[9]),
        n_nodes,
        n_ls,
        String("gpc_predict"),
    )
    var xt = read_f32(Int(py=addrs[0]), max(0, n_train * n_features))
    var y = read_f32(Int(py=addrs[1]), max(0, n_train))
    var pi = read_f32(Int(py=addrs[2]), max(0, n_train))
    var wsr = read_f32(Int(py=addrs[3]), max(0, n_train))
    var l = read_f32(Int(py=addrs[4]), max(0, n_train * n_train))
    var x_star = read_f32(Int(py=addrs[5]), max(0, n_star * n_features))
    var rc = 0
    with GILReleased(Python()):
        rc = _gpc_predict_run(
            xt, y, pi, wsr, l, spec, x_star, n_train, n_features, n_star,
            want_proba, mean_addr, var_addr, proba_addr,
        )
    if out_kind != 0:
        gpc_binary_out(mean_addr, proba_addr, n_star, out_kind, Int(py=addrs[13]))
    return PythonObject(rc)


def gpc_ovr_combine_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """DEVIATION 2833's one-vs-rest combine on the device: the k class-1
    probability columns normalized per row (ascending sum from +0.0, divided
    unless zero) and the first strictly largest column per row. Returns 0.

    `addrs`, in this exact order:

        0      proba_out  n * k float64, row-major, WRITTEN
        1      codes_out  n int32, WRITTEN
        2..k+1 col_c      n float64, class c's probabilities, read

    `params`, in this exact order: 0 n.
    """
    if len(addrs) < 3:
        raise Error(
            "gpc_ovr_combine: addrs must contain proba_out, codes_out and at"
            " least one class column, got "
            + String(len(addrs))
            + " addresses"
        )
    if len(params) != 1:
        raise Error(
            "gpc_ovr_combine: params must contain 1 value (n), got "
            + String(len(params))
        )
    var n = Int(py=params[0])
    var out_addr = Int(py=addrs[0])
    var codes_addr = Int(py=addrs[1])
    var cols = List[Int]()
    for c in range(2, len(addrs)):  # small-loop(addrs: one column address per class): pointer list, not data
        cols.append(Int(py=addrs[c]))
    with GILReleased(Python()):
        gpc_ovr_combine_host(cols, out_addr, codes_addr, n)
    return PythonObject(0)


# ===========================================================================
# GPC, EVERY CLASS IN ONE CALL (lane fam2-kernel-gp, 2026-10-04;
# gaussian_process/gpc_ovr.mojo). NEW entries: `gpc_fit`, `gpc_predict` and
# `gpc_ovr_combine` above are unchanged. `gp_idn_caps` tells the Python glue
# whether to take them (`-D MOJOLEARN_IDN_GPC_OVR_OFF` answers 0).
# ===========================================================================


def gp_idn_caps_binding() raises -> PythonObject:
    """Bit 0: `gpc_fit_all` / `gpc_predict_all` are the route
    (`GPC_IDN_OVR`, IDENTICAL builds, on by default). Bit 1: `gpr_fit_ptr` /
    `gpr_predict_ptr` are (`GPR_IDN_PTR`, the same)."""
    return PythonObject((1 if GPC_IDN_OVR else 0) | (2 if GPR_IDN_PTR else 0))


def gpr_fit_ptr_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """`gpr_fit` with the same `addrs` (9) and `params` (5) in the same
    order and the same outputs, read from and written to the caller's memory
    by the device (`gaussian_process/gpr_resident.mojo`): no host list, no
    host finiteness walk. Returns LAPACK's `info`."""
    if len(addrs) != 9:
        raise Error(
            "gpr_fit_ptr: addrs must contain 9 addresses (x, y, kinds, kparams,"
            " ls_len, ls, l_out, dual_out, scalars_out), got "
            + String(len(addrs))
        )
    if len(params) != 5:
        raise Error(
            "gpr_fit_ptr: params must contain 5 values (n_train, n_features,"
            " n_nodes, n_ls, alpha), got "
            + String(len(params))
        )
    var n_train = Int(py=params[0])
    var n_features = Int(py=params[1])
    var n_nodes = Int(py=params[2])
    var n_ls = Int(py=params[3])
    var alpha = Float32(Float64(py=params[4]))
    var spec = _rebuild_kernel_spec(
        Int(py=addrs[2]),
        Int(py=addrs[3]),
        Int(py=addrs[4]),
        Int(py=addrs[5]),
        n_nodes,
        n_ls,
        String("gpr_fit_ptr"),
    )
    var x_addr = Int(py=addrs[0])
    var y_addr = Int(py=addrs[1])
    var l_addr = Int(py=addrs[6])
    var dual_addr = Int(py=addrs[7])
    var scalars_addr = Int(py=addrs[8])
    var info = 0
    with GILReleased(Python()):
        info = gpr_fit_ptr_device(
            x_addr, y_addr, n_train, n_features, spec, alpha, l_addr, dual_addr, scalars_addr
        )
    return PythonObject(info)


def gpr_predict_ptr_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """`gpr_predict` with the same `addrs` (12) and `params` (7, or 10 with
    normalize_y, y_std, y_mean) in the same order and the same outputs, the
    model and the query read from the caller's memory by the device
    (`gaussian_process/gpr_resident.mojo`). Returns `n_clamped`."""
    if len(addrs) != 12:
        raise Error(
            "gpr_predict_ptr: addrs must contain 12 addresses (xtrain, l,"
            " dual, xstar, kinds, kparams, ls_len, ls, mean_out, var_out,"
            " std_out, clamped_out), got "
            + String(len(addrs))
        )
    if len(params) != 7 and len(params) != 10:
        raise Error(
            "gpr_predict_ptr: params must contain 7 values (n_train,"
            " n_features, n_star, n_nodes, n_ls, return_std, info), or 10"
            " (+ normalize_y, y_std, y_mean), got "
            + String(len(params))
        )
    var n_train = Int(py=params[0])
    var n_features = Int(py=params[1])
    var n_star = Int(py=params[2])
    var n_nodes = Int(py=params[3])
    var n_ls = Int(py=params[4])
    var return_std = Int(py=params[5]) != 0
    var info = Int(py=params[6])
    var unnorm = len(params) == 10 and Int(py=params[7]) != 0
    var y_std = Float32(Float64(py=params[8])) if len(params) == 10 else Float32(1.0)
    var y_mean = Float32(Float64(py=params[9])) if len(params) == 10 else Float32(0.0)
    var spec = _rebuild_kernel_spec(
        Int(py=addrs[4]),
        Int(py=addrs[5]),
        Int(py=addrs[6]),
        Int(py=addrs[7]),
        n_nodes,
        n_ls,
        String("gpr_predict_ptr"),
    )
    var xt_addr = Int(py=addrs[0])
    var l_addr = Int(py=addrs[1])
    var dual_addr = Int(py=addrs[2])
    var xs_addr = Int(py=addrs[3])
    var mean_addr = Int(py=addrs[8])
    var var_addr = Int(py=addrs[9])
    var std_addr = Int(py=addrs[10])
    var clamped_addr = Int(py=addrs[11])
    var n_clamped = 0
    with GILReleased(Python()):
        n_clamped = gpr_predict_ptr_device(
            xt_addr, l_addr, dual_addr, xs_addr, n_train, n_features, n_star, spec,
            return_std, info, unnorm, y_std, y_mean,
            mean_addr, var_addr, std_addr, clamped_addr,
        )
    return PythonObject(n_clamped)


def gpc_fit_all_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """Every binary Laplace fit of one classifier against one upload of X
    and one kernel matrix. Returns 0.

    `addrs`, in this exact order:

        0  x        n_train * n_features float32, read
        1  codes    n_train int32 class codes, read
        2  kinds    n_nodes int32, read
        3  kparams  n_nodes float32, read
        4  ls_len   n_nodes int32, read
        5  ls       max(n_ls, 1) float32, read
        then, for binary fit j = 0 .. n_fits - 1, five addresses:
        6 + 5j      y_out        n_train float32, WRITTEN (code == k_j)
        7 + 5j      l_out        n_train * n_train float32, WRITTEN
        8 + 5j      pi_out       n_train float32, WRITTEN
        9 + 5j      wsr_out      n_train float32, WRITTEN
        10 + 5j     scalars_out  3 float64 (lml, n_iter, nb), WRITTEN

    `params`, in this exact order: 0 n_train, 1 n_features, 2 n_nodes,
    3 n_ls, 4 max_iter_predict, 5 n_fits, then the n_fits class codes k_j.
    """
    if len(params) < 7:
        raise Error(
            "gpc_fit_all: params must contain n_train, n_features, n_nodes,"
            " n_ls, max_iter_predict, n_fits and one class per fit, got "
            + String(len(params))
        )
    var n_train = Int(py=params[0])
    var n_features = Int(py=params[1])
    var n_nodes = Int(py=params[2])
    var n_ls = Int(py=params[3])
    var max_iter_predict = Int(py=params[4])
    var n_fits = Int(py=params[5])
    if n_fits < 1 or len(params) != 6 + n_fits or len(addrs) != 6 + 5 * n_fits:
        raise Error(
            "gpc_fit_all: n_fits="
            + String(n_fits)
            + " needs 6 + n_fits params and 6 + 5 n_fits addresses, got "
            + String(len(params))
            + " and "
            + String(len(addrs))
        )
    var spec = _rebuild_kernel_spec(
        Int(py=addrs[2]),
        Int(py=addrs[3]),
        Int(py=addrs[4]),
        Int(py=addrs[5]),
        n_nodes,
        n_ls,
        String("gpc_fit_all"),
    )
    var x_addr = Int(py=addrs[0])
    var codes_addr = Int(py=addrs[1])
    var ks = List[Int]()
    var y_addrs = List[Int]()
    var l_addrs = List[Int]()
    var pi_addrs = List[Int]()
    var wsr_addrs = List[Int]()
    var scalar_addrs = List[Int]()
    for j in range(n_fits):  # small-loop(n_fits: one fit per one-vs-rest class): address and size lists, not data
        ks.append(Int(py=params[6 + j]))
        y_addrs.append(Int(py=addrs[6 + 5 * j]))
        l_addrs.append(Int(py=addrs[7 + 5 * j]))
        pi_addrs.append(Int(py=addrs[8 + 5 * j]))
        wsr_addrs.append(Int(py=addrs[9 + 5 * j]))
        scalar_addrs.append(Int(py=addrs[10 + 5 * j]))
    with GILReleased(Python()):
        gpc_fit_all_device(
            x_addr, codes_addr, n_train, n_features, spec, max_iter_predict,
            ks, y_addrs, l_addrs, pi_addrs, wsr_addrs, scalar_addrs,
        )
    return PythonObject(0)


def gpc_predict_all_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """`predict` or `predict_proba` of a fitted classifier, every binary fit
    against one cross kernel, the outputs formed on the device. Returns 0.

    `addrs`, in this exact order:

        0  xtrain   n_train * n_features float32, read
        1  xstar    n_star * n_features float32, read
        2  kinds    n_nodes int32, read
        3  kparams  n_nodes float32, read
        4  ls_len   n_nodes int32, read
        5  ls       max(n_ls, 1) float32, read
        6  out      out_kind 1: n_star int64 class codes, WRITTEN
                    out_kind 2: n_star * max(k, 2) float64, WRITTEN
        then, for binary fit c = 0 .. k - 1, four addresses:
        7 + 4c      y    n_train float32, read
        8 + 4c      pi   n_train float32, read
        9 + 4c      wsr  n_train float32, read
        10 + 4c     l    n_train * n_train float32, read

    `params`, in this exact order: 0 n_train, 1 n_features, 2 n_star,
    3 n_nodes, 4 n_ls, 5 out_kind, 6 k.
    """
    if len(params) != 7:
        raise Error(
            "gpc_predict_all: params must contain 7 values (n_train,"
            " n_features, n_star, n_nodes, n_ls, out_kind, k), got "
            + String(len(params))
        )
    var n_train = Int(py=params[0])
    var n_features = Int(py=params[1])
    var n_star = Int(py=params[2])
    var n_nodes = Int(py=params[3])
    var n_ls = Int(py=params[4])
    var out_kind = Int(py=params[5])
    var k = Int(py=params[6])
    if k < 1 or len(addrs) != 7 + 4 * k:
        raise Error(
            "gpc_predict_all: k="
            + String(k)
            + " needs 7 + 4 k addresses, got "
            + String(len(addrs))
        )
    var spec = _rebuild_kernel_spec(
        Int(py=addrs[2]),
        Int(py=addrs[3]),
        Int(py=addrs[4]),
        Int(py=addrs[5]),
        n_nodes,
        n_ls,
        String("gpc_predict_all"),
    )
    var xt_addr = Int(py=addrs[0])
    var xs_addr = Int(py=addrs[1])
    var out_addr = Int(py=addrs[6])
    var y_addrs = List[Int]()
    var pi_addrs = List[Int]()
    var wsr_addrs = List[Int]()
    var l_addrs = List[Int]()
    for c in range(k):  # small-loop(k: one entry per class): address lists, not data
        y_addrs.append(Int(py=addrs[7 + 4 * c]))
        pi_addrs.append(Int(py=addrs[8 + 4 * c]))
        wsr_addrs.append(Int(py=addrs[9 + 4 * c]))
        l_addrs.append(Int(py=addrs[10 + 4 * c]))
    with GILReleased(Python()):
        gpc_predict_all_device(
            xt_addr, n_train, n_features, spec, xs_addr, n_star,
            y_addrs, pi_addrs, wsr_addrs, l_addrs, out_kind, out_addr,
        )
    return PythonObject(0)


# ===========================================================================
# KERNEL HYPERPARAMETER OPTIMIZATION (lane/gp-optimizer, 2026-09-15;
# on the device since cgr4-device-optim-gp, 2026-10-03). `gpr_optimize` runs
# DEVIATION 2881's whole optimizer, every start, in one call
# (gaussian_process/gp_optim.mojo); `gpr_lml_grad` is the likelihood and its
# gradient at one kernel (DEVIATION 2880), for log_marginal_likelihood(theta).
# ===========================================================================


def _gpr_optimize_run(
    x: List[Float32],
    y: List[Float32],
    spec: GPKernelSpec,
    free: List[Int32],
    bounds: List[Float32],
    n_train: Int,
    n_features: Int,
    alpha: Float32,
    n_restarts: Int,
    seed_lo: UInt32,
    seed_hi: UInt32,
    tp: MutPointer[Float64, MutUntrackedOrigin],
    vp: MutPointer[Float64, MutUntrackedOrigin],
    rp: MutPointer[Float64, MutUntrackedOrigin],
) raises -> Int:
    """The GIL-free half of `gpr_optimize_binding`."""
    var r = gpr_optimize_device(
        x, n_train, n_features, y, spec, free, bounds, alpha, n_restarts, seed_lo, seed_hi
    )
    for i in range(len(r.theta)):
        tp.unsafe_store(i, Float64(r.theta[i]))
        vp.unsafe_store(i, Float64(r.values[i]))
    for i in range(len(r.runs)):
        rp.unsafe_store(i, Float64(r.runs[i]))
    return r.best


def gpr_optimize_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """DEVIATION 2881 on the device: every start, one call. Returns the
    winning run's index.

    `addrs`, in this exact order (mirrored in `python/mojolearn/_gp_impl.py`
    and `bindings/_mojolearn_gp_host.mojo`):

        0  x               n_train * n_features float32, read
        1  y               n_train float32, read
        2  kinds           n_nodes int32, read
        3  kparams         n_nodes float32, read (the starting values)
        4  ls_len          n_nodes int32, read
        5  ls              max(n_ls, 1) float32, read (the starting values)
        6  free            n_nodes int32 (0 fixed, 1 free), read
        7  bounds          2 * n_theta float32 (lo, hi per entry, values), read
        8  theta_out       n_theta float64, WRITTEN (float32 widened)
        9  values_out      n_theta float64, WRITTEN: the hyperparameters there
        10 runs_out        (1 + n_restarts) * 4 float64, WRITTEN: n_iter,
                           n_eval, stop code, f = -lml per run

    `params`: 0 n_train, 1 n_features, 2 n_nodes, 3 n_ls, 4 alpha,
    5 n_restarts, 6 seed low 32 bits, 7 seed high 32 bits.
    """
    if len(addrs) != 11:
        raise Error(
            "gpr_optimize: addrs must contain 11 addresses (x, y, kinds,"
            " kparams, ls_len, ls, free, bounds, theta_out, values_out,"
            " runs_out), got " + String(len(addrs))
        )
    if len(params) != 8:
        raise Error(
            "gpr_optimize: params must contain 8 values (n_train, n_features,"
            " n_nodes, n_ls, alpha, n_restarts, seed_lo, seed_hi), got "
            + String(len(params))
        )
    var n_train = Int(py=params[0])
    var n_features = Int(py=params[1])
    var n_nodes = Int(py=params[2])
    var n_ls = Int(py=params[3])
    var alpha = Float32(Float64(py=params[4]))
    var n_restarts = Int(py=params[5])
    var seed_lo = UInt32(Int(py=params[6]) & 0xFFFFFFFF)
    var seed_hi = UInt32(Int(py=params[7]) & 0xFFFFFFFF)
    var spec = _rebuild_kernel_spec(
        Int(py=addrs[2]), Int(py=addrs[3]), Int(py=addrs[4]), Int(py=addrs[5]),
        n_nodes, n_ls, String("gpr_optimize"),
    )
    var free = read_i32(Int(py=addrs[6]), max(0, n_nodes))
    var n_theta = 0
    for t in range(n_nodes):  # small-loop(n_nodes: nodes of the kernel expression tree): hyperparameter count, not data
        if Int(free[t]) != 0:
            var k = Int(spec.kinds[t])
            n_theta += Int(spec.ls_len[t]) if (k == GP_K_RBF or k == GP_K_MATERN) else 1
    var bounds = read_f32(Int(py=addrs[7]), 2 * n_theta)
    var tp = _f64_ptr(Int(py=addrs[8]))
    var vp = _f64_ptr(Int(py=addrs[9]))
    var rp = _f64_ptr(Int(py=addrs[10]))
    var x = read_f32(Int(py=addrs[0]), max(0, n_train * n_features))
    var y = read_f32(Int(py=addrs[1]), max(0, n_train))
    var best = 0
    with GILReleased(Python()):
        best = _gpr_optimize_run(
            x, y, spec, free, bounds, n_train, n_features, alpha, n_restarts,
            seed_lo, seed_hi, tp, vp, rp,
        )
    return PythonObject(best)


def _gpr_lml_grad_run(
    x: List[Float32],
    y: List[Float32],
    spec: GPKernelSpec,
    free: List[Int32],
    n_train: Int,
    n_features: Int,
    alpha: Float32,
    gp: MutPointer[Float64, MutUntrackedOrigin],
    sp: MutPointer[Float64, MutUntrackedOrigin],
) raises -> Int:
    """The GIL-free half of `gpr_lml_grad_binding`."""
    var r = gpr_lml_grad_host(x, n_train, n_features, y, spec, free, alpha)
    for i in range(len(r.grad)):
        gp.unsafe_store(i, Float64(r.grad[i]))
    sp.unsafe_store(0, Float64(r.info))
    sp.unsafe_store(1, Float64(r.lml))
    return r.info


def gpr_lml_grad_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`log_marginal_likelihood(theta, eval_gradient=True)` at the kernel
    handed in (DEVIATION 2880). Returns LAPACK's `info`.

    `addrs`, in this exact order (mirrored in `python/mojolearn/_gp_impl.py`
    and `bindings/_mojolearn_gp_host.mojo`):

        0  x               n_train * n_features float32, read
        1  y               n_train float32, read
        2  kinds           n_nodes int32, read
        3  kparams         n_nodes float32, read
        4  ls_len          n_nodes int32, read
        5  ls              max(n_ls, 1) float32, read
        6  free            n_nodes int32 (0 fixed, 1 free), read
        7  grad_out        max(n_free, 1) float64, WRITTEN (float32 widened)
        8  scalars_out     2 float64, WRITTEN: info, lml

    `params`: 0 n_train, 1 n_features, 2 n_nodes, 3 n_ls, 4 alpha.
    """
    if len(addrs) != 9:
        raise Error(
            "gpr_lml_grad: addrs must contain 9 addresses (x, y, kinds,"
            " kparams, ls_len, ls, free, grad_out, scalars_out), got "
            + String(len(addrs))
        )
    if len(params) != 5:
        raise Error(
            "gpr_lml_grad: params must contain 5 values (n_train, n_features,"
            " n_nodes, n_ls, alpha), got "
            + String(len(params))
        )
    var n_train = Int(py=params[0])
    var n_features = Int(py=params[1])
    var n_nodes = Int(py=params[2])
    var n_ls = Int(py=params[3])
    var alpha = Float32(Float64(py=params[4]))
    var spec = _rebuild_kernel_spec(
        Int(py=addrs[2]), Int(py=addrs[3]), Int(py=addrs[4]), Int(py=addrs[5]),
        n_nodes, n_ls, String("gpr_lml_grad"),
    )
    var free = read_i32(Int(py=addrs[6]), max(0, n_nodes))
    var gp = _f64_ptr(Int(py=addrs[7]))
    var sp = _f64_ptr(Int(py=addrs[8]))
    var x = read_f32(Int(py=addrs[0]), max(0, n_train * n_features))
    var y = read_f32(Int(py=addrs[1]), max(0, n_train))
    var info = 0
    with GILReleased(Python()):
        info = _gpr_lml_grad_run(x, y, spec, free, n_train, n_features, alpha, gp, sp)
    return PythonObject(info)


def gp_log64_binding(values: PythonObject) raises -> PythonObject:
    """`identical_log64` over a list of floats: theta of hyperparameter values
    and bounds (DEVIATION 2880)."""
    var out = Python.list()
    for i in range(len(values)):
        out.append(PythonObject(gp_log64(Float64(py=values[i]))))
    return out


def gp_theta_params_binding(values: PythonObject) raises -> PythonObject:
    """`Float32(identical_exp64(theta))` over a list, widened back to Python
    floats: the float32 hyperparameters that run at theta (DEVIATION 2880)."""
    var out = Python.list()
    for i in range(len(values)):
        out.append(PythonObject(Float64(gp_theta_param(Float64(py=values[i])))))
    return out


def gp_restart_uniforms_binding(params: PythonObject) raises -> PythonObject:
    """The restart draws (DEVIATION 2881): `params` = n_restarts, n_dims,
    random_state low 32 bits, high 32 bits; returns n_restarts * n_dims
    doubles in [0, 1), restart r's dimension j at r * n_dims + j."""
    if len(params) != 4:
        raise Error(
            "gp_restart_uniforms: params must contain 4 values (n_restarts,"
            " n_dims, seed_lo, seed_hi), got " + String(len(params))
        )
    var nr = Int(py=params[0])
    var nd = Int(py=params[1])
    var seed = (UInt64(Int(py=params[3])) << 32) | UInt64(Int(py=params[2]))
    var out = Python.list()
    for r in range(nr):
        for j in range(nd):
            out.append(PythonObject(gp_restart_uniform(seed, r, j)))
    return out


def gp_parallel_available() raises -> PythonObject:
    return PythonObject(1)


@export
def PyInit__mojolearn_gp() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_gp")
        m.def_function[gp_parallel_available]("gp_parallel_available")
        m.def_function[gp_vendor_binding]("gp_vendor")
        m.def_function[gp_py2mojo_binding]("gp_py2mojo")
        m.def_function[gp_numeric_mode_binding]("gp_numeric_mode")
        m.def_function[gpr_fit_binding]("gpr_fit")
        m.def_function[gpr_predict_binding]("gpr_predict")
        m.def_function[gpr_sample_y_binding]("gpr_sample_y")
        m.def_function[gpr_predict_cov_binding]("gpr_predict_cov")
        m.def_function[gpr_lml_grad_binding]("gpr_lml_grad")
        m.def_function[gpr_optimize_binding]("gpr_optimize")
        m.def_function[gp_log64_binding]("gp_log64")
        m.def_function[gp_theta_params_binding]("gp_theta_params")
        m.def_function[gp_restart_uniforms_binding]("gp_restart_uniforms")
        m.def_function[gpc_fit_binding]("gpc_fit")
        m.def_function[gpc_predict_binding]("gpc_predict")
        m.def_function[gpc_ovr_combine_binding]("gpc_ovr_combine")
        m.def_function[gp_idn_caps_binding]("gp_idn_caps")
        m.def_function[gpc_fit_all_binding]("gpc_fit_all")
        m.def_function[gpc_predict_all_binding]("gpc_predict_all")
        m.def_function[gpr_fit_ptr_binding]("gpr_fit_ptr")
        m.def_function[gpr_predict_ptr_binding]("gpr_predict_ptr")
        # The Cholesky door (workstream D, 2026-09-14).
        m.def_function[cholesky_parallel_available]("cholesky_parallel_available")
        m.def_function[cholesky_profile_jitter_binding]("cholesky_profile_jitter")
        m.def_function[cholesky_factor_binding]("cholesky_factor")
        m.def_function[cholesky_solve_binding]("cholesky_solve")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_gp: ", e))
