# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Typed host entries of `bindings/_mojolearn_x_linear_host.mojo`.

The binding unpacks PythonObject arguments and calls one function here; the
input checks, target preparation, fit dispatch and decision loops live here
with no Python import, moved word for word (lane thin-core, 2026-10-21: a
pure move, no bit change)."""
from experiments.classical_identical_ideas.linear_controls import C13_LOGCV_WEIGHTS, ENETCV_FOLD_BLOCKS
from x_linear.enetcv_blocks import fb_host_words
from x_linear.ops import FP, IP
from x_linear.dispatch import fit_dispatch, decision_one, decision_code_row, team_rows, team_own, ALGO_ISOTONIC, ALGO_LOGCV, ALGO_RIDGE, isotonic_abi_check
from x_linear.dispatch import ALGO_GLM
from x_linear.glm_ydom import XLIN_GLM_DEV_YDOM, GLM_YDOM_REFUSED, glm_ydom_host
from x_linear.logcv import logcv_fold_ids, lcv_caller_folds
from x_linear.isotonic_host import isotonic_fit_host
from x_linear.team import team_work, solo


def _fp(addr: Int) raises -> FP:
    if addr == 0:
        raise Error("x_linear: null float32 buffer address")
    return FP(unsafe_from_address=addr)


def _finite(p: FP, count: Int, name: String) raises:
    """The input check both columns run before a fit: NaN or infinity is refused by name."""
    for i in range(count):
        var v = p.unsafe_load(i)
        if not (v == v) or v > Float32(3.4028234e38) or v < Float32(-3.4028234e38):
            raise Error(String("mojolearn: ", name, " contains NaN or infinity"))


def x_linear_fit_host(
    a: Int, x_addr: Int, y_addr: Int, dims: List[Int],
    mut ipl: List[Int32], mut fpl: List[Float32], out_addr: Int,
) raises -> Int:
    """The host fit: `dims` = [n, d, n_x, n_y, n_out, n_fw, n_iw, n_ip, n_fp];
    `ipl`/`fpl` hold the n_ip / n_fp parameter words. Returns n_out."""
    var n = dims[0]
    var d = dims[1]
    var n_out = dims[4]
    var n_fw = dims[5]
    var n_iw = dims[6]
    var n_ip = dims[7]
    var n_fp = dims[8]
    if n <= 0 or d < 0 or n_out <= 0:
        raise Error("x_linear: positive dimensions required")
    isotonic_abi_check(a, n, dims[3], n_out, n_fw, n_iw, n_ip)
    comptime if ENETCV_FOLD_BLOCKS:
        if a == 9 and n_ip >= 4:
            n_fw += fb_host_words(d, Int(ipl[3]))
    comptime if C13_LOGCV_WEIGHTS:
        if a == ALGO_LOGCV:
            n_fw += Int(ipl[4]) + 1
    var fw = List[Float32](length=max(n_fw, 1), fill=Float32(0))
    var iw = List[Int32](length=max(n_iw, 1), fill=Int32(0))
    var bufs = team_rows(a, IP(unsafe_from_address=Int(ipl.unsafe_ptr())))
    var own = team_own(a, d)
    var tw = List[Float32](length=team_work(n, bufs, own), fill=Float32(0))
    var x = _fp(x_addr)
    var y = _fp(y_addr)
    comptime if C13_LOGCV_WEIGHTS:
        # the opt-in C13 weight sums assume disjoint folds (x_linear/cv_roles.mojo)
        if a == ALGO_LOGCV and n > 0 and dims[3] > n and y[n] < Float32(0):
            raise Error("x_linear: MOJOLEARN_CLASSICAL_C13_LOGCV_WEIGHTS sums disjoint folds; a cv that is not a partition (ShuffleSplit, overlapping or partial splits) is refused under it")
    var out = _fp(out_addr)
    _finite(x, dims[2], "X")
    _finite(y, dims[3], "y")
    for i in range(n_out):
        out.unsafe_store(i, Float32(0))
    # lane fam2-linear: the GLM targets' range, the device grid's rule
    # (x_linear/glm_ydom.mojo); a refused fit returns -1 in the converged word
    comptime if XLIN_GLM_DEV_YDOM:
        if a == ALGO_GLM and n_fp > 0 and n_out >= d + 3 and glm_ydom_host(y, n, fpl[0]):
            out.unsafe_store(d + 2, GLM_YDOM_REFUSED)
            return n_out
    # LogisticRegressionCV: the caller's fold ids are zeros; its StratifiedKFold
    # ids are built here from the labels (cgr-linear), into a copy of y
    var ycopy = List[Float32]()
    if a == ALGO_LOGCV:
        var ny = dims[3]
        ycopy = List[Float32](length=max(ny, 1), fill=Float32(0))
        for i in range(ny):
            ycopy[i] = y.unsafe_load(i)
        y = FP(unsafe_from_address=Int(ycopy.unsafe_ptr()))
        # an explicit cv's fold ids came in y (ip[6] = 1): kept as sent
        if not lcv_caller_folds(n_ip, Int(ipl[6]) if n_ip > 6 else 0):
            logcv_fold_ids(y, n, IP(unsafe_from_address=Int(ipl.unsafe_ptr())))
    # RidgeClassifier: ip[4] == 1 means y holds the n int32 class codes (then
    # the n weights when ip[3]); the +-1 targets as the device builds them
    # (`c1_codes_targets_kernel`), the weights after them
    if a == ALGO_RIDGE and n_ip > 4 and Int(ipl[4]) == 1:
        var t_c = Int(ipl[0])
        var has_sw = Int(ipl[3]) != 0
        ycopy = List[Float32](length=n * t_c + (n if has_sw else 0), fill=Float32(0))
        var yi = IP(unsafe_from_address=Int(y))
        for i in range(n):
            var c = Int(yi.unsafe_load(i))
            if t_c == 1:
                ycopy[i] = Float32(1) if c == 1 else Float32(-1)
            else:
                for t in range(t_c):
                    ycopy[i * t_c + t] = Float32(1) if c == t else Float32(-1)
        if has_sw:
            for i in range(n):
                ycopy[n * t_c + i] = y.unsafe_load(n + i)
        y = FP(unsafe_from_address=Int(ycopy.unsafe_ptr()))
    if a == ALGO_ISOTONIC:
        isotonic_fit_host(
            x, y, n, d,
            IP(unsafe_from_address=Int(ipl.unsafe_ptr())), FP(unsafe_from_address=Int(fpl.unsafe_ptr())),
            out, FP(unsafe_from_address=Int(fw.unsafe_ptr())), IP(unsafe_from_address=Int(iw.unsafe_ptr())),
        )
    else:
        fit_dispatch(
            solo(FP(unsafe_from_address=Int(tw.unsafe_ptr())), n, bufs, own), a, x, y, n, d,
            IP(unsafe_from_address=Int(ipl.unsafe_ptr())), FP(unsafe_from_address=Int(fpl.unsafe_ptr())),
            out, FP(unsafe_from_address=Int(fw.unsafe_ptr())), IP(unsafe_from_address=Int(iw.unsafe_ptr())),
        )
    _ = fw^
    _ = iw^
    _ = tw^
    _ = ycopy^
    return n_out


def x_linear_decision_host(
    x_addr: Int, wb_addr: Int, n: Int, d: Int, k: Int, link: Int, out_addr: Int
) raises -> Int:
    """n x k decision cells (`decision_one`) into out. Returns n * k."""
    if n < 0 or d < 0 or k <= 0:
        raise Error("x_linear: positive dimensions required")
    var x = _fp(x_addr)
    var wb = _fp(wb_addr)
    var out = _fp(out_addr)
    for t in range(n * k):
        out.unsafe_store(t, decision_one(x, t // k, d, wb, t % k, link))
    return n * k


def x_linear_decision_codes_host(
    x_addr: Int, wb_addr: Int, n: Int, d: Int, k: Int, link: Int,
    strict: Int, below: Int, above: Int, out_addr: Int,
) raises -> Int:
    """The same `decision_one` cells, then `decision_code_row` per row. Returns n."""
    if n < 0 or d < 0 or k <= 0:
        raise Error("x_linear: positive dimensions required")
    var x = _fp(x_addr)
    var wb = _fp(wb_addr)
    var out = IP(unsafe_from_address=out_addr)
    var s = List[Float32](length=max(n * k, 1), fill=Float32(0))
    var sp = s.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    for t in range(n * k):
        sp.unsafe_store(t, decision_one(x, t // k, d, wb, t % k, link))
    for i in range(n):
        out.unsafe_store(i, decision_code_row(sp, i, k, strict, below, above))
    _ = s^
    return n
