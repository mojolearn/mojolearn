# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""SVC's binary64 host epilogues in Mojo (lane/py-dn-svm, 2026-09-28).

Until this module, `python/mojolearn/_svm_impl.py` ran these in Python, one
row and one pair at a time: libsvm's `sigmoid_predict`, the [1e-7, 1 - 1e-7]
clamp and the pairwise coupling (`multiclass_probability`) of
`predict_proba`, the one-vs-one votes of `predict`, scikit-learn's
`_ovr_decision_function` of `decision_function(shape='ovr')`, the ovo
transpose, and at fit time libsvm's `sigmoid_train` (Platt's method, Lin,
Lin and Weng's Newton iteration) and the SplitMix64 shuffle. Those Python
functions stay in `_svm_impl.py` as the REFERENCE the pod job compares this
module against; the estimator no longer calls them.

THE SAME BITS AS THE PYTHON IT REPLACES. Every value is binary64, every
loop runs in the Python loop's order, and every product that feeds an add or
a subtract is `pinned_mul_f64` (CPython rounds `a * b + c` twice; a fused
multiply add would round once). `exp` and `log` are `pm_exp` and `pm_log`
below: a line for line transcription of `mojolearn_exp` and `mojolearn_log`
in packaging/portable_math/portable_math.c, the library `_portable_math`
calls. They are NOT `checks/numerics.mojo`'s `portable_exp64`, whose
reduction `k` is one fused rounding where the C (built -ffp-contract=off)
rounds `x * log2(e)` and `+ 0.5` separately. The explicit `fm` sites of the
C are `fma` here, which rounds once exactly as the C's hardware fma does.

HOST ONLY: no DeviceContext, compiled into `_mojolearn_svm` and
`_mojolearn_svm_host` alike, so the GPU column and the CPU column run the
same instructions. Rows are independent, so the row loops are split across
host tasks by shape only; no task's bits depend on the split.
"""
from std.math import fma
from std.memory import bitcast
from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from bindings.hostptr import f32_ptr, f64_ptr, i32_ptr
from checks.numerics import pinned_mul_f64
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_chunk, host_predict_task_count

comptime F64Ptr = MutPointer[Float64, MutUntrackedOrigin]
comptime F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime I64P = MutPointer[Int64, MutUntrackedOrigin]

#: the epilogue modes of `svc_pair_epilogue`
comptime EPI_OVO = 0
comptime EPI_OVR = 1
comptime EPI_VOTES = 2
comptime EPI_PROBA = 3
comptime EPI_LOG_PROBA = 4
comptime EPI_BINARY_CODES = 5


@always_inline
def _pm(a: Float64, b: Float64) -> Float64:
    return pinned_mul_f64(a, b)


@always_inline
def _bits(x: Float64) -> UInt64:
    return bitcast[DType.uint64](x)


@always_inline
def _val(u: UInt64) -> Float64:
    return bitcast[DType.float64](u)


def pm_exp(x: Float64) -> Float64:
    """`mojolearn_exp` of portable_math.c, line for line."""
    if x != x:
        return x
    if x > 709.782712893384:
        return _val(UInt64(0x7FF0000000000000))
    if x < -708.3964185322641:
        return Float64(0.0)
    var t = _pm(x, 1.4426950408889634) + 0.5
    var ki = Int(t)
    if Float64(ki) > t:
        ki -= 1
    var k = Float64(ki)
    var r = fma(k, -6.93145751953125e-1, x)
    r = fma(k, -1.42860682030941723212e-6, r)
    var xx = _pm(r, r)
    var p = fma(1.26177193074810590878e-4, xx, 3.02994407707441961300e-2)
    p = fma(p, xx, 9.99999999999999999910e-1)
    p = _pm(p, r)
    var q = fma(3.00198505138664455042e-6, xx, 2.52448340349684104192e-3)
    q = fma(q, xx, 2.27265548208155028766e-1)
    q = fma(q, xx, 2.00000000000000000009e0)
    var y = p / (q - p)
    y = fma(2.0, y, 1.0)
    var k1 = ki >> 1
    var k2 = ki - k1
    y = _pm(y, _val(UInt64(k1 + 1023) << 52))
    return _pm(y, _val(UInt64(k2 + 1023) << 52))


def pm_log(xin: Float64) -> Float64:
    """`mojolearn_log` of portable_math.c (with its `log_fraction`), line
    for line. NaN, 0, negative and +inf are its special cases."""
    if xin != xin:
        return xin
    if xin == 0.0:
        return _val(UInt64(0xFFF0000000000000))
    if xin < 0.0:
        return _val(UInt64(0x7FF8000000000000))
    if _bits(xin) == UInt64(0x7FF0000000000000):
        return xin
    # raw_e: the exponent before the mantissa normalization adjusts it
    var u = _bits(xin)
    var raw_e = 0
    if (u >> 52) == UInt64(0):
        u = _bits(_pm(xin, 18014398509481984.0))
        raw_e = -54
    raw_e += Int((u >> 52) & UInt64(0x7FF)) - 1022
    # log_fraction
    var inp = xin
    u = _bits(inp)
    var e = 0
    if (u >> 52) == UInt64(0):
        inp = _pm(inp, 18014398509481984.0)
        u = _bits(inp)
        e = -54
    e += Int((u >> 52) & UInt64(0x7FF)) - 1022
    var m = _val((u & UInt64(0x000FFFFFFFFFFFFF)) | UInt64(0x3FE0000000000000))
    var z: Float64
    var y: Float64
    var x: Float64
    if e > 2 or e < -2:
        if m < 0.70710678118654752440:
            e -= 1
            z = m - 0.5
            y = fma(0.5, z, 0.5)
        else:
            z = m - 0.5
            z = z - 0.5
            y = fma(0.5, m, 0.5)
        x = z / y
        z = _pm(x, x)
        var r = fma(-7.89580278884799154124e-1, z, 1.63866645699558079767e1)
        r = fma(r, z, -6.41409952958715622951e1)
        var q = z + -3.56722798256324312549e1
        q = fma(q, z, 3.12093766372244180303e2)
        q = fma(q, z, -7.69691943550460008604e2)
        y = _pm(x, _pm(z, r) / q)
    else:
        if m < 0.70710678118654752440:
            e -= 1
            x = fma(2.0, m, -1.0)
        else:
            x = m - 1.0
        z = _pm(x, x)
        var p = fma(1.01875663804580931796e-4, x, 4.97494994976747001425e-1)
        p = fma(p, x, 4.70579119878881725854e0)
        p = fma(p, x, 1.44989225341610930846e1)
        p = fma(p, x, 1.79368678507819816313e1)
        p = fma(p, x, 7.70838733755885391666e0)
        var q = x + 1.12873587189167450590e1
        q = fma(q, x, 4.52279145837532221105e1)
        q = fma(q, x, 8.29875266912776603211e1)
        q = fma(q, x, 7.11544750618563894466e1)
        q = fma(q, x, 2.31251620126765340583e1)
        y = _pm(x, _pm(z, p) / q)
    # mojolearn_log after log_fraction
    var fe = Float64(e)
    y = fma(fe, -2.121944400546905827679e-4, y)
    if not (raw_e > 2 or raw_e < -2):
        y = fma(_pm(x, x), -0.5, y)
    y = y + x
    return fma(fe, 0.693359375, y)


@always_inline
def sigmoid_predict(dec: Float64, a: Float64, b: Float64) -> Float64:
    """`_svm_impl._sigmoid_predict`: libsvm's P(+1) at one decision value."""
    var fapb = _pm(dec, a) + b
    if fapb >= 0.0:
        var e = pm_exp(-fapb)
        return e / (1.0 + e)
    return 1.0 / (1.0 + pm_exp(fapb))


@always_inline
def _clamp(v: Float64) -> Float64:
    """`min(max(v, 1e-7), 1.0 - 1e-7)` with CPython's builtin semantics
    (max keeps the first argument unless the second is greater; min keeps
    the first unless the second is smaller), so NaN passes through."""
    var w = v
    if 1e-7 > w:
        w = 1e-7
    var hi = 1.0 - 1e-7
    if hi < w:
        w = hi
    return w


def _div(a: Float64, b: Float64) raises -> Float64:
    """CPython's float division: a zero divisor raises."""
    if b == 0.0:
        raise Error("float division by zero")
    return a / b


def multiclass_probability(k: Int, r: F64Ptr, p: F64Ptr, q: F64Ptr, qp: F64Ptr) raises:
    """`_svm_impl._multiclass_probability` (libsvm, Wu, Lin and Weng's method
    2): r is k x k, p (k) is written, q (k x k) and qp (k) are scratch."""
    for t in range(k):
        p[t] = 1.0 / Float64(k)
    for t in range(k * k):
        q[t] = 0.0
    for t in range(k):
        for j in range(t):
            q[t * k + t] = q[t * k + t] + _pm(r[j * k + t], r[j * k + t])
            q[t * k + j] = q[j * k + t]
        for j in range(t + 1, k):
            q[t * k + t] = q[t * k + t] + _pm(r[j * k + t], r[j * k + t])
            q[t * k + j] = _pm(-r[j * k + t], r[t * k + j])
    var eps = 0.005 / Float64(k)
    for t in range(k):
        qp[t] = 0.0
    var iters = max(100, k)
    for _ in range(iters):
        var pqp = 0.0
        for t in range(k):
            qp[t] = 0.0
            for j in range(k):
                qp[t] = qp[t] + _pm(q[t * k + j], p[j])
            pqp = pqp + _pm(p[t], qp[t])
        var max_error = 0.0
        for t in range(k):
            var err = abs(qp[t] - pqp)
            if err > max_error:
                max_error = err
        if max_error < eps:
            break
        for t in range(k):
            var diff = _div(-qp[t] + pqp, q[t * k + t])
            p[t] = p[t] + diff
            var one = 1.0 + diff
            pqp = _div(_div(pqp + _pm(diff, _pm(diff, q[t * k + t]) + _pm(2.0, qp[t])), one), one)
            for j in range(k):
                qp[j] = _div(qp[j] + _pm(diff, q[t * k + j]), one)
                p[j] = _div(p[j], one)


def _row_proba(
    dec: F32P, n: Int, n_pairs: Int, k: Int, pi: I32P, ab: F64Ptr, row: Int,
    m: F64Ptr, q: F64Ptr, qp: F64Ptr, dst: F64Ptr,
) raises:
    """One row of `SVC.predict_proba` (before this lane, its Python loop):
    each pair's sigmoid of the NEGATED decision (libsvm's orientation),
    clamped, into m; two classes read m[0][1], m[1][0]; more couple."""
    for t in range(k * k):
        m[t] = 0.0
    for pr in range(n_pairs):
        var i = Int(pi[2 * pr])
        var j = Int(pi[2 * pr + 1])
        var v = sigmoid_predict(-Float64(dec[pr * n + row]), ab[2 * pr], ab[2 * pr + 1])
        v = _clamp(v)
        m[i * k + j] = v
        m[j * k + i] = 1.0 - v
    if k == 2:
        dst[0] = m[1]
        dst[1] = m[2]
        return
    multiclass_probability(k, m, dst, q, qp)


def pair_epilogue(
    mode: Int, dec: F32P, n: Int, n_pairs: Int, k: Int, pi: I32P, ab: F64Ptr,
    label1: Float64, out_addr: Int,
) raises:
    """The row loops of `SVC.decision_function`, `predict`, `predict_proba`
    and `predict_log_proba`. `dec` is n_pairs x n float32, pair-major, each
    pair machine's raw decision in this solver's orientation (>= 0 toward
    class j); `pi` the pairs' (i, j) class codes."""
    var tasks = host_predict_task_count(n)
    if tasks < 1:
        tasks = 1
    if tasks > n:
        tasks = n
    var chunk = host_predict_chunk(n, tasks)
    var failed = List[Int](length=tasks, fill=0)
    var fp = failed.unsafe_ptr()

    def _rows(c: Int) {imm mode, imm dec, imm n, imm n_pairs, imm k, imm pi, imm ab, imm label1, imm out_addr, imm chunk, imm fp}:
        try:
            var lo = c * chunk
            var hi = min(lo + chunk, n)
            if mode == EPI_OVO:
                # sklearn orientation: each pair's decision negated (exact)
                var o = F32P(unsafe_from_address=out_addr)
                for r in range(lo, hi):
                    for pr in range(n_pairs):
                        o[r * n_pairs + pr] = -dec[pr * n + r]
            elif mode == EPI_OVR:
                # `_ovr_scores`: votes plus confidences squashed, pair order
                var o = F64Ptr(unsafe_from_address=out_addr)
                var votes = List[Float64](length=k, fill=0.0)
                var conf = List[Float64](length=k, fill=0.0)
                for r in range(lo, hi):
                    for c2 in range(k):
                        votes[c2] = 0.0
                        conf[c2] = 0.0
                    for pr in range(n_pairs):
                        var v = Float64(dec[pr * n + r])
                        var i = Int(pi[2 * pr])
                        var j = Int(pi[2 * pr + 1])
                        conf[i] = conf[i] - v
                        conf[j] = conf[j] + v
                        if v > 0.0:
                            votes[j] = votes[j] + 1.0
                        else:
                            votes[i] = votes[i] + 1.0
                    for c2 in range(k):
                        o[r * k + c2] = votes[c2] + conf[c2] / _pm(3.0, abs(conf[c2]) + 1.0)
            elif mode == EPI_VOTES:
                # libsvm's vote: most votes, ties to the lowest class
                var o = I64P(unsafe_from_address=out_addr)
                var votes = List[Int](length=k, fill=0)
                for r in range(lo, hi):
                    for c2 in range(k):
                        votes[c2] = 0
                    for pr in range(n_pairs):
                        if dec[pr * n + r] >= Float32(0.0):
                            votes[Int(pi[2 * pr + 1])] += 1
                        else:
                            votes[Int(pi[2 * pr])] += 1
                    var best = 0
                    for c2 in range(1, k):
                        if votes[c2] > votes[best]:
                            best = c2
                    o[r] = Int64(best)
            elif mode == EPI_BINARY_CODES:
                # the device's class label back to a code (1 == label1)
                var o = I64P(unsafe_from_address=out_addr)
                for r in range(lo, hi):
                    o[r] = Int64(1) if Float64(dec[r]) == label1 else Int64(0)
            else:
                var o = F64Ptr(unsafe_from_address=out_addr)
                var m = List[Float64](length=k * k, fill=0.0)
                var q = List[Float64](length=k * k, fill=0.0)
                var qp = List[Float64](length=k, fill=0.0)
                for r in range(lo, hi):
                    _row_proba(dec, n, n_pairs, k, pi, ab, r, m.unsafe_ptr(), q.unsafe_ptr(),
                               qp.unsafe_ptr(), o + r * k)
                    if mode == EPI_LOG_PROBA:
                        for c2 in range(k):
                            var v = o[r * k + c2]
                            if v <= 0.0:
                                # `_portable_math.log`'s refusal
                                raise Error("math domain error")
                            o[r * k + c2] = pm_log(v)
        except:
            fp[c] = 1

    if tasks == 1:
        _rows(0)
    else:
        host_parallelize(_rows, tasks)
    for c in range(tasks):
        if failed[c] != 0:
            raise Error("svc epilogue: a row raised (float division by zero or math domain error)")


def platt_fval(dec: F64Ptr, t: F64Ptr, n: Int, a: Float64, b: Float64) -> Float64:
    """`_svm_impl._platt_fval`, sequential in row order."""
    var f = 0.0
    for i in range(n):
        var fapb = _pm(dec[i], a) + b
        if fapb >= 0.0:
            f = f + (_pm(t[i], fapb) + pm_log(1.0 + pm_exp(-fapb)))
        else:
            f = f + (_pm(t[i] - 1.0, fapb) + pm_log(1.0 + pm_exp(fapb)))
    return f


def sigmoid_train(dec: F64Ptr, labels: F64Ptr, n: Int) raises -> Tuple[Float64, Float64]:
    """`_svm_impl._sigmoid_train` (libsvm's `sigmoid_train`), its loop order
    and its roundings; labels are +1 / -1."""
    var c1 = 0
    for i in range(n):
        if labels[i] > 0.0:
            c1 += 1
    var prior1 = Float64(c1)
    var prior0 = Float64(n) - prior1
    comptime max_iter = 100
    comptime min_step = 1e-10
    comptime sigma = 1e-12
    comptime eps = 1e-5
    var hi_t = _div(prior1 + 1.0, prior1 + 2.0)
    var lo_t = _div(1.0, prior0 + 2.0)
    var t = List[Float64](length=n, fill=0.0)
    for i in range(n):
        t[i] = hi_t if labels[i] > 0.0 else lo_t
    var tp = t.unsafe_ptr()
    var a = 0.0
    var bl = _div(prior0 + 1.0, prior1 + 1.0)
    if bl <= 0.0:
        raise Error("math domain error")
    var b = pm_log(bl)
    var fval = platt_fval(dec, tp, n, a, b)
    for _ in range(max_iter):
        var h11 = sigma
        var h22 = sigma
        var h21 = 0.0
        var g1 = 0.0
        var g2 = 0.0
        for i in range(n):
            var d = dec[i]
            var fapb = _pm(d, a) + b
            var p: Float64
            var q: Float64
            if fapb >= 0.0:
                var e = pm_exp(-fapb)
                p = e / (1.0 + e)
                q = 1.0 / (1.0 + e)
            else:
                var e = pm_exp(fapb)
                p = 1.0 / (1.0 + e)
                q = e / (1.0 + e)
            var d2 = _pm(p, q)
            h11 = h11 + _pm(_pm(d, d), d2)
            h22 = h22 + d2
            h21 = h21 + _pm(d, d2)
            var d1 = tp[i] - p
            g1 = g1 + _pm(d, d1)
            g2 = g2 + d1
        if abs(g1) < eps and abs(g2) < eps:
            break
        var det = _pm(h11, h22) - _pm(h21, h21)
        var da = _div(-(_pm(h22, g1) - _pm(h21, g2)), det)
        var db = _div(-(_pm(-h21, g1) + _pm(h11, g2)), det)
        var gd = _pm(g1, da) + _pm(g2, db)
        var step = 1.0
        while step >= min_step:
            var na = a + _pm(step, da)
            var nb = b + _pm(step, db)
            var newf = platt_fval(dec, tp, n, na, nb)
            if newf < fval + _pm(_pm(0.0001, step), gd):
                a = na
                b = nb
                fval = newf
                break
            step = step / 2.0
        if step < min_step:
            break
    return (a, b)


def splitmix_perm(n: Int, seed: UInt64, dst: I32P):
    """`_svm_impl._splitmix_perm`: libsvm's shuffle with a SplitMix64
    stream (integer arithmetic, wrapping at 2^64)."""
    for i in range(n):
        dst[i] = Int32(i)
    var state = seed
    for i in range(n):
        state = state + UInt64(0x9E3779B97F4A7C15)
        var z = state
        z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
        z ^= z >> 31
        var j = i + Int(z % UInt64(n - i))
        var s = dst[i]
        dst[i] = dst[j]
        dst[j] = s


# ------------------------------------------------------------ Python doors
def _ix(v: PythonObject) raises -> Int:
    var x = Int(py=v)
    if x < 0:
        raise Error("svc epilogue: negative size")
    return x


def svc_pair_epilogue_binding(
    dec_addr: PythonObject, pairs_addr: PythonObject, ab_addr: PythonObject,
    out_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """params: [mode, n_rows, n_pairs, n_classes, label1]. Modes: 0 ovo
    (float32 n x P), 1 ovr (float64 n x K), 2 vote codes (int64 n), 3
    probabilities (float64 n x K; ab = 2P float64 (A, B) per pair), 4 their
    logs, 5 binary codes from the device's labels (int64 n). Returns n."""
    if len(params) != 5:
        raise Error("svc_pair_epilogue: params must contain 5 values")
    var mode = _ix(params[0])
    var n = _ix(params[1])
    var n_pairs = _ix(params[2])
    var k = _ix(params[3])
    var label1 = Float64(py=params[4])
    if mode > 5:
        raise Error("svc_pair_epilogue: unknown mode")
    if n == 0:
        return PythonObject(0)
    if n_pairs < 1 or k < 2:
        raise Error("svc_pair_epilogue: needs a pair and two classes")
    var dec = f32_ptr(Int(py=dec_addr))
    var pi = i32_ptr(Int(py=pairs_addr)) if mode != EPI_BINARY_CODES else I32P(unsafe_from_address=1)
    var ab = f64_ptr(Int(py=ab_addr)) if mode == EPI_PROBA or mode == EPI_LOG_PROBA else F64Ptr(unsafe_from_address=1)
    var dst = Int(py=out_addr)
    if dst == 0:
        raise Error("svc_pair_epilogue: null output")
    if mode != EPI_BINARY_CODES:
        for pr in range(n_pairs):
            var i = Int(pi[2 * pr])
            var j = Int(pi[2 * pr + 1])
            if i < 0 or j < 0 or i >= k or j >= k or i == j:
                raise Error("svc_pair_epilogue: invalid class pair")
    with GILReleased(Python()):
        pair_epilogue(mode, dec, n, n_pairs, k, pi, ab, label1, dst)
    return PythonObject(n)


def svc_platt_train_binding(
    dec_addr: PythonObject, labels_addr: PythonObject, out_addr: PythonObject, n: PythonObject
) raises -> PythonObject:
    """libsvm's `sigmoid_train` over n float64 decision values and +1/-1
    float64 labels; writes (A, B) as two float64 at out_addr. Returns n."""
    var count = _ix(n)
    var dp = f64_ptr(Int(py=dec_addr)) if count else F64Ptr(unsafe_from_address=1)
    var lp = f64_ptr(Int(py=labels_addr)) if count else F64Ptr(unsafe_from_address=1)
    var op = f64_ptr(Int(py=out_addr))
    with GILReleased(Python()):
        var r = sigmoid_train(dp, lp, count)
        op[0] = r[0]
        op[1] = r[1]
    return PythonObject(count)


def svc_splitmix_perm_binding(
    out_addr: PythonObject, n: PythonObject, seed_lo: PythonObject, seed_hi: PythonObject
) raises -> PythonObject:
    """`_splitmix_perm(n, seed)` into n int32 at out_addr, the 64-bit seed
    handed in as two 32-bit halves. Returns n."""
    var count = _ix(n)
    if count == 0:
        return PythonObject(0)
    if count > 2147483647:
        raise Error("svc_splitmix_perm: n exceeds int32")
    var lo = UInt64(_ix(seed_lo)) & UInt64(0xFFFFFFFF)
    var hi = UInt64(_ix(seed_hi)) & UInt64(0xFFFFFFFF)
    var op = i32_ptr(Int(py=out_addr))
    with GILReleased(Python()):
        splitmix_perm(count, (hi << 32) | lo, op)
    return PythonObject(count)


def svc_portable_math_binding(in_addr: PythonObject, out_addr: PythonObject, n: PythonObject, which: PythonObject) raises -> PythonObject:
    """THE TWIN'S OWN CHECK DOOR: `pm_exp` (which 0) or `pm_log` (which 1)
    over n float64, so a job can hold them to `_portable_math.exp/log` (the
    C library) bit for bit on any sweep. Returns n."""
    var count = _ix(n)
    if count == 0:
        return PythonObject(0)
    var w = Int(py=which)
    var ip = f64_ptr(Int(py=in_addr))
    var op = f64_ptr(Int(py=out_addr))
    with GILReleased(Python()):
        for i in range(count):
            op[i] = pm_exp(ip[i]) if w == 0 else pm_log(ip[i])
    return PythonObject(count)
