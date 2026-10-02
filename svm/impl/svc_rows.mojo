# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""SVC's per-row binary64 arithmetic, shared by the device kernels
(`svm/impl/svc_epilogue.mojo`) and the CPU host column
(`svm/host/svc_proba.mojo`) so both compute the same words (cgfin-c-svm,
2026-10-02). Integer instructions only (`checks/soft_f64.mojo`); no device
API here, so the CPU binding compiles it without a GPU. The module docstring
of `svc_epilogue.mojo` says what changed and why.
"""

from std.memory import bitcast

from checks.soft_f64 import (
    SF64_SIGN,
    SF64_ZERO,
    SF64_INF,
    sf64_add,
    sf64_sub,
    sf64_mul,
    sf64_div,
    sf64_fma,
    sf64_lt,
    sf64_floor,
    sf64_to_int,
    sf64_from_int,
    sf64_from_f32,
    sf64_to_f32,
    sf64_log,
    sf64_is_nan,
    sf64_neg,
)

#: one fold chunk: the block width of every svc fold launch, and the chunk
#: of the host twin (`tree_sum_sf64`); the same shape as `grid_fold.mojo`
comptime FOLD_TPB = 256


@always_inline
def fold_blocks(n: Int) -> Int:
    """The chunks of one fold level over `n` cells."""
    return (n + FOLD_TPB - 1) // FOLD_TPB


comptime U64P = MutPointer[UInt64, MutAnyOrigin]
comptime U32P = MutPointer[UInt32, MutAnyOrigin]
comptime I32P = MutPointer[Int32, MutAnyOrigin]

#: the modes of the `svc_pair_epilogue` door (the first five are the
#: epilogues; 6..8 the row glue, multiplexed on the same door so the CPU
#: binding exports the same names)
comptime EPI_OVO = 0
comptime EPI_OVR = 1
comptime EPI_VOTES = 2
comptime EPI_PROBA = 3
comptime EPI_LOG_PROBA = 4
comptime EPI_BINARY_CODES = 5
comptime GLUE_GATHER = 6
comptime GLUE_SELECT = 7
comptime GLUE_C_ROWS = 8

#: per-row status: 0 fine, 1 a zero divisor, 2 a log of a value <= 0
comptime ST_OK = 0
comptime ST_DIV0 = 1
comptime ST_DOMAIN = 2

#: Platt fold modes and their channel counts
comptime PLATT_COUNT = 0  # 1 channel: the +1 labels
comptime PLATT_FVAL = 1  # 1 channel: the objective
comptime PLATT_NEWTON = 2  # 5 channels: h11, h22, h21, g1, g2 sums

# binary64 constants (bit patterns; the decimal each one rounds is beside it)
comptime _ONE = UInt64(0x3FF0000000000000)  # 1.0
comptime _TWO = UInt64(0x4000000000000000)  # 2.0
comptime _THREE = UInt64(0x4008000000000000)  # 3.0
comptime _HALF = UInt64(0x3FE0000000000000)  # 0.5
comptime _EXP_HI = UInt64(0x40862E42FEFA39EF)  # 709.782712893384
comptime _EXP_LO = UInt64(0xC086232BDD7ABCD2)  # -708.3964185322641
comptime _LOG2E = UInt64(0x3FF71547652B82FE)  # 1.4426950408889634
comptime _C1 = UInt64(0xBFE62E4000000000)  # -6.93145751953125e-1
comptime _C2 = UInt64(0xBEB7F7D1CF79ABCA)  # -1.42860682030941723212e-6
comptime _P0 = UInt64(0x3F2089CDD5E44BE8)  # 1.26177193074810590878e-4
comptime _P1 = UInt64(0x3F9F06D10CCA2C7E)  # 3.02994407707441961300e-2
comptime _P2 = UInt64(0x3FF0000000000000)  # 9.99999999999999999910e-1
comptime _Q0 = UInt64(0x3EC92EB6BC365FA0)  # 3.00198505138664455042e-6
comptime _Q1 = UInt64(0x3F64AE39B508B6C0)  # 2.52448340349684104192e-3
comptime _Q2 = UInt64(0x3FCD17099887E074)  # 2.27265548208155028766e-1
comptime _Q3 = UInt64(0x4000000000000000)  # 2.00000000000000000009e0
comptime _CLO = UInt64(0x3E7AD7F29ABCAF48)  # 1e-7
comptime _CHI = UInt64(0x3FEFFFFFCA501ACB)  # 1.0 - 1e-7
comptime _SIGMA = UInt64(0x3D719799812DEA11)  # 1e-12
comptime _EPS = UInt64(0x3EE4F8B588E368F1)  # 1e-5
comptime _MINSTEP = UInt64(0x3DDB7CDFD9D7BDBB)  # 1e-10
comptime _ARMIJO = UInt64(0x3F1A36E2EB1C432D)  # 0.0001
comptime _EPS005 = UInt64(0x3F747AE147AE147B)  # 0.005
comptime _F32_ONE = UInt32(0x3F800000)


# ------------------------------------------------------- binary64 helpers
@always_inline
def sp_abs(x: UInt64) -> UInt64:
    return x & ~SF64_SIGN


@always_inline
def sp_is_zero(x: UInt64) -> Bool:
    return (x & ~SF64_SIGN) == 0


@always_inline
def sp_lt(a: UInt64, b: UInt64) -> Bool:
    """`a < b` with IEEE NaN semantics (a NaN compares false)."""
    if sf64_is_nan(a) or sf64_is_nan(b):
        return False
    return sf64_lt(a, b)


@always_inline
def sp_gt(a: UInt64, b: UInt64) -> Bool:
    return sp_lt(b, a)


@always_inline
def sp_ge0(x: UInt64) -> Bool:
    """`x >= 0.0`."""
    return not sf64_is_nan(x) and not sf64_lt(x, SF64_ZERO)


@always_inline
def sp_le0(x: UInt64) -> Bool:
    """`x <= 0.0`."""
    return not sf64_is_nan(x) and not sf64_lt(SF64_ZERO, x)


@always_inline
def sp_eq(a: UInt64, b: UInt64) -> Bool:
    """`a == b` (NaN unequal, +0 == -0)."""
    if sf64_is_nan(a) or sf64_is_nan(b):
        return False
    return a == b or (sp_is_zero(a) and sp_is_zero(b))


@always_inline
def sp_of_f32_bits(u: UInt32) -> UInt64:
    """The exact widening of a float32 given by its bits."""
    return sf64_from_f32(bitcast[DType.float32](u))


def sp_exp(x: UInt64) -> UInt64:
    """`pm_exp` (`mojolearn_exp` of portable_math.c), statement for
    statement: `x * log2(e)` and `+ 0.5` round separately, the explicit
    `fm` sites are fused. `sf64_exp` fuses the first pair; this does not."""
    if sf64_is_nan(x):
        return x
    if sp_gt(x, _EXP_HI):
        return SF64_INF
    if sp_lt(x, _EXP_LO):
        return SF64_ZERO
    var k = sf64_floor(sf64_add(sf64_mul(x, _LOG2E), _HALF))
    var r = sf64_fma(k, _C1, x)
    r = sf64_fma(k, _C2, r)
    var xx = sf64_mul(r, r)
    var p = sf64_fma(_P0, xx, _P1)
    p = sf64_fma(p, xx, _P2)
    p = sf64_mul(p, r)
    var q = sf64_fma(_Q0, xx, _Q1)
    q = sf64_fma(q, xx, _Q2)
    q = sf64_fma(q, xx, _Q3)
    var y = sf64_div(p, sf64_sub(q, p))
    y = sf64_fma(_TWO, y, _ONE)
    var ki = sf64_to_int(k)
    var k1 = ki >> 1
    var k2 = ki - k1
    y = sf64_mul(y, UInt64(k1 + 1023) << 52)
    return sf64_mul(y, UInt64(k2 + 1023) << 52)


@always_inline
def sp_sigmoid_predict(dec: UInt64, a: UInt64, b: UInt64) -> UInt64:
    """libsvm's P(+1) at one decision value (`sigmoid_predict`)."""
    var fapb = sf64_add(sf64_mul(dec, a), b)
    if sp_ge0(fapb):
        var e = sp_exp(sf64_neg(fapb))
        return sf64_div(e, sf64_add(_ONE, e))
    return sf64_div(_ONE, sf64_add(_ONE, sp_exp(fapb)))


@always_inline
def sp_clamp(v: UInt64) -> UInt64:
    """`min(max(v, 1e-7), 1 - 1e-7)` with CPython's builtin semantics."""
    var w = v
    if sp_gt(_CLO, w):
        w = _CLO
    if sp_lt(_CHI, w):
        w = _CHI
    return w


# -------------------------------------------------------- the epilogues
@always_inline
def multiclass_probability(k: Int, r: U64P, p: U64P, q: U64P, qp: U64P) -> Int:
    """libsvm's `multiclass_probability` (Wu, Lin and Weng, method 2):
    r is k x k, p (k) is written, q (k x k) and qp (k) are scratch. Returns
    ST_DIV0 where the Python raised `float division by zero`."""
    var kf = sf64_from_int(k)
    for t in range(k):
        p[t] = sf64_div(_ONE, kf)
    for t in range(k * k):
        q[t] = SF64_ZERO
    for t in range(k):
        for j in range(t):
            q[t * k + t] = sf64_add(q[t * k + t], sf64_mul(r[j * k + t], r[j * k + t]))
            q[t * k + j] = q[j * k + t]
        for j in range(t + 1, k):
            q[t * k + t] = sf64_add(q[t * k + t], sf64_mul(r[j * k + t], r[j * k + t]))
            q[t * k + j] = sf64_mul(sf64_neg(r[j * k + t]), r[t * k + j])
    var eps = sf64_div(_EPS005, kf)
    for t in range(k):
        qp[t] = SF64_ZERO
    var iters = max(100, k)
    for _ in range(iters):
        var pqp = SF64_ZERO
        for t in range(k):
            qp[t] = SF64_ZERO
            for j in range(k):
                qp[t] = sf64_add(qp[t], sf64_mul(q[t * k + j], p[j]))
            pqp = sf64_add(pqp, sf64_mul(p[t], qp[t]))
        var max_error = SF64_ZERO
        for t in range(k):
            var err = sp_abs(sf64_sub(qp[t], pqp))
            if sp_gt(err, max_error):
                max_error = err
        if sp_lt(max_error, eps):
            break
        for t in range(k):
            var qtt = q[t * k + t]
            if sp_is_zero(qtt):
                return ST_DIV0
            var diff = sf64_div(sf64_add(sf64_neg(qp[t]), pqp), qtt)
            p[t] = sf64_add(p[t], diff)
            var one = sf64_add(_ONE, diff)
            if sp_is_zero(one):
                return ST_DIV0
            var inner = sf64_add(
                pqp, sf64_mul(diff, sf64_add(sf64_mul(diff, qtt), sf64_mul(_TWO, qp[t])))
            )
            pqp = sf64_div(sf64_div(inner, one), one)
            for j in range(k):
                qp[j] = sf64_div(sf64_add(qp[j], sf64_mul(diff, q[t * k + j])), one)
                p[j] = sf64_div(p[j], one)
    return ST_OK


def epilogue_scratch(mode: Int, k: Int) -> Int:
    """Scratch words one row of `mode` needs."""
    if mode == EPI_OVR:
        return 2 * k
    if mode == EPI_VOTES:
        return k
    if mode == EPI_PROBA or mode == EPI_LOG_PROBA:
        return 2 * k * k + k
    return 0


@always_inline
def epilogue_row(
    mode: Int, dec: U32P, n: Int, n_pairs: Int, k: Int, pi: I32P, ab: U64P,
    label1: UInt64, row: Int, scr: U64P, out32: U32P, out64: U64P,
) -> Int:
    """One row of `SVC.decision_function`, `predict`, `predict_proba` or
    `predict_log_proba` (the device kernel and the host column both call
    this). `dec` is n_pairs x n float32 bits, pair-major, each pair
    machine's raw decision (>= 0 toward class j); `pi` the pairs' (i, j)
    class codes. Returns a status (ST_*)."""
    if mode == EPI_OVO:
        # scikit-learn's orientation: each pair's decision negated (a sign
        # flip on the bits, exact on every column)
        for pr in range(n_pairs):
            out32[row * n_pairs + pr] = dec[pr * n + row] ^ UInt32(0x80000000)
        return ST_OK
    if mode == EPI_BINARY_CODES:
        out64[row] = UInt64(1) if sp_eq(sp_of_f32_bits(dec[row]), label1) else UInt64(0)
        return ST_OK
    if mode == EPI_OVR:
        # `_ovr_scores`: votes plus confidences squashed, in pair order
        var votes = scr
        var conf = scr + k
        for c in range(k):
            votes[c] = SF64_ZERO
            conf[c] = SF64_ZERO
        for pr in range(n_pairs):
            var v = sp_of_f32_bits(dec[pr * n + row])
            var i = Int(pi[2 * pr])
            var j = Int(pi[2 * pr + 1])
            conf[i] = sf64_sub(conf[i], v)
            conf[j] = sf64_add(conf[j], v)
            if sp_gt(v, SF64_ZERO):
                votes[j] = sf64_add(votes[j], _ONE)
            else:
                votes[i] = sf64_add(votes[i], _ONE)
        for c in range(k):
            var den = sf64_mul(_THREE, sf64_add(sp_abs(conf[c]), _ONE))
            out64[row * k + c] = sf64_add(votes[c], sf64_div(conf[c], den))
        return ST_OK
    if mode == EPI_VOTES:
        # libsvm's vote: most votes, ties to the lowest class; `dec >= 0`
        # read from the bits (no float compare, so no flush on any column)
        for c in range(k):
            scr[c] = UInt64(0)
        for pr in range(n_pairs):
            var u = dec[pr * n + row]
            var mag = u & UInt32(0x7FFFFFFF)
            var ge0 = mag <= UInt32(0x7F800000) and ((u >> 31) == 0 or mag == 0)
            var c2 = Int(pi[2 * pr + 1]) if ge0 else Int(pi[2 * pr])
            scr[c2] = scr[c2] + 1
        var best = 0
        for c in range(1, k):
            if scr[c] > scr[best]:
                best = c
        out64[row] = UInt64(best)
        return ST_OK
    # EPI_PROBA / EPI_LOG_PROBA: each pair's sigmoid of the NEGATED
    # decision (libsvm's orientation), clamped; two classes read m[0][1],
    # m[1][0]; more couple
    var m = scr
    var q = scr + k * k
    var qp = scr + 2 * k * k
    var dst = out64 + row * k
    for t in range(k * k):
        m[t] = SF64_ZERO
    for pr in range(n_pairs):
        var i = Int(pi[2 * pr])
        var j = Int(pi[2 * pr + 1])
        var d = sf64_neg(sp_of_f32_bits(dec[pr * n + row]))
        var v = sp_clamp(sp_sigmoid_predict(d, ab[2 * pr], ab[2 * pr + 1]))
        m[i * k + j] = v
        m[j * k + i] = sf64_sub(_ONE, v)
    if k == 2:
        dst[0] = m[1]
        dst[1] = m[2]
    else:
        var st = multiclass_probability(k, m, dst, q, qp)
        if st != ST_OK:
            return st
    if mode == EPI_LOG_PROBA:
        for c in range(k):
            var v = dst[c]
            if sp_le0(v):
                return ST_DOMAIN
            dst[c] = sf64_log(v)
    return ST_OK


# ------------------------------------------------------------ Platt terms
@always_inline
def platt_channels(mode: Int) -> Int:
    return 5 if mode == PLATT_NEWTON else 1


@always_inline
def platt_terms(
    mode: Int, d: UInt64, label: UInt64, a: UInt64, b: UInt64, hi_t: UInt64, lo_t: UInt64
) -> SIMD[DType.uint64, 8]:
    """One row's addends of libsvm's `sigmoid_train` sums (labels +1/-1):
    PLATT_COUNT the +1 indicator; PLATT_FVAL the objective term; PLATT_NEWTON
    the Hessian and gradient terms (h11, h22, h21, g1, g2)."""
    var out = SIMD[DType.uint64, 8](0)
    var pos = sp_gt(label, SF64_ZERO)
    if mode == PLATT_COUNT:
        out[0] = _ONE if pos else SF64_ZERO
        return out
    var t = hi_t if pos else lo_t
    var fapb = sf64_add(sf64_mul(d, a), b)
    if mode == PLATT_FVAL:
        if sp_ge0(fapb):
            out[0] = sf64_add(
                sf64_mul(t, fapb), sf64_log(sf64_add(_ONE, sp_exp(sf64_neg(fapb))))
            )
        else:
            out[0] = sf64_add(
                sf64_mul(sf64_sub(t, _ONE), fapb), sf64_log(sf64_add(_ONE, sp_exp(fapb)))
            )
        return out
    var p: UInt64
    var q: UInt64
    if sp_ge0(fapb):
        var e = sp_exp(sf64_neg(fapb))
        p = sf64_div(e, sf64_add(_ONE, e))
        q = sf64_div(_ONE, sf64_add(_ONE, e))
    else:
        var e = sp_exp(fapb)
        p = sf64_div(_ONE, sf64_add(_ONE, e))
        q = sf64_div(e, sf64_add(_ONE, e))
    var d2 = sf64_mul(p, q)
    var d1 = sf64_sub(t, p)
    out[0] = sf64_mul(sf64_mul(d, d), d2)
    out[1] = d2
    out[2] = sf64_mul(d, d2)
    out[3] = sf64_mul(d, d1)
    out[4] = d1
    return out


def tree_sum_sf64(vals: List[UInt64]) -> UInt64:
    """The fold the kernels above run, for one channel on the host: the same
    pairs in the same order (the host column's twin)."""
    var cur = vals.copy()
    if len(cur) == 0:
        return SF64_ZERO
    while True:
        var nb = fold_blocks(len(cur))
        var nxt = List[UInt64](length=nb, fill=SF64_ZERO)
        var s = List[UInt64](length=FOLD_TPB, fill=SF64_ZERO)
        for blk in range(nb):
            for t in range(FOLD_TPB):
                var i = blk * FOLD_TPB + t
                s[t] = cur[i] if i < len(cur) else SF64_ZERO
            var step = FOLD_TPB // 2
            while step > 0:
                for t in range(step):
                    s[t] = sf64_add(s[t], s[t + step])
                step //= 2
            nxt[blk] = s[0]
        if nb == 1:
            return nxt[0]
        cur = nxt^


trait PlattSums:
    def platt_sums(
        mut self, mode: Int, a: UInt64, b: UInt64, hi_t: UInt64, lo_t: UInt64
    ) raises -> SIMD[DType.uint64, 8]:
        """The fixed-order sums of `platt_terms` over the rows."""
        ...


def platt_solve[T: PlattSums](mut src: T, n: Int) raises -> Tuple[UInt64, UInt64]:
    """libsvm's `sigmoid_train` (Platt's method, Lin, Lin and Weng's Newton
    iteration with backtracking) over sums `src` forms: the scalar control
    (two parameters) between the row folds. Returns (A, B) as binary64 bits."""
    var prior1 = src.platt_sums(PLATT_COUNT, SF64_ZERO, SF64_ZERO, SF64_ZERO, SF64_ZERO)[0]
    var prior0 = sf64_sub(sf64_from_int(n), prior1)
    var hi_t = sf64_div(sf64_add(prior1, _ONE), sf64_add(prior1, _TWO))
    var lo_t = sf64_div(_ONE, sf64_add(prior0, _TWO))
    var a = SF64_ZERO
    var bl = sf64_div(sf64_add(prior0, _ONE), sf64_add(prior1, _ONE))
    if sp_le0(bl):
        raise Error("math domain error")
    var b = sf64_log(bl)
    var fval = src.platt_sums(PLATT_FVAL, a, b, hi_t, lo_t)[0]
    for _ in range(100):
        var s = src.platt_sums(PLATT_NEWTON, a, b, hi_t, lo_t)
        var h11 = sf64_add(_SIGMA, s[0])
        var h22 = sf64_add(_SIGMA, s[1])
        var h21 = s[2]
        var g1 = s[3]
        var g2 = s[4]
        if sp_lt(sp_abs(g1), _EPS) and sp_lt(sp_abs(g2), _EPS):
            break
        var det = sf64_sub(sf64_mul(h11, h22), sf64_mul(h21, h21))
        if sp_is_zero(det):
            raise Error("float division by zero")
        var da = sf64_div(sf64_neg(sf64_sub(sf64_mul(h22, g1), sf64_mul(h21, g2))), det)
        var db = sf64_div(sf64_neg(sf64_add(sf64_mul(sf64_neg(h21), g1), sf64_mul(h11, g2))), det)
        var gd = sf64_add(sf64_mul(g1, da), sf64_mul(g2, db))
        var step = _ONE
        while not sp_lt(step, _MINSTEP):
            var na = sf64_add(a, sf64_mul(step, da))
            var nb = sf64_add(b, sf64_mul(step, db))
            var newf = src.platt_sums(PLATT_FVAL, na, nb, hi_t, lo_t)[0]
            if sp_lt(newf, sf64_add(fval, sf64_mul(sf64_mul(_ARMIJO, step), gd))):
                a = na
                b = nb
                fval = newf
                break
            step = sf64_div(step, _TWO)
        if sp_lt(step, _MINSTEP):
            break
    return (a, b)


# ------------------------------------------------------------ shuffle, glue
@always_inline
def shuffle_seed32(seed: UInt64) -> UInt32:
    """The Feistel key: the low 32 bits of one SplitMix64 step of `seed`."""
    var z = seed + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    z ^= z >> 31
    return UInt32(z & UInt64(0xFFFFFFFF))


@always_inline
def gather_cell(
    cell: Int, src: U32P, n_src_rows: Int, n_src_cols: Int, rows: I32P, has_rows: Bool,
    cols: I32P, has_cols: Bool, n_out_cols: Int, out: U32P,
) -> Int:
    """`out[r, c] = src[rows[r], cols[c]]` for one output cell (bits, no
    arithmetic). Returns ST_DIV0 on an index outside src."""
    var r = cell // n_out_cols
    var c = cell - r * n_out_cols
    var sr = Int(rows[r]) if has_rows else r
    var sc = Int(cols[c]) if has_cols else c
    if sr < 0 or sr >= n_src_rows or sc < 0 or sc >= n_src_cols:
        return ST_DIV0
    out[cell] = src[sr * n_src_cols + sc]
    return ST_OK


@always_inline
def c_row(
    i: Int, sw: U64P, has_sw: Bool, codes: I32P, cw: U64P, has_cw: Bool, k: Int,
    c: UInt64, out: U32P,
) -> Int:
    """One row's bound `C * (w_i * class_weight[y_i])`, binary64, rounded
    once to float32 (`_c_rows`)."""
    var w = sw[i] if has_sw else _ONE
    if has_cw:
        var code = Int(codes[i])
        if code < 0 or code >= k:
            return ST_DIV0
        w = sf64_mul(w, cw[code])
    out[i] = bitcast[DType.uint32](sf64_to_f32(sf64_mul(c, w)))
    return ST_OK
