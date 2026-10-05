# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""PCA `n_components='mle'`: Minka's MLE of the rank (scikit-learn
`_pca.py::_infer_dimension` / `_assess_dimension`) in software binary64
(checks/soft_f64.mojo) on the device, every vendor (lane cpu3-python,
2026-10-04, orchestrator decision: the rank choice left Python and its
float64 host sums).

Every quantity is IEEE double computed with integer instructions only, so
NVIDIA, AMD, Apple and the host column (`pca_mle_rank_host`, the same
per-index functions in index order) write the same words. Bits moved from
the Python form (float32 cell logs and lgammas, Python double sums): the
logs and lgammas are now double, so the log-likelihoods are closer to
scikit-learn's, never further.

The spectrum `sp` (d >= 2 values, descending) arrives as binary64 words,
two float32 words each, low first (array('d') bytes). Phases, one launch
each, in order on one context:

  0. thread k < d: lsp[k] = log(max(sp[k], TINY)) and
     lg[k] = lgamma((d - k) / 2) (lgamma(1) = 0, lgamma(1/2) = log(pi) / 2,
     then lgamma(x + 1) = lgamma(x) + log(x), ascending).
  1. thread r in [1, d): pu, pl, v and base = pu + pl + pv + pp, scikit-
     learn's expressions in its order (one thread per rank, O(d) each).
  2. thread (r, t), t < MLE_P: the cross-term partial of rank r over the
     rows i = t, t + MLE_P, ... < r, j ascending: acc += log(max(term,
     TINY)) + log(n).
  3. thread r: pa = the MLE_P partials folded in t order;
     ll[r] = base - pa / 2 - r * log(n) / 2 (-inf when sp[r - 1] < 1e-15;
     ll[0] = -inf).
  4. thread t < MLE_P: the first maximum of ll over r = t, t + MLE_P, ...
  5. one thread: the first maximum of those MLE_P candidates (a fold of
     MLE_P values, bounded); the rank, or 0 when no ll is above -inf (the
     argmax of an all -inf vector, as numpy), as an exact float.

TINY (the float32 smallest normal) keeps the floor the Python form had on
its logs; v is floored at 1e-15 as scikit-learn does."""
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast

from checks.soft_f64 import (
    SF64_INF, SF64_NAN, SF64_SIGN, SF64_ZERO, SF64_ONE, sf64_add, sf64_sub, sf64_mul, sf64_div, sf64_log,
    sf64_lt, sf64_gt, sf64_is_nan, sf64_from_int, sf64_neg,
)
from x_decomp.cells import F32Ptr

#: cross-term partials per rank (phase 2) and argmax partials (phase 4)
comptime MLE_P = 128

comptime _M_LOG2 = UInt64(0x3FE62E42FEFA39EF)  # 0.6931471805599453
comptime _M_LOGPI = UInt64(0x3FF250D048E7A1BD)  # 1.1447298858494002
comptime _M_LOG2PI = UInt64(0x3FFD67F1C864BEB4)  # 1.8378770664093453
comptime _M_TINY = UInt64(0x3810000000000000)  # 1.1754943508222875e-38
comptime _M_EPS = UInt64(0x3CD203AF9EE75616)  # 1e-15
comptime _M_LGHALF = UInt64(0x3FE250D048E7A1BD)  # lgamma(0.5) = log(pi) / 2
comptime _M_HALF = UInt64(0x3FE0000000000000)  # 0.5
comptime _M_TWO = UInt64(0x4000000000000000)  # 2.0
comptime _M_NINF = SF64_INF | SF64_SIGN


# scratch layout (float32 words; a binary64 value is two words, low first)
@always_inline
def mle_off_lsp(d: Int) -> Int:
    return 0


@always_inline
def mle_off_lg(d: Int) -> Int:
    return 2 * d


@always_inline
def mle_off_v(d: Int) -> Int:
    return 4 * d


@always_inline
def mle_off_base(d: Int) -> Int:
    return 6 * d


@always_inline
def mle_off_ll(d: Int) -> Int:
    return 8 * d


@always_inline
def mle_off_part(d: Int) -> Int:
    return 10 * d


@always_inline
def mle_off_cand(d: Int) -> Int:
    return 10 * d + 2 * d * MLE_P


@always_inline
def mle_scratch_words(d: Int) -> Int:
    """Float32 words of scratch the phases use (candidates: value, index)."""
    return 10 * d + 2 * d * MLE_P + 4 * MLE_P


@always_inline
def _ld(f: F32Ptr, at: Int) -> UInt64:
    return (UInt64(bitcast[DType.uint32](f.unsafe_load(at + 1))) << UInt64(32)) | UInt64(
        bitcast[DType.uint32](f.unsafe_load(at))
    )


@always_inline
def _st(f: F32Ptr, at: Int, v: UInt64):
    f.unsafe_store(at, bitcast[DType.float32](UInt32(v & UInt64(0xFFFFFFFF))))
    f.unsafe_store(at + 1, bitcast[DType.float32](UInt32(v >> UInt64(32))))


@always_inline
def _logf(x: UInt64) -> UInt64:
    """log(max(x, TINY)); NaN stays NaN (the cell form's floor)."""
    if sf64_is_nan(x):
        return SF64_NAN
    if sf64_lt(x, _M_TINY):
        return sf64_log(_M_TINY)
    return sf64_log(x)


@always_inline
def _better(a: UInt64, ai: Int, b: UInt64, bi: Int) -> Bool:
    """Whether (a, ai) is the earlier maximum against (b, bi): larger, or
    equal at a lower index. NaN never wins (numpy's argmax would take it;
    the Python form's `ll > best` never did)."""
    if sf64_is_nan(a):
        return False
    if sf64_is_nan(b):
        return True
    if sf64_gt(a, b):
        return True
    if sf64_lt(a, b):
        return False
    return ai < bi


@always_inline
def mle_prep_at(sp: F32Ptr, d: Int, k: Int, s: F32Ptr):
    """Phase 0 at index k < d."""
    _st(s, mle_off_lsp(d) + 2 * k, _logf(_ld(sp, 2 * k)))
    var m = d - k                       # lgamma(m / 2), m >= 1
    var acc = SF64_ZERO if (m & 1) == 0 else _M_LGHALF
    var q = 2 if (m & 1) == 0 else 1    # x = q / 2 steps up by 1 to m / 2
    while q < m:
        acc = sf64_add(acc, sf64_log(sf64_mul(sf64_from_int(q), _M_HALF)))
        q += 2
    _st(s, mle_off_lg(d) + 2 * k, acc)


@always_inline
def mle_base_at(sp: F32Ptr, d: Int, n: Int, r: Int, s: F32Ptr):
    """Phase 1 at rank r in [1, d): v[r] and base[r] = pu + pl + pv + pp."""
    var fr = sf64_from_int(r)
    var pu = sf64_neg(sf64_mul(fr, _M_LOG2))
    for i in range(1, r + 1):
        var c = sf64_div(sf64_mul(_M_LOGPI, sf64_from_int(d - i + 1)), _M_TWO)
        pu = sf64_add(pu, sf64_sub(_ld(s, mle_off_lg(d) + 2 * (i - 1)), c))
    var sl = SF64_ZERO
    for i in range(r):
        sl = sf64_add(sl, _ld(s, mle_off_lsp(d) + 2 * i))
    var fn_ = sf64_from_int(n)
    var pl = sf64_div(sf64_mul(sf64_neg(sl), fn_), _M_TWO)
    var sv = SF64_ZERO
    for j in range(r, d):
        sv = sf64_add(sv, _ld(sp, 2 * j))
    var v = sf64_div(sv, sf64_from_int(d - r))
    if sf64_is_nan(v) or sf64_lt(v, _M_EPS):
        v = _M_EPS
    var pv = sf64_div(sf64_mul(sf64_mul(sf64_neg(sf64_log(v)), fn_), sf64_from_int(d - r)), _M_TWO)
    # m = d * rank - rank * (rank + 1.0) / 2.0; pp = log(2 pi) * (m + rank) / 2.0
    var mm = sf64_sub(sf64_from_int(d * r), sf64_div(sf64_mul(fr, sf64_add(fr, SF64_ONE)), _M_TWO))
    var pp = sf64_div(sf64_mul(_M_LOG2PI, sf64_add(mm, fr)), _M_TWO)
    _st(s, mle_off_v(d) + 2 * r, v)
    _st(s, mle_off_base(d) + 2 * r, sf64_add(sf64_add(sf64_add(pu, pl), pv), pp))


@always_inline
def mle_part_at(sp: F32Ptr, d: Int, n: Int, r: Int, t: Int, s: F32Ptr):
    """Phase 2 at (rank r, partial t): rows i = t, t + MLE_P, ... < r."""
    var logn = sf64_log(sf64_from_int(n))
    var v = _ld(s, mle_off_v(d) + 2 * r)
    var iv = sf64_div(SF64_ONE, v)
    var acc = SF64_ZERO
    var i = t
    while i < r:
        var si = _ld(sp, 2 * i)
        var ivi = sf64_div(SF64_ONE, si)
        for j in range(i + 1, d):
            var sj = _ld(sp, 2 * j)
            var ivj = sf64_div(SF64_ONE, sj) if j < r else iv
            var term = sf64_mul(sf64_sub(si, sj), sf64_sub(ivj, ivi))
            acc = sf64_add(acc, sf64_add(_logf(term), logn))
        i += MLE_P
    _st(s, mle_off_part(d) + 2 * (r * MLE_P + t), acc)


@always_inline
def mle_ll_at(sp: F32Ptr, d: Int, n: Int, r: Int, s: F32Ptr):
    """Phase 3 at rank r in [0, d)."""
    var ll = _M_NINF
    if r > 0 and not sf64_lt(_ld(sp, 2 * (r - 1)), _M_EPS):
        var pa = SF64_ZERO
        for t in range(MLE_P):  # small-loop(MLE_P = 128: cross-term partials of one rank): fixed fold of the partials
            pa = sf64_add(pa, _ld(s, mle_off_part(d) + 2 * (r * MLE_P + t)))
        var logn = sf64_log(sf64_from_int(n))
        ll = sf64_sub(
            sf64_sub(_ld(s, mle_off_base(d) + 2 * r), sf64_div(pa, _M_TWO)),
            sf64_div(sf64_mul(sf64_from_int(r), logn), _M_TWO),
        )
    _st(s, mle_off_ll(d) + 2 * r, ll)


@always_inline
def mle_cand_at(d: Int, t: Int, s: F32Ptr):
    """Phase 4 at partial t: the first maximum over ranks t, t + MLE_P, ..."""
    var best = _M_NINF
    var bi = 0
    var r = t
    while r < d:
        var x = _ld(s, mle_off_ll(d) + 2 * r)
        if _better(x, r, best, bi) and sf64_gt(x, _M_NINF):
            best = x
            bi = r
        r += MLE_P
    _st(s, mle_off_cand(d) + 4 * t, best)
    _st(s, mle_off_cand(d) + 4 * t + 2, UInt64(bi))


@always_inline
def mle_pick(d: Int, s: F32Ptr) -> Int:
    """Phase 5: the first maximum of the MLE_P candidates; 0 when every ll
    is -inf (Python's `ll > best` from -inf never fired)."""
    var best = _M_NINF
    var bi = 0
    for t in range(MLE_P):  # small-loop(MLE_P = 128: argmax candidates): fixed final fold of the per-partial maxima
        var x = _ld(s, mle_off_cand(d) + 4 * t)
        var xi = Int(_ld(s, mle_off_cand(d) + 4 * t + 2))
        if sf64_gt(x, _M_NINF) and _better(x, xi, best, bi):
            best = x
            bi = xi
    return bi


# ------------------------------------------------------------ kernels ----
def mle_prep_kernel(sp: F32Ptr, d: Int32, s: F32Ptr):
    var k = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if k < Int(d):
        mle_prep_at(sp, Int(d), k, s)


def mle_base_kernel(sp: F32Ptr, d: Int32, n: Int32, s: F32Ptr):
    var r = 1 + Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if r < Int(d):
        mle_base_at(sp, Int(d), Int(n), r, s)


def mle_part_kernel(sp: F32Ptr, d: Int32, n: Int32, s: F32Ptr):
    var g = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var r = 1 + g // MLE_P
    if r < Int(d):
        mle_part_at(sp, Int(d), Int(n), r, g % MLE_P, s)


def mle_ll_kernel(sp: F32Ptr, d: Int32, n: Int32, s: F32Ptr):
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if r < Int(d):
        mle_ll_at(sp, Int(d), Int(n), r, s)


def mle_cand_kernel(d: Int32, s: F32Ptr):
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < MLE_P:
        mle_cand_at(Int(d), t, s)


def mle_pick_kernel(d: Int32, s: F32Ptr, dst: F32Ptr):
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t == 0:
        dst.unsafe_store(0, Float32(mle_pick(Int(d), s)))


# -------------------------------------------------------- host column ----
def pca_mle_rank_host(sp: F32Ptr, d: Int, n: Int, s: F32Ptr) -> Int:
    """The host column: the device phases' per-index functions, each phase
    over its indices in order (every value is a pure function of earlier
    phases, so the order inside a phase does not change a word)."""
    if d < 2:
        return 0
    for k in range(d):
        mle_prep_at(sp, d, k, s)
    for r in range(1, d):
        mle_base_at(sp, d, n, r, s)
    for r in range(1, d):
        for t in range(MLE_P):
            mle_part_at(sp, d, n, r, t, s)
    for r in range(d):
        mle_ll_at(sp, d, n, r, s)
    for t in range(MLE_P):
        mle_cand_at(d, t, s)
    return mle_pick(d, s)
