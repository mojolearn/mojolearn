# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE BLOCKED ROW FOLDS of PowerTransformer's lambda search (lane
fam-prep-metrics, 2026-10-04), IDENTICAL on every vendor and the host column.

`pt_fold_unit` and `pt_sfold_unit` (x_prep/transform.mojo) fold all n rows of
a column's transform on ONE thread, twice (the sum, then the squared
deviations), once per evaluation of the search: at the board's 1M rows that
is 2M dependent adds per (column, candidate) per round on a few threads.
Here the rows are cut into x_prep/blocked.mojo's XB-row blocks:

  ptb_part1   one unit per (block, column, candidate): the block's sum of
              the transform and its row count from zero, ascending (and, in
              the starting round, candidate 0's sum J);
  ptb_mean    one unit per (column, candidate): the block partials folded
              ascending from zero; the mean and the count (and, in the
              starting round, sum J and the count into the column's state);
  ptb_part2   one unit per (block, column, candidate): the block's squared
              deviations from that mean from zero, ascending;
  ptb_fin     one unit per (column, candidate): the partials folded
              ascending from zero, then `pt_sfold_unit`'s value;
  ptb_step    the staged search's end instead (one candidate a column): the
              same fold, then `pt_finish`'s golden-section step.

Every per-row operation is `pt_fold_unit`'s (`_pt_take1` / `_pt_take2`'s
tests and arithmetic); only the association of the three sums changes. The
same units run on the host (the program model), so the host column gets the
same order.

BITS. sum T, sum J and the squared deviations take the blocked order for
n > XB (one block at n <= XB: the serial unit's words), so a value of the
search, and with it a lambda, can move in its last bits, on all four columns
together. A blocked sum is never less accurate than the row-order one, and
the variance stays two-pass (deviations from the folded mean), unlike the
one-pass tiles FAST tried (x_prep/fastpt.mojo).

-D MOJOLEARN_IDN_PT_BLOCKED_OFF restores pt_fold / pt_sfold (the switch is
bit 8 of `x_prep_idn_fam`; python/mojolearn/_expansion_prep.py stages these
ops only when it is set).
"""
from std.sys.compile import is_defined
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from x_prep.common import FP, IP, p, ld, raw, sti, ldi, is_nan
from x_prep.prims import add, sub, mul, div, logf
from x_prep.transform import PT_STATE, pt_finish, log1pf
from x_prep.blocked import _span

comptime IDN_PT_BLOCKED = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not (is_defined["MOJOLEARN_IDN_PT_BLOCKED_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]())
)


@always_inline
def _t_base(T: Int, n: Int, M: Int, c: Int, j: Int, il: Int) -> Int:
    """Candidate j of column c: the offset of its row 0 transform
    (`pt_smap_unit`'s layouts; row i is `_t_stride` words further each)."""
    if il != 0:
        return T + c * n * M + j
    return T + (c * M + j) * n


@always_inline
def _t_stride(M: Int, il: Int) -> Int:
    if il != 0:
        return M
    return 1


def ptb_part1_unit(t: Int, f: FP, q: IP):
    """q = [X, n, d, METHOD, T, M, STATE, PS, PC, PJ, nb, FIRST, IL];
    t = (b*d + c)*M + j. Over the block's rows ascending from zero, as
    `pt_fold_unit`'s first pass takes a row: PS[t] = the sum of candidate
    j's transforms and PC[t] = the rows taken (FIRST != 0: the rows whose X
    is not NaN; else every row when STATE[9] == n, otherwise the rows whose
    transform is not NaN). FIRST != 0 and
    j == 0: PJ[b*d + c] = the block's sum J. A constant column (STATE[7] !=
    0) writes nothing."""
    var X = p(q, 0)
    var n = p(q, 1)
    var d = p(q, 2)
    var method = p(q, 3)
    var M = p(q, 5)
    var dm = d * M
    var b = t // dm
    var u = t % dm
    var c = u // M
    var j = u % M
    var S = p(q, 6) + c * PT_STATE
    if raw(f, S + 7) != Float32(0):
        return
    var first = p(q, 11) != 0
    var il = p(q, 12)
    var Tc = _t_base(p(q, 4), n, M, c, j, il)
    var ts = _t_stride(M, il)
    var r = _span(b, n)
    var cnt = 0
    var sm = Float32(0)
    var sj = Float32(0)
    if first:
        var want_j = j == 0
        for i in range(r[0], r[1]):
            var x = ld(f, X + i * d + c)
            if is_nan(x):
                continue
            sm = add(sm, raw(f, Tc + i * ts))
            if want_j:
                if method == 1:
                    sj = add(sj, logf(x))
                elif x >= Float32(0):
                    sj = add(sj, log1pf(x))
                else:
                    sj = sub(sj, log1pf(sub(Float32(0), x)))
            cnt += 1
        if want_j:
            f.unsafe_store(p(q, 9) + b * d + c, sj)
    else:
        # the starting round counted n rows (STATE[9]): no NaN row, every word is added untested
        var no_nan = ldi(f, S + 9) == n
        for i in range(r[0], r[1]):
            var tv = raw(f, Tc + i * ts)
            if not no_nan and is_nan(tv):
                continue
            sm = add(sm, tv)
            cnt += 1
    f.unsafe_store(p(q, 7) + t, sm)
    sti(f, p(q, 8) + t, cnt)


def ptb_mean_unit(t: Int, f: FP, q: IP):
    """q = [PS, PC, PJ, nb, d, M, STATE, MEAN, CNT, FIRST]; t = c*M + j.
    ptb_part1's partials, blocks ascending from zero: CNT[t] = the row count
    (an int32 word), MEAN[t] = the sum over the count (0 when no row).
    FIRST != 0 and j == 0: the column's sum J into STATE[8] and its count
    into STATE[9], where `pt_fold_unit` keeps them."""
    var nb = p(q, 3)
    var d = p(q, 4)
    var M = p(q, 5)
    var dm = d * M
    var c = t // M
    var j = t % M
    var S = p(q, 6) + c * PT_STATE
    if raw(f, S + 7) != Float32(0):
        return
    var cnt = 0
    var sm = Float32(0)
    for b in range(nb):
        cnt += ldi(f, p(q, 1) + b * dm + t)
        sm = add(sm, raw(f, p(q, 0) + b * dm + t))
    var mean = Float32(0)
    if cnt > 0:
        mean = div(sm, Float32(cnt))
    f.unsafe_store(p(q, 7) + t, mean)
    sti(f, p(q, 8) + t, cnt)
    if p(q, 9) != 0 and j == 0:
        var sj = Float32(0)
        for b in range(nb):
            sj = add(sj, raw(f, p(q, 2) + b * d + c))
        f.unsafe_store(S + 8, sj)
        sti(f, S + 9, cnt)


def ptb_part2_unit(t: Int, f: FP, q: IP):
    """q = [T, n, d, M, STATE, MEAN, CNT, PSS, nb, IL]; t = (b*d + c)*M + j.
    Over the block's rows ascending from zero, as `pt_fold_unit`'s second
    pass takes a row (every row when CNT == n, else the rows whose transform
    is not NaN): PSS[t] = the sum of (T - MEAN)^2; zero when CNT == 0."""
    var n = p(q, 1)
    var d = p(q, 2)
    var M = p(q, 3)
    var dm = d * M
    var b = t // dm
    var u = t % dm
    var c = u // M
    var j = u % M
    var S = p(q, 4) + c * PT_STATE
    if raw(f, S + 7) != Float32(0):
        return
    var il = p(q, 9)
    var Tc = _t_base(p(q, 0), n, M, c, j, il)
    var ts = _t_stride(M, il)
    var cnt = ldi(f, p(q, 6) + u)
    var mean = raw(f, p(q, 5) + u)
    var r = _span(b, n)
    var ss = Float32(0)
    if cnt == n and n > 0:
        for i in range(r[0], r[1]):
            var e = sub(raw(f, Tc + i * ts), mean)
            ss = add(ss, mul(e, e))
    elif cnt > 0:
        for i in range(r[0], r[1]):
            var tv = raw(f, Tc + i * ts)
            if is_nan(tv):
                continue
            var e = sub(tv, mean)
            ss = add(ss, mul(e, e))
    f.unsafe_store(p(q, 7) + t, ss)


@always_inline
def _ss_fold(f: FP, PSS: Int, nb: Int, dm: Int, u: Int) -> Float32:
    """ptb_part2's partials of (column, candidate) u, blocks ascending from zero."""
    var ss = Float32(0)
    for b in range(nb):
        ss = add(ss, raw(f, PSS + b * dm + u))
    return ss


def ptb_fin_unit(t: Int, f: FP, q: IP):
    """q = [PSS, nb, d, M, STATE, SPL, VALS, CNT]; t = c*M + j.
    `pt_sfold_unit`'s end: VALS[t] = the negative log-likelihood at candidate
    j's point SPL[t], from the folded squared deviations, the count CNT[t]
    and the column's sum J (STATE[8])."""
    var M = p(q, 3)
    var c = t // M
    var S = p(q, 4) + c * PT_STATE
    if raw(f, S + 7) != Float32(0):
        return
    var cnt = ldi(f, p(q, 7) + t)
    var ss = _ss_fold(f, p(q, 0), p(q, 1), p(q, 2) * M, t)
    var sj = raw(f, S + 8)
    var lam = raw(f, p(q, 5) + t)
    var val = Float32(0)
    if cnt > 0:
        var var_ = div(ss, Float32(cnt))
        val = sub(mul(mul(Float32(0.5), Float32(cnt)), logf(var_)), mul(sub(lam, Float32(1)), sj))
    f.unsafe_store(p(q, 6) + t, val)


def ptb_step_unit(t: Int, f: FP, q: IP):
    """q = [X, n, d, METHOD, T, K, STATE, LEVAL, LAMBDA, PSS, nb, CNT];
    t = column (the staged search: one candidate a column, M = 1). q[0..8]
    is `pt_fold_unit`'s layout: the squared deviations folded from
    ptb_part2's partials, then `pt_finish`'s value and golden-section step."""
    var c = t
    var S = p(q, 6) + c * PT_STATE
    if raw(f, S + 7) != Float32(0):
        return
    var cnt = ldi(f, p(q, 11) + c)
    var ss = _ss_fold(f, p(q, 9), p(q, 10), p(q, 2), c)
    pt_finish(t, f, q, cnt, raw(f, S + 8), ss)
