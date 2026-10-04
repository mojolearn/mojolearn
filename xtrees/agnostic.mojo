# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The model-agnostic explainers' UNITS (lane cgr2-metrics-shap, 2026-10-03).

KernelExplainer and PermutationExplainer walked the explained rows one by
one in Python: coalition masks and permutations built as Python lists, the
synthetic data expanded, the background means, the weighted least squares
and the marginal sums all serial host loops. Here every step is a UNIT, a
plain function of a work item `t`: xtrees/agnostic_device.mojo launches one
thread per unit over a CHUNK of rows at once (as many rows as fit the
synthetic budget), xtrees/agnostic_host.mojo (the CPU column) runs the same
units in ascending `t`. Only the model call stays in Python, once per chunk:
the model is the caller's function.

Every float64 step is `checks/soft_f64.mojo` (correctly rounded binary64 on
integer words, the Apple GPU having no float64), so the device writes the
words the host's IEEE double unit writes: the background means are the
row-order sums of `block_mean`, the logit link is `identical_log64`'s
`portable_log64`, the normal equations, the elimination with partial
pivoting (first largest pivot) and the back substitution are
`kernel_solve`'s operations in its order, and the permutation marginals are
the Python loop's adds in its order.

The draws are the counter RNG of xtrees/ops.mojo (`stream_base(seed, row)`,
`draw`, `unit`): the permutations are the same Fisher-Yates walks as before.
KernelExplainer's sampled coalitions changed (cuML `kernel_dataset`'s
schedule): sample pairs (a draw, then its complement), no de-duplication,
each sample weighted weight_left / samples_left; the full-subset part is
shap's enumeration, unranked per sample.
"""
from checks.soft_f64 import (
    SF64_ZERO, SF64_ONE, SF64_SIGN, sf64_add, sf64_sub, sf64_mul, sf64_div, sf64_lt, sf64_gt, sf64_from_int,
    sf64_from_f32, sf64_to_int, sf64_log,
)
from std.memory import bitcast
from xtrees.ops import draw, stream_base

comptime F32P = MutPointer[Float32, MutAnyOrigin]
comptime I32P = MutPointer[Int32, MutAnyOrigin]
comptime U64P = MutPointer[UInt64, MutAnyOrigin]
comptime I64P = MutPointer[Int64, MutAnyOrigin]

#: 2^-53, `unit`'s scale
comptime _TWO_M53 = UInt64(0x3CA0000000000000)
#: synthetic words per chunk (the rows of one chunk share one model call)
comptime AGN_BUDGET = 1 << 25


@always_inline
def u_unit(r: UInt64) -> UInt64:
    """xtrees/ops.mojo `unit` as binary64 bits: (r >> 11) * 2^-53, exact."""
    return sf64_mul(sf64_from_int(Int(r >> 11)), _TWO_M53)


@always_inline
def sf_zero(a: UInt64) -> Bool:
    return (a & ~SF64_SIGN) == 0


@always_inline
def sf_abs(a: UInt64) -> UInt64:
    return a & ~SF64_SIGN


@always_inline
def fy_index(base: UInt64, k: Int, a: Int) -> Int:
    """Python's int(unit(draw(base, k)) * (a + 1)): the Fisher-Yates index."""
    return sf64_to_int(sf64_mul(u_unit(draw(base, k)), sf64_from_int(a + 1)))


def binom(n: Int, k: Int) -> Int:
    """C(n, k) (exact while C(n, k) * n < 2^63; the schedule only asks for
    counts below its sample budget)."""
    if k < 0 or k > n:
        return 0
    var kk = min(k, n - k)
    var c = 1
    for i in range(1, kk + 1):
        c = c * (n - kk + i) // i
    return c


# --------------------------------------------------------------- Kernel SHAP
def kshap_mask_unit(t: Int, M: Int, m: Int, nfixed: Int, nfull: Int, npaired: Int, L: Int, seed: Int, row0: Int,
                    size_off: I64P, size_w: U64P, cdf: U64P, wrand: UInt64, perm: I32P, masks: I32P, w: U64P):
    """t = row * m + sample: masks[t, :] and w[t]. A fixed sample s <
    nfixed is the s-th mask of shap's enumeration (sizes 1..nfull in order,
    each size's combinations in itertools order, a paired size's complement
    right after each combination), weighted size_w[size - 1]. A sampled one
    (sr = s - nfixed) is pair sr // 2: the size from cdf (u * cdf[L-1]
    against the cumulative weights, shap's rule), the first `size` entries
    of a Fisher-Yates permutation of the features; odd sr takes the
    complement. Its weight is wrand."""
    var row = t // m
    var s = t - row * m
    var mk = masks + t * M
    for f in range(M):
        mk[f] = 0
    if s < nfixed:
        var size = 1
        while size < nfull and Int(size_off[size]) <= s:
            size += 1
        var j = s - Int(size_off[size - 1])
        var paired = size <= npaired
        var r = j // 2 if paired else j
        var x = 0
        for pos in range(size):
            while True:
                var c = binom(M - 1 - x, size - 1 - pos)
                if r < c:
                    break
                r -= c
                x += 1
            mk[x] = 1
            x += 1
        if paired and (j & 1) == 1:
            for f in range(M):
                mk[f] = 1 - mk[f]
        w[t] = size_w[size - 1]
        return
    var sr = s - nfixed
    var pos = sr // 2
    var base = stream_base(seed, row0 + row)
    var c = sf64_mul(u_unit(draw(base, pos * (1 + M))), cdf[L - 1])
    var ind = L - 1
    for i in range(L):
        if sf64_lt(c, cdf[i]):
            ind = i
            break
    var size = ind + nfull + 1
    var pm = perm + t * M
    for i in range(M):
        pm[i] = Int32(i)
    var i = M - 1
    while i > 0:
        var jj = fy_index(base, pos * (1 + M) + 1 + i, i)
        var tmp = pm[i]
        pm[i] = pm[jj]
        pm[jj] = tmp
        i -= 1
    for q in range(size):
        mk[Int(pm[q])] = 1
    if (sr & 1) == 1:
        for f in range(M):
            mk[f] = 1 - mk[f]
    w[t] = wrand


def kshap_synth_unit(t: Int, nb: Int, d: Int, m: Int, x: F32P, bg: F32P, masks: I32P, syn: F32P):
    """t = ((row * m + s) * nb + r) * d + f: x[row, f] where the mask is on,
    else bg[r, f] (cuML `kernel_dataset`). A copy."""
    var f = t % d
    var q = t // d
    var r = q % nb
    var rs = q // nb
    var row = rs // m
    if masks[rs * d + f] != 0:
        syn[t] = x[row * d + f]
    else:
        syn[t] = bg[r * d + f]


def bg_mean_unit(t: Int, nb: Int, k: Int, y: F32P, res: U64P):
    """t = q * k + j: the mean over nb consecutive model rows of output j,
    summed in row order (xtrees/shap.mojo `block_mean`'s words)."""
    var q = t // k
    var j = t - q * k
    var acc = SF64_ZERO
    for r in range(nb):
        acc = sf64_add(acc, sf64_from_f32(y[(q * nb + r) * k + j]))
    res[t] = sf64_div(acc, sf64_from_int(nb))


def logit_unit(t: Int, x: U64P):
    """x[t] = log(x / (1 - x)) (shap `links.logit`; xtrees/ops.mojo `logit`)."""
    var v = x[t]
    x[t] = sf64_log(sf64_div(v, sf64_sub(SF64_ONE, v)))


def kshap_gram_unit(t: Int, d: Int, m: Int, masks: I32P, w: U64P, A: U64P, perm0: I32P):
    """t = (row * q + r) * q + c, q = d - 1: A[row, r, c] = the sum over the
    samples (ascending) with e_r != 0 of (w e_r) e_c, e_i = mask_i -
    mask_last (`kernel_solve`'s normal matrix); c == 0 also starts the
    row's pivot order."""
    var q = d - 1
    var row = t // (q * q)
    var rc = t - row * q * q
    var r = rc // q
    var c = rc - r * q
    if c == 0:
        perm0[row * q + r] = Int32(r)
    var a = SF64_ZERO
    for s in range(m):
        var mk = masks + (row * m + s) * d
        var last = sf64_from_int(Int(mk[q]))
        var er = sf64_sub(sf64_from_int(Int(mk[r])), last)
        if sf_zero(er):
            continue
        var wer = sf64_mul(w[row * m + s], er)
        var ec = sf64_sub(sf64_from_int(Int(mk[c])), last)
        a = sf64_add(a, sf64_mul(wer, ec))
    A[t] = a


def kshap_rhs_unit(t: Int, d: Int, m: Int, k: Int, masks: I32P, w: U64P, ey: U64P, fx: U64P, fnull: U64P,
                   B: U64P):
    """t = (row * q + r) * k + j: B[row, r, j] = the sum over the samples
    with e_r != 0 of (w e_r) y2, y2 = (ey - fnull) - last * total, total =
    fx - fnull (`kernel_solve`'s right-hand side)."""
    var q = d - 1
    var row = t // (q * k)
    var rj = t - row * q * k
    var r = rj // k
    var j = rj - r * k
    var total = sf64_sub(fx[row * k + j], fnull[j])
    var b = SF64_ZERO
    for s in range(m):
        var mk = masks + (row * m + s) * d
        var last = sf64_from_int(Int(mk[q]))
        var y2 = sf64_sub(sf64_sub(ey[(row * m + s) * k + j], fnull[j]), sf64_mul(last, total))
        var er = sf64_sub(sf64_from_int(Int(mk[r])), last)
        if sf_zero(er):
            continue
        b = sf64_add(b, sf64_mul(sf64_mul(w[row * m + s], er), y2))
    B[t] = b


def kshap_pivot_unit(t: Int, q: Int, col: Int, A: U64P, pin: I32P, pout: I32P):
    """t = row: column col's pivot (the first largest |A| among the rows not
    yet pivoted, in pivot order); pout = pin with positions col and the
    pivot's swapped."""
    var pi = pin + t * q
    var po = pout + t * q
    var a = A + t * q * q
    var piv = col
    var best = sf_abs(a[Int(pi[col]) * q + col])
    for r in range(col + 1, q):
        var v = sf_abs(a[Int(pi[r]) * q + col])
        if sf64_gt(v, best):
            best = v
            piv = r
    for r in range(q):
        po[r] = pi[r]
    po[col] = pi[piv]
    po[piv] = pi[col]


def kshap_elim_unit(t: Int, q: Int, k: Int, col: Int, A: U64P, B: U64P, perm: I32P):
    """t = (row * nr + ri) * nc + ci, nr = q - 1 - col rows below the pivot,
    nc = nr + k: row position col + 1 + ri, matrix column col + 1 + ci (ci <
    nr) or right-hand side ci - nr, `kernel_solve`'s update a -= f * a_pivot
    with f = a[rr, col] / pv (nothing when pv or f is zero). Column col
    itself is never read again, so it is not written."""
    var nr = q - 1 - col
    var nc = nr + k
    var row = t // (nr * nc)
    var rc = t - row * nr * nc
    var ri = rc // nc
    var ci = rc - ri * nc
    var pm = perm + row * q
    var a = A + row * q * q
    var pr = Int(pm[col])
    var rr = Int(pm[col + 1 + ri])
    var pv = a[pr * q + col]
    if sf_zero(pv):
        return
    var f = sf64_div(a[rr * q + col], pv)
    if sf_zero(f):
        return
    if ci < nr:
        var c = col + 1 + ci
        a[rr * q + c] = sf64_sub(a[rr * q + c], sf64_mul(f, a[pr * q + c]))
    else:
        var j = ci - nr
        var b = B + row * q * k
        b[rr * k + j] = sf64_sub(b[rr * k + j], sf64_mul(f, b[pr * k + j]))


def kshap_back_unit(t: Int, d: Int, k: Int, A: U64P, B: U64P, perm: I32P, fx: U64P, fnull: U64P, sol: U64P,
                    phi: U64P):
    """t = row * k + j: `kernel_solve`'s back substitution (a zero pivot
    gives 0), phi[row, c, j] = sol[c], and the last feature the efficiency
    remainder total - sum(sol). d == 1: phi = total."""
    var row = t // k
    var j = t - row * k
    var total = sf64_sub(fx[row * k + j], fnull[j])
    var q = d - 1
    var ph = phi + row * d * k
    if q == 0:
        ph[j] = total
        return
    var pm = perm + row * q
    var a = A + row * q * q
    var b = B + row * q * k
    var sv = sol + t * q
    var r = q - 1
    while r >= 0:
        var pr = Int(pm[r])
        var acc = b[pr * k + j]
        for c in range(r + 1, q):
            acc = sf64_sub(acc, sf64_mul(a[pr * q + c], sv[c]))
        var pv = a[pr * q + r]
        sv[r] = SF64_ZERO if sf_zero(pv) else sf64_div(acc, pv)
        r -= 1
    var ssum = SF64_ZERO
    for c in range(q):
        ph[c * k + j] = sv[c]
        ssum = sf64_add(ssum, sv[c])
    ph[q * k + j] = sf64_sub(total, ssum)


def fx_unit(t: Int, y: F32P, res: U64P):
    """The explained rows' own outputs as `block_mean` over one row writes
    them: (0 + y) / 1."""
    res[t] = sf64_div(sf64_add(SF64_ZERO, sf64_from_f32(y[t])), SF64_ONE)


# ---------------------------------------------------------- Permutation SHAP
def pshap_perm_unit(t: Int, d: Int, np: Int, seed: Int, row0: Int, perm: I32P, inv: I32P):
    """t = row * np + p: permutation p of row row0 + row, the Fisher-Yates
    walk over draws p*d + a of stream (seed, row0 + row) (the Python loop's
    `x_trees_uniform` words); inv[t, f] = f's position."""
    var row = t // np
    var p = t - row * np
    var base = stream_base(seed, row0 + row)
    var pm = perm + t * d
    for i in range(d):
        pm[i] = Int32(i)
    var a = d - 1
    while a > 0:
        var j = fy_index(base, p * d + a, a)
        var tmp = pm[a]
        pm[a] = pm[j]
        pm[j] = tmp
        a -= 1
    for pos in range(d):
        inv[t * d + Int(pm[pos])] = Int32(pos)


def pshap_synth_unit(t: Int, nb: Int, d: Int, np: Int, x: F32P, bg: F32P, inv: I32P, syn: F32P):
    """t = (((row * np + p) * (2d + 1) + o) * nb + r) * d + f: coalition o
    of permutation p turns feature f on when its position pos satisfies pos
    < o (the forward walk, o <= d) or pos >= o - d (the backward walk)."""
    var span = 2 * d + 1
    var f = t % d
    var q = t // d
    var r = q % nb
    var s = q // nb
    var rp = s // span
    var o = s - rp * span
    var row = rp // np
    var pos = Int(inv[rp * d + f])
    var on = pos < o if o <= d else pos >= o - d
    if on:
        syn[t] = x[row * d + f]
    else:
        syn[t] = bg[r * d + f]


def pshap_marginal_unit(t: Int, d: Int, k: Int, np: Int, inv: I32P, ey: U64P, phi: U64P):
    """t = (row * d + f) * k + c: the sum over the permutations (in order)
    of feature f's forward then backward marginal, over 2 * np."""
    var row = t // (d * k)
    var fc = t - row * d * k
    var f = fc // k
    var c = fc - f * k
    var step = 2 * d + 1
    var e = ey + row * np * step * k
    var val = SF64_ZERO
    for p in range(np):
        var o = p * step
        var jj = Int(inv[(row * np + p) * d + f])
        val = sf64_add(val, sf64_sub(e[(o + jj + 1) * k + c], e[(o + jj) * k + c]))
        val = sf64_add(val, sf64_sub(e[(o + d + jj) * k + c], e[(o + d + jj + 1) * k + c]))
    phi[t] = sf64_div(val, sf64_from_int(2 * np))


# ------------------------------------- Permutation SHAP delta rows (FAST)
# MOJOLEARN_PSHAP_DELTA (lane apple-fast-w4-shap, 2026-10-04; opt-in, FAST +
# Apple only, xtrees/agnostic_device.mojo): coalition o of a permutation
# differs from coalition o - 1 in ONE feature (pshap_toggled). A background
# row r whose value of that feature has the same bits as the explained
# row's gives the same synthetic row as at o - 1, so the model is not run on
# it again (shap `MaskedModel`'s delta masking with `masker.invariants`,
# here exact bit equality): only the varying (o, r) rows are built, the
# model runs on those, and every (o, r) reads the output of its latest
# varying predecessor. The background means are bg_mean_unit's adds in the
# same row order over the same outputs, so for a model whose row output
# does not depend on the batch, phi keeps its words.

@always_inline
def pshap_toggled(o: Int, d: Int, pm: I32P) -> Int:
    """The feature coalition o (1 <= o <= 2d) toggles against o - 1: the
    feature at position o - 1 turns on (forward walk, o <= d), the one at
    position o - d - 1 turns off (backward walk)."""
    return Int(pm[o - 1]) if o <= d else Int(pm[o - d - 1])


@always_inline
def pshap_varies(g: Int, r: Int, d: Int, np: Int, x: F32P, bg: F32P, perm: I32P) -> Bool:
    """Group g = (row * np + p) * (2d + 1) + o: background row r's synthetic
    row differs from its row at o - 1 (always at o = 0)."""
    var span = 2 * d + 1
    var rp = g // span
    var o = g - rp * span
    if o == 0:
        return True
    var row = rp // np
    var f = pshap_toggled(o, d, perm + rp * d)
    return bitcast[DType.uint32](x[row * d + f]) != bitcast[DType.uint32](bg[r * d + f])


def pshap_dcount_unit(t: Int, nb: Int, d: Int, np: Int, x: F32P, bg: F32P, perm: I32P, cnt: I64P):
    """t = group: how many of its nb background rows vary."""
    var c = 0
    for r in range(nb):
        if pshap_varies(t, r, d, np, x, bg, perm):
            c += 1
    cnt[t] = Int64(c)


def scan_step_unit(t: Int, off: Int, src: I64P, dst: I64P):
    """One Hillis-Steele step of an inclusive integer prefix sum."""
    dst[t] = src[t] + src[t - off] if t >= off else src[t]


def pshap_dindex_unit(t: Int, nb: Int, d: Int, np: Int, x: F32P, bg: F32P, perm: I32P, incl: I64P, cnt: I64P,
                      idx: I64P, src: I64P):
    """t = group * nb + r: a varying row's compact index (its group's
    exclusive offset plus its rank among the group's varying rows, so the
    compact rows keep the full order), -1 for a repeat; src[index] = t."""
    var g = t // nb
    var r = t - g * nb
    if not pshap_varies(g, r, d, np, x, bg, perm):
        idx[t] = -1
        return
    var rank = 0
    for q in range(r):
        if pshap_varies(g, q, d, np, x, bg, perm):
            rank += 1
    var c = incl[g] - cnt[g] + Int64(rank)
    idx[t] = c
    src[Int(c)] = Int64(t)


def pshap_dsynth_unit(t: Int, nb: Int, d: Int, np: Int, x: F32P, bg: F32P, inv: I32P, src: I64P, syn: F32P):
    """t = c * d + f: feature f of compact row c, pshap_synth_unit's value
    for the full row src[c]."""
    var c = t // d
    var f = t - c * d
    var q = Int(src[c])
    var r = q % nb
    var s = q // nb
    var span = 2 * d + 1
    var rp = s // span
    var o = s - rp * span
    var row = rp // np
    var pos = Int(inv[rp * d + f])
    var on = pos < o if o <= d else pos >= o - d
    if on:
        syn[t] = x[row * d + f]
    else:
        syn[t] = bg[r * d + f]


def pshap_dmap_unit(t: Int, nb: Int, idx: I64P, mp: I64P):
    """t = group * nb + r: the compact row whose model output (o, r) reads,
    its latest varying predecessor in the same permutation (o = 0 always
    varies, so the walk stops there at the latest)."""
    var q = t
    while idx[q] < 0:
        q -= nb
    mp[t] = idx[q]


def bg_mean_mapped_unit(t: Int, nb: Int, k: Int, y: F32P, mp: I64P, res: U64P):
    """bg_mean_unit over the mapped compact outputs: the same adds in the
    same row order."""
    var q = t // k
    var j = t - q * k
    var acc = SF64_ZERO
    for r in range(nb):
        acc = sf64_add(acc, sf64_from_f32(y[Int(mp[q * nb + r]) * k + j]))
    res[t] = sf64_div(acc, sf64_from_int(nb))
