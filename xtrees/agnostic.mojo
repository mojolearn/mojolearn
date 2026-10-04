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
from xtrees.ops import draw, stream_base
from std.memory import bitcast
from checks.numerics import ftz, identical_div

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


# ------------------------------------------------- the model on the device
# Lane fam2-forests (2026-10-04, MOJOLEARN_IDN_SHAP_DEVICE_MODEL): when the
# explained model is one of this library's flat forests, its output on a
# chunk's synthetic rows is computed here, on the device, straight from x,
# the background and the chunk's masks or permutations: no synthetic matrix,
# no download, no model call, no upload. The arithmetic is the forest's own
# IDENTICAL predict (core/forest_inference.mojo `forest_ordered_kernel`):
# the same key comparison per node, the leaves added in increasing tree
# order from +0.0 with `forest_add`, the flushed divide by the tree count.
# The value a node compares is the one the synthetic matrix would have held
# (`kshap_synth_unit` / `pshap_synth_unit`: x where the feature is on, else
# the background row), so every output word is the word the model call
# returned for that synthetic row.
@always_inline
def agn_finite_key(value: Float32) -> UInt32:
    """core/forest_inference.mojo `finite_key`: the total order of finite
    Float32 as unsigned keys, the two zeros equal."""
    var bits = bitcast[DType.uint32](value)
    if (bits & UInt32(0x7fffffff)) == 0:
        bits = 0
    return ~bits if (bits & UInt32(0x80000000)) != 0 else bits ^ UInt32(0x80000000)


@always_inline
def agn_forest_add(a: Float32, b: Float32) -> Float32:
    """core/forest_inference.mojo `forest_add`."""
    return ftz(ftz(a) + ftz(b))


@always_inline
def agn_model_unit[RF_INPUT: Bool, PERM: Bool](q: Int, nb: Int, d: Int, mc: Int, k: Int, trees: Int, off: I32P,
                                               col: I32P, thr: F32P, left: I32P, leaf: F32P, x: F32P, bg: F32P,
                                               sel: I32P, y: F32P):
    """q = one synthetic row: y[q, 0..k) = the forest on it.
    PERM False (Kernel SHAP): q = (row * mc + s) * nb + r, mc = samples per
    row, sel = the masks ((row, s) x d), feature f on when its mask is set.
    PERM True (Permutation SHAP): q = ((row * mc + p) * (2d + 1) + o) * nb +
    r, mc = permutations per row, sel = the inverse permutations ((row, p) x
    d), feature f on by `pshap_synth_unit`'s rule. RF_INPUT flushes the
    compared feature value (the random forest's predict; ExtraTrees' does
    not), as `reached_leaf` does. Nodes: children are local ids, right =
    left + 1, a leaf has left = -1; leaf values are node x k."""
    var r = q % nb
    var s = q // nb
    var row = 0
    var selbase = 0
    var o = 0
    comptime if PERM:
        var span = 2 * d + 1
        var rp = s // span
        o = s - rp * span
        row = rp // mc
        selbase = rp * d
    else:
        row = s // mc
        selbase = s * d
    var xb = row * d
    var bb = r * d
    var yb = q * k
    for c in range(k):
        y[yb + c] = Float32(0)
    for tree in range(trees):
        var base = Int(off[tree])
        var node = base
        var child = Int(left[node])
        while child != -1:
            var f = Int(col[node])
            var on = False
            comptime if PERM:
                var pos = Int(sel[selbase + f])
                on = pos < o if o <= d else pos >= o - d
            else:
                on = sel[selbase + f] != 0
            var value = x[xb + f]
            if not on:
                value = bg[bb + f]
            comptime if RF_INPUT:
                value = ftz(value)
            var go_left = agn_finite_key(value) <= agn_finite_key(thr[node])
            node = base + child + (0 if go_left else 1)
            child = Int(left[node])
        for c in range(k):
            y[yb + c] = agn_forest_add(y[yb + c], leaf[node * k + c])
    var ft = Float32(trees)
    for c in range(k):
        y[yb + c] = ftz(identical_div(ftz(y[yb + c]), ft))
