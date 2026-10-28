# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The SVC Python reference (lane/py-dn-svm, 2026-09-28; moved out of
`_svm_impl` by lane/cgr-kernel so the estimator module holds no host row
loops). SVC never calls these: it runs their Mojo transcription
(svm/impl/svc_rows.mojo, svm/impl/svc_epilogue.mojo) through the svm
bindings, on the device on a GPU install. They stay as the reference that
transcription is held to bit for bit, and as the libsvm reading the tests
exercise."""

from mojolearn import _portable_math as math


_FEISTEL_M0 = 0xD2B74407B1CE6E93
_M64 = 0xFFFFFFFFFFFFFFFF


def _shuffle_seed32(seed):
    """The Feistel key: the low 32 bits of one SplitMix64 step of `seed`
    (`svm/impl/svc_rows.mojo::shuffle_seed32`)."""
    z = (seed + 0x9E3779B97F4A7C15) & _M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _M64
    z ^= z >> 31
    return z & 0xFFFFFFFF


def _splitmix_perm(n, seed):
    """The probability shuffle (cgfin-c-svm, 2026-10-02): `perm[i]` is
    `core/shuffle_iterator.mojo`'s Feistel bijection of `[0, n)` (CCCL's
    `random_bijection`: 24 rounds keyed by a minstd stream, cycle walk) at
    i, keyed by `_shuffle_seed32(seed)`. Every index is independent, so the
    device draws it one thread per index; pure integer arithmetic, so every
    column draws the same permutation. (It was libsvm's serial Fisher-Yates
    over a SplitMix64 stream.)"""
    x = _shuffle_seed32(seed & _M64) % 2147483647
    x = 1 if x == 0 else x
    keys = []
    for _ in range(24):
        sp = 0
        for _ in range(2):
            x = (48271 * x) % 2147483647
            u = x - 1
            while u >= 2147418112:
                x = (48271 * x) % 2147483647
                u = x - 1
            sp = ((sp << 16) + (u & 0xFFFF)) & 0xFFFFFFFF
        keys.append(sp)
    num = max(1, n)
    total = max(8, (num - 1).bit_length())
    lb = total // 2
    rb = total - lb
    lmask = (1 << lb) - 1
    rmask = (1 << rb) - 1

    def trip(v):
        left = (v >> rb) & 0xFFFFFFFF
        right = v & rmask
        for key in keys:
            product = (_FEISTEL_M0 * left) & _M64
            f_k = ((product >> 32) & 0xFFFFFFFF) ^ key
            b_k = product & 0xFFFFFFFF
            lp = f_k ^ right
            rp = ((b_k << (rb - lb)) & 0xFFFFFFFF) | (right >> lb)
            left = lp & lmask
            right = rp & rmask
        return (left << rb) | right

    perm = []
    for i in range(n):
        v = trip(i)
        while v >= num:
            v = trip(v)
        perm.append(v)
    return perm


def _tree_sum(vals):
    """The device fold's order (`svm/impl/svc_rows.mojo::tree_sum_sf64`):
    chunks of 256 cells, each folded by a halving tree (padding +0.0), the
    chunk sums the next level's cells, until one chunk remains."""
    cur = list(vals)
    if not cur:
        return 0.0
    while True:
        nxt = []
        for base in range(0, len(cur), 256):
            blk = cur[base:base + 256]
            blk += [0.0] * (256 - len(blk))
            step = 128
            while step:
                for t in range(step):
                    blk[t] = blk[t] + blk[t + step]
                step //= 2
            nxt.append(blk[0])
        if len(nxt) == 1:
            return nxt[0]
        cur = nxt


# THE PYTHON REFERENCE (lane/py-dn-svm, 2026-09-28; the row sums and the
# shuffle follow the device since cgfin-c-svm, 2026-10-02). `_splitmix_perm`,
# `_platt_fval`, `_sigmoid_train`, `_sigmoid_predict` and
# `_multiclass_probability` below are no longer called by SVC: the
# estimator runs their Mojo transcription (svm/impl/svc_rows.mojo) through
# the svm bindings. They stay as the reference that transcription is held to
# bit for bit, and as the libsvm reading the tests exercise.
# DEVIATION 6903 (IDENTITY_PATHS row 253, lane py-bugs): Platt scaling and the
# pairwise coupling are binary64 arithmetic in libsvm's loop order with no FMA
# on the pinned exp / log (`_portable_math` here, `pm_exp` / `pm_log` in the
# Mojo transcription, lane py-dn-svm).
def _platt_fval(dec, t, a, b):
    terms = []
    for d, ti in zip(dec, t):
        fapb = d * a + b
        if fapb >= 0.0:
            terms.append(ti * fapb + math.log(1.0 + math.exp(-fapb)))
        else:
            terms.append((ti - 1.0) * fapb + math.log(1.0 + math.exp(fapb)))
    return _tree_sum(terms)


def _sigmoid_train(dec, labels):
    """libsvm's `sigmoid_train` (Platt's method with Lin, Lin and Weng's
    Newton iteration and backtracking), transcribed in binary64 with the
    repository's portable exp and log: the same `(A, B)` bits on every
    host for the same decision values. `labels` are +1 / -1. The row sums
    run in the device fold's order (`_tree_sum`; cgfin-c-svm), with h11 and
    h22 `sigma + sum`."""
    prior1 = float(sum(1 for v in labels if v > 0))
    prior0 = float(len(labels)) - prior1
    max_iter, min_step, sigma, eps = 100, 1e-10, 1e-12, 1e-5
    hi = (prior1 + 1.0) / (prior1 + 2.0)
    lo = 1.0 / (prior0 + 2.0)
    t = [hi if v > 0 else lo for v in labels]
    a = 0.0
    b = math.log((prior0 + 1.0) / (prior1 + 1.0))
    fval = _platt_fval(dec, t, a, b)
    for _ in range(max_iter):
        th11, th22, th21, tg1, tg2 = [], [], [], [], []
        for d, ti in zip(dec, t):
            fapb = d * a + b
            if fapb >= 0.0:
                e = math.exp(-fapb)
                p = e / (1.0 + e)
                q = 1.0 / (1.0 + e)
            else:
                e = math.exp(fapb)
                p = 1.0 / (1.0 + e)
                q = e / (1.0 + e)
            d2 = p * q
            th11.append(d * d * d2)
            th22.append(d2)
            th21.append(d * d2)
            d1 = ti - p
            tg1.append(d * d1)
            tg2.append(d1)
        h11 = sigma + _tree_sum(th11)
        h22 = sigma + _tree_sum(th22)
        h21, g1, g2 = _tree_sum(th21), _tree_sum(tg1), _tree_sum(tg2)
        if abs(g1) < eps and abs(g2) < eps:
            break
        det = h11 * h22 - h21 * h21
        da = -(h22 * g1 - h21 * g2) / det
        db = -(-h21 * g1 + h11 * g2) / det
        gd = g1 * da + g2 * db
        step = 1.0
        while step >= min_step:
            na = a + step * da
            nb = b + step * db
            newf = _platt_fval(dec, t, na, nb)
            if newf < fval + 0.0001 * step * gd:
                a, b, fval = na, nb, newf
                break
            step = step / 2.0
        if step < min_step:
            break
    return a, b


def _sigmoid_predict(dec, a, b):
    """libsvm's `sigmoid_predict`: P(+1) at one decision value."""
    fapb = dec * a + b
    if fapb >= 0.0:
        e = math.exp(-fapb)
        return e / (1.0 + e)
    return 1.0 / (1.0 + math.exp(fapb))


def _multiclass_probability(k, r):
    """libsvm's `multiclass_probability` (Wu, Lin and Weng's pairwise
    coupling, method 2), binary64, its loop order."""
    p = [1.0 / k] * k
    q = [[0.0] * k for _ in range(k)]
    for t in range(k):
        for j in range(t):
            q[t][t] += r[j][t] * r[j][t]
            q[t][j] = q[j][t]
        for j in range(t + 1, k):
            q[t][t] += r[j][t] * r[j][t]
            q[t][j] = -r[j][t] * r[t][j]
    eps = 0.005 / k
    qp = [0.0] * k
    for _ in range(max(100, k)):
        pqp = 0.0
        for t in range(k):
            qp[t] = 0.0
            for j in range(k):
                qp[t] += q[t][j] * p[j]
            pqp += p[t] * qp[t]
        max_error = 0.0
        for t in range(k):
            err = abs(qp[t] - pqp)
            if err > max_error:
                max_error = err
        if max_error < eps:
            break
        for t in range(k):
            diff = (-qp[t] + pqp) / q[t][t]
            p[t] += diff
            pqp = (pqp + diff * (diff * q[t][t] + 2.0 * qp[t])) / (1.0 + diff) / (1.0 + diff)
            for j in range(k):
                qp[j] = (qp[j] + diff * q[t][j]) / (1.0 + diff)
                p[j] /= (1.0 + diff)
    return p


def ovr_scores(model, decisions, n_rows):
    """REFERENCE ONLY since lane/py-dn-svm (the estimator calls
    `_epilogue(_EPI_OVR, ...)`): scikit-learn's
    `_ovr_decision_function(dec < 0, -dec, K)` on the one-vs-one
    decisions `dec` (= -d here), `decisions` a list of per-pair lists:
    votes plus the summed confidences squashed into (-1/3, 1/3), in
    binary64, accumulated in pair order."""
    k = len(model.classes_)
    out = []
    for r in range(n_rows):
        votes = [0.0] * k
        conf = [0.0] * k
        for p, d in zip(model._pairs, decisions):
            v = d[r]
            i, j = p["i"], p["j"]
            conf[i] -= v
            conf[j] += v
            if v > 0.0:
                votes[j] += 1.0
            else:
                votes[i] += 1.0
        out.append([votes[c] + conf[c] / (3.0 * (abs(conf[c]) + 1.0)) for c in range(k)])
    return out


# Test-only binding glue moved from `_svm_impl` (its runtime no longer calls
# any of it): the legacy host-inference twin and tools/py_consolidated/svc_equal.py
# drive the binding entries directly through these.
import array as _array_mod
from mojolearn._array import Array
from mojolearn._buffer import addr, addr_ro, empty

array = _array_mod
(_EPI_OVO, _EPI_OVR, _EPI_VOTES, _EPI_PROBA, _EPI_LOG_PROBA, _EPI_BINARY,
 _GLUE_GATHER, _GLUE_SELECT, _GLUE_C_ROWS, _GLUE_FOLD, _GLUE_FOLD_FINISH) = range(11)


def _i32(values):
    """`values` (an int32 Array or a sequence of ints) as an int32 Array."""
    if isinstance(values, Array):
        return values
    # lane py-runtime-b: array('i') converts the ints in C (no Python
    # comprehension over the indices); integral floats take int() as before
    try:
        store = array.array("i", values)
    except TypeError:
        store = array.array("i", map(int, values))
    return Array._owned(store, (len(store),), "<i4", "C")


def _gather(binding, x, rows=None, cols=None):
    """`x[rows][:, cols]`, an exact copy of float32 cells, gathered in the
    binding (on the device on a GPU install; cgfin-c-svm). An absent index
    list keeps that axis whole."""
    n_rows, n_cols = x.shape
    ra = None if rows is None else _i32(rows)
    ca = None if cols is None else _i32(cols)
    nor = n_rows if ra is None else ra.shape[0]
    noc = n_cols if ca is None else ca.shape[0]
    out = empty((nor, noc), "<f4")
    if nor and noc:
        binding.svc_pair_epilogue(
            addr_ro(x, name="x"),
            0 if ra is None else addr_ro(ra, name="row indices"),
            0 if ca is None else addr_ro(ca, name="column indices"),
            addr(out, name="gathered"),
            [_GLUE_GATHER, n_rows, n_cols, nor, noc])
    return out


def _precomputed_columns(binding, q, support):
    """kernel='precomputed' at predict: `X[:, support_]`, the query's
    cross-kernel columns at the support vectors, copied exactly (float32
    bytes, no arithmetic). What the solver's decision reads as its kernel
    tile."""
    # `support` is the fitted int32 Array (a fit's own rows, or a load's,
    # checked against n_features_in_ by the native loader); q has
    # n_features_in_ columns (`_query`), so every index is in range and no
    # per-predict Python pass over it runs (lane cpu2-l10-linear)
    return _gather(binding, q, None, _i32(support))


def _native_perm(binding, n, seed):
    """`tests/_svm_reference.py::_splitmix_perm(n, seed)` in Mojo (`svc_splitmix_perm`): the same
    integers, an int32 `array.array`."""
    out = array.array("i", bytes(4 * n))
    seed &= 0xFFFFFFFFFFFFFFFF
    if n:
        binding.svc_splitmix_perm(out.buffer_info()[0], n, seed & 0xFFFFFFFF, seed >> 32)
    return out


def _native_sigmoid_train(binding, dec, labels):
    """`tests/_svm_reference.py::_sigmoid_train(dec, labels)` in Mojo (`svc_platt_train`): the same
    binary64 operations in the same order, the same `(A, B)` bits."""
    d = array.array("d", dec)
    lab = array.array("d", labels)
    if len(d) != len(lab):
        raise ValueError("decision values and labels differ in length")
    ab = array.array("d", [0.0, 0.0])
    binding.svc_platt_train(d.buffer_info()[0] if len(d) else 0,
                            lab.buffer_info()[0] if len(lab) else 0, ab.buffer_info()[0], len(d))
    return ab[0], ab[1]
