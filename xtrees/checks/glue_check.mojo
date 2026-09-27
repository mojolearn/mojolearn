# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The trees lane's seam gate (DEVIATIONS 5600-5605, IDENTITY_PATHS rows
160-165). For each seam of xtrees/ops.mojo: first show the fixture SEPARATES
the pinned spelling from the unpinned one (else VACUOUS, a failure), then
assert the shipped function equals the oracle (xtrees/checks/glue_oracle.mojo)
bit for bit, and record the shipped output on the card.

    tools/with_identical_mode.sh pixi run mojo run -I . xtrees/checks/glue_check.mojo

The card goes to $MOJOLEARN_XTREES_CARD (default /tmp/xtrees_glue.card);
`python3 tools/identity_trace_diff.py a.card b.card` compares two boxes.
"""
from std.os import getenv
from std.memory import bitcast
from core.identity_trace import IdentityTrace
from checks.numerics import portable_log64, pinned_mul_f64
from xtrees.shap import block_mean
from xtrees.ops import (
    sample_indices, accumulate, argmax_rows, samme_step, normalize_rows,
)
from xtrees.checks.glue_oracle import (
    counter_draw, index_mod, index_mulshift, pinned_acc, fused_acc, seq_sum, pair_sum,
    pinned_exp, libm_exp, first_max, last_max, normalized,
)


def _pi(mut l: List[Int32]) -> MutPointer[Int32, MutUntrackedOrigin]:
    return l.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()


def _pf(mut l: List[Float32]) -> MutPointer[Float32, MutUntrackedOrigin]:
    return l.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()


def _pd(mut l: List[Float64]) -> MutPointer[Float64, MutUntrackedOrigin]:
    return l.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()


def _bits(x: Float64) -> UInt64:
    return bitcast[DType.uint64](x)


def _require(ok: Bool, what: String) raises:
    if not ok:
        raise Error("FAIL: " + what)


def check_rng(mut card: IdentityTrace) raises:
    """DEVIATION 5600: index k of (seed, stream) is counter_draw mod n."""
    var n_pool = 1000
    var n = 64
    var sep = False
    for k in range(n):
        var r = counter_draw(7, 3, k)
        if index_mod(r, n_pool) != index_mulshift(r, n_pool):
            sep = True
    _require(sep, "VACUOUS rng fixture: mod and multiply-shift agree on every draw")
    var got = List[Int32](length=n, fill=0)
    sample_indices(_pi(got), n_pool, n, True, 7, 3)
    for k in range(n):
        _require(Int(got[k]) == index_mod(counter_draw(7, 3, k), n_pool), "rng: draw " + String(k))
    var short = List[Int32](length=9, fill=0)
    sample_indices(_pi(short), n_pool, 9, True, 7, 3)
    for k in range(9):
        _require(short[k] == got[k], "rng: draw k depends on how many were drawn")
    card.record_host("rng.indices", _pi(got), n)
    print("PASS rng: 64 counter draws mod 1000 (the fixture separates mod from multiply-shift)")


def check_pinned_accumulate(mut card: IdentityTrace) raises:
    """DEVIATION 5601: acc + w * x rounds the product first."""
    var n = 64
    var w: Float64 = 0.1
    var x = List[Float32](length=n, fill=0)
    var acc = List[Float64](length=n, fill=0.0)
    var want = List[Float64](length=n, fill=0.0)
    var sep = False
    for i in range(n):
        x[i] = Float32(1.0) + Float32(i) * Float32(1.1920929e-07) * 3
        acc[i] = -pinned_mul_f64(w, Float64(x[i]))
        want[i] = pinned_acc(acc[i], w, Float64(x[i]))
        if _bits(want[i]) != _bits(fused_acc(acc[i], w, Float64(x[i]))):
            sep = True
    _require(sep, "VACUOUS accumulate fixture: fused and pinned agree")
    accumulate(_pd(acc), _pf(x), n, w)
    _ = len(x)  # alive through the call: a pointer does not extend a List's life
    for i in range(n):
        _require(_bits(acc[i]) == _bits(want[i]), "accumulate: element " + String(i))
    card.record_host("accumulate", _pd(acc), n)
    print("PASS accumulate: pinned product then add (the fixture separates it from the fused form)")


def check_fold_order(mut card: IdentityTrace) raises:
    """DEVIATION 5602: the background mean folds rows in order."""
    var big = Float32(1152921504606846976.0)  # 2^60
    var y: List[Float32] = [big, 1.0, -big, 1.0]
    var v: List[Float64] = [Float64(big), 1.0, -Float64(big), 1.0]
    _require(_bits(seq_sum(v)) != _bits(pair_sum(v)), "VACUOUS fold fixture: sequential == pairwise")
    var res = List[Float64](length=1, fill=0.0)
    block_mean(_pf(y), 1, 4, 1, _pd(res))
    _ = len(y)  # alive through the call
    _require(_bits(res[0]) == _bits(seq_sum(v) / 4.0), "block_mean: not the sequential fold")
    card.record_host("block_mean", _pd(res), 1)
    print("PASS fold order: sequential over background rows (the fixture separates it from pairwise)")


def check_pinned_exp(mut card: IdentityTrace) raises:
    """DEVIATION 5603: SAMME's reweighting uses the pinned exp and log."""
    var n = 8
    var pred: List[Int32] = [0, 1, 1, 2, 0, 1, 2, 2]
    var y: List[Int32] = [0, 1, 2, 2, 1, 1, 2, 0]
    var lr: Float64 = 0.0
    var alpha: Float64 = 0.0
    var err: Float64 = 3.0 / 8.0
    var found = False
    for step in range(1, 2000):
        var cand = 0.25 + Float64(step) * 0.000731
        var a = pinned_mul_f64(cand, portable_log64((1.0 - err) / err) + portable_log64(2.0))
        if _bits(pinned_exp(a)) != _bits(libm_exp(a)):
            lr = cand
            alpha = a
            found = True
            break
    _require(found, "VACUOUS exp fixture: the pinned and libm exp agree on every candidate")
    var w = List[Float64](length=n, fill=0.125)
    var stats = List[Float64](length=4, fill=0.0)
    samme_step(_pd(w), _pi(pred), _pi(y), n, 3, lr, False, _pd(stats))
    _ = len(pred) + len(y)  # alive through the call
    _require(_bits(stats[1]) == _bits(alpha), "samme: estimator weight")
    var boost = pinned_exp(alpha)
    for i in range(n):
        var want = pinned_mul_f64(0.125, boost) if pred[i] != y[i] else 0.125
        _require(_bits(w[i]) == _bits(want), "samme: weight " + String(i))
    card.record_host("samme.w", _pd(w), n)
    card.record_host("samme.stats", _pd(stats), 4)
    print("PASS pinned exp/log: SAMME alpha and reweighting (the fixture separates pinned from libm exp)")


def check_tie_break(mut card: IdentityTrace) raises:
    """DEVIATION 5604: argmax gives a tie to the lower index."""
    var x: List[Float64] = [0.5, 0.5, 0.1, 0.2, 0.7, 0.7]
    var r0: List[Float64] = [0.5, 0.5, 0.1]
    var r1: List[Float64] = [0.2, 0.7, 0.7]
    _require(first_max(r0) != last_max(r0) and first_max(r1) != last_max(r1), "VACUOUS tie fixture")
    var got = List[Int32](length=2, fill=0)
    argmax_rows(_pd(x), 2, 3, _pi(got))
    _ = len(x)  # alive through the call
    _require(Int(got[0]) == first_max(r0) and Int(got[1]) == first_max(r1), "argmax: tie not to the lower index")
    card.record_host("argmax", _pi(got), 2)
    print("PASS tie-break: first max")


def check_zero_rows(mut card: IdentityTrace) raises:
    """DEVIATION 5605: a zero row (+0 or -0) normalises to uniform, no NaN."""
    var x: List[Float64] = [0.0, 0.0, 0.0, -0.0, 0.0, -0.0, 1.0, 3.0, 0.0]
    var rows = List[List[Float64]]()
    for r in range(3):
        var row = List[Float64]()
        for c in range(3):
            row.append(x[r * 3 + c])
        rows.append(row^)
    normalize_rows(_pd(x), 3, 3)
    for r in range(3):
        var want = normalized(rows[r])
        for c in range(3):
            var v = x[r * 3 + c]
            _require(v == v, "normalize_rows: NaN")
            _require(_bits(v) == _bits(want[c]), "normalize_rows: row " + String(r))
    card.record_host("normalize_rows", _pd(x), 9)
    print("PASS zero rows: uniform, never 0/0")


def main() raises:
    var path = getenv("MOJOLEARN_XTREES_CARD", "/tmp/xtrees_glue.card")
    var card = IdentityTrace.to_path(path)
    card.header("xtrees glue seams (DEVIATIONS 5600-5605)")
    check_rng(card)
    check_pinned_accumulate(card)
    check_fold_order(card)
    check_pinned_exp(card)
    check_tie_break(card)
    check_zero_rows(card)
    print("PASS xtrees glue: 6 seams, card", path)
