#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""SGDOneClassSVM tail-replay quality (lane/apple-fast-w2-sgdoc): dump one
arm's fitted state, compare main (A) with MOJOLEARN_SGDOC_FAST_TAIL (B).

  dump OUT.npz      fits the installed FAST x_linear binding (Metal)
  compare A.npz B.npz

Fixture: the board's sgd-ocsvm arrays (tools/bench_board_algos.py
`_load_block` + `lane_arrays`, so the board's own rows) from rows-small
(50k fit rows, restandardized on themselves as the board standardizes its
fit rows; 20 epochs = 1M steps > the 65,536-step tail), datasets taxi
and istella, board params nu=0.1, max_iter=20, tol=None, seeds 7 (the
board's), 0, 1, 2, 3; plus taxi shifted by +1 per column at seed 7 (not
centered: the tail must refuse it, so B must equal A bit for bit).
scikit-learn on the same rows and seeds is reported, never gated.

Why a seed spread and not A == B. On centered rows SGDOneClassSVM's final
state is one draw of a stationary chain (x_linear/sgdoc_tail.mojo): w ~ 1e-6,
a noise direction set by the last steps, flagged fraction anywhere in
~0.007 .. 0.093 on the board (sklearn taxi 0.00702, main 0.05557). Main's
own seed-to-seed spread is the noise of the algorithm; B passes when it
behaves as another draw of main. Gates, fixed before any B result (per
dataset over the 5 seeds; ff = fraction of Xq with predict == -1,
J = nu/2 |w|^2 + nu (1 - rho) + mean max(0, rho - w.x) on X in float64,
jac = |F_a & F_b| / |F_a | F_b| of the Xq flag sets, 1 when both empty):
  C1 median ff_B in [min ff_A, max ff_A]
  C2 median J_B <= max J_A
  C3 median |w_B| in [0.5 min |w_A|, 2 max |w_A|]
  C4 median std(decision_B on Xq) in [0.5 min, 2 max] of A's
  C5 median jac(B_s, A_s) >= min over s != s' of jac(A_s, A_s')
  C6 the tail ran: coef_B != coef_A on at least 4 of the 5 seeds
  C7 shifted taxi: coef, offset, decision of B == A exactly
"""
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
import numpy as np

NU = 0.1
SEEDS = (7, 0, 1, 2, 3)
DATASETS = ("taxi", "istella")
ROOT = Path(__file__).resolve().parents[1]


def _board():
    spec = importlib.util.spec_from_file_location("bench_board_algos", ROOT / "tools/bench_board_algos.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bench_board_algos"] = mod
    spec.loader.exec_module(mod)
    return mod


def _rows_small():
    for b in ("board-0834", "board-0833"):
        p = Path.home() / b / "cache/algos-data/rows-small"
        if (p / "SMALL_ROWS").exists() or (p / "cls-taxi.npz").exists():
            return p
    raise SystemExit("SGDOC-TAIL REFUSED: no rows-small board data")


def _restandardize(X, Xq):
    """The board's cls scaling (classical_two_datasets.standardize: float64
    mean and deviation of the fit rows, zero deviation divides by 1) redone on
    rows-small's 50k fit rows, which are the FIRST 50k rows of the full block
    (tools/gapnv2/make_small_rows.py) and so not centered on their own; the
    board's full block is centered on its fit rows, and so is this fixture."""
    f64 = X.astype(np.float64)
    mu = f64.mean(axis=0)
    sd = f64.std(axis=0)
    sd[sd == 0.0] = 1.0
    return (np.ascontiguousarray(((f64 - mu) / sd).astype(np.float32)),
            np.ascontiguousarray(((Xq.astype(np.float64) - mu) / sd).astype(np.float32)))


def objective(X, w, rho):
    sc = np.zeros(X.shape[0])
    w = np.asarray(w, np.float64)
    for lo in range(0, X.shape[0], 131072):
        sc[lo:lo + 131072] = X[lo:lo + 131072].astype(np.float64) @ w
    return float(0.5 * NU * (w @ w) + NU * (1.0 - rho) + np.maximum(0.0, rho - sc).mean())


def dump(path):
    assert os.environ.get("MOJOLEARN_NUMERIC_MODE") == "fast"
    import mojolearn as ml
    from mojolearn import _backend
    binding = _backend.binding("_mojolearn_x_linear", "fast")
    assert str(binding.x_linear_vendor()) == "metal"
    meta = dict(binding_sha256=hashlib.sha256(Path(binding.__file__).read_bytes()).hexdigest(),
                fixture="sgdoc-tail-v1-rows-small")
    bb = _board()
    data = _rows_small()
    out = {}
    from sklearn.linear_model import SGDOneClassSVM as SkOC
    cases = [(ds, s, 0.0) for ds in DATASETS for s in SEEDS] + [("taxi", 7, 1.0)]
    loaded = {}
    for ds, seed, shift in cases:
        if ds not in loaded:
            B, _ = bb._load_block("sgd-ocsvm", ds, str(data))
            D = bb.lane_arrays("sgd-ocsvm", B)
            loaded[ds] = _restandardize(D["X"], D["Xq"])
        X, Xq = loaded[ds]
        if shift:
            X, Xq = (X + np.float32(shift)).astype(np.float32), (Xq + np.float32(shift)).astype(np.float32)
        key = "%s_s%d_sh%d" % (ds, seed, int(shift))
        m = ml.SGDOneClassSVM(nu=NU, max_iter=20, tol=None, random_state=seed).fit(X)
        w = np.asarray(m.coef_, np.float32).reshape(-1)
        rho = float(np.asarray(m.offset_).reshape(-1)[0])
        dec = np.asarray(m.decision_function(Xq), np.float32).reshape(-1)
        pred = np.asarray(m.predict(Xq)).reshape(-1)
        out[key + "_coef"], out[key + "_offset"] = w, np.array([rho], np.float32)
        out[key + "_dec"], out[key + "_pred"] = dec, pred.astype(np.int8)
        out[key + "_J"] = np.array([objective(X, w, rho)])
        if not shift:
            k = SkOC(nu=NU, max_iter=20, tol=None, random_state=seed).fit(X)
            out[key + "_skff"] = np.array([float((k.predict(Xq) < 0).mean())])
            out[key + "_skJ"] = np.array([objective(X, k.coef_.reshape(-1), float(k.offset_[0]))])
            out[key + "_sknw"] = np.array([float(np.linalg.norm(k.coef_))])
        print("SGDOC-TAIL-FIT %s n=%d d=%d ff=%.5f J=%.9f |w|=%.3e offset=%.3e" % (
            key, X.shape[0], X.shape[1], float((pred < 0).mean()), out[key + "_J"][0],
            float(np.linalg.norm(w)), rho), flush=True)
    np.savez(path, **out)
    print("SGDOC-TAIL-CAPTURE " + json.dumps(dict(meta, arrays=len(out)), sort_keys=True), flush=True)


def _jac(pa, pb):
    fa, fb = pa < 0, pb < 0
    u = int((fa | fb).sum())
    return 1.0 if u == 0 else float((fa & fb).sum()) / u


def compare(pa, pb):
    A, B = np.load(pa), np.load(pb)
    ok = True

    def gate(name, cond, text):
        nonlocal ok
        ok = ok and bool(cond)
        print("SGDOC-TAIL-GATE %s %s %s" % (name, "PASS" if cond else "FAIL", text), flush=True)

    for ds in DATASETS:
        ks = ["%s_s%d_sh0" % (ds, s) for s in SEEDS]
        ff = {a: [float((Z[k + "_pred"] < 0).mean()) for k in ks] for a, Z in (("A", A), ("B", B))}
        J = {a: [float(Z[k + "_J"][0]) for k in ks] for a, Z in (("A", A), ("B", B))}
        nw = {a: [float(np.linalg.norm(Z[k + "_coef"].astype(np.float64))) for k in ks] for a, Z in (("A", A), ("B", B))}
        sd = {a: [float(Z[k + "_dec"].astype(np.float64).std()) for k in ks] for a, Z in (("A", A), ("B", B))}
        jab = [_jac(A[k + "_pred"], B[k + "_pred"]) for k in ks]
        jaa = [_jac(A[ks[i] + "_pred"], A[ks[j] + "_pred"]) for i in range(len(ks)) for j in range(i + 1, len(ks))]
        diff = sum(int(not np.array_equal(A[k + "_coef"], B[k + "_coef"])) for k in ks)
        med = lambda v: float(np.median(v))
        print("SGDOC-TAIL-ROW %s ff_A=%s ff_B=%s sk_ff=%s" % (ds, np.round(ff["A"], 5).tolist(), np.round(ff["B"], 5).tolist(),
              [round(float(A[k + "_skff"][0]), 5) for k in ks]))
        print("SGDOC-TAIL-ROW %s J_A=%s J_B=%s sk_J=%s" % (ds, ["%.9f" % v for v in J["A"]], ["%.9f" % v for v in J["B"]],
              ["%.9f" % float(A[k + "_skJ"][0]) for k in ks]))
        print("SGDOC-TAIL-ROW %s |w|_A=%s |w|_B=%s sk_|w|=%s" % (ds, ["%.3e" % v for v in nw["A"]], ["%.3e" % v for v in nw["B"]],
              ["%.3e" % float(A[k + "_sknw"][0]) for k in ks]))
        print("SGDOC-TAIL-ROW %s jac_AB=%s jac_AA=%s" % (ds, np.round(jab, 4).tolist(), np.round(jaa, 4).tolist()))
        gate("C1-ff", min(ff["A"]) <= med(ff["B"]) <= max(ff["A"]), "%s median ff_B %.5f in A [%.5f, %.5f]" % (
            ds, med(ff["B"]), min(ff["A"]), max(ff["A"])))
        gate("C2-J", med(J["B"]) <= max(J["A"]), "%s median J_B %.9f <= max J_A %.9f" % (ds, med(J["B"]), max(J["A"])))
        gate("C3-w", 0.5 * min(nw["A"]) <= med(nw["B"]) <= 2 * max(nw["A"]), "%s median |w_B| %.3e in [%.3e, %.3e]" % (
            ds, med(nw["B"]), 0.5 * min(nw["A"]), 2 * max(nw["A"])))
        gate("C4-dec", 0.5 * min(sd["A"]) <= med(sd["B"]) <= 2 * max(sd["A"]), "%s median std dec_B %.3e in [%.3e, %.3e]" % (
            ds, med(sd["B"]), 0.5 * min(sd["A"]), 2 * max(sd["A"])))
        gate("C5-jac", med(jab) >= min(jaa), "%s median jac(B,A) %.4f >= min jac(A,A') %.4f" % (ds, med(jab), min(jaa)))
        gate("C6-ran", diff >= 4, "%s coef differs on %d/5 seeds" % (ds, diff))
    k = "taxi_s7_sh1"
    same = all(np.array_equal(A[k + s], B[k + s]) for s in ("_coef", "_offset", "_dec", "_pred"))
    gate("C7-refuse", same, "shifted taxi B == A exactly: %s" % same)
    print("SGDOC-TAIL-AB status=%s" % ("PASS" if ok else "FAIL"), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "dump":
        dump(sys.argv[2])
    elif len(sys.argv) == 4 and sys.argv[1] == "compare":
        sys.exit(compare(sys.argv[2], sys.argv[3]))
    else:
        raise SystemExit(__doc__)
