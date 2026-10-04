#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""LU_FAST_TSLU quality: dump one arm's lu_factor / lu_solve outputs, then
compare arm B (candidate, -D MOJOLEARN_LU_FAST_TSLU) with arm A (main) on
the real outputs (tools/lu_fast_mma_quality.py's fixtures and rule).

  dump OUT.npz      (env MOJOLEARN_NUMERIC_MODE=fast, MOJOLEARN_VENDOR=apple;
                     whichever _mojolearn_x_decomp.so is installed)
  compare A.npz B.npz

Fixtures (seed 7 unless noted):
  board8192   the board's lu-factor / lu-solve system (tools/bench_board_algos.py
              lu_system: A = N(0,1) + 2 sqrt(n) I, B N(0,1) n x 64)
  boost{65,257,300,1000}  the same construction at smaller n (one outer block,
              one outer step + a 1-column tail, non-multiples of 4 and 32)
  plain{1000,2051}  A = N(0,1) with no diagonal boost: real row interchanges
  zero700     plain 700 with column 400 set to 0: an exactly-zero pivot
              (info = 401 in both arms; main skips that step)

Metrics per fixture, in float64 from the returned (lu, piv):
  factor_residual = ||P A - L U||_F / ||A||_F
  solve_residual  = ||A X - B||_F / ||B||_F   (the board's relative_residual)
  info            = the first zero pivot (1-based, 0 = none)

PASS rule, fixed before any result: for every fixture and both residuals,
  r_B <= max(1.5 r_A, r_A + 2e-7), info_B == info_A, every output finite;
and at least one fixture's lu bytes must DIFFER between the arms (the define
flipped: tournament pivots differ from partial pivoting's wherever there are
real row interchanges, plain1000 / plain2051; on the diagonally boosted
board system both pick the diagonal and the panel arithmetic is the same
chain, so board8192 may legitimately match bit for bit).
Why 1.5x: tournament pivoting keeps an LU backward-error bound of the
same form (c n u growth), with CALU's growth factor, measured close to
partial pivoting's on random matrices (Grigori, Demmel, Xiang 2011); correct f32 LUs on the board system already spread 4x
(torch 8.2e-7, scipy 3.25e-6, ours 3.26e-6). 1.5x is tighter than that
spread; +2e-7 absolute covers residuals near f32 resolution on small n.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

FIXTURE = "lu-fast-tslu-v1"


def _systems():
    out = []
    for n in (65, 257, 300, 1000, 8192):
        rng = np.random.default_rng(7)
        A = rng.standard_normal((n, n)).astype(np.float32)
        A[np.arange(n), np.arange(n)] += np.float32(2.0 * np.sqrt(n))
        B = rng.standard_normal((n, 64)).astype(np.float32)
        out.append(("board8192" if n == 8192 else "boost%d" % n, A, B))
    for n in (1000, 2051):
        rng = np.random.default_rng(11)
        out.append(("plain%d" % n, rng.standard_normal((n, n)).astype(np.float32),
                    rng.standard_normal((n, 64)).astype(np.float32)))
    rng = np.random.default_rng(13)
    A = rng.standard_normal((700, 700)).astype(np.float32)
    A[:, 400] = 0
    out.append(("zero700", A, rng.standard_normal((700, 64)).astype(np.float32)))
    return out


def _perm_rows(piv, n):
    """The row order getrf's swaps make: (P A)[i] = A[order[i]]."""
    order = np.arange(n)
    for k, p in enumerate(piv):
        if p != k:
            order[[k, p]] = order[[p, k]]
    return order


def _metrics(A, B, lu, piv, X):
    n = A.shape[0]
    L = np.tril(lu.astype(np.float64), -1) + np.eye(n)
    U = np.triu(lu.astype(np.float64))
    PA = A.astype(np.float64)[_perm_rows(piv, n)]
    fr = float(np.linalg.norm(PA - L @ U) / np.linalg.norm(PA))
    sr = float(np.linalg.norm(A.astype(np.float64) @ X.astype(np.float64) - B) / np.linalg.norm(B))
    return fr, sr


def dump(path):
    assert os.environ.get("MOJOLEARN_NUMERIC_MODE") == "fast"
    assert os.environ.get("MOJOLEARN_VENDOR") == "apple"
    import warnings
    import mojolearn as ml
    from mojolearn import _backend
    b = _backend.binding("_mojolearn_x_decomp", "fast")
    raw = getattr(b, "_b", b)
    so = Path(getattr(raw, "__file__", "") or (Path(ml.__file__).parent / "_mojolearn_x_decomp.so"))
    assert int(b.x_decomp_numeric_mode()) == 0, "x_decomp binding is not FAST"
    vendor = str(b.x_decomp_vendor())
    lu_factor = getattr(ml, "lu_factor", None) or ml.linalg.lu_factor
    lu_solve = getattr(ml, "lu_solve", None) or ml.linalg.lu_solve
    arrays, rows = {}, []
    for name, A, B in _systems():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pair = lu_factor(A)
        X = np.asarray(lu_solve(pair, B), dtype=np.float32)  # the board's two-call form
        lu = np.array(pair[0], dtype=np.float32)
        piv = np.array(pair[1], dtype=np.int64)
        info = 0
        for w in caught:
            msg = str(w.message)
            if msg.startswith("Diagonal number "):
                info = int(msg.split()[2])
        fr, sr = _metrics(A, B, lu, piv, X)
        finite = bool(np.isfinite(lu).all() and np.isfinite(X).all())
        arrays[name + "_lu"] = lu
        arrays[name + "_piv"] = piv
        arrays[name + "_x"] = X
        row = dict(fixture=name, n=A.shape[0], factor_residual=fr, solve_residual=sr, info=info, finite=finite)
        rows.append(row)
        print("LU-TSLU-DUMP " + json.dumps(row), flush=True)
    np.savez(path, **arrays)
    meta = dict(fixture=FIXTURE, vendor=vendor, binding_sha256=hashlib.sha256(so.read_bytes()).hexdigest()
                if so.is_file() else "", rows=rows)
    Path(str(path) + ".json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    print("LU-TSLU-CAPTURE " + json.dumps(dict(fixture=FIXTURE, vendor=vendor, binding_sha256=meta["binding_sha256"],
                                               fixtures=len(rows))), flush=True)


def compare(pa, pb):
    ma = json.loads(Path(pa + ".json").read_text())
    mb = json.loads(Path(pb + ".json").read_text())
    assert ma["fixture"] == mb["fixture"] == FIXTURE
    ra = {r["fixture"]: r for r in ma["rows"]}
    rb = {r["fixture"]: r for r in mb["rows"]}
    assert sorted(ra) == sorted(rb)
    ok = True
    for name in sorted(ra):
        a, b = ra[name], rb[name]
        good = a["finite"] and b["finite"] and a["info"] == b["info"]
        for key in ("factor_residual", "solve_residual"):
            good = good and b[key] <= max(1.5 * a[key], a[key] + 2e-7)
        ok = ok and good
        print("LU-TSLU-AB " + json.dumps(dict(
            fixture=name, status="OK" if good else "FAIL", info_A=a["info"], info_B=b["info"],
            factor_A=a["factor_residual"], factor_B=b["factor_residual"],
            solve_A=a["solve_residual"], solve_B=b["solve_residual"])), flush=True)
    za, zb = np.load(pa), np.load(pb)
    differs = {name: za[name + "_lu"].tobytes() != zb[name + "_lu"].tobytes() for name in sorted(ra)}
    flipped = any(differs.values())
    piv_same = float(np.mean(za["board8192_piv"] == zb["board8192_piv"]))
    xa, xb = za["board8192_x"].astype(np.float64), zb["board8192_x"].astype(np.float64)
    xdiff = float(np.linalg.norm(xb - xa) / np.linalg.norm(xa))
    print("LU-TSLU-AB-INFO " + json.dumps(dict(lu_differs=differs, board8192_piv_agree=piv_same,
                                              board8192_x_rel_diff=xdiff)), flush=True)
    ok = ok and flipped
    print("LU-TSLU-AB status=%s fixtures=%d define_flipped=%s" % ("PASS" if ok else "FAIL", len(ra), flipped))
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["dump", "compare"])
    p.add_argument("first")
    p.add_argument("second", nargs="?")
    args = p.parse_args()
    if args.action == "dump":
        dump(args.first)
        return 0
    return compare(args.first, args.second)


if __name__ == "__main__":
    sys.exit(main())
