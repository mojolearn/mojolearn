#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CHOL_FAST_TALL quality: dump one arm's Cholesky outputs, then compare
arm B (candidate) with arm A (main) on the real outputs.

  dump OUT.npz      (env MOJOLEARN_NUMERIC_MODE=fast, MOJOLEARN_VENDOR=apple;
                     whichever _mojolearn_gp.so is installed)
  compare A.npz B.npz

Fixtures (tools/bench_board_algos.py sym_system: A = (M + M^T)/2 + 2 sqrt(n) I,
M N(0,1) seed 7; B N(0,1) n x 64 seed 8):
  board8192         the board's cholesky system (32 outer panels of 256)
  sym{65,300,2051,4100}  one inner step + a 1-column tail, one partial outer
                    panel, a 3-column last panel, a 4-column last panel
  fail2500          (M + M^T)/2 + 0.5 sqrt(n) I: indefinite, the factor
                    stops (info > 0); both arms redo it on main's route, so
                    info and the partial L must be the SAME BYTES

Metrics per fixture, float64 from the returned L_:
  factor_residual = ||L L^T - A||_F / ||A||_F  (the board's relative_residual)
  solve_residual  = ||A X - B||_F / ||B||_F    (X = Cholesky.solve(B))
  logdet          = Cholesky.logdet_
  info            = Cholesky.info_

PASS rule, fixed before any result: for every SPD fixture
  factor_B <= max(1.5 factor_A, factor_A + 5e-8),
  solve_B  <= max(1.5 solve_A,  solve_A  + 5e-8),
  |logdet_B - logdet_A| <= 1e-4 |logdet_A|,
  info_B == info_A == 0, every output finite, L_B upper triangle exactly 0;
fail2500: info_B == info_A > 0 and L_B bytes == L_A bytes;
and the board8192 L bytes must DIFFER between the arms (the define flipped).
Why 1.5x / 5e-8: a reordered f32 blocked Cholesky keeps the backward-error
bound (c n u); correct f32 factors of the board system already spread 3.3x
(torch 5.45e-7, numpy 1.659e-7, ours 1.659e-7). 1.5x is well inside that;
5e-8 absolute covers small-n residuals near f32 resolution. logdet: a sum of
n f32 logs whose terms move by ~1e-7 each; 1e-4 relative sits between the
typical sqrt(n) u and the worst-case n u (5e-4 at n = 8192) summation bound.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

FIXTURE = "chol-fast-tall-v1"


def _sym(n, shift):
    rng = np.random.default_rng(7)
    M = rng.standard_normal((n, n)).astype(np.float32)
    A = (M + M.T) * np.float32(0.5)
    A[np.arange(n), np.arange(n)] += np.float32(shift * np.sqrt(n))
    B = np.random.default_rng(8).standard_normal((n, 64)).astype(np.float32)
    return A, B


def _systems():
    out = []
    for n in (65, 300, 2051, 4100, 8192):
        A, B = _sym(n, 2.0)
        out.append(("board8192" if n == 8192 else "sym%d" % n, A, B))
    A, B = _sym(2500, 0.5)
    out.append(("fail2500", A, B))
    return out


def dump(path):
    assert os.environ.get("MOJOLEARN_NUMERIC_MODE") == "fast"
    assert os.environ.get("MOJOLEARN_VENDOR") == "apple"
    import mojolearn as ml
    from mojolearn import _backend
    b = _backend.binding("_mojolearn_gp", "fast")
    raw = getattr(b, "_b", b)
    so = Path(getattr(raw, "__file__", "") or (Path(ml.__file__).parent / "_mojolearn_gp.so"))
    assert int(b.gp_numeric_mode()) == 0, "gp binding is not FAST"
    vendor = str(b.gp_vendor())
    Cholesky = getattr(ml, "Cholesky", None) or ml.linalg.Cholesky
    arrays, rows = {}, []
    for name, A, B in _systems():
        ch = Cholesky(jitter=0.0).fit(A)
        L = np.array(ch.L_, dtype=np.float32)
        info = int(ch.info_)
        row = dict(fixture=name, n=A.shape[0], info=info)
        if info == 0:
            X = np.asarray(ch.solve(B), dtype=np.float32)
            L64 = L.astype(np.float64)
            A64 = A.astype(np.float64)
            row.update(
                factor_residual=float(np.linalg.norm(L64 @ L64.T - A64) / np.linalg.norm(A64)),
                solve_residual=float(np.linalg.norm(A64 @ X.astype(np.float64) - B) / np.linalg.norm(B)),
                logdet=float(ch.logdet_),
                upper_zero=bool(not np.triu(L, 1).any()),
                finite=bool(np.isfinite(L).all() and np.isfinite(X).all()),
            )
            arrays[name + "_x"] = X
        arrays[name + "_L"] = L
        rows.append(row)
        print("CHOL-TALL-DUMP " + json.dumps(row), flush=True)
    np.savez(path, **arrays)
    meta = dict(fixture=FIXTURE, vendor=vendor, binding_sha256=hashlib.sha256(so.read_bytes()).hexdigest()
                if so.is_file() else "", rows=rows)
    Path(str(path) + ".json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    print("CHOL-TALL-CAPTURE " + json.dumps(dict(fixture=FIXTURE, vendor=vendor, binding_sha256=meta["binding_sha256"],
                                                  fixtures=len(rows))), flush=True)


def compare(pa, pb):
    ma = json.loads(Path(pa + ".json").read_text())
    mb = json.loads(Path(pb + ".json").read_text())
    assert ma["fixture"] == mb["fixture"] == FIXTURE
    ra = {r["fixture"]: r for r in ma["rows"]}
    rb = {r["fixture"]: r for r in mb["rows"]}
    assert sorted(ra) == sorted(rb)
    za, zb = np.load(pa), np.load(pb)
    ok = True
    for name in sorted(ra):
        a, b = ra[name], rb[name]
        if name.startswith("fail"):
            good = a["info"] > 0 and b["info"] == a["info"] and \
                za[name + "_L"].tobytes() == zb[name + "_L"].tobytes()
            print("CHOL-TALL-AB " + json.dumps(dict(fixture=name, status="OK" if good else "FAIL",
                                                     info_A=a["info"], info_B=b["info"])), flush=True)
            ok = ok and good
            continue
        good = a["info"] == 0 and b["info"] == 0 and a["finite"] and b["finite"] and b["upper_zero"]
        for key in ("factor_residual", "solve_residual"):
            good = good and b[key] <= max(1.5 * a[key], a[key] + 5e-8)
        good = good and abs(b["logdet"] - a["logdet"]) <= 1e-4 * abs(a["logdet"])
        ok = ok and good
        print("CHOL-TALL-AB " + json.dumps(dict(
            fixture=name, status="OK" if good else "FAIL", info_A=a["info"], info_B=b["info"],
            factor_A=a["factor_residual"], factor_B=b["factor_residual"],
            solve_A=a["solve_residual"], solve_B=b["solve_residual"],
            logdet_A=a["logdet"], logdet_B=b["logdet"])), flush=True)
    flipped = za["board8192_L"].tobytes() != zb["board8192_L"].tobytes()
    la, lb = za["board8192_L"].astype(np.float64), zb["board8192_L"].astype(np.float64)
    print("CHOL-TALL-AB-INFO " + json.dumps(dict(board8192_L_differs=flipped,
                                                  board8192_L_rel_diff=float(np.linalg.norm(lb - la) / np.linalg.norm(la)))),
          flush=True)
    ok = ok and flipped
    print("CHOL-TALL-AB status=%s fixtures=%d define_flipped=%s" % ("PASS" if ok else "FAIL", len(ra), flipped))
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
