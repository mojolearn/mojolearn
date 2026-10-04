#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Quality dump / exact A-B comparison for lane apple-fast-w4-small.

 dump CASE OUT.npz      (one arm installed; MOJOLEARN_NUMERIC_MODE=fast)
 compare CASE A.npz B.npz

CASE: var | knn-imputer | maxabs | resample. Every candidate of the lane
moves fixed costs only (one launch for eight with the same per-word chains,
pooled buffers, a native row copy): the outputs must be BYTE-IDENTICAL to
main's (tolerance 0, fixed before any result; a different word is a bug).
The dump also checks each output against an independent float64 / numpy
reference with a loose bound, so an arm pair that agrees on garbage fails.
Prints W4SMALL-CAPTURE {json} (binding sha, the candidate's reach flag) and
W4SMALL-QUALITY / W4SMALL-AB status lines."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

FIXTURE = "w4-small-v1"


def _sha(mod):
    return hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest()


def _var(out):
    import mojolearn as ml
    from mojolearn import _backend
    b = _backend.binding("_mojolearn_x_sequence", "fast")
    reach = int(getattr(b, "x_sequence_var_fused", lambda: 0)())
    rng = np.random.default_rng(71)
    shapes = [(1392, 16, 2, "c"), (1392, 16, 1, "n"), (700, 7, 3, "c"), (300, 3, 2, "c"),
              (40000, 2, 1, "c")]  # the last: R >= 32768, main's path on both arms
    for i, (n, K, p, trend) in enumerate(shapes):
        e = rng.normal(size=(n, K)).astype(np.float32)
        y = np.empty_like(e)
        y[0] = e[0]
        for t in range(1, n):  # fixture generation, not the measured path
            y[t] = 0.5 * y[t - 1] + e[t] + (3.0 if trend == "c" else 0.0) * 0.1
        y = y.astype(np.float32) * np.float32(1.0 + i)
        for rep in range(2):  # the second fit reuses pooled buffers
            r = ml.VAR(y, numeric_mode="fast").fit(maxlags=p, trend=trend)
            fc = np.asarray(r.forecast(y[-p:], 48))
            tag = f"s{i}_r{rep}_"
            out[tag + "params"] = np.asarray(r.params)
            out[tag + "sigma_u"] = np.asarray(r.sigma_u)
            out[tag + "resid"] = np.asarray(r.resid)
            out[tag + "forecast"] = fc
        # independent float64 OLS: the same design, lstsq
        kt = 1 if trend == "c" else 0
        rows = [np.ones((n - p, 1))] if kt else []
        rows += [y[p - l:n - l].astype(np.float64) for l in range(1, p + 1)]
        Z = np.hstack(rows)
        B = np.linalg.lstsq(Z, y[p:].astype(np.float64), rcond=None)[0]
        P = np.asarray(r.params, dtype=np.float64)
        rel = np.abs(P - B).max() / max(np.abs(B).max(), 1e-12)
        assert rel < 5e-3, (tag, rel)  # float32 normal equations vs float64 lstsq
        res = y[p:].astype(np.float64) - Z @ B
        assert np.allclose(np.asarray(r.resid, dtype=np.float64), res, atol=5e-3 * np.abs(y).max()), tag
    # a rank-deficient design (two equal series): both arms refuse
    z = rng.normal(size=(500, 3)).astype(np.float32)
    z[:, 2] = z[:, 1]
    try:
        ml.VAR(z, numeric_mode="fast").fit(maxlags=1)
        out["rankdef"] = np.array([0], dtype=np.int32)
    except np.linalg.LinAlgError as err:
        out["rankdef"] = np.frombuffer(str(err).encode(), dtype=np.uint8).copy()
    return b, reach


def _knn(out):
    import mojolearn as ml
    from mojolearn import _backend
    b = _backend.binding("_mojolearn_x_neighbors", "fast")
    reach = int(getattr(b, "x_neighbors_nc_fit_lean", lambda: 0)())
    rng = np.random.default_rng(72)
    for i, (n, d) in enumerate([(5000, 10), (3001, 7), (5000, 10)]):  # the third reuses the pool
        X = rng.normal(size=(n, d)).astype(np.float32)
        X[rng.random(X.shape) < 0.1] = np.nan
        X[:, 3] = np.nan if i == 1 else X[:, 3]   # an all-missing column
        Xq = rng.normal(size=(400, d)).astype(np.float32)
        Xq[rng.random(Xq.shape) < 0.1] = np.nan
        for keep in (False, True):
            m = ml.KNNImputer(n_neighbors=5, keep_empty_features=keep, add_indicator=keep)
            m.fit(X)
            out[f"k{i}_{int(keep)}_t"] = np.asarray(m.transform(Xq), dtype=np.float32)
            out[f"k{i}_{int(keep)}_valid"] = np.array(m._valid, dtype=np.int8)
            out[f"k{i}_{int(keep)}_miss"] = np.array(m._miss_cols, dtype=np.int32)
        # independent: the fit's column flags are numpy's NaN counts
        cm = np.isnan(X).sum(axis=0)
        assert list(m._valid) == [bool(c < n) for c in cm]
        assert list(m._miss_cols) == [f for f in range(d) if cm[f] > 0]
    return b, reach


def _maxabs(out):
    import mojolearn as ml
    from mojolearn._expansion_prep import _prep_binding
    b = _prep_binding("fast")
    reach = int(getattr(b, "x_prep_maxabs_pool", lambda: 0)())
    rng = np.random.default_rng(73)
    for i, (n, d) in enumerate([(20000, 220), (20000, 220), (5003, 9), (20000, 220)]):
        X = (rng.normal(size=(n, d)) * rng.uniform(0.1, 50, size=d)).astype(np.float32)
        X[rng.random(X.shape) < 0.01] = np.nan
        X[:, 1] = 0.0                      # a zero column scales by one
        m = ml.MaxAbsScaler().fit(X)
        out[f"m{i}_max"] = np.asarray(m.max_abs_)
        out[f"m{i}_scale"] = np.asarray(m.scale_)
        out[f"m{i}_t"] = np.asarray(m.transform(X[:500]))
        ref = np.nanmax(np.abs(X), axis=0)
        assert np.array_equal(np.asarray(m.max_abs_), ref.astype(np.float32)), i  # a max is exact
    return b, reach


def _resample(out):
    import mojolearn as ml
    from mojolearn import _backend
    from mojolearn.resample import resample, resample_indices
    b = _backend.binding("_mojolearn_resample", "fast")
    reach = int(hasattr(b, "resample_gather_rows"))
    rng = np.random.default_rng(74)
    X = rng.normal(size=(100000, 10)).astype(np.float32)
    Xw = rng.normal(size=(20000, 220)).astype(np.float32)
    y = rng.normal(size=100000).astype(np.float32)
    yi = rng.integers(0, 9, size=100000).astype(np.int64)
    X3 = rng.normal(size=(100000, 2, 3)).astype(np.float32)
    Xf = np.asfortranarray(X)               # not C-contiguous: main's gather on both arms
    cases = {"Xy": (X, y), "w": (Xw,), "i64": (yi,), "x3": (X3,), "f": (Xf,), "y2": (np.c_[y, y],)}
    for name, arrays in cases.items():
        for ns, seed in ((None, 7), (777, 11)):
            res = resample(*arrays, n_samples=ns, random_state=seed, numeric_mode="fast")
            res = res if isinstance(res, list) else [res]
            idx = np.asarray(resample_indices(len(arrays[0]), ns, True, seed, numeric_mode="fast"), dtype=np.int64)
            for j, (a, r) in enumerate(zip(arrays, res)):
                r = np.asarray(r)
                assert r.dtype == a.dtype and np.array_equal(r, a[idx]), (name, j)  # numpy reference
                out[f"r_{name}_{ns}_{j}"] = np.ascontiguousarray(r)
    lst = resample([1.0, 2.0, 3.0, 4.0], random_state=3, numeric_mode="fast")
    out["r_list"] = np.asarray(lst, dtype=np.float64)
    return b, reach


CASES = {"var": _var, "knn-imputer": _knn, "maxabs": _maxabs, "resample": _resample}
COUNTS = {"var": 41, "knn-imputer": 18, "maxabs": 12, "resample": 15}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["dump", "compare"])
    p.add_argument("case", choices=sorted(CASES))
    p.add_argument("first")
    p.add_argument("second", nargs="?")
    args = p.parse_args()
    if args.action == "dump":
        assert os.environ.get("MOJOLEARN_NUMERIC_MODE") == "fast"
        out = {}
        b, reach = CASES[args.case](out)
        np.savez(args.first, **out)
        print("W4SMALL-CAPTURE " + json.dumps(dict(case=args.case, binding_sha256=_sha(b), reach=reach,
                                                   fixture=FIXTURE, arrays=len(out)), sort_keys=True))
        print(f"W4SMALL-QUALITY case={args.case} status=PASS arrays={len(out)} path={args.first}")
        return
    a, b = np.load(args.first), np.load(args.second)
    assert sorted(a.files) == sorted(b.files) and len(a.files) == COUNTS[args.case], (len(a.files), len(b.files))
    bad = [k for k in a.files if a[k].dtype != b[k].dtype or a[k].shape != b[k].shape
           or a[k].tobytes() != b[k].tobytes()]
    for k in bad[:10]:
        x, y = a[k], b[k]
        diff = (np.abs(x.astype(np.float64) - y.astype(np.float64)).max()
                if x.shape == y.shape and x.dtype.kind == "f" else "shape/dtype")
        print(f"W4SMALL-AB case={args.case} DIFFER key={k} max_abs_diff={diff}")
    status = "PASS" if not bad else "FAIL"
    print(f"W4SMALL-AB case={args.case} status={status} exact_arrays={len(a.files) - len(bad)} of={len(a.files)}")
    raise SystemExit(0 if not bad else 1)


if __name__ == "__main__":
    main()
