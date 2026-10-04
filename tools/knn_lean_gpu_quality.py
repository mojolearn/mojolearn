#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""KNN lean GPU quality: dump knn-imputer OUT.npz; compare knn-imputer A.npz B.npz.
31 arrays must be byte-identical: predictions, fitted flags, native column
counts and total. NumPy independently checks counts including zero rows,
wide all-missing/all-present inputs and pool reuse. Run on M3 only."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

FIXTURE = "knn-lean-gpu-v2"


def _sha(mod):
    return hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest()


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
        _, got_cm, got_total = m._nan_cells(m._masked(X), 1)
        out[f"k{i}_column_counts"] = np.array(got_cm, dtype=np.int32)
        out[f"k{i}_total"] = np.array([got_total], dtype=np.int32)
        cm = np.isnan(X).sum(axis=0)
        assert got_total == int(cm.sum()) and np.array_equal(got_cm, cm)
        assert list(m._valid) == [bool(c < n) for c in cm]
        assert list(m._miss_cols) == [f for f in range(d) if cm[f] > 0]
    # Empty fit is a public refusal, not a valid route through _masked/_f32.
    # Preserve and compare the exception rather than weakening validation.
    empty_X = np.zeros((0, 7), np.float32)
    try:
        ml.KNNImputer().fit(empty_X)
    except ValueError as exc:
        out["public_empty_refusal"] = np.array([json.dumps(
            {"type": type(exc).__name__, "message": str(exc)}, sort_keys=True).encode()])
    else:
        raise AssertionError("KNNImputer.fit must refuse a zero-row input")
    # Wide and empty count inputs exercise grid tails, zero initialization,
    # and shape changes in the pool without invoking unrelated transform work.
    for i, X in enumerate((np.zeros((0, 7), np.float32),
                           np.full((513, 257), np.nan, np.float32),
                           np.zeros((513, 257), np.float32))):
        m = ml.KNNImputer()
        if X.shape[0] == 0:
            # Native nan_cells explicitly supports n=0 and zeros its outputs.
            # Use live one-slot buffers rather than passing a null empty-array
            # address or routing through public nonempty-input validation.
            dummy = np.zeros(1, np.float32)
            cells = np.full(1, -17, np.int32)
            cm = np.full(X.shape[1], -17, np.int32)
            info = np.full(1, -17, np.int32)
            b.xn_nan_cells([int(a.ctypes.data) for a in (dummy, cells, cm, info)],
                           [0, X.shape[1], 1], [])
            cols, total = cm, int(info[0])
        else:
            _, cols, total = m._nan_cells(m._masked(X), 1)
        ref = np.isnan(X).sum(axis=0)
        assert np.array_equal(cols, ref) and total == int(ref.sum())
        out[f"edge{i}_columns"] = np.array(cols, dtype=np.int32)
        out[f"edge{i}_total"] = np.array([total], dtype=np.int32)
    return b, reach


CASES = {"knn-imputer": _knn}
COUNTS = {"knn-imputer": 31}


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
