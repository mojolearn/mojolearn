#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""MOJOLEARN_PSHAP_DELTA quality: dump Permutation SHAP values of one
x_trees build, then compare arm A (main) with arm B (the define).

    pshap_delta_quality.py dump OUT.npz
    pshap_delta_quality.py compare A.npz B.npz

Fixture (seed 913): X 4000 x 40 shaped like istella's sparse columns:
columns 0-19 are 70% exact zeros, columns 20-24 hold 3 distinct values,
column 25 is constant, the rest dense normals; background 50 rows, 67
explained rows (20 rows a chunk: 4 chunks, a 7-row tail). So many
(coalition, background row) pairs repeat their predecessor (the delta path
is exercised) and many vary (the compaction and the map are exercised).
Cases: a linear model (exact SHAP values known: w_j (x_j - E_bg x_j)), a
nonlinear tanh model, a 2-output model. Each model counts the rows it is
called on (arm B must call it on FEWER rows: proof the path ran; not a
quality metric).

Tolerance, fixed before any result: arm B runs the model on a subset of
arm A's synthetic rows, bit-identical row inputs, and every (coalition,
row) reads the output of an identical row; the means and marginals are the
same soft-float adds in the same order. The only freedom is the model's
own batch dependence (a BLAS kernel may round a row differently in a batch
of another size), which moves a float64 output by an ulp and so its
float32 cast by at most one float32 ulp. The gate per case: max |phi_B -
phi_A| <= 4 * 2^-24 * max |f(bg)| (four float32 ulps of the model output
scale; phi is a difference of two means of such outputs averaged over 2 *
npermutations marginals, so one flipped output moves phi by far less),
additivity error B <= additivity A + the same bound, linear relative error
vs exact B <= A + 1e-6. Byte identity is reported (expected for these
numpy models) but not required. Shape or finiteness alone is never the
gate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

SEED = 913
N, D, NB, NQ = 4000, 40, 50, 67
FIXTURE = "pshap-delta-v1-seed913"
CASES = ("perm_linear", "perm_tanh", "perm_two")


def _fixture():
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(N, D)).astype(np.float32)
    X[:, :20] *= (rng.random(size=(N, 20)) >= 0.7)
    X[:, 20:25] = rng.integers(0, 3, size=(N, 5)).astype(np.float32)
    X[:, 25] = 1.5
    bg = X[(np.arange(NB) * N) // NB].copy()
    Xq = X[1000:1000 + NQ].copy()
    w = rng.normal(size=D)
    W1 = rng.normal(size=(D, 8)) / np.sqrt(D)
    w2 = rng.normal(size=8)
    W2 = rng.normal(size=(D, 2)) / np.sqrt(D)
    return X, bg, Xq, w, W1, w2, W2


class _Counted:
    def __init__(self, fn):
        self.fn, self.rows = fn, 0

    def __call__(self, Z):
        self.rows += int(Z.shape[0])
        return self.fn(Z)


def _models(w, W1, w2, W2):
    def linear(Z):
        return np.asarray(Z, dtype=np.float64) @ w + 0.25

    def tanh(Z):
        return np.tanh(np.asarray(Z, dtype=np.float64) @ W1) @ w2

    def two(Z):
        z = np.asarray(Z, dtype=np.float64) @ W2
        return np.stack([z[:, 0], np.sin(z[:, 1])], axis=1)
    return dict(perm_linear=linear, perm_tanh=tanh, perm_two=two)


def _f(fn, Z):
    out = np.asarray(fn(Z), dtype=np.float64)
    return out.reshape(out.shape[0], -1)


def dump(path):
    assert os.environ.get("MOJOLEARN_NUMERIC_MODE") == "fast"
    from mojolearn._expansion_trees import PermutationExplainer, _trees_x_bind
    b = _trees_x_bind()
    assert str(b.x_trees_vendor()).lower() in ("metal", "apple"), b.x_trees_vendor()
    switches = int(b.x_trees_fast_switches())
    meta = dict(binding_sha256=hashlib.sha256(Path(b.__file__).read_bytes()).hexdigest(),
                delta=(switches & 32) != 0, batch=(switches & 8) != 0, fixture=FIXTURE)
    X, bg, Xq, w, W1, w2, W2 = _fixture()
    models = _models(w, W1, w2, W2)
    out = {}
    for tag in CASES:
        fn = _Counted(models[tag])
        ex = PermutationExplainer(fn, bg, random_state=SEED)
        fn.rows = 0
        phi = np.asarray(ex.shap_values(Xq, npermutations=10), dtype=np.float64)
        assert phi.shape[:2] == (NQ, D), (tag, phi.shape)
        base = np.atleast_1d(np.asarray(ex.expected_value, dtype=np.float64))
        fx = _f(fn.fn, Xq)
        tot = phi.reshape(NQ, D, -1).sum(axis=1) + base[None, :]
        out[tag] = phi
        out[tag + "_additivity"] = np.array([np.abs(tot - fx).max()])
        out[tag + "_scale"] = np.array([np.abs(_f(fn.fn, bg)).max()])
        out[tag + "_rows"] = np.array([fn.rows])
        if tag == "perm_linear":
            exact = w[None, :] * (Xq.astype(np.float64) - bg.astype(np.float64).mean(0)[None, :])
            out[tag + "_relerr"] = np.array([np.linalg.norm(phi - exact) / np.linalg.norm(exact)])
    np.savez(path, **out)
    meta["arrays"] = len(out)
    print("PSHAP-DELTA-CAPTURE " + json.dumps(meta, sort_keys=True))
    for key in sorted(out):
        if not key.endswith(CASES):
            print(f"PSHAP-DELTA-METRIC {key}={float(out[key][0]):.6e}")
    print(f"PSHAP-DELTA-DUMP status=PASS arrays={len(out)} path={path}")


def compare(pa, pb):
    a, b = np.load(pa), np.load(pb)
    assert sorted(a.files) == sorted(b.files) and len(a.files) == 13, (a.files, b.files)
    fails = []
    for tag in CASES:
        x, y = a[tag], b[tag]
        assert x.dtype == y.dtype and x.shape == y.shape, tag
        bound = 4.0 * 2.0 ** -24 * float(a[tag + "_scale"][0])
        diff = float(np.abs(x - y).max())
        add_a, add_b = float(a[tag + "_additivity"][0]), float(b[tag + "_additivity"][0])
        rows_a, rows_b = int(a[tag + "_rows"][0]), int(b[tag + "_rows"][0])
        same = x.tobytes() == y.tobytes()
        line = (f"PSHAP-DELTA-DIFF {tag} max_abs={diff:.3e} bound={bound:.3e} byte_identical={same} "
                f"additivity A={add_a:.3e} B={add_b:.3e} model_rows A={rows_a} B={rows_b} "
                f"ratio={rows_b / max(rows_a, 1):.4f}")
        if tag + "_relerr" in a.files:
            ra, rb = float(a[tag + "_relerr"][0]), float(b[tag + "_relerr"][0])
            line += f" relerr A={ra:.6e} B={rb:.6e}"
            if not rb <= ra + 1e-6:
                fails.append(tag + " relerr")
        print(line)
        if not diff <= bound:
            fails.append(tag + " phi")
        if not add_b <= add_a + bound:
            fails.append(tag + " additivity")
        if not rows_b < rows_a:
            fails.append(tag + " delta path did not run (arm B model rows not fewer)")
    if fails:
        print("PSHAP-DELTA-AB status=FAIL " + "; ".join(fails))
        raise SystemExit(1)
    print(f"PSHAP-DELTA-AB status=PASS cases={len(CASES)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["dump", "compare"])
    p.add_argument("first")
    p.add_argument("second", nargs="?")
    args = p.parse_args()
    if args.action == "dump":
        dump(args.first)
    else:
        compare(args.first, args.second)


if __name__ == "__main__":
    main()
