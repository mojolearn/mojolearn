#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""MOJOLEARN_SHAP_FAST_PIPE quality: dump Kernel/Permutation SHAP values of
one x_trees build, then compare arm A (main) with arm B (the define).

    shap_pipe_quality.py dump OUT.npz
    shap_pipe_quality.py compare A.npz B.npz

Fixture (seed 911): X 4000 x 40, background 50 rows, 67 explained rows, so
the explainers run several chunks with a short tail (permutation: 20 rows a
chunk, 4 chunks; kernel: 7 rows a chunk, 10 chunks), i.e. the double buffer
swaps slots and reallocates for the tail. Cases: a linear model (exact SHAP
values known: w_j (x_j - E_bg x_j)), a nonlinear tanh model, a 2-output
model, KernelExplainer with link="logit" on probabilities.

Tolerance, fixed before any result: the candidate only moves WHEN the host
waits for each chunk's copy; every unit runs on the same inputs in the same
order. So phi must be BYTE-identical to arm A (max abs diff 0). The report
also prints, per case and arm, the additivity error max |sum(phi) + base -
f(x)| and, for the linear case, the relative error vs the exact values;
these must equal arm A's (they are functions of phi). Shape or finiteness
alone is never the gate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

SEED = 911
N, D, NB, NQ = 4000, 40, 50, 67


def _fixture():
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(N, D)).astype(np.float32)
    X[:, :5] *= 3.0
    bg = X[(np.arange(NB) * N) // NB].copy()
    Xq = X[1000:1000 + NQ].copy()
    w = rng.normal(size=D)
    W1 = rng.normal(size=(D, 8)) / np.sqrt(D)
    w2 = rng.normal(size=8)
    W2 = rng.normal(size=(D, 2)) / np.sqrt(D)
    return X, bg, Xq, w, W1, w2, W2


def _models(w, W1, w2, W2):
    def linear(Z):
        return np.asarray(Z, dtype=np.float64) @ w + 0.25

    def tanh(Z):
        return np.tanh(np.asarray(Z, dtype=np.float64) @ W1) @ w2

    def two(Z):
        z = np.asarray(Z, dtype=np.float64) @ W2
        return np.stack([z[:, 0], np.sin(z[:, 1])], axis=1)

    def proba(Z):
        p = 1.0 / (1.0 + np.exp(-(np.asarray(Z, dtype=np.float64) @ w) / 4.0))
        return np.stack([1.0 - p, p], axis=1)
    return dict(linear=linear, tanh=tanh, two=two, proba=proba)


def _f(fn, Z):
    out = np.asarray(fn(Z), dtype=np.float64)
    return out.reshape(out.shape[0], -1)


def dump(path):
    assert os.environ.get("MOJOLEARN_NUMERIC_MODE") == "fast"
    from mojolearn._expansion_trees import KernelExplainer, PermutationExplainer, _trees_x_bind
    b = _trees_x_bind()
    assert str(b.x_trees_vendor()).lower() in ("metal", "apple"), b.x_trees_vendor()
    switches = int(b.x_trees_fast_switches())
    meta = dict(binding_sha256=hashlib.sha256(Path(b.__file__).read_bytes()).hexdigest(),
                pipe=(switches & 16) != 0, batch=(switches & 8) != 0, fixture="shap-pipe-v1-seed911")
    X, bg, Xq, w, W1, w2, W2 = _fixture()
    models = _models(w, W1, w2, W2)
    out = {}
    cases = [("perm_linear", "perm", "linear", {}), ("perm_tanh", "perm", "tanh", {}),
             ("perm_two", "perm", "two", {}),
             ("kern_linear", "kern", "linear", {}), ("kern_tanh", "kern", "tanh", {}),
             ("kern_two", "kern", "two", {}), ("kern_logit", "kern", "proba", {"link": "logit"})]
    for tag, kind, mname, kw in cases:
        fn = models[mname]
        if kind == "perm":
            ex = PermutationExplainer(fn, bg, random_state=SEED)
            phi = np.asarray(ex.shap_values(Xq, npermutations=10), dtype=np.float64)
        else:
            ex = KernelExplainer(fn, bg, random_state=SEED, **kw)
            phi = np.asarray(ex.shap_values(Xq, nsamples=2 * D + 2048), dtype=np.float64)
        assert phi.shape[:2] == (NQ, D), (tag, phi.shape)
        base = np.atleast_1d(np.asarray(ex.expected_value, dtype=np.float64))
        fx = _f(fn, Xq)
        if kw.get("link") == "logit":
            fx = np.log(fx / (1.0 - fx))
        tot = phi.reshape(NQ, D, -1).sum(axis=1) + base[None, :]
        out[tag] = phi
        out[tag + "_additivity"] = np.array([np.abs(tot - fx).max()])
        if mname == "linear":
            exact = w[None, :] * (Xq.astype(np.float64) - bg.astype(np.float64).mean(0)[None, :])
            out[tag + "_relerr"] = np.array([np.linalg.norm(phi - exact) / np.linalg.norm(exact)])
    np.savez(path, **out)
    meta["arrays"] = len(out)
    print("SHAP-PIPE-CAPTURE " + json.dumps(meta, sort_keys=True))
    for key in sorted(out):
        if key.endswith(("_additivity", "_relerr")):
            print(f"SHAP-PIPE-METRIC {key}={float(out[key][0]):.6e}")
    print(f"SHAP-PIPE-QUALITY status=PASS arrays={len(out)} path={path}")


def compare(pa, pb):
    a, b = np.load(pa), np.load(pb)
    assert sorted(a.files) == sorted(b.files) and len(a.files) == 16, (a.files, b.files)
    worst = 0.0
    for key in sorted(a.files):
        x, y = a[key], b[key]
        assert x.dtype == y.dtype and x.shape == y.shape, key
        diff = float(np.abs(x - y).max())
        scale = float(np.abs(x).max()) or 1.0
        worst = max(worst, diff / scale)
        print(f"SHAP-PIPE-DIFF {key} max_abs={diff:.3e} max_rel={diff / scale:.3e}")
        assert x.tobytes() == y.tobytes(), key + ": arm B differs from arm A (tolerance: byte-identical)"
    print(f"SHAP-PIPE-AB status=PASS exact_arrays={len(a.files)} worst_rel={worst:.3e}")


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
