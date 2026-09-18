#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Lane kmeans-linear-speed (2026-09-17): where the PUBLIC fit's wall time goes.

    PYTHONPATH=<tree>/python python3 tools/kmeans_linear_profile.py \\
        --data /root/ctd-data --datasets taxi,istella --lanes kmeans,ols,pca --reps 5

Not a timing row. For each (lane, dataset) it runs the public estimator the
way `tools/classical_two_datasets.py` does (one warm-up, `--reps` timed fits)
with every native call of the fit wrapped in a wall clock, so a fit reads as
`total = sum(native calls) + python glue`. The device-side split of a native
call is the Mojo stage probes' job (`cluster/tools/kmeans_linear_stage_probe_
main.mojo`), not this file's. NumPy is used here to read the block; the
package under test stays NumPy-free.
"""
import argparse
import json
import os
import statistics
import sys
import time

os.environ.setdefault("MOJOLEARN_NUMERIC_MODE", "identical")
import numpy as np  # noqa: E402


class Clock:
    def __init__(self):
        self.t = {}

    def wrap(self, name, fn):
        def inner(*a, **k):
            t0 = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                self.t[name] = self.t.get(name, 0.0) + (time.perf_counter() - t0) * 1e3
        return inner


class TimedBinding:
    """A binding module proxy whose callables are clocked by name."""

    def __init__(self, mod, clock, prefix):
        self._m, self._c, self._p = mod, clock, prefix

    def __getattr__(self, name):
        v = getattr(self._m, name)
        return self._c.wrap(self._p + name, v) if callable(v) else v


def run(lane, X, y, init, reps):
    import mojolearn as ml
    from mojolearn import linear_model as lm
    from mojolearn import _buffer as buf

    rows = []
    for rep in range(reps + 1):
        clock = Clock()
        if lane == "kmeans":
            est = ml.KMeans(n_clusters=64, init="array", n_init=1, max_iter=20,
                            tol=1e-7, init_centroids=init)
        elif lane == "pca":
            est = ml.PCA(n_components=8, svd_solver="covariance_eigh")
        else:
            est = ml.LinearRegression(fit_intercept=True)
        real_bind = est._bind
        est._bind = lambda name, _rb=real_bind, _c=clock: TimedBinding(_rb(name), _c, "native.")
        saved = {}
        for fn in ("_column_means", "_vector_mean", "_center", "_shift"):
            saved[fn] = getattr(lm, fn)
            setattr(lm, fn, clock.wrap("host." + fn, saved[fn]))
        saved_native = buf._native
        try:
            t0 = time.perf_counter()
            if lane == "ols":
                est.fit(X, y)
            else:
                est.fit(X)
            total = (time.perf_counter() - t0) * 1e3
        finally:
            for fn, v in saved.items():
                setattr(lm, fn, v)
            buf._native = saved_native
        named = sum(clock.t.values())
        row = {"rep": rep, "total_ms": total, "python_glue_ms": total - named}
        row.update(clock.t)
        if rep > 0:
            rows.append(row)
    keys = sorted({k for r in rows for k in r if k != "rep"})
    return {k: statistics.median(r.get(k, 0.0) for r in rows) for k in keys}, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--datasets", default="taxi,istella")
    ap.add_argument("--lanes", default="kmeans,ols,pca")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    out = {}
    for ds in args.datasets.split(","):
        with np.load(os.path.join(args.data, "big-%s.npz" % ds)) as z:
            X = np.ascontiguousarray(z["X"], dtype=np.float32)
            y = np.ascontiguousarray(z["y"], dtype=np.float32)
            init = np.ascontiguousarray(z["init"], dtype=np.float32)
        for lane in args.lanes.split(","):
            med, rows = run(lane, X, y, init, args.reps)
            out["%s/%s" % (lane, ds)] = {"shape": list(X.shape), "median": med, "rows": rows}
            print("PROFILE %s %s %dx%d " % (lane, ds, X.shape[0], X.shape[1])
                  + " ".join("%s=%.2f" % (k, v) for k, v in sorted(med.items())), flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    sys.exit(main())
