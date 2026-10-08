#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Lane shap-xvendor-identity (2026-10-08): the witness for where the kernel-shap
and permutation-shap digests diverge between the NVIDIA and AMD boxes.

Per dataset it prints `hash=` tokens (16 hex each; the lq CMD result line
collects them in order) for the NumPy closed-form ridge the board explained
until 2026-10-08 (the Gram matrix, w, b, its float32 outputs over the
background and the query rows, its exact SHAP reference) and for this
library's device ridge (mojolearn.RidgeCV(alphas=[1.0]): coef_, intercept_,
predictions, exact), after one line naming the host CPU and the NumPy BLAS.
Two boxes' lines compare side by side: the NumPy quantities are the host
CPU's bits, the device ridge's are the same on both vendors.

--explain also runs KernelExplainer and PermutationExplainer over the first
--rows query rows with both models (the x_trees binding) and prints the phi
digests: the NumPy-model phi follows the host, the device-model phi does not.

Usage: python3 tools/shap_xvendor_probe.py --data <algos-data dir>
         [--datasets taxi,istella] [--explain] [--rows 4]
"""
import argparse
import hashlib
import importlib.util
import os
import platform


def _h(a):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).data).hexdigest()[:16]


def _bba():
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("bench_board_algos", os.path.join(here, "bench_board_algos.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cpu():
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _blas():
    import numpy as np
    try:
        cfg = np.show_config(mode="dicts")
        dep = cfg.get("Build Dependencies", {})
        b, l = dep.get("blas", {}), dep.get("lapack", {})
        return "blas=%s/%s lapack=%s/%s" % (b.get("name"), b.get("version"), l.get("name"), l.get("version"))
    except Exception as exc:  # noqa: BLE001
        return "show_config failed: %r" % (exc,)


def _fixed_mean(bg64):
    """The harness's exact reference fold: the background rows added in order."""
    import numpy as np
    acc = np.zeros(bg64.shape[1], dtype=np.float64)
    for row in range(bg64.shape[0]):
        acc += bg64[row]
    return acc / float(bg64.shape[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--datasets", default="taxi,istella")
    ap.add_argument("--explain", action="store_true")
    ap.add_argument("--rows", type=int, default=4)
    args = ap.parse_args()
    import numpy as np
    bba = _bba()
    print("host cpu=%s numpy=%s %s" % (_cpu(), np.__version__, _blas()), flush=True)
    try:
        from threadpoolctl import threadpool_info
        print("threadpools=%s" % ([(t.get("internal_api"), t.get("version"), t.get("num_threads"))
                                   for t in threadpool_info()],), flush=True)
    except Exception as exc:  # noqa: BLE001
        print("threadpools unknown: %r" % (exc,), flush=True)
    lane = "kernel-shap"
    p = bba.LANES[lane]["params"]
    for ds in args.datasets.split(","):
        B, _rec = bba._load_block(lane, ds, args.data)
        D = bba.lane_arrays(lane, B)
        for k, m in bba.LANES[lane]["sub"].items():
            if k in D:
                D[k] = bba._stride(D[k], m)
                yk = {"X": "y", "Xq": "yq"}[k]
                if yk in D:
                    D[yk] = bba._stride(D[yk], m)
        X, y, Xq = D["X"], D["y"], D["Xq"]
        bg = bba._stride(X, p["n_background"])
        print("%s data X=%s Xq=%s X hash=%s Xq hash=%s" % (ds, X.shape, Xq.shape, _h(X), _h(Xq)), flush=True)
        # the NumPy closed form (the model every arm explained until 2026-10-08)
        Xd, yd = X.astype(np.float64), y.astype(np.float64)
        mu, ym = Xd.mean(0), yd.mean()
        A = (Xd - mu).T @ (Xd - mu) + np.eye(Xd.shape[1])
        w = np.linalg.solve(A, (Xd - mu).T @ (yd - ym))
        b = ym - mu @ w

        def predict(Z, w=w, b=b):
            return np.asarray(bba._host(Z), dtype=np.float64) @ w + b
        bgm = _fixed_mean(bg.astype(np.float64))
        print("%s numpy-ridge A hash=%s w hash=%s b hash=%s bg-out-f32 hash=%s Xq-out-f32 hash=%s exact hash=%s" % (
            ds, _h(A), _h(w), _h(np.array([b])), _h(predict(bg).astype(np.float32)),
            _h(predict(Xq).astype(np.float32)), _h((Xq.astype(np.float64) - bgm) * w)), flush=True)
        # this library's device ridge
        import mojolearn as ml
        ridge = ml.RidgeCV(alphas=[1.0], fit_intercept=True).fit(X, y)
        wd = bba._arr(ridge.coef_, np.float64).reshape(-1)
        print("%s device-ridge coef hash=%s intercept hash=%s bg-pred hash=%s Xq-pred hash=%s exact hash=%s "
              "max|w-wd|/max|w|=%.3e vendor=%s" % (
                  ds, _h(wd), _h(np.array([float(ridge.intercept_)])), _h(bba._arr(ridge.predict(bg), np.float32)),
                  _h(bba._arr(ridge.predict(Xq), np.float32)), _h((Xq.astype(np.float64) - bgm) * wd),
                  float(np.max(np.abs(w - wd)) / max(np.max(np.abs(w)), 1e-300)), ml.vendor()), flush=True)
        if not args.explain:
            continue
        Xr = np.ascontiguousarray(Xq[:args.rows])
        for name, model in (("numpy-model", predict), ("device-model", ridge)):
            ex = ml.KernelExplainer(model, bg, link=p["link"], random_state=bba.SEED)
            phk = bba._arr(ex.shap_values(Xr, nsamples=p["nsamples"]), np.float64)
            ex = ml.PermutationExplainer(model, bg, random_state=bba.SEED)
            php = bba._arr(ex.shap_values(Xr, npermutations=bba.LANES["permutation-shap"]["params"]["npermutations"]),
                           np.float64)
            print("%s %s rows=%d kernel-shap phi hash=%s permutation-shap phi hash=%s" % (
                ds, name, Xr.shape[0], _h(phk), _h(php)), flush=True)


if __name__ == "__main__":
    main()
