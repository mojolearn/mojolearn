# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Is OUR Gaussian mixture broken on Istella-S, or is it the data?

The board's classical2/gmm race on Istella-S failed on EVERY arm at 2,000
rows (ours, ours-fast and scikit-learn: "some components have ill-defined
empirical covariance"). This probe fits the race's own X, UNMODIFIED, at the
race's full fit rows (tools/bench_board_more.py prep for lane gmm on
istella, then lane_arrays: 100,000 stride rows of the standardized reg
block), with the race's parameters (8 components, covariance_type='full',
tol=1e-3, reg_covar=1e-6, max_iter=100, init_params='kmeans', n_init=1,
seed 7), and prints one GMM-PROBE line per fit:

  1. ours      mojolearn.GaussianMixture (MOJOLEARN_NUMERIC_MODE, identical)
  2. sk-f32    scikit-learn GaussianMixture on the same float32 X
  3. sk-f64    scikit-learn GaussianMixture on X as float64
  4. ours-bgmm / sk-bgmm-f32  BayesianGaussianMixture, the algos/bayesian-gmm
     parameters, on the same X (the algos lane reads its own reg block)

and one GMM-PROBE-COLUMNS line: how many columns are constant, or hold at
most 2 or 10 distinct values, on these rows.

    python3 tools/gmm_istella_probe.py --data /tmp/gmm-probe      # GBM_BENCH_DATA set
"""
import argparse
import importlib.util
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def _load(name):
    spec = importlib.util.spec_from_file_location("gp_" + name, os.path.join(HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def say(tag, **kw):
    print("GMM-PROBE %s %s" % (tag, json.dumps(kw, sort_keys=True, default=str)), flush=True)


def fit(tag, make, X):
    t0 = time.perf_counter()
    try:
        est = make().fit(X)
        extra = {}
        for name in ("converged_", "n_iter_", "lower_bound_"):
            if hasattr(est, name):
                v = getattr(est, name)
                extra[name] = v.item() if hasattr(v, "item") else v
        say(tag, ok=True, seconds=round(time.perf_counter() - t0, 3), dtype=str(X.dtype), **extra)
    except BaseException as exc:  # noqa: BLE001 (report every failure by name)
        say(tag, ok=False, seconds=round(time.perf_counter() - t0, 3), dtype=str(X.dtype),
            error="%s: %s" % (type(exc).__name__, str(exc)[:1500]),
            where=traceback.format_exc().splitlines()[-3:])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True, help="where prep writes the reg block")
    ap.add_argument("--skip-prep", action="store_true", help="the block is already in --data")
    ap.add_argument("--repo-python", action="store_true",
                    help="import mojolearn from the checkout's python/ (built bindings) instead of "
                         "the interpreter's installed wheel")
    args = ap.parse_args()
    import numpy as np
    more = _load("bench_board_more")
    if not args.skip_prep:
        more.prep(argparse.Namespace(lanes="gmm", datasets="istella", max_rows=None, data=args.data))
    block = os.path.join(args.data, "reg-istella")
    with np.load(block + ".npz") as z:
        B = {k: np.ascontiguousarray(z[k]) for k in z.files}
    X = more.lane_arrays("gmm", B)["X"]
    distinct = [len(np.unique(X[:, j])) for j in range(X.shape[1])]
    say("COLUMNS", rows=X.shape[0], cols=X.shape[1], dtype=str(X.dtype),
        constant=sum(1 for c in distinct if c == 1), at_most_2=sum(1 for c in distinct if c <= 2),
        at_most_10=sum(1 for c in distinct if c <= 10),
        mode=os.environ.get("MOJOLEARN_NUMERIC_MODE"))
    kw = dict(n_components=more.GMM_COMPONENTS, covariance_type="full", tol=1e-3, reg_covar=1e-6,
              max_iter=100, init_params="kmeans", n_init=1, warm_start=False, random_state=more.SEED)
    bkw = dict(n_components=8, covariance_type="full", max_iter=100, tol=1e-3, reg_covar=1e-6,
               init_params="kmeans", random_state=more.SEED, n_init=1,
               weight_concentration_prior_type="dirichlet_process")
    # ours first, then scikit-learn (imported only after ours ran, so an
    # interpreter without it still reports ours)
    if args.repo_python:
        sys.path.insert(0, os.path.join(REPO, "python"))
    try:
        import mojolearn as ml
        say("OURS-MODULE", path=ml.__file__, version=getattr(ml, "__version__", None))
        fit("ours", lambda: ml.GaussianMixture(**kw), X)
        fit("ours-bgmm", lambda: ml.BayesianGaussianMixture(**bkw), X)
    except ImportError as exc:
        say("ours", ok=False, error="import mojolearn: %s" % exc)
    from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
    fit("sk-f32", lambda: GaussianMixture(**kw), X)
    fit("sk-f64", lambda: GaussianMixture(**kw), X.astype(np.float64))
    fit("sk-bgmm-f32", lambda: BayesianGaussianMixture(**bkw), X)
    return 0


if __name__ == "__main__":
    sys.exit(main())
