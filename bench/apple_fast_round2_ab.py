# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""One timed run of an off-board FAST workload for lane apple-fast-round2's
A/Bs (tools/afr2_ab.sh builds the two arms; this runs one arm's .so).

The board rows do not reach two of the lane's switches: kernel-shap and
permutation-shap explain a NumPy ridge closure (no device model exists for a
callable), and standard-scaler times fit(X) then transform(Xq) of a
different array. So:

  kshap-rf   KernelExplainer over a mojolearn RandomForestRegressor
             (MOJOLEARN_KSHAP_FAST_DEVICE_MODEL)
  pshap-rf   PermutationExplainer over the same forest
             (MOJOLEARN_PSHAP_FAST_DEVICE_MODEL)
  scaler-ft  StandardScaler.fit_transform(X)
             (MOJOLEARN_X_PREP_FAST_FIT_TRANSFORM_FUSED)

Prints one line: AFR2 case=<case> ms=<timed ms> quality=<metric>=<value>.
Quality (outside the clock, NumPy, a bench tool, not the runtime): SHAP
additivity max |sum(phi) + E[f] - f(x)|; scaler max |fit_transform -
fit().transform()|. Shapes are generic sizes, not board dimensions.
"""
import sys
import time

import numpy as np


def _forest(rows, cols, seed):
    import mojolearn as ml
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((rows, cols), dtype=np.float32)
    y = (X[:, :8] @ rng.standard_normal(8).astype(np.float32)
         + 0.1 * rng.standard_normal(rows).astype(np.float32)).astype(np.float32)
    model = ml.RandomForestRegressor(n_estimators=100, max_depth=6, random_state=seed,
                                     numeric_mode="fast").fit(X, y)
    return ml, model, X


def shap_case(kind):
    ml, model, X = _forest(50_000, 96, 11)
    bg = np.ascontiguousarray(X[::500][:100])
    Xq = np.ascontiguousarray(X[1::997][:40])
    t0 = time.perf_counter()
    if kind == "kshap-rf":
        ex = ml.KernelExplainer(model, bg, random_state=7)
        phi = ex.shap_values(Xq, nsamples=2048)
    else:
        ex = ml.PermutationExplainer(model, bg, random_state=7)
        phi = ex.shap_values(Xq, npermutations=10)
    ms = (time.perf_counter() - t0) * 1e3
    f = np.asarray(model.predict(Xq), dtype=np.float64)
    err = np.abs(np.asarray(phi, np.float64).sum(axis=1) + float(np.asarray(ex.expected_value)) - f).max()
    return ms, "additivity_max_abs=%.3e" % err


def scaler_case():
    import mojolearn as ml
    rng = np.random.default_rng(5)
    X = (3.0 * rng.standard_normal((750_000, 160), dtype=np.float32) + 1.5).astype(np.float32)
    ml.StandardScaler(numeric_mode="fast").fit_transform(X[:4096])   # warm the binding and context
    t0 = time.perf_counter()
    Z = ml.StandardScaler(numeric_mode="fast").fit_transform(X)
    ms = (time.perf_counter() - t0) * 1e3
    R = ml.StandardScaler(numeric_mode="fast").fit(X).transform(X)
    return ms, "max_abs_vs_two_step=%.3e" % np.abs(np.asarray(Z) - np.asarray(R)).max()


def main():
    case = sys.argv[1]
    if case in ("kshap-rf", "pshap-rf"):
        ms, q = shap_case(case)
    elif case == "scaler-ft":
        ms, q = scaler_case()
    else:
        raise SystemExit("cases: kshap-rf pshap-rf scaler-ft")
    print("AFR2 case=%s ms=%.1f quality=%s" % (case, ms, q))


if __name__ == "__main__":
    main()
