#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Which GradientBoosting TRAINING configurations refuse on a CPU-only install.

`python/mojolearn/host_surface.py::NO_CPU_PATH` says in prose what the gbdt
host binding does not train. Prose drifts: the entry this file was written for
still said eval sets refuse outside the pointwise searcher's lane a day after
Ordered boosting took one. THIS FILE IS THE LIST, REPRODUCED. Every row runs a
real fit on a real host binding and prints the exception it raised, so a
refusal that has moved shows up as a changed row and a refusal that has been
closed shows up as OK.

    MOJOLEARN_HOST_DIR=/path/to/host python3 tools/gbdt_cpu_refusal_probe.py

Needs a CPU-only load: the `identical` set must be absent (or
MOJOLEARN_HOST_DIR must name a host set and the package carry no GPU
binding), so that `mojolearn._backend._CPU_ONLY` is set and the host bindings
serve. Fits run inside `mojolearn._cpu_reference.reference_training()`,
which is the verifier's own scoped door; it is not a public CPU training API
and this file is not one either.

READ THE `## A` BLOCK FIRST. It is the control: those configurations must
print OK. If any of them refuses, the probe is measuring a broken or stale
binding and every REFUSED line below it means nothing.
"""
import argparse
import sys

import numpy as np


def _data(seed=7, n=300, d=4):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    out = dict(
        X=X,
        yb=(X[:, 0] + 0.3 * X[:, 1] > 0).astype(np.float32),
        yr=(X[:, 0] * 2.0 + X[:, 2]).astype(np.float32),
        ym=rng.integers(0, 3, n).astype(np.float32),
        w=(np.abs(rng.standard_normal(n)) + 0.5).astype(np.float32),
    )
    # a categorical column at one_hot_max_size (one-hot) and above it (CTR)
    xc = X.copy()
    xc[:, 0] = rng.integers(0, 2, n)
    xc8 = X.copy()
    xc8[:, 0] = rng.integers(0, 8, n)
    xn = X.copy()
    xn[3, 1] = np.float32("nan")
    out.update(Xc=xc, Xc8=xc8, Xn=xn)
    return out


#: The lane configuration every recorded GBDT cell is fitted at
#: (`tools/identity_break.py::_gbdt_legacy_kw`), so the probe isolates the
#: feature under test instead of the default bootstrap resolution.
PIN = dict(n_estimators=3, max_depth=3, border_count=32, random_state=0,
           learning_rate=0.03, random_strength=0.0, bootstrap_type="No")

#: Tails the binding appends to its refusals; cut so the table stays readable.
_CUTS = ("; the gbdt host binding", " (catboost_options",
         " (their TGpuTrainerFactory", " -- CatBoost's GPU registers")


def _rows(D):
    """(section, name, kwargs, fit kwargs, X, y) for every configuration."""
    X, yb, yr, ym, w = D["X"], D["yb"], D["yr"], D["ym"], D["w"]
    Xc, Xc8, Xn = D["Xc"], D["Xc8"], D["Xn"]
    ex, ey = X[:80], yb[:80]
    sym = dict(loss="Logloss", leaf_estimation_iterations=10)
    dw = dict(loss="Logloss", grow_policy="Depthwise")
    lg = dict(loss="Logloss", grow_policy="Lossguide", max_leaves=8)
    mc = dict(loss="MultiClass")
    mae = dict(loss="MAE", boost_from_average=False)
    ordered = dict(loss="Logloss", boosting_type="Ordered",
                   leaf_estimation_iterations=10)
    pw = dict(loss="Logloss", score_function="L2", use_pointwise_searcher=True,
              bootstrap_type="Bayesian", leaf_estimation_iterations=10,
              od_type="Iter", od_wait=2, boost_from_average=True)
    pwfit = dict(sample_weight=w, eval_set=(ex, ey))
    R = []
    A = "A. arms that train today (CONTROL: every row must read OK)"
    R += [(A, "SymmetricTree Logloss", sym, {}, X, yb),
          (A, "SymmetricTree RMSE", dict(loss="RMSE"), {}, X, yr),
          (A, "Depthwise Logloss", dw, {}, X, yb),
          (A, "Lossguide Logloss", lg, {}, X, yb),
          (A, "MultiClass", mc, {}, X, ym),
          (A, "MAE pointwise", mae, {}, X, yr),
          (A, "one-hot cat, SymmetricTree Logloss", dict(cat_features=[0], **sym), {}, Xc, yb),
          (A, "Ordered Logloss", ordered, {}, X, yb),
          (A, "Ordered Logloss + eval_set + Iter", dict(od_type="Iter", od_wait=2, **ordered),
           dict(eval_set=(ex, ey)), X, yb),
          (A, "pointwise searcher lane config", pw, pwfit, X, yb),
          (A, "NaN in X, SymmetricTree Logloss", sym, {}, Xn, yb),
          (A, "eval_set, SymmetricTree Logloss", sym, dict(eval_set=(ex, ey)), X, yb),
          (A, "eval_set + Iter, SymmetricTree Logloss",
           dict(od_type="Iter", od_wait=2, **sym), dict(eval_set=(ex, ey)), X, yb),
          (A, "eval_set + use_best_model, SymmetricTree Logloss",
           dict(use_best_model=True, **sym), dict(eval_set=(ex, ey)), X, yb)]
    B = "B. sample weights (F1)"
    for nm, kw, y in (("SymmetricTree Logloss", sym, yb), ("SymmetricTree RMSE", dict(loss="RMSE"), yr),
                      ("Depthwise Logloss", dw, yb), ("Lossguide Logloss", lg, yb),
                      ("MultiClass", mc, ym), ("MAE pointwise", mae, yr),
                      ("Ordered Logloss", ordered, yb)):
        R.append((B, "sample_weight, " + nm, kw, dict(sample_weight=w), X, y))
    C = "C. class weights (F2)"
    R += [(C, "class_weights, SymmetricTree Logloss", dict(class_weights=[1.0, 2.0], **sym), {}, X, yb),
          (C, "class_weights, MultiClass", dict(class_weights=[1.0, 2.0, 3.0], **mc), {}, X, ym),
          (C, "class_weights, MultiClassOneVsAll",
           dict(loss="MultiClassOneVsAll", class_weights=[1.0, 2.0, 3.0]), {}, X, ym),
          (C, "class_weights, Ordered Logloss", dict(class_weights=[1.0, 2.0], **ordered), {}, X, yb)]
    D_ = "D. eval set and the overfitting detector (F5)"
    R += [(D_, "eval_set, SymmetricTree RMSE", dict(loss="RMSE"), dict(eval_set=(X[:80], yr[:80])), X, yr),
          (D_, "eval_set, Depthwise Logloss", dw, dict(eval_set=(ex, ey)), X, yb),
          (D_, "eval_set, Lossguide Logloss", lg, dict(eval_set=(ex, ey)), X, yb),
          (D_, "eval_set, MultiClass", mc, dict(eval_set=(X[:80], ym[:80])), X, ym),
          (D_, "eval_set, MAE pointwise", mae, dict(eval_set=(X[:80], yr[:80])), X, yr),
          (D_, "od_type=IncToDec + eval_set, Ordered",
           dict(od_type="IncToDec", od_pvalue=0.01, **ordered), dict(eval_set=(ex, ey)), X, yb)]
    E = "E. categorical columns (F3, F4)"
    R += [(E, "CTR cat (8 categories), SymmetricTree Logloss", dict(cat_features=[0], **sym), {}, Xc8, yb),
          (E, "one-hot cat, SymmetricTree RMSE", dict(loss="RMSE", cat_features=[0]), {}, Xc, yr),
          (E, "one-hot cat, Depthwise Logloss", dict(cat_features=[0], **dw), {}, Xc, yb),
          (E, "one-hot cat, Lossguide Logloss", dict(cat_features=[0], **lg), {}, Xc, yb),
          (E, "one-hot cat, MultiClass", dict(cat_features=[0], **mc), {}, Xc, ym),
          (E, "one-hot cat, MAE pointwise", dict(cat_features=[0], **mae), {}, Xc, yr),
          (E, "one-hot cat, Ordered Logloss", dict(cat_features=[0], **ordered), {}, Xc, yb),
          (E, "one_hot_features explicit, SymmetricTree Logloss",
           dict(one_hot_features=[0], **sym), {}, Xc, yb)]
    F = "F. the pointwise searcher outside its configuration (F6)"
    for nm, over, fover in (
            ("no sample_weight", {}, dict(sample_weight=None)),
            ("no eval_set", {}, dict(eval_set=None)),
            ("Cosine score", dict(score_function="Cosine"), {}),
            ("Bernoulli bootstrap", dict(bootstrap_type="Bernoulli", subsample=0.8), {}),
            ("No bootstrap", dict(bootstrap_type="No"), {}),
            ("class_weights", dict(class_weights=[1.0, 2.0]), {}),
            ("cat_features", dict(cat_features=[0]), {}),
            ("IncToDec detector", dict(od_type="IncToDec", od_pvalue=0.01, od_wait=None), {}),
            ("no detector", dict(od_type=None, od_wait=None), {}),
            ("boost_from_average=False", dict(boost_from_average=False), {}),
            ("feature_fraction=0.5", dict(feature_fraction=0.5), {}),
            ("feature_border_type=Uniform", dict(feature_border_type="Uniform"), {}),
            ("Gradient leaves", dict(leaf_estimation_method="Gradient"), {}),
            ("RMSE loss", dict(loss="RMSE"), {})):
        kw = dict(pw)
        kw.update(over)
        fk = dict(pwfit)
        fk.update(fover)
        R.append((F, "pointwise: " + nm, kw, fk, X, yb))
    G = "G. searcher knobs, score functions, leaf estimators (F7 to F10, F12 to F14)"
    R += [(G, "min_split_gain, Depthwise", dict(min_split_gain=0.1, **dw), {}, X, yb),
          (G, "min_child_hessian, Depthwise", dict(min_child_hessian=1.0, score_function="NewtonL2", **dw), {}, X, yb),
          (G, "min_data_in_leaf, Depthwise", dict(min_data_in_leaf=5, **dw), {}, X, yb),
          (G, "min_split_gain, Lossguide", dict(min_split_gain=0.1, **lg), {}, X, yb),
          (G, "min_data_in_leaf, Lossguide", dict(min_data_in_leaf=5, **lg), {}, X, yb),
          (G, "feature_fraction, SymmetricTree", dict(feature_fraction=0.5, **sym), {}, X, yb),
          (G, "feature_fraction, Lossguide", dict(feature_fraction=0.5, **lg), {}, X, yb),
          (G, "score_function=L2, SymmetricTree", dict(score_function="L2", **sym), {}, X, yb),
          (G, "score_function=NewtonL2, Depthwise", dict(score_function="NewtonL2", **dw), {}, X, yb),
          (G, "Exact leaves, SymmetricTree Logloss", dict(loss="Logloss", leaf_estimation_method="Exact"), {}, X, yb),
          (G, "Gradient leaves, SymmetricTree Logloss", dict(loss="Logloss", leaf_estimation_method="Gradient"), {}, X, yb),
          (G, "Exact leaves, MAE pointwise", dict(leaf_estimation_method="Exact", **mae), {}, X, yr),
          (G, "RMSE leaf_estimation_iterations=3", dict(loss="RMSE", leaf_estimation_iterations=3), {}, X, yr),
          (G, "RMSE Depthwise", dict(loss="RMSE", grow_policy="Depthwise"), {}, X, yr),
          (G, "MAE Depthwise", dict(grow_policy="Depthwise", **mae), {}, X, yr),
          (G, "MultiClass Depthwise (the DEVICE refuses this too)",
           dict(grow_policy="Depthwise", **mc), {}, X, ym),
          (G, "border_count=300", dict(border_count=300, **sym), {}, X, yb),
          (G, "max_depth=17", dict(max_depth=17, **sym), {}, X, yb),
          (G, "boost_from_average=True, Logloss", dict(boost_from_average=True, **sym), {}, X, yb),
          (G, "random_strength=1, SymmetricTree Logloss", dict(random_strength=1.0, **sym), {}, X, yb),
          (G, "random_strength=1, Depthwise Logloss", dict(random_strength=1.0, **dw), {}, X, yb),
          (G, "random_strength=1, MultiClass", dict(random_strength=1.0, **mc), {}, X, ym),
          (G, "random_strength=1, RMSE", dict(loss="RMSE", random_strength=1.0), {}, X, yr)]
    H = "H. bootstraps (F11)"
    for bt, extra in (("Bayesian", {}), ("Bernoulli", dict(subsample=0.8)),
                      ("Poisson", dict(subsample=0.8))):
        for nm, kw, y, XX in (("SymmetricTree Logloss", sym, yb, X),
                              ("SymmetricTree RMSE", dict(loss="RMSE"), yr, X),
                              ("Depthwise Logloss", dw, yb, X),
                              ("Lossguide Logloss", lg, yb, X),
                              ("MultiClass", mc, ym, X),
                              ("MAE pointwise", mae, yr, X),
                              ("one-hot cat Logloss", dict(cat_features=[0], **sym), yb, Xc)):
            k = dict(kw)
            k.update(bootstrap_type=bt, **extra)
            R.append((H, "bootstrap=%s, %s" % (bt, nm), k, {}, XX, y))
    I = "I. NaN in X outside SymmetricTree Logloss (F15)"
    R += [(I, "NaN, SymmetricTree RMSE", dict(loss="RMSE"), {}, Xn, yr),
          (I, "NaN, Depthwise Logloss", dw, {}, Xn, yb),
          (I, "NaN, MultiClass", mc, {}, Xn, ym),
          (I, "NaN, MAE pointwise", mae, {}, Xn, yr),
          (I, "NaN, Ordered Logloss", ordered, {}, Xn, yb)]
    return R


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--allow-gpu", action="store_true",
                    help="run even though a GPU backend loaded (the refusals "
                         "are then the GPU binding's, which is a different "
                         "question); the default refuses")
    args = ap.parse_args()

    import mojolearn
    from mojolearn import _backend
    from mojolearn._cpu_reference import reference_training
    from mojolearn.ensemble import GradientBoosting

    if _backend._CPU_ONLY is None and not args.allow_gpu:
        raise SystemExit(
            "REFUSING: a GPU backend loaded, so these refusals would be the "
            "GPU binding's. Run where the `identical` set is absent, or pass "
            "--allow-gpu and read the output as a GPU answer."
        )
    print("# gbdt CPU refusal probe  vendor=%s  mode=%s" %
          (mojolearn.vendor(), mojolearn.numeric_mode()))
    print("# host set: %s" % _backend.host_dir())

    D = _data()
    section = None
    counts = {"OK": 0, "REFUSED": 0}
    control_failures = []
    for sec, name, kw, fitkw, X, y in _rows(D):
        if sec != section:
            section = sec
            print()
            print("## " + sec)
        p = dict(PIN)
        p.update(kw)
        fk = {k: v for k, v in fitkw.items() if v is not None}
        try:
            with reference_training():
                GradientBoosting(**p).fit(X, y, **fk)
            print("OK       | " + name)
            counts["OK"] += 1
        except Exception as exc:  # noqa: BLE001 -- the message IS the result
            msg = " ".join(str(exc).split())
            for cut in _CUTS:
                i = msg.find(cut)
                if i > 0:
                    msg = msg[:i]
            print("REFUSED  | " + name + " | " + msg)
            counts["REFUSED"] += 1
            if sec.startswith("A."):
                control_failures.append(name)

    print()
    print("ok=%d refused=%d" % (counts["OK"], counts["REFUSED"]))
    if control_failures:
        print("CONTROL FAILED, this probe measured nothing: " +
              ", ".join(control_failures))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
