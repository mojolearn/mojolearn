#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Which public gradient boosting configurations train on a CPU-only process.

    PYTHONPATH=python python3 tools/gbdt_cpu_parity_matrix.py OUT.tsv

{GradientBoostingRegressor, GradientBoostingClassifier} x grow_policy x
bootstrap_type with everything else at its default, then GradientBoosting at
the defaults of every loss in `ensemble.LOSSES`. Three trees per fit. One TSV
row per fit: OK, or the exception the fit raised.
"""
import sys

import numpy as np


def main():
    out = sys.argv[1]
    import mojolearn as ml
    from mojolearn import _backend
    from mojolearn.ensemble import LOSSES

    rng = np.random.default_rng(7)
    n, d = 600, 6
    X = rng.standard_normal((n, d)).astype(np.float32)
    yr = (X[:, 0] * 2.0 + X[:, 2] + 0.1 * rng.standard_normal(n)).astype(np.float32)
    yb = (X[:, 0] + 0.3 * X[:, 1] > 0).astype(np.float32)
    ym = rng.integers(0, 3, n).astype(np.float32)
    ypos = (np.abs(yr) + 0.5).astype(np.float32)
    groups = np.repeat(np.arange(n // 10), 10)

    rows = []

    def run(cls_name, loss, policy, boot, make, fit):
        try:
            fit(make())
            status, detail = "OK", ""
        except Exception as e:  # the refusal text is the datum
            status = "REFUSED"
            detail = (type(e).__name__ + ": " + str(e)).replace("\t", " ").replace("\n", " ")[:240]
        rows.append((cls_name, loss, policy, boot, status, detail))

    for cls_name, y in (("GradientBoostingRegressor", yr), ("GradientBoostingClassifier", yb)):
        cls = getattr(ml, cls_name)
        loss = "RMSE" if "Regressor" in cls_name else "Logloss"
        for policy in ("SymmetricTree", "Depthwise", "Lossguide"):
            for boot in ("default", "Bayesian", "Bernoulli", "No"):
                kw = dict(n_estimators=3, grow_policy=policy)
                if boot != "default":
                    kw["bootstrap_type"] = boot
                run(cls_name, loss, policy, boot,
                    lambda: cls(**kw), lambda m: m.fit(X, y))

    extra = {"Lq": dict(loss_q=3.0), "Expectile": dict(loss_alpha=0.3),
             "Tweedie": dict(loss_variance_power=1.5), "Huber": dict(loss_delta=1.0)}
    for loss in LOSSES:
        kw = dict(n_estimators=3, loss=loss, **extra.get(loss, {}))
        if loss in ("MultiClass", "MultiClassOneVsAll"):
            y, fk = ym, {}
        elif loss in ("Logloss", "CrossEntropy"):
            y, fk = yb, {}
        elif loss == "RMSE":
            y, fk = yr, {}
        elif loss in ("QueryRMSE", "PairLogit", "YetiRank"):
            y, fk = ypos, dict(group_id=groups)
        else:
            y, fk = ypos, {}
        run("GradientBoosting", loss, "SymmetricTree", "default",
            lambda: ml.GradientBoosting(**kw), lambda m: m.fit(X, y, **fk))

    with open(out, "w") as f:
        f.write("# cpu_only=%s\n" % (_backend._CPU_ONLY is not None))
        f.write("class\tloss\tgrow_policy\tbootstrap_type\tstatus\tdetail\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    ok = sum(1 for r in rows if r[4] == "OK")
    print("cpu_only=%s ok=%d refused=%d" % (_backend._CPU_ONLY is not None, ok, len(rows) - ok))
    for r in rows:
        print("  ".join(r))


if __name__ == "__main__":
    main()
