#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""TreeExplainer identity digests (lane gap-treeshap, 2026-10-02): one fixed
set of forests (the board's tree-shap shape, a 3-class forest, ExtraTrees,
a deep tree, DART), their expected values and SHAP values, as sha256 lines.
Run once on the GPU column and once with MOJOLEARN_VENDOR=cpu on the same
box; every line must match across NVIDIA, AMD, Apple and the host. No timing.

    python3 tools/treeshap_identity.py > digests-<column>.txt
"""
import hashlib
import numpy as np
import mojolearn as ml


def _h(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype=np.float32)).tobytes()).hexdigest()[:16]


def main():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 12)).astype(np.float32)
    yr = (X[:, 0] * 2 + np.sin(X[:, 1] * 3) + X[:, 2] * X[:, 3] + 0.1 * rng.normal(size=4000)).astype(np.float32)
    yc = (X[:, 0] + X[:, 4] > 0).astype(np.int64) + (X[:, 2] > 1).astype(np.int64)
    Xq = rng.normal(size=(10000, 12)).astype(np.float32)
    cases = [
        ("rf-reg-100x6", ml.RandomForestRegressor(n_estimators=100, max_depth=6, random_state=0).fit(X, yr), X[::40]),
        ("rf-clf3-20x8", ml.RandomForestClassifier(n_estimators=20, max_depth=8, random_state=0).fit(X, yc), X[::10]),
        ("et-reg-20x10", ml.ExtraTreesRegressor(n_estimators=20, max_depth=10, random_state=0).fit(X, yr), X[::20]),
        ("dt-reg-depth14", ml.DecisionTreeRegressor(max_depth=14).fit(X, yr), X[::4]),
        ("dart-reg-30", ml.DARTRegressor(n_estimators=30, random_state=0).fit(X, yr), X[::40]),
    ]
    for name, model, bg in cases:
        ex = ml.TreeExplainer(model, data=bg)
        phi = ex.shap_values(Xq)
        print(name, "ev", _h(np.atleast_1d(ex.expected_value)), "phi", _h(phi), "width", ex._shape[4], flush=True)


if __name__ == "__main__":
    main()
