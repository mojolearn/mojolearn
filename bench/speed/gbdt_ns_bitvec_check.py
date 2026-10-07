#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Same-bits and predict-time screen for MOJOLEARN_GBDT_NS_PREDICT_BITVEC
(lane/trees-predict-ideas, 2026-10-07). Run once per arm, in its own process,
with PYTHONPATH naming that arm's package copy; tools/gbdt_ns_bitvec_check.sh
drives both arms and compares the digests.

Shapes span every mask-word class and the fallback, not a board row:
depth 6 (<=64 leaves, 1 word), depth 7 (2 words), depth 8 (4 words),
Lossguide 300 leaves (8 words), depth 10 (1024 leaves: past the cap, the
incumbent apply), plus a MultiClass Depthwise fit (approx dim > 1).
Synthetic data from a fixed seed; one fit and one timed predict per case
(one run per arm). Prints one `NSBV` line per case.
"""
import argparse
import hashlib
import time

import numpy as np

CASES = [
    # name, grow_policy, depth, max_leaves, loss, classes
    ("dw-d6", "Depthwise", 6, 64, "Logloss", 2),
    ("dw-d7", "Depthwise", 7, 128, "Logloss", 2),
    ("dw-d8", "Depthwise", 8, 256, "Logloss", 2),
    ("lg-l300", "Lossguide", 10, 300, "Logloss", 2),
    ("dw-d10", "Depthwise", 10, 1024, "Logloss", 2),
    ("dw-d6-mc", "Depthwise", 6, 64, "MultiClass", 5),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--rows", type=int, default=200000)
    ap.add_argument("--features", type=int, default=28)
    ap.add_argument("--trees", type=int, default=100)
    a = ap.parse_args()
    import mojolearn

    rng = np.random.default_rng(7)
    X = rng.standard_normal((a.rows, a.features), dtype=np.float32)
    w = rng.standard_normal(a.features).astype(np.float32)
    z = X @ w + 0.5 * np.sin(3.0 * X[:, 0]) * X[:, 1]
    for name, policy, depth, leaves, loss, classes in CASES:
        if classes == 2:
            y = (z > 0).astype(np.float32)
        else:
            y = np.digitize(z, np.quantile(z, np.linspace(0, 1, classes + 1)[1:-1])).astype(np.float32)
        try:
            m = mojolearn.GradientBoosting(
                loss=loss, n_estimators=a.trees, max_depth=depth, max_leaves=leaves,
                learning_rate=0.1, l2_leaf_reg=1.0, border_count=254, random_state=7,
                bootstrap_type="No", grow_policy=policy, numeric_mode="identical",
            )
            m.fit(X, y)
            m.predict(X[:1024])  # warm the resident model
            t0 = time.perf_counter()
            out = m.predict(X)
            ms = (time.perf_counter() - t0) * 1e3
            digest = hashlib.sha256(np.asarray(out).tobytes()).hexdigest()[:16]
            print(f"NSBV arm={a.arm} case={name} predict_ms={ms:.2f} digest={digest}", flush=True)
        except Exception as e:  # report and continue: one case must not hide the rest
            print(f"NSBV arm={a.arm} case={name} ERROR {type(e).__name__}: {str(e)[:200]}", flush=True)


if __name__ == "__main__":
    main()
