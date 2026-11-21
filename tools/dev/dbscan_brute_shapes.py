#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Time mojolearn.DBSCAN(algorithm='brute') on generated shapes: the neighbouring-shape and non-board rows of an
A/B on the brute epsilon pass (lane sp-dbscan-filter, 2026-11-20: arm dbscan-brute-eps-filter).

A measurement controller, not product code. Run it once in the base wheel's environment and once in the arm
wheel's, on the same box in the same job, and compare the lines: `ms` is the A/B, `labels_sha256` must be equal.

    python3 tools/dev/dbscan_brute_shapes.py [--shapes 50000x8,50000x64,20000x512,200000x100] [--out FILE]

Each shape is a seeded Gaussian mixture (16 centres, unit spread, float32), unknown to the product. eps is the
1 % quantile of the pair distances of a 2,000-row sample, so every shape has the same neighbour density
(about n / 100 neighbours a row) whatever its width; min_samples = 5. One excluded warm-up fit on 2,000 rows
loads the kernels; then ONE scored fit per shape (one run per arm, one seed). The generator and the eps
quantile are single-process NumPy on the controller side and outside the clock.
"""
import argparse
import hashlib
import json
import sys
import time

DEFAULT_SHAPES = "50000x8,50000x64,20000x512,200000x100"
SEED = 20261120
CENTRES = 16
SAMPLE = 2000
QUANTILE = 0.01
MIN_SAMPLES = 5


def make_mixture(np, rows, features, seed):
    rng = np.random.default_rng(seed)
    centres = rng.normal(0.0, 4.0, size=(CENTRES, features))
    which = rng.integers(0, CENTRES, size=rows)
    x = centres[which] + rng.normal(0.0, 1.0, size=(rows, features))
    return np.ascontiguousarray(x, dtype=np.float32)


def eps_for(np, x):
    s = x[:: max(1, x.shape[0] // SAMPLE)][:SAMPLE].astype(np.float64)
    n2 = (s * s).sum(axis=1)
    d2 = n2[:, None] + n2[None, :] - 2.0 * (s @ s.T)
    iu = np.triu_indices(s.shape[0], k=1)
    q = float(np.quantile(np.maximum(d2[iu], 0.0), QUANTILE))
    return float(np.sqrt(q))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--shapes", default=DEFAULT_SHAPES, help="comma list of ROWSxFEATURES")
    ap.add_argument("--out", default="", help="also append the JSON lines to this file")
    args = ap.parse_args(argv)

    import numpy as np
    import mojolearn

    shapes = []
    for item in args.shapes.split(","):
        rows, features = item.lower().split("x")
        shapes.append((int(rows), int(features)))

    warm = make_mixture(np, 2000, 8, SEED)
    mojolearn.DBSCAN(eps=eps_for(np, warm), min_samples=MIN_SAMPLES, metric="euclidean",
                     algorithm="brute").fit(warm)

    lines = []
    for index, (rows, features) in enumerate(shapes):
        x = make_mixture(np, rows, features, SEED + 1 + index)
        eps = eps_for(np, x)
        est = mojolearn.DBSCAN(eps=eps, min_samples=MIN_SAMPLES, metric="euclidean", algorithm="brute")
        t0 = time.perf_counter()
        est.fit(x)
        ms = (time.perf_counter() - t0) * 1000.0
        labels = np.asarray(est.labels_, dtype=np.int32).reshape(-1)
        line = {
            "shape": "%dx%d" % (rows, features),
            "eps": eps,
            "min_samples": MIN_SAMPLES,
            "ms": round(ms, 3),
            "clusters": int(labels.max()) + 1 if labels.size else 0,
            "noise_fraction": float((labels < 0).mean()) if labels.size else 0.0,
            "labels_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
            "mojolearn": getattr(mojolearn, "__version__", "unknown"),
        }
        lines.append(json.dumps(line, sort_keys=True))
        print(lines[-1], flush=True)
    if args.out:
        with open(args.out, "a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
