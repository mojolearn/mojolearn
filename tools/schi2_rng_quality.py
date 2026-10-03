#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""QUALITY ONLY (no timing): SkewedChi2Sampler kernel_rel_error over several
random_states, arm A = the host MT19937 draw (sklearn's numbers), arm B = the
device draw (MOJOLEARN_SCHI2_FAST_DEVRNG), on the board's skewed-chi2 data.

    python tools/schi2_rng_quality.py <rows-full data dir> <dataset> [seeds]

The FAST x_neighbors binding must be built with -D MOJOLEARN_SCHI2_FAST_DEVRNG
(arm B); arm A turns the switch off in Python (`_kap2` -> False), so both arms
run in one process on one build. Prints one SCHI2-RNGQ line per arm:
mean, sd and every value of kernel_rel_error (tools/bench_board_algos.py's
metric: ||Z Z^T - K||_F / ||K||_F on the 1000 query rows)."""
import os
import statistics
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
os.environ.setdefault("MOJOLEARN_NUMERIC_MODE", "fast")


def main():
    data, ds = sys.argv[1], sys.argv[2]
    seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    import numpy as np
    import bench_board_algos as bb
    B, _ = bb._load_block("skewed-chi2", ds, data)
    D = bb.lane_arrays("skewed-chi2", B)
    X, Xq = D["X"], D["Xq"]
    K = bb._kernel_exact("kernel-schi2", Xq, D)
    nk = np.linalg.norm(K)
    import mojolearn._expansion_neighbors as xn
    flags = int(xn.SkewedChi2Sampler()._bind().x_neighbors_kap2_flags())
    if not flags & 2:
        print("SCHI2-RNGQ refused: binding built without MOJOLEARN_SCHI2_FAST_DEVRNG (flags=%d)" % flags)
        sys.exit(2)
    params = dict(skewedness=1.0, n_components=256)
    for arm in ("A", "B"):
        errs = []
        for s in range(seeds):
            est = xn.SkewedChi2Sampler(random_state=1000 + s, **params)
            if arm == "A":
                est._kap2 = lambda bit: False
            Z = np.asarray(est.fit(X).transform(Xq), dtype=np.float64)
            errs.append(float(np.linalg.norm(Z @ Z.T - K) / nk))
        print("SCHI2-RNGQ ds=%s arm=%s rng=%s n=%d mean=%.5f sd=%.5f all=%s" % (
            ds, arm, "host-mt19937" if arm == "A" else "device", len(errs), statistics.mean(errs),
            statistics.stdev(errs) if len(errs) > 1 else 0.0, [round(e, 5) for e in errs]))


if __name__ == "__main__":
    main()
