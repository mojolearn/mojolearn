# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""resample.bootstrap(method='BCa') against SciPy's `_bca_interval`
arithmetic in float64 over OUR bootstrap distribution (the two libraries draw
different replicates, so the distributions differ; the interval CONSTRUCTION
given a distribution is what is compared). Needs scipy. DEVIATION 1699 / 5410."""
import sys

import numpy as np
from scipy.special import ndtr, ndtri

from mojolearn import resample as rs


def _jack(data, stat):
    n = data.shape[0]
    out = []
    for i in range(n):
        d = np.delete(data, i, axis=0).astype(np.float64)
        if stat == "mean":
            out.append(d.mean())
        elif stat == "std":
            out.append(d.std(ddof=1))
        else:
            out.append(d[:, 0].mean() - d[:, 1].mean())
    return np.asarray(out)


def _scipy_levels(dist, theta_hat, jack, alpha):
    B = dist.shape[0]
    pct = ((dist < theta_hat).sum() + (dist <= theta_hat).sum()) / (2 * B)
    z0 = ndtri(pct)
    dot = jack.mean()
    num = ((dot - jack) ** 3).sum()
    den = 6.0 * ((dot - jack) ** 2).sum() ** 1.5
    a = num / den
    za = ndtri(alpha)
    n1, n2 = z0 + za, z0 - za
    return ndtr(z0 + n1 / (1 - a * n1)), ndtr(z0 + n2 / (1 - a * n2))


def main():
    rng = np.random.default_rng(7)
    x = rng.gamma(2.0, 1.5, 600).astype(np.float32)
    two = np.stack([x, rng.normal(0, 1, 600).astype(np.float32)], 1).astype(np.float32)
    bad = 0
    for stat, data in (("mean", x), ("std", x), ("diff_means", two)):
        for cl in (0.9, 0.95):
            b = rs.bootstrap(data, statistic=stat, n_resamples=4000, method="BCa", random_state=3, confidence_level=cl)
            dist = np.asarray(b.distribution, dtype=np.float64)
            a1, a2 = _scipy_levels(dist, float(b.point_estimate), _jack(data, stat), (1 - cl) / 2)
            lo, hi = np.percentile(dist, [a1 * 100, a2 * 100])
            span = hi - lo
            ok = abs(b.confidence_interval[0] - lo) <= 1e-3 * span and abs(b.confidence_interval[1] - hi) <= 1e-3 * span
            print(f"{'PASS' if ok else 'FAIL'} BCa {stat} cl={cl}: ours [{b.confidence_interval[0]:.6g}, "
                  f"{b.confidence_interval[1]:.6g}] scipy's construction [{lo:.6g}, {hi:.6g}] (levels {a1:.6f}, {a2:.6f})")
            bad += not ok
    b2 = rs.bootstrap(x, method="bca", n_resamples=64, random_state=1)
    b3 = rs.bootstrap(x, method="BCA", n_resamples=64, random_state=1)
    same = np.array_equal(np.asarray(b2.distribution), np.asarray(b3.distribution)) and b2.confidence_interval == b3.confidence_interval
    print(f"{'PASS' if same else 'FAIL'} method spelled bca / BCA (SciPy lowercases it)")
    bad += not same
    try:
        rs.bootstrap(x, statistic="quantile", q_or_prop=0.5, method="BCa", n_resamples=64)
        print("FAIL BCa on an order statistic was not refused")
        bad += 1
    except Exception as e:  # noqa: BLE001
        print("PASS BCa on an order statistic refused by name:", str(e)[:90])
    print("RESULT", "PASS" if not bad else f"FAIL ({bad})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
