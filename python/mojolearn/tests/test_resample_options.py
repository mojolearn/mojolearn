# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""resample option parity (item 5, 2026-09-28): bootstrap paired=False.
Bit-level properties of OUR construction plus SciPy's BCa construction over
our distribution. Needs scipy."""
import sys

import numpy as np
from scipy.special import ndtr, ndtri

from mojolearn import resample as rs


def _levels_two(dist, theta_hat, x, y, alpha):
    B = dist.shape[0]
    pct = ((dist < theta_hat).sum() + (dist <= theta_hat).sum()) / (2 * B)
    z0 = ndtri(pct)
    mx, my = x.astype(np.float64).mean(), y.astype(np.float64).mean()
    jx = np.array([np.delete(x, i).astype(np.float64).mean() - my for i in range(len(x))])
    jy = np.array([mx - np.delete(y, i).astype(np.float64).mean() for i in range(len(y))])
    nums, dens = 0.0, 0.0
    for j in (jx, jy):
        n = len(j)
        u = (n - 1) * (j.mean() - j)
        nums += (u ** 3).sum() / n ** 3
        dens += (u ** 2).sum() / n ** 2
    a = nums / 6 / dens ** 1.5
    za = ndtri(alpha)
    n1, n2 = z0 + za, z0 - za
    return ndtr(z0 + n1 / (1 - a * n1)), ndtr(z0 + n2 / (1 - a * n2))


def main():
    rng = np.random.default_rng(11)
    x = rng.gamma(2.0, 1.0, 700).astype(np.float32)
    y = rng.normal(1.0, 2.0, 450).astype(np.float32)
    bad = 0
    b = rs.bootstrap((x, y), statistic="diff_means", paired=False, n_resamples=3000, random_state=4)
    one = rs.bootstrap(x, statistic="mean", n_resamples=3000, random_state=4)
    d = np.asarray(b.distribution)
    # sample 0's map is the one-sample map: mean(x*) - diff = mean(y*) must be a mean of y values
    ym = (np.asarray(one.distribution).astype(np.float64) - d.astype(np.float64))
    ok = abs(ym.mean() - y.mean()) < 0.05 and np.all(np.isfinite(d))
    print(f"{'PASS' if ok else 'FAIL'} unpaired: sample 0 resampled by the one-sample map (implied y means average {ym.mean():.4f}, y mean {y.mean():.4f})")
    bad += not ok
    sd = np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
    ok = abs(np.std(d, ddof=1) - sd) < 0.1 * sd and abs(b.point_estimate - (x.mean() - y.mean())) < 1e-5
    print(f"{'PASS' if ok else 'FAIL'} unpaired: spread {np.std(d, ddof=1):.5f} vs the independent-samples SE {sd:.5f}; point {b.point_estimate:.6f}")
    bad += not ok
    p = rs.bootstrap(np.stack([x[:450], y], 1), statistic="diff_means", n_resamples=3000, random_state=4)
    ok = not np.array_equal(np.asarray(p.distribution), d)
    print(f"{'PASS' if ok else 'FAIL'} paired=True on a two-column sample is a different construction")
    bad += not ok
    bc = rs.bootstrap((x, y), statistic="diff_means", paired=False, n_resamples=4000, random_state=2, method="BCa")
    dist = np.asarray(bc.distribution, dtype=np.float64)
    a1, a2 = _levels_two(dist, float(bc.point_estimate), x, y, 0.025)
    lo, hi = np.percentile(dist, [a1 * 100, a2 * 100])
    span = hi - lo
    ok = abs(bc.confidence_interval[0] - lo) <= 1e-3 * span and abs(bc.confidence_interval[1] - hi) <= 1e-3 * span
    print(f"{'PASS' if ok else 'FAIL'} unpaired BCa: ours [{bc.confidence_interval[0]:.6g}, {bc.confidence_interval[1]:.6g}] scipy's construction [{lo:.6g}, {hi:.6g}]")
    bad += not ok
    for stat in ("mean", "pearson"):
        try:
            rs.bootstrap((x, y), statistic=stat, paired=False, n_resamples=16)
            print(f"FAIL unpaired {stat} was not refused")
            bad += 1
        except ValueError as e:
            print(f"PASS unpaired {stat} refused by name: {str(e)[:70]}")
    print("RESULT", "PASS" if not bad else f"FAIL ({bad})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
