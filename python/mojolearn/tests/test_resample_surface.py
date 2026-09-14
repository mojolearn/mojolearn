# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The gate for the Python surface of `mojolearn.resample` (`bootstrap`,
`permutation_test`, `monte_carlo_integrate`; workstream D, 2026-09-14): a
gate on the WIRING (the params lists, the arrays and scalars landing, the
batch-invariance handle, every refusal by name). The arithmetic is gated by
`pixi run check-resample`.

    cd python && python3 -m mojolearn.tests.test_resample_surface

Exit 2 naming `bindings/build_resample.sh` when unbuilt. Written on one
Apple M4 with no built binary in the worktree; the first run is owed.
"""
import sys

import numpy as np

import mojolearn
from mojolearn import resample as rs
from mojolearn.tests._expose_d_harness import Report, bind_or_exit, mode, run


def _x(n=96, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random(n, dtype=np.float32) * 2.0 - 1.0).astype(np.float32)


def _bits_same(a, b):
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return a.shape == b.shape and np.array_equal(a.view(np.uint32), b.view(np.uint32))


def arm_bootstrap(rep):
    x = _x()
    b = rs.bootstrap(x, statistic="mean", n_resamples=128, random_state=3)
    d = np.asarray(b.distribution)
    rep.check("BOOT", d.shape == (128,) and np.asarray(b.sorted_distribution).shape == (128,), "distribution and sorted_distribution (R,)")
    rep.check("BOOT", _bits_same(np.sort(d), np.asarray(b.sorted_distribution)), "sorted_distribution is the sorted distribution, bit for bit")
    rep.check("BOOT", abs(b.point_estimate - float(np.mean(x.astype(np.float64)))) < 1e-5, "point_estimate is the sample mean at 1e-5", (b.point_estimate, float(np.mean(x))))
    lo, hi = b.confidence_interval
    rep.check("BOOT", lo <= b.point_estimate <= hi and 0 <= b.order_low <= b.order_high < 128, "the interval brackets the estimate and the order positions are in range", (lo, hi, b.order_low, b.order_high))
    rep.check("BOOT", b.standard_error > 0.0, "standard_error positive", b.standard_error)
    whole = np.asarray(rs.bootstrap(x, n_resamples=64, random_state=3).distribution)
    part = np.asarray(rs.bootstrap(x, n_resamples=32, random_state=3, r_first=16).distribution)
    rep.check("BOOT", _bits_same(part, whole[16:48]), "r_first: replicates 16..47 of a 64-run equal a 32-run at r_first=16, bit for bit (the batch-invariance handle)")
    two = np.ascontiguousarray(np.stack([x, x * np.float32(0.5) + np.float32(0.1)], 1))
    for stat in ("std", "pearson", "diff_means"):
        r = rs.bootstrap(two, statistic=stat, n_resamples=32, random_state=1, method="basic")
        rep.check("BOOT", np.isfinite(r.point_estimate) and np.isfinite(np.asarray(r.distribution)).all(), "statistic %r on a two-column sample, method='basic'" % stat)
    q = rs.bootstrap(x, statistic="quantile", q_or_prop=0.25, n_resamples=32, random_state=1)
    t = rs.bootstrap(x, statistic="trimmed_mean", q_or_prop=0.1, n_resamples=32, random_state=1, alternative="less")
    rep.check("BOOT", np.isfinite(q.point_estimate) and np.isfinite(t.point_estimate), "quantile and trimmed_mean read q_or_prop", (q.point_estimate, t.point_estimate))


def arm_permutation(rep):
    x, y = _x(64, 0), (_x(48, 1) + np.float32(0.5)).astype(np.float32)
    p = rs.permutation_test(x, y, statistic="diff_means", n_resamples=128, random_state=3)
    rep.check("PERM", np.asarray(p.null_distribution).shape == (128,), "null_distribution (R,)")
    rep.check("PERM", 0.0 < p.pvalue <= 1.0, "p-value in (0, 1], never exactly zero (DEVIATION 1702's +1)", p.pvalue)
    rep.check("PERM", abs(p.statistic - (float(np.mean(x.astype(np.float64))) - float(np.mean(y.astype(np.float64))))) < 1e-4, "observed statistic is the difference of means at 1e-4")
    rep.check("PERM", p.count_less + p.count_greater >= 0 and p.count_less <= 128 and p.count_greater <= 128, "the two counts are carried and bounded", (p.count_less, p.count_greater))
    whole = np.asarray(rs.permutation_test(x, y, n_resamples=64, random_state=3).null_distribution)
    part = np.asarray(rs.permutation_test(x, y, n_resamples=32, random_state=3, r_first=16).null_distribution)
    rep.check("PERM", _bits_same(part, whole[16:48]), "r_first: permutations 16..47 equal the slice of the whole run, bit for bit")
    g = rs.permutation_test(x, y, statistic="diff_means", n_resamples=32, random_state=3, alternative="greater")
    rep.check("PERM", 0.0 < g.pvalue <= 1.0, "alternative='greater'", g.pvalue)


def arm_monte_carlo(rep):
    c = rs.monte_carlo_integrate("const", [0.0, 0.0], [1.0, 2.0], 512, random_state=1)
    rep.check("MC", c.volume == 2.0 and c.mean == 1.0 and c.integral == 2.0 and c.closed_form == 2.0, "the constant integrand is exact: mean 1, integral = volume = closed form", (c.integral, c.mean, c.volume, c.closed_form))
    s = rs.monte_carlo_integrate("sum", [0.0, 0.0], [1.0, 2.0], 4096, random_state=1)
    rep.check("MC", abs(s.integral - s.closed_form) < 0.05 * abs(s.closed_form), "the sum integrand within 5 percent of its closed form at 4096 draws", (s.integral, s.closed_form))
    p = rs.monte_carlo_integrate("product", [0.0, 0.0], [1.0, 2.0], 4096, random_state=1)
    rep.check("MC", abs(p.integral - p.closed_form) < 0.05 * abs(p.closed_form), "the product integrand within 5 percent of its closed form", (p.integral, p.closed_form))
    s2 = rs.monte_carlo_integrate("sum", [0.0, 0.0], [1.0, 2.0], 4096, random_state=1)
    if mode() == "identical":
        rep.check("MC", s2.integral == s.integral, "the same seed integrates to the same bits on this box")
    else:
        rep.report_only("MC", s2.integral == s.integral, "same seed twice")


def arm_refusals(rep):
    x = _x(32)
    rep.raises("REFUSE", ValueError, "bca", "method='bca' by name (DEVIATION 1699)", rs.bootstrap, x, method="bca", n_resamples=8)
    rep.raises("REFUSE", ValueError, "statistic", "an unknown statistic", rs.bootstrap, x, statistic="median", n_resamples=8)
    rep.raises("REFUSE", ValueError, "alternative", "an unknown alternative", rs.bootstrap, x, alternative="both", n_resamples=8)
    rep.raises("REFUSE", ValueError, "1-D or 2-D", "a 3-D sample", rs.bootstrap, np.zeros((2, 2, 2), np.float32), n_resamples=8)
    rep.raises("REFUSE", Exception, "", "n_resamples=0, refused on the Mojo host", rs.bootstrap, x, n_resamples=0)
    rep.raises("REFUSE", Exception, "", "pearson on a one-column sample, refused on the Mojo host by name", rs.bootstrap, x, statistic="pearson", n_resamples=8)
    bad = x.copy(); bad[7] = np.float32("nan")
    rep.raises("REFUSE", Exception, "", "a NaN cell, refused on the Mojo host", rs.bootstrap, bad, n_resamples=8)
    rep.raises("REFUSE", Exception, "", "confidence_level=1.5, refused on the Mojo host", rs.bootstrap, x, n_resamples=8, confidence_level=1.5)
    rep.raises("REFUSE", ValueError, "integrand", "an unknown integrand", rs.monte_carlo_integrate, "cube", [0.0, 0.0], [1.0, 1.0], 8)
    rep.raises("REFUSE", ValueError, "2 values", "a three-dimensional box", rs.monte_carlo_integrate, "sum", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 8)
    rep.raises("REFUSE", Exception, "", "n_samples=0, refused on the Mojo host", rs.monte_carlo_integrate, "sum", [0.0, 0.0], [1.0, 1.0], 0)
    rep.raises("REFUSE", TypeError, "n_resamples", "a float n_resamples", rs.permutation_test, x, x, n_resamples=8.0)


def arm_provenance(rep):
    rep.check("PROVENANCE", "resample" in mojolearn.__all__ and all(n in rs.__all__ for n in ("bootstrap", "permutation_test", "monte_carlo_integrate")), "mojolearn.resample and its three functions exported")
    from mojolearn import _backend
    rep.check("PROVENANCE", str(_backend.binding("_mojolearn_resample").resample_vendor()) == mojolearn.vendor(), "the binding's vendor read-back is the package's")


def main(out=sys.stdout):
    bind_or_exit("_mojolearn_resample", "build_resample.sh")
    rep = Report("test_resample_surface")
    return run("test_resample_surface", [("BOOT", arm_bootstrap), ("PERM", arm_permutation), ("MC", arm_monte_carlo),
                                         ("REFUSE", arm_refusals), ("PROVENANCE", arm_provenance)], rep, out)


if __name__ == "__main__":
    sys.exit(main())
