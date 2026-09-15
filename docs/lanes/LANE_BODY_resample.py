# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""identity_break lane bodies for the resampling door (workstream D,
2026-09-14), for the harness's owner to merge into tools/identity_break.py.
Function lanes in the metrics lane's style: no estimator, the probe is
n/a:function, and the hashed things are the distributions and the host
scalars. The samples are fixture columns, so `ties` hands the sorts
repeated keys and `denormal` hands the folds subnormals.

SUPERSEDED BY THE MERGED LANES (2026-09-14 night). As drafted below, both
bodies exceed a comptime limit and refuse by name on every vendor: the
sorted bootstrap statistics at 2048 x 4096 cells (RESAMPLE_MAX_SORT_CELLS is
1 << 22) and the permutation test at a pooled 4096 (PERM_MAX_POOLED is 1024).
tools/identity_break.py runs quantile and trimmed_mean at 1024 replicates and
the permutation test at 512 per group; read the lanes there, not here.
"""


@lane("bootstrap")
def _(ml, X, yc, yr, Xh=None):
    """resample.bootstrap over the first 4096 values of yr, 2048
    replicates: the Philox index map, the per-replicate folds (mean,
    std), the segmented sort (quantile), the paired two-column statistics
    (pearson, diff_means), the percentile and basic intervals and the
    standard error. Every distribution and every scalar is hashed; the
    r_first slice equality is asserted, as the surface promises it."""
    rs = ml.resample
    x = np.ascontiguousarray(yr[:4096])
    two = np.ascontiguousarray(np.stack([yr[:4096], X[:4096, 3]], 1).astype(np.float32))
    parts = {}
    for name, kw in (("mean", dict(data=x, statistic="mean")),
                     ("std", dict(data=x, statistic="std", method="basic")),
                     ("quantile", dict(data=x, statistic="quantile", q_or_prop=0.25, alternative="less")),
                     ("trimmed", dict(data=x, statistic="trimmed_mean", q_or_prop=0.1, alternative="greater")),
                     ("pearson", dict(data=two, statistic="pearson")),
                     ("diff", dict(data=two, statistic="diff_means", method="basic"))):
        b = rs.bootstrap(n_resamples=2048, random_state=3, **kw)
        parts[name] = _h(b.distribution, b.sorted_distribution,
                         np.asarray([b.point_estimate, b.standard_error, b.confidence_interval[0], b.confidence_interval[1]], dtype=np.float64),
                         np.asarray([b.order_low, b.order_high], dtype=np.int64))
    whole = np.asarray(rs.bootstrap(x, n_resamples=1024, random_state=3).distribution)
    part = np.asarray(rs.bootstrap(x, n_resamples=512, random_state=3, r_first=256).distribution)
    assert np.array_equal(whole[256:768].view(np.uint32), part.view(np.uint32)), "bootstrap lane: r_first slice is not bit-identical"
    parts["r_first"] = _h(part)
    return _fit(parts)


@lane("permutation-test")
def _(ml, X, yc, yr, Xh=None):
    """resample.permutation_test between the fixture's two label groups
    of yr (the first 2048 rows of each), 2048 permutations, three
    alternatives: the pooled Philox permutation map, the between-group
    fold and the conservative p-value (DEVIATION 1702)."""
    rs = ml.resample
    a = np.ascontiguousarray(yr[:4096][yc[:4096] == 0][:2048])
    b = np.ascontiguousarray(yr[:4096][yc[:4096] == 1][:2048])
    if a.size < 8 or b.size < 8:
        a, b = np.ascontiguousarray(yr[:2048]), np.ascontiguousarray(yr[2048:4096])
    parts = {}
    for alt in ("two-sided", "less", "greater"):
        p = rs.permutation_test(a, b, statistic="diff_means", n_resamples=2048, random_state=3, alternative=alt)
        parts[alt] = _h(p.null_distribution, np.asarray([p.statistic, p.pvalue], dtype=np.float64),
                        np.asarray([p.count_less, p.count_greater], dtype=np.int64))
    return _fit(parts)


@lane("monte-carlo")
def _(ml, X, yc, yr, Xh=None):
    """resample.monte_carlo_integrate, the three compiled integrands over
    a box whose corners come from the fixture's first two column minima
    and maxima (host min and max, exact), 65536 draws: the position-mapped
    Philox points and the chunked pinned fold. The closed forms are
    hashed beside the estimates."""
    rs = ml.resample
    lo = [float(np.min(X[:, 0])), float(np.min(X[:, 1]))]
    hi = [float(np.max(X[:, 0])), float(np.max(X[:, 1]))]
    if not (hi[0] > lo[0] and hi[1] > lo[1]):
        lo, hi = [0.0, 0.0], [1.0, 2.0]
    parts = {}
    for f in ("const", "sum", "product"):
        r = rs.monte_carlo_integrate(f, lo, hi, 65536, random_state=1)
        parts[f] = _h(np.asarray([r.integral, r.mean, r.volume, r.closed_form], dtype=np.float64))
    return _fit(parts)
