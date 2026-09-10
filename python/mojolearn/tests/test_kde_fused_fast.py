# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""DEVIATION 2490: the fused FAST score pass of KernelDensity.

The FAST tier scores queries in one kernel (distance, log-kernel, log-weight
and an online log-sum-exp per thread) instead of the staged path that
materializes two n_query x n_train matrices. The staged path is still the
code IDENTICAL runs, and it still runs in FAST whenever an identity trace is
requested, so the two arms can be compared in one process: setting
MOJOLEARN_IDENTITY_TRACE before a call selects the staged path (the trace is
read once per call).

These tests are FAST-only. Under IDENTICAL or DETERMINISTIC both calls take
the staged path and the comparison is trivially exact, so they skip.
"""
import math
import os
import random

import pytest

from mojolearn import KernelDensity
from mojolearn._array import Array

FLOAT_MIN = -3.4028234663852886e38


def _mode():
    return KernelDensity().numeric_mode_used()


pytestmark = pytest.mark.skipif(
    os.environ.get("MOJOLEARN_NUMERIC_MODE", "fast") != "fast",
    reason="DEVIATION 2490 is a FAST-only arm",
)


def _matrix(rng, rows, cols, shift=0.0):
    return Array.from_list(
        [[rng.gauss(0.0, 1.0) + shift for _ in range(cols)] for _ in range(rows)],
        dtype="<f4",
    )


def _scores(est, X, staged, tmp_path):
    if staged:
        os.environ["MOJOLEARN_IDENTITY_TRACE"] = str(tmp_path / "staged.trace")
    else:
        os.environ.pop("MOJOLEARN_IDENTITY_TRACE", None)
    try:
        return list(est.score_samples(X))
    finally:
        os.environ.pop("MOJOLEARN_IDENTITY_TRACE", None)


def _is_sentinel(v):
    return v <= -1e30 or v != v


KERNELS = ("gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine")
METRICS = ("euclidean", "sqeuclidean", "manhattan", "chebyshev", "cosine", "minkowski")


@pytest.mark.parametrize("n_train,n_query,d,seed", [
    (700, 333, 1, 1),      # DPAD 4 with three pad lanes
    (900, 257, 7, 2),      # the check fixture's width, DPAD 8
    (500, 129, 33, 3),     # DPAD 36
    (300, 65, 64, 4),      # the widest register row
    (200, 41, 70, 5),      # the wide-row kernel (feature chunks)
    (120, 17, 150, 6),     # wide rows, more than two chunks
])
@pytest.mark.parametrize("weighted", [False, True])
def test_fused_equals_staged_within_float32_fold_tolerance(
    n_train, n_query, d, seed, weighted, tmp_path
):
    if _mode() != "fast":
        pytest.skip("FAST-only arm")
    rng = random.Random(seed)
    worst = 0.0
    for metric in METRICS:
        shift = 3.0 if metric == "cosine" else 0.0
        Xt = _matrix(rng, n_train, d, shift)
        Xq = _matrix(rng, n_query, d, shift)
        w = None
        if weighted:
            w = Array.from_list([rng.uniform(0.2, 2.0) for _ in range(n_train)], dtype="<f4")
        for kernel in KERNELS:
            kw = {"metric_params": {"p": 2}} if metric == "minkowski" else {}
            est = KernelDensity(bandwidth=0.9, kernel=kernel, metric=metric, **kw)
            est.fit(Xt, sample_weight=w)
            fused = _scores(est, Xq, False, tmp_path)
            staged = _scores(est, Xq, True, tmp_path)
            assert len(fused) == len(staged) == n_query
            for a, b in zip(fused, staged):
                # The sentinels (FLOAT_MIN for a row no point reaches, -inf
                # for an overflowed row) must agree exactly, both ways.
                if _is_sentinel(a) or _is_sentinel(b):
                    assert a == b, (metric, kernel, a, b)
                    continue
                rel = abs(a - b) / max(1.0, abs(b))
                worst = max(worst, rel)
                assert rel < 2e-4, (metric, kernel, a, b)
    assert worst > 0.0 or n_train < 300  # the two folds are different code


def test_all_out_of_range_row_scores_float_min_not_minus_inf(tmp_path):
    """A compact kernel with every training point outside the bandwidth:
    the staged path folds n copies of FLOAT_MIN to FLOAT_MIN; the fused
    online fold must give the same sentinel, not -inf (a base-2 rescale
    overflowed here once)."""
    if _mode() != "fast":
        pytest.skip("FAST-only arm")
    Xt = Array.from_list([[float(i)] for i in range(10, 40)], dtype="<f4")
    Xq = Array.from_list([[0.0], [100.0], [25.1]], dtype="<f4")
    for kernel in ("tophat", "epanechnikov", "linear", "cosine"):
        est = KernelDensity(bandwidth=0.25, kernel=kernel).fit(Xt)
        fused = _scores(est, Xq, False, tmp_path)
        staged = _scores(est, Xq, True, tmp_path)
        assert fused[0] == staged[0] == pytest.approx(FLOAT_MIN, rel=1e-6)
        assert fused[1] == staged[1] == pytest.approx(FLOAT_MIN, rel=1e-6)
        assert math.isfinite(fused[2]) and fused[2] > -1e30, kernel
        assert fused[2] == pytest.approx(staged[2], rel=1e-4)


def test_query_score_does_not_depend_on_its_batch(tmp_path):
    """Case D of the native launch-invariance gate, on the fused arm: a
    query scored alone and inside a batch of 3000 gives the same bytes."""
    if _mode() != "fast":
        pytest.skip("FAST-only arm")
    rng = random.Random(9)
    Xt = _matrix(rng, 400, 5)
    q = [rng.gauss(0.0, 1.0) for _ in range(5)]
    filler = [[rng.gauss(0.0, 1.0) for _ in range(5)] for _ in range(2999)]
    est = KernelDensity(bandwidth=0.8).fit(Xt)
    alone = _scores(est, Array.from_list([q], dtype="<f4"), False, tmp_path)[0]
    big = _scores(est, Array.from_list(filler[:1500] + [q] + filler[1500:], dtype="<f4"), False, tmp_path)
    assert big[1500] == alone


def test_matches_sklearn_when_available(tmp_path):
    sklearn = pytest.importorskip("sklearn.neighbors")
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(0)
    Xt = rng.standard_normal((1500, 6)).astype(np.float32)
    Xq = rng.standard_normal((500, 6)).astype(np.float32)
    for kernel in ("gaussian", "exponential"):
        for metric in ("euclidean", "manhattan", "chebyshev"):
            ours = np.asarray(
                KernelDensity(bandwidth=0.7, kernel=kernel, metric=metric).fit(Xt).score_samples(Xq),
                dtype=np.float64,
            )
            ref = sklearn.KernelDensity(
                bandwidth=0.7, kernel=kernel, metric=metric, rtol=0, atol=0
            ).fit(Xt).score_samples(Xq)
            assert np.abs(ours - ref).max() < 2e-4, (kernel, metric)
