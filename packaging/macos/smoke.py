# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Import the installed wheel and run a real fit on every estimator family.

Not a hello-world. It plants four separated clusters and requires k-means to
recover them, and requires k-NN's nearest neighbor of a point to be itself.
An import-only smoke test would pass on a wheel whose GPU path is broken.

`--no-gpu` runs everything EXCEPT the fits: import, version, the estimator
classes, the named absences, and argument validation. It exists for one
specific environment and should not be used anywhere else.

    GitHub's hosted macOS runner is an "Apple M1 (Virtual)" that reports no
    GPU to system_profiler and CANNOT create a Metal function using
    `block_sum` -- measured 2026-08-20, tools/hosted_block_probe.mojo fails
    there while the trivial elementwise kernel passes. So CI can verify that
    the wheel BUILDS and INSTALLS on 3.10 through 3.14, and cannot verify
    that it COMPUTES. Pretending otherwise, by letting the GPU failure pass
    under continue-on-error, produced a green run in which every interpreter
    had failed. A check that cannot run must say so, not report success.

GPU execution is verified on real Apple silicon by
packaging/macos/verify_wheel.sh without this flag.
"""
import os
import sys

import numpy as np

import mojolearn

NO_GPU = "--no-gpu" in sys.argv

if NO_GPU:
    # Everything that does not touch the device. Import already exercised the
    # extension load, which is what the dylib staging and the wheel tag are
    # actually about.
    assert mojolearn.__version__
    assert hasattr(mojolearn, "KMeans")
    assert hasattr(mojolearn, "NearestNeighbors")
    # Tree estimators are on the surface as submodules whatever __init__
    # exports; their extensions must import.
    from mojolearn import ensemble, randomforest, extratrees  # noqa: F401
    for maybe in ("DBSCAN", "PCA", "LinearRegression"):
        # Either exported, or a NAMED absence. An AttributeError without the
        # named-absence text is a typo in __init__, not a decision.
        try:
            getattr(mojolearn, maybe)
        except AttributeError as exc:
            assert "no caller-facing surface" in str(exc), maybe
    # Validation happens before any device work, so it is reachable here.
    try:
        mojolearn.KMeans(n_clusters=99).fit(np.zeros((4, 2), dtype=np.float32))
    except ValueError as exc:
        assert "exceeds" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("n_clusters > n_samples should raise")
    print(
        f"v{mojolearn.__version__} py{sys.version_info.major}."
        f"{sys.version_info.minor} IMPORT+API ok (--no-gpu, device untested)"
    )
    raise SystemExit(0)

rng = np.random.default_rng(0)
centers = np.array([[0, 0], [40, 40], [0, 40], [40, 0]], dtype=np.float32)
X = np.repeat(centers, 100, axis=0) + rng.normal(0, 1, (400, 2)).astype(np.float32)

km = mojolearn.KMeans(
    n_clusters=4, init="array", init_centroids=centers + 4, max_iter=50
).fit(X)
for c in range(4):
    block = km.labels_[c * 100 : (c + 1) * 100]
    assert len(set(block.tolist())) == 1, f"cluster {c} split across labels"
assert len({tuple(r) for r in km.cluster_centers_.round(0)}) == 4, "centroids merged"

nn = mojolearn.NearestNeighbors(n_neighbors=3).fit(X)
d, i = nn.kneighbors(X[:50])
assert (i[:, 0] == np.arange(50)).all(), "a point is not its own nearest neighbour"
assert (d[:, 0] < 1e-3).all(), "self-distance is not ~0"

# THE TREES. Imported from their modules, not from the package top level, so
# this smoke holds whether or not __init__ re-exports them; the wheel ships
# the three tree extensions either way and each must load and FIT.
from mojolearn.ensemble import GradientBoosting
from mojolearn.randomforest import RandomForestClassifier
from mojolearn.extratrees import ExtraTreesRegressor

Xt = rng.random((512, 4), dtype=np.float32)
yr = Xt[:, 0] * 2.0 + Xt[:, 1]
yc = (yr > np.median(yr)).astype(np.float32)

gb = GradientBoosting(loss="RMSE", n_estimators=4, max_depth=3, border_count=16).fit(Xt, yr)
pg = gb.predict(Xt)
assert pg.shape == (512,), pg.shape
assert np.corrcoef(pg, yr)[0, 1] > 0.5, "GBDT learned nothing"

rf = RandomForestClassifier(n_estimators=4, max_depth=6, random_state=7).fit(Xt, yc)
pr = rf.predict(Xt)
assert pr.shape == (512,), pr.shape
assert (pr == yc).mean() > 0.7, "RF learned nothing"

et = ExtraTreesRegressor(n_estimators=3, max_depth=6, random_state=7).fit(Xt, yr)
pe = et.predict(Xt)
assert pe.shape == (512,), pe.shape
assert np.corrcoef(pe, yr)[0, 1] > 0.5, "ExtraTrees learned nothing"

# THE CLASSICAL ESTIMATORS, same fits as smoke_estimators.py, imported from
# their modules for the same reason as the trees above.
from mojolearn.decomposition import PCA, TruncatedSVD
from mojolearn.density import DBSCAN
from mojolearn.linear_model import LinearRegression

clouds = np.vstack([
    rng.normal((-4, -4), 0.08, (40, 2)),
    rng.normal((4, 4), 0.08, (40, 2)),
    [[-10, 8], [10, -8], [0, 10], [10, 0]],
]).astype(np.float32)
labels = DBSCAN(eps=0.35, min_samples=5).fit_predict(clouds)
assert len(set(labels[:40].tolist())) == 1 and len(set(labels[40:80].tolist())) == 1
assert labels[0] != labels[40] and (labels[-4:] == -1).all(), "DBSCAN wrong"

xc = rng.normal(size=(512, 6)).astype(np.float32)
xc[:, 1] += 0.7 * xc[:, 0]
pca = PCA(n_components=6).fit(xc)
np.testing.assert_allclose(
    pca.explained_variance_, np.linalg.eigvalsh(np.cov(xc, rowvar=False))[::-1],
    rtol=2e-3, atol=2e-4)
svd = TruncatedSVD(n_components=6).fit(xc)
np.testing.assert_allclose(
    svd.singular_values_, np.linalg.svd(xc, compute_uv=False), rtol=2e-3, atol=2e-3)
coef = np.array([1.5, -2.0, 0.25, 4.0, -1.0, 0.5], dtype=np.float32)
ols = LinearRegression().fit(xc, xc @ coef + np.float32(3.25))
np.testing.assert_allclose(ols.coef_, coef, rtol=3e-3, atol=3e-3)

# ==========================================================================
# ONE ENTRY POINT PER REMAINING BINDING. Added 2026-09-01.
# ==========================================================================
# Everything above reaches FIVE of the twelve extensions this wheel ships:
# _mojolearn, _mojolearn_estimators, _mojolearn_gbdt, _mojolearn_rf and
# _mojolearn_trees. MEASURED by wrapping `_backend.binding` and recording
# every name it was asked for, because guessing from estimator names gives
# the wrong answer -- PCA and TruncatedSVD look like they cover
# `_mojolearn_linalg` and do not (`decomposition.py:155`, `:382` both bind
# `_mojolearn_estimators`), and LinearRegression does not reach
# `_mojolearn_solver`.
#
# So SEVEN extensions shipped in every wheel, across all three tiers, had
# never been launched by wheel verification: 21 of 36 artifacts. Each is
# user-reachable, so that was untested shipped surface, not dead weight. Any
# one of them could import cleanly and die at its first kernel launch, which
# is precisely the MACOSX_DEPLOYMENT_TARGET failure this project has already
# shipped once, and verify_wheel.sh would have passed the wheel.
#
# These are LAUNCH gates, not accuracy gates. The arithmetic belongs to each
# lane's own checks; what is asserted here is that the extension loads and a
# kernel runs. They are kept to the smallest fixture that reaches a launch,
# because this file runs once per interpreter per tier.
series = (np.sin(np.arange(64, dtype=np.float64) / 3.0) + 1.0)
xs = rng.random((48, 3)).astype(np.float32)
ys = (xs[:, 0] > 0.5).astype(np.int64)

mojolearn.AgglomerativeClustering(n_clusters=2).fit(xs)   # _mojolearn_solver
mojolearn.SVC().fit(xs, ys)                               # _mojolearn_svm
mojolearn.kpss_test(series)                               # _mojolearn_tsa
assert mojolearn.metrics.accuracy_score(ys, ys) == 1.0    # _mojolearn_metrics
mojolearn.ARIMA(order=(1, 0, 0)).fit(series)              # _mojolearn_arima

from mojolearn import _training_impl as _T             # _mojolearn_training
_ce = _T.cross_entropy(
    rng.standard_normal((4, 3)).astype(np.float32),
    np.array([0, 1, 2, 0], dtype=np.int32),
)
assert np.isfinite(_ce) and _ce > 0.0, _ce

# _mojolearn_linalg IS IDENTITY-ONLY BY DESIGN, so BOTH ARMS are asserted
# here rather than the call being skipped outside the identical tier.
# `mojolearn.linalg` publishes a cross-vendor identity profile and REFUSES
# by name on any tier that does not make that claim (`_linalg_impl.py:269`),
# a refusal added after a mislabeled deterministic build on the 2026-08-29
# Apple stability run. A skip here would leave the refusal itself ungated,
# and this extension is only ever launched on the identical arm.
_mode_now = os.environ.get("MOJOLEARN_NUMERIC_MODE", "fast")
if _mode_now == "identical":
    _a = rng.random((8, 4)).astype(np.float32)
    _p = mojolearn.linalg.matmul(_a, _a.T)
    assert _p.shape == (8, 8), _p.shape
else:
    try:
        mojolearn.linalg.matmul(xs, xs.T)
    except RuntimeError as _e:
        assert "identity claim" in str(_e), str(_e)
    else:
        raise AssertionError(
            f"mojolearn.linalg.matmul did NOT refuse on the {_mode_now} tier;"
            " it publishes a cross-vendor identity profile and only the"
            " identical build may serve it"
        )

# THE MODE THAT ACTUALLY LOADED, read back from the binary where it can be.
# verify_wheel.sh runs this file once per mode and checks the word.
mode = mojolearn.numeric_mode()
assert mode == os.environ.get("MOJOLEARN_NUMERIC_MODE", "fast"), mode

# THE VENDOR THAT ACTUALLY LOADED, read back from the binary (2026-08-29,
# docs/LINUX_WHEEL.md). On the macOS wheel it is 'metal'; the Linux smoke
# (packaging/linux/smoke.py) asserts 'cuda' or 'hip' the same way.
# MOJOLEARN_SMOKE_VENDOR overrides the expectation for a Linux run of THIS
# file. A None here is a binary built without checks/vendor.mojo, which a
# release build cannot be.
vendor = mojolearn.vendor()
assert vendor == os.environ.get("MOJOLEARN_SMOKE_VENDOR", "metal"), vendor
assert rf.vendor_used() == vendor, rf.vendor_used()

print(
    f"v{mojolearn.__version__} py{sys.version_info.major}.{sys.version_info.minor}"
    f" mode={mode} vendor={vendor} kmeans n_iter={km.n_iter_} knn tile={nn.used_query_tile_}"
    f" gbdt rf et dbscan pca svd ols ok"
)
