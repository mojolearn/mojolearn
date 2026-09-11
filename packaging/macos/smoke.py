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

import math
import numpy as np

import mojolearn

# A foreign library's argtypes on the shared ctypes.pythonapi buffer pointers
# (treelite.model's spelling, as cuML imports it), armed after the import and
# before any fit, so every fit below also proves our conversions no longer
# share them (the 0.8.0 bug; tools/check_buffer_foreign_argtypes.py says why
# the order matters).
import importlib.util as _ilu
import pathlib as _pl

_spec = _ilu.spec_from_file_location(
    "check_buffer_foreign_argtypes",
    _pl.Path(__file__).resolve().parents[2] / "tools" / "check_buffer_foreign_argtypes.py")
_foreign = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_foreign)
if _foreign.poison_pythonapi() is None:
    raise SystemExit("foreign argtypes did not land on ctypes.pythonapi.PyObject_GetBuffer")

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


# ==========================================================================
# ONE TIER RULE (DEVIATION 2490, 2026-09-10). THE THREE TREE LANES ship
# fast, deterministic and identical; EVERY OTHER BINDING ships identical
# only. So this file has two halves:
#
#   * The trees, below, fit in whatever tier verify_wheel.sh asked for.
#   * Every other family is a thunk in IDENTICAL_ONLY_LAUNCHES. Under
#     `identical` each must LAUNCH. Under `fast` and `deterministic` each
#     must RAISE, naming the identical tier, and one that answers is the
#     failure: it would mean a lower-tier binary of an identical-only lane
#     got back into the wheel.
#
# A SKIP WOULD LEAVE THE REFUSAL ITSELF UNGATED, which is the failure mode
# this file has already shipped once: a binding no gate launches is untested
# shipped surface. Each thunk is checked on its own rather than the list as
# a whole, so a refusal from the first family cannot hide a lower-tier
# binary of the ninth. The refusal has to be identified by what it SAYS,
# because a bare except would pass any fault under the lower tiers.
# ==========================================================================
_tier = mojolearn._backend.requested_mode()

rng = np.random.default_rng(0)
centers = np.array([[0, 0], [40, 40], [0, 40], [40, 0]], dtype=np.float32)
X = np.repeat(centers, 100, axis=0) + rng.normal(0, 1, (400, 2)).astype(np.float32)

# THE TREES. Imported from their modules, not from the package top level, so
# this smoke holds whether or not __init__ re-exports them; the wheel ships
# the three tree extensions in every tier and each must load and FIT.
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
assert (np.asarray(pr) == yc).mean() > 0.7, "RF learned nothing"

et = ExtraTreesRegressor(n_estimators=3, max_depth=6, random_state=7).fit(Xt, yr)
pe = et.predict(Xt)
assert pe.shape == (512,), pe.shape
assert np.corrcoef(pe, yr)[0, 1] > 0.5, "ExtraTrees learned nothing"

# The sklearn-shaped GBDT adapter rides the gbdt binding, so it too fits in
# every tier.
series = (np.sin(np.arange(64, dtype=np.float64) / 3.0) + 1.0)
xs = rng.random((48, 3)).astype(np.float32)
ys = (xs[:, 0] > 0.5).astype(np.int64)
adapter = mojolearn.GradientBoostingClassifier(n_estimators=2, max_depth=2).fit(xs, ys)
assert adapter.predict(xs).shape == ys.shape
adapter_proba = adapter.predict_proba(xs)
assert adapter_proba.shape == (len(xs), 2), adapter_proba.shape
assert np.asarray(adapter_proba).dtype == np.float32, adapter_proba.dtype

# ------------------------------------------------------------------ the rest
# Everything from here to IDENTICAL_ONLY_LAUNCHES is one thunk per binding
# (or per family where a binding carries several). The fixtures are the
# smallest that reach a kernel launch; the arithmetic belongs to each lane's
# own checks, and what is asserted here is that the extension loads and a
# kernel runs. `_launched` collects what the print line at the end reports.
_launched = {}


def _kmeans():                                            # _mojolearn
    km = mojolearn.KMeans(
        n_clusters=4, init="array", init_centroids=centers + 4, max_iter=50
    ).fit(X)
    for c in range(4):
        block = km.labels_[c * 100 : (c + 1) * 100]
        assert len(set(block.tolist())) == 1, f"cluster {c} split across labels"
    # DEVIATION 2462: estimator outputs are mojolearn.Array (no arithmetic, no
    # ufuncs); np.asarray views them zero-copy so every check is unchanged.
    assert len({tuple(r) for r in np.asarray(km.cluster_centers_).round(0)}) == 4, "centroids merged"
    _launched["kmeans n_iter"] = km.n_iter_


def _knn():                                               # _mojolearn
    nn = mojolearn.NearestNeighbors(n_neighbors=3).fit(X)
    d, i = nn.kneighbors(X[:50])
    d, i = np.asarray(d), np.asarray(i)
    assert (i[:, 0] == np.arange(50)).all(), "a point is not its own nearest neighbour"
    assert (d[:, 0] < 1e-3).all(), "self-distance is not ~0"
    _launched["knn tile"] = nn.used_query_tile_


def _estimators():                                        # _mojolearn_estimators
    from mojolearn.decomposition import PCA, TruncatedSVD
    from mojolearn.density import DBSCAN
    from mojolearn.linear_model import LinearRegression

    clouds = np.vstack([
        rng.normal((-4, -4), 0.08, (40, 2)),
        rng.normal((4, 4), 0.08, (40, 2)),
        [[-10, 8], [10, -8], [0, 10], [10, 0]],
    ]).astype(np.float32)
    labels = np.asarray(DBSCAN(eps=0.35, min_samples=5).fit_predict(clouds))
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


def _solver():                                            # _mojolearn_solver
    mojolearn.AgglomerativeClustering(n_clusters=2).fit(xs)


def _svm():                                               # _mojolearn_svm
    mojolearn.SVC().fit(xs, ys)


def _tsa():                                               # _mojolearn_tsa
    mojolearn.kpss_test(series)


def _metrics():                                           # _mojolearn_metrics
    assert mojolearn.metrics.accuracy_score(ys, ys) == 1.0


def _preprocessing():                                     # _mojolearn_preprocessing
    assert mojolearn.MinMaxScaler().fit_transform(xs).shape == xs.shape
    assert mojolearn.StandardScaler().fit_transform(xs).shape == xs.shape


def _arima():                                             # _mojolearn_arima
    mojolearn.ARIMA(order=(1, 0, 0)).fit(series)


def _gp():                                                # _mojolearn_gp
    # A 16x2 slice of xs, SPREAD to [-2, 2) so a unit-length-scale RBF does
    # not drive K + 2^-20 I toward singular (points packed in [0, 1)^2 would
    # make this launch gate a conditioning test, which it is not). alpha is
    # the surface default 2^-20 spelled explicitly -- the pinned ridge,
    # accepted on every tier (DEVIATIONS 1751/1772) -- and info_ is ASSERTED
    # zero because a failed factorization here is a RESULT, not an exception
    # (DEVIATION 1634), so a smoke that ignored it would pass on a fit that
    # computed nothing.
    gpx = (xs[:16, :2] * np.float32(4.0) - np.float32(2.0))
    gpy = (np.sin(gpx[:, 0]) + 0.5 * gpx[:, 1]).astype(np.float32)
    gpm = mojolearn.GaussianProcessRegressor(
        kernel=mojolearn.RBF(1.0), alpha=2.0 ** -20).fit(gpx, gpy)
    assert gpm.info_ == 0, f"gp factorization failed: info_={gpm.info_}"
    gpp = gpm.predict(gpx)
    assert gpp.shape == (16,), gpp.shape
    assert np.isfinite(gpp).all(), gpp


def _linalg():                                            # _mojolearn_linalg
    # `mojolearn.linalg` publishes a cross-vendor identity profile and has
    # refused by name on any other tier since before DEVIATION 2490
    # (`_linalg_impl.py`, `require_identical`, after a mislabeled
    # deterministic build on the 2026-08-29 Apple stability run).
    _a = rng.random((8, 4)).astype(np.float32)
    _p = mojolearn.linalg.matmul(_a, _a.T)
    assert _p.shape == (8, 8), _p.shape


from mojolearn import _training_impl as _T             # _mojolearn_training
from mojolearn import _mamba_impl as _M                # _mojolearn_mamba

_dm = 32
_mrng = np.random.default_rng(20260903)


def _w(*shape):
    return (_mrng.standard_normal(shape) * 0.02).astype(np.float32)


# The SMALLEST fixture that reaches a launch, on the same principle as every
# arm above. d_model 32 is the smallest that keeps Mamba-1's derived shapes
# whole (dt_rank = ceil(32/16) = 2, d_inner = expand * 32) and gives the
# transformer four heads of eight.
_di = _M._M1_EXPAND * _dm
_r = int(math.ceil(_dm / 16.0))
_ds, _dc = 16, 4
_nh, _hd, _ff = 4, 8, 64


def _training():                                          # _mojolearn_training
    _ce = _T.cross_entropy(
        rng.standard_normal((4, 3)).astype(np.float32),
        np.array([0, 1, 2, 0], dtype=np.int32),
    )
    assert np.isfinite(_ce) and _ce > 0.0, _ce


def _mamba():                                             # _mojolearn_mamba
    _m1 = mojolearn.Mamba1Block({
        "norm.weight": np.ones(_dm, dtype=np.float32),
        "in_proj.weight": _w(2 * _di, _dm),
        "conv1d.weight": _w(_di, 1, _dc), "conv1d.bias": _w(_di),
        "x_proj.weight": _w(_r + 2 * _ds, _di),
        "dt_proj.weight": _w(_di, _r), "dt_proj.bias": _w(_di),
        "A_log": np.abs(_w(_di, _ds)) + 0.5, "D": _w(_di),
        "out_proj.weight": _w(_dm, _di),
    })
    _mx = _mrng.standard_normal((1, 4, _dm)).astype(np.float32)
    _mout = np.asarray(_m1.forward(_mx))
    assert _mout.shape == (1, 4, _dm), _mout.shape
    assert np.all(np.isfinite(_mout)), "Mamba1Block returned non-finite cells"


def _transformer():                                       # _mojolearn_transformer
    _tb = mojolearn.TransformerBlock({
        "input_layernorm.weight": np.ones(_dm, dtype=np.float32),
        "post_attention_layernorm.weight": np.ones(_dm, dtype=np.float32),
        "q_proj.weight": _w(_nh * _hd, _dm), "k_proj.weight": _w(_nh * _hd, _dm),
        "v_proj.weight": _w(_nh * _hd, _dm), "o_proj.weight": _w(_dm, _nh * _hd),
        "gate_proj.weight": _w(_ff, _dm), "up_proj.weight": _w(_ff, _dm),
        "down_proj.weight": _w(_dm, _ff),
    }, n_heads=_nh, head_dim=_hd)
    _tout = np.asarray(_tb.forward(
        _mrng.standard_normal((1, 4, _dm)).astype(np.float32)))
    assert _tout.shape == (1, 4, _dm), _tout.shape
    assert np.all(np.isfinite(_tout)), "TransformerBlock returned non-finite cells"


#: One entry per identical-only binding (two for `_mojolearn`, which carries
#: both k-means and k-NN). packaging/check_ext_lists.py keeps the binding
#: lists in step; this list is kept in step by the assertion right below it.
IDENTICAL_ONLY_LAUNCHES = [
    ("_mojolearn", _kmeans), ("_mojolearn", _knn),
    ("_mojolearn_estimators", _estimators),
    ("_mojolearn_solver", _solver),
    ("_mojolearn_svm", _svm),
    ("_mojolearn_tsa", _tsa),
    ("_mojolearn_metrics", _metrics),
    ("_mojolearn_preprocessing", _preprocessing),
    ("_mojolearn_arima", _arima),
    ("_mojolearn_gp", _gp),
    ("_mojolearn_linalg", _linalg),
    ("_mojolearn_training", _training),
    ("_mojolearn_mamba", _mamba),
    ("_mojolearn_transformer", _transformer),
]
_covered = {name for name, _ in IDENTICAL_ONLY_LAUNCHES}
_expected = set(mojolearn._backend._IDENTICAL_ONLY) - {"_mojolearn_byte_lm"}
assert _covered == _expected, (
    "this smoke's identical-only launches and _backend._IDENTICAL_ONLY "
    f"disagree: smoke-only {sorted(_covered - _expected)}, "
    f"package-only {sorted(_expected - _covered)}")
assert not (set(mojolearn._backend._TIERED) & _covered), "a tree lane is in the identical-only list"

for _name, _launch in IDENTICAL_ONLY_LAUNCHES:
    if _tier == "identical":
        _launch()
        continue
    try:
        _launch()
    except Exception as _exc:
        _why = str(_exc)
        if "identical" not in _why.lower():
            raise AssertionError(
                f"{_name} ({_launch.__name__}) failed on the {_tier} tier, "
                f"but not with the identical-only refusal; this is a "
                f"different fault and it is being reported rather than "
                f"passed: {_why}"
            ) from _exc
    else:
        raise AssertionError(
            f"{_name} ({_launch.__name__}) ANSWERED on the {_tier} tier, "
            f"where it must refuse: only the tree lanes ship {_tier}, and a "
            f"{_tier} binary of {_name} must not exist (DEVIATION 2490)")

# THE MODE THAT ACTUALLY LOADED, read back from the binary where it can be.
# verify_wheel.sh runs this file once per mode and checks the word.
mode = mojolearn.numeric_mode()
assert mode == _tier, (mode, _tier)

# The public UMAP, mamba and transformer suites against the installed
# package. These are ACCURACY suites (certified layout bits, corpus
# tolerances, state continuation), and every binding they touch is identical
# only, so they run under identical alone; the two lower tiers were gated
# above, where each of those bindings had to refuse by name.
import unittest
import importlib.util
from pathlib import Path
if _tier == "identical":
    _umap_spec = importlib.util.spec_from_file_location(
        "test_umap_surface", Path(__file__).resolve().parents[2]
        / "python/mojolearn/tests/test_umap_surface.py")
    test_umap_surface = importlib.util.module_from_spec(_umap_spec)
    _umap_spec.loader.exec_module(test_umap_surface)
    _umap_result = unittest.TextTestRunner(verbosity=1).run(
        unittest.defaultTestLoader.loadTestsFromModule(test_umap_surface))
    assert _umap_result.wasSuccessful(), "installed UMAP surface failed"

    # Load the drivers from source so they can find their independent
    # corpora; their mojolearn imports continue to resolve to this isolated
    # installation.
    for _sequence_family in ("mamba", "transformer"):
        _sequence_spec = importlib.util.spec_from_file_location(
            "test_" + _sequence_family + "_surface",
            Path(__file__).resolve().parents[2]
            / ("python/mojolearn/tests/test_" + _sequence_family + "_surface.py"))
        _sequence_gate = importlib.util.module_from_spec(_sequence_spec)
        _sequence_spec.loader.exec_module(_sequence_gate)
        assert _sequence_gate.main() == 0, "installed " + _sequence_family + " surface failed"

# THE VENDOR THAT ACTUALLY LOADED, read back from the binary (2026-08-29,
# docs/LINUX_WHEEL.md). On the macOS wheel it is 'metal'; the Linux smoke
# (packaging/linux/smoke.py) asserts 'cuda' or 'hip' the same way.
# MOJOLEARN_SMOKE_VENDOR overrides the expectation for a Linux run of THIS
# file. A None here is a binary built without checks/vendor.mojo, which a
# release build cannot be.
vendor = mojolearn.vendor()
assert vendor == os.environ.get("MOJOLEARN_SMOKE_VENDOR", "metal"), vendor
assert rf.vendor_used() == vendor, rf.vendor_used()

# This new ABI must execute through the installed wheel in every requested
# mode/interpreter. Keep full model/prediction evidence in the release log.
_ordered_spec = importlib.util.spec_from_file_location(
    "macos_ordered_smoke", Path(__file__).with_name("ordered_smoke.py"))
_ordered_gate = importlib.util.module_from_spec(_ordered_spec)
_ordered_spec.loader.exec_module(_ordered_gate)
_ordered_gate.run_installed_ordered(Path(__file__).resolve().parents[2], mojolearn, mode, vendor)

_extra = " ".join(f"{k}={v}" for k, v in _launched.items())
print(
    f"v{mojolearn.__version__} py{sys.version_info.major}.{sys.version_info.minor}"
    f" mode={mode} vendor={vendor} gbdt rf et ok"
    + (f" | identical-only launched: {_extra} kmeans knn dbscan pca svd ols solver svm tsa"
       f" metrics preprocessing arima gp linalg training mamba transformer ok"
       if _tier == "identical" else
       f" | {len(IDENTICAL_ONLY_LAUNCHES)} identical-only launches REFUSED by name under {_tier}")
)
