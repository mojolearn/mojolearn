# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""tools/bench_board_more.py on tiny data: the quality functions, the lane
subsets, the synthetic series, the prep of the time series blocks and the
tables tools/bench_board.py reads. NumPy only (no mojolearn, no opponent).

    .pixi/envs/test/bin/python -m pytest tools/test_bench_board_more.py

The reference values below are scikit-learn 1.9.0's answers on the same
arrays (sklearn.manifold.trustworthiness, adjusted_rand_score,
silhouette_score, GaussianMixture.score/bic, TruncatedSVD's
explained_variance_ratio_), so the checks hold where scikit-learn is absent;
where it is installed they are also compared live.
"""
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("bench_board_more", os.path.join(HERE, "bench_board_more.py"))
bbm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bbm)


@pytest.mark.parametrize("shape", [(3, 2), (2, 3), (6,)])
def test_host_cudf_forecast_uses_explicit_copy_and_preserves_layout(shape):
    values = np.arange(6, dtype=np.float32).reshape(shape)

    class Forecast:
        __module__ = "cudf.core.dataframe"

        def __array__(self, *args, **kwargs):
            raise TypeError("cuDF forbids implicit NumPy conversion")

        def to_numpy(self, *, copy):
            assert copy is True
            return values.copy()

    host = bbm._host(Forecast())
    assert host.shape == values.shape
    assert host.dtype == values.dtype
    assert not np.shares_memory(host, values)
    np.testing.assert_array_equal(host, values)
    expected = values.T if shape == (3, 2) else values.reshape(2, 3)
    np.testing.assert_array_equal(bbm._series_major(host, 2), expected)


def test_host_cupy_forecast_keeps_explicit_device_copy():
    values = np.arange(6, dtype=np.float32).reshape(3, 2)

    class Forecast:
        __module__ = "cupy._core.core"

        def __array__(self, *args, **kwargs):
            raise TypeError("CuPy forbids implicit NumPy conversion")

        def get(self):
            return values.copy()

    host = bbm._host(Forecast())
    assert host.dtype == values.dtype
    np.testing.assert_array_equal(bbm._series_major(host, 2), values.T)


def _xe():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 6))
    E = X[:, :2] + 0.3 * rng.standard_normal((300, 2))
    return rng, X, E


def test_trustworthiness_is_sklearns_formula():
    _rng, X, E = _xe()
    for chunk in (37, 256, 1000):
        assert bbm.trustworthiness(X, E, 15, chunk=chunk) == pytest.approx(0.6906385880465302, abs=1e-12)
    assert bbm.trustworthiness(X, X, 15) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        bbm.trustworthiness(X[:20], E[:20], 15)
    sk = pytest.importorskip("sklearn.manifold")
    assert bbm.trustworthiness(X, E, 15) == pytest.approx(sk.trustworthiness(X, E, n_neighbors=15))


def test_ari_and_silhouette_are_sklearns():
    rng, X, _E = _xe()
    a = rng.integers(0, 4, 300)
    b = np.where(rng.random(300) < .7, a, rng.integers(0, 5, 300))
    assert bbm.adjusted_rand(a, b) == pytest.approx(0.4856299311210968, abs=1e-12)
    assert bbm.adjusted_rand(a, a + 10) == pytest.approx(1.0)
    assert bbm.silhouette(X, b, chunk=41) == pytest.approx(-0.027979962071308026, abs=1e-9)
    assert bbm.silhouette(X, np.zeros(300)) is None
    # a singleton cluster scores 0, as scikit-learn's does
    lab = np.array([0] * 5 + [1] * 5 + [2])
    pts = np.vstack([np.zeros((5, 2)), np.ones((5, 2)) * 5, [[9, 9]]]) + np.arange(11)[:, None] * 1e-3
    s = bbm.silhouette(pts, lab)
    try:
        from sklearn.metrics import adjusted_rand_score, silhouette_score
    except ImportError:
        return
    assert s == pytest.approx(silhouette_score(pts, lab))
    assert bbm.adjusted_rand(a, b) == pytest.approx(adjusted_rand_score(a, b))


def test_gmm_quality_is_sklearns_score_and_bic():
    _rng, X, _E = _xe()
    # a fixed 2-component model: the float64 log-likelihood by hand
    w = np.array([0.4, 0.6])
    m = np.array([np.zeros(6), np.ones(6) * 0.5])
    c = np.array([np.eye(6), np.eye(6) * 2.0])
    ll = bbm.gmm_loglik(X, w, m, c)
    by_hand = np.log(w[0] * np.exp(-0.5 * (X ** 2).sum(1)) / (2 * np.pi) ** 3
                     + w[1] * np.exp(-0.25 * ((X - 0.5) ** 2).sum(1)) / ((2 * np.pi) ** 3 * 8.0))
    np.testing.assert_allclose(ll, by_hand, rtol=1e-12)
    q = bbm.quality("gmm", {"X": X, "Xq": X},
                    {"a": bbm._gmm_out(np, w, m, c, 7)})["a"]
    n_par = 2 * 6 + 2 * 21 + 1
    assert q["bic"] == pytest.approx(-2 * ll.sum() + n_par * np.log(300))
    assert q["mean_log_likelihood"] == pytest.approx(ll.mean()) and q["n_iter"] == 7
    mix = pytest.importorskip("sklearn.mixture")
    g = mix.GaussianMixture(3, random_state=0).fit(X)
    q = bbm.quality("gmm", {"X": X, "Xq": X},
                    {"a": bbm._gmm_out(np, g.weights_, g.means_, g.covariances_, g.n_iter_)})["a"]
    assert q["mean_log_likelihood"] == pytest.approx(g.score(X))
    assert q["bic"] == pytest.approx(g.bic(X))


def test_tsvd_quality():
    _rng, X, _E = _xe()
    Xs = X + 1.0
    V = np.linalg.svd(Xs, full_matrices=False)[2][:3]
    q = bbm.quality("tsvd", {"X": Xs}, {"a": {"components": V}})["a"]
    P = Xs @ V.T
    assert q["explained_variance_ratio_sum"] == pytest.approx(P.var(0).sum() / Xs.var(0).sum())
    R = Xs - P @ V
    assert q["relative_reconstruction_error"] == pytest.approx(np.linalg.norm(R) / np.linalg.norm(Xs))
    full = np.linalg.svd(Xs, full_matrices=False)[2]
    q = bbm.quality("tsvd", {"X": Xs}, {"a": {"components": full}})["a"]
    assert q["relative_reconstruction_error"] == pytest.approx(0.0, abs=1e-9)


def test_supervised_and_kernel_quality():
    yq = np.array([0, 1, 1, 0, 1], dtype=np.float32)
    q = bbm.quality("logreg", {"yq": yq}, {
        "a": {"pred": np.array([0, 1, 0, 0, 1.0]), "proba1": np.array([.1, .9, .4, .2, .8])}})["a"]
    assert q["accuracy"] == pytest.approx(0.8)
    assert q["logloss"] == pytest.approx(-np.mean(np.log([.9, .9, .4, .8, .8])))
    y = np.array([1.0, 2.0, 3.0, 4.0])
    q = bbm.quality("ridge", {"yq": y}, {"a": {"pred": y + 1.0}})["a"]
    assert q["rmse"] == pytest.approx(1.0) and q["r2"] == pytest.approx(1 - 4 / 5.0)
    q = bbm.quality("gpr", {"yq": y}, {"a": {"pred": y, "std": np.zeros(4)}})["a"]
    assert q["mean_log_predictive_density"] == pytest.approx(-0.5 * np.log(2 * np.pi * bbm.GP_ALPHA))
    rng = np.random.default_rng(1)
    Xc = rng.standard_normal((50, 4))
    g = 0.25
    K = np.exp(-g * ((Xc[:, None, :] - Xc[None, :, :]) ** 2).sum(-1))
    evals, evecs = np.linalg.eigh(K)
    Z = evecs * np.sqrt(np.maximum(evals, 0))
    D = {"X": np.zeros((10, 4)), "Xcheck": Xc}
    q = bbm.quality("nystroem", D, {"exact": {"zcheck": Z}, "zero": {"zcheck": np.zeros((50, 3))}})
    assert q["exact"]["kernel_rel_error"] == pytest.approx(0.0, abs=1e-9)
    assert q["zero"]["kernel_rel_error"] == pytest.approx(1.0)


def test_cluster_and_manifold_quality():
    _rng, X, E = _xe()
    lab = (X[:, 0] > 0).astype(np.int64)
    q = bbm.quality("spectral", {"X": X}, {"ours": {"labels": lab}, "x": {"labels": 1 - lab}})
    assert q["ours"]["n_clusters"] == 2 and "ari_vs_ours" not in q["ours"]
    assert q["x"]["ari_vs_ours"] == pytest.approx(1.0)
    assert q["x"]["silhouette"] == pytest.approx(q["ours"]["silhouette"])
    q = bbm.quality("umap", {"X": X}, {"a": {"embedding": E}, "b": {"embedding": E * np.nan}})
    assert q["a"]["trustworthiness_k15"] == pytest.approx(0.6906385880465302)
    assert q["b"]["trustworthiness_k15"] is None


def test_time_series_quality_shapes_and_values():
    fit = np.arange(24.0).reshape(2, 12)
    hold = np.ones((2, 3))
    outs = {"ours": {"llf": np.array([-10.0, -12.0]), "forecast": hold + 1.0, "insample": fit.copy()},
            "cuml": {"llf": np.array([-10.0, -12.0]), "forecast": (hold + 2.0).T,   # (h, series)
                     "insample": fit.T.copy()}}
    q = bbm.quality("arima", {"Yfit": fit, "Yhold": hold}, outs)
    assert q["ours"]["forecast_rmse"] == pytest.approx(1.0)
    assert q["cuml"]["forecast_rmse"] == pytest.approx(2.0)
    assert q["ours"]["insample_rmse"] == pytest.approx(0.0)
    assert q["ours"]["mean_llf"] == pytest.approx(-11.0)
    assert q["ours"]["mean_aic"] == pytest.approx(2 * 4 + 22.0)
    ins = fit.copy()
    ins[:, :48 if fit.shape[1] > 48 else 5] = np.nan
    q = bbm.quality("ets", {"Yfit": fit, "Yhold": hold}, {"a": {"forecast": hold, "insample": ins},
                                                          "c": {"forecast": hold}})
    assert q["a"]["forecast_rmse"] == 0.0 and "insample_rmse" not in q["c"]


def test_lane_arrays_take_the_same_rows_for_every_caller():
    rng = np.random.default_rng(3)
    B = {"X": rng.standard_normal((5000, 3)).astype(np.float32),
         "y": rng.integers(0, 2, 5000).astype(np.float32),
         "Xq": rng.standard_normal((700, 3)).astype(np.float32),
         "yq": rng.integers(0, 2, 700).astype(np.float32)}
    a, b = bbm.lane_arrays("gpc", B), bbm.lane_arrays("gpc", B)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])
    assert a["X"].shape == (bbm.GP_FIT, 3) and a["Xq"].shape == (700, 3)
    np.testing.assert_array_equal(a["X"][1], B["X"][5000 // bbm.GP_FIT])
    k = bbm.lane_arrays("nystroem", B)
    np.testing.assert_array_equal(k["Xcheck"], k["X"][:bbm.KAPPROX_CHECK])
    assert bbm.lane_arrays("logreg", B)["X"] is B["X"]
    Y = np.arange(20.0).reshape(2, 10)
    assert bbm.lane_arrays("arima", {"Y": np.hstack([Y] * 11)})["Yhold"].shape == (2, bbm.ARMA_H)


def test_synthetic_series_are_seeded_and_prep_writes_them(tmp_path):
    a1, a2 = bbm.arma_series(4, 50), bbm.arma_series(4, 50)
    np.testing.assert_array_equal(a1, a2)
    assert a1.dtype == np.float32 and a1.shape == (4, 50)
    s = bbm.seasonal_series(3, 96)
    assert s.shape == (3, 96) and np.all(np.isfinite(s)) and s.min() > 0
    rc = bbm.main(["prep", "--data", str(tmp_path), "--lanes", "arima,ets", "--datasets", "taxi",
                   "--max-rows", "300"])
    assert rc == 0
    rec = json.loads((tmp_path / "arma-synthetic.json").read_text())
    assert rec["arrays"]["Y"]["shape"] == [bbm.TS_SERIES, 300 + bbm.ARMA_H]
    rec = json.loads((tmp_path / "seasonal-synthetic.json").read_text())
    assert rec["arrays"]["Y"]["shape"] == [bbm.TS_SERIES, 300 + bbm.SEASON_H]
    assert rec["period"] == bbm.SEASON_PERIOD


def test_tables_are_complete_and_stdlib_only():
    for lane in bbm.LANE_ORDER:
        cfg = bbm.LANE_CONFIG[lane]
        for k in ("rows", "params", "timed", "quality", "mismatches"):
            assert k in cfg, (lane, k)
        assert bbm.has_fast(lane), lane      # every lane's binding ships FAST in 0.8.22
        for v in ("apple", "nvidia", "amd"):
            assert bbm.OPPONENTS[v][lane], (v, lane)
    # tools/bench_board.py imports this module on a box with no NumPy
    code = ("import sys, importlib.util; sys.modules['numpy'] = None; "
            "s = importlib.util.spec_from_file_location('m', %r); "
            "m = importlib.util.module_from_spec(s); s.loader.exec_module(m); print(len(m.LANES))"
            % os.path.join(HERE, "bench_board_more.py"))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == str(len(bbm.LANES))


def test_unknown_arm_refuses_by_name():
    with pytest.raises(SystemExit, match="no arm"):
        bbm.build("gmm", "torch-gpu", {"X": np.zeros((4, 2))}, {})


def test_mode_readback_uses_the_constant_else_the_tier_directory(tmp_path, monkeypatch):
    """0.8.22's _mojolearn_solver and _mojolearn_tsa carry no numeric-mode
    constant (measured on the wheel): the tier then comes from the loaded
    binary's directory, and the record says how."""
    import types
    base = tmp_path / "pkg"
    (base / "identical").mkdir(parents=True)
    fake = types.ModuleType("mojolearn")
    backend = types.ModuleType("mojolearn._backend")
    mods = {"_mojolearn_solver": types.SimpleNamespace(
                __name__="mojolearn._mojolearn_solver",
                __file__=str(base / "identical" / "_mojolearn_solver.so")),
            "_mojolearn_mixture": types.SimpleNamespace(
                __name__="mojolearn._mojolearn_mixture", __file__=str(base / "_m.so"),
                mixture_numeric_mode=lambda: 0)}
    backend.binding = lambda name, mode: mods[name]
    backend._vendor_fn = lambda short: short.replace("_mojolearn_", "") + "_vendor"
    backend.tier_dir = lambda m: str(base) if m == "fast" else str(base / m)
    backend._CODE_MODE = {0: "fast", 1: "identical", 2: "deterministic"}
    fake._backend = backend
    monkeypatch.setitem(sys.modules, "mojolearn", fake)
    monkeypatch.setitem(sys.modules, "mojolearn._backend", backend)
    est = types.SimpleNamespace()
    mode, how = bbm._mode_readback(fake, est, "_mojolearn_solver")
    assert mode == "identical" and "no numeric-mode constant" in how
    mode, how = bbm._mode_readback(fake, est, "_mojolearn_mixture")
    assert mode == "fast" and "constant" in how
    est = types.SimpleNamespace(numeric_mode_used=lambda: "identical")
    assert bbm._mode_readback(fake, est, "_mojolearn_solver") == ("identical", "numeric_mode_used()")


def test_gmm_lane_arrays_with_a_constant_column_describe_their_shape():
    import numpy as np
    X = np.random.default_rng(7).normal(size=(50, 4)).astype(np.float32)
    X[:, 2] = 1.0
    D = bbm.drop_constant_columns({"X": X, "Xq": X[:10]})
    assert D["_dropped_constant_columns"] == 1
    assert bbm.shape_desc(D) == "X 50x3; Xq 10x3; _dropped_constant_columns 1"
