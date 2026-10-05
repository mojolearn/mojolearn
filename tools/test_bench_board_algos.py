# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""tools/bench_board_algos.py: the algorithm-expansion races. The tables, the
SKIPPED path, the prep derivations and the conductor's quality functions, on
tiny arrays (no GPU, no mojolearn, no opponent library needed)."""
import importlib.util
import json
import os
import subprocess
import sys
import types

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    spec = importlib.util.spec_from_file_location("t_" + name, os.path.join(HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


A = _load("bench_board_algos")
bb = _load("bench_board")


def test_kernel_pca_quality_without_opponents():
    X = np.array([[0., 0.], [1., 0.], [0., 2.], [2., 3.]])
    K = np.exp(-((X[:, None] - X[None, :]) ** 2).sum(2) / X.shape[1])
    values, vectors = np.linalg.eigh(K)
    Z = vectors * np.sqrt(np.maximum(values, 0))
    q = A.quality("kernel-pca", {"Xq": X}, {"ours-fast": {"pred": Z}})
    assert q["ours-fast"]["rbf_feature_distance_relative_stress"] < 1e-12
    assert A._rbf_embedding_stress(X, Z + 7) < 1e-12
    assert A._rbf_embedding_stress(X, np.zeros_like(Z)) == pytest.approx(1)
    assert A._rbf_embedding_stress(X, Z * 2) == pytest.approx(3)
    with pytest.raises(ValueError):
        A._rbf_embedding_stress(X, Z * np.nan)
    with pytest.raises(ValueError):
        A._rbf_embedding_stress(X, Z[:-1])

XLANES = {"linear", "cluster", "neighbors", "decomp", "prep", "sequence", "trees", "cnn", "ann",
          # the lanes of the public mojolearn.training, resample, model_selection and
          # embedding surfaces (2026-09-29)
          "training", "resample", "model_selection", "embedding"}


def test_tables_are_complete_and_stdlib_only():
    assert {A.LANES[l]["xlane"] for l in A.LANE_ORDER} == XLANES
    alone = set()
    for lane in A.LANE_ORDER:
        s = A.LANES[lane]
        assert s["ours"] and all(isinstance(n, str) for n in s["ours"]), lane
        assert s["datasets"], lane
        cfg = A.lane_config(lane)
        for k in ("ours_class", "timed_fit", "quality", "params", "mismatches"):
            assert cfg[k] is not None, (lane, k)
        json.dumps(cfg)
        assert A.quality_kind(lane) in A.QUALITY_TEXT, lane
        assert A.rows_text(lane)
        for v in ("apple", "nvidia", "amd"):
            opp = A.opponents(v, lane)
            if not opp:
                alone.add(lane)
            if v != "nvidia":
                assert not any(a in A._NVIDIA_ONLY for a in opp), (v, lane)
        # two datasets of different kind, or its own named data
        tab = [d for d in s["datasets"] if d in A.TAB]
        assert len(s["datasets"]) >= 2 or s["block"] in ("tensor", "optim", "dense", "sym", "images",
                                                          "corpus"), lane
        assert not tab or set(tab) == set(A.TAB), lane
    # torch.optim has no Lion and no LAMB: those race ours alone, named in not_planned
    assert alone == {"lion", "lamb"}
    assert any("Lion and LAMB" in w for w in A.not_planned("nvidia"))
    code = ("import sys, importlib.util; sys.modules['numpy'] = None; "
            "s = importlib.util.spec_from_file_location('m', %r); "
            "m = importlib.util.module_from_spec(s); s.loader.exec_module(m); print(len(m.LANES))"
            % os.path.join(HERE, "bench_board_algos.py"))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == str(len(A.LANES))


def test_lane_names_do_not_collide_with_the_other_families():
    others = set(bb.TREE_LANES + bb.TREE_TASK_LANES + bb.CLASSICAL_LANES + bb.MORE_LANES
                 + bb.NEURAL_LANES)
    assert not others & set(A.LANE_ORDER)


def test_every_opponent_has_a_pin_or_is_in_an_existing_set():
    pinned = " ".join(A.PINS["nvidia"] + A.RAPIDS_EXTRA["nvidia"])
    for lib in ("statsforecast", "statsmodels", "arch", "prophet", "networkx", "shap", "implicit",
                "torch-geometric", "gpytorch", "faiss-cpu", "cugraph-cu12"):
        assert lib + "==" in pinned, lib
    assert all("==" in p for v in A.PINS.values() for p in v)


@pytest.mark.parametrize("vendor", ["apple", "nvidia", "amd"])
def test_plan_has_every_lane_with_our_arms(vendor):
    races = bb.plan_races(vendor, bb.modes_for(vendor), ["algos"])
    assert {r["lane"] for r in races} == set(A.LANE_ORDER)
    want = {"ours": "identical"}          # the board never races our CPU (Oct 2 2026)
    if vendor == "apple":
        want["ours-fast"] = "fast"
    for r in races:
        assert r["our_arms"] == want, r["id"]
        assert sorted(r["arms"][:len(want)]) == sorted(want)
        # a race with any GPU opponent races only GPU opponents
        opp = r["opponents"]
        if any("-cpu" not in a for a in opp):
            assert not any(bb._is_cpu_arm(a) for a in opp), r["id"]
    # --datasets narrows taxi/Istella; a lane's own data always runs
    only = bb.plan_races(vendor, bb.modes_for(vendor), ["algos"], datasets=["taxi"])
    assert not any(r["dataset"] == "istella" for r in only)
    assert any(r["dataset"] == "text" for r in only)


def test_opponent_rosters():
    nv = {l: A.opponents("nvidia", l) for l in A.LANE_ORDER}
    assert nv["sgd-clf"] == ("sklearn-cpu", "cuml-gpu")
    assert nv["dart"] == ("lightgbm-cpu", "xgboost-cpu", "xgboost-gpu")
    assert nv["ivf-pq"] == ("faiss-cpu", "cuvs-gpu")
    assert nv["pagerank"] == ("networkx-cpu", "cugraph-gpu")
    assert nv["lstm-clf"][0] == "torch-eager-fp32" and "torch-compile-tf32" in nv["lstm-clf"]
    assert nv["dart-reg"] == ("lightgbm-cpu", "xgboost-cpu", "xgboost-gpu")
    assert nv["autoarima"] == ("statsforecast-cpu", "cuml-gpu")
    assert A.opponents("amd", "ivf-pq") == ("faiss-cpu",)
    assert not any("tf32" in a for a in A.opponents("apple", "conv2d"))


def test_skipped_when_the_wheel_lacks_the_class(monkeypatch):
    fake = types.ModuleType("mojolearn")
    fake.__version__ = "0.0-test"
    fake.Ridge = object
    monkeypatch.setitem(sys.modules, "mojolearn", fake)
    with pytest.raises(A.Skipped, match="SKIPPED: not built yet"):
        A._ours_class("sgd-clf")
    with pytest.raises(A.Skipped):
        A.build("lu-solve", "ours", {})
    fake.linalg = types.SimpleNamespace(solve=len)
    assert A._ours_class("lu-solve") == ("linalg.solve", len)
    fake.solve = abs                              # the first exported candidate wins
    assert A._ours_class("lu-solve") == ("solve", abs)
    # a nested base estimator the wheel lacks is a skip too, not an error
    with pytest.raises(A.Skipped, match="GaussianNB"):
        A._resolve(A._E("GaussianNB"), "ours")


def test_board_marks_skipped_cells():
    cells = [{"arm": "ours", "status": "REFUSED(x)"}, {"arm": "sklearn-cpu", "status": "ok"}]
    r = {"arms": {"ours": {"status": "skipped", "error": "SKIPPED: not built yet (...)"},
                  "sklearn-cpu": {"status": "ok"}}}
    out = bb.algos_skips(cells, r)
    assert out[0]["status"] == bb.ALGOS_SKIPPED and out[1]["status"] == "ok"


def test_lane_arrays_derivations():
    rng = np.random.default_rng(0)
    B = {"X": rng.standard_normal((200, 6)).astype(np.float32),
         "Xq": rng.standard_normal((50, 6)).astype(np.float32),
         "y": (rng.random(200) > 0.5).astype(np.float32), "yq": (rng.random(50) > 0.5).astype(np.float32)}
    D = A.lane_arrays("additive-chi2", dict(B))
    assert D["X"].min() >= 0 and D["Xq"].min() >= 0
    D = A.lane_arrays("knn-imputer", dict(B))
    assert np.isnan(D["X"]).any() and not np.isnan(D["X_true"]).any()
    D = A.lane_arrays("label-propagation", dict(B))
    assert (D["y_semi"] == -1).sum() == 180
    D = A.lane_arrays("multioutput-clf", dict(B))
    assert D["X"].shape[1] == 5 and D["Y"].shape == (200, 2)
    D = A.lane_arrays("isotonic", dict(B))
    assert D["X"].ndim == 1
    D = A.lane_arrays("incremental-pca", {"X": B["X"]})
    assert D["X"].shape[0] + D["Xq"].shape[0] == 200
    # 'half' resolves to d // 2; gaussian-rp itself races the cuML benchmark's 10 components
    p = A._derived_params("gaussian-rp", {"X": B["X"]}, dict(A.LANES["gaussian-rp"]["params"],
                                                              n_components="half"))
    assert p["n_components"] == 3
    assert A.LANES["gaussian-rp"]["params"]["n_components"] == 10
    Y = np.random.default_rng(3).standard_normal((2, 100)).astype(np.float32)
    D = A.lane_arrays("gru-clf", {"Y": Y})
    n_fit = int(100 * 0.8) - A.SEQ_T
    assert D["X"].shape == (2 * n_fit, A.SEQ_T, 1) and D["Xq"].shape[1:] == (A.SEQ_T, 1)
    assert set(np.unique(D["y"])) <= {0.0, 1.0} and D["X"].dtype == np.float32
    assert A.block_file("gru-clf", "synthetic") == "ts-synthetic"
    D = A.lane_arrays("autoarima", {"Y": np.arange(2 * 100, dtype=np.float32).reshape(2, 100)})
    assert D["Yfit"].shape == (2, 100 - A.TS_H) and D["Yhold"].shape == (2, A.TS_H)


def test_taxi_hourly_recovers_the_month():
    # Jan 1 2024 is weekday 0 in the decode's convention, Feb 1 weekday 3
    x = np.zeros((4, 16), dtype=np.float32)
    x[:, A.TAXI_PU] = 7
    x[:, A.TAXI_HOUR] = 5
    x[:, A.TAXI_DAY] = [1, 1, 2, 1]
    x[:, A.TAXI_WDAY] = [0, 3, 1, 5]          # Jan 1, Feb 1, Jan 2, neither
    C = A.taxi_hourly(x)
    assert C[7, 5] == 1 and C[7, 31 * 24 + 5] == 1 and C[7, 24 + 5] == 1 and C.sum() == 3


def test_graph_and_quality_helpers():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((60, 3))
    ip, ix = A.knn_graph(X, 4)
    assert ip.shape == (61,) and ip[-1] == ix.shape[0]
    src = np.repeat(np.arange(60), np.diff(ip))
    assert set(zip(src.tolist(), ix.tolist())) == set(zip(ix.tolist(), src.tolist()))
    assert not (src == ix).any()
    one = np.zeros(60, dtype=np.int64)
    assert abs(A._modularity(ip, ix, one)) < 1e-12
    M = rng.standard_normal((20, 4))
    assert A._subspace(M, M @ rng.standard_normal((4, 4))) == pytest.approx(1.0)
    Xc = np.abs(rng.standard_normal((5, 3)))
    K = A._kernel_exact("kernel-achi2", Xc, {"X": Xc})
    assert K[0, 0] == pytest.approx(Xc[0].sum())
    D = {"index": rng.standard_normal((100, 3)).astype(np.float32),
         "queries": rng.standard_normal((7, 3)).astype(np.float32)}
    X64, Q64 = D["index"].astype(np.float64), D["queries"].astype(np.float64)
    truth = np.argsort(((Q64[:, None] - X64[None]) ** 2).sum(-1), axis=1)[:, :10]
    assert A._recall(D, truth) == 1.0
    A_, B_ = A.lu_system(64)
    x = np.linalg.solve(A_.astype(np.float64), B_)
    q = A.quality("lu-solve", {}, {"numpy-cpu": {"x": x.reshape(-1)}})
    assert q["numpy-cpu"]["relative_residual"] < 1e-10


def test_quality_by_kind():
    rng = np.random.default_rng(2)
    yq = (rng.random(40) > 0.5).astype(np.float32)
    q = A.quality("perceptron", {"yq": yq}, {"sklearn-cpu": {"pred": yq.astype(np.float64)}})
    assert q["sklearn-cpu"]["accuracy"] == 1.0
    ref = {"pred": rng.standard_normal((40, 3))}
    q = A.quality("robust-scaler", {}, {"sklearn-cpu": ref, "ours": {"pred": ref["pred"] + 1e-3}})
    assert q["ours"]["max_abs_diff_vs_sklearn"] == pytest.approx(1e-3)
    sup = {"support": np.array([1, 0, 1, 0])}
    q = A.quality("select-chi2", {}, {"sklearn-cpu": sup, "ours": {"support": np.array([1, 1, 1, 0])}})
    assert q["ours"]["jaccard_vs_sklearn"] == pytest.approx(2 / 3)
    phi = rng.standard_normal((5, 3))
    q = A.quality("tree-shap", {}, {"x": {"phi": phi, "base": np.ones(5), "margin": phi.sum(1) + 1}})
    assert q["x"]["max_additivity_error"] < 1e-12
    Y = rng.standard_normal((3, 60))
    q = A.quality("theta", {"Yfit": Y[:, :-A.TS_H], "Yhold": Y[:, -A.TS_H:]},
                  {"a": {"forecast": Y[:, -A.TS_H:]}})
    assert q["a"]["forecast_rmse"] == 0.0
    y = rng.standard_normal((1, 8, 4)).astype(np.float32)
    q = A.quality("conv2d", {}, {"torch-eager-fp32": {"y": y}, "ours": {"y": y}})
    assert q["ours"]["rel_fro_vs_torch_eager_fp32"] == 0.0
    # a quality function that raises is recorded by name, never fatal
    q = A.quality("perceptron", {"yq": yq}, {"x": {}})
    assert "error" in q["x"]


def test_device_ndarray_is_copied_to_host():
    """cuVS returns pylibraft device_ndarray; np.array() of it is garbage, so
    the host copy must go through copy_to_host (the classical2 ivf cuvs arm
    and every algos cuvs arm read their neighbours through it)."""
    class Dev(object):
        __module__ = "pylibraft.common.device_ndarray"

        def copy_to_host(self):
            return np.arange(6).reshape(2, 3)

        def __array__(self, dtype=None, copy=None):
            return np.full((2, 3), -1)
    ctd = _load("classical_two_datasets")
    assert ctd._to_host(Dev()).tolist() == [[0, 1, 2], [3, 4, 5]]
    assert A._arr(Dev(), np.int64).tolist() == [[0, 1, 2], [3, 4, 5]]


def test_bayesian_gmm_reg_covar_per_dataset(tmp_path, monkeypatch):
    # Istella-S takes reg_covar 3e-3 on every arm (every arm refused at 1e-6);
    # taxi keeps 1e-6. The dataset is the one _load_block read.
    D = {"X": np.zeros((4, 3), np.float32)}
    s = A.LANES["bayesian-gmm"]
    for ds, want in (("istella", 3e-3), ("taxi", 1e-6)):
        monkeypatch.setattr(A, "_DATASET", ds)
        for params in (s["params"], s.get("sk_params", s["params"])):
            assert A._derived_params("bayesian-gmm", D, params)["reg_covar"] == want
    assert s["params"]["reg_covar"] == 1e-6          # the table itself is not changed
    assert A.lane_config("bayesian-gmm")["dataset_params"] == {"istella": {"reg_covar": 3e-3}}
    # _load_block records the dataset (the synthetic path needs no file)
    monkeypatch.setattr(A, "block_file", lambda lane, ds: None)
    A._load_block("bayesian-gmm", "istella", str(tmp_path))
    assert A._DATASET == "istella"


def test_no_other_lane_has_dataset_params():
    assert sorted(k for k, s in A.LANES.items() if s.get("dataset_params")) == ["bayesian-gmm"]
