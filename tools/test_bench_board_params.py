# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""tools/bench_board_params.py: the same-seed, same-parameters check.

    .pixi/envs/test/bin/python -m pytest tools/test_bench_board_params.py

Stub estimators only (no mojolearn, no opponent library): a scikit-learn-shaped
class per library, placed in a module named after the library so the check
reads the library the way it does on a box.
"""
import importlib.util
import io
import json
import os
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("bench_board_params",
                                               os.path.join(HERE, "bench_board_params.py"))
BP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(BP)


def _est(library, cls_name, **params):
    """A get_params() object whose class lives in module `library`."""
    mod = types.ModuleType(library)

    def get_params(self, deep=True):
        return dict(self._p)

    cls = type(cls_name, (), {"get_params": get_params, "__module__": library})
    obj = cls()
    obj._p = params
    return obj


def _matched():
    ours = _est("mojolearn.ensemble", "GradientBoosting", n_estimators=100, max_depth=6,
                learning_rate=0.1, l2_leaf_reg=1.0, border_count=254, random_state=7,
                numeric_mode="identical")
    fast = _est("mojolearn.ensemble", "GradientBoosting", n_estimators=100, max_depth=6,
                learning_rate=0.1, l2_leaf_reg=1.0, border_count=254, random_state=7,
                numeric_mode="fast")
    cat = _est("catboost.core", "CatBoostClassifier", iterations=100, depth=6,
               learning_rate=0.1, l2_leaf_reg=1.0, border_count=254, random_seed=7,
               thread_count=-1)
    xgb = _est("xgboost.sklearn", "XGBClassifier", n_estimators=100, max_depth=6,
               learning_rate=0.1, reg_lambda=1.0, max_bin=255, random_state=7, n_jobs=-1)
    return {"ours": ours, "ours-fast": fast, "catboost-cpu": cat, "xgboost-cpu": xgb}


def test_matched_race_passes_and_maps_names():
    buf = io.StringIO()
    rep = BP.enforce("gbdt-depthwise", _matched(), family="trees", stream=buf)
    assert rep["verdict"] == "MATCHED"
    # aliases resolved: CatBoost depth/iterations/border_count, XGBoost max_bin
    cat = rep["arms"]["catboost-cpu"]["params"]
    assert cat["max_depth"] == 6 and cat["n_estimators"] == 100 and cat["max_bin"] == 255
    assert rep["arms"]["xgboost-cpu"]["params"]["reg_lambda"] == 1.0
    # execution-only settings are not compared
    assert "n_jobs" not in rep["arms"]["xgboost-cpu"]["params"]
    assert "numeric_mode" not in rep["arms"]["ours-fast"]["params"]
    line = buf.getvalue().splitlines()[0]
    assert line.startswith(BP.MARK + " {")
    assert BP.parse_lines(buf.getvalue())[0]["verdict"] == "MATCHED"


def test_planted_parameter_mismatch_is_refused():
    arms = _matched()
    arms["xgboost-cpu"]._p["max_depth"] = 8               # the plant
    buf = io.StringIO()
    with pytest.raises(BP.ParamsRefused, match="max_depth"):
        BP.enforce("gbdt-depthwise", arms, stream=buf)
    assert BP.MARK_REFUSED in buf.getvalue()


def test_planted_seed_mismatch_is_refused():
    arms = _matched()
    arms["catboost-cpu"]._p["random_seed"] = 0
    with pytest.raises(BP.ParamsRefused, match="seed"):
        BP.enforce("gbdt-depthwise", arms, stream=io.StringIO())


def test_library_default_is_not_a_match():
    arms = _matched()
    arms["xgboost-cpu"]._p["learning_rate"] = None        # left to XGBoost's default
    with pytest.raises(BP.ParamsRefused, match="learning_rate is 0.1 on ours and unset"):
        BP.enforce("gbdt-depthwise", arms, stream=io.StringIO())


def test_arm_without_seed_argument_is_recorded_deterministic():
    arms = _matched()
    arms["faiss-ivf"] = {"__library__": "faiss", "nlist": 1024}
    arms["ours"]._p["nlist"] = 1024
    rep = BP.enforce("zz-no-row", arms, stream=io.StringIO())
    assert rep["verdict"] == "MATCHED"
    assert rep["arms"]["faiss-ivf"]["params"]["seed"] == BP.NO_SEED_ARGUMENT
    assert rep["seed_none_deterministic"] == ["faiss-ivf"]
    # a third-party arm that draws without a seed argument: its row's reason is recorded
    rep = BP.enforce("zz-no-row", arms, stream=io.StringIO(),
                     extra_exceptions=[("seed", "faiss-ivf", "faiss IVF-Flat training draws; no seed")])
    assert rep["verdict"] == "MATCHED"
    assert rep["exceptions"][0]["reason"].startswith("faiss")
    assert rep["arms"]["faiss-ivf"]["params"]["seed"] == BP.NO_SEED_DRAWS


def test_seed_argument_lost_in_read_back_is_refused():
    class Lost:
        __module__ = "sklearn.fake"

        def __init__(self, random_state=7):
            pass

        def get_params(self, deep=True):
            return {"n_estimators": 100}
    arms = {"ours": _matched()["ours"], "sk": Lost()}
    with pytest.raises(BP.ParamsRefused, match="takes a seed but none was read back"):
        BP.enforce("gbdt-depthwise", arms, stream=io.StringIO())


def test_seed_argument_left_none_is_refused():
    arms = _matched()
    arms["sk"] = {"__library__": "sklearn", "random_state": None}
    with pytest.raises(BP.ParamsRefused, match="seed is None"):
        BP.enforce("gbdt-depthwise", arms, stream=io.StringIO())


def test_lane_seed_is_the_harness_seed():
    assert BP.seed_for("spectral") == 42 and BP.seed_for("kmeans") == 7
    arms = {"ours": _est("mojolearn.cluster", "SpectralClustering", random_state=42),
            "sk": _est("sklearn.cluster", "SpectralClustering", random_state=42)}
    assert BP.enforce("spectral", arms, stream=io.StringIO())["verdict"] == "MATCHED"
    arms["sk"]._p["random_state"] = 7
    with pytest.raises(BP.ParamsRefused, match="seed is 7"):
        BP.enforce("spectral", arms, stream=io.StringIO())


def test_scale_pos_weight_matches_class_weights():
    ours = _est("mojolearn.ensemble", "GradientBoosting", random_state=7, class_weights=[1.0, 4.0])
    xgb = _est("xgboost.sklearn", "XGBClassifier", random_state=7, scale_pos_weight=4.0)
    rep = BP.enforce("gbdt-depthwise", {"ours": ours, "xgboost-cpu": xgb}, stream=io.StringIO())
    assert rep["arms"]["ours"]["params"]["scale_pos_weight"] == 4.0
    xgb._p["scale_pos_weight"] = 2.0
    with pytest.raises(BP.ParamsRefused, match="scale_pos_weight"):
        BP.enforce("gbdt-depthwise", {"ours": ours, "xgboost-cpu": xgb}, stream=io.StringIO())


def test_exception_table_names_lane_param_and_arm(monkeypatch):
    arms = _matched()
    arms["xgboost-cpu"]._p["max_depth"] = 8
    monkeypatch.setattr(BP, "EXCEPTIONS", [("gbdt-depth*", "max_depth", "xgboost-*", "planted")])
    rep = BP.enforce("gbdt-depthwise", arms, stream=io.StringIO())
    assert rep["verdict"] == "MATCHED" and rep["exceptions"][0]["param"] == "max_depth"
    # the same exception does not cover another lane
    with pytest.raises(BP.ParamsRefused):
        BP.enforce("gbdt-lossguide", arms, stream=io.StringIO())


def test_torch_optimizer_defaults_are_read():
    opt = type("AdamW", (), {"__module__": "torch.optim.adamw"})()
    opt.defaults = {"lr": 1e-3, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.01}
    opt.param_groups = []
    lib, source, got = BP.read_params(opt)
    assert (lib, source) == ("torch", "optimizer.defaults")
    canon = BP.canonical(lib, got)
    assert canon["learning_rate"][0] == 1e-3 and canon["betas"][0] == [0.9, 0.999]


def test_worker_records_are_checked_like_objects():
    arms = {n: BP.arm_record(o) for n, o in _matched().items()}
    assert BP.enforce("gbdt-depthwise", arms, stream=io.StringIO())["verdict"] == "MATCHED"
    arms["catboost-cpu"]["params"]["depth"] = 4
    with pytest.raises(BP.ParamsRefused, match="max_depth"):
        BP.enforce("gbdt-depthwise", arms, stream=io.StringIO())


def test_cuml_pca_full_is_covariance_eigh_and_its_tol_is_an_exception():
    ours = _est("mojolearn.decomposition", "PCA", n_components=10, svd_solver="covariance_eigh",
                tol=0.0, whiten=False, random_state=7)
    cuml = _est("cuml.decomposition.pca", "PCA", n_components=10, svd_solver="full", tol=1e-7,
                whiten=False)
    rep = BP.enforce("pca", {"ours": ours, "cuml-gpu": cuml}, family="classical",
                     stream=io.StringIO())
    assert rep["verdict"] == "MATCHED"
    assert rep["arms"]["cuml-gpu"]["params"]["svd_solver"] == "covariance_eigh"
    assert "tol" in json.dumps(rep["exceptions"])
    # jacobi is another algorithm: still refused
    cuml._p["svd_solver"] = "jacobi"
    with pytest.raises(BP.ParamsRefused, match="svd_solver"):
        BP.enforce("pca", {"ours": ours, "cuml-gpu": cuml}, family="classical",
                   stream=io.StringIO())


def test_cuml_tsvd_full_solver_ignores_n_iter_and_tol():
    ours = _est("mojolearn.decomposition", "TruncatedSVD", n_components=10, algorithm="covariance_eigh",
                n_iter=5, tol=0.0, random_state=7)
    cuml = _est("cuml.decomposition.tsvd", "TruncatedSVD", n_components=10, algorithm="full",
                n_iter=15, tol=1e-7, random_state=7)
    rep = BP.enforce("tsvd", {"ours": ours, "cuml-gpu": cuml}, family="classical2",
                     stream=io.StringIO())
    assert rep["verdict"] == "MATCHED"
    cuml._p["n_components"] = 12                           # a real difference still refuses
    with pytest.raises(BP.ParamsRefused, match="n_components"):
        BP.enforce("tsvd", {"ours": ours, "cuml-gpu": cuml}, family="classical2",
                   stream=io.StringIO())


def test_holt_winters_add_is_additive():
    ours = _est("mojolearn.forecast", "ExponentialSmoothing", seasonal="additive",
                seasonal_periods=24, eps=0.00224, start_periods=2)
    cuml = _est("cuml.tsa.holtwinters", "ExponentialSmoothing", seasonal="add",
                seasonal_periods=24, eps=0.00224, start_periods=2)
    rep = BP.enforce("ets", {"ours": ours, "cuml-gpu": cuml}, family="classical2",
                     stream=io.StringIO())
    assert rep["verdict"] == "MATCHED"
    cuml._p["seasonal"] = "mul"
    with pytest.raises(BP.ParamsRefused, match="seasonal"):
        BP.enforce("ets", {"ours": ours, "cuml-gpu": cuml}, family="classical2",
                   stream=io.StringIO())


def _hdbscan_arms(sk_min_samples, cuml_min_samples=10):
    common = dict(min_cluster_size=100, metric="euclidean", cluster_selection_method="eom",
                  cluster_selection_epsilon=0.0, alpha=1.0, allow_single_cluster=False)
    ours = _est("mojolearn.hdbscan", "HDBSCAN", min_samples=10, max_cluster_size=0, **common)
    cuml = _est("cuml.cluster.hdbscan", "HDBSCAN", min_samples=cuml_min_samples, max_cluster_size=0,
                **common)
    sk = _est("sklearn.cluster._hdbscan.hdbscan", "HDBSCAN", min_samples=sk_min_samples,
              max_cluster_size=None, n_jobs=-1, **common)
    return {"ours": ours, "cuml-gpu": cuml, "sklearn-cpu": sk}


def test_hdbscan_min_samples_counts_the_point_in_sklearn_only():
    """scikit-learn's HDBSCAN counts the point itself in min_samples; ours and
    cuML's do not. scikit-learn 11 selects ours' 10th neighbour: MATCHED."""
    rep = BP.enforce("hdbscan", _hdbscan_arms(11), family="classical", stream=io.StringIO())
    assert rep["verdict"] == "MATCHED"
    assert rep["arms"]["sklearn-cpu"]["params"]["min_samples"] == 10
    assert rep["arms"]["cuml-gpu"]["params"]["min_samples"] == 10


def test_hdbscan_same_number_is_a_different_k_and_refuses():
    """The same NUMBER on scikit-learn is a different neighbour: refused."""
    with pytest.raises(BP.ParamsRefused):
        BP.enforce("hdbscan", _hdbscan_arms(10), family="classical", stream=io.StringIO())


def test_hdbscan_cuml_takes_no_transform():
    """cuML passes min_samples straight to runner.h, which adds 1 as ours does."""
    with pytest.raises(BP.ParamsRefused):
        BP.enforce("hdbscan", _hdbscan_arms(11, cuml_min_samples=11), family="classical",
                   stream=io.StringIO())


def test_sklearn_dbscan_min_samples_is_not_transformed():
    """Only scikit-learn's HDBSCAN is keyed: its DBSCAN counts the point, as ours does."""
    assert BP.library_of(_est("sklearn.cluster._dbscan", "DBSCAN", min_samples=2)) == "sklearn"
    assert BP.library_of(_est("sklearn.cluster._hdbscan.hdbscan", "HDBSCAN")) == "sklearn/HDBSCAN"
