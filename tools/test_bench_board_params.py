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
